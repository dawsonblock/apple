from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from storage import (
    RFSNConfig,
    QuantizedMatrix,
    CompressedTensorMLX,
    SUPPORTED_GROUP_SIZES,
    _concat0,
    _numpy_ints,
)


logger = logging.getLogger("RFSN_QUANT")


def _choose_group_size(last_dim: int, preferred: int) -> Optional[int]:
    candidates: List[int] = []
    if preferred in SUPPORTED_GROUP_SIZES:
        candidates.append(preferred)
    for size in SUPPORTED_GROUP_SIZES:
        if size not in candidates:
            candidates.append(size)
    for size in candidates:
        if last_dim % size == 0:
            return size
    return None


def _quantize_matrix(
    matrix: mx.array,
    bits: int,
    preferred_group_size: int,
) -> Optional[QuantizedMatrix]:
    if not hasattr(mx, "quantize"):
        return None
    if len(matrix.shape) != 2:
        return None

    original_shape = tuple(int(dim) for dim in matrix.shape)
    candidate = matrix.astype(mx.float32)
    transposed = False

    group_size = _choose_group_size(int(candidate.shape[-1]), preferred_group_size)
    if group_size is None:
        transposed_candidate = mx.swapaxes(candidate, 0, 1)
        group_size = _choose_group_size(int(transposed_candidate.shape[-1]), preferred_group_size)
        if group_size is None:
            return None
        candidate = transposed_candidate
        transposed = True

    try:
        q_weights, q_scales, q_biases = mx.quantize(
            candidate,
            bits=bits,
            group_size=group_size,
        )
    except Exception:
        return None

    return QuantizedMatrix(
        weights=q_weights,
        scales=q_scales,
        biases=q_biases,
        bits=bits,
        group_size=group_size,
        transposed=transposed,
        original_shape=original_shape,
    )


def _dequantize_matrix(matrix: QuantizedMatrix, dtype: mx.Dtype = mx.float32) -> mx.array:
    recovered = mx.dequantize(
        matrix.weights,
        matrix.scales,
        matrix.biases,
        bits=matrix.bits,
        group_size=matrix.group_size,
    )
    if matrix.transposed:
        recovered = mx.swapaxes(recovered, 0, 1)
    return recovered.astype(dtype)


def _squared_l2_distance(lhs: mx.array, rhs: mx.array) -> mx.array:
    lhs_sq = mx.sum(lhs * lhs, axis=1, keepdims=True)
    rhs_sq = mx.sum(rhs * rhs, axis=1, keepdims=True).T
    cross = lhs @ rhs.T
    return lhs_sq - (2.0 * cross) + rhs_sq


class ProductQuantizerMLX(nn.Module):
    """MLX-native product quantizer with optional codebook quantization."""

    def __init__(self, config: RFSNConfig):
        super().__init__()
        self.config = config
        self.num_subspaces = config.num_subspaces
        self.subspace_dim = config.subspace_dim
        self.codebook_size = 1 << config.pq_bits
        scale = (2.0 / self.subspace_dim) ** 0.5
        self.codebooks = (
            mx.random.normal(
                shape=(self.num_subspaces, self.codebook_size, self.subspace_dim),
                dtype=mx.float32,
            )
            * scale
        ).astype(mx.float16)
        self._ane_codebooks: List[Optional[QuantizedMatrix]] = [None] * self.num_subspaces

    def _codebook(self, subspace: int, dtype: mx.Dtype = mx.float32) -> mx.array:
        quantized = self._ane_codebooks[subspace]
        if quantized is not None:
            return _dequantize_matrix(quantized, dtype=dtype)
        return self.codebooks[subspace].astype(dtype)

    def quantize(self, vectors: mx.array) -> Tuple[mx.array, mx.array]:
        num_tokens, dim = vectors.shape
        expected_dim = self.num_subspaces * self.subspace_dim
        if dim != expected_dim:
            raise ValueError(f"Expected vectors with dim {expected_dim}, got {dim}")

        codes_np = np.zeros((num_tokens, self.num_subspaces), dtype=np.uint8)
        residuals = mx.zeros_like(vectors)

        for subspace in range(self.num_subspaces):
            start = subspace * self.subspace_dim
            end = start + self.subspace_dim
            sub_vectors = vectors[:, start:end].astype(mx.float32)
            codebook = self._codebook(subspace, dtype=mx.float32)
            dists = _squared_l2_distance(sub_vectors, codebook)
            idx = mx.argmin(dists, axis=1).astype(mx.int32)
            codes_np[:, subspace] = np.asarray(idx, dtype=np.uint8)
            reconstructed = mx.take(codebook, idx, axis=0)
            residuals[:, start:end] = (sub_vectors - reconstructed).astype(vectors.dtype)

        return mx.array(codes_np, dtype=mx.uint8), residuals

    def decode(self, codes: mx.array) -> mx.array:
        num_tokens = int(codes.shape[0])
        result = mx.zeros((num_tokens, self.num_subspaces * self.subspace_dim), dtype=mx.float16)

        for subspace in range(self.num_subspaces):
            start = subspace * self.subspace_dim
            end = start + self.subspace_dim
            codebook = self._codebook(subspace, dtype=mx.float16)
            idx = codes[:, subspace].astype(mx.int32)
            result[:, start:end] = mx.take(codebook, idx, axis=0)

        return result

    def quantize_codebooks_for_ane(self) -> Dict[str, int]:
        quantized = 0
        for subspace in range(self.num_subspaces):
            packed = _quantize_matrix(
                self.codebooks[subspace],
                bits=self.config.ane_quant_bits,
                preferred_group_size=self.config.ane_group_size,
            )
            self._ane_codebooks[subspace] = packed
            if packed is not None:
                quantized += 1
        logger.info(
            "PQ codebook fast path prepared for %d/%d subspaces",
            quantized,
            self.num_subspaces,
        )
        return {"quantized_subspaces": quantized, "total_subspaces": self.num_subspaces}


class ResidualVQMLX(nn.Module):
    """Sparse residual vector quantizer."""

    def __init__(self, config: RFSNConfig):
        super().__init__()
        self.config = config
        self.num_layers = config.num_rvq_layers
        self.codebook_size = config.rvq_codebook_size
        self.head_dim = config.head_dim
        self.sparsity_threshold = config.rvq_sparsity_threshold
        scale = (2.0 / self.head_dim) ** 0.5
        self.codebooks = (
            mx.random.normal(
                shape=(self.num_layers, self.codebook_size, self.head_dim),
                dtype=mx.float32,
            )
            * scale
        ).astype(mx.float16)
        for layer in range(self.num_layers):
            self.codebooks[layer, 0] = mx.zeros((self.head_dim,), dtype=mx.float16)
        self._ane_codebooks: List[Optional[QuantizedMatrix]] = [None] * self.num_layers

    def _codebook(self, layer_idx: int, dtype: mx.Dtype = mx.float32) -> mx.array:
        quantized = self._ane_codebooks[layer_idx]
        if quantized is not None:
            return _dequantize_matrix(quantized, dtype=dtype)
        return self.codebooks[layer_idx].astype(dtype)

    def encode(
        self,
        residuals: mx.array,
        base_codes: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array, mx.array]:
        del base_codes
        num_tokens, dim = residuals.shape
        if dim != self.head_dim:
            raise ValueError(f"Expected residual dim {self.head_dim}, got {dim}")

        residual_norms = mx.sqrt(mx.sum(residuals.astype(mx.float32) ** 2, axis=1))
        mask_np = np.asarray(residual_norms > self.sparsity_threshold, dtype=bool)
        active_indices_np = np.flatnonzero(mask_np)
        mask = mx.array(mask_np, dtype=mx.bool_)

        if active_indices_np.size == 0:
            return (
                mx.zeros((0, self.num_layers), dtype=mx.uint16),
                mask,
                mx.zeros((0,), dtype=mx.int32),
            )

        active_indices = mx.array(active_indices_np, dtype=mx.int32)
        running_residual = mx.take(residuals.astype(mx.float32), active_indices, axis=0)
        num_active = int(active_indices.shape[0])
        codes = mx.zeros((num_active, self.num_layers), dtype=mx.uint16)

        chunk_size = min(512, self.codebook_size)
        for layer_idx in range(self.num_layers):
            codebook = self._codebook(layer_idx, dtype=mx.float32)
            best_dists = mx.full((num_active,), float("inf"), dtype=mx.float32)
            best_idx = mx.zeros((num_active,), dtype=mx.int32)
            for start in range(0, self.codebook_size, chunk_size):
                end = min(start + chunk_size, self.codebook_size)
                chunk = codebook[start:end]
                dists = _squared_l2_distance(running_residual, chunk)
                local_idx = mx.argmin(dists, axis=1).astype(mx.int32)
                local_best = mx.min(dists, axis=1)
                update = local_best < best_dists
                best_dists = mx.where(update, local_best, best_dists)
                best_idx = mx.where(update, local_idx + start, best_idx)

            codes[:, layer_idx] = best_idx.astype(mx.uint16)
            chosen = mx.take(codebook, best_idx, axis=0)
            running_residual = running_residual - chosen

        return codes, mask, active_indices.astype(mx.int32)

    def decode_correction(
        self,
        rvq_codes: mx.array,
        rvq_mask: mx.array,
        rvq_offsets: mx.array,
    ) -> mx.array:
        total_tokens = int(rvq_mask.shape[0])
        correction = mx.zeros((total_tokens, self.head_dim), dtype=mx.float16)
        num_active = int(rvq_codes.shape[0])
        if num_active == 0:
            return correction

        total = mx.zeros((num_active, self.head_dim), dtype=mx.float32)
        for layer_idx in range(self.num_layers):
            codebook = self._codebook(layer_idx, dtype=mx.float32)
            idx = rvq_codes[:, layer_idx].astype(mx.int32)
            total = total + mx.take(codebook, idx, axis=0)

        total = total.astype(mx.float16)
        for row_idx, token_offset in enumerate(_numpy_ints(rvq_offsets).tolist()):
            correction[token_offset] = total[row_idx]
        return correction

    def quantize_codebooks_for_ane(self) -> Dict[str, int]:
        quantized = 0
        for layer_idx in range(self.num_layers):
            packed = _quantize_matrix(
                self.codebooks[layer_idx],
                bits=self.config.ane_quant_bits,
                preferred_group_size=self.config.ane_group_size,
            )
            self._ane_codebooks[layer_idx] = packed
            if packed is not None:
                quantized += 1
        logger.info(
            "RVQ codebook fast path prepared for %d/%d layers",
            quantized,
            self.num_layers,
        )
        return {"quantized_layers": quantized, "total_layers": self.num_layers}


class HybridQuantizerMLX(nn.Module):
    """PQ base quantization plus sparse RVQ refinement."""

    def __init__(self, config: RFSNConfig):
        super().__init__()
        self.config = config
        self.pq = ProductQuantizerMLX(config)
        self.rvq = ResidualVQMLX(config)

    def encode(self, vectors: mx.array) -> Tuple[mx.array, mx.array, mx.array, mx.array]:
        pq_codes, residuals = self.pq.quantize(vectors)
        rvq_codes, rvq_mask, rvq_offsets = self.rvq.encode(residuals, pq_codes)
        return pq_codes, rvq_codes, rvq_mask, rvq_offsets

    def decode(
        self,
        pq_codes: mx.array,
        rvq_codes: mx.array,
        rvq_mask: mx.array,
        rvq_offsets: mx.array,
    ) -> mx.array:
        base = self.pq.decode(pq_codes)
        correction = self.rvq.decode_correction(rvq_codes, rvq_mask, rvq_offsets)
        return base + correction

    def quantize_codebooks_for_ane(self) -> Dict[str, int]:
        pq_report = self.pq.quantize_codebooks_for_ane()
        rvq_report = self.rvq.quantize_codebooks_for_ane()
        return {
            "pq_quantized": pq_report["quantized_subspaces"],
            "rvq_quantized": rvq_report["quantized_layers"],
        }


def _compress_tensor_sequence(
    tensor: mx.array,
    quantizer: HybridQuantizerMLX,
) -> CompressedTensorMLX:
    num_tokens, num_heads, head_dim = tensor.shape
    flat = tensor.reshape(num_tokens * num_heads, head_dim)
    pq_codes, rvq_codes, rvq_mask, rvq_offsets = quantizer.encode(flat)
    return CompressedTensorMLX(
        pq_codes=pq_codes.reshape(num_tokens, num_heads, quantizer.config.num_subspaces),
        rvq_codes=rvq_codes,
        rvq_mask=rvq_mask.reshape(num_tokens, num_heads),
        rvq_offsets=rvq_offsets.astype(mx.int32),
        num_heads=num_heads,
    )


def _append_compressed_tensor(
    base: CompressedTensorMLX,
    addition: CompressedTensorMLX,
) -> CompressedTensorMLX:
    if base.num_heads != addition.num_heads:
        raise ValueError("Compressed tensors must have the same number of heads")

    offset_shift = base.num_tokens * base.num_heads
    shifted_offsets = addition.rvq_offsets + offset_shift if addition.rvq_offsets.shape[0] else addition.rvq_offsets
    return CompressedTensorMLX(
        pq_codes=_concat0(base.pq_codes, addition.pq_codes),
        rvq_codes=_concat0(base.rvq_codes, addition.rvq_codes),
        rvq_mask=_concat0(base.rvq_mask, addition.rvq_mask),
        rvq_offsets=_concat0(base.rvq_offsets, shifted_offsets.astype(mx.int32)),
        num_heads=base.num_heads,
    )


def _slice_compressed_tensor(
    tensor: CompressedTensorMLX,
    start: int,
    end: int,
) -> CompressedTensorMLX:
    if start < 0 or end < start or end > tensor.num_tokens:
        raise ValueError("Invalid slice for compressed tensor")

    flat_start = start * tensor.num_heads
    flat_end = end * tensor.num_heads
    offsets_np = _numpy_ints(tensor.rvq_offsets)
    select_np = np.where((offsets_np >= flat_start) & (offsets_np < flat_end))[0]

    if select_np.size:
        select_idx = mx.array(select_np, dtype=mx.int32)
        rvq_codes = mx.take(tensor.rvq_codes, select_idx, axis=0)
        rvq_offsets = mx.take(tensor.rvq_offsets, select_idx, axis=0) - flat_start
    else:
        rvq_codes = mx.zeros((0, tensor.rvq_codes.shape[1]), dtype=tensor.rvq_codes.dtype)
        rvq_offsets = mx.zeros((0,), dtype=mx.int32)

    return CompressedTensorMLX(
        pq_codes=tensor.pq_codes[start:end],
        rvq_codes=rvq_codes,
        rvq_mask=tensor.rvq_mask[start:end],
        rvq_offsets=rvq_offsets.astype(mx.int32),
        num_heads=tensor.num_heads,
    )


def _decode_compressed_tensor(
    tensor: CompressedTensorMLX,
    quantizer: HybridQuantizerMLX,
) -> mx.array:
    if tensor.num_tokens == 0:
        return mx.zeros((0, tensor.num_heads, quantizer.config.head_dim), dtype=mx.float16)
    flat_codes = tensor.pq_codes.reshape(tensor.num_tokens * tensor.num_heads, quantizer.config.num_subspaces)
    flat_mask = tensor.rvq_mask.reshape(tensor.num_tokens * tensor.num_heads)
    decoded = quantizer.decode(flat_codes, tensor.rvq_codes, flat_mask, tensor.rvq_offsets)
    return decoded.reshape(tensor.num_tokens, tensor.num_heads, quantizer.config.head_dim)