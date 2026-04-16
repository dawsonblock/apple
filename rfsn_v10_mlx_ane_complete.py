from __future__ import annotations

"""
RFSN v10.2 - Native MLX Apple Silicon implementation.

This module keeps the original component layout from the design draft while
hardening the behavior for the MLX runtime that is actually installed.
Quantized codebooks are capability-gated and attention is computed with a
blockwise online-softmax pass over exact or reconstructed tensors.
"""

import asyncio
import logging
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("RFSN_MLX_APPLE")

SUPPORTED_GROUP_SIZES = (128, 64, 32)
MASK_FILL_VALUE = -1e30


@dataclass
class RFSNConfig:
    """Hyperparameters and platform defaults."""

    hidden_dim: int = 4096
    num_heads: int = 32
    head_dim: int = 128
    num_layers: int = 32

    num_subspaces: int = 8
    pq_bits: int = 8
    subspace_dim: int = 16
    pq_codebook_dtype: str = "float16"

    num_rvq_layers: int = 4
    rvq_codebook_size: int = 4096
    rvq_sparsity_threshold: float = 0.005
    max_rvq_sparse: int = 64

    hot_capacity: int = 8192
    warm_capacity: int = 65536
    cold_capacity: int = 2_000_000

    block_size_seq: int = 128

    ane_quant_bits: int = 6
    ane_group_size: int = 128

    disk_cache_dir: str = "./rfsn_disk_cache"
    max_open_files: int = 8
    prefetch_throttle_s: float = 0.5

    cpu_threads: int = 4

    def __post_init__(self) -> None:
        expected = self.num_subspaces * self.subspace_dim
        if expected != self.head_dim:
            raise ValueError(
                "head_dim must equal num_subspaces * subspace_dim: "
                f"{self.head_dim} != {expected}"
            )


@dataclass
class QuantizedMatrix:
    """A quantized 2D tensor plus the metadata needed to recover it."""

    weights: mx.array
    scales: mx.array
    biases: mx.array
    bits: int
    group_size: int
    transposed: bool
    original_shape: Tuple[int, int]


@dataclass
class CompressedTensorMLX:
    """Compressed representation of a [T, H, D] tensor."""

    pq_codes: mx.array
    rvq_codes: mx.array
    rvq_mask: mx.array
    rvq_offsets: mx.array
    num_heads: int

    @property
    def num_tokens(self) -> int:
        return int(self.pq_codes.shape[0])

    @classmethod
    def empty(cls, config: RFSNConfig, num_heads: Optional[int] = None) -> "CompressedTensorMLX":
        heads = config.num_heads if num_heads is None else num_heads
        return cls(
            pq_codes=mx.zeros((0, heads, config.num_subspaces), dtype=mx.uint8),
            rvq_codes=mx.zeros((0, config.num_rvq_layers), dtype=mx.uint16),
            rvq_mask=mx.zeros((0, heads), dtype=mx.bool_),
            rvq_offsets=mx.zeros((0,), dtype=mx.int32),
            num_heads=heads,
        )


def _concat0(lhs: mx.array, rhs: mx.array) -> mx.array:
    if lhs.shape[0] == 0:
        return rhs
    if rhs.shape[0] == 0:
        return lhs
    return mx.concatenate([lhs, rhs], axis=0)


def _numpy_ints(values: mx.array) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


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


def _normalize_query_positions(
    query_positions: Optional[Sequence[int] | mx.array],
    batch_size: int,
    num_heads: int,
    default_position: int,
) -> mx.array:
    if query_positions is None:
        return mx.full((batch_size, num_heads), default_position, dtype=mx.int32)

    if hasattr(query_positions, "shape"):
        positions = query_positions
    else:
        positions = mx.array(query_positions, dtype=mx.int32)

    if len(positions.shape) == 1:
        if int(positions.shape[0]) != batch_size:
            raise ValueError("query_positions length must equal batch size")
        return mx.broadcast_to(positions.reshape(batch_size, 1).astype(mx.int32), (batch_size, num_heads))
    if tuple(int(dim) for dim in positions.shape) == (batch_size, num_heads):
        return positions.astype(mx.int32)
    raise ValueError("query_positions must have shape [B] or [B, H]")


def _squared_l2_distance(lhs: mx.array, rhs: mx.array) -> mx.array:
    lhs_sq = mx.sum(lhs * lhs, axis=1, keepdims=True)
    rhs_sq = mx.sum(rhs * rhs, axis=1, keepdims=True).T
    cross = lhs @ rhs.T
    return lhs_sq - (2.0 * cross) + rhs_sq


def _blockwise_exact_attention(
    q: mx.array,
    keys: mx.array,
    values: mx.array,
    scale: float,
    block_size: int,
    causal: bool = False,
    query_positions: Optional[Sequence[int] | mx.array] = None,
) -> mx.array:
    batch_size, num_heads, head_dim = q.shape
    seq_len = int(keys.shape[0])
    if seq_len == 0:
        return mx.zeros((batch_size, num_heads, head_dim), dtype=q.dtype)

    q_f32 = q.astype(mx.float32)
    running_max = mx.full((batch_size, num_heads), MASK_FILL_VALUE, dtype=mx.float32)
    running_sum = mx.zeros((batch_size, num_heads), dtype=mx.float32)
    out_accum = mx.zeros((batch_size, num_heads, head_dim), dtype=mx.float32)

    normalized_positions = _normalize_query_positions(
        query_positions,
        batch_size,
        num_heads,
        default_position=seq_len - 1,
    )

    for start in range(0, seq_len, block_size):
        end = min(start + block_size, seq_len)
        key_block = keys[start:end].astype(mx.float32)
        value_block = values[start:end].astype(mx.float32)

        scores = mx.sum(
            q_f32[:, :, None, :] * mx.swapaxes(key_block, 0, 1)[None, :, :, :],
            axis=-1,
        ) * scale

        if causal:
            positions = mx.arange(start, end, dtype=mx.int32)
            allowed = positions[None, None, :] <= normalized_positions[:, :, None]
            scores = mx.where(
                allowed,
                scores,
                mx.full(scores.shape, MASK_FILL_VALUE, dtype=mx.float32),
            )
        else:
            allowed = None

        block_max = mx.max(scores, axis=-1)
        new_max = mx.maximum(running_max, block_max)
        prev_rescale = mx.exp(running_max - new_max)

        running_sum = running_sum * prev_rescale
        out_accum = out_accum * prev_rescale[:, :, None]

        exp_scores = mx.exp(scores - new_max[:, :, None])
        if allowed is not None:
            exp_scores = exp_scores * allowed.astype(mx.float32)

        running_sum = running_sum + mx.sum(exp_scores, axis=-1)
        out_accum = out_accum + mx.sum(
            exp_scores[:, :, :, None] * mx.swapaxes(value_block, 0, 1)[None, :, :, :],
            axis=2,
        )
        running_max = new_max

    safe_sum = mx.where(running_sum > 0, running_sum, mx.ones_like(running_sum))
    return (out_accum / safe_sum[:, :, None]).astype(q.dtype)


def _compress_tensor_sequence(
    tensor: mx.array,
    quantizer: "HybridQuantizerMLX",
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
    quantizer: "HybridQuantizerMLX",
) -> mx.array:
    if tensor.num_tokens == 0:
        return mx.zeros((0, tensor.num_heads, quantizer.config.head_dim), dtype=mx.float16)
    flat_codes = tensor.pq_codes.reshape(tensor.num_tokens * tensor.num_heads, quantizer.config.num_subspaces)
    flat_mask = tensor.rvq_mask.reshape(tensor.num_tokens * tensor.num_heads)
    decoded = quantizer.decode(flat_codes, tensor.rvq_codes, flat_mask, tensor.rvq_offsets)
    return decoded.reshape(tensor.num_tokens, tensor.num_heads, quantizer.config.head_dim)


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


class RFSNHybridAttentionMLX(nn.Module):
    """Attention over compressed key/value stores using MLX-native kernels."""

    def __init__(self, config: RFSNConfig):
        super().__init__()
        self.config = config
        self.scale = config.head_dim ** -0.5
        self.block_size = config.block_size_seq

    def __call__(
        self,
        q: mx.array,
        key_store: CompressedTensorMLX,
        value_store: CompressedTensorMLX,
        quantizer: HybridQuantizerMLX,
        causal: bool = False,
        query_positions: Optional[Sequence[int] | mx.array] = None,
    ) -> mx.array:
        if key_store.num_tokens != value_store.num_tokens:
            raise ValueError("Key/value stores must have the same token count")
        decoded_keys = _decode_compressed_tensor(key_store, quantizer)
        decoded_values = _decode_compressed_tensor(value_store, quantizer)
        return _blockwise_exact_attention(
            q=q,
            keys=decoded_keys,
            values=decoded_values,
            scale=self.scale,
            block_size=self.block_size,
            causal=causal,
            query_positions=query_positions,
        )


class RFSNv10KVCacheMLX:
    """Tiered KV cache with exact hot storage and compressed warm/cold storage."""

    def __init__(self, config: RFSNConfig, layer_idx: int = 0):
        self.config = config
        self.layer_idx = layer_idx
        self.hot_capacity = config.hot_capacity
        self.warm_capacity = config.warm_capacity

        self.hot_keys = mx.zeros((0, config.num_heads, config.head_dim), dtype=mx.float16)
        self.hot_values = mx.zeros((0, config.num_heads, config.head_dim), dtype=mx.float16)

        self.warm_keys = CompressedTensorMLX.empty(config)
        self.warm_values = CompressedTensorMLX.empty(config)

        self.cold_chunk_paths: List[Path] = []
        self.cold_chunk_metadata: List[dict] = []

        self.num_hot = 0
        self.num_warm = 0
        self.num_cold = 0
        self.total_tokens = 0
        self.quantizer: Optional[HybridQuantizerMLX] = None

    @property
    def current_tier(self) -> str:
        if self.total_tokens < self.hot_capacity:
            return "hot"
        if self.total_tokens < self.hot_capacity + self.warm_capacity:
            return "warm"
        return "cold"

    def update(
        self,
        new_keys: mx.array,
        new_values: mx.array,
        quantizer: HybridQuantizerMLX,
        disk_dir: Optional[Path] = None,
    ) -> None:
        if new_keys.shape != new_values.shape:
            raise ValueError("new_keys and new_values must have identical shapes")

        self.quantizer = quantizer
        num_tokens = int(new_keys.shape[0])
        offset = 0

        hot_space = max(self.hot_capacity - self.num_hot, 0)
        if hot_space > 0 and offset < num_tokens:
            take = min(hot_space, num_tokens - offset)
            self.hot_keys = _concat0(self.hot_keys, new_keys[offset : offset + take])
            self.hot_values = _concat0(self.hot_values, new_values[offset : offset + take])
            self.num_hot += take
            offset += take

        warm_space = max(self.warm_capacity - self.num_warm, 0)
        if warm_space > 0 and offset < num_tokens:
            take = min(warm_space, num_tokens - offset)
            self._add_to_warm(new_keys[offset : offset + take], new_values[offset : offset + take], quantizer)
            offset += take

        if offset < num_tokens:
            self._add_to_cold(new_keys[offset:], new_values[offset:], quantizer, disk_dir)

        self.total_tokens += num_tokens

    def _add_to_warm(
        self,
        keys: mx.array,
        values: mx.array,
        quantizer: HybridQuantizerMLX,
    ) -> None:
        compressed_keys = _compress_tensor_sequence(keys, quantizer)
        compressed_values = _compress_tensor_sequence(values, quantizer)
        self.warm_keys = _append_compressed_tensor(self.warm_keys, compressed_keys)
        self.warm_values = _append_compressed_tensor(self.warm_values, compressed_values)
        self.num_warm += int(keys.shape[0])

    def _add_to_cold(
        self,
        keys: mx.array,
        values: mx.array,
        quantizer: HybridQuantizerMLX,
        disk_dir: Optional[Path],
    ) -> None:
        disk_root = Path(self.config.disk_cache_dir) if disk_dir is None else disk_dir
        disk_root.mkdir(parents=True, exist_ok=True)

        compressed_keys = _compress_tensor_sequence(keys, quantizer)
        compressed_values = _compress_tensor_sequence(values, quantizer)

        chunk_id = len(self.cold_chunk_paths)
        chunk_path = disk_root / f"layer{self.layer_idx}_chunk{chunk_id}.npz"
        start_token = self.num_hot + self.num_warm + self.num_cold
        num_tokens = int(keys.shape[0])
        np.savez_compressed(
            chunk_path,
            num_tokens=np.array([num_tokens], dtype=np.int32),
            num_heads=np.array([keys.shape[1]], dtype=np.int32),
            key_pq_codes=np.asarray(compressed_keys.pq_codes),
            key_rvq_codes=np.asarray(compressed_keys.rvq_codes),
            key_rvq_mask=np.asarray(compressed_keys.rvq_mask),
            key_rvq_offsets=np.asarray(compressed_keys.rvq_offsets),
            value_pq_codes=np.asarray(compressed_values.pq_codes),
            value_rvq_codes=np.asarray(compressed_values.rvq_codes),
            value_rvq_mask=np.asarray(compressed_values.rvq_mask),
            value_rvq_offsets=np.asarray(compressed_values.rvq_offsets),
        )

        self.cold_chunk_paths.append(chunk_path)
        self.cold_chunk_metadata.append(
            {
                "chunk_id": chunk_id,
                "start_token": start_token,
                "end_token": start_token + num_tokens,
                "num_tokens": num_tokens,
                "path": str(chunk_path),
            }
        )
        self.num_cold += num_tokens

    def load_cold_chunk(self, chunk_id: int) -> Dict[str, mx.array]:
        path = self.cold_chunk_paths[chunk_id]
        data = np.load(path)
        return {key: mx.array(data[key]) for key in data.files}

    def _chunk_store(self, loaded: Dict[str, mx.array], prefix: str) -> CompressedTensorMLX:
        num_heads = int(np.asarray(loaded["num_heads"])[0])
        return CompressedTensorMLX(
            pq_codes=loaded[f"{prefix}_pq_codes"],
            rvq_codes=loaded[f"{prefix}_rvq_codes"],
            rvq_mask=loaded[f"{prefix}_rvq_mask"],
            rvq_offsets=loaded[f"{prefix}_rvq_offsets"].astype(mx.int32),
            num_heads=num_heads,
        )

    def _load_cold_exact(
        self,
        router: Optional["AsyncHierarchicalRouterMLX"] = None,
        current_position: Optional[int] = None,
        context_window: Optional[int] = None,
    ) -> Tuple[List[mx.array], List[mx.array]]:
        if self.quantizer is None or not self.cold_chunk_paths:
            return [], []

        if router is not None and current_position is not None and context_window is not None:
            router.prefetch_sync(current_position=current_position, context_window=context_window)

        cold_keys: List[mx.array] = []
        cold_values: List[mx.array] = []
        for chunk_id in range(len(self.cold_chunk_paths)):
            loaded = router.get_chunk(chunk_id) if router is not None else None
            if loaded is None:
                loaded = self.load_cold_chunk(chunk_id)
            key_store = self._chunk_store(loaded, "key")
            value_store = self._chunk_store(loaded, "value")
            cold_keys.append(_decode_compressed_tensor(key_store, self.quantizer))
            cold_values.append(_decode_compressed_tensor(value_store, self.quantizer))
        return cold_keys, cold_values

    def attention_forward(
        self,
        q: mx.array,
        pq_codebook: Optional[mx.array] = None,
        rvq_codebook: Optional[mx.array] = None,
        causal: bool = True,
        query_positions: Optional[Sequence[int] | mx.array] = None,
        router: Optional["AsyncHierarchicalRouterMLX"] = None,
        current_position: Optional[int] = None,
        context_window: Optional[int] = None,
    ) -> mx.array:
        del pq_codebook, rvq_codebook
        if self.quantizer is None:
            raise RuntimeError("Cache cannot run attention before update() provides a quantizer")

        key_segments: List[mx.array] = []
        value_segments: List[mx.array] = []

        if self.num_hot:
            key_segments.append(self.hot_keys)
            value_segments.append(self.hot_values)
        if self.num_warm:
            key_segments.append(_decode_compressed_tensor(self.warm_keys, self.quantizer))
            value_segments.append(_decode_compressed_tensor(self.warm_values, self.quantizer))

        cold_keys, cold_values = self._load_cold_exact(
            router=router,
            current_position=current_position,
            context_window=context_window,
        )
        key_segments.extend(cold_keys)
        value_segments.extend(cold_values)

        if not key_segments:
            return mx.zeros_like(q)

        keys = key_segments[0] if len(key_segments) == 1 else mx.concatenate(key_segments, axis=0)
        values = value_segments[0] if len(value_segments) == 1 else mx.concatenate(value_segments, axis=0)
        return _blockwise_exact_attention(
            q=q,
            keys=keys,
            values=values,
            scale=self.config.head_dim ** -0.5,
            block_size=self.config.block_size_seq,
            causal=causal,
            query_positions=query_positions,
        )

    def memory_usage_bytes(self) -> Dict[str, int]:
        hot_bytes = self.num_hot * self.config.num_heads * self.config.head_dim * 2 * 2
        warm_pq_bytes = self.num_warm * self.config.num_heads * self.config.num_subspaces * 2
        warm_rvq_bytes = int(self.warm_keys.rvq_codes.shape[0] + self.warm_values.rvq_codes.shape[0])
        warm_rvq_bytes *= self.config.num_rvq_layers * 2
        return {
            "hot_bytes": hot_bytes,
            "warm_pq_bytes": warm_pq_bytes,
            "warm_rvq_bytes": warm_rvq_bytes,
            "cold_chunks": len(self.cold_chunk_paths),
            "cold_tokens": self.num_cold,
        }


class AsyncHierarchicalRouterMLX:
    """Simple chunk prefetcher for cold cache files."""

    def __init__(self, config: RFSNConfig, layer_idx: int = 0):
        self.config = config
        self.layer_idx = layer_idx
        self.disk_dir = Path(config.disk_cache_dir)
        self.throttle = config.prefetch_throttle_s
        self._cache: Dict[int, Dict[str, mx.array]] = {}
        self._max_cache_size = 16
        self._pending_prefetch: set[int] = set()

    async def predict_and_prefetch(
        self,
        current_position: int,
        context_window: int,
        top_k: int = 2,
    ) -> List[int]:
        chunk_size = max(self.config.block_size_seq, 1)
        start_chunk = max(0, (current_position - context_window) // chunk_size)
        end_chunk = max(start_chunk, (current_position + context_window) // chunk_size)

        candidates: List[int] = []
        for chunk_id in range(start_chunk, end_chunk + 1):
            path = self.disk_dir / f"layer{self.layer_idx}_chunk{chunk_id}.npz"
            if path.exists():
                candidates.append(chunk_id)

        candidates.sort(key=lambda cid: abs(cid * chunk_size - current_position))
        loaded: List[int] = []
        for chunk_id in candidates[:top_k]:
            if chunk_id in self._cache:
                loaded.append(chunk_id)
                continue
            if chunk_id in self._pending_prefetch:
                continue
            self._pending_prefetch.add(chunk_id)
            await self._load_chunk(chunk_id)
            self._pending_prefetch.discard(chunk_id)
            loaded.append(chunk_id)
            if self.throttle > 0:
                await asyncio.sleep(self.throttle)

        while len(self._cache) > self._max_cache_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        return loaded

    def prefetch_sync(self, current_position: int, context_window: int, top_k: int = 2) -> List[int]:
        try:
            return asyncio.run(
                self.predict_and_prefetch(
                    current_position=current_position,
                    context_window=context_window,
                    top_k=top_k,
                )
            )
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(
                    self.predict_and_prefetch(
                        current_position=current_position,
                        context_window=context_window,
                        top_k=top_k,
                    )
                )
            finally:
                loop.close()

    async def _load_chunk(self, chunk_id: int) -> None:
        loop = asyncio.get_running_loop()
        loaded = await loop.run_in_executor(None, self._load_chunk_sync, chunk_id)
        self._cache[chunk_id] = loaded

    def _load_chunk_sync(self, chunk_id: int) -> Dict[str, mx.array]:
        path = self.disk_dir / f"layer{self.layer_idx}_chunk{chunk_id}.npz"
        if not path.exists():
            return {}
        data = np.load(path)
        return {key: mx.array(data[key]) for key in data.files}

    def get_chunk(self, chunk_id: int) -> Optional[Dict[str, mx.array]]:
        return self._cache.get(chunk_id)


def calibrate_quantizer(
    quantizer: HybridQuantizerMLX,
    calibration_vectors: mx.array,
    num_iterations: int = 20,
) -> Dict[str, List[float]]:
    logger.info("Calibrating PQ codebooks with %d vectors", int(calibration_vectors.shape[0]))
    num_vectors = int(calibration_vectors.shape[0])
    codebook_size = quantizer.pq.codebook_size
    sub_dim = quantizer.config.subspace_dim

    for subspace in range(quantizer.config.num_subspaces):
        start = subspace * sub_dim
        end = start + sub_dim
        sample_idx = mx.random.randint(0, num_vectors, shape=(codebook_size,))
        quantizer.pq.codebooks[subspace] = mx.take(calibration_vectors[:, start:end], sample_idx, axis=0).astype(mx.float16)

    metrics = {"avg_distortion": [], "sparsity_fraction": []}

    calibration_np = np.asarray(calibration_vectors, dtype=np.float32)
    for iteration in range(num_iterations):
        pq_codes, residuals = quantizer.pq.quantize(calibration_vectors)
        pq_codes_np = np.asarray(pq_codes, dtype=np.int64)

        for subspace in range(quantizer.config.num_subspaces):
            start = subspace * sub_dim
            end = start + sub_dim
            new_codebook = np.asarray(quantizer.pq.codebooks[subspace], dtype=np.float32).copy()
            sub_vectors = calibration_np[:, start:end]
            sub_codes = pq_codes_np[:, subspace]
            for centroid_idx in range(codebook_size):
                member_idx = np.where(sub_codes == centroid_idx)[0]
                if member_idx.size:
                    new_codebook[centroid_idx] = sub_vectors[member_idx].mean(axis=0)
            quantizer.pq.codebooks[subspace] = mx.array(new_codebook, dtype=mx.float16)

        reconstructed = quantizer.pq.decode(pq_codes)
        distortion = mx.mean((calibration_vectors.astype(mx.float32) - reconstructed.astype(mx.float32)) ** 2)
        residual_norms = mx.sqrt(mx.sum(residuals.astype(mx.float32) ** 2, axis=1))
        sparsity = mx.mean((residual_norms < quantizer.config.rvq_sparsity_threshold).astype(mx.float32))
        metrics["avg_distortion"].append(float(distortion))
        metrics["sparsity_fraction"].append(float(sparsity))

        if (iteration + 1) % max(1, num_iterations // 4) == 0:
            logger.info(
                "  iter %d/%d distortion=%.6f sparsity=%.3f",
                iteration + 1,
                num_iterations,
                float(distortion),
                float(sparsity),
            )

    return metrics


def run_tests() -> bool:
    config = RFSNConfig(
        hidden_dim=512,
        num_heads=4,
        head_dim=128,
        num_layers=2,
        num_subspaces=8,
        subspace_dim=16,
        num_rvq_layers=3,
        rvq_codebook_size=256,
        rvq_sparsity_threshold=0.25,
        hot_capacity=16,
        warm_capacity=24,
        cold_capacity=128,
        block_size_seq=16,
        disk_cache_dir="./test_rfsn_disk_cache",
        prefetch_throttle_s=0.0,
    )

    try:
        if hasattr(mx.random, "seed"):
            mx.random.seed(7)
        np.random.seed(7)
    except Exception:
        pass

    logger.info("=" * 60)
    logger.info("RFSN v10.2 MLX Apple Silicon test suite")
    logger.info("=" * 60)

    hq = HybridQuantizerMLX(config)
    attention = RFSNHybridAttentionMLX(config)
    tests: List[Tuple[str, callable]] = []

    def test_pq_roundtrip() -> None:
        vectors = mx.random.normal(shape=(96, config.head_dim), dtype=mx.float32).astype(mx.float16)
        codes, residuals = hq.pq.quantize(vectors)
        reconstructed = hq.pq.decode(codes)
        mse = float(mx.mean((vectors.astype(mx.float32) - reconstructed.astype(mx.float32)) ** 2))
        assert codes.shape == (96, config.num_subspaces)
        assert residuals.shape == vectors.shape
        assert np.isfinite(mse)
        logger.info("  PQ MSE: %.6f", mse)

    def test_rvq_sparsity() -> None:
        sparse = mx.random.normal(shape=(32, config.head_dim), dtype=mx.float32).astype(mx.float16) * 0.01
        dense = mx.random.normal(shape=(32, config.head_dim), dtype=mx.float32).astype(mx.float16)
        residuals = mx.concatenate([sparse, dense], axis=0)
        rvq_codes, rvq_mask, rvq_offsets = hq.rvq.encode(residuals)
        inactive_fraction = 1.0 - float(mx.mean(rvq_mask.astype(mx.float32)))
        assert inactive_fraction > 0.2
        assert rvq_offsets.shape[0] == rvq_codes.shape[0]
        logger.info("  RVQ inactive fraction: %.3f", inactive_fraction)

    def test_hybrid_roundtrip() -> None:
        vectors = mx.random.normal(shape=(96, config.head_dim), dtype=mx.float32).astype(mx.float16)
        pq_codes, residuals = hq.pq.quantize(vectors)
        pq_recon = hq.pq.decode(pq_codes)
        pq_mse = float(mx.mean((vectors.astype(mx.float32) - pq_recon.astype(mx.float32)) ** 2))
        hybrid = hq.decode(*hq.encode(vectors))
        hybrid_mse = float(mx.mean((vectors.astype(mx.float32) - hybrid.astype(mx.float32)) ** 2))
        assert hybrid_mse <= pq_mse + 1e-5
        logger.info("  Hybrid MSE %.6f <= PQ MSE %.6f", hybrid_mse, pq_mse)

    def test_quantized_fast_path() -> None:
        report = hq.quantize_codebooks_for_ane()
        vectors = mx.random.normal(shape=(32, config.head_dim), dtype=mx.float32).astype(mx.float16)
        decoded = hq.decode(*hq.encode(vectors))
        mse = float(mx.mean((vectors.astype(mx.float32) - decoded.astype(mx.float32)) ** 2))
        assert report["rvq_quantized"] >= 1
        assert report["pq_quantized"] >= 1
        assert np.isfinite(mse)
        logger.info(
            "  Quantized fast path enabled for PQ=%d RVQ=%d",
            report["pq_quantized"],
            report["rvq_quantized"],
        )

    def test_hybrid_attention_matches_reference() -> None:
        q = mx.random.normal(shape=(2, config.num_heads, config.head_dim), dtype=mx.float32).astype(mx.float16)
        keys = mx.random.normal(shape=(24, config.num_heads, config.head_dim), dtype=mx.float32).astype(mx.float16)
        values = mx.random.normal(shape=(24, config.num_heads, config.head_dim), dtype=mx.float32).astype(mx.float16)
        key_store = _compress_tensor_sequence(keys, hq)
        value_store = _compress_tensor_sequence(values, hq)
        decoded_keys = _decode_compressed_tensor(key_store, hq)
        decoded_values = _decode_compressed_tensor(value_store, hq)
        query_positions = mx.array([12, 18], dtype=mx.int32)
        out = attention(
            q=q,
            key_store=key_store,
            value_store=value_store,
            quantizer=hq,
            causal=True,
            query_positions=query_positions,
        )
        ref = _blockwise_exact_attention(
            q=q,
            keys=decoded_keys,
            values=decoded_values,
            scale=config.head_dim ** -0.5,
            block_size=config.block_size_seq,
            causal=True,
            query_positions=query_positions,
        )
        max_diff = float(mx.max(mx.abs(out.astype(mx.float32) - ref.astype(mx.float32))))
        assert max_diff < 1e-4
        logger.info("  Hybrid attention max diff: %.2e", max_diff)

    def test_kv_cache_tiers_and_attention() -> None:
        cache = RFSNv10KVCacheMLX(config, layer_idx=0)
        keys = mx.random.normal(shape=(60, config.num_heads, config.head_dim), dtype=mx.float32).astype(mx.float16)
        values = mx.random.normal(shape=(60, config.num_heads, config.head_dim), dtype=mx.float32).astype(mx.float16)
        cache.update(keys, values, hq, disk_dir=Path(config.disk_cache_dir))

        assert cache.num_hot == config.hot_capacity
        assert cache.num_warm == config.warm_capacity
        assert cache.num_cold == 60 - config.hot_capacity - config.warm_capacity
        assert len(cache.cold_chunk_paths) == 1

        q = mx.random.normal(shape=(1, config.num_heads, config.head_dim), dtype=mx.float32).astype(mx.float16)
        cache_out = cache.attention_forward(
            q,
            causal=True,
            query_positions=mx.array([59], dtype=mx.int32),
        )

        cold_loaded = cache.load_cold_chunk(0)
        cold_keys = _decode_compressed_tensor(cache._chunk_store(cold_loaded, "key"), hq)
        cold_values = _decode_compressed_tensor(cache._chunk_store(cold_loaded, "value"), hq)
        exact_keys = mx.concatenate([cache.hot_keys, _decode_compressed_tensor(cache.warm_keys, hq), cold_keys], axis=0)
        exact_values = mx.concatenate([cache.hot_values, _decode_compressed_tensor(cache.warm_values, hq), cold_values], axis=0)
        ref = _blockwise_exact_attention(
            q=q,
            keys=exact_keys,
            values=exact_values,
            scale=config.head_dim ** -0.5,
            block_size=config.block_size_seq,
            causal=True,
            query_positions=mx.array([59], dtype=mx.int32),
        )
        max_diff = float(mx.max(mx.abs(cache_out.astype(mx.float32) - ref.astype(mx.float32))))
        assert max_diff < 1e-4
        mem = cache.memory_usage_bytes()
        assert mem["cold_chunks"] == 1
        logger.info("  Cache tiering validated, attention max diff: %.2e", max_diff)

    def test_calibration() -> None:
        fresh = HybridQuantizerMLX(config)
        calibration_vectors = mx.random.normal(shape=(256, config.head_dim), dtype=mx.float32).astype(mx.float16)
        metrics = calibrate_quantizer(fresh, calibration_vectors, num_iterations=6)
        assert metrics["avg_distortion"][-1] <= metrics["avg_distortion"][0]
        logger.info(
            "  Calibration distortion %.6f -> %.6f",
            metrics["avg_distortion"][0],
            metrics["avg_distortion"][-1],
        )

    def test_async_router() -> None:
        cache = RFSNv10KVCacheMLX(config, layer_idx=0)
        keys = mx.random.normal(shape=(48, config.num_heads, config.head_dim), dtype=mx.float32).astype(mx.float16)
        values = mx.random.normal(shape=(48, config.num_heads, config.head_dim), dtype=mx.float32).astype(mx.float16)
        cache.update(keys, values, hq, disk_dir=Path(config.disk_cache_dir))
        router = AsyncHierarchicalRouterMLX(config, layer_idx=0)
        loaded_ids = router.prefetch_sync(current_position=0, context_window=1024, top_k=1)
        assert loaded_ids
        assert router.get_chunk(loaded_ids[0]) is not None
        logger.info("  Router prefetched chunks: %s", loaded_ids)

    tests.extend(
        [
            ("PQ roundtrip", test_pq_roundtrip),
            ("RVQ sparsity", test_rvq_sparsity),
            ("Hybrid roundtrip", test_hybrid_roundtrip),
            ("Quantized fast path", test_quantized_fast_path),
            ("Hybrid attention", test_hybrid_attention_matches_reference),
            ("KV cache tiers", test_kv_cache_tiers_and_attention),
            ("Calibration", test_calibration),
            ("Async router", test_async_router),
        ]
    )

    all_passed = True
    try:
        for name, fn in tests:
            logger.info("\n[Test] %s", name)
            start = time.time()
            try:
                fn()
                logger.info("  PASS (%.1f ms)", (time.time() - start) * 1000.0)
            except Exception:
                all_passed = False
                logger.exception("  FAIL")
    finally:
        if Path(config.disk_cache_dir).exists():
            shutil.rmtree(config.disk_cache_dir)

    logger.info("\n%s", "=" * 60)
    if all_passed:
        logger.info("All MLX Apple Silicon tests passed")
    else:
        logger.info("One or more MLX tests failed")
    logger.info("%s", "=" * 60)
    return all_passed


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")

    logger.info("RFSN v10.2 - MLX Apple Silicon implementation")
    logger.info("MLX default device: %s", mx.default_device() if hasattr(mx, "default_device") else "unknown")

    ok = run_tests()
    if not ok:
        sys.exit(1)