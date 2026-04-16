from __future__ import annotations

import logging
from typing import Optional, Sequence

import mlx.core as mx
import mlx.nn as nn

from quantization import HybridQuantizerMLX, _decode_compressed_tensor
from storage import CompressedTensorMLX, MASK_FILL_VALUE, RFSNConfig


logger = logging.getLogger("RFSN_ATTENTION")


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


def _dense_exact_attention(
    q: mx.array,
    keys: mx.array,
    values: mx.array,
    scale: float,
    causal: bool = True,
) -> mx.array:
    q_f32 = q.astype(mx.float32)
    k_f32 = keys.astype(mx.float32)
    v_f32 = values.astype(mx.float32)
    scores = mx.sum(q_f32[:, :, None, :, :] * k_f32[:, None, :, :, :], axis=-1)
    scores = mx.transpose(scores, (0, 3, 1, 2)) * scale

    if causal:
        q_positions = mx.arange(int(q.shape[1]), dtype=mx.int32)
        k_positions = mx.arange(int(keys.shape[1]), dtype=mx.int32)
        mask = k_positions[None, :] <= q_positions[:, None]
        scores = mx.where(
            mask[None, None, :, :],
            scores,
            mx.full(scores.shape, MASK_FILL_VALUE, dtype=mx.float32),
        )

    probs = mx.softmax(scores, axis=-1)
    values_by_head = mx.transpose(v_f32, (0, 2, 1, 3))
    context = mx.sum(probs[..., None] * values_by_head[:, :, None, :, :], axis=3)
    return mx.transpose(context, (0, 2, 1, 3)).astype(q.dtype)


def _gelu(x: mx.array) -> mx.array:
    return 0.5 * x * (1.0 + mx.tanh(0.7978845608 * (x + 0.044715 * (x ** 3))))


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