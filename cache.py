from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import mlx.core as mx
import numpy as np

from attention import _blockwise_exact_attention
from quantization import (
    HybridQuantizerMLX,
    _append_compressed_tensor,
    _compress_tensor_sequence,
    _decode_compressed_tensor,
    _slice_compressed_tensor,
)
from storage import (
    AsyncHierarchicalRouterMLX,
    CompressedTensorMLX,
    RFSNConfig,
    _concat0,
    chunk_file_path,
    compressed_tensor_from_loaded,
    load_compressed_chunk,
    save_compressed_chunk,
)


logger = logging.getLogger("RFSN_CACHE")


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
        self.cold_chunk_paths_by_id: Dict[int, Path] = {}
        self.cold_chunk_metadata: List[dict] = []

        self.num_hot = 0
        self.num_warm = 0
        self.num_cold = 0
        self.total_tokens = 0
        self.quantizer: Optional[HybridQuantizerMLX] = None
        self.last_access_stats = self._empty_access_stats()

    def _empty_access_stats(self) -> Dict[str, int]:
        return {
            "window_start": 0,
            "window_end": 0,
            "window_tokens": 0,
            "context_window": -1,
            "query_tokens": 0,
            "used_router": 0,
            "hot_tokens_materialized": 0,
            "warm_tokens_materialized": 0,
            "cold_tokens_materialized": 0,
            "reconstructed_tokens": 0,
            "warm_chunk_decodes": 0,
            "cold_chunk_decodes": 0,
            "cold_chunks_touched": 0,
            "cold_chunk_cache_hits": 0,
            "cold_chunk_cache_misses": 0,
        }

    def get_last_access_stats(self) -> Dict[str, int]:
        return dict(self.last_access_stats)

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

        chunk_tokens = max(1, self.config.block_size_seq)
        start_token = self.num_hot + self.num_warm + self.num_cold
        for offset in range(0, int(keys.shape[0]), chunk_tokens):
            end = min(offset + chunk_tokens, int(keys.shape[0]))
            chunk_keys = keys[offset:end]
            chunk_values = values[offset:end]
            chunk_len = int(chunk_keys.shape[0])
            chunk_start_token = start_token + offset
            chunk_id = chunk_start_token // chunk_tokens
            chunk_path = chunk_file_path(disk_root, self.layer_idx, chunk_id)

            compressed_keys = _compress_tensor_sequence(chunk_keys, quantizer)
            compressed_values = _compress_tensor_sequence(chunk_values, quantizer)
            existing_metadata = next((item for item in self.cold_chunk_metadata if item["chunk_id"] == chunk_id), None)
            if existing_metadata is not None and chunk_id in self.cold_chunk_paths_by_id:
                if chunk_start_token != existing_metadata["end_token"]:
                    raise ValueError(
                        "Cold chunk append must be contiguous with existing metadata: "
                        f"chunk_id={chunk_id} start={chunk_start_token} expected={existing_metadata['end_token']}"
                    )
                loaded = load_compressed_chunk(self.cold_chunk_paths_by_id[chunk_id])
                merged_keys = _append_compressed_tensor(
                    compressed_tensor_from_loaded(loaded, "key"),
                    compressed_keys,
                )
                merged_values = _append_compressed_tensor(
                    compressed_tensor_from_loaded(loaded, "value"),
                    compressed_values,
                )
                save_compressed_chunk(chunk_path, merged_keys, merged_values)
                existing_metadata["end_token"] += chunk_len
                existing_metadata["num_tokens"] += chunk_len
                existing_metadata["path"] = str(chunk_path)
            else:
                save_compressed_chunk(chunk_path, compressed_keys, compressed_values)
                self.cold_chunk_paths.append(chunk_path)
                self.cold_chunk_paths_by_id[chunk_id] = chunk_path
                self.cold_chunk_metadata.append(
                    {
                        "chunk_id": chunk_id,
                        "start_token": chunk_start_token,
                        "end_token": chunk_start_token + chunk_len,
                        "num_tokens": chunk_len,
                        "path": str(chunk_path),
                    }
                )

        self.num_cold += int(keys.shape[0])

    def load_cold_chunk(self, chunk_id: int) -> Dict[str, mx.array]:
        return load_compressed_chunk(self.cold_chunk_paths_by_id[chunk_id])

    def _chunk_store(self, loaded: Dict[str, mx.array], prefix: str) -> CompressedTensorMLX:
        return compressed_tensor_from_loaded(loaded, prefix)

    def _max_query_position(
        self,
        query_positions: Optional[Sequence[int] | mx.array],
        current_position: Optional[int],
    ) -> int:
        if query_positions is not None:
            return int(np.max(np.asarray(query_positions, dtype=np.int64)))
        if current_position is not None:
            return current_position
        return max(self.total_tokens - 1, 0)

    def _window_bounds(
        self,
        query_positions: Optional[Sequence[int] | mx.array],
        context_window: Optional[int],
        current_position: Optional[int],
    ) -> Tuple[int, int, Optional[mx.array]]:
        def to_query_array(value: Optional[Any]) -> Optional[mx.array]:
            if value is None:
                return None
            if hasattr(value, "astype"):
                return value.astype(mx.int32)
            return mx.array(value, dtype=mx.int32)

        if self.total_tokens == 0:
            return 0, 0, to_query_array(query_positions)

        max_query = self._max_query_position(query_positions, current_position)
        window_end = min(max_query + 1, self.total_tokens)
        if context_window is None:
            window_start = 0
        else:
            window_start = max(0, window_end - context_window)

        if query_positions is None:
            return window_start, window_end, None

        adjusted = to_query_array(query_positions)
        if adjusted is None:
            return window_start, window_end, None
        adjusted = adjusted - window_start
        adjusted = mx.maximum(adjusted, mx.full(adjusted.shape, -1, dtype=mx.int32))
        return window_start, window_end, adjusted.astype(mx.int32)

    def _overlap(self, start: int, end: int, span_start: int, span_end: int) -> Optional[Tuple[int, int]]:
        overlap_start = max(start, span_start)
        overlap_end = min(end, span_end)
        if overlap_start >= overlap_end:
            return None
        return overlap_start, overlap_end

    def _hot_window_exact(
        self,
        start: int,
        end: int,
        stats: Dict[str, int],
    ) -> Tuple[List[mx.array], List[mx.array]]:
        overlap = self._overlap(start, end, 0, self.num_hot)
        if overlap is None:
            return [], []
        overlap_start, overlap_end = overlap
        stats["hot_tokens_materialized"] += overlap_end - overlap_start
        return [self.hot_keys[overlap_start:overlap_end]], [self.hot_values[overlap_start:overlap_end]]

    def _warm_window_exact(
        self,
        start: int,
        end: int,
        stats: Dict[str, int],
    ) -> Tuple[List[mx.array], List[mx.array]]:
        warm_start_global = self.num_hot
        warm_end_global = self.num_hot + self.num_warm
        overlap = self._overlap(start, end, warm_start_global, warm_end_global)
        if overlap is None or self.quantizer is None:
            return [], []

        overlap_start, overlap_end = overlap
        local_start = overlap_start - warm_start_global
        local_end = overlap_end - warm_start_global
        key_chunks: List[mx.array] = []
        value_chunks: List[mx.array] = []
        chunk_tokens = max(1, self.config.block_size_seq)
        for chunk_start in range(local_start, local_end, chunk_tokens):
            chunk_end = min(chunk_start + chunk_tokens, local_end)
            chunk_len = chunk_end - chunk_start
            key_store = _slice_compressed_tensor(self.warm_keys, chunk_start, chunk_end)
            value_store = _slice_compressed_tensor(self.warm_values, chunk_start, chunk_end)
            key_chunks.append(_decode_compressed_tensor(key_store, self.quantizer))
            value_chunks.append(_decode_compressed_tensor(value_store, self.quantizer))
            stats["warm_tokens_materialized"] += chunk_len
            stats["reconstructed_tokens"] += chunk_len
            stats["warm_chunk_decodes"] += 1
        return key_chunks, value_chunks

    def _cold_window_exact(
        self,
        start: int,
        end: int,
        stats: Dict[str, int],
        router: Optional[AsyncHierarchicalRouterMLX] = None,
        current_position: Optional[int] = None,
        context_window: Optional[int] = None,
    ) -> Tuple[List[mx.array], List[mx.array]]:
        if self.quantizer is None or not self.cold_chunk_paths:
            return [], []

        if router is not None and current_position is not None and context_window is not None:
            chunk_tokens = max(1, self.config.block_size_seq)
            prefetch_top_k = max(1, (context_window + chunk_tokens - 1) // chunk_tokens)
            router.prefetch_sync(
                current_position=current_position,
                context_window=context_window,
                top_k=prefetch_top_k,
            )

        cold_keys: List[mx.array] = []
        cold_values: List[mx.array] = []
        for metadata in self.cold_chunk_metadata:
            overlap = self._overlap(start, end, metadata["start_token"], metadata["end_token"])
            if overlap is None:
                continue

            chunk_id = metadata["chunk_id"]
            loaded = router.get_chunk(chunk_id) if router is not None else None
            if loaded is None:
                loaded = self.load_cold_chunk(chunk_id)
                if router is not None:
                    stats["cold_chunk_cache_misses"] += 1
            else:
                stats["cold_chunk_cache_hits"] += 1

            local_start = overlap[0] - metadata["start_token"]
            local_end = overlap[1] - metadata["start_token"]
            token_count = local_end - local_start
            key_store = _slice_compressed_tensor(self._chunk_store(loaded, "key"), local_start, local_end)
            value_store = _slice_compressed_tensor(self._chunk_store(loaded, "value"), local_start, local_end)
            cold_keys.append(_decode_compressed_tensor(key_store, self.quantizer))
            cold_values.append(_decode_compressed_tensor(value_store, self.quantizer))
            stats["cold_tokens_materialized"] += token_count
            stats["reconstructed_tokens"] += token_count
            stats["cold_chunk_decodes"] += 1
            stats["cold_chunks_touched"] += 1

        return cold_keys, cold_values

    def materialize_window(
        self,
        query_positions: Optional[Sequence[int] | mx.array] = None,
        context_window: Optional[int] = None,
        router: Optional[AsyncHierarchicalRouterMLX] = None,
        current_position: Optional[int] = None,
    ) -> Tuple[mx.array, mx.array, Optional[mx.array]]:
        if self.quantizer is None:
            raise RuntimeError("Cache cannot materialize values before update() provides a quantizer")

        window_start, window_end, adjusted_positions = self._window_bounds(
            query_positions=query_positions,
            context_window=context_window,
            current_position=current_position,
        )
        stats = self._empty_access_stats()
        stats["window_start"] = window_start
        stats["window_end"] = window_end
        stats["window_tokens"] = max(0, window_end - window_start)
        stats["context_window"] = -1 if context_window is None else context_window
        stats["used_router"] = int(router is not None)

        key_segments: List[mx.array] = []
        value_segments: List[mx.array] = []
        hot_keys, hot_values = self._hot_window_exact(window_start, window_end, stats)
        warm_keys, warm_values = self._warm_window_exact(window_start, window_end, stats)
        cold_keys, cold_values = self._cold_window_exact(
            window_start,
            window_end,
            stats,
            router=router,
            current_position=current_position,
            context_window=context_window,
        )
        key_segments.extend(hot_keys)
        value_segments.extend(hot_values)
        key_segments.extend(warm_keys)
        value_segments.extend(warm_values)
        key_segments.extend(cold_keys)
        value_segments.extend(cold_values)

        if not key_segments:
            empty = mx.zeros((0, self.config.num_heads, self.config.head_dim), dtype=mx.float16)
            self.last_access_stats = dict(stats)
            return empty, empty, adjusted_positions

        keys = key_segments[0] if len(key_segments) == 1 else mx.concatenate(key_segments, axis=0)
        values = value_segments[0] if len(value_segments) == 1 else mx.concatenate(value_segments, axis=0)
        self.last_access_stats = dict(stats)
        return keys, values, adjusted_positions

    def attention_forward(
        self,
        q: mx.array,
        pq_codebook: Optional[mx.array] = None,
        rvq_codebook: Optional[mx.array] = None,
        causal: bool = True,
        query_positions: Optional[Sequence[int] | mx.array] = None,
        router: Optional[AsyncHierarchicalRouterMLX] = None,
        current_position: Optional[int] = None,
        context_window: Optional[int] = None,
    ) -> mx.array:
        del pq_codebook, rvq_codebook
        if self.quantizer is None:
            raise RuntimeError("Cache cannot run attention before update() provides a quantizer")

        keys, values, adjusted_positions = self.materialize_window(
            query_positions=query_positions,
            context_window=context_window,
            router=router,
            current_position=current_position,
        )
        self.last_access_stats["query_tokens"] = int(q.shape[0])
        if int(keys.shape[0]) == 0:
            return mx.zeros_like(q)

        return _blockwise_exact_attention(
            q=q,
            keys=keys,
            values=values,
            scale=self.config.head_dim ** -0.5,
            block_size=self.config.block_size_seq,
            causal=causal,
            query_positions=adjusted_positions,
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