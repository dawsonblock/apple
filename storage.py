from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import numpy as np


logger = logging.getLogger("RFSN_STORAGE")

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
    original_shape: Tuple[int, ...]


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


def chunk_file_path(root: Path, layer_idx: int, chunk_id: int) -> Path:
    return root / f"layer{layer_idx}_chunk{chunk_id}.npz"


def save_compressed_chunk(
    chunk_path: Path,
    keys: CompressedTensorMLX,
    values: CompressedTensorMLX,
) -> None:
    chunk_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        chunk_path,
        num_tokens=np.array([keys.num_tokens], dtype=np.int32),
        num_heads=np.array([keys.num_heads], dtype=np.int32),
        key_pq_codes=np.asarray(keys.pq_codes),
        key_rvq_codes=np.asarray(keys.rvq_codes),
        key_rvq_mask=np.asarray(keys.rvq_mask),
        key_rvq_offsets=np.asarray(keys.rvq_offsets),
        value_pq_codes=np.asarray(values.pq_codes),
        value_rvq_codes=np.asarray(values.rvq_codes),
        value_rvq_mask=np.asarray(values.rvq_mask),
        value_rvq_offsets=np.asarray(values.rvq_offsets),
    )


def load_compressed_chunk(chunk_path: Path) -> Dict[str, mx.array]:
    data = np.load(chunk_path)
    return {key: mx.array(data[key]) for key in data.files}


def compressed_tensor_from_loaded(loaded: Dict[str, mx.array], prefix: str) -> CompressedTensorMLX:
    num_heads = int(np.asarray(loaded["num_heads"])[0])
    return CompressedTensorMLX(
        pq_codes=loaded[f"{prefix}_pq_codes"],
        rvq_codes=loaded[f"{prefix}_rvq_codes"],
        rvq_mask=loaded[f"{prefix}_rvq_mask"],
        rvq_offsets=loaded[f"{prefix}_rvq_offsets"].astype(mx.int32),
        num_heads=num_heads,
    )


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
            path = chunk_file_path(self.disk_dir, self.layer_idx, chunk_id)
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
        path = chunk_file_path(self.disk_dir, self.layer_idx, chunk_id)
        if not path.exists():
            return {}
        return load_compressed_chunk(path)

    def get_chunk(self, chunk_id: int) -> Optional[Dict[str, mx.array]]:
        return self._cache.get(chunk_id)