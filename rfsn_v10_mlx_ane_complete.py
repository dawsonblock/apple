from __future__ import annotations

"""
RFSN v10.2 - Native MLX Apple Silicon implementation.

The core math and runtime pieces now live in dedicated modules:
  - quantization.py
  - attention.py
  - cache.py
  - storage.py

This file remains the public entrypoint for tests, benchmarks, and the decoder
layer API that composes those modules together.
"""

import argparse
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from attention import _blockwise_exact_attention, _dense_exact_attention, _gelu, RFSNHybridAttentionMLX
from cache import RFSNv10KVCacheMLX
from quantization import (
    HybridQuantizerMLX,
    ProductQuantizerMLX,
    ResidualVQMLX,
    _compress_tensor_sequence,
    _decode_compressed_tensor,
)
from storage import (
    AsyncHierarchicalRouterMLX,
    CompressedTensorMLX,
    MASK_FILL_VALUE,
    QuantizedMatrix,
    RFSNConfig,
    SUPPORTED_GROUP_SIZES,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("RFSN_MLX_APPLE")


class RFSNDecoderLayerMLX(nn.Module):
    """Decoder layer that owns its projections, quantizer, and cache-backed decode path."""

    def __init__(self, config: RFSNConfig, layer_idx: int = 0, ffn_multiplier: int = 2):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.model_dim = config.hidden_dim
        self.ffn_hidden_dim = max(config.hidden_dim * ffn_multiplier, config.hidden_dim)
        self.scale = config.head_dim ** -0.5

        self.attn_norm = nn.RMSNorm(config.hidden_dim)
        self.ffn_norm = nn.RMSNorm(config.hidden_dim)
        self.q_proj = nn.Linear(config.hidden_dim, config.num_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_dim, config.num_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_dim, config.num_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_heads * config.head_dim, config.hidden_dim, bias=False)
        self.ffn_up = nn.Linear(config.hidden_dim, self.ffn_hidden_dim, bias=False)
        self.ffn_down = nn.Linear(self.ffn_hidden_dim, config.hidden_dim, bias=False)

        self.quantizer = HybridQuantizerMLX(config)
        self.disk_cache_dir = Path(config.disk_cache_dir) / f"decoder_layer_{layer_idx}"
        self.cache = RFSNv10KVCacheMLX(config, layer_idx=layer_idx)

    @property
    def cache_tokens(self) -> int:
        return self.cache.total_tokens

    def reset_cache(self, remove_disk: bool = True) -> None:
        if remove_disk and self.disk_cache_dir.exists():
            shutil.rmtree(self.disk_cache_dir)
        self.cache = RFSNv10KVCacheMLX(self.config, layer_idx=self.layer_idx)

    def prefill_cache(self, x: mx.array, reset: bool = True) -> None:
        if len(x.shape) != 3 or int(x.shape[0]) != 1:
            raise ValueError("prefill_cache expects input shape [1, seq_len, hidden_dim]")
        if reset:
            self.reset_cache(remove_disk=True)

        attn_input = self.attn_norm(x).astype(mx.float16)
        keys = self._project_heads(self.k_proj(attn_input))
        values = self._project_heads(self.v_proj(attn_input))
        self.cache.update(
            keys[0].astype(mx.float16),
            values[0].astype(mx.float16),
            self.quantizer,
            disk_dir=self.disk_cache_dir,
        )

    def _project_heads(self, projected: mx.array) -> mx.array:
        batch_size, seq_len, _ = projected.shape
        return projected.reshape(batch_size, seq_len, self.num_heads, self.head_dim).astype(mx.float16)

    def _merge_heads(self, attended: mx.array) -> mx.array:
        batch_size, seq_len, _, _ = attended.shape
        return attended.reshape(batch_size, seq_len, self.num_heads * self.head_dim)

    def __call__(
        self,
        x: mx.array,
        use_cache: bool = False,
        router: Optional[AsyncHierarchicalRouterMLX] = None,
        context_window: Optional[int] = None,
    ) -> mx.array:
        if len(x.shape) != 3:
            raise ValueError("Decoder layer expects input shape [batch, seq_len, hidden_dim]")
        if int(x.shape[-1]) != self.model_dim:
            raise ValueError(f"Expected hidden_dim {self.model_dim}, got {int(x.shape[-1])}")

        residual = x.astype(mx.float16)
        attn_input = self.attn_norm(residual).astype(mx.float16)
        q_heads = self._project_heads(self.q_proj(attn_input))
        k_heads = self._project_heads(self.k_proj(attn_input))
        v_heads = self._project_heads(self.v_proj(attn_input))

        if use_cache:
            if tuple(int(dim) for dim in x.shape[:2]) != (1, 1):
                raise ValueError(
                    "cache-backed decoding currently supports input shape [1, 1, hidden_dim]; "
                    "use prefill_cache() for longer prompts"
                )
            self.cache.update(
                k_heads[0].astype(mx.float16),
                v_heads[0].astype(mx.float16),
                self.quantizer,
                disk_dir=self.disk_cache_dir,
            )
            query_position = mx.array([self.cache.total_tokens - 1], dtype=mx.int32)
            effective_window = context_window
            if effective_window is None:
                effective_window = max(
                    self.config.block_size_seq,
                    min(self.cache.total_tokens, self.config.warm_capacity),
                )
            attended = self.cache.attention_forward(
                q_heads[:, 0].astype(mx.float16),
                causal=True,
                query_positions=query_position,
                router=router,
                current_position=self.cache.total_tokens - 1,
                context_window=effective_window,
            ).reshape(1, 1, self.num_heads, self.head_dim)
        else:
            attended = _dense_exact_attention(
                q=q_heads,
                keys=k_heads,
                values=v_heads,
                scale=self.scale,
                causal=True,
            )

        attn_out = self.o_proj(self._merge_heads(attended)).astype(residual.dtype)
        hidden = residual + attn_out
        ffn_input = self.ffn_norm(hidden).astype(mx.float16)
        ffn_out = self.ffn_down(_gelu(self.ffn_up(ffn_input))).astype(hidden.dtype)
        return hidden + ffn_out


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


def run_benchmarks() -> List[Dict[str, object]]:
    benchmark_root = Path("./benchmark_rfsn_disk_cache")
    if benchmark_root.exists():
        shutil.rmtree(benchmark_root)
    benchmark_root.mkdir(parents=True, exist_ok=True)

    base_config = dict(
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
        cold_capacity=96,
        block_size_seq=16,
        prefetch_throttle_s=0.0,
    )
    scenarios = [
        {"name": "hot_only", "tokens": 12, "iterations": 8, "use_router": False, "context_window": 12},
        {"name": "warm_mixed", "tokens": 28, "iterations": 6, "use_router": False, "context_window": 12},
        {"name": "cold_mixed", "tokens": 52, "iterations": 4, "use_router": True, "context_window": 16},
    ]
    results: List[Dict[str, object]] = []

    try:
        for scenario in scenarios:
            cache_dir = benchmark_root / scenario["name"]
            config = RFSNConfig(**base_config, disk_cache_dir=str(cache_dir))
            quantizer = HybridQuantizerMLX(config)
            cache = RFSNv10KVCacheMLX(config, layer_idx=0)
            keys = mx.random.normal(
                shape=(scenario["tokens"], config.num_heads, config.head_dim),
                dtype=mx.float32,
            ).astype(mx.float16)
            values = mx.random.normal(
                shape=(scenario["tokens"], config.num_heads, config.head_dim),
                dtype=mx.float32,
            ).astype(mx.float16)
            cache.update(keys, values, quantizer, disk_dir=cache_dir)
            query = mx.random.normal(shape=(1, config.num_heads, config.head_dim), dtype=mx.float32).astype(mx.float16)
            router = AsyncHierarchicalRouterMLX(config, layer_idx=0) if scenario["use_router"] else None
            query_positions = mx.array([scenario["tokens"] - 1], dtype=mx.int32)

            warmup = cache.attention_forward(
                query,
                causal=True,
                query_positions=query_positions,
                router=router,
                current_position=scenario["tokens"] - 1 if router is not None else None,
                context_window=scenario["context_window"],
            )
            mx.eval(warmup)

            start = time.perf_counter()
            for _ in range(scenario["iterations"]):
                out = cache.attention_forward(
                    query,
                    causal=True,
                    query_positions=query_positions,
                    router=router,
                    current_position=scenario["tokens"] - 1 if router is not None else None,
                    context_window=scenario["context_window"],
                )
                mx.eval(out)
            mean_ms = ((time.perf_counter() - start) * 1000.0) / scenario["iterations"]

            mem = cache.memory_usage_bytes()
            access_stats = cache.get_last_access_stats()
            cold_disk_bytes = sum(path.stat().st_size for path in cache.cold_chunk_paths if path.exists())
            result = {
                "scenario": scenario["name"],
                "tokens": scenario["tokens"],
                "iterations": scenario["iterations"],
                "context_window": scenario["context_window"],
                "mean_latency_ms": round(mean_ms, 3),
                "hot_bytes": mem["hot_bytes"],
                "warm_pq_bytes": mem["warm_pq_bytes"],
                "warm_rvq_bytes": mem["warm_rvq_bytes"],
                "cold_tokens": mem["cold_tokens"],
                "cold_chunks": mem["cold_chunks"],
                "cold_disk_bytes": cold_disk_bytes,
                "window_tokens": access_stats["window_tokens"],
                "reconstructed_tokens": access_stats["reconstructed_tokens"],
                "warm_chunk_decodes": access_stats["warm_chunk_decodes"],
                "cold_chunk_decodes": access_stats["cold_chunk_decodes"],
            }
            results.append(result)
            logger.info(
                "[Bench] %-10s tokens=%d window=%d mean=%.3fms hot=%.1fKB warm=%.1fKB cold_disk=%.1fKB recon=%d warm_decodes=%d cold_decodes=%d",
                result["scenario"],
                result["tokens"],
                result["context_window"],
                result["mean_latency_ms"],
                result["hot_bytes"] / 1024.0,
                (result["warm_pq_bytes"] + result["warm_rvq_bytes"]) / 1024.0,
                result["cold_disk_bytes"] / 1024.0,
                result["reconstructed_tokens"],
                result["warm_chunk_decodes"],
                result["cold_chunk_decodes"],
            )
    finally:
        if benchmark_root.exists():
            shutil.rmtree(benchmark_root)

    return results


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
    tests: List[Tuple[str, Callable[[], None]]] = []

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
        pq_codes, _ = hq.pq.quantize(vectors)
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
        assert len(cache.cold_chunk_paths) == 2

        q = mx.random.normal(shape=(1, config.num_heads, config.head_dim), dtype=mx.float32).astype(mx.float16)
        cache_out = cache.attention_forward(
            q,
            causal=True,
            query_positions=mx.array([59], dtype=mx.int32),
        )
        access_stats = cache.get_last_access_stats()

        keys_exact, values_exact, adjusted_positions = cache.materialize_window(
            query_positions=mx.array([59], dtype=mx.int32),
        )
        ref = _blockwise_exact_attention(
            q=q,
            keys=keys_exact,
            values=values_exact,
            scale=config.head_dim ** -0.5,
            block_size=config.block_size_seq,
            causal=True,
            query_positions=adjusted_positions,
        )
        max_diff = float(mx.max(mx.abs(cache_out.astype(mx.float32) - ref.astype(mx.float32))))
        assert max_diff < 1e-4
        mem = cache.memory_usage_bytes()
        assert mem["cold_chunks"] == 2
        assert access_stats["window_tokens"] == 60
        assert access_stats["hot_tokens_materialized"] == config.hot_capacity
        assert access_stats["warm_tokens_materialized"] == config.warm_capacity
        assert access_stats["cold_tokens_materialized"] == 60 - config.hot_capacity - config.warm_capacity
        assert access_stats["reconstructed_tokens"] == access_stats["warm_tokens_materialized"] + access_stats["cold_tokens_materialized"]
        assert access_stats["warm_chunk_decodes"] == 2
        assert access_stats["cold_chunk_decodes"] == 2
        logger.info("  Cache tiering validated, attention max diff: %.2e", max_diff)

    def test_warm_window_materialization() -> None:
        cache = RFSNv10KVCacheMLX(config, layer_idx=0)
        keys = mx.random.normal(shape=(36, config.num_heads, config.head_dim), dtype=mx.float32).astype(mx.float16)
        values = mx.random.normal(shape=(36, config.num_heads, config.head_dim), dtype=mx.float32).astype(mx.float16)
        cache.update(keys, values, hq, disk_dir=Path(config.disk_cache_dir))

        full_keys, full_values, _ = cache.materialize_window(
            query_positions=mx.array([35], dtype=mx.int32),
            context_window=None,
        )

        window_keys, window_values, adjusted = cache.materialize_window(
            query_positions=mx.array([35], dtype=mx.int32),
            context_window=8,
        )
        access_stats = cache.get_last_access_stats()
        assert window_keys.shape == (8, config.num_heads, config.head_dim)
        assert window_values.shape == (8, config.num_heads, config.head_dim)
        assert int(np.asarray(adjusted)[0]) == 7
        max_key_diff = float(mx.max(mx.abs(window_keys.astype(mx.float32) - full_keys[-8:].astype(mx.float32))))
        max_value_diff = float(mx.max(mx.abs(window_values.astype(mx.float32) - full_values[-8:].astype(mx.float32))))
        assert max_key_diff < 1e-4
        assert max_value_diff < 1e-4
        assert access_stats["window_tokens"] == 8
        assert access_stats["hot_tokens_materialized"] == 0
        assert access_stats["warm_tokens_materialized"] == 8
        assert access_stats["cold_tokens_materialized"] == 0
        assert access_stats["reconstructed_tokens"] == 8
        assert access_stats["warm_chunk_decodes"] == 1
        logger.info("  Warm-tier window materialization validated")

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
        q = mx.random.normal(shape=(1, config.num_heads, config.head_dim), dtype=mx.float32).astype(mx.float16)
        _ = cache.attention_forward(
            q,
            causal=True,
            query_positions=mx.array([47], dtype=mx.int32),
            router=router,
            current_position=47,
            context_window=16,
        )
        access_stats = cache.get_last_access_stats()
        assert access_stats["cold_chunk_cache_hits"] >= 1
        logger.info("  Router prefetched chunks: %s", loaded_ids)

    def test_decoder_layer_api() -> None:
        layer = RFSNDecoderLayerMLX(config, layer_idx=0)
        batch_input = mx.random.normal(shape=(2, 6, config.hidden_dim), dtype=mx.float32).astype(mx.float16)
        batch_out = layer(batch_input, use_cache=False)
        assert batch_out.shape == batch_input.shape

        prompt = mx.random.normal(shape=(1, 4, config.hidden_dim), dtype=mx.float32).astype(mx.float16)
        layer.prefill_cache(prompt)
        assert layer.cache_tokens == 4

        decode_token = mx.random.normal(shape=(1, 1, config.hidden_dim), dtype=mx.float32).astype(mx.float16)
        decode_out = layer(decode_token, use_cache=True, context_window=8)
        assert decode_out.shape == decode_token.shape
        assert layer.cache_tokens == 5

        layer.reset_cache(remove_disk=True)
        assert layer.cache_tokens == 0
        logger.info("  Decoder layer cache-backed path validated")

    tests.extend(
        [
            ("PQ roundtrip", test_pq_roundtrip),
            ("RVQ sparsity", test_rvq_sparsity),
            ("Hybrid roundtrip", test_hybrid_roundtrip),
            ("Quantized fast path", test_quantized_fast_path),
            ("Hybrid attention", test_hybrid_attention_matches_reference),
            ("KV cache tiers", test_kv_cache_tiers_and_attention),
            ("Warm window materialization", test_warm_window_materialization),
            ("Calibration", test_calibration),
            ("Async router", test_async_router),
            ("Decoder layer API", test_decoder_layer_api),
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


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="RFSN v10.2 MLX Apple Silicon tooling")
    parser.add_argument("--test", action="store_true", help="Run the MLX test suite")
    parser.add_argument("--bench", action="store_true", help="Run the MLX cache benchmarks")
    args = parser.parse_args(argv)

    run_default_tests = args.test or not args.bench
    success = True
    if run_default_tests:
        success = run_tests()
    if args.bench:
        run_benchmarks()
    return 0 if success else 1


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")

    logger.info("RFSN v10.2 - MLX Apple Silicon implementation")
    logger.info("MLX default device: %s", mx.default_device() if hasattr(mx, "default_device") else "unknown")

    sys.exit(main())