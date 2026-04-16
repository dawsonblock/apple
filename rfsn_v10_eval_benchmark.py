from __future__ import annotations

"""
RFSN v10.2 - Dense vs hot+warm evaluation benchmark.

This script compares exact dense attention against the current hot+warm cache
path on synthetic long sequences. It reports latency, memory estimates, and
output drift, then writes a CSV for further analysis.
"""

import argparse
import csv
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import mlx.core as mx
import numpy as np

from rfsn_v10_mlx_ane_complete import (
    RFSNConfig,
    HybridQuantizerMLX,
    RFSNv10KVCacheMLX,
    _blockwise_exact_attention,
    _decode_compressed_tensor,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("RFSN_EVAL")


def _dense_bytes(config: RFSNConfig, seq_len: int) -> int:
    return seq_len * config.num_heads * config.head_dim * 2 * 2


def _cache_bytes(cache: RFSNv10KVCacheMLX) -> int:
    mem = cache.memory_usage_bytes()
    return mem["hot_bytes"] + mem["warm_pq_bytes"] + mem["warm_rvq_bytes"]


def _reconstruct_hot_warm(cache: RFSNv10KVCacheMLX, quantizer: HybridQuantizerMLX) -> tuple[mx.array, mx.array]:
    key_segments: List[mx.array] = []
    value_segments: List[mx.array] = []
    if cache.num_hot:
        key_segments.append(cache.hot_keys)
        value_segments.append(cache.hot_values)
    if cache.num_warm:
        key_segments.append(_decode_compressed_tensor(cache.warm_keys, quantizer))
        value_segments.append(_decode_compressed_tensor(cache.warm_values, quantizer))
    keys = key_segments[0] if len(key_segments) == 1 else mx.concatenate(key_segments, axis=0)
    values = value_segments[0] if len(value_segments) == 1 else mx.concatenate(value_segments, axis=0)
    return keys, values


def _score_table(q: mx.array, keys: mx.array) -> mx.array:
    return mx.sum(
        q.astype(mx.float32)[:, :, None, :] * mx.swapaxes(keys.astype(mx.float32), 0, 1)[None, :, :, :],
        axis=-1,
    )


def _masked_score_metrics(
    dense_scores: mx.array,
    reconstructed_scores: mx.array,
    query_positions: mx.array,
) -> Dict[str, float]:
    dense_np = np.asarray(dense_scores, dtype=np.float32)
    recon_np = np.asarray(reconstructed_scores, dtype=np.float32)
    query_np = np.asarray(query_positions, dtype=np.int64)
    seq_len = dense_np.shape[-1]
    positions = np.arange(seq_len, dtype=np.int64)[None, None, :]
    valid = positions <= query_np[:, None, None]
    delta = dense_np - recon_np
    valid = np.broadcast_to(valid, delta.shape)
    valid_delta = delta[valid]
    return {
        "score_mae": float(np.mean(np.abs(valid_delta))),
        "score_rmse": float(np.sqrt(np.mean(valid_delta ** 2))),
    }


def _time_call(fn, repeats: int) -> tuple[float, mx.array]:
    output = fn()
    mx.eval(output)
    start = time.perf_counter()
    for _ in range(repeats):
        output = fn()
        mx.eval(output)
    elapsed_ms = ((time.perf_counter() - start) * 1000.0) / repeats
    return elapsed_ms, output


def evaluate_sequence_length(
    seq_len: int,
    config: RFSNConfig,
    repeats: int,
    query_count: int,
) -> Dict[str, float]:
    quantizer = HybridQuantizerMLX(config)
    cache = RFSNv10KVCacheMLX(config, layer_idx=0)
    keys = mx.random.normal(shape=(seq_len, config.num_heads, config.head_dim), dtype=mx.float32).astype(mx.float16)
    values = mx.random.normal(shape=(seq_len, config.num_heads, config.head_dim), dtype=mx.float32).astype(mx.float16)
    cache.update(keys, values, quantizer)

    if cache.num_cold:
        raise ValueError("Evaluation harness expects hot+warm only scenarios; increase warm capacity or shorten lengths")

    reconstructed_keys, reconstructed_values = _reconstruct_hot_warm(cache, quantizer)
    eval_queries = min(query_count, seq_len)
    start_query = max(0, seq_len - eval_queries)
    query_positions = mx.array(np.arange(start_query, seq_len, dtype=np.int32), dtype=mx.int32)
    q = mx.random.normal(shape=(eval_queries, config.num_heads, config.head_dim), dtype=mx.float32).astype(mx.float16)

    dense_fn = lambda: _blockwise_exact_attention(
        q=q,
        keys=keys,
        values=values,
        scale=config.head_dim ** -0.5,
        block_size=config.block_size_seq,
        causal=True,
        query_positions=query_positions,
    )
    cache_fn = lambda: cache.attention_forward(
        q,
        causal=True,
        query_positions=query_positions,
    )

    dense_latency_ms, dense_out = _time_call(dense_fn, repeats)
    cache_latency_ms, cache_out = _time_call(cache_fn, repeats)

    output_delta = np.asarray(dense_out.astype(mx.float32) - cache_out.astype(mx.float32), dtype=np.float32)
    dense_scores = _score_table(q, keys)
    reconstructed_scores = _score_table(q, reconstructed_keys)
    score_metrics = _masked_score_metrics(dense_scores, reconstructed_scores, query_positions)

    dense_bytes = _dense_bytes(config, seq_len)
    cache_total_bytes = _cache_bytes(cache)
    mem = cache.memory_usage_bytes()

    return {
        "sequence_length": seq_len,
        "query_count": eval_queries,
        "hot_tokens": cache.num_hot,
        "warm_tokens": cache.num_warm,
        "dense_latency_ms": dense_latency_ms,
        "cache_latency_ms": cache_latency_ms,
        "latency_ratio": cache_latency_ms / dense_latency_ms if dense_latency_ms else 0.0,
        "dense_bytes": dense_bytes,
        "cache_bytes": cache_total_bytes,
        "memory_saved_bytes": dense_bytes - cache_total_bytes,
        "compression_ratio": dense_bytes / cache_total_bytes if cache_total_bytes else 0.0,
        "output_mae": float(np.mean(np.abs(output_delta))),
        "output_rmse": float(np.sqrt(np.mean(output_delta ** 2))),
        "output_max_abs": float(np.max(np.abs(output_delta))),
        "score_mae": score_metrics["score_mae"],
        "score_rmse": score_metrics["score_rmse"],
        "hot_bytes": mem["hot_bytes"],
        "warm_pq_bytes": mem["warm_pq_bytes"],
        "warm_rvq_bytes": mem["warm_rvq_bytes"],
    }


def run_evaluation(
    lengths: Sequence[int],
    repeats: int,
    query_count: int,
    hot_capacity: int,
    warm_capacity: Optional[int],
    output_path: Path,
) -> List[Dict[str, float]]:
    if hasattr(mx.random, "seed"):
        mx.random.seed(17)
    np.random.seed(17)

    max_len = max(lengths)
    effective_warm = warm_capacity if warm_capacity is not None else max(1, max_len - hot_capacity)
    if max_len > hot_capacity + effective_warm:
        raise ValueError("All sequence lengths must fit into hot + warm capacity for this hot+warm evaluation")

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
        hot_capacity=hot_capacity,
        warm_capacity=effective_warm,
        cold_capacity=0,
        block_size_seq=16,
        disk_cache_dir="./eval_unused_disk_cache",
        prefetch_throttle_s=0.0,
    )

    results = [evaluate_sequence_length(seq_len, config, repeats, query_count) for seq_len in lengths]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "sequence_length",
        "query_count",
        "hot_tokens",
        "warm_tokens",
        "dense_latency_ms",
        "cache_latency_ms",
        "latency_ratio",
        "dense_bytes",
        "cache_bytes",
        "memory_saved_bytes",
        "compression_ratio",
        "output_mae",
        "output_rmse",
        "output_max_abs",
        "score_mae",
        "score_rmse",
        "hot_bytes",
        "warm_pq_bytes",
        "warm_rvq_bytes",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    for row in results:
        logger.info(
            "len=%3d dense=%.3fms cache=%.3fms mem=%.2fx drift=%.4f score=%.4f",
            row["sequence_length"],
            row["dense_latency_ms"],
            row["cache_latency_ms"],
            row["compression_ratio"],
            row["output_rmse"],
            row["score_rmse"],
        )

    logger.info("Wrote CSV: %s", output_path)
    return results


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Dense vs hot+warm cache evaluation benchmark")
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=[32, 64, 128, 192],
        help="Sequence lengths to evaluate",
    )
    parser.add_argument("--repeats", type=int, default=8, help="Timing repeats per scenario")
    parser.add_argument("--query-count", type=int, default=8, help="Number of decode-style queries per sequence")
    parser.add_argument("--hot-capacity", type=int, default=32, help="Hot-tier token capacity")
    parser.add_argument(
        "--warm-capacity",
        type=int,
        default=None,
        help="Warm-tier token capacity; defaults to max(lengths) - hot_capacity",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./benchmark_outputs/dense_vs_hot_warm.csv"),
        help="CSV output path",
    )
    args = parser.parse_args(argv)

    run_evaluation(
        lengths=args.lengths,
        repeats=args.repeats,
        query_count=args.query_count,
        hot_capacity=args.hot_capacity,
        warm_capacity=args.warm_capacity,
        output_path=args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())