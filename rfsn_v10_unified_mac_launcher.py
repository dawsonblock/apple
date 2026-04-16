from __future__ import annotations

"""
RFSN v10.2 - Unified Mac launcher.

Backend order:
  1. MLX on Apple Silicon
  2. PyTorch with MPS
  3. PyTorch on CPU
"""

import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("RFSN_LAUNCHER")


@dataclass
class BackendInfo:
    name: str
    priority: int
    device_str: str
    is_available: bool
    reason: str


def detect_backends() -> List[BackendInfo]:
    backends: List[BackendInfo] = []

    mlx_available = False
    mlx_reason = "mlx not installed"
    try:
        import mlx.core as mx

        probe = mx.zeros((1,), dtype=mx.float16)
        del probe
        mlx_available = True
        mlx_reason = f"mlx import ok (device={mx.default_device() if hasattr(mx, 'default_device') else 'unknown'})"
    except Exception as exc:
        mlx_reason = f"mlx unavailable: {exc}"

    backends.append(
        BackendInfo(
            name="mlx_apple",
            priority=1,
            device_str="mlx",
            is_available=mlx_available,
            reason=mlx_reason,
        )
    )

    mps_available = False
    mps_reason = "torch not installed"
    cpu_available = False
    cpu_reason = "torch not installed"

    try:
        import torch

        cpu_available = True
        cpu_reason = f"torch {torch.__version__} CPU available"

        if torch.backends.mps.is_available():
            try:
                probe = torch.zeros((1,), device="mps")
                del probe
                mps_available = True
                mps_reason = f"torch {torch.__version__} MPS ready"
            except Exception as exc:
                mps_reason = f"MPS allocation failed: {exc}"
        elif torch.backends.mps.is_built():
            mps_reason = "MPS support built but unavailable on this machine"
        else:
            mps_reason = f"torch {torch.__version__} built without MPS"
    except Exception as exc:
        mps_reason = f"torch unavailable: {exc}"
        cpu_reason = f"torch unavailable: {exc}"

    backends.append(
        BackendInfo(
            name="pytorch_mps",
            priority=2,
            device_str="mps",
            is_available=mps_available,
            reason=mps_reason,
        )
    )
    backends.append(
        BackendInfo(
            name="pytorch_cpu",
            priority=3,
            device_str="cpu",
            is_available=cpu_available,
            reason=cpu_reason,
        )
    )
    return backends


def select_best_backend(backends: List[BackendInfo]) -> BackendInfo:
    available = [backend for backend in backends if backend.is_available]
    if not available:
        raise RuntimeError("No supported backend is available. Install mlx or torch.")
    return sorted(available, key=lambda backend: backend.priority)[0]


def run_mlx_ane() -> None:
    try:
        from rfsn_v10_mlx_ane_complete import run_tests
    except Exception as exc:
        logger.warning("Failed to import full MLX implementation (%s); using embedded smoke tests", exc)
        run_mlx_embedded_tests()
        return

    logger.info("Running MLX Apple Silicon test suite")
    if not run_tests():
        raise RuntimeError("MLX backend tests failed")


def run_mlx_embedded_tests() -> None:
    import mlx.core as mx

    logger.info("Running embedded MLX smoke tests")

    tensor = mx.random.normal(shape=(32, 128), dtype=mx.float32).astype(mx.float16)
    assert tensor.shape == (32, 128)
    logger.info("  Tensor allocation ok: %s", tensor.shape)

    codebook = mx.random.normal(shape=(8, 256, 16), dtype=mx.float32).astype(mx.float16)
    codes = mx.random.randint(0, 256, shape=(32, 8)).astype(mx.uint8)
    recon = mx.zeros((32, 128), dtype=mx.float16)
    for subspace in range(8):
        start = subspace * 16
        end = start + 16
        recon[:, start:end] = mx.take(codebook[subspace], codes[:, subspace].astype(mx.int32), axis=0)
    assert recon.shape == (32, 128)
    logger.info("  PQ lookup ok")

    weights = mx.random.normal(shape=(64, 32), dtype=mx.float32)
    q_weights, q_scales, q_biases = mx.quantize(weights, bits=6, group_size=32)
    dequantized = mx.dequantize(q_weights, q_scales, q_biases, bits=6, group_size=32)
    mse = float(mx.mean((weights - dequantized) ** 2))
    assert np.isfinite(mse)
    logger.info("  Quantize/dequantize ok, mse=%.6f", mse)

    scores = mx.random.normal(shape=(1, 4, 64), dtype=mx.float32)
    values = mx.random.normal(shape=(64, 4, 32), dtype=mx.float32)
    max_scores = mx.max(scores, axis=-1, keepdims=True)
    probs = mx.exp(scores - max_scores)
    probs = probs / mx.sum(probs, axis=-1, keepdims=True)
    output = mx.sum(probs[:, :, :, None] * mx.swapaxes(values, 0, 1)[None, :, :, :], axis=2)
    assert output.shape == (1, 4, 32)
    logger.info("  Attention smoke test ok")


def _run_pytorch_backend(device_str: str) -> None:
    import torch
    import torch.nn.functional as F

    @dataclass
    class Config:
        num_heads: int = 4
        head_dim: int = 128
        num_subspaces: int = 8
        subspace_dim: int = 16
        codebook_size: int = 256
        block_size_seq: int = 16

    config = Config()
    if device_str == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if device.type == "mps":
        try:
            torch.backends.mps.set_high_precision(True)
        except Exception:
            pass

    pq_codebook = torch.randn(
        config.num_subspaces,
        config.codebook_size,
        config.subspace_dim,
        device=device,
        dtype=torch.float16,
    )
    value_codebook = torch.randn(
        config.num_subspaces,
        config.codebook_size,
        config.subspace_dim,
        device=device,
        dtype=torch.float16,
    )

    def decode_codes(codes: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
        batch, heads, seq_len, _ = codes.shape
        out = torch.zeros(batch, heads, seq_len, config.head_dim, device=device, dtype=torch.float32)
        for subspace in range(config.num_subspaces):
            start = subspace * config.subspace_dim
            end = start + config.subspace_dim
            out[..., start:end] = F.embedding(codes[..., subspace].long(), codebook[subspace].float())
        return out

    def hybrid_attention_pytorch(
        q: torch.Tensor,
        key_codes: torch.Tensor,
        value_codes: torch.Tensor,
        causal: bool = False,
        query_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, heads, head_dim = q.shape
        seq_len = key_codes.shape[2]
        keys = decode_codes(key_codes, pq_codebook).permute(0, 2, 1, 3)
        values = decode_codes(value_codes, value_codebook).permute(0, 2, 1, 3)

        running_max = torch.full((batch, heads), -1e30, device=device, dtype=torch.float32)
        running_sum = torch.zeros((batch, heads), device=device, dtype=torch.float32)
        out = torch.zeros((batch, heads, head_dim), device=device, dtype=torch.float32)

        if query_positions is None:
            query_positions = torch.full((batch,), seq_len - 1, device=device, dtype=torch.long)
        elif query_positions.dim() == 2:
            query_positions = query_positions[:, 0]

        scale = head_dim ** -0.5
        for start in range(0, seq_len, config.block_size_seq):
            end = min(start + config.block_size_seq, seq_len)
            key_block = keys[:, start:end].permute(0, 2, 1, 3)
            value_block = values[:, start:end].permute(0, 2, 1, 3)
            scores = (q.float().unsqueeze(2) * key_block).sum(dim=-1) * scale
            if causal:
                positions = torch.arange(start, end, device=device)
                allowed = positions.view(1, 1, -1) <= query_positions.view(batch, 1, 1)
                scores = torch.where(allowed, scores, torch.full_like(scores, -1e30))
            else:
                allowed = None

            block_max = scores.max(dim=-1).values
            new_max = torch.maximum(running_max, block_max)
            prev_rescale = torch.exp(running_max - new_max)
            running_sum = running_sum * prev_rescale
            out = out * prev_rescale.unsqueeze(-1)
            exp_scores = torch.exp(scores - new_max.unsqueeze(-1))
            if allowed is not None:
                exp_scores = exp_scores * allowed
            running_sum = running_sum + exp_scores.sum(dim=-1)
            out = out + (exp_scores.unsqueeze(-1) * value_block).sum(dim=2)
            running_max = new_max

        safe_sum = torch.where(running_sum > 0, running_sum, torch.ones_like(running_sum))
        return (out / safe_sum.unsqueeze(-1)).to(q.dtype)

    compiled_attention = hybrid_attention_pytorch
    if hasattr(torch, "compile"):
        try:
            candidate_attention = torch.compile(hybrid_attention_pytorch)
            probe_q = torch.randn(1, config.num_heads, config.head_dim, device=device, dtype=torch.float16)
            probe_codes = torch.randint(
                0,
                config.codebook_size,
                (1, config.num_heads, 8, config.num_subspaces),
                device=device,
                dtype=torch.int32,
            )
            probe_positions = torch.tensor([6], device=device)
            eager_probe = hybrid_attention_pytorch(
                probe_q,
                probe_codes,
                probe_codes,
                causal=True,
                query_positions=probe_positions,
            )
            compiled_probe = candidate_attention(
                probe_q,
                probe_codes,
                probe_codes,
                causal=True,
                query_positions=probe_positions,
            )
            compile_diff = (compiled_probe.float() - eager_probe.float()).abs().max().item()
            compile_tolerance = 1e-4 if device.type != "mps" else 5e-3
            if compile_diff <= compile_tolerance:
                compiled_attention = candidate_attention
                logger.info("torch.compile enabled for %s", device.type)
            else:
                logger.warning(
                    "torch.compile validation diff %.2e exceeded tolerance %.2e on %s; using eager",
                    compile_diff,
                    compile_tolerance,
                    device.type,
                )
        except Exception as exc:
            logger.info("torch.compile not used: %s", exc)

    logger.info("Running PyTorch backend tests on %s", device)
    q = torch.randn(2, config.num_heads, config.head_dim, device=device, dtype=torch.float16)
    key_codes = torch.randint(
        0,
        config.codebook_size,
        (2, config.num_heads, 24, config.num_subspaces),
        device=device,
        dtype=torch.int32,
    )
    value_codes = torch.randint(
        0,
        config.codebook_size,
        (2, config.num_heads, 24, config.num_subspaces),
        device=device,
        dtype=torch.int32,
    )

    start = time.time()
    out = compiled_attention(
        q,
        key_codes,
        value_codes,
        causal=True,
        query_positions=torch.tensor([12, 18], device=device),
    )
    if device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()
    elapsed_ms = (time.time() - start) * 1000.0

    ref = hybrid_attention_pytorch(
        q,
        key_codes,
        value_codes,
        causal=True,
        query_positions=torch.tensor([12, 18], device=device),
    )
    max_diff = (out.float() - ref.float()).abs().max().item()
    assert out.shape == (2, config.num_heads, config.head_dim)
    assert max_diff < 1e-4
    logger.info("  Hybrid attention ok, max diff %.2e, time %.1f ms", max_diff, elapsed_ms)

    vectors = torch.randn(64, config.head_dim, device=device, dtype=torch.float16)
    pq_codes = torch.zeros(64, config.num_subspaces, device=device, dtype=torch.int64)
    decoded = torch.zeros_like(vectors, dtype=torch.float32)
    for subspace in range(config.num_subspaces):
        start_idx = subspace * config.subspace_dim
        end_idx = start_idx + config.subspace_dim
        sub_vectors = vectors[:, start_idx:end_idx].float()
        centroids = pq_codebook[subspace].float()
        dists = torch.cdist(sub_vectors, centroids) ** 2
        idx = dists.argmin(dim=1)
        pq_codes[:, subspace] = idx
        decoded[:, start_idx:end_idx] = F.embedding(idx, centroids)

    mse = (vectors.float() - decoded).pow(2).mean().item()
    assert np.isfinite(mse)
    logger.info("  PQ roundtrip ok, mse %.6f", mse)

    logger.info("PyTorch backend completed successfully on %s", device)


def run_pytorch_mps() -> None:
    _run_pytorch_backend("mps")


def run_pytorch_cpu() -> None:
    _run_pytorch_backend("cpu")


def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")

    logger.info("=" * 64)
    logger.info("RFSN v10.2 - Unified Mac launcher")
    logger.info("=" * 64)

    backends = detect_backends()
    logger.info("Detected backends:")
    for backend in backends:
        status = "OK" if backend.is_available else "NO"
        logger.info("  [%s] %-12s %s", status, backend.name, backend.reason)

    best = select_best_backend(backends)
    logger.info("Selected backend: %s (%s)", best.name, best.reason)

    if best.name == "mlx_apple":
        run_mlx_ane()
    elif best.name == "pytorch_mps":
        run_pytorch_mps()
    else:
        logger.warning("Running CPU fallback")
        run_pytorch_cpu()

    logger.info("Launcher completed successfully")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.error("Launcher failed: %s", exc)
        sys.exit(1)