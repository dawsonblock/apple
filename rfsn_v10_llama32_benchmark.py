from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean
from typing import Dict, List

from llama32_adapter import (
    DEFAULT_MODEL_ID,
    DEFAULT_PROMPTS,
    Llama32DecoderLayerMLX,
    build_rfsn_config_from_hf_config,
    capture_layer_trace,
    get_decoder_layers,
    get_rotary_embedding_module,
    load_model_and_tokenizer,
    load_prompts_from_file,
    run_layer_parity,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark real prompts and generated continuations against the optional Llama 3.2 layer adapter.",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Hugging Face model id to load")
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    parser.add_argument("--torch-dtype", choices=["auto", "float16", "float32", "bfloat16"], default="auto")
    parser.add_argument("--hot-capacity", type=int, default=64)
    parser.add_argument("--warm-capacity", type=int, default=256)
    parser.add_argument("--cold-capacity", type=int, default=512)
    parser.add_argument("--block-size-seq", type=int, default=32)
    parser.add_argument("--context-window", type=int, default=None)
    parser.add_argument("--disable-rvq", action="store_true")
    parser.add_argument("--use-router", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--prompt", action="append", default=[], help="Repeatable prompt value")
    parser.add_argument("--prompts-file", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_outputs/llama32_layer_benchmark.csv"),
    )
    return parser


def resolve_prompts(args: argparse.Namespace) -> List[str]:
    if args.prompt:
        return args.prompt
    if args.prompts_file is not None:
        prompts = load_prompts_from_file(args.prompts_file)
        if prompts:
            return prompts
    return list(DEFAULT_PROMPTS)


def main() -> None:
    args = build_parser().parse_args()
    prompts = resolve_prompts(args)
    model, tokenizer, resolved_device, resolved_dtype = load_model_and_tokenizer(
        model_id=args.model_id,
        device=args.device,
        torch_dtype=args.torch_dtype,
    )
    layers = get_decoder_layers(model)
    hf_layer = layers[args.layer_index]
    adapter_config = build_rfsn_config_from_hf_config(
        model.config,
        hot_capacity=args.hot_capacity,
        warm_capacity=args.warm_capacity,
        cold_capacity=args.cold_capacity,
        block_size_seq=args.block_size_seq,
        disable_rvq=args.disable_rvq,
    )
    adapter = Llama32DecoderLayerMLX.from_hf_layer(
        hf_layer=hf_layer,
        hf_config=model.config,
        rfsn_config=adapter_config,
        layer_idx=args.layer_index,
        rotary_module=get_rotary_embedding_module(model, hf_layer),
    )

    rows: List[Dict[str, object]] = []
    for prompt_index, prompt in enumerate(prompts):
        trace = capture_layer_trace(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            layer_index=args.layer_index,
            max_new_tokens=args.max_new_tokens,
        )

        runs: List[Dict[str, float]] = []
        for _ in range(args.repeats):
            runs.append(
                run_layer_parity(
                    adapter=adapter,
                    trace=trace,
                    context_window=args.context_window,
                    use_router=args.use_router,
                )
            )

        baseline = dict(runs[0])
        baseline["dense_latency_ms"] = mean(run["dense_latency_ms"] for run in runs)
        baseline["cache_latency_ms"] = mean(run["cache_latency_ms"] for run in runs)
        row: Dict[str, object] = {
            "prompt_index": prompt_index,
            "prompt_preview": prompt.replace("\n", " ")[:96],
            "model_id": args.model_id,
            "layer_index": args.layer_index,
            "device": str(resolved_device),
            "torch_dtype": str(resolved_dtype),
            "prompt_length": trace.prompt_length,
            "generated_length": trace.generated_length,
            "total_tokens": trace.total_tokens,
            "generation_latency_ms": trace.generation_latency_ms,
            "capture_latency_ms": trace.capture_latency_ms,
            "context_window": -1 if args.context_window is None else args.context_window,
            "use_router": int(args.use_router),
            "disable_rvq": int(args.disable_rvq),
            **baseline,
        }
        rows.append(row)
        print(
            f"prompt {prompt_index}: total_tokens={trace.total_tokens} "
            f"dense_latency_ms={row['dense_latency_ms']:.3f} "
            f"cache_latency_ms={row['cache_latency_ms']:.3f} "
            f"cache_last_token_rmse={row['cache_last_token_rmse']:.6f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()