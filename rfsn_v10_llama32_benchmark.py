from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean
from typing import Dict, List

from llama32_adapter import (
    DEFAULT_MODEL_ID,
    DEFAULT_PROMPTS,
    DEFAULT_REPEAT_SEPARATOR,
    Llama32DecoderLayerMLX,
    build_rfsn_config_from_hf_config,
    capture_layer_trace,
    decode_escape_sequences,
    get_decoder_layers,
    get_rotary_embedding_module,
    load_model_and_tokenizer,
    load_prompts_from_file,
    prepare_prompt,
    require_min_metric,
    require_min_total_tokens,
    run_layer_parity,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark real prompts and generated continuations against the optional Llama 3.2 layer adapter.",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Hugging Face model id to load")
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--prompt-repeat", type=int, default=1, help="Repeat each prompt this many times before capture")
    parser.add_argument(
        "--repeat-separator",
        default=DEFAULT_REPEAT_SEPARATOR.encode("unicode_escape").decode("ascii"),
        help="Separator inserted between repeated prompt copies; escape sequences like \\n are decoded",
    )
    parser.add_argument(
        "--min-prompt-tokens",
        type=int,
        default=None,
        help="Keep repeating each prompt until its tokenized length reaches at least this value",
    )
    parser.add_argument(
        "--min-total-tokens",
        type=int,
        default=None,
        help="Fail if a prompt plus generated tokens stays below this value",
    )
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    parser.add_argument("--torch-dtype", choices=["auto", "float16", "float32", "bfloat16"], default="auto")
    parser.add_argument("--hot-capacity", type=int, default=64)
    parser.add_argument("--warm-capacity", type=int, default=256)
    parser.add_argument("--cold-capacity", type=int, default=512)
    parser.add_argument("--block-size-seq", type=int, default=32)
    parser.add_argument("--context-window", type=int, default=None)
    parser.add_argument("--disable-rvq", action="store_true")
    parser.add_argument("--use-router", action="store_true")
    parser.add_argument(
        "--min-reconstructed-tokens",
        type=int,
        default=None,
        help="Fail if the cache path reconstructs fewer tokens than this value",
    )
    parser.add_argument(
        "--min-warm-chunk-decodes",
        type=int,
        default=None,
        help="Fail if the cache path performs fewer warm chunk decodes than this value",
    )
    parser.add_argument(
        "--min-cold-chunk-decodes",
        type=int,
        default=None,
        help="Fail if the cache path performs fewer cold chunk decodes than this value",
    )
    parser.add_argument(
        "--min-cold-chunk-cache-hits",
        type=int,
        default=None,
        help="Fail if router-assisted cold fetches produce fewer cache hits than this value",
    )
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
    repeat_separator = decode_escape_sequences(args.repeat_separator)
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
        prepared_prompt = prepare_prompt(
            prompt=prompt,
            tokenizer=tokenizer,
            repeat_count=args.prompt_repeat,
            min_prompt_tokens=args.min_prompt_tokens,
            repeat_separator=repeat_separator,
        )
        trace = capture_layer_trace(
            model=model,
            tokenizer=tokenizer,
            prompt=prepared_prompt.prompt_text,
            layer_index=args.layer_index,
            max_new_tokens=args.max_new_tokens,
        )
        require_min_total_tokens(trace, args.min_total_tokens, label=f"prompt {prompt_index}")

        runs: List[Dict[str, float]] = []
        for _ in range(args.repeats):
            run = run_layer_parity(
                adapter=adapter,
                trace=trace,
                context_window=args.context_window,
                use_router=args.use_router,
            )
            require_min_metric(run, "reconstructed_tokens", args.min_reconstructed_tokens, label=f"prompt {prompt_index}")
            require_min_metric(run, "warm_chunk_decodes", args.min_warm_chunk_decodes, label=f"prompt {prompt_index}")
            require_min_metric(run, "cold_chunk_decodes", args.min_cold_chunk_decodes, label=f"prompt {prompt_index}")
            require_min_metric(run, "cold_chunk_cache_hits", args.min_cold_chunk_cache_hits, label=f"prompt {prompt_index}")
            runs.append(run)

        baseline = dict(runs[0])
        baseline["dense_latency_ms"] = mean(run["dense_latency_ms"] for run in runs)
        baseline["cache_latency_ms"] = mean(run["cache_latency_ms"] for run in runs)
        row: Dict[str, object] = {
            "prompt_index": prompt_index,
            "source_prompt_preview": prepared_prompt.source_prompt.replace("\n", " ")[:96],
            "prompt_preview": prepared_prompt.prompt_text.replace("\n", " ")[:96],
            "model_id": args.model_id,
            "layer_index": args.layer_index,
            "device": str(resolved_device),
            "torch_dtype": str(resolved_dtype),
            "source_prompt_tokens": prepared_prompt.source_prompt_tokens,
            "prompt_length": trace.prompt_length,
            "prompt_repeat_count": prepared_prompt.repeat_count,
            "generated_length": trace.generated_length,
            "total_tokens": trace.total_tokens,
            "generation_latency_ms": trace.generation_latency_ms,
            "capture_latency_ms": trace.capture_latency_ms,
            "min_prompt_tokens": -1 if args.min_prompt_tokens is None else args.min_prompt_tokens,
            "min_total_tokens": -1 if args.min_total_tokens is None else args.min_total_tokens,
            "min_reconstructed_tokens": -1 if args.min_reconstructed_tokens is None else args.min_reconstructed_tokens,
            "min_warm_chunk_decodes": -1 if args.min_warm_chunk_decodes is None else args.min_warm_chunk_decodes,
            "min_cold_chunk_decodes": -1 if args.min_cold_chunk_decodes is None else args.min_cold_chunk_decodes,
            "min_cold_chunk_cache_hits": -1 if args.min_cold_chunk_cache_hits is None else args.min_cold_chunk_cache_hits,
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
            f"prompt_repeat_count={prepared_prompt.repeat_count} "
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