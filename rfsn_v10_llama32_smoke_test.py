from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from llama32_adapter import (
    DEFAULT_MODEL_ID,
    DEFAULT_REPEAT_SEPARATOR,
    Llama32DecoderLayerMLX,
    build_rfsn_config_from_hf_config,
    capture_layer_trace,
    decode_escape_sequences,
    get_decoder_layers,
    get_rotary_embedding_module,
    load_model_and_tokenizer,
    prepare_prompt,
    require_min_total_tokens,
    run_layer_parity,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a real-prompt Llama 3.2 layer parity smoke test against the MLX RFSN cache path.",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Hugging Face model id to load")
    parser.add_argument("--prompt", default="Explain why cache compression trades memory for latency.")
    parser.add_argument("--prompt-repeat", type=int, default=1, help="Repeat the prompt this many times before capture")
    parser.add_argument(
        "--repeat-separator",
        default=DEFAULT_REPEAT_SEPARATOR.encode("unicode_escape").decode("ascii"),
        help="Separator inserted between repeated prompt copies; escape sequences like \\n are decoded",
    )
    parser.add_argument(
        "--min-prompt-tokens",
        type=int,
        default=None,
        help="Keep repeating the prompt until its tokenized length reaches at least this value",
    )
    parser.add_argument(
        "--min-total-tokens",
        type=int,
        default=None,
        help="Fail if prompt plus generated tokens stays below this value",
    )
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
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model, tokenizer, resolved_device, resolved_dtype = load_model_and_tokenizer(
        model_id=args.model_id,
        device=args.device,
        torch_dtype=args.torch_dtype,
    )
    prepared_prompt = prepare_prompt(
        prompt=args.prompt,
        tokenizer=tokenizer,
        repeat_count=args.prompt_repeat,
        min_prompt_tokens=args.min_prompt_tokens,
        repeat_separator=decode_escape_sequences(args.repeat_separator),
    )
    trace = capture_layer_trace(
        model=model,
        tokenizer=tokenizer,
        prompt=prepared_prompt.prompt_text,
        layer_index=args.layer_index,
        max_new_tokens=args.max_new_tokens,
    )
    require_min_total_tokens(trace, args.min_total_tokens)
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
    metrics = run_layer_parity(
        adapter=adapter,
        trace=trace,
        context_window=args.context_window,
        use_router=args.use_router,
    )

    result: Dict[str, Any] = {
        "model_id": args.model_id,
        "layer_index": args.layer_index,
        "device": str(resolved_device),
        "torch_dtype": str(resolved_dtype),
        "source_prompt": prepared_prompt.source_prompt,
        "prompt": prepared_prompt.prompt_text,
        "source_prompt_tokens": prepared_prompt.source_prompt_tokens,
        "prepared_prompt_tokens": prepared_prompt.prompt_tokens,
        "prompt_repeat_count": prepared_prompt.repeat_count,
        "prompt_length": trace.prompt_length,
        "generated_length": trace.generated_length,
        "generated_text": trace.generated_text,
        "generation_latency_ms": trace.generation_latency_ms,
        "capture_latency_ms": trace.capture_latency_ms,
        "min_prompt_tokens": args.min_prompt_tokens,
        "min_total_tokens": args.min_total_tokens,
        "context_window": args.context_window,
        "use_router": args.use_router,
        "disable_rvq": args.disable_rvq,
        **metrics,
    }

    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()