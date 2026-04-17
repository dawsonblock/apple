from __future__ import annotations

import importlib.util
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import mlx.core as mx
import numpy as np
import torch

from attention import _dense_exact_attention
from cache import RFSNv10KVCacheMLX
from quantization import HybridQuantizerMLX
from storage import AsyncHierarchicalRouterMLX, RFSNConfig


DEFAULT_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
DEFAULT_REPEAT_SEPARATOR = "\n\n"
DEFAULT_PROMPTS = [
    "Summarize why tiered KV caches are useful on memory-constrained hardware in two sentences.",
    "Write a Python function that merges two sorted lists into one sorted list.",
    "A store sells notebooks in packs of 3 and pens in packs of 5. What is the smallest number of items that can be bought using both pack sizes exactly once?",
]


@dataclass
class LayerTrace:
    prompt_text: str
    generated_text: str
    prompt_length: int
    generated_length: int
    input_ids: torch.Tensor
    input_hidden_states: torch.Tensor
    output_hidden_states: torch.Tensor
    position_ids: Optional[torch.Tensor]
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]]
    generation_latency_ms: float
    capture_latency_ms: float

    @property
    def total_tokens(self) -> int:
        return int(self.input_ids.shape[-1])


@dataclass
class PreparedPrompt:
    source_prompt: str
    prompt_text: str
    source_prompt_tokens: int
    prompt_tokens: int
    repeat_count: int


def _require_transformers() -> Tuple[Any, Any]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Install optional model dependencies with `python3 -m pip install -r requirements-models-optional.txt` before using the Llama adapter."
        ) from exc
    return AutoModelForCausalLM, AutoTokenizer


def decode_escape_sequences(value: str) -> str:
    return value.encode("utf-8").decode("unicode_escape")


def count_prompt_tokens(tokenizer: Any, prompt: str) -> int:
    encoded = tokenizer(prompt, return_tensors="pt")
    return int(encoded["input_ids"].shape[-1])


def prepare_prompt(
    prompt: str,
    tokenizer: Any,
    repeat_count: int = 1,
    min_prompt_tokens: Optional[int] = None,
    repeat_separator: str = DEFAULT_REPEAT_SEPARATOR,
) -> PreparedPrompt:
    if repeat_count < 1:
        raise ValueError("repeat_count must be at least 1")
    if min_prompt_tokens is not None and min_prompt_tokens < 1:
        raise ValueError("min_prompt_tokens must be at least 1 when provided")

    source_prompt_tokens = count_prompt_tokens(tokenizer, prompt)
    prompt_parts = [prompt] * repeat_count
    prompt_text = repeat_separator.join(prompt_parts)
    prompt_tokens = count_prompt_tokens(tokenizer, prompt_text)
    effective_repeat_count = repeat_count

    while min_prompt_tokens is not None and prompt_tokens < min_prompt_tokens:
        prompt_parts.append(prompt)
        prompt_text = repeat_separator.join(prompt_parts)
        updated_prompt_tokens = count_prompt_tokens(tokenizer, prompt_text)
        if updated_prompt_tokens <= prompt_tokens:
            raise RuntimeError(
                "Prompt expansion stalled without increasing token count. Use a different base prompt or separator."
            )
        prompt_tokens = updated_prompt_tokens
        effective_repeat_count += 1

    return PreparedPrompt(
        source_prompt=prompt,
        prompt_text=prompt_text,
        source_prompt_tokens=source_prompt_tokens,
        prompt_tokens=prompt_tokens,
        repeat_count=effective_repeat_count,
    )


def require_min_total_tokens(trace: LayerTrace, min_total_tokens: Optional[int], label: str = "prompt") -> None:
    if min_total_tokens is None:
        return
    if min_total_tokens < 1:
        raise ValueError("min_total_tokens must be at least 1 when provided")
    if trace.total_tokens >= min_total_tokens:
        return
    raise RuntimeError(
        f"{label} produced {trace.total_tokens} total tokens, below the required minimum of {min_total_tokens}. "
        "Increase --prompt-repeat, raise --max-new-tokens, or set --min-prompt-tokens higher."
    )


def resolve_torch_device(requested: str) -> torch.device:
    choice = requested.lower()
    if choice == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if choice == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available in this environment.")
        return torch.device("mps")
    if choice == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device {requested!r}; expected one of auto, mps, cpu")


def resolve_torch_dtype(requested: str, device: torch.device) -> torch.dtype:
    choice = requested.lower()
    if choice == "auto":
        return torch.float16 if device.type == "mps" else torch.float32
    mapping = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    if choice not in mapping:
        raise ValueError(f"Unsupported dtype {requested!r}; expected auto, float16, float32, or bfloat16")
    return mapping[choice]


def load_model_and_tokenizer(
    model_id: str = DEFAULT_MODEL_ID,
    device: str = "auto",
    torch_dtype: str = "auto",
) -> Tuple[Any, Any, torch.device, torch.dtype]:
    AutoModelForCausalLM, AutoTokenizer = _require_transformers()
    resolved_device = resolve_torch_device(device)
    resolved_dtype = resolve_torch_dtype(torch_dtype, resolved_device)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        load_kwargs: Dict[str, Any] = {"torch_dtype": resolved_dtype}
        if importlib.util.find_spec("accelerate") is not None:
            load_kwargs["low_cpu_mem_usage"] = True
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    except Exception as exc:
        message = str(exc)
        if "gated repo" in message.lower() or "restricted" in message.lower():
            raise RuntimeError(
                "Failed to load the requested model. For Meta Llama 3.2 checkpoints, accept the Hugging Face license terms for the repo and authenticate locally before retrying."
            ) from exc
        raise RuntimeError(f"Failed to load the requested model: {message}") from exc

    model.to(resolved_device)
    model.eval()
    return model, tokenizer, resolved_device, resolved_dtype


def get_decoder_layers(model: torch.nn.Module) -> Sequence[torch.nn.Module]:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise TypeError("Expected a decoder-only Hugging Face model with model.layers")


def get_rotary_embedding_module(model: torch.nn.Module, layer: torch.nn.Module) -> Optional[torch.nn.Module]:
    if hasattr(model, "model") and hasattr(model.model, "rotary_emb"):
        return model.model.rotary_emb
    if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "rotary_emb"):
        return layer.self_attn.rotary_emb
    return None


def _pick_subspace_layout(head_dim: int) -> Tuple[int, int]:
    upper_bound = min(16, head_dim)
    for subspace_dim in range(upper_bound, 0, -1):
        if head_dim % subspace_dim == 0:
            return head_dim // subspace_dim, subspace_dim
    return head_dim, 1


def build_rfsn_config_from_hf_config(
    hf_config: Any,
    hot_capacity: int = 64,
    warm_capacity: int = 256,
    cold_capacity: int = 512,
    block_size_seq: int = 32,
    disable_rvq: bool = False,
    disk_cache_dir: str = "./rfsn_llama32_disk_cache",
) -> RFSNConfig:
    hidden_dim = int(hf_config.hidden_size)
    num_heads = int(hf_config.num_attention_heads)
    head_dim = hidden_dim // num_heads
    num_subspaces, subspace_dim = _pick_subspace_layout(head_dim)
    return RFSNConfig(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        num_layers=int(hf_config.num_hidden_layers),
        num_subspaces=num_subspaces,
        subspace_dim=subspace_dim,
        num_rvq_layers=0 if disable_rvq else 4,
        rvq_codebook_size=256,
        rvq_sparsity_threshold=0.01,
        max_rvq_sparse=64,
        hot_capacity=hot_capacity,
        warm_capacity=warm_capacity,
        cold_capacity=cold_capacity,
        block_size_seq=block_size_seq,
        disk_cache_dir=disk_cache_dir,
        prefetch_throttle_s=0.0,
    )


def capture_layer_trace(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    layer_index: int,
    max_new_tokens: int = 32,
) -> LayerTrace:
    device = next(model.parameters()).device
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids)).to(device)
    prompt_length = int(input_ids.shape[1])
    layers = get_decoder_layers(model)
    if layer_index < 0 or layer_index >= len(layers):
        raise IndexError(f"Layer index {layer_index} is out of range for a {len(layers)}-layer model")

    generation_latency_ms = 0.0
    with torch.no_grad():
        if max_new_tokens > 0:
            generation_start = time.perf_counter()
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
            )
            generation_latency_ms = (time.perf_counter() - generation_start) * 1000.0
        else:
            generated_ids = input_ids

    capture: Dict[str, Any] = {}

    def hook(module: torch.nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any], output: Any) -> None:
        del module
        hidden_states = args[0] if args else kwargs.get("hidden_states")
        if hidden_states is None:
            raise RuntimeError("Unable to locate decoder-layer input hidden states from the forward hook")
        layer_output = output[0] if isinstance(output, tuple) else output
        position_ids = kwargs.get("position_ids")
        position_embeddings = kwargs.get("position_embeddings")
        capture["input_hidden_states"] = hidden_states.detach().cpu()
        capture["output_hidden_states"] = layer_output.detach().cpu()
        capture["position_ids"] = position_ids.detach().cpu() if isinstance(position_ids, torch.Tensor) else None
        if (
            isinstance(position_embeddings, (tuple, list))
            and len(position_embeddings) == 2
            and all(isinstance(item, torch.Tensor) for item in position_embeddings)
        ):
            capture["position_embeddings"] = (
                position_embeddings[0].detach().cpu(),
                position_embeddings[1].detach().cpu(),
            )
        else:
            capture["position_embeddings"] = None

    handle = layers[layer_index].register_forward_hook(hook, with_kwargs=True)
    capture_start = time.perf_counter()
    with torch.no_grad():
        generated_attention_mask = torch.ones_like(generated_ids, device=generated_ids.device)
        model(input_ids=generated_ids, attention_mask=generated_attention_mask, use_cache=False)
    capture_latency_ms = (time.perf_counter() - capture_start) * 1000.0
    handle.remove()

    if not capture:
        raise RuntimeError("Decoder-layer hook did not capture any activations")

    if capture["position_ids"] is None and capture["position_embeddings"] is None:
        seq_len = int(generated_ids.shape[1])
        capture["position_ids"] = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

    generated_length = int(generated_ids.shape[1]) - prompt_length
    generated_text = ""
    if generated_length > 0:
        generated_text = tokenizer.decode(generated_ids[0, prompt_length:], skip_special_tokens=True)

    return LayerTrace(
        prompt_text=prompt,
        generated_text=generated_text,
        prompt_length=prompt_length,
        generated_length=generated_length,
        input_ids=generated_ids.detach().cpu(),
        input_hidden_states=capture["input_hidden_states"],
        output_hidden_states=capture["output_hidden_states"],
        position_ids=capture["position_ids"],
        position_embeddings=capture["position_embeddings"],
        generation_latency_ms=generation_latency_ms,
        capture_latency_ms=capture_latency_ms,
    )


def _to_mx_array(value: Any, dtype: Optional[mx.Dtype] = None) -> mx.array:
    if isinstance(value, torch.Tensor):
        array = mx.array(value.detach().cpu().numpy())
    else:
        array = mx.array(np.asarray(value))
    return array.astype(dtype) if dtype is not None else array


def _rmse(lhs: mx.array, rhs: mx.array) -> float:
    diff = lhs.astype(mx.float32) - rhs.astype(mx.float32)
    return float(mx.sqrt(mx.mean(diff * diff)))


def _max_abs(lhs: mx.array, rhs: mx.array) -> float:
    diff = lhs.astype(mx.float32) - rhs.astype(mx.float32)
    return float(mx.max(mx.abs(diff)))


def _rotate_half(x: mx.array) -> mx.array:
    half = int(x.shape[-1]) // 2
    return mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def _repeat_kv(x: mx.array, repeats: int) -> mx.array:
    if repeats == 1:
        return x
    batch, seq_len, num_kv_heads, head_dim = (int(dim) for dim in x.shape)
    expanded = x[:, :, :, None, :]
    expanded = mx.broadcast_to(expanded, (batch, seq_len, num_kv_heads, repeats, head_dim))
    return expanded.reshape(batch, seq_len, num_kv_heads * repeats, head_dim)


def _normalize_position_component(component: Any, seq_len: int, head_dim: int) -> mx.array:
    data = np.asarray(component.detach().cpu().numpy() if isinstance(component, torch.Tensor) else component)
    data = np.squeeze(data)
    if data.ndim == 1:
        data = data.reshape(1, 1, -1)
    elif data.ndim == 2:
        data = data.reshape(1, data.shape[0], data.shape[1])
    elif data.ndim == 3:
        if data.shape[0] != 1:
            raise ValueError("Only batch size 1 is supported for the Llama adapter")
    else:
        raise ValueError(f"Unsupported rotary embedding rank {data.ndim}")
    if data.shape[1] != seq_len:
        raise ValueError(f"Expected {seq_len} position rows, got {data.shape[1]}")
    if data.shape[2] != head_dim:
        raise ValueError(f"Expected rotary head dim {head_dim}, got {data.shape[2]}")
    return mx.array(data).astype(mx.float16)


def slice_position_embeddings(
    position_embeddings: Tuple[mx.array, mx.array],
    start: int,
    end: int,
) -> Tuple[mx.array, mx.array]:
    cos, sin = position_embeddings
    return cos[:, start:end], sin[:, start:end]


class Llama32DecoderLayerMLX:
    def __init__(
        self,
        config: RFSNConfig,
        layer_idx: int,
        num_key_value_heads: int,
        rms_norm_eps: float,
        input_layernorm_weight: mx.array,
        post_attention_layernorm_weight: mx.array,
        q_proj_weight: mx.array,
        k_proj_weight: mx.array,
        v_proj_weight: mx.array,
        o_proj_weight: mx.array,
        gate_proj_weight: mx.array,
        up_proj_weight: mx.array,
        down_proj_weight: mx.array,
        rotary_module: Optional[torch.nn.Module] = None,
    ):
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim ** -0.5
        self.rms_norm_eps = rms_norm_eps

        self.input_layernorm_weight = input_layernorm_weight.astype(mx.float16)
        self.post_attention_layernorm_weight = post_attention_layernorm_weight.astype(mx.float16)
        self.q_proj_weight = q_proj_weight.astype(mx.float16)
        self.k_proj_weight = k_proj_weight.astype(mx.float16)
        self.v_proj_weight = v_proj_weight.astype(mx.float16)
        self.o_proj_weight = o_proj_weight.astype(mx.float16)
        self.gate_proj_weight = gate_proj_weight.astype(mx.float16)
        self.up_proj_weight = up_proj_weight.astype(mx.float16)
        self.down_proj_weight = down_proj_weight.astype(mx.float16)

        self.quantizer = HybridQuantizerMLX(config)
        self.disk_cache_dir = Path(config.disk_cache_dir)
        self.cache = RFSNv10KVCacheMLX(config, layer_idx=layer_idx)
        self.rotary_module = rotary_module

    @classmethod
    def from_hf_layer(
        cls,
        hf_layer: torch.nn.Module,
        hf_config: Any,
        rfsn_config: RFSNConfig,
        layer_idx: int,
        rotary_module: Optional[torch.nn.Module] = None,
    ) -> "Llama32DecoderLayerMLX":
        num_key_value_heads = int(getattr(hf_config, "num_key_value_heads", hf_config.num_attention_heads))
        return cls(
            config=rfsn_config,
            layer_idx=layer_idx,
            num_key_value_heads=num_key_value_heads,
            rms_norm_eps=float(getattr(hf_config, "rms_norm_eps", 1e-5)),
            input_layernorm_weight=_to_mx_array(hf_layer.input_layernorm.weight, mx.float16),
            post_attention_layernorm_weight=_to_mx_array(hf_layer.post_attention_layernorm.weight, mx.float16),
            q_proj_weight=_to_mx_array(hf_layer.self_attn.q_proj.weight, mx.float16),
            k_proj_weight=_to_mx_array(hf_layer.self_attn.k_proj.weight, mx.float16),
            v_proj_weight=_to_mx_array(hf_layer.self_attn.v_proj.weight, mx.float16),
            o_proj_weight=_to_mx_array(hf_layer.self_attn.o_proj.weight, mx.float16),
            gate_proj_weight=_to_mx_array(hf_layer.mlp.gate_proj.weight, mx.float16),
            up_proj_weight=_to_mx_array(hf_layer.mlp.up_proj.weight, mx.float16),
            down_proj_weight=_to_mx_array(hf_layer.mlp.down_proj.weight, mx.float16),
            rotary_module=rotary_module,
        )

    def reset_cache(self, remove_disk: bool = True) -> None:
        if remove_disk and self.disk_cache_dir.exists():
            shutil.rmtree(self.disk_cache_dir)
        self.cache = RFSNv10KVCacheMLX(self.config, layer_idx=self.layer_idx)

    def _linear(self, x: mx.array, weight: mx.array) -> mx.array:
        output = mx.matmul(x.astype(mx.float32), mx.transpose(weight.astype(mx.float32)))
        return output.astype(mx.float16)

    def _rms_norm(self, x: mx.array, weight: mx.array) -> mx.array:
        variance = mx.mean(x.astype(mx.float32) ** 2, axis=-1, keepdims=True)
        normalized = x.astype(mx.float32) / mx.sqrt(variance + self.rms_norm_eps)
        return (normalized * weight.astype(mx.float32)).astype(mx.float16)

    def _silu(self, x: mx.array) -> mx.array:
        x_f32 = x.astype(mx.float32)
        return (x_f32 * mx.sigmoid(x_f32)).astype(mx.float16)

    def _reshape_query(self, projected: mx.array) -> mx.array:
        batch, seq_len, _ = (int(dim) for dim in projected.shape)
        return projected.reshape(batch, seq_len, self.num_heads, self.head_dim).astype(mx.float16)

    def _reshape_kv(self, projected: mx.array) -> mx.array:
        batch, seq_len, _ = (int(dim) for dim in projected.shape)
        return projected.reshape(batch, seq_len, self.num_key_value_heads, self.head_dim).astype(mx.float16)

    def _merge_heads(self, attended: mx.array) -> mx.array:
        batch, seq_len, _, _ = (int(dim) for dim in attended.shape)
        return attended.reshape(batch, seq_len, self.num_heads * self.head_dim)

    def _position_embeddings_for_sequence(
        self,
        seq_len: int,
        position_embeddings: Optional[Tuple[Any, Any]],
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[mx.array, mx.array]:
        if position_embeddings is not None:
            return (
                _normalize_position_component(position_embeddings[0], seq_len, self.head_dim),
                _normalize_position_component(position_embeddings[1], seq_len, self.head_dim),
            )
        if self.rotary_module is None:
            raise RuntimeError(
                "No position embeddings were captured and no rotary embedding module is available to recompute them."
            )
        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
        rotary_device = position_ids.device
        rotary_buffer = next(self.rotary_module.buffers(), None)
        if rotary_buffer is not None:
            rotary_device = rotary_buffer.device
        else:
            rotary_param = next(self.rotary_module.parameters(), None)
            if rotary_param is not None:
                rotary_device = rotary_param.device
        position_ids = position_ids.to(rotary_device)
        rotary_hidden = torch.zeros((1, seq_len, self.hidden_dim), dtype=torch.float32, device=rotary_device)
        with torch.no_grad():
            try:
                cos, sin = self.rotary_module(rotary_hidden, position_ids)
            except TypeError:
                cos, sin = self.rotary_module(position_ids)
        return (
            _normalize_position_component(cos, seq_len, self.head_dim),
            _normalize_position_component(sin, seq_len, self.head_dim),
        )

    def _apply_rope(
        self,
        queries: mx.array,
        keys: mx.array,
        position_embeddings: Tuple[mx.array, mx.array],
    ) -> Tuple[mx.array, mx.array]:
        cos, sin = position_embeddings
        cos_f32 = cos.astype(mx.float32)[:, :, None, :]
        sin_f32 = sin.astype(mx.float32)[:, :, None, :]
        q_f32 = queries.astype(mx.float32)
        k_f32 = keys.astype(mx.float32)
        rotated_q = q_f32 * cos_f32 + _rotate_half(q_f32) * sin_f32
        rotated_k = k_f32 * cos_f32 + _rotate_half(k_f32) * sin_f32
        return rotated_q.astype(mx.float16), rotated_k.astype(mx.float16)

    def _project_qkv(
        self,
        hidden_states: mx.array,
        position_embeddings: Tuple[mx.array, mx.array],
    ) -> Tuple[mx.array, mx.array, mx.array]:
        normed = self._rms_norm(hidden_states, self.input_layernorm_weight)
        queries = self._reshape_query(self._linear(normed, self.q_proj_weight))
        keys = self._reshape_kv(self._linear(normed, self.k_proj_weight))
        values = self._reshape_kv(self._linear(normed, self.v_proj_weight))
        queries, keys = self._apply_rope(queries, keys, position_embeddings)
        repeated_keys = _repeat_kv(keys, self.num_key_value_groups)
        repeated_values = _repeat_kv(values, self.num_key_value_groups)
        return queries, repeated_keys, repeated_values

    def _forward_mlp(self, hidden_states: mx.array) -> mx.array:
        mlp_input = self._rms_norm(hidden_states, self.post_attention_layernorm_weight)
        gate = self._linear(mlp_input, self.gate_proj_weight)
        up = self._linear(mlp_input, self.up_proj_weight)
        activated = self._silu(gate) * up
        return self._linear(activated, self.down_proj_weight)

    def prefill_cache(
        self,
        hidden_states: mx.array,
        position_embeddings: Optional[Tuple[Any, Any]] = None,
        position_ids: Optional[torch.Tensor] = None,
        reset: bool = True,
    ) -> None:
        if int(hidden_states.shape[0]) != 1:
            raise ValueError("prefill_cache currently supports batch size 1 only")
        if int(hidden_states.shape[1]) == 0:
            if reset:
                self.reset_cache(remove_disk=True)
            return
        if reset:
            self.reset_cache(remove_disk=True)
        seq_len = int(hidden_states.shape[1])
        normalized_positions = self._position_embeddings_for_sequence(seq_len, position_embeddings, position_ids)
        _, keys, values = self._project_qkv(hidden_states.astype(mx.float16), normalized_positions)
        self.cache.update(
            keys[0].astype(mx.float16),
            values[0].astype(mx.float16),
            self.quantizer,
            disk_dir=self.disk_cache_dir,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        position_embeddings: Optional[Tuple[Any, Any]] = None,
        position_ids: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        router: Optional[AsyncHierarchicalRouterMLX] = None,
        context_window: Optional[int] = None,
    ) -> mx.array:
        if int(hidden_states.shape[0]) != 1:
            raise ValueError("The Llama adapter currently supports batch size 1 only")
        seq_len = int(hidden_states.shape[1])
        normalized_positions = self._position_embeddings_for_sequence(seq_len, position_embeddings, position_ids)

        residual = hidden_states.astype(mx.float16)
        queries, keys, values = self._project_qkv(residual, normalized_positions)

        if use_cache:
            if seq_len != 1:
                raise ValueError("cache-backed decode expects input shape [1, 1, hidden_dim]")
            self.cache.update(
                keys[0].astype(mx.float16),
                values[0].astype(mx.float16),
                self.quantizer,
                disk_dir=self.disk_cache_dir,
            )
            query_position = mx.array([self.cache.total_tokens - 1], dtype=mx.int32)
            attended = self.cache.attention_forward(
                q=queries[:, 0].astype(mx.float16),
                causal=True,
                query_positions=query_position,
                router=router,
                current_position=self.cache.total_tokens - 1,
                context_window=context_window,
            ).reshape(1, 1, self.num_heads, self.head_dim)
        else:
            attended = _dense_exact_attention(
                q=queries,
                keys=keys,
                values=values,
                scale=self.scale,
                causal=True,
            )

        attn_output = self._linear(self._merge_heads(attended), self.o_proj_weight)
        hidden = residual + attn_output
        return hidden + self._forward_mlp(hidden)


def trace_position_embeddings(
    adapter: Llama32DecoderLayerMLX,
    trace: LayerTrace,
) -> Tuple[mx.array, mx.array]:
    return adapter._position_embeddings_for_sequence(
        trace.total_tokens,
        trace.position_embeddings,
        trace.position_ids,
    )


def run_layer_parity(
    adapter: Llama32DecoderLayerMLX,
    trace: LayerTrace,
    context_window: Optional[int] = None,
    use_router: bool = False,
) -> Dict[str, float]:
    hidden_in = _to_mx_array(trace.input_hidden_states, mx.float16)
    expected_out = _to_mx_array(trace.output_hidden_states, mx.float16)
    position_embeddings = trace_position_embeddings(adapter, trace)

    dense_start = time.perf_counter()
    dense_out = adapter(hidden_in, position_embeddings=position_embeddings, use_cache=False)
    dense_latency_ms = (time.perf_counter() - dense_start) * 1000.0

    adapter.reset_cache(remove_disk=True)
    prefix_hidden = hidden_in[:, :-1]
    last_hidden = hidden_in[:, -1:]
    prefix_embeddings = slice_position_embeddings(position_embeddings, 0, max(trace.total_tokens - 1, 0))
    last_embeddings = slice_position_embeddings(position_embeddings, trace.total_tokens - 1, trace.total_tokens)
    if int(prefix_hidden.shape[1]) > 0:
        adapter.prefill_cache(prefix_hidden, position_embeddings=prefix_embeddings, reset=True)

    router = AsyncHierarchicalRouterMLX(adapter.config, layer_idx=adapter.layer_idx) if use_router else None
    cache_start = time.perf_counter()
    cache_out = adapter(
        last_hidden,
        position_embeddings=last_embeddings,
        use_cache=True,
        router=router,
        context_window=context_window,
    )
    cache_latency_ms = (time.perf_counter() - cache_start) * 1000.0

    expected_last = expected_out[:, -1:]
    dense_last = dense_out[:, -1:]
    access_stats = adapter.cache.get_last_access_stats()
    return {
        "dense_latency_ms": dense_latency_ms,
        "cache_latency_ms": cache_latency_ms,
        "dense_output_rmse": _rmse(dense_out, expected_out),
        "dense_output_max_abs": _max_abs(dense_out, expected_out),
        "dense_last_token_rmse": _rmse(dense_last, expected_last),
        "cache_last_token_rmse": _rmse(cache_out, expected_last),
        "cache_last_token_max_abs": _max_abs(cache_out, expected_last),
        "cache_vs_dense_last_token_rmse": _rmse(cache_out, dense_last),
        "window_tokens": float(access_stats["window_tokens"]),
        "reconstructed_tokens": float(access_stats["reconstructed_tokens"]),
        "warm_chunk_decodes": float(access_stats["warm_chunk_decodes"]),
        "cold_chunk_decodes": float(access_stats["cold_chunk_decodes"]),
        "cold_chunk_cache_hits": float(access_stats["cold_chunk_cache_hits"]),
        "cold_chunk_cache_misses": float(access_stats["cold_chunk_cache_misses"]),
    }


def load_prompts_from_file(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]