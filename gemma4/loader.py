"""
Weight loader for Gemma 4 E4B — loads safetensors from a HuggingFace directory.

Handles the weight name mapping between HuggingFace's checkpoint format
and our from-scratch module names.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from safetensors import safe_open

from gemma4.config import Gemma4Config
from gemma4.model import Gemma4ForCausalLM

logger = logging.getLogger(__name__)


def _build_weight_map(config: Gemma4Config) -> dict[str, str]:
    """Build mapping from our parameter names to HF checkpoint names."""
    wmap: dict[str, str] = {}

    hf_prefix = "model.language_model"

    # Main token embedding
    wmap["model.embed_tokens.weight"] = f"{hf_prefix}.embed_tokens.weight"

    # Final norm
    wmap["model.norm.weight"] = f"{hf_prefix}.norm.weight"

    # Centralized PLE components
    if config.hidden_size_per_layer_input > 0:
        wmap["model.ple.embed_tokens_per_layer.weight"] = f"{hf_prefix}.embed_tokens_per_layer.weight"
        wmap["model.ple.per_layer_model_projection.weight"] = f"{hf_prefix}.per_layer_model_projection.weight"
        wmap["model.ple.per_layer_projection_norm.weight"] = f"{hf_prefix}.per_layer_projection_norm.weight"

    # Decoder layers
    for i in range(config.num_hidden_layers):
        our = f"model.layers.{i}"
        hf = f"{hf_prefix}.layers.{i}"

        # Attention
        wmap[f"{our}.self_attn.q_proj.weight"] = f"{hf}.self_attn.q_proj.weight"
        wmap[f"{our}.self_attn.k_proj.weight"] = f"{hf}.self_attn.k_proj.weight"
        wmap[f"{our}.self_attn.v_proj.weight"] = f"{hf}.self_attn.v_proj.weight"
        wmap[f"{our}.self_attn.o_proj.weight"] = f"{hf}.self_attn.o_proj.weight"

        # Attention QK norms
        wmap[f"{our}.self_attn.q_norm.weight"] = f"{hf}.self_attn.q_norm.weight"
        wmap[f"{our}.self_attn.k_norm.weight"] = f"{hf}.self_attn.k_norm.weight"

        # MLP
        wmap[f"{our}.mlp.gate_proj.weight"] = f"{hf}.mlp.gate_proj.weight"
        wmap[f"{our}.mlp.up_proj.weight"] = f"{hf}.mlp.up_proj.weight"
        wmap[f"{our}.mlp.down_proj.weight"] = f"{hf}.mlp.down_proj.weight"

        # Norms
        wmap[f"{our}.input_layernorm.weight"] = f"{hf}.input_layernorm.weight"
        wmap[f"{our}.post_attention_layernorm.weight"] = f"{hf}.post_attention_layernorm.weight"
        wmap[f"{our}.pre_feedforward_layernorm.weight"] = f"{hf}.pre_feedforward_layernorm.weight"
        wmap[f"{our}.post_feedforward_layernorm.weight"] = f"{hf}.post_feedforward_layernorm.weight"

        # Layer scalar
        wmap[f"{our}.layer_scalar"] = f"{hf}.layer_scalar"

        # Per-layer PLE components
        if config.hidden_size_per_layer_input > 0:
            wmap[f"{our}.per_layer_input_gate.weight"] = f"{hf}.per_layer_input_gate.weight"
            wmap[f"{our}.per_layer_projection.weight"] = f"{hf}.per_layer_projection.weight"
            wmap[f"{our}.post_per_layer_input_norm.weight"] = f"{hf}.post_per_layer_input_norm.weight"

    # LM head (if not tied)
    if not config.tie_word_embeddings:
        wmap["lm_head.weight"] = "model.language_model.lm_head.weight"

    return wmap


def _find_safetensor_files(model_dir: Path) -> list[Path]:
    """Find all safetensor shard files in a directory."""
    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No .safetensors files found in {model_dir}")
    return files


def load_weights(
    model: Gemma4ForCausalLM,
    model_dir: str | Path,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cpu",
) -> Gemma4ForCausalLM:
    """
    Load HuggingFace safetensors weights into our model.
    Loads weights **in-place** to avoid doubling memory usage.
    """
    model_dir = Path(model_dir)
    config = model.config

    wmap = _build_weight_map(config)
    # Invert: hf_name -> our_name
    hf_to_ours = {v: k for k, v in wmap.items()}

    safetensor_files = _find_safetensor_files(model_dir)

    # Build a lookup from our param names to the actual parameter objects
    param_dict = dict(model.named_parameters())
    buf_dict = dict(model.named_buffers())
    all_params = {**param_dict, **buf_dict}

    loaded_keys: set[str] = set()

    for st_file in safetensor_files:
        with safe_open(str(st_file), framework="pt", device=device) as f:
            for hf_key in f.keys():
                if hf_key not in hf_to_ours:
                    continue  # Skip vision/audio weights, etc.

                our_key = hf_to_ours[hf_key]
                if our_key not in all_params:
                    logger.warning(f"Key {our_key} in map but not in model, skipping")
                    continue

                tensor = f.get_tensor(hf_key).to(dtype=dtype)
                param = all_params[our_key]

                if tensor.shape != param.shape:
                    logger.warning(f"Shape mismatch for {our_key}: model={param.shape}, checkpoint={tensor.shape}")
                    continue

                # Copy in-place — no extra allocation
                with torch.no_grad():
                    param.copy_(tensor)
                del tensor  # free immediately

                loaded_keys.add(our_key)

    # Check for missing keys
    missing = [k for k in wmap if k not in loaded_keys]
    if missing:
        logger.warning(f"Missing {len(missing)} keys from checkpoint:\n" + "\n".join(f"  - {k}" for k in missing[:20]))

    logger.info(f"Loaded {len(loaded_keys)} / {len(wmap)} weight tensors")
    return model


def load_model(
    model_dir: str | Path,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
) -> Gemma4ForCausalLM:
    """
    Convenience function: load config + weights in one call.
    Initializes the model directly in the target dtype to minimize memory.
    """
    model_dir = Path(model_dir)
    config_path = model_dir / "config.json"

    if config_path.exists():
        config = Gemma4Config.from_json(config_path)
    else:
        logger.info("No config.json found, using default E4B config")
        config = Gemma4Config()

    logger.info(
        f"Initializing model: {config.num_hidden_layers} layers, hidden={config.hidden_size}, vocab={config.vocab_size}"
    )

    # Initialize directly in target dtype to halve memory (bf16 = 2 bytes vs float32 = 4 bytes)
    with torch.device("cpu"):
        torch.set_default_dtype(dtype)
        model = Gemma4ForCausalLM(config)
        torch.set_default_dtype(torch.float32)  # restore default

    logger.info(f"Loading weights from {model_dir}")
    load_weights(model, model_dir, dtype=dtype, device="cpu")

    logger.info(f"Moving model to {device}...")
    model = model.to(device=device)

    model.eval()
    return model
