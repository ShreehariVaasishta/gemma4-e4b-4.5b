#!/usr/bin/env python3
"""
Validate our from-scratch implementation against HuggingFace transformers.

Compares logits for a given prompt to verify correctness.

Usage:
    pip install -e .[dev]
    python validate.py --model-dir ./weights --prompt "Hello, world!"
"""

import argparse
import logging

import torch
from sentencepiece import SentencePieceProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Validate against HuggingFace")
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="Hello, world!")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device
    dtype = torch.bfloat16

    # ── Load tokenizer ──────────────────────────────────────────────
    tokenizer = SentencePieceProcessor(model_file=f"{args.model_dir}/tokenizer.model")
    token_ids = [tokenizer.bos_id()] + tokenizer.encode(args.prompt)
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    logger.info(f"Prompt: {args.prompt!r} -> {len(token_ids)} tokens")

    # ── Our model ───────────────────────────────────────────────────
    logger.info("Loading our model...")
    from gemma4.loader import load_model
    our_model = load_model(args.model_dir, dtype=dtype, device=device)

    with torch.inference_mode():
        our_logits = our_model(input_ids)  # [1, seq_len, vocab]

    our_top5 = our_logits[0, -1].topk(5)
    logger.info(f"Our top-5 tokens:  {our_top5.indices.tolist()}")
    logger.info(f"Our top-5 logits:  {our_top5.values.tolist()}")

    # Free memory
    del our_model
    torch.cuda.empty_cache()

    # ── HuggingFace model ───────────────────────────────────────────
    logger.info("Loading HuggingFace model...")
    from transformers import AutoModelForCausalLM

    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=dtype,
        device_map=device,
    )

    with torch.inference_mode():
        hf_output = hf_model(input_ids)
        hf_logits = hf_output.logits

    hf_top5 = hf_logits[0, -1].topk(5)
    logger.info(f"HF top-5 tokens:   {hf_top5.indices.tolist()}")
    logger.info(f"HF top-5 logits:   {hf_top5.values.tolist()}")

    # ── Compare ─────────────────────────────────────────────────────
    diff = (our_logits.float() - hf_logits.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    logger.info(f"Max abs diff:  {max_diff:.6f}")
    logger.info(f"Mean abs diff: {mean_diff:.6f}")

    # Check top-1 agreement
    our_pred = our_logits[0, -1].argmax().item()
    hf_pred = hf_logits[0, -1].argmax().item()

    if our_pred == hf_pred:
        logger.info(f"✅ Top-1 prediction matches: token {our_pred} "
                     f"({tokenizer.decode([our_pred])!r})")
    else:
        logger.warning(f"❌ Top-1 mismatch: ours={our_pred}, HF={hf_pred}")

    if max_diff < 0.1:
        logger.info("✅ Logits match within tolerance (max_diff < 0.1)")
    elif max_diff < 1.0:
        logger.warning(f"⚠️ Logits close but not exact (max_diff={max_diff:.4f})")
    else:
        logger.error(f"❌ Logits diverge significantly (max_diff={max_diff:.4f})")


if __name__ == "__main__":
    main()
