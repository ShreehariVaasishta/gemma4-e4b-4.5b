#!/usr/bin/env python3
"""
CLI entrypoint for text generation with Gemma 4 E4B.

Usage:
    python generate.py --model-dir ./weights --prompt "What is gravity?"
    python generate.py --model-dir ./weights --prompt "Hello" --temperature 0.0  # greedy
"""

from pathlib import Path
import argparse
import logging
import time

import torch
from tokenizers import Tokenizer

from gemma4.loader import load_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate text with Gemma 4 E4B")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="/media/sln/codeit/projs/gemma4-e4b-4.5b/llm-models/Gemma-4-E4B",
        help="Path to HF model directory with safetensors + tokenizer",
    )
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt for generation")
    parser.add_argument("--max-tokens", type=int, default=256, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    args = parser.parse_args()

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    # ── Load tokenizer ──────────────────────────────────────────────
    tokenizer_path = Path(args.model_dir, "tokenizer.json").as_posix()
    logger.info(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = Tokenizer.from_file(tokenizer_path)

    # ── Load model ──────────────────────────────────────────────────
    model = load_model(
        args.model_dir,
        dtype=dtype_map[args.dtype],
        device=args.device,
    )
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model loaded: {total_params / 1e9:.2f}B parameters")

    # ── Tokenize prompt ─────────────────────────────────────────────
    # Gemma uses <bos> prefix (token id 2)
    encoded = tokenizer.encode(args.prompt)
    token_ids = [2] + encoded.ids  # prepend BOS
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=args.device)
    logger.info(f"Prompt tokens: {len(token_ids)}")

    # ── Generate ────────────────────────────────────────────────────
    logger.info("Generating...")
    t0 = time.perf_counter()

    output_ids = model.generate(
        input_ids,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )

    elapsed = time.perf_counter() - t0
    new_tokens = output_ids.shape[1] - input_ids.shape[1]
    tps = new_tokens / elapsed if elapsed > 0 else 0

    # ── Decode output ───────────────────────────────────────────────
    output_text = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)

    print("\n" + "=" * 72)
    print(f"PROMPT: {args.prompt}")
    print("-" * 72)
    print(f"OUTPUT: {output_text}")
    print("=" * 72)
    print(f"\n  Generated {new_tokens} tokens in {elapsed:.2f}s ({tps:.1f} tok/s)")


if __name__ == "__main__":
    main()
