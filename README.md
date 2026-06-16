# Gemma 4 E4B — From-Scratch Inference

A minimal, from-scratch implementation of **Gemma 4 E4B** (4.5B effective parameters)
for text-only inference in PyTorch.

## Architecture Highlights

| Property | Value |
|---|---|
| Effective Parameters | 4.5B |
| Total Parameters | ~8B (incl. PLE embeddings) |
| Hidden Size | 2560 |
| Layers | 42 |
| Attention Heads | 8 (GQA, 2 KV heads) |
| Head Dim | 256 (sliding), 512 (global) |
| Intermediate Size | 10240 |
| Vocab Size | 262,144 |
| Context Length | 128K tokens |
| Sliding Window | 512 tokens |
| KV Shared Layers | 18 |
| Activation | GELU (tanh approx) |
| Logit Softcapping | 30.0 |

### Key Innovations
- **Per-Layer Embeddings (PLE)** — each decoder layer gets its own token embedding lookup
  (dim 256, full vocab), gated into the residual stream
- **Hybrid Attention** — 5 sliding-window layers then 1 global layer, repeated 7×
- **KV Sharing** — last 18 layers reuse KV from earlier layers
- **Proportional RoPE** — global layers use partial rotary (25%) with θ=1M
- **Logit Softcapping** — final logits capped at 30.0

## Project Structure

```
gemma4-e4b-4.5b/
├── pyproject.toml
├── README.md
├── gemma4/
│   ├── __init__.py
│   ├── config.py          # Model hyperparameters from config.json
│   ├── model.py           # Full model: embeddings → decoder → LM head
│   ├── layers/
│   │   ├── __init__.py
│   │   ├── attention.py   # Sliding window + global attention with GQA
│   │   ├── mlp.py         # GeLU feed-forward
│   │   ├── norm.py        # RMSNorm
│   │   └── rope.py        # RoPE + Proportional RoPE
│   ├── sampling.py        # Top-k, top-p, temperature sampling
│   └── loader.py          # Load safetensors weights from HF hub
├── generate.py            # CLI entrypoint for text generation
└── validate.py            # Compare outputs vs HuggingFace transformers
```

## Usage

```bash
# Install
pip install -e .

# Generate text
python generate.py --prompt "Explain quantum computing" --max-tokens 256

# Validate against HuggingFace (requires `pip install -e .[dev]`)
python validate.py --prompt "Hello, world!"
```

## Model Weights

Download from [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it):

```bash
huggingface-cli download google/gemma-4-E4B-it --local-dir ./weights
```
