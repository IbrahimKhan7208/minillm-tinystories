# MiniLLM: A Modern Transformer Language Model Built From Scratch

A small decoder-only language model, trained from scratch on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset, implementing several modern architectural techniques used in current large language models.

This project was built as a hands-on learning exercise — every component (tokenizer, attention mechanism, positional embeddings, training loop) was implemented and understood line-by-line, not copied from an existing implementation.

## Architecture

- **RMSNorm** — simplified LayerNorm (scale-only, no mean-centering)
- **RoPE (Rotary Positional Embeddings)** — relative position encoding via rotation, applied to queries/keys
- **Grouped Query Attention (GQA)** — 8 query heads sharing 2 key/value heads for compute efficiency
- **SwiGLU** — gated feed-forward network with SiLU activation
- **Pre-norm Transformer blocks** with residual connections and dropout
- **Weight-tied embeddings** between input embedding and output projection
- **Custom byte-level BPE tokenizer** — trained from scratch (not pretrained), vocab size 16,000

## Model Details

| | |
|---|---|
| Parameters | ~15-18M |
| Layers | 8 |
| d_model | 384 |
| Attention heads | 8 (query) / 2 (key-value) |
| Context length | 256 tokens |
| Training data | 1,000,000 TinyStories, ~220M tokens |
| Training steps | 13,000 (≈1 epoch) |
| Final train / val loss | 1.62 / 1.63 |
| Final perplexity | 5.05 / 5.10 |

## Usage

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download model weights
Download `minillm_final.pt` and `tokenizer.json` from [Hugging Face](#) *(link added after upload)* and place them in the `weights/` folder.

### 3. Generate text
```bash
python inference.py --prompt "Once upon a time" --temperature 0.8 --max_new_tokens 150
```

Optional arguments:
- `--temperature` (default 0.8) — higher values produce more random/diverse output
- `--max_new_tokens` (default 150) — number of tokens to generate

## Training

The full training script is included in `train.py` for transparency and reproducibility. It was run on a Google Colab T4 GPU; training on CPU is not practical due to time (~30-45 min on GPU vs. many hours on CPU).

```bash
python train.py
```

## Sample Output

**Prompt:** "Once upon a time"

> Once upon a time, there was a little girl named Lily. She loved to play outside and explore. One day, she went on a walk and saw a big bright sun in the sky. It was very pretty and she wanted to touch it. Lily walked over to the sun and looked up at it...

More examples in [`examples/sample_outputs.md`](examples/sample_outputs.md).

## Known Limitations

- Trained on a relatively small (1M story), simple-vocabulary dataset — not suitable for general-purpose text generation.
- No learning rate scheduling or mixed-precision training were used in this iteration (deliberate scope decisions, not oversights).
- As with most small language models, plot logic can wander over longer generations even when grammar stays coherent.

## Project Background

This was built as a staged learning project: starting with a 50k-story debug run, then scaling data and model size incrementally (500k → 1M stories) while validating the pipeline at each stage before committing to longer training runs.

## License

MIT