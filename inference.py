"""
Run text generation with a trained MiniLLM model.

Usage:
    python inference.py --prompt "Once upon a time" --temperature 0.8 --max_new_tokens 150
"""

import argparse
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from model import MiniLLM

DEFAULT_MODEL_PATH = "weights/minillm_final.pt"
DEFAULT_TOKENIZER_PATH = "weights/tokenizer.json"


def load_model_and_tokenizer(model_path, tokenizer_path, device):
    tokenizer = Tokenizer.from_file(tokenizer_path)

    checkpoint = torch.load(model_path, map_location=device)
    model = MiniLLM(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, tokenizer, checkpoint


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=100, temperature=0.8, device="cpu"):
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")

    ids = [bos_id] + tokenizer.encode(prompt).ids
    tokens = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        context = tokens[:, -model.config["max_seq_len"]:]
        logits, _ = model(context)
        last_logits = logits[:, -1, :]
        probs = F.softmax(last_logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat([tokens, next_token], dim=1)
        if next_token.item() == eos_id:
            break

    output_ids = tokens[0].tolist()
    return tokenizer.decode(output_ids, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Generate text with MiniLLM")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt to start generation")
    parser.add_argument("--max_new_tokens", type=int, default=150, help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (higher = more random)")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help="Path to model checkpoint (.pt)")
    parser.add_argument("--tokenizer_path", type=str, default=DEFAULT_TOKENIZER_PATH, help="Path to tokenizer.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model, tokenizer, checkpoint = load_model_and_tokenizer(args.model_path, args.tokenizer_path, device)
    print(f"Loaded model (trained {checkpoint['step']} steps, val loss {checkpoint['val_losses'][-1]:.4f})\n")

    output = generate(
        model, tokenizer, args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        device=device,
    )

    print(f"Prompt: {args.prompt}")
    print("-" * 60)
    print(output)


if __name__ == "__main__":
    main()