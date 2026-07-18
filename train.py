"""
Train MiniLLM from scratch on the TinyStories dataset.

Usage:
    python train.py
"""

import torch
import torch.nn.functional as F
import math
import time
import json
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

from model import MiniLLM

# ============================================================
# CONFIG
# ============================================================
TRAIN_STORIES = 1_000_000
VOCAB_SIZE = 16000

config = {
    "vocab_size": VOCAB_SIZE,
    "d_model": 384,
    "n_layers": 8,
    "n_heads": 8,
    "n_kv_heads": 2,
    "ffn_hidden_dim": 1024,
    "max_seq_len": 256,
    "dropout": 0.1,
}

BATCH_SIZE = 64
CONTEXT_LEN = 256
LEARNING_RATE = 3e-4
MAX_STEPS = 13000
EVAL_INTERVAL = 500
EVAL_STEPS = 50
LOG_INTERVAL = 200

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# ============================================================
# DATA
# ============================================================
print("Loading dataset...")
dataset = load_dataset("roneneldan/TinyStories")
train_data = dataset["train"].select(range(TRAIN_STORIES))
val_data = dataset["validation"]

print("Training tokenizer...")
tokenizer = Tokenizer(BPE(unk_token="<unk>"))
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
tokenizer.decoder = ByteLevelDecoder()

trainer = BpeTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=["<unk>", "<pad>", "<bos>", "<eos>"],
    min_frequency=2,
    initial_alphabet=ByteLevel.alphabet(),
)

def text_iterator():
    for row in train_data:
        yield row["text"]

tokenizer.train_from_iterator(text_iterator(), trainer=trainer)
tokenizer.save("weights/tokenizer.json")

BOS_ID = tokenizer.token_to_id("<bos>")
EOS_ID = tokenizer.token_to_id("<eos>")


def encode(text):
    ids = tokenizer.encode(text).ids
    return [BOS_ID] + ids + [EOS_ID]


def tokenize_dataset(hf_dataset, batch_size=1000):
    all_ids = []
    texts = hf_dataset["text"]
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        encodings = tokenizer.encode_batch(batch_texts)
        for enc in encodings:
            all_ids.append(BOS_ID)
            all_ids.extend(enc.ids)
            all_ids.append(EOS_ID)
        if i % 50000 == 0:
            print(f"  Tokenized {i}/{len(texts)} stories...")
    return torch.tensor(all_ids, dtype=torch.long)


print("Tokenizing train split...")
train_ids = tokenize_dataset(train_data)
print("Tokenizing validation split...")
val_ids = tokenize_dataset(val_data)
print(f"Train tokens: {train_ids.shape[0]:,} | Validation tokens: {val_ids.shape[0]:,}")


def get_batch(split, batch_size, context_len):
    data = train_ids if split == "train" else val_ids
    ix = torch.randint(0, len(data) - context_len - 1, (batch_size,))
    x = torch.stack([data[i : i + context_len] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + context_len] for i in ix])
    return x.to(device), y.to(device)


# ============================================================
# MODEL + TRAINING
# ============================================================
model = MiniLLM(config).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)


@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = []
        for _ in range(EVAL_STEPS):
            xb, yb = get_batch(split, BATCH_SIZE, CONTEXT_LEN)
            _, loss = model(xb, yb)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


print("Starting training...")
train_losses, val_losses, step_log = [], [], []
start_time = time.time()

model.train()
for step in range(MAX_STEPS):
    xb, yb = get_batch("train", BATCH_SIZE, CONTEXT_LEN)
    logits, loss = model(xb, yb)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    if step % LOG_INTERVAL == 0:
        elapsed = time.time() - start_time
        print(f"  Step {step:6d}/{MAX_STEPS} | Loss: {loss.item():.4f} | Time: {elapsed:.0f}s")

    if step % EVAL_INTERVAL == 0 or step == MAX_STEPS - 1:
        losses = estimate_loss()
        train_losses.append(losses["train"])
        val_losses.append(losses["val"])
        step_log.append(step)
        print(f"  >>> Eval @ step {step}: train={losses['train']:.4f}, val={losses['val']:.4f}")

total_time = time.time() - start_time
print(f"Training complete! Total time: {total_time/60:.1f} min")
print(f"Final train loss: {train_losses[-1]:.4f} | Final val loss: {val_losses[-1]:.4f}")

# Save final model
torch.save({
    "step": MAX_STEPS,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "config": config,
    "train_losses": train_losses,
    "val_losses": val_losses,
    "step_log": step_log,
}, "weights/minillm_final.pt")
print("Saved to weights/minillm_final.pt")