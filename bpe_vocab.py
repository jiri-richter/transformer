import math
import re                              # regex for whitespace pre-tokenization and word splitting.
from collections import Counter
import torch                           # tensor data structures, automatic differentiation (autograd), and GPU acceleration primitives.
import torch.nn as nn                  # PyTorch's neural network module containing base classes and layers.
import torch.nn.functional as F        # PyTorch's functional API (F.softmax, F.gelu, F.cross_entropy).


if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# Dimenze pro model
# Dimensionality of the model's hidden representation (the vector size for each token throughout the transformer architecture).
D_MODEL = 256

# Dimensionality of the inner hidden layer of the Position-wise Feed-Forward Network (typically 4x D_MODEL).
D_FF = 4 * D_MODEL

# Struktura modelu
# Number of parallel attention heads in multi-head attention.
N_HEADS = 8

# Number of stacked Transformer decoder blocks.
N_LAYERS = 6

# Rozsah vstupu/výstupu - BPE vocabulary
# Sequence length of training context windows (number of tokens in input sequence during training).
SEQ_LEN = 128

# Maximum sequence length supported by the positional embedding matrix and causal mask.
MAX_LEN = 256

# Reduced BPE vocab size (1024 instead of 2048 so every subword merge is well-sampled)
BPE_VOCAB_SIZE = 1024

# Lecture milestone checkpoints (Underfitted -> Sweet Spot -> Overfitted)
CHECKPOINT_STEPS = {
    200:  "ckpt_1_underfit.pt",    # Stage 1: Czech syllables/diacritics, broken words
    1450: "ckpt_2_sweetspot.pt",   # Stage 2: Generalization sweet spot (~loss 1.5 - 2.2)
    5000: "ckpt_3_overfit.pt",     # Stage 3: Memorized training corpus (~loss < 0.3)
}


class PositionalEmbedding(nn.Module):                                             # Defines a custom PyTorch module inheriting from nn.Module.
    def __init__(self, max_len, d_model):                                         # Constructor initializing the fixed sinusoidal positional encoding matrix.
        super().__init__()                                                        # Calls the superclass initializer to register module hooks and buffers.
        assert d_model % 2 == 0, "d_model must be even for sinusoidal PE."        # Ensures an even embedding dimension so sine and cosine pairs align cleanly.

        pe = torch.zeros(max_len, d_model)                                        # Initializes a 2D tensor of shape (max_len, d_model) to hold positional encodings.
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)       # Creates position indices [0, 1, ..., max_len-1] of shape (max_len, 1).
        div_term = torch.exp(                                                     # Computes the frequency divisor 1 / (10000^(2i / d_model)) in log-space
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)  # for numerical stability, yielding a 1D tensor of shape (d_model / 2,).
        )

        pe[:, 0::2] = torch.sin(position * div_term)                              # Applies sine to even embedding indices (2i) across all positions.
        pe[:, 1::2] = torch.cos(position * div_term)                              # Applies cosine to odd embedding indices (2i + 1) across all positions.
        pe = pe.unsqueeze(0)                                                      # Adds a leading batch dimension to get shape (1, max_len, d_model) for broadcasting.

        self.register_buffer("pe", pe)                                            # Registers 'pe' as a non-learnable buffer (excluded from model.parameters(), but moves with .to(device)
                                                                                  # and saves in state_dict).

    def forward(self, x):                                                         # Defines the forward execution path receiving an input tensor x of shape (B, T).
        return self.pe[:, :x.size(1), :]                                          # Slices the precomputed encodings along the sequence dimension to length T = x.size(1),
                                                                                  # returning shape (1, T, d_model).


class MultiHeadAttention(nn.Module):                                                     # Custom class for multi-head self-attention
    def __init__(self, d_model, n_heads):                                                # Constructor accepting total embedding dimension d_model and number of heads n_heads.
        super().__init__()                                                               # Initializes base module infrastructure.
        assert d_model % n_heads == 0                                                    # Enforces that d_model is evenly divisible by n_heads, ensuring each head receives an equal subspace dimension d_head.
        self.d_head, self.n_heads = d_model // n_heads, n_heads                          # Computes dimension per head (d_head = d_model // n_heads) and stores head count.

        self.W_q = nn.Linear(d_model, d_model)                                           # Linear projection layer for Queries.
        self.W_k = nn.Linear(d_model, d_model)                                           # Linear projection layer for Keys.
        self.W_v = nn.Linear(d_model, d_model)                                           # Linear projection layer for Values.
        self.W_o = nn.Linear(d_model, d_model)                                           # Final linear projection layer (out projection).

    def forward(self, x, mask=None):                                                     # Receives input x of shape (B, T, d_model) and optional binary mask.
        B, T, _ = x.size()                                                               # Unpacks batch size B, sequence length T, and ignores hidden dimension size.

        q = self.W_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)            # self.W_q(x) projects (B, T, d_model) to (B, T, d_model).
                                                                                         # .view(B, T, n_heads, d_head) splits hidden dimension into n_heads heads of size d_head.
                                                                                         # .transpose(1, 2) swaps sequence length T and head count, resulting in shape (B, n_heads, T, d_head).
        k = self.W_k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)            # same as q
        v = self.W_v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)            # same as q

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)  # (B,heads,T,T)     # k.transpose(-2, -1) transposes the last two dimensions of k from (B, n_heads, T, d_head) to (B, n_heads, d_head, T).
                                                                                         # q @ k.transpose(...) computes batched matrix multiplication of (B, n_heads, T, d_head) by (B, n_heads, d_head, T), yielding raw attention logits of shape (B, n_heads, T, T).
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))                        # Replaces all elements in scores where mask == 0 (or False) with negative infinity (-inf).
                                                                                         # When passed through softmax, $e^{-\infty} = 0$, completely preventing future tokens from influencing past tokens.

        attn = F.softmax(scores, dim=-1)                                                 # Applies softmax across the last dimension (dim=-1), converting raw attention scores to probability distributions
                                                                                         # where each row sums to 1.0. Shape remains (B, n_heads, T, T).

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, -1)                     # Multiplies attention weights by values, restores (B, T, d_model) shape.

        return self.W_o(out)                                                             # Applies linear out-projection layer (B, T, d_model) -> (B, T, d_model) to mix information across attention heads.


class FeedForward(nn.Module):                                                                         # Defines the position-wise feed-forward block.

    def __init__(self, d_model, d_ff):                                                                # Accepts model dimension d_model and feed-forward inner dimension d_ff.
        super().__init__()                                                                            # Initializes parent class.

        self.net = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))       # Expands dimension from d_model to d_ff, applies GELU, projects back down to d_model.

    def forward(self, x):                                                                             # Pass input (B, T, d_model) through self.net.
        return self.net(x)                                                                            # Executes sequential transformations and returns output of shape (B, T, d_model).


class TransformerBlock(nn.Module):                                                                    # Defines single Transformer layer block combining self-attention and feed-forward layers.

    def __init__(self, d_model, n_heads, d_ff):                                                       # Constructor initializing sub-modules.
        super().__init__()                                                                            # Calls superclass initializer.

        self.attn, self.ffn = MultiHeadAttention(d_model, n_heads), FeedForward(d_model, d_ff)        # Instantiates multi-head attention module and feed-forward network.
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)                             # Instantiates two Layer Normalization modules operating over final dimension d_model to stabilize training gradients.

    def forward(self, x, mask):                                                                       # Takes input tensor x (B, T, d_model) and causal mask mask.
        x = self.ln1(x + self.attn(x, mask))   # residual + attention                                 # Computes self-attention sub-layer output, adds original input x (residual connection), and applies Layer Normalization.
        x = self.ln2(x + self.ffn(x))          # residual + FFN                                       # Computes feed-forward network output, adds residual connection, and applies second Layer Normalization.

        return x                                                                                      # Returns transformed tensor (B, T, d_model)


class MiniGPT(nn.Module):                                                                                             # Defines overall autoregressive language model class.

    def __init__(self, vocab_size=None, d_model=None, n_heads=None, n_layers=None, d_ff=None, max_len=None):
        super().__init__()
        self.vocab_size = vocab_size if vocab_size is not None else VOCAB
        self.d_model = d_model if d_model is not None else D_MODEL
        self.n_heads = n_heads if n_heads is not None else N_HEADS
        self.n_layers = n_layers if n_layers is not None else N_LAYERS
        self.d_ff = d_ff if d_ff is not None else D_FF
        self.max_len = max_len if max_len is not None else MAX_LEN

        self.tok_emb = nn.Embedding(self.vocab_size, self.d_model)                                                   # Token embedding lookup table mapping discrete token IDs to d_model-dimensional dense vectors.
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.d_model ** -0.5)                                     # Renormalization of variance of embedding initialization so the variance is around 1.

        self.pos_emb = PositionalEmbedding(self.max_len, self.d_model)                                                # Instance of PositionalEmbedding providing positional encoding vectors up to max_len.
        self.blocks = nn.ModuleList([
            TransformerBlock(self.d_model, self.n_heads, self.d_ff) for _ in range(self.n_layers)
        ])                                                                                                            # Container module holding list of n_layers Transformer blocks.
        self.head = nn.Linear(self.d_model, self.vocab_size, bias=False)                                              # Final language modeling linear layer mapping hidden state (B, T, d_model) to logits (B, T, vocab_size).
        self.head.weight = self.tok_emb.weight                                                                        # Ties input & output embeddings.

        self.register_buffer("mask", torch.tril(torch.ones(self.max_len, self.max_len)).bool())                       # Lower-triangular causal mask buffer of shape (max_len, max_len).

    def forward(self, idx, targets=None):                                                                             # Receives input token IDs tensor idx of shape (B, T) and optional target token IDs targets of shape (B, T).
        B, T = idx.size()                                                                                             # Extracts batch size B and sequence length T.
        x = (self.tok_emb(idx) * math.sqrt(self.d_model)) + self.pos_emb(idx)                                         # Combines scaled token embeddings and positional encodings.

        mask = self.mask[:T, :T]

        for block in self.blocks:                                                                                     # Iterates through each Transformer block in nn.ModuleList.
            x = block(x, mask)                                                                                        # Sequentially updates token hidden states x through Transformer blocks.

        logits = self.head(x)                                                                                         # Projects final hidden states (B, T, d_model) to vocabulary space (B, T, vocab_size).

        loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1)) if targets is not None else None

        return logits, loss                                                                                           # Returns predicted logits tensor (B, T, vocab_size) and scalar loss tensor (or None).


# BPE tokenizer with whitespace pre-tokenization
class BPE:
    def __init__(self, vocab_size=1024):
        self.vocab_size = vocab_size
        self.merges = {}       # (a, b) -> new_id
        self.vocab = {}        # id -> bytes (human-readable token)
        self.inv_vocab = {}    # bytes -> id

    def _pretokenize(self, text):
        # Keep leading whitespace attached to each word; never merge across word boundaries
        return [list(chunk.encode("utf-8")) for chunk in re.findall(r"\s*\S+|\s+", text)]

    def train(self, text):
        # Start with 256 byte tokens
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.inv_vocab = {bytes([i]): i for i in range(256)}

        chunks = self._pretokenize(text)

        for new_id in range(256, self.vocab_size):
            pairs = Counter()
            for chunk in chunks:
                pairs.update(zip(chunk, chunk[1:]))
            if not pairs:
                break

            best = max(pairs, key=pairs.get)
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            self.inv_vocab[self.vocab[new_id]] = new_id

            # Apply merge inside each word chunk independently
            new_chunks = []
            for chunk in chunks:
                merged = []
                i = 0
                while i < len(chunk):
                    if i < len(chunk) - 1 and (chunk[i], chunk[i + 1]) == best:
                        merged.append(new_id)
                        i += 2
                    else:
                        merged.append(chunk[i])
                        i += 1
                new_chunks.append(merged)
            chunks = new_chunks

    def encode(self, text):
        chunks = self._pretokenize(text)
        # Apply merges in order of priority inside each word chunk
        for (a, b), new_id in sorted(self.merges.items(), key=lambda x: x[1]):
            new_chunks = []
            for chunk in chunks:
                merged = []
                i = 0
                while i < len(chunk):
                    if i < len(chunk) - 1 and chunk[i] == a and chunk[i + 1] == b:
                        merged.append(new_id)
                        i += 2
                    else:
                        merged.append(chunk[i])
                        i += 1
                new_chunks.append(merged)
            chunks = new_chunks
        return [tok for chunk in chunks for tok in chunk]

    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def save(self, path):
        torch.save({
            "vocab_size": self.vocab_size,
            "merges": self.merges,
            "vocab": self.vocab,
        }, path)

    @classmethod
    def load(cls, path):
        state = torch.load(path, weights_only=True)
        bpe = cls(vocab_size=state["vocab_size"])
        bpe.merges = state["merges"]
        bpe.vocab = state["vocab"]
        bpe.inv_vocab = {v: k for k, v in bpe.vocab.items()}
        return bpe


# BPE slovník & dataset preparation
torch.manual_seed(0)

with open("corpus_words.txt", "r", encoding="utf-8") as f:
    corpus = f.read() + " "

corpus = corpus.lower()

bpe = BPE(vocab_size=BPE_VOCAB_SIZE)
bpe.train(corpus)
bpe.save("bpe_tokenizer_1024.pt")

VOCAB = len(bpe.vocab)
data = torch.tensor(bpe.encode(corpus), dtype=torch.long, device=device)
print(f"VOCAB: {VOCAB} | Total BPE tokens: {len(data):,}")

# Verify consistent subword splits regardless of surrounding whitespace
bpe = BPE.load("bpe_tokenizer_1024.pt")
for sample in ["šílenství", " šílenství", "šílenství ", " šílenství "]:
    ids = bpe.encode(sample)
    pieces = [bpe.vocab[i].decode("utf-8", errors="replace") for i in ids]
    print(f"{repr(sample):<15} -> {ids} -> {pieces}")


# Checkpoint save/load helpers
def save_checkpoint(path, model, opt, bpe, step, loss_val):
    torch.save({
        "config": {
            "vocab_size": len(bpe.vocab),
            "d_model": D_MODEL,
            "n_heads": N_HEADS,
            "n_layers": N_LAYERS,
            "d_ff": D_FF,
            "max_len": MAX_LEN,
        },
        "seq_len": SEQ_LEN,
        "step": step,
        "loss": loss_val,
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "bpe": {
            "vocab_size": bpe.vocab_size,
            "merges": bpe.merges,
            "vocab": bpe.vocab,
        },
    }, path)
    print(f"  -> [Checkpoint saved: {path}] step={step}, loss={loss_val:.4f}")


def load_checkpoint(path, load_optimizer=False):
    checkpoint = torch.load(path, map_location=device, weights_only=True)

    loaded_bpe = BPE(vocab_size=checkpoint["bpe"]["vocab_size"])
    loaded_bpe.merges = checkpoint["bpe"]["merges"]
    loaded_bpe.vocab = checkpoint["bpe"]["vocab"]
    loaded_bpe.inv_vocab = {v: k for k, v in loaded_bpe.vocab.items()}

    loaded_model = MiniGPT(**checkpoint["config"]).to(device)
    loaded_model.load_state_dict(checkpoint["model"])
    loaded_model.eval()

    loaded_opt = None
    if load_optimizer and "optimizer" in checkpoint:
        loaded_opt = torch.optim.AdamW(loaded_model.parameters(), lr=3e-4)
        loaded_opt.load_state_dict(checkpoint["optimizer"])

    return loaded_model, loaded_bpe, loaded_opt, checkpoint


# Training loop with milestone checkpoints (underfit -> sweet spot -> overfit)
torch.manual_seed(42)

model = MiniGPT().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)


def get_batch():
    BATCH = 64
    B, T = BATCH, SEQ_LEN
    ix = torch.randint(len(data) - T, (B,), device=device)
    idx = ix.unsqueeze(1) + torch.arange(T, device=device)
    return data[idx], data[idx + 1]


epochs = max(CHECKPOINT_STEPS) + 1
for step in range(epochs):
    x, y = get_batch()
    _, loss = model(x, y)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

    if step % 100 == 0:
        print(f"step {step:4d}  loss {loss.item():.4f}")

    if step in CHECKPOINT_STEPS:
        save_checkpoint(CHECKPOINT_STEPS[step], model, opt, bpe, step, loss.item())


# Inference & Logit Lens
@torch.no_grad()
def logit_lens(model, bpe, prompt, seq_len=SEQ_LEN):
    model.eval()
    norm_prompt = " " + prompt.lower().lstrip()
    ids = torch.tensor([bpe.encode(norm_prompt)], device=device)[:, -seq_len:]
    B, T = ids.size()

    x = (model.tok_emb(ids) * math.sqrt(model.d_model)) + model.pos_emb(ids)
    mask = model.mask[:T, :T]

    stages = [("Emb", x)]
    for i, block in enumerate(model.blocks):
        x = block(x, mask)
        stages.append((f"Block {i}", x))

    print(f"Prompt: {repr(prompt)} -> Predicting next token:")
    for name, hidden in stages:
        probs = F.softmax(model.head(hidden)[0, -1], dim=-1)
        top_p_vals, top_id = torch.topk(probs, k=3)
        preds = [
            f"{repr(bpe.vocab[idx.item()].decode('utf-8', errors='replace'))} ({p.item():.0%})"
            for idx, p in zip(top_id, top_p_vals)
        ]
        print(f"  {name:<8}: {', '.join(preds)}")


@torch.no_grad()
def generate(model, bpe, prompt, max_new=150, seq_len=SEQ_LEN, temperature=0.7, top_k=15, top_p=0.85, rep_penalty=1.15):
    model.eval()
    norm_prompt = " " + prompt.lower().lstrip()
    ids = torch.tensor([bpe.encode(norm_prompt)], device=device)

    for _ in range(max_new):
        idx_cond = ids[:, -seq_len:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature

        # Penalize recently generated tokens to prevent subword echo loops
        for tok_id in set(ids[0, -32:].tolist()):
            if logits[0, tok_id] > 0:
                logits[0, tok_id] /= rep_penalty
            else:
                logits[0, tok_id] *= rep_penalty

        # Top-k filter
        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")

        # Top-p (nucleus) filter
        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = -float("inf")

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, next_token], dim=1)

    return bpe.decode(ids[0].tolist()).lstrip()


def run_lecture_demo(prompt="Třetí den ráno byla půda natolik suchá,", max_new=120):
    stages = [
        ("1. UNDERFITTED", "ckpt_1_underfit.pt"),
        ("2. SWEET SPOT",  "ckpt_2_sweetspot.pt"),
        ("3. OVERFITTED",  "ckpt_3_overfit.pt"),
    ]

    for label, path in stages:
        ckpt_model, ckpt_bpe, _, ckpt = load_checkpoint(path)
        ckpt_seq_len = ckpt["seq_len"]

        print("=" * 78)
        print(f"{label} (Step {ckpt['step']}) | Step: {ckpt['step']} | Train Loss: {ckpt['loss']:.4f}")
        print("=" * 78)
        print("Generated text:\n", generate(ckpt_model, ckpt_bpe, prompt, max_new=max_new, seq_len=ckpt_seq_len))
        print("-" * 78)
        logit_lens(ckpt_model, ckpt_bpe, prompt, seq_len=ckpt_seq_len)
        print()


# Run live during the lecture (loads all 3 checkpoints from disk in milliseconds)
run_lecture_demo(prompt="Třetí den ráno byla půda natolik suchá,", max_new=120)
run_lecture_demo(prompt="hluboko pod ledem jsme spatřili podivné", max_new=120)
