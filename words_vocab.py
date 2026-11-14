import math
import re                              # regex for splitting corpus text into words.
import torch                           # tensor data structures, automatic differentiation (autograd), and GPU acceleration primitives.
import torch.nn as nn                  # PyTorch's neural network module containing base classes like nn.Module, parameter wrappers nn.Parameter, layer types (nn.Linear, nn.Embedding, nn.LayerNorm), and container modules (nn.Sequential, nn.ModuleList).
import torch.nn.functional as F        # PyTorch's functional API containing stateless functions like activation functions (F.softmax, F.gelu) and loss functions (F.cross_entropy).


if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

# Rozsah vstupu/výstupu - word-level vocabulary
# Sequence length of training context windows (number of tokens in input sequence during training).
SEQ_LEN = 32

# Maximum sequence length supported by the positional embedding matrix and causal mask.
MAX_LEN = 64


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

        self.W_q = nn.Linear(d_model, d_model)                                           # Linear projection layer for Queries, projecting input (B, T, d_model) to Query representation (B, T, d_model).
        self.W_k = nn.Linear(d_model, d_model)                                           # Linear projection layer for Keys, projecting input (B, T, d_model) to Key representation (B, T, d_model).
        self.W_v = nn.Linear(d_model, d_model)                                           # Linear projection layer for Values, projecting input (B, T, d_model) to Value representation (B, T, d_model).
        self.W_o = nn.Linear(d_model, d_model)                                           # Final linear projection layer (out projection) to combine multi-head outputs back into unified hidden dimension d_model.

    def forward(self, x, mask=None):                                                     # Receives input x of shape (B, T, d_model) and optional binary mask.
        B, T, _ = x.size()                                                               # Unpacks batch size B, sequence length T, and ignores hidden dimension size.
        # (B,T,d_model) -> (B, n_heads, T, d_head)

        q = self.W_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)            # self.W_q(x) projects (B, T, d_model) to (B, T, d_model).
                                                                                         # .view(B, T, n_heads, d_head) splits hidden dimension into n_heads heads of size d_head.
                                                                                         # .transpose(1, 2) swaps sequence length T and head count, resulting in shape (B, n_heads, T, d_head).
        k = self.W_k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)            # same as q
        v = self.W_v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)            # same as q

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)  # (B,heads,T,T)     # k.transpose(-2, -1) transposes the last two dimensions of k from (B, n_heads, T, d_head) to (B, n_heads, d_head, T).
                                                                                         # q @ k.transpose(...) computes batched matrix multiplication of (B, n_heads, T, d_head) by (B, n_heads, d_head, T), yielding raw attention logits of shape (B, n_heads, T, T).
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))                        # scores = scores.masked_fill(mask == 0, float('-inf')): Replaces all elements in scores where mask == 0 (or False) with negative infinity (-inf).
                                                                                         # When passed through softmax, $e^{-\infty} = 0$, completely preventing future tokens from influencing past tokens.

        attn = F.softmax(scores, dim=-1)                                                 # attn = F.softmax(scores, dim=-1): Applies softmax across the last dimension (dim=-1), converting raw attention scores to probability distributions
                                                                                         # where each row sums to $1.0$. Shape remains (B, n_heads, T, T).

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, -1)                     # attn @ v multiplies attention weights (B, n_heads, T, T) by values (B, n_heads, T, d_head) to produce weighted context vectors (B, n_heads, T, d_head).
                                                                                         # .transpose(1, 2) swaps head dimension back to (B, T, n_heads, d_head).
                                                                                         # .contiguous() rearranges memory layout to be contiguous in memory, enabling .view().
                                                                                         # .view(B, T, -1) flattens the last two dimensions (d_head x n_heads = d_model) back into hidden dimension d_model, yielding shape (B, T, d_model).

        return self.W_o(out)                                                             # Applies linear out-projection layer (B, T, d_model) -> (B, T, d_model) to mix information across attention heads.


class FeedForward(nn.Module):                                                                         # Defines the position-wise feed-forward block.

    def __init__(self, d_model, d_ff):                                                                # Accepts model dimension d_model and feed-forward inner dimension d_ff.
        super().__init__()                                                                            # Initializes parent class.

        self.net = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))       # nn.Sequential(...): Constructs a sequential pipeline of three operations:
                                                                                                      # nn.Linear(d_model, d_ff): Expands dimension from d_model to d_ff.
                                                                                                      # nn.GELU(): Gaussian Error Linear Unit activation function introducing non-linearity.
                                                                                                      # nn.Linear(d_ff, d_model): Projects high-dimensional feature representations back down to original dimension d_model.

    def forward(self, x):                                                                             # Pass input (B, T, d_model) through self.net.
        return self.net(x)                                                                            # Executes sequential transformations and returns output of shape (B, T, d_model).


class TransformerBlock(nn.Module):                                                                    # Defines single Transformer layer block combining self-attention and feed-forward layers.

    def __init__(self, d_model, n_heads, d_ff):                                                       # Constructor initializing sub-modules.

        super().__init__()                                                                            # Calls superclass initializer.

        self.attn, self.ffn = MultiHeadAttention(d_model, n_heads), FeedForward(d_model, d_ff)        # Instantiates multi-head attention module and feed-forward network.
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)                             # Instantiates two Layer Normalization modules operating over final dimension d_model to stabilize training gradients.

    def forward(self, x, mask):                                                                       # Takes input tensor x (B, T, d_model) and causal mask mask.
        x = self.ln1(x + self.attn(x, mask))   # residual + attention                                 # Computes self-attention sub-layer output, adds original input x (residual connection / skip connection), and applies Layer Normalization.
        x = self.ln2(x + self.ffn(x))          # residual + FFN                                       # Computes feed-forward network output, adds residual connection, and applies second Layer Normalization.

        return x                                                                                      # Returns transformed tensor (B, T, d_model)


class MiniGPT(nn.Module):                                                                                             # Defines overall autoregressive language model class.

    def __init__(self):                                                                                               # Constructor initializing sub-modules.

        super().__init__()                                                                                            # Calls superclass initializer.

        self.tok_emb = nn.Embedding(VOCAB, D_MODEL)                                                                   # Token embedding lookup table mapping discrete token IDs to D_MODEL-dimensional dense vectors.
                                                                                                                      # Weight shape is (VOCAB, d_model).

        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=D_MODEL ** -0.5)                                           # Renormalization of variance of embedding initialization so the variance is around 1.
                                                                                                                      # We do this so that the positional embedding does not dominate the input of the model.

        self.pos_emb = PositionalEmbedding(MAX_LEN, D_MODEL)                                                          # Instance of PositionalEmbedding providing positional encoding vectors up to MAX_LEN.

        self.blocks = nn.ModuleList([TransformerBlock(D_MODEL, N_HEADS, D_FF) for _ in range(N_LAYERS)])              # Container module holding list of N_LAYERS Transformer blocks. Using nn.ModuleList ensures PyTorch
                                                                                                                      # registers sub-module parameters correctly.
        self.head = nn.Linear(D_MODEL, VOCAB, bias=False)
        self.head.weight = self.tok_emb.weight                                                                        # Optional: ties input & output embeddings as in the paper.

        self.register_buffer("mask", torch.tril(torch.ones(MAX_LEN, MAX_LEN)).bool())                                 # Generates lower-triangular matrix of shape (MAX_LEN, MAX_LEN) filled with True on and below main diagonal, and False above it.

    def forward(self, idx, targets=None):                                                                             # Receives input token IDs tensor idx of shape (B, T) and optional target token IDs targets of shape (B, T).
        B, T = idx.size()                                                                                             # Extracts batch size B and sequence length T.

        x = (self.tok_emb(idx) * math.sqrt(D_MODEL)) + self.pos_emb(idx)                                              # self.tok_emb(idx) retrieves token embeddings of shape (B, T, d_model).
                                                                                                                      # self.pos_emb(idx) retrieves position embeddings of shape (1, T, d_model).
                                                                                                                      # Sum adds them elementwise (positional embedding broadcasts across batch dimension B) yielding combined embedding (B, T, d_model).

        mask = self.mask[:T, :T]

        for block in self.blocks:                                                                                     # Iterates through each Transformer block in nn.ModuleList.
            x = block(x, mask)                                                                                        # Sequentially updates token hidden states x through Transformer blocks.

        logits = self.head(x)                                                                                         # Projects final hidden states (B, T, d_model) to vocabulary space (B, T, VOCAB).

        loss = F.cross_entropy(logits.reshape(-1, VOCAB), targets.reshape(-1)) if targets is not None else None       # logits.reshape(-1, VOCAB) flattens batch and sequence dimensions into single batch dimension (B*T, VOCAB).
                                                                                                                      # targets.reshape(-1) flattens targets into (B*T,).
                                                                                                                      # F.cross_entropy computes negative log-likelihood loss over predicted word distributions against ground-truth target
                                                                                                                      # tokens. If targets is None (inference generation mode), loss evaluates to None.

        return logits, loss                                                                                           # Returns predicted logits tensor (B, T, VOCAB) and scalar loss tensor (or None).


# word-level slovník
def make_dataset():                                                             # Utility function creating word-level training dataset.

    with open("corpus_words.txt", "r", encoding="utf-8") as f:
        corpus = f.read()

    tokens = re.findall(r"\w+|[,.]", corpus.lower())
    vocab = sorted(set(tokens))
    vocab = ["<unk>", " "] + vocab

    global VOCAB
    VOCAB = len(vocab)

    stoi = {w: i for i, w in enumerate(vocab)}
    itos = {i: w for i, w in enumerate(vocab)}
    UNK = stoi["<unk>"]

    return torch.tensor([stoi[t] for t in tokens]), stoi, itos, UNK


# kontrola, jak se inicializoval slovník
data, stoi, itos, UNK = make_dataset()
print(f"VOCAB size: {VOCAB} | Sample tokens:", list(stoi.items())[:10])


torch.manual_seed(42)                                                           # Sets pseudo-random generator seed for PyTorch to ensure reproducible weight initialization and batch sampling.

data, stoi, itos, UNK = make_dataset()

model = MiniGPT().to(device)

opt = torch.optim.AdamW(model.parameters(), lr=3e-4)                            # Instantiates AdamW optimizer (Adam with decoupled weight decay) configured with learning rate 3e-4.


def get_batch():                                                                # Helper function sampling random context windows from dataset tensor.
    BATCH = 64

    # To ensure the model can generate tokens beyond SEQ_LEN, we set SEQ_LEN = MAX_LEN to allow for maximum contextualization
    # More robust than sliding context window, but takes more time to train the model
    # SEQ_LEN = MAX_LEN

    ix = torch.randint(len(data) - SEQ_LEN, (BATCH,))                           # Generates BATCH random starting index integers between 0 and len(data) - SEQ_LEN.

    x = torch.stack([data[i:i+SEQ_LEN] for i in ix])                            # Extracts input sequences of length SEQ_LEN starting at index i for each index in ix, stacking them into tensor (BATCH, SEQ_LEN).

    y = torch.stack([data[i+1:i+SEQ_LEN+1] for i in ix])                        # Extracts target sequences shifted by 1 token index to the right (i+1), enabling autoregressive next-token prediction supervision.

    text_input = [" ".join(itos[tok.item()] for tok in row) for row in x]

    return x, y, text_input                                                     # Returns input batch x, target batch y, and decoded text_input.


# # training loop (15 000 epochs pre-trained and saved into words_vocab.pt)
# epochs = 15000
# for step in range(epochs):
#
#     # if step == 0:
#     #     x, y, text_input = get_batch()
#     # else:
#     #     x, y, _ = get_batch()
#     x, y, _ = get_batch()
#
#     x = x.to(device, non_blocking=True)
#     y = y.to(device, non_blocking=True)
#
#     _, loss = model(x, y)
#
#     opt.zero_grad(); loss.backward()
#
#     torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#
#     opt.step()
#
#     if step == 0:
#         # print("Text input (for training, first batch):\n")
#         # print(text_input[0])
#
#         print("Model input (first batch):\n")
#         print(x[0])
#
#         print("First batch of tokens:\n")
#         print(model.tok_emb(x)[0])
#
#         print("Positional encoding:\n")
#         print(model.pos_emb(x)[0])
#
#         print("Sum of token embeddings and positional encoding:\n")
#         print((model.tok_emb(x) + model.pos_emb(x))[0])
#
#     if step % 100 == 0:
#         print(f"step {step:3d}  loss {loss.item():.4f}")
#
# # saving a model
# torch.save({
#     "step": step,
#     "model": model.state_dict(),
#     "optimizer": opt.state_dict(),
#     "loss": loss.item(),
# }, "words_vocab.pt")


# loading a saved model (for word-level vocabulary)
model = MiniGPT().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

checkpoint = torch.load("words_vocab.pt", map_location=device, weights_only=True)
model.load_state_dict(checkpoint["model"])
opt.load_state_dict(checkpoint["optimizer"])
start_step = checkpoint["step"] + 1
last_loss = checkpoint["loss"]

print(f"Loaded checkpoint from step {checkpoint['step']} (loss: {last_loss:.4f})")


# generation loop
@torch.no_grad()                                                                # Decorator disabling autograd computation graph construction during execution, saving memory and accelerating inference.
def logit_lens(model, prompt):                                                  # Pomocná funkce pro vypsání pravděpodobností dalších tokenů (pro začátek generace)
    ids = torch.tensor([[stoi.get(t, UNK) for t in re.findall(r"\w+|[,.]", prompt.lower())]], device=device)
    B, T = ids.size()
    x = (model.tok_emb(ids) * math.sqrt(D_MODEL)) + model.pos_emb(ids)
    mask = model.mask[:T, :T]

    stages = [("Emb", x)]
    for i, block in enumerate(model.blocks):
        x = block(x, mask)
        stages.append((f"Block {i}", x))

    print(f"Prompt: {repr(prompt)} -> Predicting next token:")
    for name, hidden in stages:
        probs = F.softmax(model.head(hidden)[0, -1], dim=-1)
        top_p, top_id = torch.topk(probs, k=3)
        preds = [f"'{itos[idx.item()]}' ({p.item():.0%})" for idx, p in zip(top_id, top_p)]
        print(f"  {name:<8}: {', '.join(preds)}")


@torch.no_grad()
def generate(prompt, max_new=20):                                               # Inference sampling function taking text string prompt and generating max_new additional words.

    tokens = re.findall(r"\w+|[,.]", prompt.lower())
    ids = torch.tensor([[stoi.get(t, UNK) for t in tokens]], device=device)     # Encodes prompt words into 2D tensor of token IDs with shape (1, len(tokens)).

    for _ in range(max_new):                                                    # Iterates max_new times to generate new words sequentially.

        # sliding context window - prevents gibberish/crash when max_new + prompt + generated_token >= SEQ_LEN
        # alternative to setting SEQ_LEN = MAX_LEN in training loop, but context from earlier tokens is discarded
        idx_cond = ids[:, -SEQ_LEN:]
        logits, _ = model(idx_cond)                                             # Runs forward pass on sliding context window, returning output logits (1, current_len, VOCAB).

        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)       # dim=-1 finds max across vocab, keepdim=True keeps shape as (1, 1)
        ids = torch.cat([ids, next_token], dim=1)                               # Concatenates predicted token ID onto existing sequence ids along sequence dimension (dim=1), extending context length by 1.

    return " ".join(itos[i.item()] for i in ids[0])                             # Converts tensor of generated token IDs back to strings using itos dictionary and joins them with spaces.


prompt = "šílenství"
print("\nSample output:", generate(prompt))                                    # Calls text generation with seed prompt and prints output string.
logit_lens(model, prompt)
