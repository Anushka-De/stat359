import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm

# Hyperparameters (MUST MATCH HANDOUT)
EMBEDDING_DIM = 100
BATCH_SIZE = 128
EPOCHS = 25
LEARNING_RATE = 0.01
NEGATIVE_SAMPLES = 5  # Number of negative samples per positive pair


# -----------------------------
# Custom Dataset for Skip-gram
# -----------------------------
class SkipGramDataset(Dataset):
    """
    Dataset of (center_idx, context_idx) pairs from processed_data.pkl.
    """
    def __init__(self, centers, contexts):
        self.centers = torch.as_tensor(centers, dtype=torch.long)
        self.contexts = torch.as_tensor(contexts, dtype=torch.long)

    def __len__(self):
        return self.centers.numel()

    def __getitem__(self, idx):
        return self.centers[idx], self.contexts[idx]


# -----------------------------
# Word2Vec Model (two embeddings)
# forward: dot(center, context)
# -----------------------------
class Word2Vec(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, embedding_dim)   # center words
        self.out_embed = nn.Embedding(vocab_size, embedding_dim)  # context words

    def forward(self, center_idx: torch.Tensor, context_idx: torch.Tensor) -> torch.Tensor:
        """
        Returns dot product logits for (center, context).
        Shape: [batch]
        """
        v = self.in_embed(center_idx)    # [B, D]
        u = self.out_embed(context_idx)  # [B, D]
        return (v * u).sum(dim=1)        # [B]

    def get_embeddings(self):
        # Standard: save input embeddings
        return self.in_embed.weight.detach().cpu().numpy()


# -----------------------------
# Load processed data
# -----------------------------
with open("processed_data.pkl", "rb") as f:
    data = pickle.load(f)

# Find the skip-gram pairs DataFrame inside the pickle.
pairs = None
if isinstance(data, pd.DataFrame):
    pairs = data
elif isinstance(data, dict):
    # common key names
    for k in ["skipgram_pairs", "pairs", "skip_gram_pairs", "skipgrams", "df"]:
        if k in data and isinstance(data[k], pd.DataFrame):
            pairs = data[k]
            break
    # otherwise: first DataFrame value in dict
    if pairs is None:
        for v in data.values():
            if isinstance(v, pd.DataFrame):
                pairs = v
                break

if pairs is None:
    raise ValueError(
        "Could not find skip-gram pairs DataFrame in processed_data.pkl. "
        "Expected a pandas DataFrame with columns ['center','context']."
    )

if "center" not in pairs.columns or "context" not in pairs.columns:
    raise ValueError(f"Skip-gram DataFrame must have columns ['center','context'], got {list(pairs.columns)}")

centers = pairs["center"].to_numpy()
contexts = pairs["context"].to_numpy()

# vocab mappings
if not isinstance(data, dict) or "word2idx" not in data or "idx2word" not in data:
    raise KeyError("processed_data.pkl must include 'word2idx' and 'idx2word' dictionaries for saving embeddings.")

vocab_size = data.get("vocab_size", len(data["word2idx"]))


# -----------------------------
# Precompute negative sampling distribution (unigram^(3/4))
# -----------------------------
# Prefer precomputed counts if present; otherwise build from pairs.
counts = None
for k in ["word_counts", "counts", "freq", "frequency", "unigram_counts"]:
    if isinstance(data, dict) and k in data:
        obj = data[k]
        # dict: can be idx->count or word->count
        if isinstance(obj, dict):
            c = torch.zeros(vocab_size, dtype=torch.float)
            if all(isinstance(kk, int) for kk in obj.keys()):
                for idx, val in obj.items():
                    if 0 <= idx < vocab_size:
                        c[idx] = float(val)
            else:
                # word -> count
                for w, val in obj.items():
                    if w in data["word2idx"]:
                        c[data["word2idx"][w]] = float(val)
            counts = c
        else:
            # list/array aligned to vocab
            t = torch.tensor(obj, dtype=torch.float)
            if t.numel() == vocab_size:
                counts = t
        break

if counts is None:
    # build counts from skip-gram indices (fast)
    c_centers = torch.bincount(torch.as_tensor(centers, dtype=torch.long), minlength=vocab_size).float()
    c_contexts = torch.bincount(torch.as_tensor(contexts, dtype=torch.long), minlength=vocab_size).float()
    counts = c_centers + c_contexts

# unigram^(3/4)
neg_dist = counts.pow(0.75)
neg_dist_sum = neg_dist.sum()
if neg_dist_sum.item() == 0:
    raise ValueError("Negative sampling distribution sum is zero; counts appear to be empty.")
neg_dist = neg_dist / neg_dist_sum  # probability distribution over vocab


# -----------------------------
# Device selection: CUDA > MPS > CPU
# -----------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)


# -----------------------------
# Dataset and DataLoader
# -----------------------------
dataset = SkipGramDataset(centers, contexts)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)


# -----------------------------
# Model, Loss, Optimizer
# -----------------------------
model = Word2Vec(vocab_size=vocab_size, embedding_dim=EMBEDDING_DIM).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


def sample_negative_context(batch_size: int, num_neg: int, dist: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Samples negative context word indices using torch.multinomial.
    Returns shape: [B, num_neg]
    """
    # multinomial operates on CPU tensor fine; move sampled indices to device
    neg = torch.multinomial(dist, batch_size * num_neg, replacement=True).view(batch_size, num_neg)
    return neg.to(device)


def make_targets(pos_logits: torch.Tensor, neg_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build concatenated logits and targets for BCEWithLogitsLoss.
    pos_logits: [B]
    neg_logits: [B, K]
    returns:
      logits:  [B + B*K]
      targets: [B + B*K]
    """
    bsz = pos_logits.shape[0]
    k = neg_logits.shape[1]
    logits = torch.cat([pos_logits, neg_logits.reshape(-1)], dim=0)  # [B + B*K]
    targets = torch.cat(
        [torch.ones(bsz, device=pos_logits.device), torch.zeros(bsz * k, device=pos_logits.device)],
        dim=0,
    )
    return logits, targets


# -----------------------------
# Training loop
# -----------------------------
model.train()

for epoch in range(EPOCHS):
    total_loss = 0.0

    for center_batch, context_batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        center_batch = center_batch.to(device)     # [B]
        context_batch = context_batch.to(device)   # [B]

        # Positive logits: dot(center, true_context)
        pos_logits = model(center_batch, context_batch)  # [B]

        # Negative samples for each positive pair
        neg_context = sample_negative_context(BATCH_SIZE, NEGATIVE_SAMPLES, neg_dist, device)  # [B, K]

        # Ensure negatives don't include the actual positive context word (resample masked positions)
        # (Usually very rare, but required by handout.)
        for _ in range(3):  # a few attempts is enough
            mask = neg_context.eq(context_batch.unsqueeze(1))  # [B, K]
            if not mask.any():
                break
            # resample only the masked positions
            num_bad = int(mask.sum().item())
            replacement = torch.multinomial(neg_dist, num_bad, replacement=True).to(device)
            neg_context[mask] = replacement

        # Compute negative logits: dot(center, sampled_context)
        center_emb = model.in_embed(center_batch)           # [B, D]
        neg_emb = model.out_embed(neg_context)              # [B, K, D]
        neg_logits = (neg_emb * center_emb.unsqueeze(1)).sum(dim=2)  # [B, K]

        # BCE loss over combined logits/targets
        logits, targets = make_targets(pos_logits, neg_logits)
        loss = criterion(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{EPOCHS} - avg loss: {avg_loss:.4f}")


# -----------------------------
# Save embeddings and mappings
# -----------------------------
embeddings = model.get_embeddings()
with open("word2vec_embeddings.pkl", "wb") as f:
    pickle.dump({"embeddings": embeddings, "word2idx": data["word2idx"], "idx2word": data["idx2word"]}, f)

print("Embeddings saved to word2vec_embeddings.pkl")
