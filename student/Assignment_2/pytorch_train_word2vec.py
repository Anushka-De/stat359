import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
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
    Returns (center_idx, context_idx) for each skip-gram pair.
    """
    def __init__(self, centers, contexts):
        self.centers = torch.tensor(centers, dtype=torch.long)
        self.contexts = torch.tensor(contexts, dtype=torch.long)

    def __len__(self):
        return len(self.centers)

    def __getitem__(self, idx):
        return self.centers[idx], self.contexts[idx]


# -----------------------------
# Word2Vec Module (two embeddings)
# forward: dot(center, context)
# -----------------------------
class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, embedding_dim)   # center
        self.out_embed = nn.Embedding(vocab_size, embedding_dim)  # context

    def forward(self, center_idx, context_idx):
        """
        Returns logits = dot(input_embedding(center), output_embedding(context))
        Shape: [batch]
        """
        v = self.in_embed(center_idx)     # [B, D]
        u = self.out_embed(context_idx)   # [B, D]
        logits = (v * u).sum(dim=1)       # [B]
        return logits

    def get_embeddings(self):
        # Save input embeddings (standard)
        return self.in_embed.weight.detach().cpu().numpy()


# -----------------------------
# Load processed data
# -----------------------------
with open("processed_data.pkl", "rb") as f:
    data = pickle.load(f)

# --- Robustly locate skip-gram pairs and vocab ---
# Try common key names; adapt if your data.py used different ones.
def first_existing_key(d, keys):
    for k in keys:
        if k in d:
            return k
    return None

center_key = first_existing_key(data, ["centers", "center_words", "center", "X_center"])
context_key = first_existing_key(data, ["contexts", "context_words", "context", "X_context"])
if center_key is None or context_key is None:
    raise KeyError(
        f"Could not find skip-gram pair keys in processed_data.pkl. Found keys: {list(data.keys())}. "
        "Expected something like centers/contexts or center_words/context_words."
    )

centers = data[center_key]
contexts = data[context_key]

if "word2idx" not in data or "idx2word" not in data:
    raise KeyError("processed_data.pkl must contain 'word2idx' and 'idx2word' mappings.")

vocab_size = len(data["word2idx"])

# -----------------------------
# Precompute negative sampling distribution (unigram^(3/4))
# -----------------------------
# Handout says: use word frequency counts; may be in data dict or build from pairs. :contentReference[oaicite:2]{index=2}
freq_key = first_existing_key(data, ["word_counts", "counts", "freq", "frequency", "unigram_counts"])
if freq_key is not None:
    # Expect dict-like mapping index->count or word->count; handle both.
    freq_obj = data[freq_key]
    counts = torch.zeros(vocab_size, dtype=torch.float)
    if isinstance(freq_obj, dict):
        # If keys are ints (indices)
        if all(isinstance(k, int) for k in freq_obj.keys()):
            for idx, c in freq_obj.items():
                if 0 <= idx < vocab_size:
                    counts[idx] = float(c)
        else:
            # keys are words; map via word2idx
            for w, c in freq_obj.items():
                if w in data["word2idx"]:
                    counts[data["word2idx"][w]] = float(c)
    else:
        # If it's already a list/array aligned to vocab
        counts = torch.tensor(freq_obj, dtype=torch.float)
        if counts.numel() != vocab_size:
            raise ValueError("Frequency array length doesn't match vocab_size.")
else:
    # Build counts from centers and contexts (both are indices)
    counts = torch.zeros(vocab_size, dtype=torch.float)
    # Count appearances in pairs (simple approximation)
    for idx in centers:
        counts[int(idx)] += 1.0
    for idx in contexts:
        counts[int(idx)] += 1.0

# Apply 3/4 smoothing and normalize
neg_sampling_dist = counts.pow(0.75)
neg_sampling_dist = neg_sampling_dist / neg_sampling_dist.sum()


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


def sample_negatives(batch_size, num_negatives, dist, device):
    """
    Sample negative word indices using torch.multinomial.
    Returns shape: [B * num_negatives]
    """
    # multinomial expects probabilities on CPU ok; move to device after sampling
    neg = torch.multinomial(dist, batch_size * num_negatives, replacement=True)
    return neg.to(device)


# -----------------------------
# Training loop
# -----------------------------
model.train()
for epoch in range(EPOCHS):
    total_loss = 0.0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)

    for center_batch, context_batch in pbar:
        center_batch = center_batch.to(device)   # [B]
        context_batch = context_batch.to(device) # [B]

        # Positive logits and labels (1)
        pos_logits = model(center_batch, context_batch)           # [B]
        pos_labels = torch.ones_like(pos_logits, device=device)   # [B]

        # Negative sampling: sample NEGATIVE_SAMPLES per positive
        neg_context = sample_negatives(BATCH_SIZE, NEGATIVE_SAMPLES, neg_sampling_dist, device)
        # Repeat centers to match negatives
        neg_center = center_batch.repeat_interleave(NEGATIVE_SAMPLES)  # [B*NEG]
        # logits for negative pairs
        neg_logits = model(neg_center, neg_context)                    # [B*NEG]
        neg_labels = torch.zeros_like(neg_logits, device=device)       # [B*NEG]

        # Optional: ensure negatives do not equal the true context (handout note) :contentReference[oaicite:3]{index=3}
        # Simple fix: if any sampled negative equals the positive context for that center,
        # resample those positions once (good enough for coursework).
        with torch.no_grad():
            pos_context_rep = context_batch.repeat_interleave(NEGATIVE_SAMPLES)
            mask = neg_context.eq(pos_context_rep)
            if mask.any():
                neg_context2 = sample_negatives(int(mask.sum().item()), 1, neg_sampling_dist, device).view(-1)
                neg_context[mask] = neg_context2

        # Compute losses
        loss_pos = criterion(pos_logits, pos_labels)
        loss_neg = criterion(neg_logits, neg_labels)
        loss = loss_pos + loss_neg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{EPOCHS} - avg loss: {avg_loss:.4f}")


# -----------------------------
# Save embeddings and mappings
# -----------------------------
embeddings = model.get_embeddings()
with open("word2vec_embeddings.pkl", "wb") as f:
    pickle.dump(
        {"embeddings": embeddings, "word2idx": data["word2idx"], "idx2word": data["idx2word"]},
        f
    )
print("Embeddings saved to word2vec_embeddings.pkl")
