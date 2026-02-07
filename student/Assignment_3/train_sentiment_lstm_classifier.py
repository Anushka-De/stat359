#!/usr/bin/env python3
# coding: utf-8
"""
Task 2: LSTM with Padded FastText Word Vectors
- Pad/truncate each sentence to exactly 32 tokens
- Represent each sentence as a (32, 300) tensor (FastText vectors)
- No nn.Embedding (we precompute word vectors directly)
- LSTM + final hidden state -> classifier
"""

# ========== Imports ==========
import os
import re
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import datasets
import gensim.downloader as api

from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm


# ========== Config ==========
SEED = 42
BATCH_SIZE = 64

MIN_EPOCHS = 30
MAX_EPOCHS = 80       # you can raise if needed to hit F1>=0.70
PATIENCE = 10         # early stopping patience (active only after epoch >= MIN_EPOCHS)

LR = 2e-3             # slightly higher than Task1 often helps LSTM here
WEIGHT_DECAY = 1e-4

EMB_DIM = 300
SEQ_LEN = 32          # REQUIRED: exactly 32 tokens

HIDDEN_SIZE = 256
NUM_LAYERS = 1
BIDIRECTIONAL = True
DROPOUT = 0.3         # for classifier head

OUTPUT_DIR = "outputs_task2"
BEST_CKPT_PATH = os.path.join(OUTPUT_DIR, "best_lstm_fasttext.pt")

_token_re = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+")  # simple tokenizer


# ========== Utilities ==========
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    return "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


def tokenize(text: str):
    return _token_re.findall(text.lower())


def compute_class_weights(labels, num_classes=3, device="cpu"):
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    weights = 1.0 / np.maximum(counts, 1.0)
    weights = weights / weights.sum()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def save_curve_plot(train_vals, val_vals, ylabel, title, outpath):
    epochs = np.arange(1, len(train_vals) + 1)
    plt.figure()
    plt.plot(epochs, train_vals, marker="o", label="train")
    plt.plot(epochs, val_vals, marker="o", label="val")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_confusion_matrix(cm, class_names, outpath, title="Confusion Matrix"):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


# ========== Dataset ==========
class FastTextPaddedDataset(Dataset):
    """
    Returns:
      x: FloatTensor [32, 300]  (padded/truncated to exactly 32)
      y: LongTensor
    """
    def __init__(self, sentences, labels, ft_model, seq_len=32):
        self.sentences = list(sentences)
        self.labels = np.array(labels, dtype=np.int64)
        self.ft = ft_model
        self.seq_len = seq_len

        self.zero_vec = np.zeros((EMB_DIM,), dtype=np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sent = self.sentences[idx]
        y = int(self.labels[idx])

        toks = tokenize(sent)

        # truncate/pad to exactly seq_len tokens
        toks = toks[: self.seq_len]
        if len(toks) < self.seq_len:
            toks = toks + ["<PAD>"] * (self.seq_len - len(toks))

        # build (32, 300) array
        vecs = []
        for w in toks:
            if w == "<PAD>":
                vecs.append(self.zero_vec)
            else:
                if w in self.ft:
                    vecs.append(self.ft[w])
                else:
                    vecs.append(self.zero_vec)

        x = torch.tensor(np.stack(vecs, axis=0), dtype=torch.float32)  # [32,300]
        return x, torch.tensor(y, dtype=torch.long)


# ========== Model ==========
class LSTMSentimentClassifier(nn.Module):
    """
    Takes x: [B, 32, 300] and uses final hidden state to classify.
    """
    def __init__(self, emb_dim=300, hidden_size=256, num_layers=1, bidirectional=True, dropout=0.3, num_classes=3):
        super().__init__()
        self.bidirectional = bidirectional
        self.num_dirs = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.0  # LSTM dropout only applies when num_layers > 1
        )

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * self.num_dirs, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        # x: [B, 32, 300]
        _, (h_n, _) = self.lstm(x)
        # h_n: [num_layers*num_dirs, B, hidden]

        if self.bidirectional:
            h_fwd = h_n[-2, :, :]
            h_bwd = h_n[-1, :, :]
            h = torch.cat([h_fwd, h_bwd], dim=1)  # [B, 2H]
        else:
            h = h_n[-1, :, :]  # [B, H]

        logits = self.head(h)
        return logits


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    losses = []
    all_preds, all_y = [], []

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        losses.append(loss.item())

        preds = torch.argmax(logits, dim=1)
        all_preds.append(preds.cpu().numpy())
        all_y.append(yb.cpu().numpy())

    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_preds)
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    return float(np.mean(losses)), acc, macro_f1


def main():
    print("\n========== Loading Dataset ==========")
    set_seed(SEED)
    device = get_device()
    print(f"Using device: {device}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dataset = datasets.load_dataset("financial_phrasebank", "sentences_50agree", trust_remote_code=True)
    full = dataset["train"]
    print("Dataset loaded. Example:", full[0])

    print("\n========== Splitting Data ==========")
    tmp = full.train_test_split(test_size=0.15, seed=SEED, stratify_by_column="label")
    trainval = tmp["train"]
    test_hf = tmp["test"]

    tmp2 = trainval.train_test_split(test_size=0.15, seed=SEED, stratify_by_column="label")
    train_hf = tmp2["train"]
    val_hf = tmp2["test"]

    print(f"Train: {len(train_hf)}, Val: {len(val_hf)}, Test: {len(test_hf)}")

    print("\n========== Loading FastText Model ==========")
    ft = api.load("fasttext-wiki-news-subwords-300")
    print("FastText loaded.")

    print("\n========== Creating DataLoaders ==========")
    train_dataset = FastTextPaddedDataset(train_hf["sentence"], train_hf["label"], ft, seq_len=SEQ_LEN)
    val_dataset   = FastTextPaddedDataset(val_hf["sentence"],   val_hf["label"],   ft, seq_len=SEQ_LEN)
    test_dataset  = FastTextPaddedDataset(test_hf["sentence"],  test_hf["label"],  ft, seq_len=SEQ_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

    print("\n========== Setting Up Training ==========")
    model = LSTMSentimentClassifier(
        emb_dim=EMB_DIM,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        bidirectional=BIDIRECTIONAL,
        dropout=DROPOUT,
        num_classes=3,
    ).to(device)

    class_weights = compute_class_weights(np.array(train_hf["label"]), num_classes=3, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    print("Training setup complete.")

    print("\n========== Starting Training Loop ==========")
    best_val_f1 = -1.0
    best_epoch = -1
    epochs_since_improve = 0

    train_loss_history = []
    val_loss_history = []
    train_f1_history = []
    val_f1_history = []
    train_acc_history = []
    val_acc_history = []

    for epoch in range(MAX_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{MAX_EPOCHS} ---")

        # ---- Train ----
        model.train()
        running_loss = 0.0
        all_train_preds = []
        all_train_labels = []

        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch+1} Training", leave=False):
            xb, yb = xb.to(device), yb.to(device)

            logits = model(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)
            preds = torch.argmax(logits, dim=1)
            all_train_preds.extend(preds.detach().cpu().numpy())
            all_train_labels.extend(yb.detach().cpu().numpy())

        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_f1 = f1_score(all_train_labels, all_train_preds, average="macro")
        train_acc = (np.array(all_train_preds) == np.array(all_train_labels)).mean()

        train_loss_history.append(epoch_train_loss)
        train_f1_history.append(train_f1)
        train_acc_history.append(train_acc)

        print(f"Train Loss: {epoch_train_loss:.4f}, Train F1: {train_f1:.4f}, Train Acc: {train_acc:.4f}")

        # ---- Validation ----
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)

        val_loss_history.append(val_loss)
        val_f1_history.append(val_f1)
        val_acc_history.append(val_acc)

        print(f"Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}, Val Acc: {val_acc:.4f}")

        scheduler.step(val_f1)

        # ---- checkpoint ----
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            epochs_since_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_val_f1": best_val_f1,
                    "best_epoch": best_epoch,
                    "config": {
                        "seq_len": SEQ_LEN,
                        "hidden_size": HIDDEN_SIZE,
                        "bidirectional": BIDIRECTIONAL,
                        "seed": SEED,
                    },
                    "histories": {
                        "train_loss": train_loss_history,
                        "val_loss": val_loss_history,
                        "train_f1": train_f1_history,
                        "val_f1": val_f1_history,
                        "train_acc": train_acc_history,
                        "val_acc": val_acc_history,
                    },
                },
                BEST_CKPT_PATH,
            )
            print(f">>> Saved new best model (Val F1: {best_val_f1:.4f}) at epoch {best_epoch}")
        else:
            epochs_since_improve += 1

        # ---- early stopping only after 30 epochs ----
        if (epoch + 1) >= MIN_EPOCHS and epochs_since_improve >= PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1} (best epoch {best_epoch}).")
            break

    # ========== Plot Learning Curves ==========
    print("\n========== Plotting Learning Curves ==========")

    # Combined plot (3 subplots)
    plt.figure(figsize=(12, 15))
    plt.subplot(3, 1, 1)
    plt.plot(train_loss_history, label="Train Loss")
    plt.plot(val_loss_history, label="Val Loss")
    plt.title("Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(train_f1_history, label="Train F1")
    plt.plot(val_f1_history, label="Val F1")
    plt.title("Macro F1 Score Curve")
    plt.xlabel("Epochs")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(train_acc_history, label="Train Accuracy")
    plt.plot(val_acc_history, label="Val Accuracy")
    plt.title("Accuracy Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    combined_path = os.path.join(OUTPUT_DIR, "lstm_learning_curves.png")
    plt.savefig(combined_path, dpi=200)
    plt.close()
    print(f"Learning curves saved as '{combined_path}'.")

    # Also save separate plots (matches evaluation requirement clearly)
    save_curve_plot(train_loss_history, val_loss_history, "Loss", "Loss vs Epochs",
                    os.path.join(OUTPUT_DIR, "loss_vs_epochs.png"))
    save_curve_plot(train_acc_history, val_acc_history, "Accuracy", "Accuracy vs Epochs",
                    os.path.join(OUTPUT_DIR, "acc_vs_epochs.png"))
    save_curve_plot(train_f1_history, val_f1_history, "Macro F1", "Macro F1 vs Epochs",
                    os.path.join(OUTPUT_DIR, "macro_f1_vs_epochs.png"))

    # ========== Test Evaluation ==========
    print("\n========== Evaluating on Test Set ==========")
    ckpt = torch.load(BEST_CKPT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_preds = []
    all_labels = []
    test_losses = []

    with torch.no_grad():
        for xb, yb in tqdm(test_loader, desc="Testing", leave=False):
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            test_losses.append(loss.item())

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())

    test_loss = float(np.mean(test_losses))
    test_acc = (np.array(all_preds) == np.array(all_labels)).mean()
    test_f1_macro = f1_score(all_labels, all_preds, average="macro")
    test_f1_weighted = f1_score(all_labels, all_preds, average="weighted")

    print("\n" + "=" * 50)
    print(f"Best Val F1: {ckpt['best_val_f1']:.4f} at epoch {ckpt['best_epoch']}")
    print(f"Final Test Loss: {test_loss:.4f}")
    print(f"Final Test Accuracy: {test_acc:.4f}")
    print(f"Test F1 Macro: {test_f1_macro:.4f}")
    print(f"Test F1 Weighted: {test_f1_weighted:.4f}")
    print("=" * 50 + "\n")

    class_names = ["Negative (0)", "Neutral (1)", "Positive (2)"]
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    cm = confusion_matrix(all_labels, all_preds)
    np.savetxt(os.path.join(OUTPUT_DIR, "confusion_matrix.txt"), cm, fmt="%d")
    plot_confusion_matrix(cm, class_names, os.path.join(OUTPUT_DIR, "confusion_matrix.png"),
                          title="Confusion Matrix (LSTM + FastText)")
    print(f"Confusion matrix saved as '{os.path.join(OUTPUT_DIR, 'confusion_matrix.png')}'.")

    print("\n========== Script Complete ==========")


if __name__ == "__main__":
    main()
