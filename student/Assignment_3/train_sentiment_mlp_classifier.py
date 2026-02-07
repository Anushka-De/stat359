
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

from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm


# ========== Config ==========
SEED = 42
BATCH_SIZE = 64

MIN_EPOCHS = 30          # must train at least 30 epochs
MAX_EPOCHS = 60          # you may train longer
PATIENCE = 10            # early stopping patience (only active after epoch >= MIN_EPOCHS)

LR = 1e-3
WEIGHT_DECAY = 1e-4
HIDDEN_DIM = 256
DROPOUT = 0.3
EMB_DIM = 300

OUTPUT_DIR = "outputs"
BEST_CKPT_PATH = os.path.join(OUTPUT_DIR, "best_mlp_fasttext.pt")

_token_re = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+")  # simple tokenizer


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    # mimic pseudo-code style
    return "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


def tokenize(text: str):
    return _token_re.findall(text.lower())


def mean_pool_fasttext(tokens, ft_model):
    """Mean of word vectors (300-d). If no known tokens, return zeros."""
    vecs = []
    for w in tokens:
        if w in ft_model:
            vecs.append(ft_model[w])
    if len(vecs) == 0:
        return np.zeros((EMB_DIM,), dtype=np.float32)
    return np.mean(np.stack(vecs, axis=0), axis=0).astype(np.float32)


def compute_class_weights(labels, num_classes=3, device="cpu"):
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    weights = 1.0 / np.maximum(counts, 1.0)
    weights = weights / weights.sum()  # normalize
    return torch.tensor(weights, dtype=torch.float32, device=device)


def plot_confusion_matrix(cm, class_names, outpath, title="Confusion Matrix"):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    # annotate values
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


# ========== PyTorch Dataset ==========
class FastTextDataset(Dataset):
    def __init__(self, sentences, labels, ft_model):
        self.sentences = list(sentences)
        self.labels = np.array(labels, dtype=np.int64)
        self.ft_model = ft_model

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sent = self.sentences[idx]
        tokens = tokenize(sent)
        x = mean_pool_fasttext(tokens, self.ft_model)  # (300,)
        y = int(self.labels[idx])
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


# ========== Model Definition ==========
class MLPClassifier(nn.Module):
    def __init__(self, in_dim=300, hidden_dim=256, num_classes=3, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ========== Main ==========
def main():
    print("\n========== Loading Dataset ==========")
    set_seed(SEED)
    device = get_device()
    print(f"Using device: {device}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dataset = datasets.load_dataset("financial_phrasebank", "sentences_50agree", trust_remote_code=True)
    full = dataset["train"]
    print("Dataset loaded. Example:", full[0])

    # ========== Splitting Data ==========
    print("\n========== Splitting Data ==========")
    tmp = full.train_test_split(test_size=0.15, seed=SEED, stratify_by_column="label")
    trainval = tmp["train"]
    test_hf = tmp["test"]

    tmp2 = trainval.train_test_split(test_size=0.15, seed=SEED, stratify_by_column="label")
    train_hf = tmp2["train"]
    val_hf = tmp2["test"]

    print(f"Train: {len(train_hf)}, Val: {len(val_hf)}, Test: {len(test_hf)}")

    # ========== Load FastText ==========
    print("\n========== Loading FastText Model ==========")
    ft = api.load("fasttext-wiki-news-subwords-300")
    print("FastText loaded.")

    # ========== DataLoaders ==========
    print("\n========== Creating DataLoaders ==========")
    train_dataset = FastTextDataset(train_hf["sentence"], train_hf["label"], ft)
    val_dataset = FastTextDataset(val_hf["sentence"], val_hf["label"], ft)
    test_dataset = FastTextDataset(test_hf["sentence"], test_hf["label"], ft)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ========== Training Setup ==========
    print("\n========== Setting Up Training ==========")
    num_classes = 3
    model = MLPClassifier(in_dim=EMB_DIM, hidden_dim=HIDDEN_DIM, num_classes=num_classes, dropout=DROPOUT).to(device)

    # class weights based on train labels
    class_weights = compute_class_weights(np.array(train_hf["label"]), num_classes=num_classes, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    print("Training setup complete.")

    # ========== Training Loop ==========
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
            xb = xb.to(device)
            yb = yb.to(device)

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
        model.eval()
        val_running_loss = 0.0
        all_val_preds = []
        all_val_labels = []

        with torch.no_grad():
            for xb, yb in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation", leave=False):
                xb = xb.to(device)
                yb = yb.to(device)

                logits = model(xb)
                loss = criterion(logits, yb)

                val_running_loss += loss.item() * xb.size(0)
                preds = torch.argmax(logits, dim=1)
                all_val_preds.extend(preds.detach().cpu().numpy())
                all_val_labels.extend(yb.detach().cpu().numpy())

        epoch_val_loss = val_running_loss / len(val_loader.dataset)
        val_f1 = f1_score(all_val_labels, all_val_preds, average="macro")
        val_acc = (np.array(all_val_preds) == np.array(all_val_labels)).mean()

        val_loss_history.append(epoch_val_loss)
        val_f1_history.append(val_f1)
        val_acc_history.append(val_acc)

        print(f"Val Loss: {epoch_val_loss:.4f}, Val F1: {val_f1:.4f}, Val Acc: {val_acc:.4f}")

        scheduler.step(val_f1)

        # ---- Best checkpoint by validation F1 ----
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
                        "seed": SEED,
                        "batch_size": BATCH_SIZE,
                        "hidden_dim": HIDDEN_DIM,
                        "dropout": DROPOUT,
                        "lr": LR,
                        "weight_decay": WEIGHT_DECAY,
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

        # ---- Early stopping allowed only AFTER epoch >= MIN_EPOCHS ----
        if (epoch + 1) >= MIN_EPOCHS and epochs_since_improve >= PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1} (best epoch {best_epoch}).")
            break

    # ========== Plot Learning Curves ==========
    print("\n========== Plotting Learning Curves ==========")
    epochs_ran = len(train_loss_history)

    # Combined plot like pseudo code (3 subplots)
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
    combined_path = os.path.join(OUTPUT_DIR, "mlp_learning_curves.png")
    plt.savefig(combined_path, dpi=200)
    plt.close()
    print(f"Learning curves saved as '{combined_path}'.")

    # Separate plots (optional but matches requirement nicely)
    plt.figure(figsize=(8, 6))
    plt.plot(train_loss_history, label="Train Loss")
    plt.plot(val_loss_history, label="Val Loss")
    plt.title("Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    loss_path = os.path.join(OUTPUT_DIR, "mlp_loss_curve.png")
    plt.tight_layout()
    plt.savefig(loss_path, dpi=200)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(train_f1_history, label="Train F1")
    plt.plot(val_f1_history, label="Val F1")
    plt.title("Macro F1 Score Curve")
    plt.xlabel("Epochs")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.grid(True)
    f1_path = os.path.join(OUTPUT_DIR, "mlp_f1_curve.png")
    plt.tight_layout()
    plt.savefig(f1_path, dpi=200)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(train_acc_history, label="Train Accuracy")
    plt.plot(val_acc_history, label="Val Accuracy")
    plt.title("Accuracy Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    acc_path = os.path.join(OUTPUT_DIR, "mlp_accuracy_curve.png")
    plt.tight_layout()
    plt.savefig(acc_path, dpi=200)
    plt.close()

    print(f"Saved: {loss_path}")
    print(f"Saved: {f1_path}")
    print(f"Saved: {acc_path}")

    # ========== Test Evaluation ==========
    print("\n========== Evaluating on Test Set ==========")
    ckpt = torch.load(BEST_CKPT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_test_preds = []
    all_test_labels = []
    test_running_loss = 0.0

    with torch.no_grad():
        for xb, yb in tqdm(test_loader, desc="Testing", leave=False):
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)

            test_running_loss += loss.item() * xb.size(0)
            preds = torch.argmax(logits, dim=1)
            all_test_preds.extend(preds.detach().cpu().numpy())
            all_test_labels.extend(yb.detach().cpu().numpy())

    test_loss = test_running_loss / len(test_loader.dataset)
    test_acc = (np.array(all_test_preds) == np.array(all_test_labels)).mean()
    test_f1_macro = f1_score(all_test_labels, all_test_preds, average="macro")
    test_f1_weighted = f1_score(all_test_labels, all_test_preds, average="weighted")

    print("\n" + "=" * 50)
    print(f"Best Val F1: {ckpt['best_val_f1']:.4f} at epoch {ckpt['best_epoch']}")
    print(f"Final Test Loss: {test_loss:.4f}")
    print(f"Final Test Accuracy: {test_acc:.4f}")
    print(f"Test F1 Macro: {test_f1_macro:.4f}")
    print(f"Test F1 Weighted: {test_f1_weighted:.4f}")
    print("=" * 50 + "\n")

    class_names = ["Negative (0)", "Neutral (1)", "Positive (2)"]
    print("Classification Report:")
    print(classification_report(all_test_labels, all_test_preds, target_names=class_names, digits=4))

    cm = confusion_matrix(all_test_labels, all_test_preds)
    cm_path = os.path.join(OUTPUT_DIR, "mlp_confusion_matrix.png")
    plot_confusion_matrix(cm, class_names, cm_path, title="Confusion Matrix (MLP + FastText)")
    print(f"Confusion matrix saved as '{cm_path}'.")

    # also save raw cm values (nice for notebook)
    np.savetxt(os.path.join(OUTPUT_DIR, "mlp_confusion_matrix.txt"), cm, fmt="%d")

    print("\n========== Script Complete ==========")


if __name__ == "__main__":
    main()
