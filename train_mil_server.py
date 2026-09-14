"""
Improved Gated-Attention MIL training — server version (GPU).

Key fixes over the first run:
  1. Bag-size capping: bags > MAX_BAG_SIZE are randomly subsampled
     down to MAX_BAG_SIZE patches. This kills the 65x size variance
     (304 to 19,771 patches) that was destabilizing training.
  2. Lower LR (1e-4 instead of 2e-4) + gradient clipping (norm 1.0).
  3. 100 epochs instead of 30 — loss was still dropping at epoch 30.
  4. Validation-calibrated decision threshold (not hardcoded 0.5).

Save to:  /home/subhabrat/sanskar/Aira/camelyon_remaining/train_mil_server.py
Run from: /home/subhabrat/sanskar/Aira/camelyon_remaining/
Command:  CUDA_VISIBLE_DEVICES=0 python3 train_mil_server.py
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

# ---- CONFIG ----
FEATURES_ROOT  = "camelyon16_features"
RESULTS_ROOT   = "camelyon16_results"
SEED           = 42
NUM_EPOCHS     = 100
LR             = 1e-4
WEIGHT_DECAY   = 1e-5
GRAD_CLIP      = 1.0
MAX_BAG_SIZE   = 2000   # randomly subsample bags larger than this

FEATURE_DIM    = 2048
HIDDEN_DIM     = 512
ATTENTION_DIM  = 256

N_TRAIN_PER_CLASS = 15
N_VAL_PER_CLASS   = 5
N_TEST_PER_CLASS   = 5


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Dataset
# ============================================================

class MILDataset(Dataset):
    def __init__(self, slide_names, slide_labels, features_root,
                 max_bag_size=None, training=False):
        self.slide_names   = slide_names
        self.slide_labels  = slide_labels
        self.features_root = features_root
        self.max_bag_size  = max_bag_size
        self.training      = training

    def __len__(self):
        return len(self.slide_names)

    def __getitem__(self, idx):
        name  = self.slide_names[idx]
        label = self.slide_labels[name]
        feats = torch.load(
            os.path.join(self.features_root, f"{name}.pt"),
            weights_only=True
        )
        # Bag-size capping: randomly subsample during training,
        # deterministically take the first MAX_BAG_SIZE at test time
        if self.max_bag_size and feats.shape[0] > self.max_bag_size:
            if self.training:
                idx_sel = torch.randperm(feats.shape[0])[:self.max_bag_size]
            else:
                idx_sel = torch.arange(self.max_bag_size)
            feats = feats[idx_sel]
        return feats, torch.tensor(label, dtype=torch.long), name


def mil_collate(batch):
    features = [item[0] for item in batch]
    labels   = torch.stack([item[1] for item in batch])
    names    = [item[2] for item in batch]
    return features, labels, names


def make_splits(slide_labels, seed):
    normal_slides = [s for s, l in slide_labels.items() if l == 0]
    tumor_slides  = [s for s, l in slide_labels.items() if l == 1]
    rng = random.Random(seed)
    rng.shuffle(normal_slides)
    rng.shuffle(tumor_slides)

    def split_class(slides):
        train = slides[:N_TRAIN_PER_CLASS]
        val   = slides[N_TRAIN_PER_CLASS:N_TRAIN_PER_CLASS + N_VAL_PER_CLASS]
        test  = slides[N_TRAIN_PER_CLASS + N_VAL_PER_CLASS:]
        return train, val, test

    n_tr, n_va, n_te = split_class(normal_slides)
    t_tr, t_va, t_te = split_class(tumor_slides)
    return (n_tr + t_tr), (n_va + t_va), (n_te + t_te)


# ============================================================
# Models
# ============================================================

class GatedAttentionMIL(nn.Module):
    def __init__(self, feature_dim=2048, hidden_dim=512,
                 attention_dim=256, n_classes=1):
        super().__init__()
        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
        )
        self.attention_V = nn.Linear(hidden_dim, attention_dim)
        self.attention_U = nn.Linear(hidden_dim, attention_dim)
        self.attention_w = nn.Linear(attention_dim, 1)
        self.classifier  = nn.Linear(hidden_dim, n_classes)

    def forward(self, features):
        h   = self.feature_proj(features)
        a_V = torch.tanh(self.attention_V(h))
        a_U = torch.sigmoid(self.attention_U(h))
        a   = self.attention_w(a_V * a_U)
        a   = torch.softmax(a, dim=0)
        z   = (a * h).sum(dim=0, keepdim=True)
        logit = self.classifier(z).squeeze()
        return logit, a.squeeze()


class MeanPoolMIL(nn.Module):
    def __init__(self, feature_dim=2048, hidden_dim=512, n_classes=1):
        super().__init__()
        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
        )
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, features):
        h     = self.feature_proj(features)
        z     = h.mean(dim=0, keepdim=True)
        logit = self.classifier(z).squeeze()
        return logit


# ============================================================
# Train / Evaluate
# ============================================================

def train_one_epoch(model, loader, optimizer, device, is_attn):
    model.train()
    criterion  = nn.BCEWithLogitsLoss()
    total_loss = 0.0

    for features_list, labels, _ in loader:
        labels = labels.float().to(device)
        for bag_feats, label in zip(features_list, labels):
            bag_feats = bag_feats.to(device)
            optimizer.zero_grad()
            if is_attn:
                logit, _ = model(bag_feats)
            else:
                logit = model(bag_feats)
            loss = criterion(logit.unsqueeze(0), label.unsqueeze(0))
            loss.backward()
            # Gradient clipping — stabilises training with variable bag sizes
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            total_loss += loss.item()

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device, is_attn, threshold=0.5,
             save_attention=False, attention_dir=None):
    model.eval()
    all_probs, all_labels = [], []

    for features_list, labels, names in loader:
        for bag_feats, label, name in zip(features_list, labels, names):
            bag_feats = bag_feats.to(device)
            if is_attn:
                logit, attn = model(bag_feats)
                if save_attention and attention_dir:
                    os.makedirs(attention_dir, exist_ok=True)
                    torch.save(attn.cpu(),
                               os.path.join(attention_dir, f"{name}_attn.pt"))
            else:
                logit = model(bag_feats)
            prob = torch.sigmoid(logit).item()
            all_probs.append(prob)
            all_labels.append(label.item())

    auc   = roc_auc_score(all_labels, all_probs)
    preds = [1 if p >= threshold else 0 for p in all_probs]
    acc   = sum(p == l for p, l in zip(preds, all_labels)) / len(all_labels)
    return auc, acc, all_probs, all_labels


def find_best_threshold(probs, labels):
    best_thresh, best_acc = 0.5, 0.0
    for t in np.arange(0.05, 0.96, 0.01):
        preds = [1 if p >= t else 0 for p in probs]
        acc   = sum(p == l for p, l in zip(preds, labels)) / len(labels)
        if acc > best_acc:
            best_acc, best_thresh = acc, t
    return best_thresh, best_acc


def run_experiment(model_name, model, train_loader, val_loader,
                   test_loader, device, results_dir):
    is_attn   = isinstance(model, GatedAttentionMIL)
    optimizer = optim.Adam(model.parameters(), lr=LR,
                           weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, NUM_EPOCHS)
    best_val_auc = 0.0
    best_ckpt    = os.path.join(results_dir, f"{model_name}_best.pt")

    print(f"\n{'='*55}")
    print(f"Training: {model_name}")
    print(f"{'='*55}")

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer,
                                     device, is_attn)
        val_auc, val_acc, _, _ = evaluate(model, val_loader, device, is_attn)
        scheduler.step()

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), best_ckpt)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | loss {train_loss:.4f} | "
                  f"val AUC {val_auc:.4f} | val acc {val_acc:.4f}")

    # Calibrate threshold on val set, then evaluate test
    model.load_state_dict(torch.load(best_ckpt, weights_only=True))
    _, _, val_probs, val_labels = evaluate(model, val_loader, device, is_attn)
    best_thresh, cal_val_acc = find_best_threshold(val_probs, val_labels)
    print(f"  Calibrated threshold: {best_thresh:.2f} "
          f"(val acc: {cal_val_acc:.4f})")

    attn_dir = os.path.join(results_dir, f"{model_name}_attention")
    test_auc, test_acc, _, _ = evaluate(
        model, test_loader, device, is_attn,
        threshold=best_thresh,
        save_attention=is_attn, attention_dir=attn_dir
    )
    print(f"\n  >>> {model_name} TEST — "
          f"AUC: {test_auc:.4f} | Acc: {test_acc:.4f} "
          f"(threshold={best_thresh:.2f})")
    return test_auc, test_acc


# ============================================================
# Main
# ============================================================

def main():
    set_seed(SEED)
    os.makedirs(RESULTS_ROOT, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    slide_labels = torch.load(
        os.path.join(FEATURES_ROOT, "slide_labels.pt"), weights_only=True
    )
    print(f"Total slides: {len(slide_labels)}")
    print(f"Max bag size cap: {MAX_BAG_SIZE} patches")

    train_slides, val_slides, test_slides = make_splits(slide_labels, SEED)
    print(f"Split -> train: {len(train_slides)} | "
          f"val: {len(val_slides)} | test: {len(test_slides)}")

    def make_loader(slides, training):
        ds = MILDataset(slides, slide_labels, FEATURES_ROOT,
                        max_bag_size=MAX_BAG_SIZE, training=training)
        return DataLoader(ds, batch_size=1, shuffle=training,
                          collate_fn=mil_collate)

    train_loader = make_loader(train_slides, training=True)
    val_loader   = make_loader(val_slides,   training=False)
    test_loader  = make_loader(test_slides,  training=False)

    results = {}

    attn_model = GatedAttentionMIL(FEATURE_DIM, HIDDEN_DIM,
                                    ATTENTION_DIM).to(device)
    auc, acc = run_experiment("GatedAttentionMIL", attn_model,
                               train_loader, val_loader, test_loader,
                               device, RESULTS_ROOT)
    results["GatedAttentionMIL"] = {"AUC": auc, "Acc": acc}

    mean_model = MeanPoolMIL(FEATURE_DIM, HIDDEN_DIM).to(device)
    auc, acc = run_experiment("MeanPoolMIL", mean_model,
                               train_loader, val_loader, test_loader,
                               device, RESULTS_ROOT)
    results["MeanPoolMIL"] = {"AUC": auc, "Acc": acc}

    print(f"\n{'='*55}")
    print("FINAL RESULTS")
    print(f"{'='*55}")
    for name, r in results.items():
        print(f"  {name:<25} AUC: {r['AUC']:.4f}  Acc: {r['Acc']:.4f}")

    torch.save(results, os.path.join(RESULTS_ROOT, "results_summary.pt"))
    print(f"\nResults saved to {RESULTS_ROOT}/")


if __name__ == "__main__":
    main()
