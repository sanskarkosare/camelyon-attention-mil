"""
Ensemble evaluation: averages sigmoid predictions across all 6 trained models
(3 seeds × 2 architectures). Run after train_mil_150.py has completed.

Legitimate ensemble -- we average ALL models, never select on test.

Save to:  /home/subhabrat/sanskar/Aira/camelyon_remaining/ensemble_eval.py
Run from: /home/subhabrat/sanskar/Aira/camelyon_remaining/
Command:  CUDA_VISIBLE_DEVICES=0 python3 ensemble_eval.py
"""

import os, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

FEATURES_ROOT = "camelyon16_features"
RESULTS_ROOT  = "camelyon16_results_150"
SEEDS         = [42, 7, 123]
FEATURE_DIM   = 2048
HIDDEN_DIM    = 512
ATTENTION_DIM = 256
DROPOUT       = 0.5
N_TRAIN = 45; N_VAL = 15; N_TEST = 15


def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


class MILDataset(Dataset):
    def __init__(self, slides, labels, root):
        self.slides=slides; self.labels=labels; self.root=root
    def __len__(self): return len(self.slides)
    def __getitem__(self, i):
        name=self.slides[i]; label=self.labels[name]
        feats=torch.load(os.path.join(self.root,f"{name}.pt"),weights_only=True)
        feats=torch.nn.functional.normalize(feats,p=2,dim=1)
        return feats,torch.tensor(label,dtype=torch.long),name

def collate(b): return [x[0] for x in b],torch.stack([x[1] for x in b]),[x[2] for x in b]

def make_split(labels, seed):
    normal=[s for s,l in labels.items() if l==0]
    tumor=[s for s,l in labels.items() if l==1]
    rng=random.Random(seed); rng.shuffle(normal); rng.shuffle(tumor)
    def sp(lst): return lst[:N_TRAIN],lst[N_TRAIN:N_TRAIN+N_VAL],lst[N_TRAIN+N_VAL:]
    nt,nv,ne=sp(normal); tt,tv,te=sp(tumor)
    return nt+tt,nv+tv,ne+te


class GatedAttentionMIL(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj=nn.Sequential(nn.Linear(FEATURE_DIM,HIDDEN_DIM),nn.ReLU(),nn.Dropout(DROPOUT))
        self.aV=nn.Linear(HIDDEN_DIM,ATTENTION_DIM)
        self.aU=nn.Linear(HIDDEN_DIM,ATTENTION_DIM)
        self.aw=nn.Linear(ATTENTION_DIM,1)
        self.cls=nn.Sequential(nn.Dropout(DROPOUT),nn.Linear(HIDDEN_DIM,1))
    def forward(self,x):
        h=self.proj(x)
        a=torch.softmax(self.aw(torch.tanh(self.aV(h))*torch.sigmoid(self.aU(h))),dim=0)
        return self.cls((a*h).sum(0,keepdim=True)).squeeze(),a.squeeze()

class MeanPoolMIL(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(FEATURE_DIM,HIDDEN_DIM),nn.ReLU(),
                                nn.Dropout(DROPOUT),nn.Linear(HIDDEN_DIM,1))
    def forward(self,x): return self.net(x.mean(0,keepdim=True)).squeeze()


@torch.no_grad()
def get_probs(model, loader, device, is_attn):
    model.eval()
    slide_probs = {}
    for fl, lbs, names in loader:
        for bag, label, name in zip(fl, lbs, names):
            bag = bag.to(device)
            logit = model(bag)[0] if is_attn else model(bag)
            slide_probs[name] = (torch.sigmoid(logit).item(), label.item())
    return slide_probs


def calibrate(probs, labs):
    best_t, best_acc = 0.5, 0.0
    for t in np.arange(0.05, 0.96, 0.01):
        acc = sum((p>=t)==l for p,l in zip(probs,labs))/len(labs)
        if acc > best_acc: best_acc, best_t = acc, t
    return best_t


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    labels = torch.load(os.path.join(FEATURES_ROOT,"slide_labels.pt"),weights_only=True)

    # Collect per-slide probability predictions from every model
    # Key: slide_name → list of sigmoid outputs across all 6 models
    all_val_probs   = {}  # slide → [prob, prob, ...]
    all_test_probs  = {}
    val_labels_map  = {}
    test_labels_map = {}

    for seed in SEEDS:
        set_seed(seed)
        tr, va, te = make_split(labels, seed)

        def mk(slides):
            return DataLoader(MILDataset(slides,labels,FEATURES_ROOT),
                              batch_size=1,shuffle=False,collate_fn=collate)

        for mname, model, is_attn in [
            ("GatedAttentionMIL", GatedAttentionMIL(), True),
            ("MeanPoolMIL",       MeanPoolMIL(),       False),
        ]:
            ckpt = os.path.join(RESULTS_ROOT, f"seed{seed}_{mname}.pt")
            if not os.path.exists(ckpt):
                print(f"[MISSING] {ckpt} — run train_mil_150.py first")
                continue
            model.load_state_dict(torch.load(ckpt, weights_only=True))
            model = model.to(device)

            vp = get_probs(model, mk(va), device, is_attn)
            tp = get_probs(model, mk(te), device, is_attn)

            for name,(prob,lab) in vp.items():
                all_val_probs.setdefault(name,[]).append(prob)
                val_labels_map[name] = lab
            for name,(prob,lab) in tp.items():
                all_test_probs.setdefault(name,[]).append(prob)
                test_labels_map[name] = lab

    # Average across all models per slide
    def ensemble_eval(probs_map, labels_map, label):
        names  = sorted(probs_map.keys())
        probs  = [np.mean(probs_map[n]) for n in names]
        labs   = [labels_map[n]         for n in names]
        auc    = roc_auc_score(labs, probs)
        return probs, labs, auc

    val_probs, val_labs, val_auc   = ensemble_eval(all_val_probs,  val_labels_map,  "val")
    test_probs, test_labs, test_auc = ensemble_eval(all_test_probs, test_labels_map, "test")

    # Calibrate threshold on val ensemble (never test)
    thresh = calibrate(val_probs, val_labs)
    test_preds = [1 if p>=thresh else 0 for p in test_probs]
    test_acc   = sum(p==l for p,l in zip(test_preds,test_labs)) / len(test_labs)

    print(f"{'='*55}")
    print(f"ENSEMBLE RESULTS (6 models: 3 seeds x 2 architectures)")
    print(f"{'='*55}")
    print(f"  Val  AUC (calibration basis): {val_auc:.4f}")
    print(f"  Calibrated threshold:          {thresh:.2f}")
    print(f"  Test AUC:                      {test_auc:.4f}")
    print(f"  Test Accuracy:                 {test_acc:.4f}  "
          f"({int(test_acc*len(test_labs))}/{len(test_labs)} correct)")
    print(f"\n  Test slides: {len(test_labs)} | "
          f"Val slides: {len(val_labs)}")

    torch.save({"test_auc": test_auc, "test_acc": test_acc,
                "thresh": thresh, "val_auc": val_auc},
               os.path.join(RESULTS_ROOT, "ensemble_results.pt"))


if __name__ == "__main__":
    main()
