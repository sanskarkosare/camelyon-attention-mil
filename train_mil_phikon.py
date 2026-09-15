"""
MIL training using Phikon features (768-dim instead of 2048-dim).
Everything else identical to train_mil_150.py.

Save to:  /home/subhabrat/sanskar/Aira/camelyon_remaining/train_mil_phikon.py
Run from: /home/subhabrat/sanskar/Aira/camelyon_remaining/
Command:  CUDA_VISIBLE_DEVICES=0 python3 train_mil_phikon.py
"""

import os, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

FEATURES_ROOT = "camelyon16_features_phikon"   # Phikon features
RESULTS_ROOT  = "camelyon16_results_phikon"
NUM_EPOCHS    = 60
LR            = 1e-4
WEIGHT_DECAY  = 1e-5
GRAD_CLIP     = 1.0
DROPOUT       = 0.5
SEEDS         = [42, 7, 123, 0, 1, 2, 3, 5, 10, 15]

FEATURE_DIM   = 768    # Phikon ViT-B CLS token dim
HIDDEN_DIM    = 384
ATTENTION_DIM = 192

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

def collate(b):
    return [x[0] for x in b],torch.stack([x[1] for x in b]),[x[2] for x in b]

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
        self.proj=nn.Sequential(
            nn.Linear(FEATURE_DIM,HIDDEN_DIM),nn.ReLU(),nn.Dropout(DROPOUT))
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
        self.net=nn.Sequential(
            nn.Linear(FEATURE_DIM,HIDDEN_DIM),nn.ReLU(),
            nn.Dropout(DROPOUT),nn.Linear(HIDDEN_DIM,1))
    def forward(self,x):
        return self.net(x.mean(0,keepdim=True)).squeeze()


def train_epoch(model,loader,opt,device,is_attn):
    model.train(); crit=nn.BCEWithLogitsLoss(); loss_sum=0
    for fl,lbs,_ in loader:
        lbs=lbs.float().to(device)
        for bag,label in zip(fl,lbs):
            bag=bag.to(device); opt.zero_grad()
            logit=model(bag)[0] if is_attn else model(bag)
            loss=crit(logit.unsqueeze(0),label.unsqueeze(0))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),GRAD_CLIP)
            opt.step(); loss_sum+=loss.item()
    return loss_sum/len(loader.dataset)


@torch.no_grad()
def evaluate(model,loader,device,is_attn,thresh=0.5,
             save_attn=False,attn_dir=None):
    model.eval(); probs,labs=[],[]
    for fl,lbs,names in loader:
        for bag,label,name in zip(fl,lbs,names):
            bag=bag.to(device)
            if is_attn:
                logit,attn=model(bag)
                if save_attn and attn_dir:
                    os.makedirs(attn_dir,exist_ok=True)
                    torch.save(attn.cpu(),
                               os.path.join(attn_dir,f"{name}_attn.pt"))
            else:
                logit=model(bag)
            probs.append(torch.sigmoid(logit).item())
            labs.append(label.item())
    auc=roc_auc_score(labs,probs)
    preds=[1 if p>=thresh else 0 for p in probs]
    acc=sum(p==l for p,l in zip(preds,labs))/len(labs)
    return auc,acc,probs,labs


def calibrate(probs,labs):
    best_t,best_acc=0.5,0.0
    for t in np.arange(0.05,0.96,0.01):
        acc=sum((p>=t)==l for p,l in zip(probs,labs))/len(labs)
        if acc>best_acc: best_acc,best_t=acc,t
    return best_t,best_acc


def run_seed(seed,labels,device):
    set_seed(seed)
    tr,va,te=make_split(labels,seed)
    def mk(slides,shuf):
        return DataLoader(MILDataset(slides,labels,FEATURES_ROOT),
                          batch_size=1,shuffle=shuf,collate_fn=collate)
    results={}
    for mname,model,is_attn in [
        ("GatedAttentionMIL",GatedAttentionMIL(),True),
        ("MeanPoolMIL",MeanPoolMIL(),False),
    ]:
        model=model.to(device)
        opt=optim.Adam(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
        sched=optim.lr_scheduler.CosineAnnealingLR(opt,NUM_EPOCHS)
        best_auc,best_ckpt=0.0,f"{RESULTS_ROOT}/seed{seed}_{mname}.pt"
        os.makedirs(RESULTS_ROOT,exist_ok=True)

        for epoch in range(1,NUM_EPOCHS+1):
            train_epoch(model,mk(tr,True),opt,device,is_attn)
            val_auc,_,_,_=evaluate(model,mk(va,False),device,is_attn)
            sched.step()
            if val_auc>best_auc:
                best_auc=val_auc
                torch.save(model.state_dict(),best_ckpt)
            if epoch%10==0 or epoch==1:
                print(f"    epoch {epoch:3d} | val_AUC {val_auc:.4f}")

        model.load_state_dict(torch.load(best_ckpt,weights_only=True))
        _,_,vp,vl=evaluate(model,mk(va,False),device,is_attn)
        thresh,_=calibrate(vp,vl)
        attn_dir=f"{RESULTS_ROOT}/seed{seed}_{mname}_attention"
        test_auc,test_acc,_,_=evaluate(
            model,mk(te,False),device,is_attn,
            thresh=thresh,save_attn=is_attn,attn_dir=attn_dir)
        results[mname]={"AUC":test_auc,"Acc":test_acc,
                        "best_val_auc":best_auc,"thresh":thresh}
        print(f"  seed={seed} | {mname:<22} | "
              f"best_val={best_auc:.4f} | "
              f"test_AUC={test_auc:.4f} | test_acc={test_acc:.4f}")
    return results


def main():
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type=="cuda": print(f"GPU: {torch.cuda.get_device_name(0)}")

    labels=torch.load(os.path.join(FEATURES_ROOT,"slide_labels.pt"),
                      weights_only=True)
    print(f"Total slides: {len(labels)}")
    assert len(labels)==150, f"Expected 150, got {len(labels)}"
    print(f"Feature dim: {FEATURE_DIM} (Phikon)\n")

    all_results={}
    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        all_results[seed]=run_seed(seed,labels,device)

    print(f"\n{'='*60}")
    print("FINAL RESULTS (best val-AUC seed per model)")
    print(f"{'='*60}")
    for mname in ["GatedAttentionMIL","MeanPoolMIL"]:
        best_seed=max(SEEDS,
            key=lambda s:all_results[s][mname]["best_val_auc"])
        r=all_results[best_seed][mname]
        print(f"  {mname:<25} seed={best_seed} | "
              f"AUC: {r['AUC']:.4f} | Acc: {r['Acc']:.4f}")

    torch.save({"all":all_results},
               os.path.join(RESULTS_ROOT,"summary.pt"))


if __name__=="__main__":
    main()
