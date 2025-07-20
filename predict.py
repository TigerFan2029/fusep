import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score

# ─── USER CONFIG ──────────────────────────────────────────────────────────────
MODEL_DIR    = Path("/Users/tiger/Desktop/FUSEP/models/more_data")
DATA_DIR     = Path("/Users/tiger/Desktop/FUSEP")
CSV_FILE     = DATA_DIR / "id_selected.csv"
RGRAM_DIR    = DATA_DIR / "rgram"
RELOC_DIR    = DATA_DIR / "reloc_01"
OUTPUT_DIR   = DATA_DIR / "predictions"

BLOCK_SIZE = 100

# ─── MODEL DEFINITION ─────────────────────────────────────────────────────────
class PeakPicker1D(nn.Module):
    def __init__(self, in_ch=1):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_ch,  32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv1d(32,  64, kernel_size=7, padding=3, stride=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 128, kernel_size=9, padding=4, stride=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 1, kernel_size=1)
        )
    def forward(self,x):
        z = self.cnn(x)
        return F.interpolate(z, size=x.shape[-1], mode="linear", align_corners=False)

def load_model(device):
    model = PeakPicker1D().to(device)
    model.load_state_dict(torch.load(MODEL_DIR/"best.pt", map_location=device))
    model.eval()
    thresh = float(np.load(MODEL_DIR/"best_threshold.npy"))
    return model, thresh

def zscore_cols(arr):
    μ = arr.mean(axis=0, keepdims=True)
    σ = arr.std(axis=0, keepdims=True) + 1e-6
    return (arr - μ)/σ

def predict_mask(model, x_norm, T, device):
    inp = torch.from_numpy(x_norm.T).unsqueeze(1).to(device)
    with torch.no_grad():
        logit = model(inp)
        prob  = torch.sigmoid(logit).cpu().numpy()
    return (prob > T).astype(np.uint8).squeeze().T   # H×W binary mask

# ─── LAYER ANALYSIS ───────────────────────────────────────────────────────────
def analyze_ranges(mask, ranges):
    H,W = mask.shape
    stats = []
    for c0,c1 in ranges:
        c1 = min(c1, W-1)
        pres = mask[:, c0:c1+1].any(axis=1)  # collapse to 1D
        d = np.diff(pres.astype(int))
        starts = np.where(d==1)[0]+1
        ends   = np.where(d==-1)[0]
        if pres[0]:  starts = np.r_[0, starts]
        if pres[-1]: ends   = np.r_[ends, H-1]
        runs = list(zip(starts, ends))
        nlay = len(runs)
        c1_ctr = c2_ctr = dist = np.nan
        if nlay>=1:
            s,e = runs[0]; c1_ctr = (s+e)/2
        if nlay>=2:
            s,e = runs[1]; c2_ctr = (s+e)/2
            dist = abs(c2_ctr - c1_ctr)
        stats.append({
            "col_start": c0, "col_end": c1,
            "n_layers": nlay,
            "center1": float(c1_ctr),
            "center2": float(c2_ctr),
            "distance": float(dist)
        })
    return stats

# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, T = load_model(DEVICE)

    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR/"plots").mkdir(exist_ok=True)

    df    = pd.read_csv(CSV_FILE)
    names = (df["ProductId"]
               .str.replace(r"^s_","",regex=True)
               .str.replace(r"_rgram$","",regex=True))

    all_stats = []
    for name in names:
        # load precomputed radargram and reloc
        X = np.loadtxt(RGRAM_DIR/f"{name}_rgram.txt", dtype=np.float32)
        yfile = RELOC_DIR/f"{name}_reloc_01.txt"
        Y = np.loadtxt(yfile, dtype=np.uint8) if yfile.exists() else None

        # predict
        Xn   = zscore_cols(X)
        mask = predict_mask(model, Xn, T, DEVICE)

        # save mask
        np.savetxt(OUTPUT_DIR/f"{name}_mask.txt", mask, fmt="%d", delimiter="\t")

        # optional metrics
        if Y is not None:
            P = precision_score(Y.flatten(), mask.flatten(), zero_division=1)
            R = recall_score   (Y.flatten(), mask.flatten(), zero_division=1)
            print(f"{name}: Precision={P:.3f}, Recall={R:.3f}")

        # overlay plot
        fig, ax = plt.subplots(1,3,figsize=(15,5), sharex=True, sharey=True)
        vmin,vmax = np.percentile(X,0.5), np.percentile(X,99.5)
        ax[0].imshow(X, cmap="gray", vmin=vmin, vmax=vmax, origin="upper", aspect="auto")
        ax[0].set_title("Radargram")
        if Y is not None:
            ax[1].imshow(Y, cmap="Reds", origin="upper", aspect="auto")
            ax[1].set_title("Ground Truth")
        else:
            ax[1].text(0.5,0.5,"no label",ha="center",va="center")
            ax[1].set_title("Ground Truth")
        ax[2].imshow(X, cmap="gray", vmin=vmin, vmax=vmax, origin="upper", aspect="auto")
        ax[2].imshow(mask, cmap="Reds", alpha=0.4, origin="upper", aspect="auto")
        ax[2].set_title("Prediction Overlay")
        for a in ax:
            a.axis("off")
        fig.suptitle(name, fontsize=14)
        fig.savefig(OUTPUT_DIR/"plots"/f"{name}_overlay.png", dpi=150)
        plt.close(fig)

        # layer analysis
        H, W = mask.shape
        col_ranges = [
            (i, min(i + BLOCK_SIZE - 1, W - 1))
            for i in range(0, W, BLOCK_SIZE)
        ]
        stats = analyze_ranges(mask, col_ranges)

        for s in stats:
            s["name"] = name
            all_stats.append(s)

    # write out summary CSV
    stats_df = pd.DataFrame(all_stats,
        columns=["name","col_start","col_end","n_layers","center1","center2","distance"])
    stats_df.to_csv(OUTPUT_DIR/"layer_stats.csv", index=False)
    print(f"Layer stats written to {OUTPUT_DIR/'layer_stats.csv'}")

if __name__ == "__main__":
    main()
