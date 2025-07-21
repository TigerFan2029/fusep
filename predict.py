import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ─── USER CONFIG ──────────────────────────────────────────────────────────────
MODEL_DIR    = Path("/Users/tiger/Desktop/FUSEP/models/noise_reduc_oldcode")
DATA_DIR     = Path("/Users/tiger/Desktop/FUSEP")
CSV_FILE     = DATA_DIR / "id_selected_165.csv"
RGRAM_DIR    = DATA_DIR / "rgram_full"
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
from sklearn.linear_model import RANSACRegressor
import numpy as np

def analyze_block_ransac_global(
    block, c0, c1,
    resid_thresh1=12, min_inliers1=20,
    resid_thresh2=3,  min_inliers2=20
):
    """
    block: H×W binary mask for columns c0..c1
    c0, c1: global column indices of this block

    resid_thresh1: max vertical residual for layer #1
    min_inliers1 : min points needed to accept layer #1

    resid_thresh2: max vertical residual for layer #2
    min_inliers2 : min points needed to attempt & accept layer #2

    Returns: n_layers, (a1,b1), (a2,b2) or None, distance
    """
    rows, cols = np.where(block)
    # require enough points for even the first layer
    if len(rows) < min_inliers1:
        return 0, None, None, np.nan

    # global x‐coords
    xg = cols + c0
    pts = np.column_stack([xg, rows])

    # ── fit layer #1 with its own threshold ────────────────────────────
    r1 = RANSACRegressor(residual_threshold=resid_thresh1)
    r1.fit(pts[:, [0]], pts[:, 1])
    in1 = r1.inlier_mask_
    a1, b1 = r1.estimator_.coef_[0], r1.estimator_.intercept_

    # remove layer #1 inliers and check if enough remain for #2
    rem = pts[~in1]
    if len(rem) < min_inliers2:
        return 1, (a1, b1), None, np.nan

    # ── fit layer #2 with its own threshold ────────────────────────────
    r2 = RANSACRegressor(residual_threshold=resid_thresh2)
    r2.fit(rem[:, [0]], rem[:, 1])
    a2, b2 = r2.estimator_.coef_[0], r2.estimator_.intercept_

    # distance at midpoint
    x_mid = (c0 + c1) / 2
    y1m   = r1.predict([[x_mid]])[0]
    y2m   = r2.predict([[x_mid]])[0]
    dist  = abs(y2m - y1m)

    return 2, (a1, b1), (a2, b2), dist

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, T = load_model(device)

    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR/"plots").mkdir(exist_ok=True)

    df    = pd.read_csv(CSV_FILE)
    names = (df["ProductId"]
               .str.replace(r"^s_","",regex=True)
               .str.replace(r"_rgram$","",regex=True))

    all_stats = []
    for name in names:
        print(f"\n▶️ Processing {name}…")
        # load radargram
        try:
            X = np.loadtxt(RGRAM_DIR/f"{name}_rgram.txt", dtype=np.float32)
        except Exception as e:
            print(f"⚠️  Couldn’t load {name}_rgram.txt: {e}")
            continue

        # predict mask
        Xn   = zscore_cols(X)
        mask = predict_mask(model, Xn, T, device)

        # save mask
        np.savetxt(OUTPUT_DIR/f"{name}_mask.txt", mask, fmt="%d", delimiter="\t")

        # setup plot
        fig, ax = plt.subplots(1,3,figsize=(18,5), sharex=True, sharey=True)
        vmin, vmax = np.percentile(X, 0.5), np.percentile(X, 99.5)
        ax[0].imshow(X, cmap="gray", vmin=vmin, vmax=vmax, origin="upper", aspect="auto", interpolation="nearest" )
        ax[0].set_title("Radargram")

        # middle panel: mask only
        ax[1].imshow(mask, cmap="Reds", origin="upper", aspect="auto", interpolation="nearest" )
        ax[1].set_title("Mask Only")

        # right panel: mask + RANSAC lines
        ax[2].imshow(mask, cmap="Reds", alpha=0.4, origin="upper", aspect="auto", interpolation="nearest" )
        ax[2].set_title("Mask + Lines")

        # Layer analysis with RANSAC and plotting layer lines
        H, W = mask.shape
        col_ranges = [(i, min(i+BLOCK_SIZE-1, W-1)) for i in range(0, W, BLOCK_SIZE)]

        stats = []
        for c0, c1 in col_ranges:
            block = mask[:, c0:c1+1]
            n, l1, l2, dist = analyze_block_ransac_global(block, c0, c1)

            # plot lines
            x_block = np.arange(c0, c1+1)
            if n >= 1:
                a1, b1 = l1
                y1 = a1*x_block + b1
                ax[2].plot(x_block, y1, color='blue', linewidth=2)
            if n == 2:
                a2, b2 = l2
                y2 = a2*x_block + b2
                ax[2].plot(x_block, y2, color='green', linewidth=2)

            stats.append({
                "col_start": c0, "col_end": c1,
                "n_layers": n,
                "center1": l1 and (a1*c0 + b1),   # approx center at block start
                "center2": l2 and (a2*c0 + b2),
                "distance": dist
            })

        # collect stats
        for idx, s in enumerate(stats):
            s["name"] = name
            s["block_idx"] = idx
            all_stats.append(s)

        print(f"✅ Finished {name}: {len(stats)} blocks")
        
        for idx, s in enumerate(stats):
            print(f"   Block {idx:02d} cols {s['col_start']}-{s['col_end']}: {s['n_layers']} layer(s)")

        # Save the plot with lines overlaid
        for a in ax:
            a.axis("off")
        fig.suptitle(name, fontsize=14)
        fig.savefig(OUTPUT_DIR/"plots"/f"{name}_overlay.pdf", dpi=300)
        plt.close(fig)

    # Write out summary CSV
    stats_df = pd.DataFrame(all_stats,
        columns=["name", "block_idx", "col_start", "col_end", "n_layers", "center1", "center2", "distance"])

    stats_df.to_csv(OUTPUT_DIR/"layer_stats.csv", index=False)

    for name, grp in stats_df.groupby("name"):
        out_fn = OUTPUT_DIR/f"{name}_layer_stats.csv"
        grp.to_csv(out_fn, index=False)
        print(f"Wrote {out_fn.name}")

    print(f"\n🏁 Done! {len(names)} files → masks, plots + stats in {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
