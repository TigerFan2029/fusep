import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import math
from sklearn.linear_model import RANSACRegressor

#config
model_name   = "PRE_denoise_Final"
MODEL_DIR    = Path(f"/Users/tiger/Desktop/FUSEP/models/{model_name}")
DATA_DIR     = Path("/Users/tiger/Desktop/FUSEP")
CSV_FILE     = DATA_DIR / "138_path.csv"
RGRAM_DIR    = DATA_DIR / "rgram_clipped"
OUTPUT_DIR   = DATA_DIR / "predictions"

BLOCK_SIZE = 4

#model
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
    return (prob > T).astype(np.uint8).squeeze().T # H×W binary mask

#layer analysis
def analyze_block_ransac_global(
    block, c0, c1,
    resid_thresh1=1, min_inliers1=4,
    resid_thresh2=1,  min_inliers2=4,
    min_distance=5, # minimum vertical separation in pixels
    max_slope_deg=60 # max allowed slope in degrees
):
    rows, cols = np.where(block)
    if len(rows) < min_inliers1:
        return 0, None, None, 0

    # Convert to global x-coordinates
    xg = cols + c0
    pts = np.column_stack([xg, rows])  # (N,2)

    # find the topmost layer for layer 1 by scanning from the top of the block
    r1 = RANSACRegressor(residual_threshold=resid_thresh1, max_trials=1000)
    try:
        r1.fit(pts[:, [0]], pts[:, 1])
    except ValueError as e:
        print(f"⚠️ RANSAC failed for layer 1: {e}")
        return 0, None, None, 0

    # Check the slope of the first layer
    a1, b1 = r1.estimator_.coef_[0], r1.estimator_.intercept_
    slope_deg1 = math.degrees(math.atan(abs(a1)))
    
    if slope_deg1 > max_slope_deg:
        print(f"⚠️ Layer 1 slope {slope_deg1:.2f}° exceeds {max_slope_deg}° threshold, ignoring this layer.")
        return 0, None, None, np.nan
    
    # Calculate vertical position at the midpoint
    x_mid = (c0 + c1) / 2
    y1m  = r1.predict([[x_mid]])[0]
    
    if len(r1.inlier_mask_) < min_inliers1:
        return 0, None, None, np.nan

    # check for the second layer using the outliers of the first
    rem = pts[~r1.inlier_mask_]
    if len(rem) < min_inliers2:
        return 1, (a1, b1), None, 0

    # Fit second line with its own threshold
    r2 = RANSACRegressor(residual_threshold=resid_thresh2, max_trials=1000)
    try:
        r2.fit(rem[:, [0]], rem[:, 1])
    except ValueError as e:
        print(f"⚠️ RANSAC failed for layer 2: {e}")
        return 1, (a1, b1), None, 0
    
    a2, b2 = r2.estimator_.coef_[0], r2.estimator_.intercept_

    # Calculate slope for second layer
    slope_deg2 = math.degrees(math.atan(abs(a2)))
    if slope_deg2 > max_slope_deg:
        print(f"⚠️ Layer 2 slope {slope_deg2:.2f}° exceeds {max_slope_deg}° threshold, ignoring this layer.")
        return 1, (a1, b1), None, 0

    # Calculate vertical separation between the two layers at the block’s midpoint
    y2m  = r2.predict([[x_mid]])[0]
    dist = abs(y2m - y1m)

    # Ensure layer 1 is above layer 2
    if y1m > y2m:
        a1, b1, a2, b2 = a2, b2, a1, b1
        slope_deg1, slope_deg2 = slope_deg2, slope_deg1

        dist = abs(y2m - y1m)

    if dist < min_distance:
        print(f"⚠️ Layers are too close (distance = {dist:.2f} px), returning 1 layer.")
        return 1, (a1, b1), None, 0

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

        if (OUTPUT_DIR / f"{name}_mask.txt").exists():
            print(f"⚠️ Skipping {name}, mask already exists.")
            continue
    
        # load radargram
        try:
            X = np.loadtxt(RGRAM_DIR/f"{name}_rgram.txt", dtype=np.float32)
        except Exception as e:
            print(f"⚠️  Couldn’t load {name}_rgram.txt: {e}")
            continue

        # predict mask
        Xn   = zscore_cols(X)
        mask = predict_mask(model, Xn, T, device)

        np.savetxt(OUTPUT_DIR/f"{name}_mask.txt", mask, fmt="%d", delimiter="\t")

        fig, ax = plt.subplots(
            1, 3, figsize=(15, 5),
            sharex=True, sharey=True,
            gridspec_kw={"width_ratios": [1, 1, 1]},
            constrained_layout=True)
        vmin, vmax = np.percentile(X, 0), np.percentile(X, 100)
        ax[0].imshow(X, cmap="gray", vmin=vmin, vmax=vmax, origin="upper", aspect="auto")
        ax[0].set_title("Radargram")

        ax[1].imshow(mask, cmap="Reds", origin="upper", aspect="auto")
        ax[1].set_title("Mask Only")

        ax[2].imshow(mask, cmap="Reds", alpha=0.2, origin="upper", aspect="auto")
        ax[2].set_title("Mask + Lines")

        # Layer analysis with RANSAC
        H, W = mask.shape
        col_ranges = [(i, min(i+BLOCK_SIZE-1, W-1)) for i in range(0, W, BLOCK_SIZE)]

        stats = []
        for c0, c1 in col_ranges:
            block = mask[:, c0:c1+1]
            n, l1, l2, dist = analyze_block_ransac_global(block, c0, c1)

            x_block = np.arange(c0, c1+1)
            if n == 1:
                a1, b1 = l1
                y1 = a1 * x_block + b1
                ax[2].plot(x_block, y1, color='blue', linewidth=1)
            elif n == 2:
                a1, b1 = l1
                y1 = a1 * x_block + b1
                ax[2].plot(x_block, y1, color='blue', linewidth=1)
                
                a2, b2 = l2
                y2 = a2 * x_block + b2
                ax[2].plot(x_block, y2, color='green', linewidth=1)


            stats.append({
                "col_start": c0, "col_end": c1,
                "n_layers": n,
                "center1": l1 and (a1 * c0 + b1),
                "center2": l2 and (a2 * c0 + b2),
                "distance": dist})

        # collect stats
        for idx, s in enumerate(stats):
            s["name"] = name
            s["block_idx"] = idx
            all_stats.append(s)

        print(f"✅ Finished {name}: {len(stats)} blocks")
        
        for idx, s in enumerate(stats):
            print(f"   Block {idx:02d} cols {s['col_start']}-{s['col_end']}: {s['n_layers']} layer(s)")

        stats_df = pd.DataFrame(stats,
            columns=["name", "block_idx", "col_start", "col_end", "n_layers", "center1", "center2", "distance"])

        # Save individual CSV
        out_fn = OUTPUT_DIR / f"{name}_layer_stats.csv"
        stats_df.to_csv(out_fn, index=False)
        print(f"Wrote {out_fn.name}")

        for a in ax:
            for spine in a.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1)
                spine.set_color("black")
            a.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        fig.suptitle(f"{name}_rgram.txt model={model_name}", fontsize=14)

        fig.savefig(OUTPUT_DIR/"plots"/f"{name}_overlay.pdf", dpi=300)
        # plt.show()
        # plt.close(fig)
        # break

    all_stats_df = pd.concat([pd.read_csv(OUTPUT_DIR / f"{name}_layer_stats.csv") for name in names], ignore_index=True)

    # final summary CSV
    all_stats_df.to_csv(OUTPUT_DIR / "layer_stats.csv", index=False)
    print(f"Wrote final layer_stats.csv")
    print(f"\n🏁 Done! {len(names)} files → masks, plots + stats in {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
