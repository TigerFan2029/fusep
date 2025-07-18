from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score
from cnn1d_h import SimpleFCN2D, preprocess_rgram, extract_horizon
import cv2



MODEL_DIR = Path("/Users/tiger/Desktop/FUSEP/models")
DATA_DIR = Path("/Users/tiger/Desktop/FUSEP/data")
FILE = "00588302"
X_FILE = DATA_DIR / f"rgram/{FILE}_rgram.npy"
Y_FILE = DATA_DIR / f"reloc_01/{FILE}_reloc_01.npy"

def load_model(device: torch.device):
    model = SimpleFCN2D().to(device)
    model.load_state_dict(torch.load(MODEL_DIR / "best.pt", map_location=device))
    model.eval()
    thresh = float(np.load(MODEL_DIR / "best_threshold.npy"))
    return model, thresh

def zscore_cols(arr: np.ndarray):
    μ = arr.mean(axis=0, keepdims=True)
    σ = arr.std(axis=0, keepdims=True) + 1e-6
    return (arr - μ) / σ


def predict(model: SimpleFCN2D, x: np.ndarray, T: float, device):
    # 1) denoise & clip/stretch exactly as during training
    x = preprocess_rgram(x.astype(np.float32))
    # 2) per-column z-score
    μ = x.mean(axis=0, keepdims=True)
    σ = x.std(axis=0, keepdims=True) + 1e-6
    x_norm = (x - μ) / σ

    # 3) inference
    inp    = torch.from_numpy(x_norm[None]).unsqueeze(0).to(device)  # (1,1,H,W)
    with torch.no_grad():
        logits = model(inp)                                          # (1,1,H,W)
        prob   = torch.sigmoid(logits).cpu().numpy()[0,0]            # (H,W)

    # 4) threshold + median filter
    mask = (prob > T).astype(np.uint8)
    mask = cv2.medianBlur(mask, ksize=5)

    return mask


def main():
    dev = torch.device("mps" if torch.backends.mps.is_available() else
                       "cuda" if torch.cuda.is_available() else "cpu")
    model, best_T = load_model(dev)

    X = np.load(X_FILE)
    Y = np.load(Y_FILE) if Y_FILE.exists() else None

    pred_mask = predict(model, X, best_T, dev)

    if Y is not None:
        P = precision_score(Y.flatten(), pred_mask.flatten(), zero_division=0)
        R = recall_score   (Y.flatten(), pred_mask.flatten(), zero_division=0)
        metric_txt = f"(P={P:.3f}, R={R:.3f})"
        print(f"Precision: {P:.3f}, Recall: {R:.3f}")
    else:
        metric_txt = "(no ground-truth)"

    fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    vmin, vmax = np.percentile(X, 1), np.percentile(X, 99)
    ax[0].imshow(X, cmap="gray", vmin=vmin, vmax=vmax, origin="upper")
    ax[0].set_title("Radargram amplitude")
    if Y is not None:
        ax[1].imshow(Y, cmap="Reds", origin="upper")
        ax[1].set_title("Ground-truth mask")
    else:
        ax[1].text(0.5, 0.5, "no label", ha="center", va="center")
        ax[1].set_title("Ground-truth mask")

    # 4) plot only the extracted horizon
    # ax[2].imshow(X, cmap="gray", vmin=vmin, vmax=vmax, origin="upper")
    pts = extract_horizon(pred_mask, min_votes=1)  # Nx2 array: (col, row)
    if pts.size:
        ax[2].plot(pts[:,0], pts[:,1], '-r', linewidth=1.5)
    ax[2].set_title("Extracted horizon")

    for a in ax:
        a.axis("off")
    fig.suptitle(f"{X_FILE.name} {metric_txt}", fontsize=14)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
