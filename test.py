from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score

model = "more_data_oldcode"
MODEL_DIR = Path(f"/Users/tiger/Desktop/FUSEP/models/more_data_oldcode{model}")
DATA_DIR = Path("/Users/tiger/Desktop/FUSEP/")
FILE = "00357601"
X_FILE = DATA_DIR / f"rgram/{FILE}_rgram.txt"
Y_FILE = DATA_DIR / f"reloc_01/{FILE}_reloc_01.txt"

class PeakPicker1D(nn.Module):
    def __init__(self, in_ch: int = 1):
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

    def forward(self, x):
        z = self.cnn(x)
        return F.interpolate(z, size=x.shape[-1], mode="linear", align_corners=False)

def load_model(device: torch.device):
    model = PeakPicker1D().to(device)
    model.load_state_dict(torch.load(MODEL_DIR / "best.pt", map_location=device))
    model.eval()
    thresh = float(np.load(MODEL_DIR / "best_threshold.npy"))
    return model, thresh

def zscore_cols(arr: np.ndarray):
    μ = arr.mean(axis=0, keepdims=True)
    σ = arr.std(axis=0, keepdims=True) + 1e-6
    return (arr - μ) / σ

def predict(model: PeakPicker1D, x_norm: np.ndarray, T: float, device):
    h, w = x_norm.shape
    with torch.no_grad():
        logit = model(torch.from_numpy(x_norm.T).unsqueeze(1).to(device))
        prob  = torch.sigmoid(logit).cpu().numpy()
    return (prob > T).astype(np.uint8).squeeze().T

def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, best_T = load_model(dev)

    X = np.loadtxt(X_FILE).astype(np.float32)
    Y     = np.loadtxt(Y_FILE).astype(np.uint8) if Y_FILE.exists() else None

    X_norm    = zscore_cols(X)
    pred_mask = predict(model, X_norm, best_T, dev)

    if Y is not None:
        precision = precision_score(Y.flatten(), pred_mask.flatten(), zero_division=1)
        recall    = recall_score   (Y.flatten(), pred_mask.flatten(), zero_division=1)
        metric_txt = f" (P={precision:.3f}, R={recall:.3f})"
        print(f"Precision: {precision:.3f}, Recall: {recall:.3f}")
    else:
        metric_txt = " (no ground-truth)"

    #plot
    fig, ax = plt.subplots(
    1, 3, figsize=(15, 5),
    sharex=True, sharey=True,
    gridspec_kw={"width_ratios": [1, 1, 1]},
    constrained_layout=True)

    vmin = np.percentile(X, 0.5)
    vmax = np.percentile(X, 99.5)
    ax[0].imshow(X,         cmap="gray", vmin=vmin, vmax=vmax,
             origin="upper", aspect="auto")
    ax[0].set_title("Radargram amplitude")

    if Y is not None:
        ax[1].imshow(Y,         cmap="Reds", origin="upper",    aspect="auto")
        ax[1].set_title("Ground-truth mask")
    else:
        ax[1].text(0.5, 0.5, "no label", ha="center", va="center")
        ax[1].set_title("Ground-truth mask")

    # ax[2].imshow(X_disp, cmap="gray", origin="upper")
    ax[2].imshow(pred_mask, cmap="Reds", origin="upper",
             alpha=0.45,        aspect="auto")
    ax[2].set_title("Predicted mask overlay")

    for a in ax:
        for spine in a.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1)
            spine.set_color("black")
        a.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    fig.suptitle(f"{X_FILE.name}{metric_txt} model={model}", fontsize=14)
    plt.show()

if __name__ == "__main__":
    main()