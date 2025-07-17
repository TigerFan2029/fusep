import argparse
import re
from pathlib import Path
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy.ndimage import median_filter, gaussian_filter

DATA_ROOT   = Path("/Users/tiger/Desktop/FUSEP/data")
NUM_EPOCHS  = 10
BATCH_SIZE  = 4
LR          = 1e-3
MODEL_PATH  = Path("/Users/tiger/Desktop/FUSEP/models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

BEST_PATH   = MODEL_PATH / "best.pt"
FINAL_PATH  = MODEL_PATH / "final.pt"

n = os.cpu_count()
torch.set_num_threads(n-2)
torch.set_num_interop_threads(n-2)

#model
class SimpleFCN2D(nn.Module):
    def __init__(self, in_ch=1, base_ch=16):
        super().__init__()
        # Encoder: anisotropic → square convs
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, base_ch,    kernel_size=(3,11), padding=(1,5)),
            nn.BatchNorm2d(base_ch), nn.ReLU(),
            nn.MaxPool2d(2)   # ↓H/2, W/2
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_ch*2), nn.ReLU(),
            nn.MaxPool2d(2)   # ↓H/4, W/4
        )
        # Bottleneck: larger RF via dilation
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_ch*2, base_ch*2, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(base_ch*2), nn.ReLU()
        )
        # Decoder: simple up-sampling
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(base_ch*2, base_ch, kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(base_ch, 1, kernel_size=2, stride=2)
        )

    def forward(self, x):
        # x: (B,1,H,W)
        e1 = self.enc1(x)          # (B,base,H/2,W/2)
        e2 = self.enc2(e1)         # (B,2base,H/4,W/4)
        b  = self.bottleneck(e2)   # same shape
        d1 = self.dec1(b)          # (B,base,H/2,W/2)
        d2 = self.dec2(d1)         # (B,1,H,W)

        d2 = F.interpolate(d2, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)
        
        return d2


#loss function
class DiceBCELoss(nn.Module):
    def __init__(self, pos_weight, bce_weight=0.5, smooth=1):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.bce_weight = bce_weight
        self.smooth = smooth

    def forward(self, z, target):
        pred = torch.sigmoid(z)
        pred_f = pred.view(-1)
        target_f = target.view(-1)

        #dice
        inter = (pred_f * target_f).sum()
        dice = (2 * inter + self.smooth) / (pred_f.sum() + target_f.sum() + self.smooth)
        dice_loss = 1 - dice

        #bce
        bce_loss = self.bce(z, target)

        return (1 - self.bce_weight) * dice_loss + self.bce_weight * bce_loss

#data prep
def collect_pairs(root, x_key="_rgram.npy", y_key="_reloc_01.npy"):
    pattern = re.compile(rf"(.+){re.escape(x_key)}$")
    pairs   = []

    x_dir = root / "rgram"
    y_dir = root / "reloc_01"

    for x_file in x_dir.rglob(f"*{x_key}"):
        match = pattern.match(x_file.name)
        if not match:
            continue
        stem   = match.group(1)
        y_file = y_dir / (stem + y_key)
        if y_file.exists():
            pairs.append((x_file, y_file))
        else:
            print("[skip] label missing for", x_file)
    if not pairs:
        raise RuntimeError("No rgram / reloc_01 pairs found!")
    print("Found", len(pairs), "radargrams")
    print(f"epoch = {NUM_EPOCHS}")
    print(f"batch size = {BATCH_SIZE}")
    print(f"learning rate = {LR}")
    return pairs

def make_splits(pairs, val_frac=0.15, test_frac=0.15, seed=42):
    train_pairs, tmp_pairs = train_test_split(
        pairs,
        test_size=val_frac + test_frac,
        shuffle=True,
        random_state=seed
    )
    val_pairs, test_pairs = train_test_split(
        tmp_pairs,
        test_size=test_frac / (val_frac + test_frac),
        shuffle=True,
        random_state=seed
    )
    return train_pairs, val_pairs, test_pairs


def preprocess_rgram(img, p_lo=1, p_hi=99):
    # intensity clip / stretch
    vmin, vmax = np.percentile(img, (p_lo, p_hi))
    img = np.clip(img, vmin, vmax)
    img = (img - vmin) / (vmax - vmin + 1e-6)

    # speckle suppression – preserve sharp horizontal edges
    img = median_filter(img, size=(1, 5))
    img = gaussian_filter(img, sigma=(0, 0.8))

    return img.astype(np.float32)

class RadargramDataset(Dataset):
    def __init__(self, pairs):
        self.images, self.masks = [], []
        for x_path, y_path in pairs:
            X = preprocess_rgram(np.load(x_path).astype(np.float32))
            Y = (np.load(y_path) > 0.5).astype(np.float32)
            self.images.append(X)
            self.masks.append(Y)
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        # (1, H, W) tensors for image and binary mask
        return ( torch.from_numpy(self.images[idx])[None],
                 torch.from_numpy(self.masks[idx])[None] )

#loops
def run_epoch(model, loader, loss_fn, optimizer, device, threshold=0.7):
    is_train = optimizer is not None
    model.train(is_train)

    loss_sum, correct = 0.0, 0
    total_dice, total_iou = 0.0, 0.0
    total_precision, total_recall = 0.0, 0.0

    for X, Y in loader:
        X, y = X.to(device), Y.to(device)
        if is_train:
            optimizer.zero_grad()
        logits = model(X)
        base_loss = loss_fn(logits, y)

        pred = torch.sigmoid(logits)
        dh = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :]).mean()
        λ = 0.1   # tune this weight

        loss = base_loss + λ * dh


        #Dice and IoU calculation
        pred_binary = (torch.sigmoid(logits) > threshold)
        intersection = (pred_binary & (y > 0.5)).sum()
        union = (pred_binary | (y > 0.5)).sum()
        dice = (2*intersection + 1e-6) / (pred_binary.sum() + y.sum() + 1e-6)
        iou = intersection / (union + 1e-6)
        total_dice += dice.item()
        total_iou += iou.item()

        precision, recall = calculate_precision_recall(logits, y, threshold)
        total_precision += precision
        total_recall += recall

        if is_train:
            loss.backward()
            optimizer.step()

        loss_sum += loss.item()
        correct += ((torch.sigmoid(logits) > threshold) == (y > 0.5)).sum().item()

    avg_dice = total_dice / len(loader)
    avg_iou = total_iou / len(loader)

    avg_precision = total_precision / len(loader)
    avg_recall = total_recall / len(loader)

    return loss_sum / len(loader), avg_precision, avg_recall,  avg_dice, avg_iou

#train
def train():
    pairs = collect_pairs(DATA_ROOT)
    tr_pairs, va_pairs, _ = make_splits(pairs)

    ds_train = RadargramDataset(tr_pairs)
    ds_val   = RadargramDataset(va_pairs)

    # class weighting calc
    # total_pos = sum(block[1].sum() for block in ds_train.blocks)
    # total_pix = len(ds_train) * ds_train.depth
    # total_neg = total_pix - total_pos
    # pw_value  = total_neg / (total_pos + 1e-6)
    # device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # pos_weight = torch.tensor([pw_value], device=device)
    # print(f"Using pos_weight = {pw_value:.2f}")

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pos_weight_value = 50.0
    pos_weight = torch.tensor([pos_weight_value], device=device)
    print(f"Using pos_weight = {pos_weight_value:.2f}")

    train_dl = DataLoader(ds_train, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=3, collate_fn=pad_collate,
                          prefetch_factor=4, persistent_workers=True)
    
    val_dl   = DataLoader(ds_val,   batch_size=BATCH_SIZE * 2,
                          num_workers=3, collate_fn=pad_collate,
                          prefetch_factor=4, persistent_workers=True)

    get_device()

    model = SimpleFCN2D().to(device)
    loss_fn = DiceBCELoss(pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(opt, mode="min",
                                factor=0.5,
                                patience=1)

    best_val = float("inf")

    train_losses = []
    val_losses = []

    for epoch in range(NUM_EPOCHS):
        tr_loss, tr_precision, tr_recall, tr_dice, tr_iou = run_epoch(model, train_dl, loss_fn, opt, device)
        va_loss, va_precision, va_recall, va_dice, va_iou = run_epoch(model, val_dl, loss_fn, None, device, threshold=0.7)
        scheduler.step(va_loss)

        print(f"[{epoch:02}/{NUM_EPOCHS}] "
            f"train {tr_loss:.4f}  precision {tr_precision:.3f} recall {tr_recall:.3f} dice {tr_dice:.3f} iou {tr_iou:.3f} | "
            f"val {va_loss:.4f} precision {va_precision:.3f} recall {va_recall:.3f} dice {va_dice:.3f} iou {va_iou:.3f}")

        train_losses.append(tr_loss)
        val_losses.append(va_loss)

        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), BEST_PATH)
            print(f"   ↳ new best saved → {BEST_PATH}")

    torch.save(model.state_dict(), FINAL_PATH)
    print(f"Final model saved to {FINAL_PATH}")

    model.load_state_dict(torch.load(BEST_PATH, map_location=device))
    model.eval()

    best_T = sweep_threshold(model, val_dl, device)
    np.save(MODEL_PATH / "best_threshold.npy", best_T)
    print(f"Best T = {best_T}")

    plot_loss_curve(train_losses, val_losses)

mpl.rcParams["image.aspect"] = "auto"


@torch.no_grad()
def test():
    pairs = collect_pairs(DATA_ROOT)
    _, _, test_pairs = make_splits(pairs)

    ds_test = RadargramDataset(test_pairs)
    test_dl = DataLoader(
        ds_test,
        batch_size=BATCH_SIZE * 2,
        num_workers=3, 
        collate_fn=pad_collate,
        prefetch_factor=4,
        persistent_workers=True
    )
    #pin_memory=True
    
    #weight calc
    total_pos = sum(block[1].sum() for block in ds_test.blocks)
    total_pix = len(ds_test) * ds_test.depth
    total_neg = total_pix - total_pos
    #pw_value  = total_neg / (total_pos + 1e-6)          # >1 if rare
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pw_value = 50.0
    pos_weight = torch.tensor([pw_value], device=device)
    print(f"pos_weight for test = {pw_value:.2f}")

    get_device()

    model = SimpleFCN2D().to(device)
    model.load_state_dict(torch.load(BEST_PATH, map_location=device))
    model.eval()
    print("✓ Loaded best checkpoint from", BEST_PATH)

    #test
    loss_fn = DiceBCELoss(pos_weight)
    best_T = float(np.load(MODEL_PATH / "best_threshold.npy"))
    te_loss, tr_precision, tr_recall, tr_dice, tr_iou = run_epoch(model, test_dl, loss_fn, None, device, threshold=best_T)
    print(f"TEST  loss {te_loss:.4f}, precision {tr_precision:.3f} recall {tr_recall:.3f} dice {tr_dice:.3f} iou {tr_iou:.3f}")

    #visualization
    for blk_idx, (X_norm, Y_norm) in enumerate(zip(ds_test.images, ds_test.masks)):
        H, W = X_norm.shape
        X_batch = torch.from_numpy(X_norm.T)      # (W, H)
        X_batch = X_batch.unsqueeze(1).to(device) # (W, 1, H)

        logits = model(X_batch)                   # (W,1,H)
        probs     = torch.sigmoid(logits)
        pred_mask = (probs > best_T).cpu().numpy().squeeze().T  # shape (H, W)

        # new: extract horizon as one point per strong column
        horizon_pts = extract_horizon(pred_mask, min_votes=30)  # (N, 2) array

        # plotting
        fig, ax = plt.subplots(1, 2, figsize=(12, 6),
                               sharex=True, sharey=True)
        ax[0].imshow(X_norm, cmap="gray", origin="lower")
        ax[0].set_title(f"Radargram {blk_idx}  (amplitude)")

        ax[1].imshow(pred_mask, cmap="Reds", alpha=0.45, origin="lower")
        # overlay the extracted polyline:
        if horizon_pts.size:
            cols, rows = horizon_pts[:,0], horizon_pts[:,1]
            ax[1].plot(cols, rows, '-o', markersize=2, linewidth=1.5)
        ax[1].set_title("Predicted mask + extracted horizon")
        plt.tight_layout()
        plt.show()

def plot_loss_curve(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(train_losses)), train_losses, label='Training Loss', color='blue')
    plt.plot(range(len(val_losses)), val_losses, label='Validation Loss', color='red')
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

def calculate_precision_recall(preds, target, threshold=0.7):
    #calculate precision and recall
    pred_binary = (torch.sigmoid(preds) > threshold).cpu().numpy().flatten()
    target_binary = target.cpu().numpy().flatten()

    precision = precision_score(target_binary, pred_binary, zero_division=1)
    recall = recall_score(target_binary, pred_binary, zero_division=1)

    return precision, recall

@torch.no_grad()
def sweep_threshold(model, val_dl, device, metric="dice", steps=11):
    model.eval()
    best_score, best_t = -1, 0.5
    Ts = np.linspace(0.2, 0.8, steps)

    for T in Ts:
        scores = []
        for X, y in val_dl:
            X, y = X.to(device), y.to(device)
            prob = torch.sigmoid(model(X))
            pred = (prob > T)
            inter = (pred & (y > 0.5)).sum()
            union = pred.sum() + y.sum()
            dice = (2 * inter) / (union + 1e-6)
            scores.append(dice.item())

        score = np.mean(scores)
        if score > best_score:
            best_score, best_t = score, T

    print(f"✓  Best threshold {best_t:.2f} → Dice {best_score:.3f}")
    return best_t

def extract_horizon(mask, min_votes=30):
    """
    Run a Hough‐style voting on the binary mask and return the best
    (column, row) points of the horizon as an N×2 array.
    
    mask : 2-D numpy array, shape (H, W), dtype=bool or {0,1}
    min_votes : int, minimum positives in a column to count that column
    """
    ys, xs = np.where(mask > 0)
    # count hits per column
    votes, _ = np.histogram(xs, bins=mask.shape[1])
    # select columns with enough votes
    strong_cols = np.where(votes > min_votes)[0]
    # for each, take median row index
    median_y = [int(np.median(ys[xs == c])) for c in strong_cols]
    return np.column_stack([strong_cols, median_y])

def pad_collate(batch):
    # batch is a list of (X, Y) pairs, each X: (1, H, Wi), Y: (1, H, Wi)
    Xs, Ys = zip(*batch)
    max_w = max(x.shape[2] for x in Xs)

    Xs_pad = [F.pad(x, (0, max_w - x.shape[2]), "constant", 0) for x in Xs]
    Ys_pad = [F.pad(y, (0, max_w - y.shape[2]), "constant", 0) for y in Ys]

    return torch.stack(Xs_pad, 0), torch.stack(Ys_pad, 0)

def get_device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available():          return torch.device("cuda")
    return torch.device("cpu")

device = get_device()
print(f"→ Using {device}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test"],
                        default="train")
    args = parser.parse_args()

    if args.mode == "train":
        train()
    else:
        test()
