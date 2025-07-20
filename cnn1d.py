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

DATA_ROOT   = Path("/Users/tiger/Desktop/FUSEP/data")
NUM_EPOCHS  = 15
BATCH_SIZE  = 128
LR          = 1e-3
MODEL_PATH  = Path("/Users/tiger/Desktop/FUSEP/models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

BEST_PATH   = MODEL_PATH / "best.pt"
FINAL_PATH  = MODEL_PATH / "final.pt"

n = os.cpu_count()
torch.set_num_threads(n-2)
torch.set_num_interop_threads(n-2)

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

    def forward(self, x):
        z = self.cnn(x)
        z = F.interpolate(z, size=x.shape[-1], mode="linear", align_corners=False)
        return z

#loss function
class DiceBCELoss(nn.Module):
    def __init__(self, pos_weight, bce_weight=0.4, smooth=1):
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

def make_splits(pairs, val_frac=0.1, test_frac=0.1, seed=42):
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

class MultiRadarColumnDataset(Dataset):
    #Treat every column of every radargram as an independent sample
    def __init__(self, pairs):
        self.blocks      = []   # xy numpy list
        self.col_offsets = []
        col_total = 0

        for x_path, y_path in pairs:
            X = np.load(x_path).astype(np.float32)
            Y = np.load(y_path).astype(np.float32)
            if X.shape != Y.shape:
                raise ValueError("shape mismatch:", x_path)

            #normalise
            mean = X.mean(axis=0, keepdims=True)
            std  = X.std(axis=0, keepdims=True) + 1e-6
            X    = (X - mean) / std

            self.blocks.append((X, Y))
            col_total += X.shape[1]
            self.col_offsets.append(col_total)

        self.depth = self.blocks[0][0].shape[0]  #rows

    #need torch.utils.data.Dataset
    def __len__(self):
        return self.col_offsets[-1]

    #indexing
    def _lookup(self, global_col):
        block_idx = np.searchsorted(self.col_offsets, global_col, side="right")
        prev_cols = 0 if block_idx == 0 else self.col_offsets[block_idx - 1]
        local_col = global_col - prev_cols
        return block_idx, local_col

    def __getitem__(self, idx):
        b, c = self._lookup(idx)
        X, Y = self.blocks[b]          #(depth, width)
        col_x = torch.from_numpy(X[:, c])[None]  # add (1, depth)
        col_y = torch.from_numpy(Y[:, c])[None]
        return col_x, col_y

#loops
def run_epoch(model, loader, loss_fn, optimizer, device, threshold=0.5):
    is_train = optimizer is not None
    model.train(is_train)

    loss_sum, correct = 0.0, 0
    total_dice, total_iou = 0.0, 0.0
    total_precision, total_recall = 0.0, 0.0

    for X, y in tqdm(loader, leave=False):
        X = X.to(device)
        y = y.to(device)

        if is_train:
            optimizer.zero_grad()

        logits = model(X)
        loss  = loss_fn(logits, y)

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

    n_pixels = len(loader.dataset) * loader.dataset.depth
    avg_dice = total_dice / len(loader)
    avg_iou = total_iou / len(loader)

    avg_precision = total_precision / len(loader)
    avg_recall = total_recall / len(loader)

    return loss_sum / len(loader), correct / n_pixels, avg_precision, avg_recall,  avg_dice, avg_iou

#train
def train():
    pairs = collect_pairs(DATA_ROOT)
    tr_pairs, va_pairs, _ = make_splits(pairs)

    ds_train = MultiRadarColumnDataset(tr_pairs)
    ds_val   = MultiRadarColumnDataset(va_pairs)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pos_weight_value = 150.0
    pos_weight = torch.tensor([pos_weight_value], device=device)
    print(f"Using pos_weight = {pos_weight_value:.2f}")

    train_dl = DataLoader(ds_train, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=6, 
                          prefetch_factor=4, persistent_workers=True)
    
    val_dl   = DataLoader(ds_val,   batch_size=BATCH_SIZE * 2,
                          num_workers=6,
                          prefetch_factor=4, persistent_workers=True)

    model   = PeakPicker1D().to(device)
    loss_fn = DiceBCELoss(pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(opt, mode="min",
                                factor=0.5,
                                patience=1)

    best_val = float("inf")

    train_losses = []
    val_losses = []

    patience = 3
    no_improve = 0

    for epoch in range(NUM_EPOCHS):
        tr_loss, tr_acc, tr_precision, tr_recall, tr_dice, tr_iou = run_epoch(model, train_dl, loss_fn, opt, device)
        va_loss, va_acc, va_precision, va_recall, va_dice, va_iou = run_epoch(model, val_dl, loss_fn, None, device, threshold=0.5)
        scheduler.step(va_loss)

        print(f"[{epoch:02}/{NUM_EPOCHS}] "
            f"train {tr_loss:.4f} acc {tr_acc:.3f} precision {tr_precision:.3f} recall {tr_recall:.3f} dice {tr_dice:.3f} iou {tr_iou:.3f} | "
            f"val {va_loss:.4f} acc {va_acc:.3f} precision {va_precision:.3f} recall {va_recall:.3f} dice {va_dice:.3f} iou {va_iou:.3f}")

        train_losses.append(tr_loss)
        val_losses.append(va_loss)

        if va_loss < best_val:
            best_val = va_loss
            no_improve = 0
            torch.save(model.state_dict(), BEST_PATH)
            print(f"   ↳ new best saved → {BEST_PATH}")
        
        else:
            no_improve += 1
            print(f"   ↳ no improvement for {no_improve}/{patience} epochs")
            if no_improve >= patience:
                print(f"Stopping early at epoch {epoch}")
                break

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

    ds_test = MultiRadarColumnDataset(test_pairs)
    test_dl = DataLoader(
        ds_test,
        batch_size=BATCH_SIZE * 2,
        num_workers=6,
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
    pw_value = 150.0
    pos_weight = torch.tensor([pw_value], device=device)
    print(f"pos_weight for test = {pw_value:.2f}")

    model = PeakPicker1D().to(device)
    model.load_state_dict(torch.load(BEST_PATH, map_location=device))
    model.eval()
    print("✓ Loaded best checkpoint from", BEST_PATH)

    #test
    loss_fn = DiceBCELoss(pos_weight)
    best_T = float(np.load(MODEL_PATH / "best_threshold.npy"))
    te_loss, te_acc, tr_precision, tr_recall, tr_dice, tr_iou = run_epoch(model, test_dl, loss_fn, None, device, threshold=best_T)
    print(f"TEST  loss {te_loss:.4f}, acc {te_acc:.3f} precision {tr_precision:.3f} recall {tr_recall:.3f} dice {tr_dice:.3f} iou {tr_iou:.3f}")

    #visualization
    for blk_idx, (X_norm, _) in enumerate(ds_test.blocks):
        H, W = X_norm.shape
        X_batch = torch.from_numpy(X_norm.T)      # (W, H)
        X_batch = X_batch.unsqueeze(1).to(device) # (W, 1, H)

        logits = model(X_batch)                   # (W,1,H)
        probs  = torch.sigmoid(logits)            # (W,1,H)
        pred_mask = (probs > best_T).cpu().numpy().T

        fig, ax = plt.subplots(1, 2, figsize=(12, 6),
                               sharex=True, sharey=True)
        ax[0].imshow(X_norm, cmap="gray", origin="lower")
        ax[0].set_title(f"Radargram {blk_idx}  (amplitude)")

        #ax[1].imshow(X_norm, cmap="gray", origin="lower")
        ax[1].imshow(pred_mask.squeeze(), cmap="Reds", alpha=0.45, origin="lower")
        ax[1].set_title("Predicted mask overlay")
        plt.tight_layout()
        plt.show()

        if blk_idx == 2:
            break

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

def calculate_precision_recall(preds, target, threshold=0.5):
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test"],
                        default="train")
    args = parser.parse_args()

    if args.mode == "train":
        train()
    else:
        test()