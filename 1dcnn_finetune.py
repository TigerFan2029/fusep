"""
Fine-tune the 1-D CNN on *all* rgram / reloc_01 pairs in a directory tree.
"""

import numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import re

# ------------------------------------------------------------------ #
# 0.  Re-use the model & loss from the previous file                  #
# ------------------------------------------------------------------ #
from train_sharad_cnn import PeakPicker1D, DiceBCELoss   # import earlier defs

# ------------------------------------------------------------------ #
# 1.  Helper: find all matching pairs                                #
# ------------------------------------------------------------------ #
def collect_pairs(root: Path, x_key="_rgram.npy", y_key="_reloc_01.npy"):
    pairs = []
    pattern = re.compile(rf"(.+){re.escape(x_key)}$")
    for x_file in root.rglob(f"*{x_key}"):
        m = pattern.match(x_file.name)
        if not m:
            continue
        stem = m.group(1)
        y_file = x_file.with_name(stem + y_key)
        if y_file.exists():
            pairs.append((x_file, y_file))
        else:
            print(f"⚠️  No label file for {x_file}")
    if not pairs:
        raise RuntimeError("No rgram / reloc_01 pairs found!")
    print(f"Found {len(pairs)} radargrams")
    return pairs


# ------------------------------------------------------------------ #
# 2.  Dataset that streams *columns* from *all* radargrams            #
# ------------------------------------------------------------------ #
class MultiRadarColumnDataset(Dataset):
    def __init__(self, pairs):
        self.data = []  # holds (X, Y) arrays
        self.col_offsets = []  # cumulative column counts for _global_ indexing
        col_total = 0

        for x_path, y_path in pairs:
            X = np.load(x_path).astype(np.float32)  # (H, W_i)
            Y = np.load(y_path).astype(np.float32)
            assert X.shape == Y.shape, f"Shape mismatch for {x_path}"
            # column-wise z-score
            mu  = X.mean(axis=0, keepdims=True)
            std = X.std(axis=0, keepdims=True) + 1e-6
            X   = (X - mu) / std

            self.data.append((X, Y))
            col_total += X.shape[1]
            self.col_offsets.append(col_total)

        self.H = self.data[0][0].shape[0]  # assume all have same depth

    def __len__(self):          # total number of columns across all files
        return self.col_offsets[-1]

    def _lookup(self, global_idx):
        # figure out which radargram this column belongs to
        file_idx = np.searchsorted(self.col_offsets, global_idx, side="right")
        prev_cols = 0 if file_idx == 0 else self.col_offsets[file_idx-1]
        local_col = global_idx - prev_cols
        return file_idx, local_col

    def __getitem__(self, idx):
        fidx, c = self._lookup(idx)
        X, Y = self.data[fidx]
        x = torch.from_numpy(X[:, c])[None]   # (1, H)
        y = torch.from_numpy(Y[:, c])[None]   # (1, H)
        return x, y


# ------------------------------------------------------------------ #
# 3.  Training helpers (unchanged)                                   #
# ------------------------------------------------------------------ #
def train_one_epoch(model, loader, loss_fn, opt, device):
    model.train()
    running = 0.0
    for X, y in tqdm(loader, leave=False):
        X, y = X.to(device), y.to(device)
        opt.zero_grad()
        p = model(X)
        loss = loss_fn(p, y)
        loss.backward()
        opt.step()
        running += loss.item()
    return running / len(loader)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    loss_tot, correct = 0.0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        p = model(X)
        loss_tot += loss_fn(p, y).item()
        correct += ((p > 0.5) == (y > 0.5)).sum().item()
    pixels = len(loader.dataset) * loader.dataset.H
    return loss_tot / len(loader), correct / pixels


# ------------------------------------------------------------------ #
# 4.  Main fine-tune routine                                         #
# ------------------------------------------------------------------ #
def main():
    root = Path("./data")              # folder containing many npys
    pairs = collect_pairs(root)

    ds = MultiRadarColumnDataset(pairs)
    # ─── simple 90/10 split by columns ──────────────────────────────
    val_fraction = 0.10
    val_size = int(len(ds) * val_fraction)
    train_size = len(ds) - val_size
    ds_train, ds_val = torch.utils.data.random_split(ds, [train_size, val_size])

    train_loader = DataLoader(ds_train, batch_size=128, shuffle=True)
    val_loader   = DataLoader(ds_val,   batch_size=256, shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ---------------------------------------------------------------
    # Load the *previously trained* weights
    # ---------------------------------------------------------------
    model = PeakPicker1D().to(device)
    ckpt  = Path("peakpicker_cnn.pth")
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print("✔️  Loaded existing weights")
    else:
        print("⚠️  No checkpoint found; training from scratch")

    loss_fn = DiceBCELoss()
    opt     = torch.optim.AdamW(model.parameters(), lr=3e-4)  # lower LR for fine-tune

    for epoch in range(15):            # a handful of extra epochs is usually enough
        tr = train_one_epoch(model, train_loader, loss_fn, opt, device)
        vl, acc = evaluate(model, val_loader, loss_fn, device)
        print(f"[{epoch:02}] train {tr:.4f} | val {vl:.4f} | acc {acc:.3f}")

    torch.save(model.state_dict(), "peakpicker_cnn_finetuned.pth")
    print("✅  Saved fine-tuned weights")


if __name__ == "__main__":
    main()
