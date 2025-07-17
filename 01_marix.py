import os
import numpy as np
from pathlib import Path

def _reloc_to_indices(reloc_file, *, ignore_bad_row=True):
    cols, rows = [], []
    with open(reloc_file, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) != 2:
                continue
            c, r = map(int, parts)
            if ignore_bad_row and r == 9999:
                continue
            cols.append(c)
            rows.append(r)

    if not cols:
        raise ValueError("no valid reloc points found")

    left_edge = min(cols)
    cols = [c - left_edge          for c in cols]
    rows = [r - 1                  for r in rows]
    return rows, cols


def create_label_array(rgram_file, reloc_file):
    rgram = np.loadtxt(rgram_file)
    n_rows, n_cols = rgram.shape
    label = np.zeros((n_rows, n_cols), dtype=np.uint8)

    rows, cols = _reloc_to_indices(reloc_file)

    mask = (
        (0 <= np.array(rows)) & (np.array(rows) < n_rows) &
        (0 <= np.array(cols)) & (np.array(cols) < n_cols)
    )
    label[ np.array(rows)[mask], np.array(cols)[mask] ] = 1
    return label

root = "/Volumes/data/mars/sharad/"
rgram_path   = Path("/Users/tiger/Desktop/FUSEP/rgram")
output_dir   = Path("/Users/tiger/Desktop/FUSEP/reloc_01")

for rgram_file in rgram_path.glob("*_rgram.txt"):
    file_name=rgram_file.stem.replace("_rgram", "")
    reloc_file=f"/Users/tiger/Desktop/FUSEP/reloc/{file_name}_reloc.txt"
    outfile = output_dir / f"{file_name}_reloc_01.txt"

    if outfile.exists():
        print(f"[SKIP] {outfile.name} already exists")
        continue

    try:
        label_array = create_label_array(rgram_file, reloc_file) 
        np.savetxt(outfile, label_array, delimiter="\t", fmt="%.8f")
        print(f"{file_name}: {label_array.shape} successful")

    except Exception as exc:
        print(f"[WARN] Skipped {file_name}: {exc}")