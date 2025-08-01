import numpy as np
from pathlib import Path

def _reloc_to_indices(reloc_file, *, ignore_bad_row=True):
    # split file into blocks separated by blank lines
    blocks, cur = [], []
    with open(reloc_file, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                if cur:
                    blocks.append(cur)
                    cur = []
                continue
            parts = s.split()
            if len(parts) != 2:
                continue
            try:
                c, r = map(int, parts)
            except ValueError:
                continue
            if ignore_bad_row and r == 9999:
                continue
            cur.append((c, r))
    if cur:
        blocks.append(cur)

    if len(blocks) < 2:
        raise ValueError("reloc file has fewer than 2 blocks (layers)")

    second_cols = [c for c, _ in blocks[1]]
    c_min_file  = min(second_cols)
    c_max_file  = max(second_cols)
    c_min_0     = c_min_file - 1 # 1-based indices in reloc file

    rows, cols = [], []
    for block in blocks:
        for c, r in block:
            if not (c_min_file <= c <= c_max_file):
                continue
            rows.append(r - 1) # 0-based row
            cols.append(c - 1 - c_min_0)  # 0-based col inside crop

    if not cols:
        raise ValueError("no picks lie inside 2nd-block window")

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

rgram_path   = Path("/Users/tiger/Desktop/FUSEP/rgram_full")
output_dir   = Path("/Users/tiger/Desktop/FUSEP/reloc_01_full")

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