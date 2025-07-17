from pathlib import Path
import numpy as np

SRC_ROOT  = Path("/Users/tiger/Desktop/FUSEP")
DEST_ROOT = Path("/Users/tiger/Desktop/FUSEP/data")

for txt in (SRC_ROOT / "rgram").glob("*_rgram.txt"):
    outfile = DEST_ROOT / "rgram" / (txt.stem + ".npy")
    if outfile.exists():
        print(f"[SKIP] {outfile.name} already exists")
        continue

    arr = np.loadtxt(txt, dtype=np.float32)
    np.save(DEST_ROOT / "rgram" / (txt.stem + ".npy"), arr)
    print (f"saved {txt.stem}.npy")

for txt in (SRC_ROOT / "reloc_01").glob("*_reloc_01.txt"):
    outfile = DEST_ROOT / "reloc_01" / (txt.stem + ".npy")
    if outfile.exists():
        print(f"[SKIP] {outfile.name} already exists")
        continue

    arr = np.loadtxt(txt, dtype=np.float32)
    np.save(DEST_ROOT / "reloc_01" / (txt.stem + ".npy"), arr)
    print (f"saved {txt.stem}.npy")