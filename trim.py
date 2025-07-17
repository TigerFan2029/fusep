import numpy as np
import os

def trim_radargram_height(input_file, output_file, min_row, max_row):
    rgram = np.loadtxt(input_file)

    if min_row < 0 or max_row > rgram.shape[0] or min_row >= max_row:
        raise ValueError("Invalid trimming bounds.")

    trimmed = rgram[min_row:max_row, :]

    np.savetxt(output_file, trimmed, delimiter="\t", fmt="%.8f")

    print(f"Trimmed radargram saved to: {output_file}")
    print(f"New shape: {trimmed.shape}")

type = "reloc_01"
input_file = f"/Users/tiger/Desktop/FUSEP/{type}/00176903_{type}.txt"
output_file = f"/Users/tiger/Desktop/FUSEP/processed/00176903_{type}_t.txt"
trim_radargram_height(input_file, output_file, min_row=2000, max_row=3000)


