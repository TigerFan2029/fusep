import os
import re
import rasterio
import numpy as np
#import pds4_tools
from pathlib import Path
import pywt

def scale(data):  # 进行线性变换的过程
    # Clip negative values to avoid issues with log10
    img_scale = np.log10(np.clip(data + 1e-30, 1e-30, np.inf))  # Clip to avoid invalid log
    img_vaild = img_scale[data != 0]  # 提取有效值
    p10 = np.percentile(img_vaild, 10)  # 计算分位数值
    m = 255 / (img_vaild.max() - p10)
    b = - p10 * m
    img_map = (m * img_scale) + b

    # Handle NaNs or invalid values
    img_map[np.isnan(img_map)] = 0  # Replace NaNs with 0

    img_map[img_map < 0] = 0  # Ensure all values are non-negative
    img_uint = img_map.astype(np.uint8)  # Convert to uint8
    return img_uint  # 生成的是8位无整型的数据，0-255

from scipy.ndimage import gaussian_filter

def gaussian_filter_denoise(img, sigma=1):
    return gaussian_filter(img, sigma=sigma)  # `sigma` controls the amount of smoothing (larger value = more smoothing)

from scipy.ndimage import median_filter

def median_filter_denoise(img, size=3):
    return median_filter(img, size=size)  # `size` controls the filter window, e.g., 3x3, 5x5

def dwt_denoise_trace(x, wavelet="db4", level=None):
    coeffs = pywt.wavedec(x, wavelet=wavelet, level=level)
    detail_last = coeffs[-1]
    sigma = np.median(np.abs(detail_last)) / 0.6745
    n = x.size
    thr = sigma * np.sqrt(2 * np.log(n))
    new_coeffs = [coeffs[0]]
    for c in coeffs[1:]:
        new_coeffs.append(pywt.threshold(c, thr, mode="soft"))
    y = pywt.waverec(new_coeffs, wavelet=wavelet)
    return y[:n]

def read_radar(path, name):
    path_rgram = os.path.join(path, 'rgram', 's_' + name + '_rgram.lbl')
    data_rgram = rasterio.open(path_rgram)
    img_rgram = data_rgram.read()[0]  #(1, 3600, m)
    
    if True in np.isnan(img_rgram):
        mask = np.isnan(img_rgram)
        img_rgram = np.where(mask, 0, img_rgram)
    
    # Apply wavelet denoising
    img_rgram_denoised = np.apply_along_axis(dwt_denoise_trace, 0, img_rgram)  # 沿列（trace）进行去噪
    
    # Apply median filtering after wavelet denoising
    # img_rgram_denoised = median_filter_denoise(img_rgram_denoised, size=2)
    
    # Apply Gaussian filtering after wavelet denoising
    # img_rgram_denoised = gaussian_filter_denoise(img_rgram_denoised, sigma=0.2)
    
    return scale(img_rgram_denoised)


root         = "/Volumes/data/mars/sharad/"
reloc_path   = Path("/Users/tiger/Desktop/FUSEP/reloc")
output_dir   = Path("/Users/tiger/Desktop/FUSEP/test")

# for reloc_file in reloc_path.glob("*_reloc.txt"):
reloc_file = Path("/Users/tiger/Desktop/FUSEP/reloc/00357601_reloc.txt")
name = reloc_file.stem.replace("_reloc", "")
outfile = output_dir / f"{name}_rgram.txt"

# if outfile.exists():
#     print(f"[SKIP] {outfile.name} already exists")
#     continue

blocks = []
current = []
with open(reloc_file, "r") as f:
    for line in f:
        stripped = line.strip()
        if not stripped:
            if current:
                blocks.append(current)
                current = []
        else:
            try:
                col = int(stripped.split()[0])
                current.append(col)
            except ValueError:
                pass
    if current:
        blocks.append(current)

# if len(blocks) < 2:
#     print(f"[SKIP] {name}: only {len(blocks)} layer(s) found")
#     continue

second = blocks[1]
c_min = min(second) - 1
c_max = max(second) - 1
# if c_min < 0:
#     print(f"[WARN] {name}: second block min index < 1 after conversion, skipping")
#     continue

full_rgram = read_radar(root, name)
keep_rgram = full_rgram[:, c_min : c_max + 1]
np.savetxt(outfile, keep_rgram, delimiter="\t", fmt="%.8f")
print(f"{name}: saved second block cols {c_min}-{c_max}  →  shape {keep_rgram.shape}")