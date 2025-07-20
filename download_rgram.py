import os
import rasterio
import numpy as np
from pathlib import Path
import pywt
import pandas as pd

def scale(data):  # Perform linear transformation
    # Clip negative values to avoid issues with log10
    img_scale = np.log10(np.clip(data + 1e-30, 1e-30, np.inf))  # Clip to avoid invalid log
    img_vaild = img_scale[data != 0]  # Extract valid values
    p10 = np.percentile(img_vaild, 10)  # Calculate percentile value
    m = 255 / (img_vaild.max() - p10)
    b = - p10 * m
    img_map = (m * img_scale) + b

    # Handle NaNs or invalid values
    img_map[np.isnan(img_map)] = 0  # Replace NaNs with 0

    img_map[img_map < 0] = 0  # Ensure all values are non-negative
    img_uint = img_map.astype(np.uint8)  # Convert to uint8
    return img_uint  # Generates 8-bit unsigned data, 0-255

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
    path_rgram = os.path.join(path, 'rgram', 's_' + name + '.lbl')
    data_rgram = rasterio.open(path_rgram)
    img_rgram = data_rgram.read()[0]  #(1, 3600, m)
    
    if True in np.isnan(img_rgram):
        mask = np.isnan(img_rgram)
        img_rgram = np.where(mask, 0, img_rgram)
    
    # Apply wavelet denoising
    img_rgram_denoised = np.apply_along_axis(dwt_denoise_trace, 0, img_rgram)  # 沿列（trace）进行去噪
    
    # Apply median filtering after wavelet denoising
    img_rgram_denoised = median_filter_denoise(img_rgram_denoised, size=2)
    
    return scale(img_rgram_denoised)
    

root         = "/Volumes/data/mars/sharad/"
output_dir   = Path("/Users/tiger/Desktop/FUSEP/rgram_full")

# Read the CSV file to get the radargram names
csv_file = pd.read_csv('/Users/tiger/Desktop/FUSEP/id_selected.csv')
names = csv_file['ProductId'].str.replace('s_', '')  # Remove "s_" prefix

for name in names:
    outfile = output_dir / f"{name}.txt"

    if outfile.exists():
        print(f"[SKIP] {outfile.name} already exists")
        continue

    try:
        full_rgram = read_radar(root, name)
        
        # Save the radargram data
        np.savetxt(outfile, full_rgram, delimiter="\t", fmt="%.8f")
        print(f"{name}: saved radargram data →  shape {full_rgram.shape}")

    except:
        print(f"{name}_rgram.lbl does not exist")
