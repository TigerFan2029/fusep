import os
import rasterio
import numpy as np
from pathlib import Path
import pywt
import pandas as pd

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

def dwt_denoise_img(img, wavelet="db4", level=None):
    # 对整个图像进行小波去噪：按行处理（即每一行都会被去噪）
    coeffs = pywt.wavedec2(img, wavelet=wavelet, level=level)
    
    # 如果是多层分解，coeffs[0] 是近似系数，coeffs[1:] 是细节系数
    cA = coeffs[0]
    details = coeffs[1:]
    
    # 使用多层分解后的细节系数（水平、垂直、对角线）进行去噪
    for detail in details:
        cH, cV, cD = detail  # 细节系数
        sigma = np.median(np.abs(cD)) / 0.6745  # 计算噪声的标准差
        n = img.size
        thr = sigma * np.sqrt(2 * np.log(n))  # 设置阈值

        # 对细节系数进行软阈值去噪
        cH = pywt.threshold(cH, thr, mode="soft")
        cV = pywt.threshold(cV, thr, mode="soft")
        cD = pywt.threshold(cD, thr, mode="soft")
    
    # 使用去噪后的系数重构图像
    new_coeffs = [cA] + details  # 组合去噪后的系数
    y = pywt.waverec2(new_coeffs, wavelet)  # 重构图像
    return y

def read_radar(path, name):
    path_rgram = os.path.join(path, 'rgram', 's_' + name + '.lbl')
    data_rgram = rasterio.open(path_rgram)
    img_rgram = data_rgram.read()[0]  #(1, 3600, m)
    
    if True in np.isnan(img_rgram):
        mask = np.isnan(img_rgram)
        img_rgram = np.where(mask, 0, img_rgram)
    
    # Apply wavelet denoising
    img_rgram_denoised = dwt_denoise_img(img_rgram)
    
    # Apply median filtering after wavelet denoising
    img_rgram_denoised = median_filter_denoise(img_rgram_denoised, size=3)

    # Apply Gaussian filtering after wavelet denoising
    img_rgram_denoised = gaussian_filter_denoise(img_rgram_denoised, sigma=1)
    
    return scale(img_rgram_denoised)
    

root         = "/Volumes/data/mars/sharad/"
output_dir   = Path("/Users/tiger/Desktop/FUSEP/rgram_full")

# Read the CSV file to get the radargram names
csv_file = pd.read_csv('/Users/tiger/Desktop/FUSEP/id_selected_165.csv')
names = csv_file['ProductId'].str.replace('s_', '')  # Remove "s_" prefix

expected = set(names)
for f in output_dir.glob("*.txt"):
    if f.stem not in expected:
        try:
            f.unlink()
            print(f"[DELETE] {f.name} (not in current CSV)")
        except Exception as e:
            print(f"[WARN] Could not delete {f.name}: {e}")

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

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Script stopped by user")
        exit(0)

    except Exception:
        print(f"{name}_rgram.lbl does not exist")
    
