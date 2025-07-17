import os
import re
import rasterio
import numpy as np
#import pds4_tools
from pathlib import Path

def scale(data):  # 进行线性变换的过程
    img_scale = np.log10(data + 1e-30)  # 转换成对数
    img_vaild = img_scale[data != 0]  # 提取有效值
    p10 = np.percentile(img_vaild, 10)  # 计算分位数值
    m = 255 / (img_vaild.max() - p10)  #
    b = - p10 * m
    img_map = (m * img_scale) + b
    img_map[img_map < 0] = 0
    img_uint = img_map.astype(np.uint8)
    return img_uint  # 生成的是8位无整型的数据，0-255

def read_radar(path, name):
    path_rgram = os.path.join(path, 'rgram', 's_' + name + '_rgram.lbl')
    data_rgram = rasterio.open(path_rgram)
    img_rgram = data_rgram.read()[0]  #(1,3600,m) .transpose((1, 2, 0))
    # 把雷达图中的nana值转换成0
    if True in np.isnan(img_rgram):
        mask = np.isnan(img_rgram)
        img_rgram = np.where(mask, 0, img_rgram)
    return scale(img_rgram)


def read_simu(path, name):
    # 打开simu文件
    path_sim = os.path.join(path, 'simu', 's_' + name + '_sim.xml')
    data_sim = pds4_tools.read(path_sim, quiet=True)  # quiet为True不输出警告
    img_sim = data_sim[2].data  # Left, Right, Combined
    return img_sim


def read_geom(path, name):
    lat, lon = [], []
    # 打开geom文件
    path_geom = os.path.join(path, 'geom', 's_' + name + '_geom.tab')
    with open(path_geom, 'r') as file:
        lines = file.readlines()

    for line in lines:
        tmp = re.split(',', line)
        lat.append(float(tmp[2]))
        lon360 = float(tmp[3])  #Change coordinates from 0-360 to -180 to 180 (consistent with ArcGIS)
        if lon360 > 180:
            lon360 = lon360 - 360
        lon.append(lon360)

    return [lon, lat]

#Example use:
root         = "/Volumes/data/mars/sharad/"
reloc_path   = Path("/Users/tiger/Desktop/FUSEP/reloc")
output_dir   = Path("/Users/tiger/Desktop/FUSEP/rgram")

for reloc_file in reloc_path.glob("*_reloc.txt"):
    name = reloc_file.stem.replace("_reloc", "")
    outfile = output_dir / f"{name}_rgram.txt"

    if outfile.exists():
        print(f"[SKIP] {outfile.name} already exists")
        continue

    # --- parse into blocks separated by blank lines ---
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

    # --- require at least two blocks ---
    if len(blocks) < 2:
        print(f"[SKIP] {name}: only {len(blocks)} layer(s) found")
        continue

    # --- take extent of the second block, convert to 0-based ---
    second = blocks[1]
    c_min = min(second) - 1
    c_max = max(second) - 1
    if c_min < 0:
        print(f"[WARN] {name}: second block min index < 1 after conversion, skipping")
        continue

    # --- extract and save ---
    full_rgram = read_radar(root, name)
    keep_rgram = full_rgram[:, c_min : c_max + 1]
    np.savetxt(outfile, keep_rgram, delimiter="\t", fmt="%.8f")
    print(f"{name}: saved second block cols {c_min}-{c_max}  →  shape {keep_rgram.shape}")

