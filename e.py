import numpy as np
import matplotlib.pyplot as plt
import pywt

file_path = "/Users/tiger/Desktop/FUSEP/rgram/05698701_rgram.txt"
radar = np.loadtxt(file_path)          # shape (samples, traces)
print(radar.shape)

trace = radar[:, 0]                    # 取第一条迹

# 小波参数（可调）
wavelet = "db4"
level = None   # None -> pywt 会给最大可能分解层数；你也可以设 4、5 等

# DWT 分解
coeffs = pywt.wavedec(trace, wavelet=wavelet, level=level)

# 估计噪声 sigma：通常用最高频（细节系数最后一个）中位绝对偏差
detail_last = coeffs[-1]
sigma = np.median(np.abs(detail_last)) / 0.6745

# 通用软阈值（VisuShrink）
n = trace.size
universal_thresh = 100 #sigma * np.sqrt(2 * np.log(n))

# 对所有 detail 系数做软阈值（保留近似系数 coeffs[0]）
new_coeffs = [coeffs[0]]
for c in coeffs[1:]:
    new_c = pywt.threshold(c, universal_thresh, mode="soft")
    new_coeffs.append(new_c)

# 重构
trace_denoised = pywt.waverec(new_coeffs, wavelet=wavelet)

# 长度可能因重构微调，与原长度对齐（截断或填充）
trace_denoised = trace_denoised[:n]

plt.plot(trace, label="orig", alpha=0.7)
plt.plot(trace_denoised, label="dwt denoised")
plt.legend()
plt.show()
