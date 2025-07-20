import matplotlib.pyplot as plt
import numpy as np

def plot_radargram(img):
    vmin = np.percentile(img, 1)
    vmax = np.percentile(img, 99)

    plt.figure(figsize=(10, 6))
    im = plt.imshow(img, cmap="gray", vmin=vmin, vmax=vmax, origin="upper")
    plt.colorbar(im, label="Power / Intensity")
    plt.xlabel("Along-track sample")
    plt.ylabel("Time delay sample")

    #plt.gca().invert_yaxis()
    plt.title(f"Radargram {img.shape}")
    plt.tight_layout()
    plt.show()

# path = "/Volumes/data/mars/sharad"
# name = "04581501"

# radar_img = read_radar(path, name)
# plot_radargram(radar_img)

file_path = "/Users/tiger/Desktop/FUSEP/data_whole_3_1/rgram/00357601_rgram.txt"
radar_data = np.loadtxt(file_path)
plot_radargram(radar_data)