# visualisation helpers for data
import numpy as np
import torch
try:
    import seaborn as sns
except ImportError:
    sns = None
import matplotlib.pyplot as plt
import matplotlib
if sns is not None:
    sns.set()

def array_plot(points, filepath):
    data = points[0]
    period = len(points) + 1
    a = np.zeros((period*data.shape[0], data.shape[1]))
    a[period*np.array(range(data.shape[0])),:] = data
    if period > 2:
        recon = points[1]
        a[period*np.array(range(data.shape[0]))+1,:] = recon
    if sns is not None:
        sns.heatmap(a, linewidth=0.5, vmin=-1, vmax=1, cmap=sns.color_palette('RdBu_r', 100))
    else:
        plt.imshow(a, aspect="auto", vmin=-1, vmax=1, cmap="RdBu_r")
        plt.colorbar()
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.clf()
