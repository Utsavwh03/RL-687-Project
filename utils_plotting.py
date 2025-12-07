import matplotlib.pyplot as plt
import numpy as np

def plot_curve(values, path, title):
    plt.figure(figsize=(9,5))
    plt.plot(values, alpha=0.4, label="Raw")
    
    # Moving average window
    if len(values) > 50:
        ma = np.convolve(values, np.ones(50)/50, mode='valid')
        plt.plot(ma, label="Moving Avg (50 steps)", linewidth=2)
    
    plt.title(title)
    plt.xlabel("Step" if len(values) > 1000 else "Episode")
    plt.ylabel(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(path)
    plt.close()
