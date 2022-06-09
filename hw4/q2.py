from matplotlib import pyplot as plt
import numpy as np
import scipy.io

data = scipy.io.loadmat('Q2_data.mat')
trials = np.array(data["trials"])


def plot_raster(num):
    for i in range(num):
        for j in range(500):
            if trials[i, j] == 1:
                x1 = [i, i + 0.5]
                x2 = [j / 2 - 50, j / 2 - 50]
                plt.plot(x2, x1, color='black')
    plt.axvline(0, 0, 1)
    plt.title('Spike raster plot')
    plt.xlabel('Times')
    plt.ylabel('Trials')
    plt.show()


plot_raster(1)
plot_raster(20)
plot_raster(100)


def plot_PETH(Dt):
    res = np.zeros((250,))
    for j in range(int(500 / Dt)):
        firing_rate = 0
        for i in range(100):
            firing_rate += np.sum(trials[i, j * Dt:j * Dt + Dt])
        res[int(j * Dt / 2):int((j * Dt + Dt) / 2)] = firing_rate
    plt.title("PETH plot for a bin size of: " + str(Dt))
    plt.xlabel("Time (ms)")
    plt.ylabel("Firing rate")
    plt.plot(np.arange(-50, 200), res)
    plt.axvline(0, c='r')
    plt.show()


plot_PETH(5)
plot_PETH(20)
plot_PETH(35)
