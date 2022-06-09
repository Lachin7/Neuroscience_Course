from matplotlib import pyplot as plt
import numpy as np
import scipy.io

data = scipy.io.loadmat('Q3_data.mat')
spike_times = data['Spike_times']
stim = data['Stim']
print(spike_times.shape)
print(stim.shape)

plt.plot(np.linspace(0, 1, 2000), stim[0, 0:2000])
plt.show()

random_indices = np.random.choice(spike_times.shape[0], size=20, replace=False)
for indx in random_indices:
    spike_time = spike_times[indx]
    plt.title("index of the spike: " + str(indx))
    stim_spike = stim[0, int((spike_time - 0.075) * 2000):int(spike_time * 2000)]
    plt.plot(np.linspace(-75, 0, stim_spike.shape[0]), stim_spike)
    plt.show()

stim_75 = np.zeros((598, 149))
i = 0
for row in spike_times:
    spike_time = row[0]
    stim_75[i, :] = stim[0, int((spike_time - 0.075) * 2000):int(spike_time * 2000)][0:149]
    i += 1

spike_triggered_average = np.mean(stim_75, axis=0)
plt.plot(np.linspace(-75, 0, spike_triggered_average.shape[0]), spike_triggered_average)
plt.show()

for indx in random_indices:
    spike_time = spike_times[indx]
    plt.title("index of the spike: " + str(indx))
    stim_spike = stim[0, int((spike_time - 0.075) * 2000):int(spike_time * 2000)]
    plt.plot(np.linspace(-75, 0, stim_spike.shape[0]), stim_spike)
    plt.plot(np.linspace(-75, 0, spike_triggered_average.shape[0]), spike_triggered_average, c='r', linewidth=2.5)
    plt.show()


