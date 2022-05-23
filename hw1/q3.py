import matplotlib.pyplot as plt
import numpy as np

# some constants here:
dt = 0.01  # Simulation time step
Duration = 20  # Simulation length
T = int(np.ceil(Duration / dt))
t = np.arange(1, T + 1) * dt  # Simulation time points in ms

# Q1: calculating rho(t):
V_th = -45
noise_range = 22
V_t = np.linspace(V_th - noise_range, V_th + noise_range, T)
x = V_t - V_th
beta = 1 / 2


def calc_rho(gama):
    return beta * (1 + np.tanh(x * gama))


fig, ax = plt.subplots(nrows=2, ncols=3)
ax[0, 0].plot(calc_rho(5))
ax[0, 1].plot(calc_rho(2))
ax[0, 2].plot(calc_rho(1))
ax[1, 0].plot(calc_rho(0.7))
ax[1, 1].plot(calc_rho(0.5))
ax[1, 2].plot(calc_rho(0.25))
plt.show()


# calculate PDF:
def calc_pdf(cdf):
    res = np.zeros(len(cdf))
    res[0] = cdf[0]
    for i in range(1, len(cdf)):
        res[i] = cdf[i] - cdf[i - 1]
    return res / np.sum(res)


fig, ax = plt.subplots(nrows=2, ncols=3)
ax[0, 0].plot(calc_pdf(calc_rho(5)))
ax[0, 1].plot(calc_pdf(calc_rho(2)))
ax[0, 2].plot(calc_pdf(calc_rho(1)))
ax[1, 0].plot(calc_pdf(calc_rho(0.7)))
ax[1, 1].plot(calc_pdf(calc_rho(0.5)))
ax[1, 2].plot(calc_pdf(calc_rho(0.25)))
plt.show()

# Q2:
V_rest = -70
V = V_rest * np.ones(T)
spike_val = 35


def simulate(RI, V_rest, tau_m):
    pdf = calc_pdf(calc_rho(0.7))
    spike_Vs = []  # voltages that a spike occurred in them
    for i in range(1, len(t)):
        # dv/dt = 1/tau_m (-v(t) + RI)
        V[i] = V[i - 1] + dt * (1 / tau_m) * (- (V[i - 1] - V_rest) + RI)
        th = np.random.choice(np.linspace(V_th - noise_range, V_th + noise_range, T), p=pdf)
        if V[i] > th:
            spike_Vs.append(V[i])
            V[i] = V_rest
            V[i - 1] = spike_val
    return V, spike_Vs


def plot_LIF(RI, V_rest, tau_m):
    V, spike_Vs = simulate(RI, V_rest, tau_m)
    plt.plot(t, V)
    plt.show()


plot_LIF(20, -70, 1.5)


# Q3:
V, spike_Vs = simulate(45, -70, 2)
plt.hist(spike_Vs)
plt.show()


# Q4:

def calc_firing_rate(I, duration):
    V, spike_Vs = simulate(I, -70, 2)
    return len(spike_Vs) / (duration / 1000)


I_arr = np.linspace(2, 80, 80)
F = np.zeros(I_arr.shape)
for i in range(80):
    F[i] = calc_firing_rate(I_arr[i], Duration)

plt.plot(I_arr, F)
plt.show()

