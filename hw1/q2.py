
import numpy as np
from matplotlib import pyplot as plt

dt = 0.01  # Simulation time step
Duration = 100  # Simulation length
T = int(np.ceil(Duration / dt))
t = np.arange(1, T + 1) * dt  # Simulation time points in ms
vRest = -60  # in mv, resting potential
vThresh = 30  # in mv
V = vRest * np.ones(T)  # Vector of output voltage
U = np.zeros(T)


def plot_pattern(a, b, c, d, h):
    U[0] = b * V[0]
    I = h * np.heaviside((t - 10), 0)

    for i in range(1, len(t)):
        dv = (0.04 * V[i - 1] ** 2) + (5 * V[i - 1]) + 140 - U[i - 1] + I[i - 1]
        du = a * (b * V[i - 1] - U[i - 1])

        V[i] = V[i - 1] + dv * dt
        U[i] = U[i - 1] + du * dt

        if V[i] > vThresh:
            # spike
            V[i] = c
            U[i] += d

    plt.plot(t, V)
    plt.show()
    return U, V


# Tonic Spiking: a = 0.02, b = 0.20, c = -65, d = 2, h = 15
plot_pattern(0.02, 0.20, -65, 2, 15)
# Phasic Spiking: a = 0.02, b = 0.25, c = -65, d = 6, h = 1
plot_pattern(0.02, 0.25, -65, 6, 1)
# Tonic Bursting: a = 0.02, b = 0.20, c = -50, d = 2, h = 15
plot_pattern(0.02, 0.2, -50, 2, 15)
# Phasic Bursting: a = 0.02, b = 0.25, c = -55, d = 0.05, h = 0.6
plot_pattern(0.02, 0.25, -55, 0.05, 0.6)
# Mixed Model: a = 0.02, b = 0.20, c = -55, d = 4, h = 10
plot_pattern(0.02, 0.2, -55, 4, 10)

# 4:
# Phasic Spiking: a = 0.02, b = 0.25, c = -65, d = 6, h = 1
u, v = plot_pattern(0.02, 0.25, -65, 6, 1)
plt.plot(u, v)
plt.show()
plt.plot(t, v)
plt.plot(t, u)
plt.show()



