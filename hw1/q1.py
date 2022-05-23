import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp

dt = 0.01  # Simulation time step
Duration = 200  # Simulation length
T = int(np.ceil(Duration / dt))
t = np.arange(1, T + 1) * dt  # Simulation time points in ms
Cm = 1  # Membrane capacitance in micro Farads
gNa = 120  # in Siemens, maximum conductivity of Na+ Channel
gK = 36  # in Siemens, maximum conductivity of K+ Channel
gl = 0.3  # in Siemens, conductivity of leak Channel
ENa = 55  # in mv, Na+ nernst potential
EK = -72  # in mv, K+ nernst potential
El = -49.4  # in mv, nernst potential for leak channel
vRest = -60  # in mv, resting potential
V = vRest * np.ones(T)  # Vector of output voltage
I = np.zeros(T)  # in uA, external stimulus (external current)


def alpha_n(v): return ((0.1 * (-v + vRest) + 1) / 10) / (np.exp(1 + 0.1 * (-v + vRest)) - 1)


def beta_n(v): return 0.125 * np.exp((-v + vRest) / 80)


def alpha_m(v): return (((-v + vRest) + 25) / 10) / (np.exp(2.5 + 0.1 * (-v + vRest)) - 1)


def beta_m(v): return 4 * np.exp((-v + vRest) / 18)


def alpha_h(v): return 0.07 * np.exp((-v + vRest) / 20)


def beta_h(v): return 1 / (1 + np.exp(3 + 0.1 * (-v + vRest)))


# Q1:
vt = np.arange(-80, 10, 0.01)
time_cons_n = 1 / (alpha_n(vt) + beta_n(vt))
time_cons_m = 1 / (alpha_m(vt) + beta_m(vt))
time_cons_h = 1 / (alpha_h(vt) + beta_h(vt))

steady_state_n = alpha_n(vt) / (alpha_n(vt) + beta_n(vt))
steady_state_m = alpha_m(vt) / (alpha_m(vt) + beta_m(vt))
steady_state_h = alpha_h(vt) / (alpha_h(vt) + beta_h(vt))


fig, ax = plt.subplots(1, 2)

ax[0].plot(vt, time_cons_n, 'r')
ax[0].plot(vt, time_cons_m, 'b')
ax[0].plot(vt, time_cons_h, 'g')

ax[1].plot(vt, steady_state_n, 'r')
ax[1].plot(vt, steady_state_m, 'b')
ax[1].plot(vt, steady_state_h, 'g')
plt.show()


# Q2:
def HH_model(I):
    n, m, h = np.zeros(len(t)), np.zeros(len(t)), np.zeros(len(t))
    n[0], m[0], h[0] = alpha_n(V[0]) / (alpha_n(V[0]) + beta_n(V[0])), alpha_m(V[0]) / (
            alpha_m(V[0]) + beta_m(V[0])), alpha_h(V[0]) / (
                               alpha_h(V[0]) + beta_h(V[0]))

    for i in range(1, len(t)):
        m[i] = m[i - 1] + dt * (alpha_m(V[i - 1]) * (1 - m[i - 1]) - beta_m(V[i - 1]) * m[i - 1])
        n[i] = n[i - 1] + dt * (alpha_n(V[i - 1]) * (1 - n[i - 1]) - beta_n(V[i - 1]) * n[i - 1])
        h[i] = h[i - 1] + dt * (alpha_h(V[i - 1]) * (1 - h[i - 1]) - beta_h(V[i - 1]) * h[i - 1])

        I_L = gl * (V[i - 1] - El)
        I_K = (gK * np.power(n[i - 1], 4)) * (V[i - 1] - EK)
        I_Na = (gNa * h[i - 1] * np.power(m[i - 1], 3)) * (V[i - 1] - ENa)

        V[i] = V[i - 1] + dt * ((-1 / Cm) * (I_L + I_K + I_Na - I[i - 1]))

    return n, m, h, V


def plot_model(I):
    n, m, h, V = HH_model(I)
    plt.plot(t, I)
    plt.plot(t, V)
    plt.show()


def plot_for_constant_I(c):
    I = np.zeros(T)  # in uA, external stimulus (external current)
    I[600:T] = c
    plot_model(I)


plot_for_constant_I(10)

# Q3:
plot_for_constant_I(7)
plot_for_constant_I(6) # so it should be between 6 and 7
plot_for_constant_I(6.1)
plot_for_constant_I(6.3) # it should be between 6.1 and 6.3
plot_for_constant_I(6.2)
plot_for_constant_I(6.22)
plot_for_constant_I(6.23)

# Q4:
def plot_for_interval_I(c):
    I = np.zeros(T)  # in uA, external stimulus (external current)
    I[600:600 + c] = 6.23
    plot_model(I)


plot_for_interval_I(100)
plot_for_interval_I(200)
plot_for_interval_I(400)
plot_for_interval_I(500)
plot_for_interval_I(600)
#

# Q5:
plot_for_constant_I(6.23)
plot_for_constant_I(6.5)
plot_for_constant_I(10)
plot_for_constant_I(20)
plot_for_constant_I(30)
plot_for_constant_I(45)

plot_for_constant_I(50)
plot_for_constant_I(50)
plot_for_constant_I(80)
plot_for_constant_I(100)
plot_for_constant_I(111)
plot_for_constant_I(140)


# Q7:
I = np.linspace(2, 160, T)
plot_model(I)

# Q8:
def plot_subPlots(c):
    I = np.zeros(T)
    I[600:T] = c
    n, m, h, V = HH_model(I)
    plot_model(I)
    fig, ax = plt.subplots(nrows=1, ncols=3)
    ax[0].plot(n, V)
    ax[1].plot(m, V)
    ax[2].plot(h, V)
    plt.show()


plot_subPlots(2)
plot_subPlots(7)
plot_subPlots(200)

# Q9:

def triangle(length, amplitude):
    section = length // 4
    for direction in (1, -1):
        for i in range(section):
            yield i * (amplitude / section) * direction
        for i in range(section):
            yield (amplitude - (i * (amplitude / section))) * direction


I = np.asarray(list(triangle(T, 20)))
plot_model(I)

sig = (np.arange(T) % 2000 < 1000) * 10
plot_model(sig)

I = chirp(np.linspace(0, Duration, T), f0=0.11, f1=0.05, t1=10, method='linear') * 7
plot_model(I)

I = np.sin(np.linspace(0, Duration, T)) * 10
plot_model(I)

# Q10
def calc_firing_rate(I, duration):
    n, m, h, V = HH_model(I)
    firing_threshold = 20
    spikes = 0
    for i in range(len(V)):
        if V[i] > firing_threshold >= V[i - 1]:
            spikes += 1
    if spikes == 1:
        spikes = 0
    return spikes / (duration / 1000)


I_arr = np.linspace(2, 80, 80)
F = np.zeros(I_arr.shape)
for i in range(80):
    I = np.ones(T) * I_arr[i]
    F[i] = calc_firing_rate(I, Duration)

plt.plot(I_arr, F)
plt.show()
