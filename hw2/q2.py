from matplotlib import pyplot as plt
import numpy as np

# some constants here:
dt = 0.01  # Simulation time step
Duration = 200  # Simulation length
T = int(np.ceil(Duration / dt))
t = np.arange(1, T + 1) * dt  # Simulation time points in ms

V_rest = -80
E_L = -70
tau_m = 20
IR = 25
r_m = 100
V_th = -54

E_s_exc = 0
E_s_inh = -80
spike_val = 25
tau_peak = 10


def simulate(E_s, p, rho_val, spiking_length, v_initial):
    V = v_initial * np.ones(T)
    for i in range(1, len(t)):
        gt = g(i, p, rho_val, spiking_length)
        V[i] = ((1 / tau_m) * (-((V[i - 1] - E_L) + gt * (V[i - 1] - E_s) * r_m) + IR)) * dt + V[i - 1]
        if V[i] > V_th:
            V[i] = V_rest
            V[i - 1] = spike_val
    return V


def g(i, p, val, spiking_length):
    rho = np.zeros(T)
    kt = np.arange(0, Duration, dt)
    # calculate the spike train with a period of p
    for j in range(T):
        if j % p == 0:
            rho[j: j + spiking_length] = val
    return np.sum(np.roll(np.flipud(K(np.arange(0, Duration, dt))), i) * rho) / 1000


def K(t):
    return (t / tau_peak) * np.exp(1 - (t / tau_peak))


def plot_simulation(E_s, p, rho_val=0.01, spiking_length=50, v_initial=V_rest):
    V = simulate(E_s, p, rho_val, spiking_length, v_initial)
    plt.plot(t, V)
    plt.show()


# plot_simulation(E_s_exc, p=200)
# plot_simulation(E_s_exc, p=2000, rho_val=0.1)
# plot_simulation(E_s_inh, p=2000)
# plot_simulation(E_s_inh, p=2000, rho_val=0.1)
#
# # check the periods of 100, 1000 and 4000 on excitatory neuron:
# plot_simulation(E_s_exc, p=100)
# plot_simulation(E_s_exc, p=1000)
# plot_simulation(E_s_exc, p=4000)
#
# # check the periods of 100, 1000 and 4000 on inhibitory neuron:
# plot_simulation(E_s_inh, p=100)
# plot_simulation(E_s_inh, p=1000)
# plot_simulation(E_s_inh, p=4000)
#
# # v_initial = -20, -50, -70
# plot_simulation(E_s_exc, p=200, v_initial=20)
# plot_simulation(E_s_exc, p=200, v_initial=-60)
# plot_simulation(E_s_exc, p=200, v_initial=-70)


def simulate2(E_s, rho_val, spiking_length, v_initial_1, v_initial_2):
    V1, V2 = v_initial_1 * np.ones(T), v_initial_2 * np.ones(T)
    rho1, rho2 = np.zeros(T), np.zeros(T)
    for i in range(1, len(t)):

        gt = g2(i, rho1)
        V1[i] = ((1 / tau_m) * (-((V1[i - 1] - E_L) + gt * (V1[i - 1] - E_s) * r_m) + IR)) * dt + V1[i - 1]
        if V1[i] > V_th:
            V1[i] = V_rest
            V1[i - 1] = spike_val
            update_rho(rho2, i, rho_val, spiking_length)

        gt = g2(i, rho2)
        V2[i] = ((1 / tau_m) * (-((V2[i - 1] - E_L) + gt * (V2[i - 1] - E_s) * r_m) + IR)) * dt + V2[i - 1]
        if V2[i] > V_th:
            V2[i] = V_rest
            V2[i - 1] = spike_val
            update_rho(rho1, i, rho_val, spiking_length)
    return V1, V2


def update_rho(rho, i, val, spiking_length):
    l = i + spiking_length
    if l > T:
        l = T
    rho[i: l] = val


def g2(i, rho):
    return np.sum(np.roll(np.flipud(K(np.arange(0, Duration, dt))), i) * rho) / 1000


def plot_simulation2(E_s, rho_val=0.01, spiking_length=50, v_initial_1=-80, v_initial_2=-60):
    V1, V2 = simulate2(E_s, rho_val, spiking_length, v_initial_1, v_initial_2)
    plt.plot(t, V1)
    plt.plot(t, V2)
    plt.savefig("resQ2/"+str(E_s)+".jpg")
    plt.show()


# plot_simulation2(E_s_exc)
plot_simulation2(E_s_inh, rho_val=0.02, spiking_length=100)
