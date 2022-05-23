from matplotlib import pyplot as plt

import numpy as np


def acquisition_extinction(epsilon, initial_w, acq_r, ex_r):
    w = np.zeros((200, 1))
    # in all the 200 trials, we have the same stimuli = 1
    w[0] = initial_w
    for i in range(100):
        v = w[i - 1] * 1  # v = w u
        w[i] = w[i - 1] + epsilon * (acq_r - v) * 1

    for i in range(100, 200):
        v = w[i - 1] * 1  # v = w u
        w[i] = w[i - 1] + epsilon * (ex_r - v) * 1

    plt.plot(np.linspace(0, 199, num=200), w)
    plt.xlabel('trial number')
    plt.ylabel('w')
    plt.suptitle('Evolution of the weight w over 200 trials in Acquisition & Extinction Conditioning')
    plt.show()


acquisition_extinction(0.05, 0, 1, 0)

import random


def partial(alpha, initial_w, r, epsilon):
    w = np.zeros((200, 1))
    # in all the 200 trials, we have the same stimuli = 1
    w[0] = initial_w
    for i in range(1, 200):
        v = w[i - 1] * 1  # v = w u
        if random.random() < alpha:
            w[i] = w[i - 1] + epsilon * (r - v) * 1
        else:
            w[i] = w[i - 1] + epsilon * (0 - v) * 1

    plt.plot(np.linspace(0, 199, num=200), w)
    plt.xlabel('trial number')
    plt.ylabel('w')
    plt.suptitle(
        'Evolution of the weight w over 200 trials in Partial Conditioning, alpha=' + str(alpha) + ' and r=' + str(r))
    plt.show()
    return w[199]


# %%
w_final = partial(0.5, 0, 1, 0.05)
print('final w value after 200 trials with alpha being 0.5 : ', str(w_final))

w_final = partial(0.8, 0, 1, 0.05)
print('final w value after 200 trials with alpha being 0.8 : ', str(w_final))

w_final = partial(0.5, 0, 2, 0.05)
print('final w value after 200 trials with alpha being 0.5 and reward being 2 ', str(w_final))


def blocking():
    w1 = np.zeros((200, 1))
    w2 = np.zeros((200, 1))
    w1[0] = 1
    for i in range(1, 200):
        v1 = w1[i - 1] * 1  # v = w u
        v2 = w2[i - 1] * 1
        v = v1 + v2
        w1[i] = w1[i - 1] + 0.05 * (1 - v) * 1
        w2[i] = w2[i - 1] + 0.05 * (1 - v) * 1

    plt.plot(np.linspace(0, 199, num=200), w1, 'r')
    plt.plot(np.linspace(0, 199, num=200), w2, 'b')
    plt.xlabel('trial number')
    plt.ylabel('w')
    plt.suptitle(
        'Evolution of the weight w over 200 trials in blocking conditioning with red one corresponding to the weights of s1 and blue to s2')
    plt.show()


# %%
blocking()


def inhibitory():
    w1 = np.zeros((400, 1))  # s1
    w2 = np.zeros((400, 1))  # s1 + s2
    for i in range(1, 400):
        if random.random() < 0.5:
            v1 = w1[i - 1] * 1  # v = w u
            v2 = w2[i - 1] * 0
            v = v1 + v2
            w1[i] = w1[i - 1] + 0.05 * (1 - v) * 1
            w2[i] = w2[i - 1] + 0.05 * (1 - v) * 0
        else:
            v1 = w1[i - 1] * 1  # v = w u
            v2 = w2[i - 1] * 1
            v = v1 + v2
            w1[i] = w1[i - 1] + 0.05 * (0 - v) * 1
            w2[i] = w2[i - 1] + 0.05 * (0 - v) * 1

    plt.plot(np.linspace(0, 399, num=400), w1, 'r')
    plt.plot(np.linspace(0, 399, num=400), w2, 'b')
    plt.xlabel('trial number')
    plt.ylabel('w')
    plt.suptitle('Evolution of the weight w over 400 trials in inhibitory conditioning with red one corresponding to '
                 'the weights of s1 and blue to s2')
    plt.show()


# %%

# %%
inhibitory()


def overshadow(epsilon1, epsilon2):
    w1 = np.zeros((200, 1))  # s1
    w2 = np.zeros((200, 1))  # s1 + s2
    for i in range(1, 200):
        v1 = w1[i - 1] * 1  # v = w u
        v2 = w2[i - 1] * 1
        v = v1 + v2
        w1[i] = w1[i - 1] + epsilon1 * (1 - v) * 1
        w2[i] = w2[i - 1] + epsilon2 * (1 - v) * 1

    plt.plot(np.linspace(0, 199, num=200), w1, 'r')
    plt.plot(np.linspace(0, 199, num=200), w2, 'b')
    plt.xlabel('trial number')
    plt.ylabel('w')
    plt.suptitle(
        'Evolution of the weight w over 200 trials in overshadow conditioning with red one corresponding to the weights of s1 and blue to s2')
    plt.show()


# %%
overshadow(0.3, 0.3)
overshadow(0.1, 0.3)
