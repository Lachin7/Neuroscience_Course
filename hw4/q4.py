import numpy as np


def linear_perceptron(w, x, mu):
    result = np.matmul(w.T, x) - mu
    if result > 0:
        return 1
    else:
        return -1



def is_point_inside(a):
    mu1, mu2, mu3, mu4 = -9, -2, -1, 55

    l1 = np.asarray([-0.5, -1]).reshape((2, 1))
    l2 = np.asarray([0.5, -1]).reshape((2, 1))
    l3 = np.asarray([-3, -1]).reshape((2, 1))
    l4 = np.asarray([-3, 1]).reshape((2, 1))
    l5 = np.asarray([0.5, 1]).reshape((2, 1))
    l6 = np.asarray([-0.5, 1]).reshape((2, 1))
    l7 = np.asarray([3, 1]).reshape((2, 1))
    l8 = np.asarray([3, -1]).reshape((2, 1))

    r1 = linear_perceptron(l1, a, mu2)
    r2 = linear_perceptron(l2, a, mu2)
    r3 = linear_perceptron(l3, a, mu1)
    r4 = linear_perceptron(l4, a, mu1)
    r5 = linear_perceptron(l5, a, mu2)
    r6 = linear_perceptron(l6, a, mu2)
    r7 = linear_perceptron(l7, a, mu1)
    r8 = linear_perceptron(l8, a, mu1)
    r12 = linear_perceptron(np.asarray([1, 1]).reshape((2, 1)), np.asarray([r1, r2]).reshape((2, 1)), mu3)
    r56 = linear_perceptron(np.asarray([1, 1]).reshape((2, 1)), np.asarray([r5, r6]).reshape((2, 1)), mu3)

    w_and = np.ones((6, 1)) * 10
    r = np.asarray([r3, r4, r7, r8, r12, r56]).reshape((6, 1))
    result = linear_perceptron(w_and, r, mu4)
    return result
