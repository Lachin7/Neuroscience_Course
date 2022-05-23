import cv2
from matplotlib import pyplot as plt
import numpy as np
import random

path = "ponyo.jpeg"
img = cv2.imread(path)
height, width, _ = img.shape
B, G, R = cv2.split(img)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()



def K_means(r, g, b, n, k, h, w, max_num_iter=20):
    centers_change = True
    iterations = 0
    J = []
    clusters = np.array(random.choices(range(0, k), k=n))  # at first, clusters are chosen randomly

    centers_r, centers_g, centers_b = assign_center(r, g, b, k, clusters)
    # then iterate until the centers get fixed
    while centers_change:
        j = 0
        for i in range(n):
            clusters[i], distance = assign_cluster(r[i], g[i], b[i], k, centers_r, centers_g, centers_b)
            j += distance
        J.append(j)

        prev_r, prev_g, prev_b = centers_r.copy(), centers_g.copy(), centers_b.copy()
        centers_r, centers_g, centers_b = assign_center(r, g, b, k, clusters)

        diff = np.abs(centers_r - prev_r) + np.abs(centers_g - prev_g) + np.abs(centers_b - prev_b)
        if np.sum(diff) < k or iterations == max_num_iter:
            centers_change = False
        iterations += 1

    res = get_result(k, h, w, clusters, centers_r, centers_g, centers_b)
    cv2.imwrite("resQ4/k=" + str(k) + "-iter=" + str(iterations) + ".jpg", res)
    plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
    plt.show()

    plt.plot(np.arange(0, iterations), J)
    plt.show()


def get_result(k, h, w, clusters, centers_r, centers_g, centers_b):
    r, g, b = np.ndarray((n, 1)), np.ndarray((n, 1)), np.ndarray((n, 1))
    for i in range(k):
        region = clusters == i
        r[region], g[region], b[region] = centers_r[i], centers_g[i], centers_b[i]
    r, g, b = np.reshape(r, (h, w)), np.reshape(g, (h, w)), np.reshape(b, (h, w))
    return cv2.merge((b.astype('uint8'), g.astype('uint8'), r.astype('uint8')))


def assign_center(r, g, b, k, clusters):
    centers_r, centers_g, centers_b = np.ndarray((k, 1)), np.ndarray((k, 1)), np.ndarray((k, 1))
    for cluster_num in range(k):
        region = clusters == cluster_num
        if region.size > 0:
            rk, gk, bk = r[region], g[region], b[region]
            r_mean, g_mean, b_mean = np.mean(rk), np.mean(gk), np.mean(bk)
            centers_r[cluster_num], centers_g[cluster_num], centers_b[cluster_num] = r_mean, g_mean, b_mean
    return centers_r, centers_g, centers_b


def assign_cluster(r, g, b, k, centers_r, centers_g, centers_b):
    min_distance, cluster = None, 0
    for i in range(k):
        distance = calculate_distance(r, g, b, centers_r[i], centers_g[i], centers_b[i])
        if min_distance is None or distance < min_distance:
            min_distance, cluster = distance, i
    return cluster, min_distance


def calculate_distance(r1, g1, b1, r2, g2, b2):
    return np.sqrt(np.power(r1 - r2, 2) + np.power(g1 - g2, 2) + np.power(b1 - b2, 2))


n = height * width
R, G, B = np.reshape(R, (n, 1)), np.reshape(G, (n, 1)), np.reshape(B, (n, 1))
K_means(R, G, B, n, 4, height, width)
K_means(R, G, B, n, 4, height, width)
K_means(R, G, B, n, 5, height, width)
K_means(R, G, B, n, 6, height, width)
K_means(R, G, B, n, 10, height, width)
