import cv2 as cv2
import os

import matplotlib.pyplot as plt
import numpy as np


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    return images


def show_images(images):
    for image in images:
        plt.imshow(image)
        plt.show()


def pre_process_images(images):
    processed = []
    for image in images:
        t = np.where(image < np.mean(image), -1, 1)
        t = t.flatten()
        processed.append(t)
    return processed


def train(N, images):
    W = np.zeros((N, N), dtype=np.int8)
    for i in images:
        W += np.outer(i, i)
    return W / N


def predict(images):
    for image in images:
        error = None
        S = image
        while error is None or error > 100:
            Si = np.where(S < np.mean(S), -1, 1)
            Si = Si.flatten()
            S = np.matmul(W, Si)
            S = np.where(S < 0, -1, 1)
            error = np.sum(np.abs(S - Si))
            print(error)
        print("error: " + str(error))
        np.where(S == 1, 255, 0)
        result_image = np.reshape(S, image.shape)
        plt.title("matched prototype")
        plt.imshow(result_image)
        plt.show()
        plt.title("inital image")
        plt.imshow(image)
        plt.show()


def replaceRandom(image, size):
    temp = np.asarray(image)
    shape = temp.shape
    temp = temp.flatten()
    random_indices = np.random.choice(temp.size, size=size)
    temp[random_indices] = np.random.randint(0, 255, size)
    return temp.reshape(shape)


def apply_noise(images, size):
    noisy_images = []
    for image in images:
        result_image = replaceRandom(image, size)
        noisy_images.append(result_image)
        plt.imshow(result_image)
        plt.show()
    return noisy_images


def compute_correlation():
    for i in range(M):
        for j in range(M):
            noisy = np.where(noisy_images_3000[i] < np.mean(noisy_images_3000[i]), -1, 1)
            initial = np.where(initial_images[j] < np.mean(initial_images[j]), -1, 1)
            correlation = (1/N) * np.dot(noisy.flatten(), initial.flatten())
            print("the correlation between the " + str(i) + "th (noisy) and " + str(j) + "th (initial) images is: " + str(correlation))


initial_images = load_images_from_folder("train")
show_images(initial_images)

M = len(initial_images)
N = initial_images[0].shape[0] * initial_images[0].shape[1]
image_shape = initial_images[0].shape
processed_images = pre_process_images(initial_images)
W = train(N, processed_images)
plt.imshow(W)
plt.show()

predict(initial_images)
noisy_images_3000 = apply_noise(initial_images, 3000)
compute_correlation()
predict(noisy_images_3000)

noisy_images_8000 = apply_noise(initial_images, 8000)
predict(noisy_images_8000)