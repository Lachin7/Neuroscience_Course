import matplotlib.pyplot as plt
import numpy as np
import scipy.io


def pca1(X):
    X = normalize_data(X)
    plt.scatter(X[:, 0], X[:, 1])
    # calc covariance matrix
    # cov = np.cov(X, rowvar=False)
    # calc eigen vectors
    # eigen_values, eigen_vectors = np.linalg.eig(cov)
    P, V, S = get_principle_components(X, 1)
    # plot eigen vectors
    origin = np.array([[0, 0], [0, 0]])
    plt.quiver(*origin, V[:, 0], V[:, 1], color=['r', 'b'], scale=1)
    plt.show()
    print(P)


def normalize_data(X):
    # normalization: (assuming we have training sets as rows)
    miu = np.reshape(np.average(X, axis=0), (1, X.shape[1]))
    std = np.reshape(np.std(X, axis=0), (1, X.shape[1]))
    return (X - miu) / std


def get_principle_components(X, k):
    u, s, vh = np.linalg.svd(X, full_matrices=False)
    v = vh[:k, :].transpose()
    # principle components, eigen vector, eigen values
    return X @ v, vh, s


data = scipy.io.loadmat('1d_pcadata.mat')
pca1(np.array(data["X"]))


def pca2(X, k):
    X = normalize_data(X)
    return get_principle_components(X, k)


def plot_pics(X, a):
    # draws a^2 faces:
    pic = np.zeros((a * 32, a * 32))
    for i in range(a):
        for j in range(a):
            pic[i * 32:i * 32 + 32, j * 32:j * 32 + 32] = np.reshape(X[i + j, :], (32, 32), order='F')
    plt.imshow(pic, cmap=plt.get_cmap('gray'))
    plt.show()


data = scipy.io.loadmat('faces.mat')
X = np.array(data["X"])
plot_pics(X, 6)

# assume we have k = 900
P, V, S = pca2(X, 900)
# the first 5 eigen value
print(S[0:5])
# plot the 36 eigen faces
plot_pics(V, 6)
# plot the 16 eigen faces
plot_pics(V, 4)

# if we use the first 20 pics:
# assume we have k = 900
P, V, S = pca2(X[0:20, :], 900)
print(S[0:5])
plot_pics(V, 4)
