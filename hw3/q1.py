# two layer neural network

# Q1:
from matplotlib import pyplot as plt
import numpy as np
import scipy.io

data = scipy.io.loadmat('data.mat')
X = np.array(data["X"])  # 5000 * 400
y = np.array(data["y"])  # 5000 * 1

input_layer_size = 400
hidden_layer_size = 25
num_labels = 10

rand_100 = X[np.random.choice(X.shape[0], 100, replace=False)]  # array which contains 100 rows of X chosen randomly
pics = np.zeros((200, 200))  # convert each row in to a 20 * 20 image and add it to pics
for i in range(10):
    for j in range(10):
        pics[i * 20:i * 20 + 20, j * 20:j * 20 + 20] = np.reshape(rand_100[i + 10 * j, :], (20, 20), order='F')

plt.imshow(pics, cmap=plt.get_cmap('gray'))
plt.show()

# Q2:
# we have 500 rows for each number. we use the first 300 in the training set.
train_set = np.zeros((300 * 10, 400))
test_set = np.zeros((200 * 10, 400))
y_train = np.zeros((300 * 10, 1), dtype='uint8')
y_test = np.zeros((200 * 10, 1), dtype='uint8')
for i in range(10):
    train_set[i * 300:i * 300 + 300, :] = X[i * 500:i * 500 + 300, :]
    test_set[i * 200:i * 200 + 200, :] = X[i * 500:i * 500 + 200, :]
    y_train[i * 300:i * 300 + 300, :] = y[i * 500:i * 500 + 300, :]
    y_test[i * 200:i * 200 + 200, :] = y[i * 500:i * 500 + 200, :]


# Q3
def random_weight_initialization(L_in, L_out, epsilon):
    W = np.random.rand(L_out, 1 + L_in) * 2 * epsilon - epsilon
    return W


theta1 = random_weight_initialization(400, 25, 0.12)
theta2 = random_weight_initialization(25, 10, 0.12)


# Q4
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_gradient(z):
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))


# Q5
def compute_cost(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambdaa):
    m = X.shape[0]

    # derive theta1 and theta2 from the nn_params which contains all the wights
    theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                        (hidden_layer_size, (input_layer_size + 1)),
                        order='F')  # hidden_layer_size *  (input_layer_size + 1)
    theta2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1)):], (num_labels, (hidden_layer_size + 1)),
                        order='F')  # num_labels * (hidden_layer_size + 1)

    # performing forward propagation
    # first add the bias unit of a vector of ones to a1
    bias_unit = np.ones((m, 1))
    a1 = np.concatenate((bias_unit, X), axis=1)  # m * 401
    z2 = a1 @ np.transpose(theta1)  # m * hidden_layer_size
    a2 = sigmoid(z2)
    a2 = np.concatenate((bias_unit, a2), axis=1)  # m * (hidden_layer_size+1)
    z3 = a2 @ np.transpose(theta2)  # m * num_labels
    a3 = sigmoid(z3)  # a3 =  h_theta(x) m * num_labels

    # also lets create the m * num_labels matrix of the actual values stored in y
    Y = np.zeros((m, num_labels))
    for i in range(m):
        s = y[i]
        Y[i, s - 1] = 1

    # compute cost
    J = (1 / m) * np.sum(np.sum(-np.multiply(Y, np.log(a3)) - np.multiply((1 - Y), np.log(1 - a3)))) + (
                lambdaa / (2 * m)) * (np.sum(np.sum(np.power(theta1[:, 1:], 2), axis=1)) + np.sum(
        np.sum(np.power(theta2[:, 1:], 2), axis=1)))
    return J


def compute_gradient(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambdaa):
    m = X.shape[0]
    bias_unit = np.ones((m, 1))

    # derive theta1 and theta2 from the nn_params which contains all the wights
    theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                        (hidden_layer_size, (input_layer_size + 1)),
                        order='F')  # hidden_layer_size *  (input_layer_size + 1)
    theta2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1)):], (num_labels, (hidden_layer_size + 1)),
                        order='F')  # num_labels * (hidden_layer_size + 1)

    # backpropagation
    theta1_grad = np.zeros_like(theta1)
    theta2_grad = np.zeros_like(theta2)
    A1 = np.concatenate((bias_unit, X), axis=1)  # m * 401
    Z2 = A1 @ np.transpose(theta1)  # m * hidden_layer_size
    A2 = np.concatenate((bias_unit, sigmoid(Z2)), axis=1)  # m * (hidden_layer_size + 1)
    Z3 = A2 @ np.transpose(theta2)  # m * num_labels
    A3 = sigmoid(Z3)

    Y = np.zeros((m, num_labels))
    for i in range(m):
        s = y[i]
        Y[i, s - 1] = 1

    delta3 = A3 - Y  # m * num_labels
    delta2 = np.multiply((delta3 @ theta2), np.c_[bias_unit, sigmoid_gradient(Z2)])
    delta2 = delta2[:, 1:]
    theta1_grad = np.transpose(delta2) @ A1
    theta2_grad = np.transpose(delta3) @ A2
    theta1_grad, theta2_grad = theta1_grad / m, theta2_grad / m
    # add the regularization term
    theta1_grad[:, 1:] += (lambdaa / m) * theta1[:, 1:]
    theta2_grad[:, 1:] += (lambdaa / m) * theta2[:, 1:]
    gradient = np.hstack((theta1_grad.ravel(order='F'), theta2_grad.ravel(order='F')))
    return gradient

    ## non-vectorized implementation
    # one = np.ones((1,1))
    # for i in range(m):
    #     a1 = np.transpose(X[i,:]).reshape((X.shape[1], 1)) # 401 * 1
    #     z2 = theta1 @ a1 # hidden_layer_size * 1
    #     a2 = np.r_[one, z2]# (hidden_layer_size + 1) * 1
    #     z3 = theta2 @ a2 # num_labels * 1
    #     a3 = sigmoid(z3)
    #     # we need y[i] to be a num_labels * 1 vector where only the actual number is one
    #     yi = np.zeros((num_labels,1))
    #     yi[y[i]-1] = 1
    #     # calculate deltas
    #     # L = 3, so we should only compute d3 and d2
    #     delta3 = a3 - yi
    #     delta2 = np.multiply((np.transpose(theta2) @ delta3), np.r_[one, sigmoid_gradient(z2)])
    #     delta2 = delta2[1:]
    #     theta1_grad += delta2 @ np.transpose(a1)
    #     theta2_grad += delta3 @ np.transpose(a2)
    #
    # theta1_grad, theta2_grad = theta1_grad/m, theta2_grad/m
    # # add the regularization term
    # theta1_grad[:, 1:] += (lambdaa/m) * theta1[:,1:]
    # theta2_grad[:, 1:] += (lambdaa/m) * theta2[:,1:]
    # gather all the elements of gradients in one vector


# Q6

from scipy.optimize import fmin_cg

nn_params = np.hstack((theta1.ravel(order='F'), theta2.ravel(order='F')))
# use lambda = 1
theta = fmin_cg(f=compute_cost, fprime=compute_gradient, x0=nn_params,
                args=(input_layer_size, hidden_layer_size, num_labels, test_set, y_test, 1), maxiter=80)

theta1_res = np.reshape(theta[:hidden_layer_size * (input_layer_size + 1)], (hidden_layer_size, input_layer_size + 1),
                        order='F')
theta2_res = np.reshape(theta[hidden_layer_size * (input_layer_size + 1):], (num_labels, hidden_layer_size + 1),
                        order='F')


# Q7
def predict(theta1, theta2, X_test, y_test):
    m = X_test.shape[0]
    bias_unit = np.ones((m, 1))
    a1 = np.concatenate((bias_unit, X_test), axis=1)  # m * 401
    z2 = a1 @ np.transpose(theta1)  # m * hidden_layer_size
    a2 = sigmoid(z2)
    a2 = np.concatenate((bias_unit, a2), axis=1)  # m * (hidden_layer_size+1)
    z3 = a2 @ np.transpose(theta2)  # m * num_labels
    a3 = sigmoid(z3)
    prediction = np.argmax(a3, axis=1).reshape(y_test.shape) + 1
    accuracy = np.mean(prediction == y_test) * 100
    return prediction, accuracy


prediction, accuracy = predict(theta1_res, theta2_res, test_set, y_test)
print(accuracy)


def show_100_random_images(X, y):
    rand_idx = np.random.choice(X.shape[0], 100, replace=False)
    x_rand = X[rand_idx]
    y_rand = y[rand_idx]
    fig, axs = plt.subplots(10, 10, figsize=(20, 20))
    for i in range(10):
        for j in range(10):
            img = np.reshape(x_rand[i + 10 * j, :], (20, 20), order='F')
            axs[i, j].imshow(img)
            axs[i, j].set_title(str(y_rand[i + 10 * j]))
            axs[i, j].axis('off')

    fig.show()


show_100_random_images(test_set, prediction)


# Q8
def display_hidden_units(theta, image_size, figure_size):
    res = np.zeros((image_size * figure_size, image_size * figure_size))
    for i in range(figure_size):
        for j in range(figure_size):
            res[i * image_size:(i + 1) * image_size, j * image_size:(j + 1) * image_size] = np.reshape(
                theta[i * figure_size + j, 1:], (image_size, image_size), order='F')

    plt.imshow(res, cmap=plt.get_cmap('gray'))
    plt.show()


display_hidden_units(theta1_res, 20, 5)
