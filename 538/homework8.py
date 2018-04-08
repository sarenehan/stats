from matplotlib import pyplot as plt
from sklearn.svm import SVC
from mpl_toolkits.mplot3d import Axes3D  # noqa
import math
import numpy as np
import pickle
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split


mnist_data_location = '/Users/stewart/projects/stats/538/mnist_data.pkl'
hidden_layer_size = 1600
rnd = np.random.RandomState(1)


def gaussian_rbf_kernel(x1, x2, sigma):
    return np.exp(-sum((x1 - x2) ** 2) / sigma ** 2)


def get_gaussian_rbf_kernel_matrix(data, sigma):
    n = len(data)
    kernel_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            kernel_matrix[i][j] = gaussian_rbf_kernel(data[i], data[j], sigma)
    return kernel_matrix


def download_mnist_data():
    data = fetch_mldata('MNIST Original')
    with open(mnist_data_location, 'wb') as f:
        pickle.dump(data, f)


def load_mnist_data():
    with open(mnist_data_location, 'rb') as f:
        data = pickle.load(f)
    x_data = data.data
    y_data = data.target
    return x_data, y_data


def preprocess_data():
    x_data, y_data = load_mnist_data()
    x_mean = np.array([x or 1 for x in x_data.mean(axis=0)])
    x_data = x_data / x_mean
    x_data = np.array([
        point for idx, point in enumerate(x_data)
        if y_data[idx] in [0, 1]
    ])
    y_data = np.array([point for point in y_data if point in [0, 1]])
    return x_data, y_data


def gradient(K, y_train, lambda_, alpha):
    K_alpha = np.dot(K, alpha)
    n = len(K)
    term_1 = (1 / n) * sum(
        ((-y_train[i] * np.exp(-y_train[i] * K_alpha[i])) /
         (1 + np.exp(-y_train[i] * K_alpha[i]))) * K[i]
        for i in range(n)
    )
    term_2 = lambda_ * alpha.T.dot(K)
    return term_1 + term_2


def compute_train_error(K, y_train, alpha, lambda_):
    K_alpha = np.dot(K, alpha)
    term_1 = (1 / len(K)) * sum(
        np.log(1 + np.exp(-y_train[i] * K_alpha[i]))
        for i in range(len(K))
    )
    term_2 = (lambda_ / 2) * (alpha.T.dot(K_alpha))
    return term_1 + term_2


def predict(train_x, new_x, alpha, lambda_, sigma):
    fx = sum(
        alpha[i] * gaussian_rbf_kernel(train_x[i], new_x, sigma)
        for i in range(len(train_x))
    )
    return int(1 / (1 + np.exp(-fx)) > 0.5)


def compute_test_error(test_x, test_y, train_x, alpha, lambda_, sigma):
    incorrect_preds = [
        predict(train_x, new_x, alpha, lambda_, sigma) != new_y
        for new_x, new_y in zip(test_x, test_y)
    ]
    return sum(incorrect_preds) / len(incorrect_preds)


def mysfm_grad(K, y_train, step_size, stopping_crit, lambda_, sigma):
    alpha = np.array([0] * len(K))
    t = 0
    grad_norm = float('inf')
    while grad_norm > stopping_crit:
        grad = gradient(K, y_train, lambda_, alpha)
        alpha = alpha - (step_size * grad)
        t = t + 1
        print(compute_train_error(K, y_train, alpha, lambda_))
        grad_norm = np.linalg.norm(grad) ** .5
    return alpha


def mysvm_fast_grad(K, y_train, step_size, stopping_crit, lambda_, sigma):
    alpha = np.array([0] * len(K))
    beta = np.array([0] * len(K))
    t = 0
    n_iters = 0
    grad_norm = float('inf')
    while grad_norm > stopping_crit and n_iters < 1000:
        alpha_prev = alpha
        alpha = beta - (step_size * gradient(K, y_train, lambda_, beta))
        beta = alpha + ((t / (t + 3)) * (alpha - alpha_prev))
        t = t + 1
        n_iters += 1
        grad_norm = np.linalg.norm(gradient(K, y_train, lambda_, alpha)) ** .5
    return alpha


def get_step_size(K, y_train, lambda_):
    x_ = rnd.normal(scale=10, size=(len(K),))
    y_ = rnd.normal(scale=10, size=(len(K),))
    return (np.linalg.norm(x_ - y_) ** .5) / (np.linalg.norm(
        gradient(K, y_train, lambda_, x_) -
        gradient(K, y_train, lambda_, y_)
    ) ** .5)


def plot_error_vs_params(test_errors, lambdas, sigmas):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(lambdas, sigmas, test_errors)
    ax.set_xlabel('log10(Lambda)')
    ax.set_ylabel('log10(Sigma)')
    ax.set_zlabel('Error Rate')
    ax.set_title('Error Rate vs Lambda and Sigma')
    plt.show()


def plot_error_vs_df(test_errors, dfs):
    plt.scatter(dfs, test_errors)
    plt.xlabel("Degrees of Freedom")
    plt.ylabel("Test Misclassification Error Rate")
    plt.title('Error Rate vs DF')
    plt.show()


def build_sklearn_svms(x_train, x_test, y_train, y_test):
    for kernel in ['linear', 'rbf', 'poly', 'sigmoid']:
        for c_ in [.01, .1, 1]:
            clf = SVC(C=c_, kernel=kernel).fit(x_train, y_train)
            error = 1 - clf.score(x_test, y_test)
            print('{} kernel with C = {}; test error = {}'.format(
                kernel, c_, error))


if __name__ == '__main__':
    x_data, y_data = preprocess_data()
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.99)
    x_test, _, y_test, _ = train_test_split(
        x_test, y_test, test_size=0.99)
    build_sklearn_svms(x_train, x_test, y_train, y_test)
    stopping_crit = 0.01
    sigmas = []
    lambdas = []
    test_errors = []
    dfs = []
    for sigma in [0.01, 0.1, 1, 10, 100]:
        for lambda_ in [0.00001, 0.0001, 0.001, 0.01]:
            sigmas.append(math.log(sigma, 10))
            lambdas.append(math.log(lambda_, 10))
            print('Sigma: {}'.format(sigma))
            print('Lambda: {}'.format(lambda_))
            K = get_gaussian_rbf_kernel_matrix(x_train, sigma)
            step_size = get_step_size(K, y_train, lambda_)
            # mysfm_grad(K, y_train, step_size, stopping_crit, lambda_, sigma)
            alpha = mysvm_fast_grad(
                K, y_train, step_size, stopping_crit, lambda_, sigma)
            test_error = compute_test_error(
                x_test, y_test, x_train, alpha, lambda_, sigma
            )
            print('Test error: {}'.format(test_error))
            test_errors.append(test_error)
            df = np.trace(
                np.linalg.inv(K + (lambda_ * np.identity(len(K)))).dot(K)
            )
            dfs.append(df)
    plot_error_vs_df(test_errors, dfs)
    plot_error_vs_params(test_errors, lambdas, sigmas)
