from sklearn.model_selection import train_test_split
import pandas as pd
import io
import requests
import math
from matplotlib import pyplot as plt

cacher = {}


def load_and_preprocess_data():
    url = 'https://web.stanford.edu/~hastie/ElemStatLearn/datasets/spam.data'
    s = requests.get(url).content
    data = pd.read_csv(io.StringIO(s.decode('utf-8')), sep=' ', header=None)
    x = data[data.columns[:-1]]
    y = data[data.columns[-1]]
    x = (x - x.mean()) / (x.max() - x.min())
    y[y == 0] = -1
    return x.values.tolist(), y.tolist()


def gaussian_rbf_kernel(x1, x2, sigma):
    return math.exp(
        -sum((xi - xj)**2 for xi, xj in zip(x1, x2)) / (sigma ** 2))


def vokvs_polynomial_kernel(x1, x2, p):
    return (1 - math.pow(
        sum(x_i * x_j for x_i, x_j in zip(x1, x2)), p)) / (
        1 - sum(x_i * x_j for x_i, x_j in zip(x1, x2))
    )


def sum_cross_points(kernel_function, kernel_param, x_points):
    global cacher
    key_ = str(kernel_function) + str(kernel_param) + str(len(x_points))
    if key_ not in cacher:
        cacher[key_] = sum(
            sum(kernel_function(x_i, x_j, kernel_param) for x_i in x_points)
            for x_j in x_points
        )
    return cacher[key_]


def distance_to_centroid(x, x_points, kernel_function, kernel_param):
    n = len(x_points)
    return kernel_function(x, x, kernel_param) - (
        (2 / n) * sum(
            kernel_function(x, x_point, kernel_param)
            for x_point in x_points
        )) + (
        (1 / (n**2)) * sum_cross_points(
            kernel_function, kernel_param, x_points)
    )


def predict(x_new, x_data, y_data, kernel_function, kernel_param):
    clusters = {}
    for idx, x_row in enumerate(x_data):
        if y_data[idx] not in clusters:
            clusters[y_data[idx]] = [x_row]
        else:
            clusters[y_data[idx]].append(x_row)
    best_label = None
    closest_centroid = float('inf')
    for y_label, x_points in clusters.items():
        centroid_distance = distance_to_centroid(
            x_new, x_points, kernel_function, kernel_param)
        if centroid_distance < closest_centroid:
            best_label = y_label
            closest_centroid = centroid_distance
    return best_label


def compute_test_error(
        x_train, x_test, y_train, y_test, kernel_function, kernel_param):
    incorrect_pred_count = 0
    for x, y in zip(x_test, y_test):
        incorrect_pred_count += y != predict(
            x, x_train, y_train, kernel_function, kernel_param)
    return incorrect_pred_count / len(x_test)


def optimize_parameters(kernel_function, params_to_try):
    x_data, y_data = load_and_preprocess_data()
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.33)
    test_errors = []
    for param in params_to_try:
        print(param)
        test_errors.append(
            compute_test_error(
                x_train, x_test, y_train, y_test, kernel_function, param
            )
        )
        print(test_errors)
    return test_errors


if __name__ == '__main__':
    sigmas = [0.01, 0.1, 1, 10, 100, 1000]
    gaussian_rbf_errors = optimize_parameters(gaussian_rbf_kernel, sigmas)
    p_params = [1, 2, 3, 4, 5, 10]
    voks_poly_errors = optimize_parameters(vokvs_polynomial_kernel, p_params)
    sigmas_transformed = [math.log(sigma, 10) for sigma in sigmas]
    plt.scatter(sigmas_transformed, gaussian_rbf_errors)
    plt.xlabel("log10(Sigma)")
    plt.ylabel("Misclassification Error Rate")
    plt.title('Gaussian RBF Error Rate by Sigma')
    plt.show()
    plt.scatter(p_params, voks_poly_errors)
    plt.xlabel("P Parameter")
    plt.ylabel("Misclassification Error Rate")
    plt.title('Vokvs Polynomial Error Rate by p')
    plt.show()
