import numpy as np
from itertools import product
from seaborn import heatmap
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor


sigma = 1
x1_scale = np.arange(1, 101)
x2_scale = np.arange(1, 101)
indices = product(x1_scale, x2_scale)


def f_nonlinear(x1, x2):
    return (x1 / x2) + (np.log(x1) ** 2) + x2 + np.random.normal(scale=sigma)


def f_linear(x1, x2):
    return (3 * x1) + (5 * x2) - 12 + np.random.normal(scale=sigma)


def plot_heatmap_nonlinear():
    matrix_of_values = np.zeros((100, 100))
    for x1, x2 in indices:
        matrix_of_values[x1 - 1, x2 - 1] = f_nonlinear(x1, x2)

    heatmap(matrix_of_values)
    plt.ylabel('x1')
    plt.xlabel('x2')
    plt.title('Heatmap of x1 vs x2')
    plt.savefig('/Users/stewart/Desktop/hw_1_problem_6_b.png')


def plot_heatmap_linear():
    matrix_of_values = np.zeros((100, 100))
    for x1, x2 in indices:
        matrix_of_values[x1 - 1, x2 - 1] = f_linear(x1, x2)

    heatmap(matrix_of_values)
    plt.ylabel('x1')
    plt.xlabel('x2')
    plt.title('Heatmap of x1 vs x2')
    plt.savefig('/Users/stewart/Desktop/hw_1_problem_6_f_b.png')


def generate_training_and_testing_data():
    x1s_train = np.random.choice(x1_scale, 100, replace=True)
    x1s_test = np.random.choice(x1_scale, 100, replace=True)
    x2s_train = np.random.choice(x2_scale, 100, replace=True)
    x2s_test = np.random.choice(x2_scale, 100, replace=True)
    x_train = np.vstack([[x1s_train, x2s_train]]).T
    x_test = np.vstack([[x1s_test, x2s_test]]).T

    y_train = np.array([f(*x_) for x_ in x_train])
    y_test = np.array([f(*x_) for x_ in x_test])
    return x_train, y_train, x_test, y_test


def mse(preds, true):
    return ((preds - true) ** 2).mean()


def get_linear_model_error(x_train, y_train, x_test, y_test):
    model = LinearRegression()
    model.fit(x_train, y_train)
    train_mse = mse(model.predict(x_train), y_train)
    test_mse = mse(model.predict(x_test), y_test)
    return train_mse, test_mse


def get_knn_error(x_train, y_train, x_test, y_test, k_):
    model = KNeighborsRegressor(n_neighbors=k_)
    model.fit(x_train, y_train)
    train_mse = mse(model.predict(x_train), y_train)
    test_mse = mse(model.predict(x_test), y_test)
    return train_mse, test_mse


def generate_plot_nonlinear():
    global f
    f = f_nonlinear
    x_train, y_train, x_test, y_test = generate_training_and_testing_data()
    lm_train_error, lm_test_error = get_linear_model_error(
        x_train, y_train, x_test, y_test)
    knn_train_errors = []
    knn_test_errors = []
    ks_to_test = [1, 3, 5, 7, 9, 11, 13, 15]
    for k_ in ks_to_test:
        train_error, test_error = get_knn_error(
            x_train, y_train, x_test, y_test, k_)
        knn_train_errors.append(train_error)
        knn_test_errors.append(test_error)
    plt.scatter(ks_to_test, knn_train_errors, label='Knn Train')
    plt.scatter(ks_to_test, knn_test_errors, label='Knn Test')
    plt.xticks(ks_to_test)
    plt.hlines(
        lm_train_error, 1, 15, label='Linear Model Train', color='purple')
    plt.hlines(
        lm_test_error, 1, 15, label='Linear Model Test', color='black')
    plt.xlabel('K Neighbors')
    plt.ylabel('Mean Squared Error')
    plt.title('Linear Regression vs KNN on Nonlinear Simulated Data')
    plt.legend(loc='best')
    plt.savefig('/Users/stewart/Desktop/hw_1_problem_6.png')


f = f_linear
x_train, y_train, x_test, y_test = generate_training_and_testing_data()
lm_train_error, lm_test_error = get_linear_model_error(
    x_train, y_train, x_test, y_test)
knn_train_errors = []
knn_test_errors = []
ks_to_test = [1, 3, 5, 7, 9, 11, 13, 15]
for k_ in ks_to_test:
    train_error, test_error = get_knn_error(
        x_train, y_train, x_test, y_test, k_)
    knn_train_errors.append(train_error)
    knn_test_errors.append(test_error)
plt.scatter(ks_to_test, knn_train_errors, label='Knn Train')
plt.scatter(ks_to_test, knn_test_errors, label='Knn Test')
plt.xticks(ks_to_test)
plt.hlines(
    lm_train_error, 1, 15, label='Linear Model Train', color='purple')
plt.hlines(
    lm_test_error, 1, 15, label='Linear Model Test', color='black')
plt.xlabel('K Neighbors')
plt.ylabel('Mean Squared Error')
plt.title('Linear Regression vs KNN on Nonlinear Linear Simulated Data')
plt.legend(loc='best')
plt.savefig('/Users/stewart/Desktop/hw_1_problem_6_f.png')
