import math
import numpy as np
from matplotlib import pyplot as plt

stopping_criterion = 0.00001
f_star = 0


def d_w_squared(w):
    return w


def d_abs_w(w):
    if w < 0:
        return -1
    if w > 0:
        return 1
    return 0


def gradient_descent(grad_func=d_w_squared, w0=1, learning_rate=.75):
    gradient = float('inf')
    w = w0
    errors = []
    it_num = 0
    while abs(gradient) > stopping_criterion and it_num < 30:
        it_num += 1
        error = (0.5 * (w ** 2)) - f_star
        errors.append(error)
        gradient = grad_func(w)
        w = w - learning_rate * gradient
    return np.array(errors)


def adagrad(grad_func=d_w_squared, w0=1, learning_rate=.75):
    gradient = float('inf')
    gradient_norm_sum = 0
    w = w0
    errors = []
    it_num = 0
    while abs(gradient) > stopping_criterion and it_num < 30:
        it_num += 1
        gradient = grad_func(w)
        gradient_norm_sum += gradient ** 2
        error = (0.5 * (w ** 2)) - f_star
        errors.append(error)
        step_size_adagrad = learning_rate / (
            math.sqrt(gradient_norm_sum)
        )
        w = w - step_size_adagrad * gradient
    return np.array(errors)


def plot_convergence(errors, title, ylabel):
    x_ = list(range(len(errors)))
    plt.plot(x_, errors)
    plt.xlabel('Iteration')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    # gd_errors_w2 = gradient_descent()
    # plot_convergence(
    #     np.log(gd_errors_w2),
    #     'Gradient Descent Convergence', 'log(F(w_k) - F*)')
    # adagrad_errors_w2 = adagrad()
    # plot_convergence(
    #     np.log(adagrad_errors_w2),
    #     'Adagrad Convergence', 'log(F(w_k) - F*)')
    gd_errors_abs_w = gradient_descent(grad_func=d_abs_w)
    plot_convergence(
        gd_errors_abs_w, 'Gradient Descent Convergence', 'F(w_k) - F*')
    adagrad_errors_abs_w = adagrad(grad_func=d_abs_w)
    plot_convergence(
        adagrad_errors_abs_w, 'Adagrad Convergence', 'F(w_k) - F*')
