from matplotlib import pyplot as plt
import math
import numpy as np
from statistics import mean

rnd = np.random.RandomState(2)
sigma = 1


def get_other_elipse_val_from_one(y, a, b):
    x = a * math.sqrt(1 - ((y ** 2) / (b ** 2)))
    return rnd.choice([x, -x])


def generate_data_point_around_ellipse(a, b):
    if rnd.rand() > .5:
        height = 2 * b
        y = -b + (rnd.rand() * height)
        x = get_other_elipse_val_from_one(y, a, b)
    else:
        width = 2 * a
        x = -a + (rnd.rand() * width)
        y = get_other_elipse_val_from_one(x, b, a)
    y = y + rnd.normal(0, .25)
    x = x + rnd.normal(0, .25)
    return [x, y]


def generate_sample_data(a1=2, b1=1, a2=6, b2=3, n_per_class=50):
    class_1 = [
        generate_data_point_around_ellipse(a1, b1)
        for _ in range(n_per_class)
    ]
    class_2 = [
        generate_data_point_around_ellipse(a2, b2)
        for _ in range(n_per_class)
    ]
    return class_1 + class_2


def gaussian_rbf_kernel(x1, x2, sigma):
    return math.exp(
        -sum((xi - xj)**2 for xi, xj in zip(x1, x2)) / (sigma ** 2))


def compute_matrix_vector_product(m, v):
    """
    Computes the product of square matrix M * V
    """
    return np.array([
        sum(m[i][j] * v[j] for j in range(len(m[i])))
        for i in range(len(m))
    ])


def get_covariance_matrix(x):
    p = len(x[0])
    n = len(x)
    cov_matrix = np.array([[None] * p] * p)
    mean_x = [
        mean(point[idx] for point in x) for idx in range(p)
    ]
    for i in range(p):
        for j in range(p):
            cov_matrix[i][j] = sum(
                ((x[idx][i] - mean_x[i]) * (x[idx][j] - mean_x[j]))
                for idx in range(n)
            ) / (n - 1)
    return cov_matrix


def get_gaussian_rbf_kernel_matrix(data, sigma):
    n = len(data)
    kernel_matrix = np.array([[None] * n] * n)
    for i in range(n):
        for j in range(n):
            kernel_matrix[i][j] = gaussian_rbf_kernel(data[i], data[j], sigma)
    return kernel_matrix


def norm(vect):
    return math.sqrt(sum(v**2 for v in vect))


def power_iteration(x, max_iterations=500):
    vk = rnd.normal(size=(1, len(x)))[0]
    for k in range(max_iterations):
        zk = compute_matrix_vector_product(x, vk)
        vk = np.array([zi / norm(zk) for zi in zk])
    x_vk = compute_matrix_vector_product(x, vk)
    lambdak = sum(vk[i] * x_vk[i] for i in range(len(vk)))
    return lambdak, vk


def square_matrix_product(a, b):
    assert len(a) == len(b)
    n = len(a)
    return np.array(
        [
            [sum(a[i][k] * b[k][j] for k in range(n)) for j in range(n)]
            for i in range(n)
        ]
    )


def center_kernel_matrix(K):
    n = len(K)
    off_diagonal_ons = np.array(
        [[int(i != j) - (1 / n) for i in range(n)] for j in range(n)]
    )
    return square_matrix_product(
        off_diagonal_ons, square_matrix_product(K, off_diagonal_ons)
    )


def subtract_pc(K, ev, lambda_):
    n = len(ev)
    to_subtract = np.array([
        [ev[i] * ev[j] * lambda_ for j in range(n)]
        for i in range(n)
    ])
    return np.array([
        [K[i][j] - to_subtract[i][j] for j in range(n)]
        for i in range(n)
    ])


def alpha_t_k(alpha, K):
    return [
        sum(alpha[j] * K[i][j] for j in range(len(alpha)))
        for i in range(len(alpha))
    ]


def perform_kernel_pca(x, k, sigma):
    K = get_gaussian_rbf_kernel_matrix(x, sigma)
    K = center_kernel_matrix(K)
    principal_components = []
    lambdas = []
    for _ in range(k):
        lambdak, ev = power_iteration(K)
        principal_components.append(ev)
        lambdas.append(lambdak)
        K = subtract_pc(K, ev, lambdak)
    return principal_components, lambdas


def project_data_onto_pc(data, pc, lambda_, sigma):
    K = get_gaussian_rbf_kernel_matrix(data, sigma)
    alpha = [pc_i / math.sqrt(lambda_) for pc_i in pc]
    return alpha_t_k(alpha, K)


def visualize_projected_data(kpcs, lambdas, data, sigma):
    x_ = project_data_onto_pc(data, kpcs[0], lambdas[0], sigma)
    y_ = project_data_onto_pc(data, kpcs[1], lambdas[1], sigma)
    col = [1] * 50 + [2] * 50
    plt.scatter(x_, y_, c=col)
    plt.xlabel('KPC1')
    plt.ylabel('KPC2')
    plt.title('Ellipse Data Projected onto PCs; sigma = {}'.format(sigma))
    plt.show()


if __name__ == '__main__':
    plot_data = False
    x_data = generate_sample_data()
    if plot_data:
        x = [p[0] for p in x_data]
        y = [p[1] for p in x_data]
        plt.scatter(x, y, c=[1] * 50 + [2] * 50)
        plt.title('Simulated Ellipse Data')
        plt.show()
    for sigma in [.01, .1, 1, 3, 5, 10, 100, 1000]:
        sigma_ = sigma
        kpcs, lambdas = perform_kernel_pca(
            x_data,
            2,
            sigma)
        visualize_projected_data(kpcs, lambdas, x_data, sigma)
