import numpy as np
import math

rnd = np.random.RandomState()


def gaussian_rbf_kernel(x1, x2, sigma):
    return math.exp(
        -sum((x1 - x2)**2) / (sigma ** 2))


def get_gaussian_rbf_kernel_matrix(data, sigma):
    n = len(data)
    kernel_matrix = np.array([[None] * n] * n)
    for i in range(n):
        for j in range(n):
            kernel_matrix[i][j] = gaussian_rbf_kernel(data[i], data[j], sigma)
    return kernel_matrix


def norm(vect):
    return math.sqrt(sum(v**2 for v in vect))


def center_kernel_matrix(K):
    n = len(K)
    off_diagonal_ones = np.array(
        [[int(i != j) - (1 / n) for i in range(n)] for j in range(n)]
    )
    return np.dot(off_diagonal_ones, np.dot(K, off_diagonal_ones))


def power_iteration(x, max_iterations=500):
    vk = rnd.normal(size=(1, len(x)))[0]
    for k in range(max_iterations):
        zk = np.dot(x, vk)
        vk = np.array([zi / norm(zk) for zi in zk])
    x_vk = np.dot(x, vk)
    lambdak = sum(vk[i] * x_vk[i] for i in range(len(vk)))
    return lambdak, vk


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


def perform_kernel_pca(x, k, sigma):
    K = get_gaussian_rbf_kernel_matrix(x, sigma)
    print('Centering kernel matrix ... ')
    K = center_kernel_matrix(K)
    print('Finished centering kernel matrix')
    principal_components = []
    lambdas = []
    for idx in range(k):
        print(idx)
        lambdak, ev = power_iteration(K)
        principal_components.append(ev)
        lambdas.append(lambdak)
        K = subtract_pc(K, ev, lambdak)
    return principal_components, lambdas, K


def project_data_onto_kpc(pc, lambda_, sigma, K):
    alpha = np.array([pc_i / math.sqrt(lambda_) for pc_i in pc])
    return np.dot(alpha.T, K)


def project_data_onto_kpcs(data, k, sigma):
    pcs, lambdas, K = perform_kernel_pca(data, k, sigma)
    return np.array(
        [project_data_onto_kpc(pc, lambda_, sigma, K)
         for pc, lambda_ in zip(pcs, lambdas)]
    ).T
