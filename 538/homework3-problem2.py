import numpy as np
import math
from statistics import mean
from sklearn.decomposition import PCA

rnd = np.random.RandomState(1)
x1 = rnd.normal(size=(20, 50))
x2 = rnd.normal(5, 1, size=(20, 50))
x3 = rnd.normal(10, 1, size=(20, 50))
y = np.repeat([0, 1, 2], 20)
x = np.concatenate((x1, x2, x3))


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


def norm(vect):
    return math.sqrt(sum(v**2 for v in vect))


def compute_matrix_vector_product(m, v):
    """
    Computes the product of square matrix M * V
    """
    return [
        sum(m[i][j] * v[j] for j in range(len(m[i])))
        for i in range(len(m))
    ]


def power_iteration(x, max_iterations=1000, existing_evs=[]):
    vk = rnd.normal(size=(1, len(x)))[0]
    for ev in existing_evs:
        dot = sum(ev[i] * vk[i] for i in range(len(ev)))
        norm_ev = norm(ev)
        vk = [vk[i] - (dot * ev[i] / norm_ev) for i in range(len(ev))]
    for k in range(max_iterations):
        zk = compute_matrix_vector_product(x, vk)
        vk = [zi / norm(zk) for zi in zk]
    x_vk = compute_matrix_vector_product(x, vk)
    lambdak = sum(vk[i] * x_vk[i] for i in range(len(vk)))
    return lambdak, vk


def compute_cosine_similarity(v1, v2):
    return sum(
        v1[i] * v2[i] for i in range(len(v1))
    ) / (math.sqrt(sum(x_i ** 2 for x_i in v1)) * math.sqrt(
        sum(x_i ** 2 for x_i in v2)))


def compare_pc1_to_sklearn_pc1(pc1, x):
    pca = PCA(n_components=len(x[0]))
    pca.fit(x)
    sklearn_pc1 = pca.components_[0]
    cosine_similarity = compute_cosine_similarity(sklearn_pc1, pc1)
    print(cosine_similarity)
    print('Cosine Similarity between sklean pc1 and my pc1: {}'.format(
        cosine_similarity))
    print('They are essentially equivalent')


def perform_pca(x, k):
    x_cov = get_covariance_matrix(x)
    principal_components = []
    lambdas = []
    for _ in range(k):
        lambdak, ev = power_iteration(x_cov, existing_evs=principal_components)
        principal_components.append(ev)
        lambdas.append(lambdak)
        norm_ev = norm(ev)
        x = np.array([
            [
                x[i][j] - (
                    (sum(x[i][k] * ev[k] for k in range(len(ev))) * ev[j]
                     ) / norm_ev) for j in range(len(ev))
            ]
            for i in range(len(x))
        ])
        x_cov = get_covariance_matrix(x)
    return principal_components, lambdas


if __name__ == '__main__':
    x_cov = get_covariance_matrix(x)
    lambda1, e1 = power_iteration(x_cov)
    compare_pc1_to_sklearn_pc1(e1, x)
    # Cosine Similarity between sklean pc1 and my pc1: 0.9999999999999999
    # They are essentially equivalent

    print('\n\n')
    pca = PCA(n_components=len(x[0]))
    pca.fit(x)
    pcs, lambdas = perform_pca(x, 2)
    for idx, pc in enumerate(pcs):
        csm = compute_cosine_similarity(pc, pca.components_[idx])
        print('Cosine similarity between my pc{} and sklearn pc{}: {}'.format(
            idx + 1, idx + 1, csm))
    # Cosine similarity between my pc1 and sklearn pc1: 0.9999999999999999
    # Cosine similarity between my pc2 and sklearn pc2: 1.0000000000000002
