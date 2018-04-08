#!/usr/bin/env python3

from pyspark import SparkContext, SparkConf
import math
import numpy as np
import pickle
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


mnist_data_location = '/Users/stewart/projects/stats/538/mnist_data.pkl'
hidden_layer_size = 1600
rnd = np.random.RandomState(1)


def preprocess_data(x_data, y_data, test_size=0.5):
    y_data = np.array(
        [[int(point == idx) for idx in range(10)] for point in y_data]
    )
    var_x = np.array([point or 1 for point in x_data.var(axis=0)])
    x_data = (x_data - x_data.mean(axis=0)) / var_x
    return train_test_split(x_data, y_data, test_size=test_size)


def gaussian_rbf_kernel(x1, x2, sigma):
    return np.exp(-sum((x1 - x2) ** 2) / sigma ** 2)


def get_gaussian_rbf_kernel_matrix(data, sigma):
    n = len(data)
    kernel_matrix = np.zeros((n, n))
    # Method 1
    # cols = sc.parallelize(list(range(n)))
    # for i in range(n):
    #     kernel_matrix[i] = cols.map(
    #         lambda j: gaussian_rbf_kernel(data[i], data[j], sigma)
    #     ).collect()
    # return kernel_matrix


    # Method 2
    # dat = sc.parallelize(data)
    # kernel_matrix = np.array(dat.map(
    #     lambda row: np.array(
    #         [gaussian_rbf_kernel(row, data[j], sigma) for j in range(n)])).collect())

    # # Method 3
    for i in range(n):
        for j in range(i, n):
            kernel_matrix[i][j] = gaussian_rbf_kernel(data[i], data[j], sigma)

    return kernel_matrix - np.diag(
        np.diag(kernel_matrix)) + np.transpose(kernel_matrix)


def center_kernel_matrix(K):
    K = K - np.mean(K, axis=1)
    return K - np.mean(K, axis=0)


def power_iteration(x, max_iterations=100):
    vk = rnd.normal(size=(1, len(x)))[0]
    for k in range(max_iterations):
        zk = np.dot(x, vk)
        vk = zk / np.linalg.norm(zk)
    x_vk = np.dot(x, vk)
    lambdak = sum(vk[i] * x_vk[i] for i in range(len(vk)))
    return lambdak, vk


def subtract_pc(K, ev, lambda_):
    n = len(ev)
    return K - lambda_ * np.multiply(ev.reshape((n, 1)), ev.reshape((1, n)))


def perform_kernel_pca(x, k, sigma):
    K = get_gaussian_rbf_kernel_matrix(x, sigma)
    K_init = center_kernel_matrix(K)
    K_current = K_init
    principal_components = []
    lambdas = []
    for idx in range(k):
        lambdak, ev = power_iteration(K_current)
        principal_components.append(ev)
        lambdas.append(lambdak)
        K_current = subtract_pc(K_current, ev, lambdak)
    return principal_components, lambdas, K


def project_data_onto_kpc(pc, lambda_, sigma, K):
    alpha = pc / math.sqrt(lambda_)
    return np.dot(alpha.T, K)


def project_data_onto_kpcs(data, k, sigma):
    pcs, lambdas, K = perform_kernel_pca(data, k, sigma)
    return np.array(
        [project_data_onto_kpc(pc, lambda_, sigma, K)
         for pc, lambda_ in zip(pcs, lambdas)]
    ).T


def download_mnist_data():
    data = fetch_mldata('MNIST Original')
    with open(mnist_data_location, 'wb') as f:
        pickle.dump(data, f)


def load_mnist_data():
    data = fetch_mldata('MNIST Original')
    x_data = data.data
    y_data = data.target
    return x_data, y_data


def get_layer_weight_init(n_inputs, layer_size):
    w = rnd.normal(size=(n_inputs, layer_size))
    beta = rnd.normal(size=(layer_size,))
    return w, beta


def get_output_weight_init(input_size, n_outputs=10):
    return rnd.normal(size=(input_size, n_outputs))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def layer_function(x, w, beta):
    return sigmoid(np.dot(w.T, x) + beta)


def softmax(x):
    return np.exp(x) / sum(np.exp(x))


def output_layer(x, v):
    return softmax(np.dot(v.T, x))


def compute_error(model_output, y_data):
    preds = np.argmax(model_output, axis=1)
    true_values = np.argmax(y_data, axis=1)
    incorrect_preds = (np.argmax(y_data, axis=1) != preds)
    return sum(incorrect_preds) / len(incorrect_preds)


def predicted_number(model_output):
    return np.where(model_output == model_output.max())[0][0]


def predict(x, w1, beta_1, w2, beta_2, v):
    return predicted_number(
        output_layer(
            layer_function(
                layer_function(x, w1, beta_1),
                w2, beta_2),
            v
        )
    )


def compute_misclassification_error(x_data, y_data, w1, beta_1, w2, beta_2, v):
    incorrect_predictions = []
    for idx, x in enumerate(x_data):
        incorrect_predictions.append(
            y_data[idx][predict(x, w1, beta_1, w2, beta_2, v)] != 1
        )
    error = sum(incorrect_predictions) / len(incorrect_predictions)
    print('Error rate: {}'.format(error))
    return error


def get_weight_partition_updates(
        x_data,
        y_data,
        w1,
        beta_1,
        w2,
        beta_2,
        v
    ):
    n = len(x_data)
    z1 = x_data.dot(w1) + beta_1
    a1 = sigmoid(z1)
    z2 = a1.dot(w2) + beta_2
    a2 = sigmoid(z2)
    z3 = a2.dot(v)
    preds = np.array([softmax(row) for row in z3])
    d4 = preds - y_data
    dv = a2.T.dot(d4)
    d3 = np.multiply(d4.dot(v.T), np.multiply(a2, 1 - a2))
    dw2 = a1.T.dot(d3)
    d2 = np.multiply(d3.dot(w2.T), np.multiply(a1, 1 - a1))
    db2 = sum(d3)
    dw1 = x_data.T.dot(d2)
    db1 = sum(d2)
    error = compute_error(preds, y_data)
    return np.array([dv, dw2, db2, dw1, db1, error])



def update_weights(
        x_data,
        y_data,
        w1,
        beta_1,
        w2,
        beta_2,
        v,
        alpha,
        lambda_,
        vy,
        w2y,
        beta_2y,
        w1y,
        beta_1y,
        t
        ):
    split_locations = list(range(
        round(len(x_data) / n_splits),
        len(x_data),
        round(len(x_data) / n_splits)
    ))
    indeces = np.split(list(range(len(x_data))), split_locations)
    weight_updates = np.array(sc.parallelize(indeces, n_splits).map(
        lambda index: get_weight_partition_updates(
            x_data[index],
            y_data[index],
            w1y,
            beta_1y,
            w2y,
            beta_2y,
            vy
        )
    ).collect()).mean(axis=0)
    dv = weight_updates[0]
    dw2 = weight_updates[1]
    db2 = weight_updates[2]
    dw1 = weight_updates[3]
    db1 = weight_updates[4]
    error = weight_updates[5]
    print('Error rate: {}'.format(error))
    vinit = v
    w2init = w2
    beta_2init = beta_2
    w1init = w1
    beta_1init = beta_1

    w2 = w2y - (alpha * (dw2 + (lambda_ * w2y)))
    beta_2 = beta_2y - (alpha * db2)
    w1 = w1y - (alpha * (dw1 + (lambda_ * w1y)))
    beta_1 = beta_1y - (alpha * db1)
    v = vy - (alpha * (dv + (lambda_ * vy)))

    vy = v + ((t / (t + 3)) * (v - vinit))
    w2y = w2 + ((t / (t + 3)) * (w2 - w2init))
    beta_2y = beta_2 + ((t / (t + 3)) * (beta_2 - beta_2init))
    w1y = w1 + ((t / (t + 3)) * (w1 - w1init))
    beta_1y = beta_1 + ((t / (t + 3)) * (beta_1 - beta_1init))

    return (
        w1,
        beta_1,
        w2,
        beta_2,
        v,
        error,
        vy,
        w2y,
        beta_2y,
        w1y,
        beta_1y,
    )


def train_mlp(x_data, y_data, alpha, lambda_, stopping_error_rate):
    point = x_data[0]
    n_inputs = len(point)
    w1, beta_1 = get_layer_weight_init(n_inputs, hidden_layer_size)
    w2, beta_2 = get_layer_weight_init(hidden_layer_size, hidden_layer_size)
    v = get_output_weight_init(hidden_layer_size)
    vy = v
    w2y = w2
    beta_2y = beta_2
    w1y = w1
    beta_1y = beta_1
    classification_err_rate = 1
    t = 1
    while classification_err_rate > stopping_error_rate:
        if not t % 50:
            print('\nValidation Error: {}\n'.format(
                compute_misclassification_error(
                    x_test, y_test, w1, beta_1, w2, beta_2, v)
                )
            )
        (
            w1,
            beta_1,
            w2,
            beta_2,
            v,
            classification_err_rate,
            vy,
            w2y,
            beta_2y,
            w1y,
            beta_1y,
        ) = update_weights(
            x_data,
            y_data,
            w1,
            beta_1,
            w2,
            beta_2,
            v,
            alpha,
            lambda_,
            vy,
            w2y,
            beta_2y,
            w1y,
            beta_1y,
            t
        )
        t = t + 1
    return w1, beta_1, w2, beta_2, v


def factorize_y_data(y_data):
    return np.argmax(y_data, axis=1)


def build_linear_model(x_data, y_data):
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=.3)
    model = LogisticRegression().fit(x_train, y_train)
    print(1 - model.score(x_test, y_test))
    return 1 - model.score(x_test, y_test)


def layerwise_kernel_pca(
        x_data,
        y_data,
        w1,
        beta_1,
        w2,
        beta_2,
        v):
    errors_by_layer = [[], [], [], []]
    z1 = x_data.dot(w1) + beta_1
    layer_one_output = sigmoid(z1)
    z2 = layer_one_output.dot(w2) + beta_2
    layer_two_output = sigmoid(z2)
    z3 = layer_two_output.dot(v)
    final_layer_output = np.array([softmax(row) for row in z3])
    y_data = factorize_y_data(y_data)
    for sigma in [.01, .1, 1, 10]:
        layer_zero_kps = project_data_onto_kpcs(x_data, 10, sigma)
        errors_by_layer[0].append(build_linear_model(layer_zero_kps, y_data))
        layer_one_kps = project_data_onto_kpcs(layer_one_output, 10, sigma)
        errors_by_layer[1].append(build_linear_model(layer_one_kps, y_data))
        layer_two_kps = project_data_onto_kpcs(layer_two_output, 10, sigma)
        errors_by_layer[2].append(build_linear_model(layer_two_kps, y_data))
        final_layer_kps = project_data_onto_kpcs(final_layer_output, 10, sigma)
        errors_by_layer[3].append(build_linear_model(final_layer_kps, y_data))
    return [min(errors) for errors in errors_by_layer]


def save_model(w1, beta_1, w2, beta_2, v, alpha, lambda_):
    if not local:
        sc.parallelize(w1).repartition(1).saveAsTextFile(
            's3://stat-538/models/w1_mlp_model_{}_{}.txt'.format(
            alpha, lambda_)
        )
        sc.parallelize(w2).repartition(1).saveAsTextFile(
            's3://stat-538/models/w2_mlp_model_{}_{}.txt'.format(
            alpha, lambda_)
        )
        sc.parallelize(v).repartition(1).saveAsTextFile(
            's3://stat-538/models/v_mlp_model_{}_{}.txt'.format(
            alpha, lambda_)
        )
        sc.parallelize(beta_1).repartition(1).saveAsTextFile(
            's3://stat-538/models/beta_1_mlp_model_{}_{}.txt'.format(
            alpha, lambda_)
        )
        sc.parallelize(beta_2).repartition(1).saveAsTextFile(
            's3://stat-538/models/beta_2_mlp_model_{}_{}.txt'.format(
            alpha, lambda_)
        )
    else:
        save_location_prefix = '/Users/stewart/projects/stats/538/models/'
        file_name = 'mlp_model_{}_{}'.format(
            alpha, lambda_)
        with open(save_location_prefix + file_name, 'wb') as f:
            pickle.dump((w1, beta_1, w2, beta_2, v), f)


def save_error_rates(error_rates, alpha, lambda_):
    sc.parallelize(error_rates).repartition(1).saveAsTextFile(
        's3://stat-538/error_rates/error_rates_{}_{}.pkl'.format(
        alpha, lambda_)
    )


def plot_effect_of_weight_penalty(x_data, y_data):
    x_data_for_kpca, _, y_data_for_kpca, _  = train_test_split(
        x_data, y_data, test_size=.9)
    for lambda_ in [0, 0.0001, 0.001, 0.01, .1]:
        print('Lambda: {}'.format(lambda_))
        w1, beta_1, w2, beta_2, v = train_mlp(
            x_data, y_data, alpha, lambda_, stopping_error_rate)
        print('\n\n\nValidation Error: {}'.format(
            compute_misclassification_error(
                x_test, y_test, w1, beta_1, w2, beta_2, v)))
        try:
            save_model(w1, beta_1, w2, beta_2, v, alpha, lambda_)
        except:
            print('\n\nCould not save model...\n\n')
        error_rates = layerwise_kernel_pca(
                             x_data_for_kpca,
                             y_data_for_kpca,
                             w1,
                             beta_1,
                             w2,
                             beta_2,
                             v)
        print('Kernel PCA error rates {}'.format(error_rates))
        if local:
            plt.plot(error_rates, label='Weight Penalty = {}'.format(lambda_))
    if local:
        plt.legend(loc='best')
        plt.xlabel('layer l')
        plt.ylabel('error e(d=10)')
        plt.title('Effect of Weight Penalty')
        plt.show()


def load_model(lambda_, alpha='1e-05'):
    prefix = '/Users/stewart/projects/stats/538/models/'
    weights = []
    for weight in ['beta_1', 'beta_2', 'w1', 'w2', 'v']:
        weights.append(
            np.genfromtxt(
                prefix +
                '{}_mlp_model_{}_{}.pkl/part-00000'.format(
                    weight, alpha, lambda_)
            )
        )
    return weights


if __name__ == '__main__':
    lambda_ = 0
    local = False
    if local:
        conf = SparkConf().setAppName("App")
        conf = (conf.setMaster('local[*]')
                .set('spark.executor.memory', '50G')
                .set('spark.driver.memory', '50G')
                .set('spark.driver.maxResultSize', '10G'))
        sc = SparkContext(conf=conf)
        x_, y_ = load_mnist_data()
        n_splits = 4
        test_size = .999
        alpha = .00001
        stopping_error_rate = 0.5
        from matplotlib import pyplot as plt
    else:
        sc = SparkContext(appName="DeepLearning")
        sc.setCheckpointDir('s3://stat-538/checkpoint/')
        data = sc.textFile('s3://stat-538/mnist_data.csv')
        data = data.map(lambda l: np.array(l.split(',')).astype(
            float).astype(int))
        x_ = np.array(
            data.map(lambda l: np.array(l[1:]).astype(int)).collect())
        y_ = np.array(data.map(lambda l: int(float(l[0]))).collect())
        n_splits = 18
        test_size = .9
        alpha = .00001
        stopping_error_rate = 0.1

    x_data, x_test, y_data, y_test = preprocess_data(x_, y_, test_size=test_size)
    x_test, _, y_test, _  = train_test_split(x_test, y_test, test_size=.99)
    plot_effect_of_weight_penalty(x_data, y_data)
