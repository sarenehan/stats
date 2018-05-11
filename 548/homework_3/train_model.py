import pickle
import random
import torch
import numpy as np
from torch.autograd import Variable
from sklearn.metrics import average_precision_score
from utils.coco_utils import dataDir


def load_data_for_category():
    with open(dataDir + '/small_train2014_feature_23.pkl', 'rb') as f:
        data = pickle.load(f)
    return data


def initialize_weights(d):
    return Variable((torch.rand(d) - .5) / 1000, requires_grad=True)


def compute_error_logistic_regression(w, lambda_, x, y):
    preds = 1 / (1 + torch.exp(-x.matmul(w)))
    loss = (0.5 * (y - preds).pow(2)) + (lambda_ / 2) * w.norm()
    return loss.sum()


def predict(x, w):
    return 1 / (1 + torch.exp(-x.matmul(w)))


def get_average_precision_score(x_data, y_data, w):
    return average_precision_score(y_data.data, predict(x_data, w).data)


def construct_batches(n_, batch_size):
    indices = list(range(n_))
    random.shuffle(indices)
    batches = []
    start_idx = 0
    while start_idx < n_:
        batches.append(indices[start_idx:start_idx+batch_size])
        start_idx += batch_size
    return batches


def train_logistic_regression(x, y, lambda_, training_rate, batch_size):
    iter_ = 0
    w = initialize_weights(x.shape[1])
    v_t = Variable(torch.zeros(w.shape), requires_grad=True)
    total_error = float('inf')
    errors = []
    n_ = len(x)
    half_epoch = int(x.size()[0] / (2 * batch_size))
    while total_error > .001 and iter_ < 30000:
        total_error = float(
            compute_error_logistic_regression(w, lambda_, x, y))
        ap_score = get_average_precision_score(x, y, w)
        print('{}'.format(iter_))
        print('\tError: {}'.format(total_error))
        print('\tAP Score: {}'.format(ap_score))
        errors.append(total_error)
        batches = construct_batches(n_, batch_size)
        for idx, batch in enumerate(batches):
            if idx == half_epoch:
                total_error = float(compute_error_logistic_regression(
                    w, lambda_, x, y))
                ap_score = get_average_precision_score(x, y, w)
                print('{}'.format(iter_))
                print('\tError: {}'.format(total_error))
                print('\tAP Score: {}'.format(ap_score))
                errors.append(total_error)
            # Nesterov momentum
            err_est = compute_error_logistic_regression(
                w - 0.9 * v_t, lambda_, x[batch], y[batch])
            err_est.backward()
            v_t.data = (0.9 * v_t.data) + (training_rate * (
                w.grad.data - (0.9 * v_t.grad.data)))
            w.data = w.data - v_t.data
            w.grad.data.zero_()
            v_t.grad.data.zero_()
            iter_ += 1
    return w, errors


def train_model(
        data, lambda_, training_rate, batch_size, fit_intercept=True):
    positive_features = data['positive_features']
    negative_features = data['negative_features']
    y = np.array([1] * len(positive_features) + [0] * len(negative_features))
    x = np.array(positive_features + negative_features)
    if fit_intercept:
        x = np.hstack((np.ones((len(x), 1)), x))
    x = Variable(torch.from_numpy(x).float(), requires_grad=True)
    y = Variable(torch.from_numpy(y).float(), requires_grad=True)
    return train_logistic_regression(
        x, y, lambda_, training_rate, batch_size)


if __name__ == '__main__':
    data = load_data_for_category()
    lambda_ = 0
    training_rate = 0.000001
    batch_size = 10
    train_model(data, lambda_, training_rate, batch_size)
