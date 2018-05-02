import os
import torch
import random
import pickle
from copy import copy
from utils.redis_utils import cache
from torch.autograd import Variable
from utils.coco_utils import load_category_level_data_hw2
from sklearn.metrics import average_precision_score
# from utils.graph_utils import draw_error_over_time


if 'stewart' in os.getenv('PYTHONPATH').lower():
    save_dir = '/Users/stewart/projects/stats/548/model_results/'
else:
    save_dir = '/home/ubuntu/548/'


def initialize_weights(d, k):
    return Variable((torch.rand(d, k) - .5) / 1000, requires_grad=True)


def squared_loss(y_hat, y):
    return (1 / 2) * (y - y_hat).pow(2)


def log_loss(y_hat, y):
    return y.mul(torch.log(1 + torch.exp(-y_hat))) + (1 - y).mul(
        torch.log(1 + torch.exp(y_hat)))


def predict(x, w):
    return x.matmul(w)


def compute_error(w, x, y, lambda_, loss_function):
    if len(x.size()) == 1:
        n = 1
    else:
        n = x.size()[0]
    regularization = (lambda_ / 2) * w.norm()
    y_hat = predict(x, w)
    return regularization + ((1 / n) * loss_function(
        y_hat, y).sum())


def get_average_precision_score(x_data, y_data, w):
    return average_precision_score(y_data.data, predict(x_data, w).data)


@cache.cached(timeout=60 * 60 * 24 * 60)
def train_linear_regression(
        lambda_,
        training_rate,
        loss_function=squared_loss,
        mini_batch_size=1,
        feature_size='tiny',
        stopping_criterion=10):
    x_data, y_data = load_category_level_data_hw2(feature_size, 'train2014')
    x_val, y_val = load_category_level_data_hw2(feature_size, 'val2014')
    x_data = Variable(torch.from_numpy(x_data), requires_grad=True)
    y_data = Variable(torch.from_numpy(y_data).float(), requires_grad=True)
    x_val = Variable(torch.from_numpy(x_val))
    y_val = Variable(torch.from_numpy(y_val).float())
    w = initialize_weights(x_data.shape[1], y_data.shape[1])
    indexes = list(range(x_data.size()[0]))
    train_error = float('inf')
    grad_norm = 100
    iteration_num = 0
    half_epoch = int(x_data.size()[0] / (2 * mini_batch_size))
    print('Half ephoc: {}'.format(half_epoch))
    train_errors = []
    val_errors = []
    train_average_precision_scores = []
    val_average_precision_scores = []
    while grad_norm > stopping_criterion and iteration_num < 150000:
        if not iteration_num % half_epoch:
            train_error = compute_error(
                w, x_data, y_data, lambda_, loss_function)
            train_error.backward()
            grad_norm = copy(w.grad.data.norm())
            train_error = float(train_error)
            train_average_precision_score = get_average_precision_score(
                x_data, y_data, w)
            val_error = float(compute_error(
                w, x_val, y_val, lambda_, loss_function))
            val_average_precision_score = get_average_precision_score(
                x_val, y_val, w)
            print('\n\n{}'.format(iteration_num))
            print('Train Error: {}'.format(train_error))
            print('Train Avg Prec Score: {}'.format(
                train_average_precision_score))
            print('Validation Error: {}'.format(val_error))
            print('Validation Avg Prec Score: {}'.format(
                val_average_precision_score))
            print('Gradient Norm: {}'.format(grad_norm))
            train_errors.append(train_error)
            val_errors.append(val_error)
            train_average_precision_scores.append(
                train_average_precision_score)
            val_average_precision_scores.append(
                val_average_precision_score)
        iteration_num += 1
        idx = random.sample(indexes, mini_batch_size)
        err_est = compute_error(
            w, x_data[idx], y_data[idx], lambda_, loss_function)
        err_est.backward()
        w.data = w.data - (training_rate * w.grad.data)
        w.grad.data.zero_()
    return (
        train_errors,
        val_errors,
        train_average_precision_scores,
        val_average_precision_scores,
        w.data
    )


@cache.cached(timeout=60 * 60 * 24 * 60)
def train_linear_regression_nesterov(
        lambda_,
        training_rate,
        loss_function=squared_loss,
        mini_batch_size=1,
        feature_size='tiny',
        stopping_criterion=10):
    x_data, y_data = load_category_level_data_hw2(feature_size, 'train2014')
    x_val, y_val = load_category_level_data_hw2(feature_size, 'val2014')
    x_data = Variable(torch.from_numpy(x_data), requires_grad=True)
    y_data = Variable(torch.from_numpy(y_data).float(), requires_grad=True)
    x_val = Variable(torch.from_numpy(x_val))
    y_val = Variable(torch.from_numpy(y_val).float())
    w = initialize_weights(x_data.shape[1], y_data.shape[1])
    v_t = Variable(torch.zeros(w.shape), requires_grad=True)
    indexes = list(range(x_data.size()[0]))
    train_error = float('inf')
    grad_norm = 100
    iteration_num = 0
    half_epoch = int(x_data.size()[0] / (2 * mini_batch_size))
    print('Half ephoc: {}'.format(half_epoch))
    train_errors = []
    val_errors = []
    train_average_precision_scores = []
    val_average_precision_scores = []
    while grad_norm > stopping_criterion and iteration_num < 150000:
        if not iteration_num % half_epoch:
            train_error = compute_error(
                w, x_data, y_data, lambda_, loss_function)
            train_error.backward()
            grad_norm = copy(w.grad.data.norm())
            train_error = float(train_error)
            train_average_precision_score = get_average_precision_score(
                x_data, y_data, w)
            val_error = float(compute_error(
                w, x_val, y_val, lambda_, loss_function))
            val_average_precision_score = get_average_precision_score(
                x_val, y_val, w)
            print('\n\n{}'.format(iteration_num))
            print('Train Error: {}'.format(train_error))
            print('Train Avg Prec Score: {}'.format(
                train_average_precision_score))
            print('Validation Error: {}'.format(val_error))
            print('Validation Avg Prec Score: {}'.format(
                val_average_precision_score))
            print('Gradient Norm: {}'.format(grad_norm))
            train_errors.append(train_error)
            val_errors.append(val_error)
            train_average_precision_scores.append(
                train_average_precision_score)
            val_average_precision_scores.append(
                val_average_precision_score)
        iteration_num += 1
        idx = random.sample(indexes, mini_batch_size)
        err_est = compute_error(
            w - 0.9 * v_t, x_data[idx], y_data[idx], lambda_, loss_function)
        err_est.backward()
        v_t.data = (0.9 * v_t.data) + (training_rate * (
            w.grad.data - (0.9 * v_t.grad.data)))
        w.data = w.data - v_t.data
        w.grad.data.zero_()
        v_t.grad.data.zero_()
    return (
        train_errors,
        val_errors,
        train_average_precision_scores,
        val_average_precision_scores,
        w.data
    )


def train_sgd_model():
    mini_batch_size = 10
    lambda_ = 1
    training_rate = 0.000001
    print('Labmda: {}'.format(lambda_))
    print('Trainig Rate: {}'.format(training_rate))
    print('Mini batch size: {}'.format(mini_batch_size))
    (
        train_errors,
        val_errors,
        train_average_precision_scores,
        val_average_precision_scores,
        w
    ) = train_linear_regression(
        lambda_,
        training_rate,
        loss_function=squared_loss,
        mini_batch_size=mini_batch_size)
    print('Train Errors: {}'.format(train_errors))
    print('Val Errors: {}'.format(val_errors))
    print('Train APS: {}'.format(train_average_precision_scores))
    print('Val APS: {}'.format(train_average_precision_scores))
    print('w: {}'.format(w))
    with open(save_dir + '{}_linear_regression_w.pkl'.format(
            lambda_), 'wb') as f:
        pickle.dump(w, f)
    with open(save_dir + '{}_linear_regression_errors.pkl'.format(
            lambda_), 'wb') as f:
        pickle.dump((
            train_errors,
            val_errors,
            train_average_precision_scores,
            val_average_precision_scores,
            w), f)


def train_sgd_model_nesterov():
    mini_batch_size = 10
    lambda_ = 1
    training_rate = 0.0000001
    print('Labmda: {}'.format(lambda_))
    print('Trainig Rate: {}'.format(training_rate))
    print('Mini batch size: {}'.format(mini_batch_size))
    (
        train_errors,
        val_errors,
        train_average_precision_scores,
        val_average_precision_scores,
        w
    ) = train_linear_regression_nesterov(
        lambda_,
        training_rate,
        loss_function=squared_loss,
        mini_batch_size=mini_batch_size)
    print('Train Errors: {}'.format(train_errors))
    print('Val Errors: {}'.format(val_errors))
    print('Train APS: {}'.format(train_average_precision_scores))
    print('Val APS: {}'.format(train_average_precision_scores))
    print('w: {}'.format(w))
    with open(save_dir + '{}_linear_regression_w_nest.pkl'.format(
            lambda_), 'wb') as f:
        pickle.dump(w, f)
    with open(save_dir + '{}_linear_regression_errors_nest.pkl'.format(
            lambda_), 'wb') as f:
        pickle.dump((
            train_errors,
            val_errors,
            train_average_precision_scores,
            val_average_precision_scores,
            w), f)


if __name__ == '__main__':
    # train_sgd_model()
    train_sgd_model_nesterov()
