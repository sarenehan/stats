import os
import pickle
from utils.redis_utils import cache
import torch
import random
from torch.autograd import Variable
from utils.coco_utils import load_category_level_data_hw2
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
# from utils.graph_utils import draw_error_over_time


n_hidden_nodes = 250
if 'stewart' in os.getenv('PYTHONPATH').lower():
    save_dir = '/Users/stewart/projects/stats/548/model_results/'
else:
    save_dir = '/home/ubuntu/548/'


def initialize_weights(n_pixels, n_hidden_nodes, n_outputs):
    w1 = Variable(
        (torch.rand(n_pixels, n_hidden_nodes) - .5) / 1000,
        requires_grad=True)
    w2 = Variable((torch.rand(n_hidden_nodes, n_outputs) - .5) / 1000,
                  requires_grad=True)
    return w1, w2


def squared_loss(y_hat, y):
    return (1 / 2) * (y - y_hat).pow(2)


def predict(x, w1, w2):
    layer_one_out = F.relu(x.matmul(w1))
    return layer_one_out.matmul(w2)


def compute_error(w1, w2, x, y):
    if len(x.size()) == 1:
        n = 1
    else:
        n = x.size()[0]
    y_hat = predict(x, w1, w2)
    return ((1 / n) * squared_loss(
        y_hat, y).sum())


def get_average_precision_score(x_data, y_data, w1, w2):
    return average_precision_score(y_data.data, predict(x_data, w1, w2).data)


@cache.cached(timeout=60 * 60 * 24 * 60)
def train_mlp(
        training_rate,
        mini_batch_size=1,
        feature_size='tiny'):
    x_data, y_data = load_category_level_data_hw2(feature_size, 'train2014')
    x_val, y_val = load_category_level_data_hw2(feature_size, 'val2014')
    x_data = Variable(torch.from_numpy(x_data), requires_grad=True)
    y_data = Variable(torch.from_numpy(y_data).float(), requires_grad=True)
    x_val = Variable(torch.from_numpy(x_val))
    y_val = Variable(torch.from_numpy(y_val).float())
    w1, w2 = initialize_weights(
        x_data.shape[1],
        n_hidden_nodes,
        y_data.shape[1]
    )
    indexes = list(range(x_data.size()[0]))
    train_error = float('inf')
    half_epoch = int(x_data.size()[0] / (2 * mini_batch_size))
    print('Half ephoc: {}'.format(half_epoch))
    train_errors = []
    val_errors = []
    train_average_precision_scores = []
    val_average_precision_scores = []
    iteration_num = 0
    while train_error > 0.01 and iteration_num < 150000:
        if not iteration_num % half_epoch:
            train_error = float(compute_error(w1, w2, x_data, y_data))
            val_error = float(compute_error(w1, w2, x_val, y_val))
            train_average_precision_score = get_average_precision_score(
                x_data, y_data, w1, w2)
            val_average_precision_score = get_average_precision_score(
                x_val, y_val, w1, w2)
            print('\n\n{}'.format(iteration_num))
            print('Train Error: {}'.format(train_error))
            print('Train Avg Prec Score: {}'.format(
                train_average_precision_score))
            print('Validation Error: {}'.format(val_error))
            print('Validation Avg Prec Score: {}'.format(
                val_average_precision_score))
            train_errors.append(train_error)
            val_errors.append(val_error)
            train_average_precision_scores.append(
                train_average_precision_score)
            val_average_precision_scores.append(
                val_average_precision_score)
        iteration_num += 1
        idx = random.sample(indexes, mini_batch_size)
        err_est = compute_error(
            w1, w2, x_data[idx], y_data[idx])
        err_est.backward()
        w1.data = w1.data - (training_rate * w1.grad.data)
        w1.grad.data.zero_()
        w2.data = w2.data - (training_rate * w2.grad.data)
        w2.grad.data.zero_()
    return (
        train_errors,
        val_errors,
        train_average_precision_scores,
        val_average_precision_scores,
        w1.data,
        w2.data
    )


@cache.cached(timeout=60 * 60 * 24 * 60)
def train_mlp_nesterov(
        training_rate,
        mini_batch_size=1,
        feature_size='small'):
    x_data, y_data = load_category_level_data_hw2(feature_size, 'train2014')
    x_val, y_val = load_category_level_data_hw2(feature_size, 'val2014')
    x_data = Variable(torch.from_numpy(x_data), requires_grad=True)
    y_data = Variable(torch.from_numpy(y_data).float(), requires_grad=True)
    x_val = Variable(torch.from_numpy(x_val))
    y_val = Variable(torch.from_numpy(y_val).float())
    w1, w2 = initialize_weights(
        x_data.shape[1],
        n_hidden_nodes,
        y_data.shape[1]
    )
    v_t1 = Variable(torch.zeros(w1.shape), requires_grad=True)
    v_t2 = Variable(torch.zeros(w2.shape), requires_grad=True)
    indexes = list(range(x_data.size()[0]))
    train_error = float('inf')
    half_epoch = int(x_data.size()[0] / (2 * mini_batch_size))
    print('Half ephoc: {}'.format(half_epoch))
    train_errors = []
    val_errors = []
    train_average_precision_scores = []
    val_average_precision_scores = []
    iteration_num = 0
    while train_error > 0.01 and iteration_num < 150000:
        if not iteration_num % half_epoch:
            train_error = float(compute_error(w1, w2, x_data, y_data))
            val_error = float(compute_error(w1, w2, x_val, y_val))
            train_average_precision_score = get_average_precision_score(
                x_data, y_data, w1, w2)
            val_average_precision_score = get_average_precision_score(
                x_val, y_val, w1, w2)
            print('\n\n{}'.format(iteration_num))
            print('Train Error: {}'.format(train_error))
            print('Train Avg Prec Score: {}'.format(
                train_average_precision_score))
            print('Validation Error: {}'.format(val_error))
            print('Validation Avg Prec Score: {}'.format(
                val_average_precision_score))
            train_errors.append(train_error)
            val_errors.append(val_error)
            train_average_precision_scores.append(
                train_average_precision_score)
            val_average_precision_scores.append(
                val_average_precision_score)
        iteration_num += 1
        idx = random.sample(indexes, mini_batch_size)
        err_est = compute_error(
            w1 - 0.9 * v_t1, w2 - 0.9 * v_t2, x_data[idx], y_data[idx])
        err_est.backward()
        v_t1.data = (0.9 * v_t1.data) + (training_rate * (
            w1.grad.data - (0.9 * v_t1.grad.data)))
        v_t2.data = (0.9 * v_t2.data) + (training_rate * (
            w2.grad.data - (0.9 * v_t2.grad.data)))
        w1.data = w1.data - v_t1.data
        w2.data = w2.data - v_t2.data
        w1.grad.data.zero_()
        w2.grad.data.zero_()
        v_t1.grad.data.zero_()
        v_t2.grad.data.zero_()
    return (
        train_errors,
        val_errors,
        train_average_precision_scores,
        val_average_precision_scores,
        w1.data,
        w2.data
    )


def get_mlp_sgd():
    mini_batch_size = 10
    training_rate = .0001
    print('Training rate: {}'.format(training_rate))
    print('Batch size: {}'.format(mini_batch_size))
    print('N Hidden Nodes: {}'.format(n_hidden_nodes))
    (
        train_errors,
        val_errors,
        train_average_precision_scores,
        val_average_precision_scores,
        w1,
        w2
    ) = train_mlp(training_rate, mini_batch_size=mini_batch_size)
    with open(save_dir + '{}_mlp_weights.pkl'.format(
            n_hidden_nodes), 'wb') as f:
        pickle.dump((w1, w2), f)
    with open(save_dir + '{}_mlp_errors.pkl'.format(
            n_hidden_nodes), 'wb') as f:
        pickle.dump((
            train_errors,
            val_errors,
            train_average_precision_scores,
            val_average_precision_scores), f)


def get_mlp_nesterov():
    mini_batch_size = 10
    training_rate = .0001
    print('Training rate: {}'.format(training_rate))
    print('Batch size: {}'.format(mini_batch_size))
    print('N Hidden Nodes: {}'.format(n_hidden_nodes))
    (
        train_errors,
        val_errors,
        train_average_precision_scores,
        val_average_precision_scores,
        w1,
        w2
    ) = train_mlp_nesterov(training_rate, mini_batch_size=mini_batch_size)
    with open(save_dir + '{}_mlp_weights_nesterov.pkl'.format(
            n_hidden_nodes), 'wb') as f:
        pickle.dump((w1, w2), f)
    with open(save_dir + '{}_mlp_errors_nesterov.pkl'.format(
            n_hidden_nodes), 'wb') as f:
        pickle.dump((
            train_errors,
            val_errors,
            train_average_precision_scores,
            val_average_precision_scores), f)


if __name__ == '__main__':
    get_mlp_sgd()
    # get_mlp_nesterov()
