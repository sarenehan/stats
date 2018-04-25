import torch
from torch.autograd import Variable
from utils.coco_utils import load_data
from utils.redis_utils import cache
from homework_1.hw1_graphs import draw_error_over_time
import torch.nn.functional as F


def predict(x_data, w1, w2):
    layer_one_out = F.relu(x_data.matmul(w1))
    return layer_one_out.matmul(w2)


def compute_error(x_data, y_data, w1, w2):
    y_hat = predict(x_data, w1, w2)
    return (1 / x_data.size()[0]) * (
        0.5 * (y_data - y_hat).pow(2)
    ).sum()


def get_misclassification_error(x_data, y_data, w1, w2):
    preds = [1 if float(x) > 0 else -1 for x in predict(x_data, w1, w2)]
    incorrect_preds = [
        1 if (preds[idx] != int(y_data[idx])) else 0
        for idx in range(len(x_data))
    ]
    return sum(incorrect_preds) / len(incorrect_preds)


def print_misclassification_error(x_data, y_data, w1, w2):
    print('Misclassification Error: {}'.format(
        get_misclassification_error(x_data, y_data, w1, w2)))


def initialize_weights(d, h):
    w1 = Variable((torch.rand(d, h) - .5) / 1000, requires_grad=True)
    w2 = Variable((torch.rand(h) - .5) / 1000, requires_grad=True)
    return w1, w2


@cache.cached(timeout=60 * 60 * 24 * 60)
def train_mlp(n_hidden_layers, training_rate, stopping_criterion):
    x_data, y_data = load_data('train2014')
    x_test, y_test = load_data('test2014')
    x_val, y_val = load_data('val2014')
    d = len(x_data[0])
    x_data = Variable(torch.from_numpy(x_data), requires_grad=True)
    y_data = Variable(torch.from_numpy(y_data).float(), requires_grad=True)
    x_test = Variable(torch.from_numpy(x_test))
    y_test = Variable(torch.from_numpy(y_test).float())
    x_val = Variable(torch.from_numpy(x_val))
    y_val = Variable(torch.from_numpy(y_val).float())
    total_err = float('inf')
    idx = 0
    train_errors = []
    test_errors = []
    val_errors = []
    train_mc_errors = []
    test_mc_errors = []
    val_mc_errors = []
    w1, w2 = initialize_weights(d, n_hidden_layers)
    while float(total_err) > stopping_criterion:
        if not idx % 50:
            train_errors.append(float(compute_error(x_data, y_data, w1, w2)))
            test_errors.append(float(compute_error(x_test, y_test, w1, w2)))
            val_errors.append(float(compute_error(x_val, y_val, w1, w2)))
            train_mc_errors.append(
                get_misclassification_error(x_data, y_data, w1, w2))
            test_mc_errors.append(
                get_misclassification_error(x_test, y_test, w1, w2))
            val_mc_errors.append(
                get_misclassification_error(x_val, y_val, w1, w2))
        print_misclassification_error(x_test, y_test, w1, w2)
        total_err = compute_error(x_data, y_data, w1, w2)
        print('Error: {}'.format(float(total_err)))
        total_err.backward()
        w1.data = w1.data - (training_rate * w1.grad.data)
        w1.grad.data.zero_()
        w2.data = w2.data - (training_rate * w2.grad.data)
        w2.grad.data.zero_()
        idx += 1
    return (
        test_errors,
        val_errors,
        train_errors,
        train_mc_errors,
        test_mc_errors,
        val_mc_errors
    )

if __name__ == '__main__':
    training_rate = 0.0001
    stopping_criterion = 0.03
    for n_hidden_layers in [10, 100, 500]:
        print('\n\n\nN hidden layers: {}\n\n\n'.format(n_hidden_layers))
        (
            test_errors,
            val_errors,
            train_errors,
            train_mc_errors,
            test_mc_errors,
            val_mc_errors
        ) = train_mlp(
            n_hidden_layers, training_rate, stopping_criterion)
        print('test_errors: {}'.format(test_errors))
        print('val_errors: {}'.format(val_errors))
        print('train_errors: {}'.format(train_errors))
        print('train_misclassification_errors: {}'.format(train_mc_errors))
        print('test_misclassification_errors: {}'.format(test_mc_errors))
        print('val_misclassification_errors: {}'.format(val_mc_errors))
        draw_error_over_time(
            train_errors,
            test_errors,
            val_errors,
            '{} hidden layer MLP Loss by Training Iteration'.format(
                n_hidden_layers), 'Loss')
        draw_error_over_time(
            train_mc_errors,
            test_mc_errors,
            val_mc_errors,
            '{} Hidden Layer MLP Misclassification Error by Training Iteration'.format(
                n_hidden_layers),
            'Misclassification Error'
        )
