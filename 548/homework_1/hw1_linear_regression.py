import torch
from torch.autograd import Variable
from utils.coco_utils import load_data_small_supercategory
from utils.redis_utils import cache
from hw1_graphs import draw_error_over_time


def initialize_weight_vector(n_weights):
    return Variable(
        (torch.rand(n_weights) - .5) / 1000, requires_grad=True
    )


def error_function(x, y, w, lambda_):
    return ((lambda_ / 2) * w.norm()) + (
        0.5 * (y - x.dot(w)).pow(2)
    )


def total_error(x_data, y_data, w, lambda_):
    return ((lambda_ / 2) * w.norm()) + ((1 / x_data.size()[0]) * (
        0.5 * (y_data - x_data.matmul(w)).pow(2)
    ).sum())


def get_misclassification_error(x_data, y_data, w):
    preds = [1 if float(x) > 0 else -1 for x in x_data.matmul(w)]
    incorrect_preds = [
        1 if (preds[idx] != int(y_data[idx])) else 0
        for idx in range(len(x_data))
    ]
    return sum(incorrect_preds) / len(incorrect_preds)


def print_misclassification_error(x_data, y_data, w):
    print('Misclassification Error: {}'.format(
        get_misclassification_error(x_data, y_data, w)))


def train_model_sgd(training_rate, lambda_):
    x_data, y_data = load_data_small_supercategory('train2014')
    w = initialize_weight_vector(len(x_data[0]))
    n_ = len(x_data)
    for i in range(1000):
        print_misclassification_error(x_data, y_data, w)
        total_err = total_error(x_data, y_data, w, lambda_)
        print('Error: {}'.format(float(total_err)))
        total_err.backward()
        for idx in range(n_):
            err = error_function(x_data[idx], y_data[idx], w, lambda_)
            err.backward()
            w.data = w.data - (training_rate * w.grad.data / n_)
            w.grad.data.zero_()


@cache.cached(timeout=60 * 60 * 24 * 60)
def train_linear_model(training_rate, lambda_, stopping_criterion):
    x_data, y_data = load_data_small_supercategory('train2014')
    x_test, y_test = load_data_small_supercategory('test2014')
    x_val, y_val = load_data_small_supercategory('val2014')
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
    w = initialize_weight_vector(len(x_data[0]))
    while float(total_err) > stopping_criterion:
        if not idx % 50:
            train_errors.append(float(total_error(x_data, y_data, w, lambda_)))
            test_errors.append(float(total_error(x_test, y_test, w, lambda_)))
            val_errors.append(float(total_error(x_val, y_val, w, lambda_)))
            train_mc_errors.append(
                get_misclassification_error(x_data, y_data, w))
            test_mc_errors.append(
                get_misclassification_error(x_test, y_test, w))
            val_mc_errors.append(
                get_misclassification_error(x_val, y_val, w))
        print_misclassification_error(x_test, y_test, w)
        total_err = total_error(x_data, y_data, w, lambda_)
        print('Error: {}'.format(float(total_err)))
        total_err.backward()
        w.data = w.data - (training_rate * w.grad.data)
        w.grad.data.zero_()
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
    training_rate = 0.000001
    lambda_ = 1
    stopping_criterion = 0.065
    (
        test_errors,
        val_errors,
        train_errors,
        train_mc_errors,
        test_mc_errors,
        val_mc_errors
    ) = train_linear_model(
        training_rate, lambda_, stopping_criterion)
    print('test_errors: {}'.format(test_errors))
    print('val_errors: {}'.format(val_errors))
    print('train_errors: {}'.format(train_errors))
    print('train_misclassification_errors: {}'.format(train_mc_errors))
    print('test_misclassification_errors: {}'.format(test_mc_errors))
    print('val_misclassification_errors: {}'.format(val_mc_errors))
    draw_error_over_time(
        train_errors, test_errors, val_errors,
        'Linear Regression Loss by Training Iteration', 'Loss'
    )
    draw_error_over_time(
        train_mc_errors,
        test_mc_errors,
        val_mc_errors,
        'Linear Regression Misclassification Error by Training Iteration',
        'Misclassification Error'
    )
