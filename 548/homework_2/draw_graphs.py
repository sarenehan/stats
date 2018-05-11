from utils.graph_utils import draw_train_and_dev_errors_over_time
from utils.coco_utils import load_category_level_data_hw2
from homework_2.hw2_linear_regression import (
    get_average_precision_score as get_average_precision_score_lr,
    predict as predict_lr,
)
from homework_2.hw_2_mlp import (
    get_average_precision_score as get_average_precision_score_mlp,
    predict as predict_mlp,
)
import torch
from torch.autograd import Variable
from sklearn.metrics import average_precision_score
import pandas as pd
import os
import pickle


graph_data_dir = os.environ['PYTHONPATH'] + '/model_results/'
categories = {
    0: 'bicycle',
    1: 'car',
    2: 'motorcycle',
    3: 'airplane',
    4: 'bus',
    5: 'train',
    6: 'truck',
    7: 'boat',
    8: 'bird',
    9: 'cat',
    10: 'dog',
    11: 'horse',
    12: 'sheep',
    13: 'cow',
    14: 'elephant',
    15: 'bear',
    16: 'zebra',
    17: 'giraffe'
}


def open_pickle_file(filename):
    with open(graph_data_dir + filename + '.pkl', 'rb') as f:
        data = pickle.load(f)
    return data


def sgd_linear_regression():
    (
        train_errors,
        val_errors,
        train_average_precision_scores,
        val_average_precision_scores,
        w
    ) = open_pickle_file('1_linear_regression_errors')
    draw_train_and_dev_errors_over_time(
        train_errors,
        val_errors,
        'Linear Regression L2 Loss vs Iteration',
        'L2 Loss'
    )
    draw_train_and_dev_errors_over_time(
        train_average_precision_scores,
        val_average_precision_scores,
        'Linear Regression Average Precision Score vs Iteration',
        'Average Precision Score'
    )


def linear_regression_test_error(w, file_prefix):
    w = Variable(w)
    x_test_, y_test_ = load_category_level_data_hw2('small', 'test2014')
    x_test = Variable(torch.from_numpy(x_test_))
    y_test = Variable(torch.from_numpy(y_test_))
    ap_score = get_average_precision_score_lr(x_test, y_test, w)
    print('Overall {} Linear Regression Avg precision score: {}'.format(
        file_prefix, ap_score))
    preds = predict_lr(x_test, w).data.numpy()
    ap_score_by_category = []
    for column in range(y_test_.shape[1]):
        ap_score_by_category.append({
            'Category': categories[column],
            'Average Precision Score': average_precision_score(
                y_test_[:, column], preds[:, column])
        })
    df = pd.DataFrame(ap_score_by_category)
    df.to_csv('/Users/stewart/Desktop/{}_linear_regression.csv'.format(
        file_prefix), index=False)


def sgd_linear_regression_test_error():
    _, _, _, _, w = open_pickle_file('1_linear_regression_errors')
    linear_regression_test_error(w, 'sgd')


def nesterov_linear_regression():
    (
        train_errors,
        val_errors,
        train_average_precision_scores,
        val_average_precision_scores,
        w
    ) = open_pickle_file('10_linear_regression_errors_nest')
    draw_train_and_dev_errors_over_time(
        train_errors,
        val_errors,
        'Nesterov Linear Regression Momentum L2 Loss vs Iteration',
        'L2 Loss'
    )
    draw_train_and_dev_errors_over_time(
        train_average_precision_scores,
        val_average_precision_scores,
        'Nesterov Linear Regression Momentum Avg Prec Score vs Iteration',
        'Average Precision Score'
    )


def nesterov_linear_regression_test_error():
    _, _, _, _, w = open_pickle_file('10_linear_regression_errors_nest')
    linear_regression_test_error(w, 'nesterov')


def adagrad_linear_regression():
    (
        train_errors,
        val_errors,
        train_average_precision_scores,
        val_average_precision_scores,
        w
    ) = open_pickle_file('10_linear_regression_errors_adagrad')
    draw_train_and_dev_errors_over_time(
        train_errors,
        val_errors,
        'Adagrad Linear Regression L2 Loss vs Iteration',
        'L2 Loss'
    )
    draw_train_and_dev_errors_over_time(
        train_average_precision_scores,
        val_average_precision_scores,
        'Adagrad Linear Regression Avg Prec Score vs Iteration',
        'Average Precision Score'
    )


def adagrad_linear_regression_test_error():
    _, _, _, _, w = open_pickle_file('10_linear_regression_errors_adagrad')
    linear_regression_test_error(w, 'adagrad')


def mlp_test_error(w1, w2, file_prefix):
    x_test_, y_test_ = load_category_level_data_hw2('small', 'test2014')
    x_test = Variable(torch.from_numpy(x_test_))
    y_test = Variable(torch.from_numpy(y_test_))
    w1 = Variable(w1)
    w2 = Variable(w2)
    ap_score = get_average_precision_score_mlp(x_test, y_test, w1, w2)
    print('Overall {} MLP Avg precision score: {}'.format(
        file_prefix, ap_score))
    preds = predict_mlp(x_test, w1, w2).data.numpy()
    ap_score_by_category = []
    for column in range(y_test_.shape[1]):
        ap_score_by_category.append({
            'Category': categories[column],
            'Average Precision Score': average_precision_score(
                y_test_[:, column], preds[:, column])
        })
    df = pd.DataFrame(ap_score_by_category)
    df.to_csv('/Users/stewart/Desktop/{}_mlp.csv'.format(
        file_prefix), index=False)


def get_class_imbalance():
    x_test_, y_test_ = load_category_level_data_hw2('small', 'test2014')
    count_by_class = []
    n_ = len(y_test_)
    for idx in range(y_test_.shape[1]):
        count_by_class.append(
            {
                'Category': categories[idx],
                'Pct in class': y_test_[:, idx].sum() / n_
            }
        )
    pd.DataFrame(count_by_class).to_csv(
        '/Users/stewart/Desktop/class_imbalance.csv', index=False)


def sgd_mlp():
    (
        train_errors,
        val_errors,
        train_average_precision_scores,
        val_average_precision_scores
    ) = open_pickle_file('250_mlp_errors')
    draw_train_and_dev_errors_over_time(
        train_errors,
        val_errors,
        'MLP L2 Loss vs Iteration',
        'L2 Loss'
    )
    draw_train_and_dev_errors_over_time(
        train_average_precision_scores,
        val_average_precision_scores,
        'MLP Average Precision Score vs Iteration',
        'Average Precision Score'
    )


def sgd_mlp_test_error():
    w1, w2 = open_pickle_file('250_mlp_weights')
    mlp_test_error(w1, w2, 'sgd')


def nesterov_mlp():
    (
        train_errors,
        val_errors,
        train_average_precision_scores,
        val_average_precision_scores
    ) = open_pickle_file('250_mlp_errors_nesterov')
    draw_train_and_dev_errors_over_time(
        train_errors,
        val_errors,
        'Nesterov MLP L2 Loss vs Iteration',
        'L2 Loss'
    )
    draw_train_and_dev_errors_over_time(
        train_average_precision_scores,
        val_average_precision_scores,
        'Nesterov MLP Average Precision Score vs Iteration',
        'Average Precision Score'
    )


def nesterov_mlp_test_error():
    w1, w2 = open_pickle_file('250_mlp_weights_nesterov')
    mlp_test_error(w1, w2, 'nesterov')


def adagrad_mlp():
    (
        train_errors,
        val_errors,
        train_average_precision_scores,
        val_average_precision_scores
    ) = open_pickle_file('250_mlp_errors_adagrad')
    draw_train_and_dev_errors_over_time(
        train_errors,
        val_errors,
        'Adagrad MLP L2 Loss vs Iteration',
        'L2 Loss'
    )
    draw_train_and_dev_errors_over_time(
        train_average_precision_scores,
        val_average_precision_scores,
        'Adagrad MLP Average Precision Score vs Iteration',
        'Average Precision Score'
    )


def adagrad_mlp_test_error():
    w1, w2 = open_pickle_file('250_mlp_weights_adagrad')
    mlp_test_error(w1, w2, 'adagrad')


if __name__ == '__main__':
    get_class_imbalance()
    # sgd_linear_regression()
    # sgd_mlp()
    # sgd_linear_regression_test_error()
    # sgd_mlp_test_error()
    # nesterov_linear_regression()
    # nesterov_mlp()
    # nesterov_linear_regression_test_error()
    # nesterov_mlp_test_error()
    # adagrad_linear_regression()
    # adagrad_mlp()
    # adagrad_linear_regression_test_error()
    # adagrad_mlp_test_error()
