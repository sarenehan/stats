from utils.coco_utils import dataDir, get_hw3_categories
from utils.aws_utils import upload_to_s3
import os
import pickle
import boto3
import random
import torch
import numpy as np
from torch.autograd import Variable
from sklearn.metrics import average_precision_score


data_type = 'train2014'
save_location = dataDir + '/hw3_model_results_category_{}.pkl'


def save_results_to_s3(
        category_id,
        w,
        train_errors,
        val_errors,
        train_ap_scores,
        val_ap_scores):
    dict_to_save = {
        'weights': w,
        'train_errors': train_errors,
        'val_errors': val_errors,
        'train_ap_scores': train_ap_scores,
        'val_ap_scores': val_ap_scores
    }
    model_save_location = save_location.format(category_id)
    with open(model_save_location, 'wb') as f:
        pickle.dump(dict_to_save, f)
    upload_to_s3(
        'data/hw3_model_results/' + model_save_location.split('/')[-1],
        model_save_location
    )
    os.remove(model_save_location)


def load_data_for_category(category_id):
    session = boto3.session.Session()
    s3client = session.client('s3')  # credentials
    response_train = s3client.get_object(
        Bucket='stat-548',
        Key='data/training_features/small_train2014_feature_{}.pkl'.format(
            category_id))
    response_val = s3client.get_object(
        Bucket='stat-548',
        Key='data/training_features/small_val2014_feature_{}.pkl'.format(
            category_id))
    train = pickle.loads(response_train['Body'].read())
    val = pickle.loads(response_val['Body'].read())
    return train, val


def initialize_weights(d):
    return Variable((torch.rand(d) - .5) / 1000, requires_grad=True)


def compute_error_logistic_regression(w, lambda_, x, y):
    preds = 1 / (1 + torch.exp(-x.matmul(w)))
    loss = (0.5 * (y - preds).pow(2)) + ((lambda_ / 2) * w.norm())
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
        batches.append(indices[start_idx:start_idx + batch_size])
        start_idx += batch_size
    return batches


def train_logistic_regression(
        x_train,
        y_train,
        x_val,
        y_val,
        lambda_,
        training_rate,
        batch_size):
    iter_ = 0
    w = initialize_weights(x_train.shape[1])
    v_t = Variable(torch.zeros(w.shape), requires_grad=True)
    total_error = float('inf')
    best_val_error = float('inf')
    ap_score = 0
    train_errors = []
    val_errors = []
    train_ap_scores = []
    val_ap_scores = []
    best_weight = None
    n_ = len(x_train)
    half_epoch = int(x_train.size()[0] / (2 * batch_size))
    while ap_score < .99 and iter_ < 30000:
        total_error = float(
            compute_error_logistic_regression(w, lambda_, x_train, y_train))
        ap_score = get_average_precision_score(x_train, y_train, w)
        total_error_val = float(
            compute_error_logistic_regression(w, lambda_, x_val, y_val))
        ap_score_val = get_average_precision_score(x_val, y_val, w)
        print('{}'.format(iter_))
        print('\tTrain Error: {}'.format(total_error))
        print('\tTrain AP Score: {}'.format(ap_score))
        print('\tVal Error: {}'.format(total_error_val))
        print('\tVal AP Score: {}'.format(ap_score_val))
        train_errors.append(total_error)
        val_errors.append(total_error_val)
        train_ap_scores.append(ap_score)
        val_ap_scores.append(ap_score_val)
        if total_error_val < best_val_error:
            best_val_error = total_error_val
            best_weight = w.data.numpy()
        batches = construct_batches(n_, batch_size)
        for idx, batch in enumerate(batches):
            if idx == half_epoch:
                total_error = float(
                    compute_error_logistic_regression(
                        w, lambda_, x_train, y_train))
                ap_score = get_average_precision_score(x_train, y_train, w)
                total_error_val = float(
                    compute_error_logistic_regression(w, lambda_, x_val, y_val)
                )
                ap_score_val = get_average_precision_score(x_val, y_val, w)
                print('{}'.format(iter_))
                print('\tTrain Error: {}'.format(total_error))
                print('\tTrain AP Score: {}'.format(ap_score))
                print('\tVal Error: {}'.format(total_error_val))
                print('\tVal AP Score: {}'.format(ap_score_val))
                train_errors.append(total_error)
                val_errors.append(total_error_val)
                train_ap_scores.append(ap_score)
                val_ap_scores.append(ap_score_val)
                if total_error_val < best_val_error:
                    best_val_error = total_error_val
                    best_weight = w.data.numpy()
            # Nesterov momentum
            err_est = compute_error_logistic_regression(
                w - 0.9 * v_t, lambda_, x_train[batch], y_train[batch])
            err_est.backward()
            v_t.data = (0.9 * v_t.data) + (training_rate * (
                w.grad.data - (0.9 * v_t.grad.data)))
            w.data = w.data - v_t.data
            w.grad.data.zero_()
            v_t.grad.data.zero_()
            iter_ += 1
    return (
        best_weight,
        train_errors,
        val_errors,
        train_ap_scores,
        val_ap_scores
    )


def preprocess_data(data, fit_intercept):
    positive_features = data['positive_features']
    negative_features = data['negative_features']
    y = np.array([1] * len(positive_features) + [0] * len(
        negative_features))
    x = np.array(positive_features + negative_features)
    if fit_intercept:
        x = np.hstack((np.ones((len(x), 1)), x))
    x = Variable(torch.from_numpy(x).float(), requires_grad=True)
    y = Variable(torch.from_numpy(y).float(), requires_grad=True)
    return x, y


def train_model(
        train,
        val,
        lambda_,
        training_rate,
        batch_size,
        fit_intercept=True):
    x_train, y_train = preprocess_data(train, fit_intercept)
    x_val, y_val = preprocess_data(val, fit_intercept)
    return train_logistic_regression(
        x_train, y_train, x_val, y_val, lambda_, training_rate, batch_size)


def train_and_save_model_for_category(category_id):
    train, val = load_data_for_category(category_id)
    lambda_ = 1
    print('Lambda: {}'.format(lambda_))
    training_rate = 0.000001
    batch_size = 10
    (
        w,
        train_errors,
        val_errors,
        train_ap_scores,
        val_ap_scores
    ) = train_model(train, val, lambda_, training_rate, batch_size)
    save_results_to_s3(
        category_id,
        w,
        train_errors,
        val_errors,
        train_ap_scores,
        val_ap_scores)


if __name__ == '__main__':
    # idxes_for_node = list(range(9))
    # idxes_for_node = list(range(9, 18))
    # category_ids = get_hw3_categories('small', data_type)
    # category_ids = [
    #     cat for idx, cat in enumerate(category_ids) if idx in idxes_for_node
    # ]
    category_ids = [2]
    for category_id in category_ids:
        train_and_save_model_for_category(category_id)
