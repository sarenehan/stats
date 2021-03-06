from utils.coco_utils import (
    dataDir,
    get_hw3_categories,
    get_features_for_projected_bboxes,
)
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
save_location = dataDir + '/hw3_model_results_for_category_{}.pkl'


def save_results_to_s3(
        category_id,
        ws,
        train_errors,
        val_errors,
        train_ap_scores,
        val_ap_scores,
        total_train_ap_scores,
        total_val_ap_scores):
    dict_to_save = {
        'weights': ws,
        'train_errors': train_errors,
        'val_errors': val_errors,
        'train_ap_scores': train_ap_scores,
        'val_ap_scores': val_ap_scores,
        'total_train_ap_scores': total_train_ap_scores,
        'total_val_ap_scores': total_val_ap_scores
    }
    model_save_location = save_location.format(category_id)
    with open(model_save_location, 'wb') as f:
        pickle.dump(dict_to_save, f)
    upload_to_s3(
        'data/hw3_models/' + model_save_location.split('/')[-1],
        model_save_location
    )
    os.remove(model_save_location)


def load_data_for_category(category_id):
    session = boto3.session.Session()
    s3client = session.client('s3')
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


def initialize_weights_mlp(n_outputs):
    w1 = Variable(
        (torch.rand(11776, 250) - .5) / 1000,
        requires_grad=True)
    w2 = Variable((torch.rand(250) - .5) / 1000,
                  requires_grad=True)
    return w1, w2


def initialize_weights(d):
    return Variable((torch.rand(d) - .5) / 1000, requires_grad=True)


def log_loss(y_hat, y):
    return -y.mul(torch.log(y_hat)) - (1 - y).mul(
        torch.log(1 - y_hat))


def compute_error_logistic_regression(w, lambda_, x, y, prev_w):
    preds = 1 / (1 + torch.exp(-x.matmul(w)))
    loss = log_loss(preds, y)
    regularization = (lambda_ / 2) * (w[1:].norm().pow(2))
    return loss.mean() + regularization + (lambda_ * (w - prev_w).norm() ** 2)


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
        batch_size,
        w,
        v_t,
        iter_init=0,
        prev_w=None,
        max_iter=20000):
    iter_ = 0
    best_val_total_error = float('inf')
    ap_score = 0
    train_errors = []
    val_errors = []
    train_ap_scores = []
    val_ap_scores = []
    best_weight = None
    best_v_t = None
    if prev_w is None:
        prev_w = Variable(torch.zeros(len(w)), requires_grad=True)
    n_ = len(x_train)
    half_epoch = (int(x_train.size()[0] / (2 * batch_size))) + 1
    iter_ = iter_init
    while iter_ < iter_init + 20000:
        batches = construct_batches(n_, batch_size)
        for idx, batch in enumerate(batches):
            if idx % half_epoch == 0:
                total_error = float(compute_error_logistic_regression(
                    w, lambda_, x_train, y_train, prev_w))
                ap_score = get_average_precision_score(x_train, y_train, w)
                total_error_val = float(
                    compute_error_logistic_regression(w, lambda_, x_val, y_val, prev_w)
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
                if total_error_val < best_val_total_error:
                    best_val_total_error = total_error_val
                    best_weight = w.data.numpy()
                    best_v_t = v_t.data.numpy()
                elif len(val_errors) - val_errors.index(
                        best_val_total_error) > 8:
                    print('Best val error: {}'.format(best_val_total_error))
                    return (
                        best_weight,
                        best_v_t,
                        train_errors,
                        val_errors,
                        train_ap_scores,
                        val_ap_scores,
                        iter_
                    )
            # Nesterov momentum
            err_est = compute_error_logistic_regression(
                w - 0.9 * v_t, lambda_, x_train[batch], y_train[batch], prev_w)
            err_est.backward()
            v_t.data = (0.9 * v_t.data) + ((training_rate / (
                (iter_ + 1) ** (1 / 3))) * (
                w.grad.data - (0.9 * v_t.grad.data)))
            w.data = w.data - v_t.data
            w.grad.data.zero_()
            v_t.grad.data.zero_()
            iter_ += 1
    print('Best val error: {}'.format(best_val_total_error))
    return (
        best_weight,
        best_v_t,
        train_errors,
        val_errors,
        train_ap_scores,
        val_ap_scores,
        iter_
    )


def preprocess_data(data):
    positive_features = data['positive_features']
    negative_features = data['negative_features']
    y = np.array([1] * len(positive_features) + [0] * len(
        negative_features))
    x = np.array(positive_features + negative_features)
    return x, y


def predict_numpy(x, w):
    return 1 / (1 + np.exp(-np.matmul(x, w)))


def determine_negative_features(x, y, w, data_type):
    positive_indices = np.where(y == 1)[0]
    negative_indices = np.where(y == 0)[0]
    if len(negative_indices) > 100000:
        negative_indices = np.random.choice(
            negative_indices,
            replace=False,
            size=100000)
        all_indices = np.concatenate(
            [positive_indices, negative_indices]
        )
        y = y[all_indices]
        x = x[all_indices]
        positive_indices = np.where(y == 1)[0]
        negative_indices = np.where(y == 0)[0]
    x = get_features_for_projected_bboxes(x, data_type=data_type)
    # x = np.hstack((np.ones((len(x), 1)), x))  # for the intercept...
    if w is not None and 'val' not in data_type:
        total_ap_score = average_precision_score(y, predict_numpy(x, w))
        print('Total AP score {}: {}'.format(data_type, total_ap_score))
        predictions = predict_numpy(x[negative_indices], w)
        sorted_preds_with_indices = sorted(
            list(enumerate(predictions)), key=lambda x: x[1])
        hard_negative_indices_to_use = [
            negative_indices[idx] for idx, prediction
            in sorted_preds_with_indices[-len(positive_indices):]
        ]
        random_neg_indices_to_use = np.random.choice(
            list(set(negative_indices).difference(
                set(hard_negative_indices_to_use))),
            replace=False,
            size=len(positive_indices)
        )
        neg_indices_to_use = np.concatenate(
            [hard_negative_indices_to_use, random_neg_indices_to_use]
        )
    elif 'val' not in data_type:
        neg_indices_to_use = np.random.choice(
            negative_indices,
            replace=False,
            size=len(positive_indices) * 2
        )
        total_ap_score = None
    else:
        neg_indices_to_use = negative_indices
        if w is not None:
            total_ap_score = average_precision_score(y, predict_numpy(x, w))
        else:
            total_ap_score = None
    indices_to_use = np.concatenate([neg_indices_to_use, positive_indices])
    x = x[indices_to_use]
    y = y[indices_to_use]
    x = Variable(torch.from_numpy(x).float(), requires_grad=True)
    y = Variable(torch.from_numpy(y).float(), requires_grad=True)
    return x, y, total_ap_score


def train_model(
        train,
        val,
        lambda_,
        training_rate,
        batch_size):
    x_train, y_train, _ = determine_negative_features(
        *preprocess_data(train), None, 'train2014')
    x_val, y_val, _ = determine_negative_features(
        *preprocess_data(val), None, 'val2014')
    w = initialize_weights(x_train.shape[1])
    v_t = Variable(torch.zeros(w.shape), requires_grad=True)
    (
        w,
        v_t,
        train_errors_list,
        val_errors_list,
        train_ap_scores_list,
        val_ap_scores_list,
        iter_
    ) = train_logistic_regression(
        x_train,
        y_train,
        x_val,
        y_val,
        lambda_,
        training_rate,
        batch_size,
        w,
        v_t)
    total_train_ap_scores = []
    total_val_ap_scores = []
    ws = [w]
    for i in range(10):
        x_train, y_train, total_train_ap_score = determine_negative_features(
            *preprocess_data(train), w, 'train2014')
        x_val, y_val, total_val_ap_score = determine_negative_features(
            *preprocess_data(val), w, 'val2014')
        total_train_ap_scores.append(total_train_ap_score)
        total_val_ap_scores.append(total_val_ap_score)
        (
            w,
            v_t,
            train_errors,
            val_errors,
            train_ap_scores,
            val_ap_scores,
            iter_
        ) = train_logistic_regression(
            x_train,
            y_train,
            x_val,
            y_val,
            lambda_,
            training_rate,
            batch_size,
            Variable(torch.from_numpy(w).float(), requires_grad=True),
            Variable(torch.from_numpy(v_t).float(), requires_grad=True),
            iter_init=iter_,
            prev_w=Variable(torch.from_numpy(w).float(), requires_grad=True))
        ws.append(w)
        train_ap_scores_list.extend(train_ap_scores)
        val_ap_scores_list.extend(val_ap_scores)
    return (
        ws,
        train_errors_list,
        val_errors_list,
        train_ap_scores_list,
        val_ap_scores_list,
        total_train_ap_scores,
        total_val_ap_scores
    )


def train_and_save_model_for_category(category_id):
    print('Generating model for cat id {}'.format(category_id))
    train, val = load_data_for_category(category_id)
    lambda_ = 1000
    training_rate = 0.00000001
    batch_size = 10
    print('Lambda: {}'.format(lambda_))
    print('Training rate: {}'.format(training_rate))
    (
        ws,
        train_errors,
        val_errors,
        train_ap_scores,
        val_ap_scores,
        total_train_ap_scores,
        total_val_ap_scores
    ) = train_model(train, val, lambda_, training_rate, batch_size)
    save_results_to_s3(
        category_id,
        ws,
        train_errors,
        val_errors,
        train_ap_scores,
        val_ap_scores,
        total_train_ap_scores,
        total_val_ap_scores)


if __name__ == '__main__':
    # idxes_for_node = list(range(9))
    # idxes_for_node = list(range(9, 18))
    # category_ids = get_hw3_categories('small', data_type)
    # category_ids = [
    #     cat for idx, cat in enumerate(category_ids) if idx in idxes_for_node
    # ]
    category_ids = [23]
    for category_id in category_ids:
        train_and_save_model_for_category(category_id)
