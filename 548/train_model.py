from utils.coco_utils import (
    dataDir,
    get_hw3_categories,
    get_features_for_bboxes,
    get_features_for_bboxes_large,
    plot_bbox,
)
from utils.aws_utils import upload_to_s3
import os
import pickle
import boto3
import random
import numpy as np
from sklearn.metrics import average_precision_score


use_full_dataset = True
if use_full_dataset:
    save_location = dataDir + '/hw3_model_results_for_category_{}_large.pkl'
else:
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
        'data/final_project_models/' + model_save_location.split('/')[-1],
        model_save_location
    )
    os.remove(model_save_location)


def load_data_for_category(category_id):
    session = boto3.session.Session()
    s3client = session.client('s3')
    if use_full_dataset:
        prefix = 'small2'
    else:
        prefix = 'small'
    response_train = s3client.get_object(
        Bucket='stat-548',
        Key='data/training_features/{}_train2014_feature_{}.pkl'.format(
            prefix, category_id))
    response_val = s3client.get_object(
        Bucket='stat-548',
        Key='data/training_features/{}_val2014_feature_{}.pkl'.format(
            prefix, category_id))
    train = pickle.loads(response_train['Body'].read())
    val = pickle.loads(response_val['Body'].read())
    return train, val


def initialize_weights(d):
    return (np.random.random(d) - .5) / 1000


def log_loss(y_hat, y):
    return -(y * np.log(y_hat)) - ((1 - y) * np.log(1 - y_hat))


def compute_error_logistic_regression(w, lambda_, x, y):
    preds = 1 / (1 + np.exp(-np.matmul(x, w)))
    loss = log_loss(preds, y)
    regularization = (lambda_ / 2) * (np.linalg.norm(w) ** 2)
    return loss.mean() + regularization


def predict(x, w):
    return 1 / (1 + np.exp(-np.matmul(x, w)))


def get_average_precision_score(x_data, y_data, w):
    return average_precision_score(y_data, predict(x_data, w))


def construct_batches(n_, batch_size):
    indices = list(range(n_))
    random.shuffle(indices)
    batches = []
    start_idx = 0
    while start_idx < n_:
        batches.append(indices[start_idx:start_idx + batch_size])
        start_idx += batch_size
    return batches


def compute_gradient(x, y, w, lambda_):
    preds = 1 / (1 + np.exp(-np.matmul(x, w)))
    return (2 * (np.matmul((preds - y), x) / len(x))) + (lambda_ * w)


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
        max_iter=60):
    n_half_ephocs = 0
    iter_ = 0
    train_errors = []
    val_errors = []
    train_ap_scores = []
    val_ap_scores = []
    n_ = len(x_train)
    half_epoch = (int(len(x_train) / (2 * batch_size))) + 1
    grad_norm = float('inf')
    while (n_half_ephocs < max_iter) and (grad_norm > 1):
        batches = construct_batches(n_, batch_size)
        for idx, batch in enumerate(batches):
            if idx % half_epoch == 0:
                n_half_ephocs += 1
                total_error = float(compute_error_logistic_regression(
                    w, lambda_, x_train, y_train))
                ap_score = get_average_precision_score(x_train, y_train, w)
                total_error_val = float(
                    compute_error_logistic_regression(
                        w, lambda_, x_val, y_val)
                )
                ap_score_val = get_average_precision_score(x_val, y_val, w)
                print('{}'.format(n_half_ephocs))
                print('\tTrain Error: {}'.format(total_error))
                print('\tTrain AP Score: {}'.format(ap_score))
                print('\tVal Error: {}'.format(total_error_val))
                print('\tVal AP Score: {}'.format(ap_score_val))
                grad_norm = np.linalg.norm(compute_gradient(
                    x_train, y_train, w, lambda_))
                print('\tGrad norm: {}'.format(grad_norm))
                train_errors.append(total_error)
                val_errors.append(total_error_val)
                train_ap_scores.append(ap_score)
                val_ap_scores.append(ap_score_val)
            # Nesterov momentum
            grad = compute_gradient(
                x_train[batch], y_train[batch], w - (0.9 * v_t), lambda_)
            v_t = (0.9 * v_t) + ((training_rate / (
                (iter_ + 1) ** (1 / 3))) * grad)
            w = w - v_t
            iter_ += 1
    return (
        w,
        v_t,
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
    if len(positive_indices) > 10000:
        positive_indices = np.random.choice(
            positive_indices,
            replace=False,
            size=10000)
        all_indices = np.concatenate(
            [positive_indices, negative_indices]
        )
        y = y[all_indices]
        x = x[all_indices]
        positive_indices = np.where(y == 1)[0]
        negative_indices = np.where(y == 0)[0]
    if len(negative_indices) > 100000:
        new_negative_indices = np.random.choice(
            negative_indices,
            replace=False,
            size=100000)
        neg_index_to_init_neg_index = {
            idx: neg_index for idx, neg_index in enumerate(
                new_negative_indices)
        }
        negative_indices = new_negative_indices
        all_indices = np.concatenate(
            [positive_indices, negative_indices]
        )
        y = y[all_indices]
        x = x[all_indices]
        positive_indices = np.where(y == 1)[0]
        negative_indices = np.where(y == 0)[0]
    else:
        neg_index_to_init_neg_index = {x: x for x in negative_indices}
    if use_full_dataset:
        x = get_features_for_bboxes_large(x, data_type)
    else:
        x = get_features_for_bboxes(x, data_type)
    # x = np.hstack((np.ones((len(x), 1)), x))  # for the intercept...
    if w is not None:
        total_ap_score = average_precision_score(y, predict_numpy(x, w))
        print('Total AP score {}: {}'.format(data_type, total_ap_score))
        predictions = predict_numpy(x[negative_indices], w)
        sorted_preds_with_indices = sorted(
            list(enumerate(predictions)), key=lambda x: x[1])
        hard_negative_indices_to_use = [
            negative_indices[idx] for idx, prediction
            in sorted_preds_with_indices[-len(positive_indices):]
        ]
        hard_negative_indices_to_return = [
            neg_index_to_init_neg_index[idx] for idx in
            hard_negative_indices_to_use
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
    else:
        neg_indices_to_use = np.random.choice(
            negative_indices,
            replace=False,
            size=len(positive_indices) * 2
        )
        hard_negative_indices_to_return = None
        total_ap_score = None
    indices_to_use = np.concatenate([neg_indices_to_use, positive_indices])
    x = x[indices_to_use]
    y = y[indices_to_use]
    return x, y, total_ap_score, hard_negative_indices_to_return


def plot_some_hard_negatives(train, hard_negative_idxs, i):
    n_positive = len(train['positive_features'])
    for idx, hard_negative_idx in enumerate(
            random.sample(hard_negative_idxs, 4)):
        neg_idx = hard_negative_idx - n_positive
        image = train['negative_features'][neg_idx]
        bbox = image['bbox']
        img_id = image['img_id']
        plot_bbox(
            bbox,
            img_id,
            'train2014',
            is_large=use_full_dataset,
            save_fig=True,
            save_prefix='hard_negatives_cat_id_{}_round_{}_{}_'.format(
                category_id, i + 1, idx + 1),
            upload_img_to_s3=True,
            category_id=category_id)


def train_model(
        train,
        val,
        lambda_,
        training_rate,
        batch_size):
    x_train, y_train, _, _ = determine_negative_features(
        *preprocess_data(train), None, 'train2014')
    x_val, y_val, _, _ = determine_negative_features(
        *preprocess_data(val), None, 'val2014')
    w = initialize_weights(x_train.shape[1])
    v_t = np.zeros(len(w))
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
        x_train, y_train, total_train_ap_score, hard_negative_idxs = \
            determine_negative_features(
                *preprocess_data(train), w, 'train2014')
        x_val, y_val, total_val_ap_score, _ = \
            determine_negative_features(*preprocess_data(val), w, 'val2014')
        plot_some_hard_negatives(train, hard_negative_idxs, i)
        total_train_ap_scores.append(total_train_ap_score)
        total_val_ap_scores.append(total_val_ap_score)
        (
            w,
            v_t,
            _,
            _,
            _,
            _,
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
        ws.append(w)
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
    lambda_ = 10
    training_rate = 0.000001
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
    # idxes_for_node = list(range(4))
    # idxes_for_node = list(range(4, 8))
    # idxes_for_node = list(range(8, 13))
    # idxes_for_node = list(range(13, 18))
    category_ids = get_hw3_categories('small', 'train2014')
    # category_ids = [
    #     cat for idx, cat in enumerate(category_ids) if idx in idxes_for_node
    # ]
    for category_id in category_ids:
        train_and_save_model_for_category(category_id)
