import random
import pickle
import boto3
from utils.graph_utils import draw_train_and_dev_errors_over_time
from utils.coco_utils import (
    get_hw3_categories,
    get_features_for_projected_bboxes,
)
import numpy as np
from statistics import mean
from utils.redis_utils import cache
from sklearn.metrics import average_precision_score

graph_save_location = '/Users/stewart/Desktop/'


def load_model_for_category(category_id):
    session = boto3.session.Session()
    s3client = session.client('s3')
    model_results = s3client.get_object(
        Bucket='stat-548',
        Key='data/hw3_models/hw3_model_results_for_category_{}.pkl'.format(
            category_id))
    model_results = pickle.loads(model_results['Body'].read())
    return model_results


def draw_optimization_graphs_for_category(category_id):
    model_results = load_model_for_category(category_id)
    train_errors = model_results['train_errors']
    val_errors = model_results['val_errors']
    draw_train_and_dev_errors_over_time(
        train_errors,
        val_errors,
        'Log Loss vs Iteration for Category Id {}'.format(category_id),
        y_label='Log Loss',
        save_fig=True,
        save_location_dir=graph_save_location
    )
    train_ap_scores = model_results['train_ap_scores'][:len(train_errors)]
    val_ap_scores = model_results['val_ap_scores'][:len(train_errors)]
    draw_train_and_dev_errors_over_time(
        train_ap_scores,
        val_ap_scores,
        'Average Precision Score vs Iteration for Category Id {}'.format(
            category_id),
        y_label='Average Precision Score',
        save_fig=True,
        save_location_dir=graph_save_location
    )


def load_test_data(category_id):
    session = boto3.session.Session()
    s3client = session.client('s3')
    test_data = s3client.get_object(
        Bucket='stat-548',
        Key='data/training_features/small_test2014_feature_{}.pkl'.format(
            category_id))
    data = pickle.loads(test_data['Body'].read())
    positive_features = data['positive_features']
    negative_features = data['negative_features']
    y = np.array([1] * len(positive_features) + [0] * len(
        negative_features))
    x = np.array(positive_features + negative_features)
    if len(x) > 100000:
        indices = list(range(len(x)))
        indices_to_use = random.sample(indices, 100000)
        y = y[indices_to_use]
        x = x[indices_to_use]
    x = get_features_for_projected_bboxes(x, data_type='test2014')
    x = np.hstack((np.ones((len(x), 1)), x))  # for the intercept...
    return x, y


def predict(x, w):
    return 1 / (1 + np.exp(-np.matmul(x, w)))


@cache.cached(timeout=60 * 60 * 24 * 60)
def compute_test_average_precision_score(category_id):
    model_results = load_model_for_category(category_id)
    test_x, test_y = load_test_data(category_id)
    w_ = model_results['weights'][0]
    predictions = predict(test_x, w_)
    return average_precision_score(test_y, predictions)


def get_average_precision_score_per_class_and_total():
    ap_scores = []
    category_ids = get_hw3_categories('small', 'test2014')
    for category_id in category_ids:
        try:
            ap_scores.append(compute_test_average_precision_score(category_id))
            print('{}\t{}'.format(category_id, ap_scores[-1]))
        except:
            print('Category {} is messed up'.format(category_id))
    print('\n\n\nMean Average Precision Score: {}'.format(mean(ap_scores)))


if __name__ == '__main__':
    # for category_id in category_ids:
    #     draw_optimization_graphs_for_category(category_id)
    get_average_precision_score_per_class_and_total()
