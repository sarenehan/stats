from sklearn.metrics import average_precision_score
import boto3
import numpy as np
import pickle
from utils.coco_utils import (
    get_hw3_categories,
    get_features_for_bboxes_large,
    category_id_to_info,
)
from utils.graph_utils import draw_train_and_dev_errors_over_time
from train_model import preprocess_data
import pandas as pd
from statistics import mean


def load_model_for_category(category_id):
    session = boto3.session.Session()
    s3client = session.client('s3')
    response = s3client.get_object(
        Bucket='stat-548',
        Key='data/final_project_models/' +
            'hw3_model_results_for_category_{}_large.pkl'.format(category_id))
    return pickle.loads(response['Body'].read())


def load_test_data_for_category(category_id):
    session = boto3.session.Session()
    s3client = session.client('s3')
    prefix = 'small2'
    response = s3client.get_object(
        Bucket='stat-548',
        Key='data/training_features/{}_test2014_feature_{}.pkl'.format(
            prefix, category_id))
    return pickle.loads(response['Body'].read())


def get_test_data_to_predict(category_id):
    test_data = load_test_data_for_category(category_id)
    x, y = preprocess_data(test_data)
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
        negative_indices = np.random.choice(
            negative_indices,
            replace=False,
            size=100000)
        all_indices = np.concatenate(
            [positive_indices, negative_indices]
        )
        y = y[all_indices]
        x = x[all_indices]
    x = get_features_for_bboxes_large(x, 'test2014')
    return x, y


def predict(x, w):
    return 1 / (1 + np.exp(-np.matmul(x, w)))


def get_w_for_best_predictor(category_id):
    model = load_model_for_category(category_id)
    weights_idx = np.argmax(model['total_val_ap_scores'])
    return model['weights'][weights_idx]


def get_test_ap_for_category(category_id):
    test_x, test_y = get_test_data_to_predict(category_id)
    w = get_w_for_best_predictor(category_id)
    preds = predict(test_x, w)
    actual_for_ap_score = []
    pred_for_ap_score = []
    for pred, label in zip(preds, test_y):
        if pred > 0.2 or label == 1:
            actual_for_ap_score.append(label)
            pred_for_ap_score.append(pred)
    return average_precision_score(actual_for_ap_score, pred_for_ap_score)


def draw_optimization_graphs_for_cat_2():
    model_results = load_model_for_category(2)
    train_errors = model_results['train_errors']
    val_errors = model_results['val_errors']
    draw_train_and_dev_errors_over_time(
        train_errors,
        val_errors,
        'Log Loss vs Iteration for Category Id {}'.format(2),
        y_label='Log Loss',
        save_fig=True,
        save_location_dir='/Users/stewart/Desktop/'
    )
    train_ap_scores = model_results['train_ap_scores'][:len(train_errors)]
    val_ap_scores = model_results['val_ap_scores'][:len(train_errors)]
    draw_train_and_dev_errors_over_time(
        train_ap_scores,
        val_ap_scores,
        'Average Precision Score vs Iteration for Category Id {}'.format(
            2),
        y_label='Average Precision Score',
        save_fig=True,
        save_location_dir='/Users/stewart/Desktop/'
    )


def get_test_errors():
    ap_scores = []
    categories = []
    super_categories = []
    category_id_to_info_dict = category_id_to_info()
    cat_ids = get_hw3_categories('small', 'train2014')
    for category_id in cat_ids:
        ap_score = get_test_ap_for_category(category_id)
        print('{}: {}'.format(category_id, ap_score))
        ap_scores.append(ap_score)
        categories.append(
            category_id_to_info_dict[category_id]['name']
        )
        super_categories.append(
            category_id_to_info_dict[category_id]['supercategory']
        )
    df = pd.DataFrame(
        {
            'Average Precision Score on Test Data': ap_scores,
            'Category': categories,
            'SuperCategory': super_categories,
            'Category ID': cat_ids
        }
    )
    df.to_csv(
        '/Users/stewart/Desktop/obj_detection_test_errors.csv',
        index=False
    )
    print('\n\nMean ap score: {}'.format(mean(ap_scores)))


if __name__ == '__main__':
    get_test_errors()
