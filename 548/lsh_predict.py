import boto3
import pickle
import random
from utils.coco_utils import (
    get_hw3_categories,
    features_from_img_id,
    Featurizer,
    project_onto_feature_space,
    features_from_img_id_large,
    category_id_to_info,
)
from sklearn.metrics import average_precision_score
import sqlite3
from collections import Counter
import pandas as pd
from statistics import mean
import numpy as np
from utils.python_utils import unpickle_big_data
from utils.graph_utils import draw_scatterplot


r_ = 500
K = 40
featurizer = Featurizer()
category_ids = [0] + get_hw3_categories('small', 'train2014')
cat_id_to_idx = {cat_id: idx for idx, cat_id in enumerate(category_ids)}

use_full_dataset = True
if use_full_dataset:
    image_features_dict = features_from_img_id_large('test2014')
    image_features_dict_train = features_from_img_id_large('train2014')
else:
    image_features_dict = features_from_img_id('test2014')
    image_features_dict_train = features_from_img_id('train2014')


def get_retrieval_data(data, category_id):
    data_to_return = []
    positive_features = data['positive_features']
    negative_features = data['negative_features']
    for bbox in positive_features:
        bbox['category_id'] = category_id
        data_to_return.append(bbox)
    for bbox in negative_features:
        bbox['category_id'] = 0
        data_to_return.append(bbox)
    if len(data_to_return) > 1000:
        data_to_return = random.sample(data_to_return, 1000)
    return np.array(data_to_return)


def load_data_for_category(category_id):
    session = boto3.session.Session()
    s3client = session.client('s3')
    if use_full_dataset:
        prefix = 'small2'
    else:
        prefix = 'small'
    response = s3client.get_object(
        Bucket='stat-548',
        Key='data/training_features/{}_test2014_knn_features_{}.pkl'.format(
            prefix, category_id))
    data = pickle.loads(response['Body'].read())
    return get_retrieval_data(data, category_id)


def load_hash_functions():
    if use_full_dataset:
        file_ = 'hash_functions_large.pkl'
    else:
        file_ = 'hash_functions_small.pkl'
    filename = '/Users/stewart/projects/stats/data/' + file_
    return unpickle_big_data(filename)


def extract_features_from_bbox_test(bbox):
    return featurizer.featurize(
        project_onto_feature_space(bbox['bbox'], bbox['image_shape']),
        image_features_dict[bbox['img_id']]
    )


def extract_features_from_bbox_train(bbox):
    return featurizer.featurize(
        project_onto_feature_space(bbox['bbox'], bbox['image_shape']),
        image_features_dict_train[bbox['img_id']]
    )


def compute_distance(val_features, train_bbox):
    train_features = extract_features_from_bbox_train(train_bbox)
    return np.linalg.norm(val_features - train_features)


def convert_retrieval_query_to_dict(retrieval_query):
    return {
        'category_id': retrieval_query[0],
        'bbox': np.array(
            [
                retrieval_query[1],
                retrieval_query[2],
                retrieval_query[3],
                retrieval_query[4]
            ]),
        'img_id': retrieval_query[5],
        'image_shape': (
            retrieval_query[6], retrieval_query[7])
    }


def get_neighbors(bbox_to_predict, hash_functions, features):
    if use_full_dataset:
        suffix = 'large'
    else:
        suffix = 'small'
    conn = sqlite3.connect('/Users/stewart/projects/stats/data/548.db')
    c = conn.cursor()
    neighbors = []
    for idx, hash_function in enumerate(hash_functions):
        hash_ = tuple(
            np.round(
                np.matmul(hash_function, features) / r_).astype(int)
        )
        new_neighbors = c.execute(
            """
            SELECT
                category_id,
                bbox_x,
                bbox_y,
                bbox_width,
                bbox_height,
                img_id,
                img_shape_x,
                img_shape_y
            FROM retrieval_{suffix}
            INNER JOIN lsh_model_{suffix}_{lsh_id}
                ON lsh_model_{suffix}_{lsh_id}.retrieval_id =
                    retrieval_{suffix}.id
            WHERE hash_value like {hash_value}
            """.format(
                suffix=suffix,
                lsh_id=idx,
                hash_value='\"{}%\"'.format(
                    ':'.join([str(numb) for numb in hash_])
                )
            )
        ).fetchall()
        neighbors.extend(
            [convert_retrieval_query_to_dict(res) for res in new_neighbors]
        )
    return neighbors


def predict(bbox_to_predict, hash_functions, K=K):
    features = extract_features_from_bbox_test(bbox_to_predict)
    neighbors = get_neighbors(bbox_to_predict, hash_functions, features)
    neighbors_with_dist = [
        (neighbor['category_id'], compute_distance(features, neighbor))
        for neighbor in neighbors
    ]
    neighbors_with_dist.sort(key=lambda x: x[1])
    preds = [neighbor[0] for neighbor in neighbors_with_dist[:K]]
    distances = [float(neighbor[1]) for neighbor in neighbors_with_dist[:K]]
    output = np.zeros(len(category_ids))
    for pred, n_occurences in Counter(preds).items():
        output[cat_id_to_idx[pred]] = n_occurences / K
    return output, distances, len(neighbors)


def plot_ap_score_vs_search_time():
    hash_functions = load_hash_functions()
    data = load_data_for_category(2).tolist()
    if len(data) > 1000:
        data = random.sample(data, 1000)
    x = []
    y = []
    for l_ in range(1, len(hash_functions) + 1):
        preds = []
        true_values = []
        times = []
        for idx, bbox in enumerate(data):
            print('{} of {}'.format(idx, len(data)))
            pred, distances, time = predict(bbox, hash_functions[:l_])
            preds.append(pred[1])
            true_values.append(int(bbox['category_id'] == 2))
            times.append(time)
        x.append(mean(times))
        y.append(average_precision_score(true_values, preds))
        print(mean(times))
        print(y[-1])
    draw_scatterplot(
        x,
        y,
        xlabel='Search Time',
        ylabel='Average Precision Score',
        title='Average Precision Score vs Mean Search Time',
        save_location='/Users/stewart/Desktop/ap_score_vs_mean_search_time.png'
    )


def plot_mean_distance_vs_search_time():
    hash_functions = load_hash_functions()
    data = load_data_for_category(2).tolist()
    if len(data) > 1000:
        data = random.sample(data, 1000)
    x = []
    y = []
    blacklist = []
    for l_ in range(1, len(hash_functions) + 1):
        all_distances = []
        times = []
        hash_funcs = hash_functions[:l_]
        for idx, bbox in enumerate(data):
            print('{} of {}'.format(idx, len(data)))
            if idx not in blacklist:
                pred, distances, time = predict(bbox, hash_funcs)
                if not len(distances) == K:
                    blacklist.append(idx)
                else:
                    all_distances.extend(distances)
                    times.append(time)
        x.append(mean(times))
        y.append(mean(all_distances))
        print(mean(all_distances))
        print(mean(times))
    draw_scatterplot(
        x,
        y,
        xlabel='Search Time',
        ylabel='Mean Euclidean Distance to 10 Nearest Neighbors',
        title='Average Distance to 10 NNs vs Search Time',
        save_location='/Users/stewart/Desktop/mean_distance_vs_search_time.png'
    )


def plot_map_vs_k():
    hash_functions = load_hash_functions()
    data = load_data_for_category(2).tolist()
    x = []
    y = []
    for K in [10, 20, 30, 40, 50]:
        true_values = []
        preds = []
        for idx, bbox in enumerate(data):
            pred, distances, time = predict(bbox, hash_functions, K=K)
            true_values.append(int(bbox['category_id'] == 2))
            preds.append(pred[1])
        y.append(average_precision_score(true_values, preds))
        x.append(K)
    draw_scatterplot(
        x,
        y,
        xlabel='K',
        ylabel='Average Precision Score',
        title='Average Precision Score vs K',
        save_location='/Users/stewart/Desktop/ap_score_vs_k.png'
    )


def get_test_error():
    hash_functions = load_hash_functions()[:7]
    category_id_to_info_dict = category_id_to_info()
    y = []
    preds = []
    cat_ids = []
    ap_scores = []
    categories = []
    super_categories = []
    for category_id in get_hw3_categories('small', 'test2014'):
        data = load_data_for_category(category_id)
        preds_for_category = []
        labels_for_category = []
        for idx, bbox in enumerate(data):
            print('{} of {}'.format(idx, len(data)))
            y_row = np.zeros(len(category_ids))
            y_row[cat_id_to_idx[bbox['category_id']]] = 1
            y.append(y_row)
            pred_row, _, _ = predict(bbox, hash_functions)
            preds.append(pred_row)
            preds_for_category.append(pred_row[cat_id_to_idx[category_id]])
            labels_for_category.append(int(bbox['category_id'] == category_id))
        ap_score_for_category = average_precision_score(
            labels_for_category, preds_for_category)
        print('Average Precision Score for {}: {}'.format(
            category_id, ap_score_for_category))
        ap_scores.append(ap_score_for_category)
        categories.append(
            category_id_to_info_dict[category_id]['name']
        )
        cat_ids.append(category_id)
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
    if use_full_dataset:
        suffix = 'large'
    else:
        suffix = 'small'
    df.to_csv(
        '/Users/stewart/Desktop/lsh_test_errors_{}.csv'.format(suffix),
        index=False
    )
    print('\n\nMean ap score: {}'.format(mean(ap_scores)))


if __name__ == '__main__':
    ap_score_by_category = get_test_error()
    # plot_mean_distance_vs_search_time()
    # plot_ap_score_vs_search_time()
    # plot_map_vs_k()
