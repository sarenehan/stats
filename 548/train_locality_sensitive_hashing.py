from collections import defaultdict, Counter
import boto3
import sqlite3
import random
import pickle
import numpy as np
from utils.coco_utils import (
    get_hw3_categories,
    features_from_img_id_large,
    features_from_img_id,
    Featurizer,
    project_onto_feature_space,
    dataDir,
)
from utils.python_utils import pickle_big_data
from sklearn.metrics import average_precision_score
from statistics import mean


use_full_dataset = False
r_ = 500
l_ = 10
k_ = 10
K = 10
dim = 11776
category_ids = [0] + get_hw3_categories('small', 'train2014')
cat_id_to_idx = {cat_id: idx for idx, cat_id in enumerate(category_ids)}
print('r_: {}'.format(r_))
print('l_: {}'.format(l_))
print('k_: {}'.format(k_))
print('K: {}'.format(K))
featurizer = Featurizer()


def load_data_for_category(category_id):
    session = boto3.session.Session()
    s3client = session.client('s3')
    if use_full_dataset:
        prefix = 'small2'
    else:
        prefix = 'small'
    response_train = s3client.get_object(
        Bucket='stat-548',
        Key='data/training_features/{}_train2014_knn_features_{}.pkl'.format(
            prefix, category_id))
    response_val = s3client.get_object(
        Bucket='stat-548',
        Key='data/training_features/{}_val2014_knn_features_{}.pkl'.format(
            prefix, category_id))
    train = pickle.loads(response_train['Body'].read())
    val = pickle.loads(response_val['Body'].read())
    return train, val


def get_random_hash_vector(k_=k_):
    return np.array([np.random.normal(0, 1, dim) for _ in range(k_)])


def construct_hash_functions(l_=l_):
    return np.array([get_random_hash_vector() for _ in range(l_)])


def get_retrieval_data(data, category_id):
    data_to_return = []
    positive_features = data['positive_features']
    negative_features = data['negative_features']
    if len(data['positive_features']) > 10000:
        positive_features = random.sample(
            positive_features, 10000
        )
    if len(data['negative_features']) > 20000:
        negative_features = random.sample(
            negative_features, 20000
        )
    for bbox in positive_features:
        bbox['category_id'] = category_id
        data_to_return.append(bbox)
    for bbox in negative_features:
        bbox['category_id'] = 0
        data_to_return.append(bbox)
    return np.array(data_to_return)


def extract_features_from_bbox(bbox, data_type='train2014'):
    if data_type == 'train2014':
        return featurizer.featurize(
            project_onto_feature_space(
                bbox['bbox'], bbox['image_shape']),
            image_features_dict[bbox['img_id']]
        )
    elif data_type == 'val2014':
        return featurizer.featurize(
            project_onto_feature_space(
                bbox['bbox'], bbox['image_shape']),
            val_image_features_dict[bbox['img_id']]
        )


def insert_bbox_row(bbox, c, id_):
    row_ = [
        str(bbox['bbox'][0]),
        str(bbox['bbox'][1]),
        str(bbox['bbox'][2]),
        str(bbox['bbox'][3]),
        str(bbox['image_shape'][0]),
        str(bbox['image_shape'][1]),
        str(bbox['img_id']),
        str(bbox['category_id']),
        str(id_)
    ]
    if use_full_dataset:
        c.execute('insert into retrieval_large values ({})'.format(
            ', '.join(row_)))
    else:
        c.execute('insert into retrieval_small values ({})'.format(
            ', '.join(row_)))


def insert_hash_row(hash_, lsh_id, retrieval_id, c):
    hash_str = "\"" + ':'.join([str(numb) for numb in hash_]) + "\""
    row_ = [hash_str, str(retrieval_id)]
    if use_full_dataset:
        c.execute('insert into lsh_model_large_{} values ({})'.format(
            lsh_id, ', '.join(row_)))
    else:
        c.execute('insert into lsh_model_small_{} values ({})'.format(
            lsh_id, ', '.join(row_)))


def create_retrieval_table(c):
    if use_full_dataset:
        suffix = 'large'
    else:
        suffix = 'small'
    c.execute('drop table if exists retrieval_{}'.format(suffix))
    c.execute(
        """
        CREATE TABLE retrieval_{}
            (bbox_x int,
             bbox_y int,
             bbox_width int,
             bbox_height int,
             img_shape_x int,
             img_shape_y int,
             img_id int,
             category_id int,
             id integer not null,
             PRIMARY KEY (id));
        """.format(suffix)
    )
    c.execute('create index id_idx_{x} on retrieval_{x} (id)'.format(x=suffix))


def create_hash_tables(hash_functions, c):
    c.execute('PRAGMA foreign_keys = ON;')
    for idx, hash_function in enumerate(hash_functions):
        if use_full_dataset:
            c.execute('drop table if exists lsh_model_large_{}'.format(idx))
            c.execute(
                """
                CREATE TABLE lsh_model_large_{}
                    (
                        hash_value text,
                        retrieval_id int,
                        FOREIGN KEY (retrieval_id) REFERENCES
                            retrieval_large(id)
                    )
                """.format(idx))
        else:
            c.execute('drop table if exists lsh_model_small_{}'.format(idx))
            c.execute(
                """
                CREATE TABLE lsh_model_small_{}
                    (
                        hash_value text,
                        retrieval_id int,
                        FOREIGN KEY (retrieval_id) REFERENCES
                            retrieval_small(id)
                    )
                """.format(idx))


def populate_sqllite_table():
    hash_functions = construct_hash_functions()
    conn = sqlite3.connect('/Users/stewart/projects/stats/data/548.db')
    category_ids = get_hw3_categories('small', 'train2014')
    c = conn.cursor()
    create_retrieval_table(c)
    conn.commit()
    create_hash_tables(hash_functions, c)
    conn.commit()
    id_ = 0
    for category_id in category_ids:
        print(category_id)
        train_bboxes, _ = load_data_for_category(category_id)
        train_bboxes = get_retrieval_data(train_bboxes, category_id)
        for bbox in train_bboxes:
            features = extract_features_from_bbox(bbox)
            insert_bbox_row(bbox, c, id_)
            for idx, hash_function in enumerate(hash_functions):
                hash_ = tuple(
                    np.round(
                        np.matmul(hash_function, features) / r_).astype(int)
                )
                insert_hash_row(hash_, idx, id_, c)
            id_ += 1
        conn.commit()
    conn.commit()
    conn.close()
    if use_full_dataset:
        pickle_big_data(
            hash_functions,
            '/Users/stewart/projects/stats/data/hash_functions_large.pkl')
    else:
        pickle_big_data(
            hash_functions,
            '/Users/stewart/projects/stats/data/hash_functions_small.pkl')


def determine_k():
    train_bboxes, _ = load_data_for_category(2)
    train_bboxes = get_retrieval_data(train_bboxes, 2)
    for k_ in range(1, 20):
        train_hash_dict = defaultdict(int)
        hash_function = get_random_hash_vector(k_)
        for i, bbox in enumerate(train_bboxes):
            features = extract_features_from_bbox(bbox)
            hash_ = tuple(
                np.round(
                    np.matmul(hash_function, features) / r_).astype(int)
            )
            train_hash_dict[hash_] += 1
        avg_sizes = mean(train_hash_dict.values())
        print('{}: {}'.format(k_, avg_sizes))


def compute_distance(val_features, train_bbox):
    train_features = extract_features_from_bbox(train_bbox, 'train2014')
    return np.linalg.norm(val_features - train_features)


def get_prediction(val_features, neighbors):
    distances = []
    for neighbor in neighbors:
        distances.append(
            [compute_distance(val_features, neighbor), neighbor]
        )
    nearest_neigbors = [x[1] for x in sorted(distances, key=lambda x: x[0])]
    predictions = []
    img_ids_in_preds = []
    idx = 0
    while len(predictions) < K and idx < len(neighbors):
        if nearest_neigbors[idx]['img_id'] not in img_ids_in_preds:
            img_ids_in_preds.append(nearest_neigbors[idx]['img_id'])
            predictions.append(nearest_neigbors[idx]['category_id'])
        idx += 1
    return {
        key: (val / len(predictions))
        for key, val in Counter(predictions).items()
    }, len(predictions)


def predict(bbox_to_predict, hash_functions, train_hash_dict):
    neighbors = []
    preds = np.zeros(len(category_ids))
    features = extract_features_from_bbox(bbox_to_predict, 'val2014')
    for idx, hash_function in enumerate(hash_functions):
        hash_ = tuple(
            np.round(
                np.matmul(hash_function, features) / r_).astype(int)
        )
        neighbors.extend(train_hash_dict[idx][hash_])
    if len(neighbors) > 0:
        predictions = get_prediction(features, neighbors)
    else:
        preds[0] = 1
        return preds
    for cat_id, proportion_in_category in predictions.items():
        preds[cat_id_to_idx[cat_id]] = proportion_in_category
    return preds


def determine_l():
    train_bboxes, val_bboxes = load_data_for_category(2)
    train_bboxes = get_retrieval_data(train_bboxes, 2)
    val_bboxes = get_retrieval_data(val_bboxes, 2)
    for l_ in [1, 50, 100, 300]:
        hash_functions = construct_hash_functions(l_)
        train_hash_dict = {
            idx: defaultdict(list) for idx in range(len(hash_functions))
        }
        for bbox in train_bboxes:
            features = extract_features_from_bbox(bbox, 'train2014')
            for idx, hash_function in enumerate(hash_functions):
                hash_ = tuple(
                    np.round(
                        np.matmul(hash_function, features) / r_).astype(int)
                )
                train_hash_dict[idx][hash_].append(bbox)
        prob_preds = []
        true_values = []
        n_neighbors_s = []
        n_dist_funcs = []
        for bbox in val_bboxes:
            neighbors = []
            features = extract_features_from_bbox(bbox, 'val2014')
            for idx, hash_function in enumerate(hash_functions):
                hash_ = tuple(
                    np.round(
                        np.matmul(hash_function, features) / r_).astype(int)
                )
                neighbors.extend(train_hash_dict[idx][hash_])
            if len(neighbors) > 0:
                predictions, n_neighbors = get_prediction(features, neighbors)
            else:
                n_neighbors = 0
            if bbox['category_id'] == 2:
                true_values.append(1)
            else:
                true_values.append(0)
            if len(neighbors) > 0:
                prob_preds.append(predictions.get(2, 0))
            else:
                prob_preds.append(0)
            n_neighbors_s.append(n_neighbors)
            n_dist_funcs.append(len(neighbors))
        ap_score = average_precision_score(true_values, prob_preds)
        print('Mean N neighbors: {}'.format(mean(n_neighbors_s)))
        print('Mean N computations: {}'.format(mean(n_dist_funcs)))
        print('{}: {}'.format(l_, ap_score))


if __name__ == '__main__':
    # train_and_save_model()
    for use_full_dataset in [True]:
        if use_full_dataset:
            image_features_dict = features_from_img_id_large('train2014')
            val_image_features_dict = features_from_img_id_large('val2014')
        else:
            image_features_dict = features_from_img_id('train2014')
            val_image_features_dict = features_from_img_id('val2014')
        populate_sqllite_table()
    # determine_k()
    # determine_l()
