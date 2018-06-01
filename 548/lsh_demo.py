import boto3
import pickle
import random
from utils.coco_utils import (
    get_hw3_categories,
    Featurizer,
    project_onto_feature_space,
    features_from_img_id_large,
    category_id_to_info,
    plot_bbox,
)
import sqlite3
from collections import Counter
import numpy as np
from test_examples_for_demo import good_examples
from utils.python_utils import unpickle_big_data


r_ = 500
K = 40
featurizer = Featurizer()
category_ids = [0] + get_hw3_categories('small', 'train2014')
cat_id_to_idx = {cat_id: idx for idx, cat_id in enumerate(category_ids)}
idx_to_cat_id = dict(enumerate(category_ids))
image_features_dict = features_from_img_id_large('test2014')
image_features_dict_train = features_from_img_id_large('train2014')


def get_retrieval_data(data, category_id):
    data_to_return = []
    positive_features = data['positive_features']
    for bbox in positive_features:
        bbox['category_id'] = category_id
        data_to_return.append(bbox)
    if len(data_to_return) > 1000:
        data_to_return = random.sample(data_to_return, 1000)
    random.shuffle(data_to_return)
    return np.array(data_to_return)


def load_data_for_category_(category_id):
    session = boto3.session.Session()
    s3client = session.client('s3')
    prefix = 'small2'
    response = s3client.get_object(
        Bucket='stat-548',
        Key='data/training_features/{}_test2014_knn_features_{}.pkl'.format(
            prefix, category_id))
    data = pickle.loads(response['Body'].read())
    return get_retrieval_data(data, category_id)


def load_data_for_category(category_id):
    bboxes = good_examples[category_id]
    random.shuffle(bboxes)
    return bboxes


def load_hash_functions():
    file_ = 'hash_functions_large.pkl'
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
    suffix = 'large'
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
        (neighbor, compute_distance(features, neighbor))
        for neighbor in neighbors
    ]
    neighbors_with_dist.sort(key=lambda x: x[1])
    preds_w_bboxes = [neighbor[0] for neighbor in neighbors_with_dist[:K]]
    preds = [neighbor['category_id'] for neighbor in preds_w_bboxes]
    output = np.zeros(len(category_ids))
    for pred, n_occurences in Counter(preds).items():
        output[cat_id_to_idx[pred]] = n_occurences / len(preds)
    return output, preds_w_bboxes


def print_category_ids(cat_id_to_info_dict, valid_cat_ids):
    for cat_id, info in cat_id_to_info_dict.items():
        if cat_id in valid_cat_ids:
            print('{:5d}: {:20s}'.format(cat_id, info['name']))


def print_predictions(predictions, cat_id_to_info_dict):
    print('{:20s} {:20s}\n'.format('Category', 'Prediction'))
    for idx, pred in enumerate(predictions):
        cat_id = idx_to_cat_id[idx]
        if cat_id in cat_id_to_info_dict:
            category = cat_id_to_info_dict[cat_id]['name']
        else:
            category = 'None'
        print('{:20s} {:20f}'.format(category, pred))


def get_good_test_examples():
    hash_functions = load_hash_functions()
    test_example_dict = {}
    for cat_id in get_hw3_categories('small', 'train2014'):
        print(cat_id)
        rows_for_category = []
        data = load_data_for_category_(cat_id)
        i = 0
        while len(rows_for_category) < 20 and i < len(data):
            output, _ = predict(data[i], hash_functions)
            if output[cat_id_to_idx[cat_id]] > 0.2:
                rows_for_category.append(data[i])
            i += 1
        test_example_dict[cat_id] = rows_for_category
    print(test_example_dict)


def main():
    hash_functions = load_hash_functions()
    cat_id_to_info_dict = category_id_to_info()
    valid_cat_ids = get_hw3_categories('small', 'train2014')
    while True:
        is_valid = False
        print_category_ids(cat_id_to_info_dict, valid_cat_ids)
        while not is_valid:
            try:
                category_id = int(input(
                    '\n\nEnter a category id to find neighbors for: '))
                if category_id in valid_cat_ids:
                    is_valid = True
            except ValueError:
                pass
        data = load_data_for_category(category_id)
        if not len(data):
            data = load_data_for_category_(category_id)
        bbox = None
        i = 0
        while bbox is None:
            answer = input(
                """
                \n\Find neighbors for this bounding box? \n{}  \ny/N\n
                """.format(data[i])
            )
            if answer.lower().strip() == 'y':
                bbox = data[i]
            else:
                i += 1
        print('You have selected this bounding in this image. (may take up to 10 seconds to display)')  # noqa
        plot_bbox(
            bbox['bbox'],
            bbox['img_id'],
            'test2014',
            is_large=True
        )
        output, preds_w_bboxes = predict(bbox, hash_functions, K=K)
        print('\n\n')
        print_predictions(output, cat_id_to_info_dict)
        print('Here is the nearest neighbor! (may take up to 10 seconds to display)')  # noqa
        plot_bbox(
            preds_w_bboxes[0]['bbox'],
            preds_w_bboxes[0]['img_id'],
            'train2014',
            is_large=True
        )


if __name__ == '__main__':
    main()
    # get_good_test_examples()
