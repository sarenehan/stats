from pycocotools.coco import COCO
from utils.redis_utils import cache
import pickle
import os
import numpy as np


if 'stewart' in os.getenv('PYTHONPATH').lower():
    dataDir = '/Users/stewart/projects/stats/548/data'
else:
    dataDir = '/home/ubuntu/548/data'


def get_coco(data_type):
    annFile = '{}/annotations/instances_{}.json'.format(dataDir, data_type)
    return COCO(annFile)


@cache.cached(timeout=60 * 60 * 24 * 60)
def load_data_small_supercategory(data_type):
    file_location = os.path.join(
        dataDir, 'features_small', '{}.p'.format(data_type)
    )
    with open(file_location, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        [img_list, feats] = u.load()
    coco = get_coco(data_type)
    annotation_ids = coco.getAnnIds(imgIds=img_list,  iscrowd=None)
    annotations = coco.loadAnns(annotation_ids)
    categories = coco.loadCats(coco.getCatIds())
    cat_id_to_supercat = {
        cat['id']: cat['supercategory']
        for cat in categories
    }
    image_id_to_super_cat = {
        annotation['image_id']: cat_id_to_supercat[annotation['category_id']]
        for annotation in annotations
    }
    y = np.array([
        1 if image_id_to_super_cat[image_id] == 'animal' else -1
        for image_id in img_list
    ])
    x = np.array([point.ravel() for point in feats])
    return x, y


def load_data_tiny_supercategory(data_type):
    file_location = os.path.join(
        dataDir, 'features_tiny', '{}.p'.format(data_type)
    )
    with open(file_location, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        [img_list, feats] = u.load()
    coco = get_coco(data_type)
    annotation_ids = coco.getAnnIds(imgIds=img_list,  iscrowd=None)
    annotations = coco.loadAnns(annotation_ids)
    categories = coco.loadCats(coco.getCatIds())
    cat_id_to_supercat = {
        cat['id']: cat['supercategory']
        for cat in categories
    }
    image_id_to_super_cat = {
        annotation['image_id']: cat_id_to_supercat[annotation['category_id']]
        for annotation in annotations
    }
    y = np.array([
        1 if image_id_to_super_cat[image_id] == 'animal' else -1
        for image_id in img_list
    ])
    x = np.array([point.ravel() for point in feats])
    return x, y


def enocode_feature_set(y):
    unique_features = list(set(y))
    unique_features.sort()
    feature_to_idx = {
        feature: idx for idx, feature in enumerate(unique_features)
    }
    y_new = np.zeros((len(y), len(unique_features)))
    for row, y_label in zip(y_new, y):
        row[feature_to_idx[y_label]] = 1
    return y_new


def load_category_level_data(size, data_type):
    file_location = os.path.join(
        dataDir, 'features_{}'.format(size), '{}.p'.format(data_type)
    )
    with open(file_location, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        [img_list, feats] = u.load()
    coco = get_coco(data_type)
    annotation_ids = coco.getAnnIds(imgIds=img_list,  iscrowd=None)
    annotations = coco.loadAnns(annotation_ids)
    image_id_to_category = {
        annotation['image_id']: annotation['category_id']
        for annotation in annotations
    }
    y = np.array([image_id_to_category[image_id] for image_id in img_list])
    y = enocode_feature_set(y)
    x = np.array([point.ravel() for point in feats])
    return x, y
