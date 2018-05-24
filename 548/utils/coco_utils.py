from collections import defaultdict
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from pycocotools.coco import COCO
from utils.redis_utils import cache
import pickle
from math import floor, ceil
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import cv2
import random


ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
if 'stewart' in os.getenv('PYTHONPATH').lower():
    dataDir = '/Users/stewart/projects/stats/data'
else:
    dataDir = '/home/ubuntu/548/data'


@cache.cached(timeout=60 * 60 * 24 * 60)
def get_coco(data_type):
    annFile = '{}/annotations/instances_{}.json'.format(dataDir, data_type)
    return COCO(annFile)


def get_all_categories(data_type='train2014'):
    coco = get_coco(data_type)
    return coco.loadCats(coco.getCatIds())


def category_id_to_info(data_type='train2014'):
    return {cat['id']: cat for cat in get_all_categories()}


def get_all_annotations(data_type='train2014'):
    coco = get_coco(data_type)
    return coco.loadAnns(coco.getAnnIds())


@cache.cached(timeout=60 * 60 * 24 * 60)
def image_id_to_categories(data_type='train2014'):
    annotations = get_all_annotations(data_type=data_type)
    image_id_to_category_dict = defaultdict(set)
    for ann in annotations:
        image_id_to_category_dict[ann['category_id']].add(
            ann['image_id']
        )
    return image_id_to_category_dict


def get_image_ids_for_category(category_id, data_type='train2014'):
    annotations = get_all_annotations(data_type=data_type)
    img_ids = [
        ann['image_id'] for ann in annotations
        if ann['category_id'] == category_id
    ]
    return list(set(img_ids))


def get_annotations_for_images(image_ids, data_type='train2014'):
    annotations = get_all_annotations(data_type=data_type)
    img_id_to_annotation = defaultdict(list)
    for ann in annotations:
        if ann['image_id'] in image_ids:
            img_id_to_annotation[ann['image_id']].append(ann)
    return img_id_to_annotation


def get_all_img_ids(data_type='train2014'):
    return get_coco(data_type).getImgIds()


class Featurizer:
    dim = 11776  # for small features

    def __init__(self):
        # pyramidal pooling of sizes 1, 3, 6
        self.pool1 = nn.AdaptiveMaxPool2d(1)
        self.pool3 = nn.AdaptiveMaxPool2d(3)
        self.pool6 = nn.AdaptiveMaxPool2d(6)
        self.lst = [self.pool1, self.pool3, self.pool6]

    def featurize(self, projected_bbox, image_features):
        # projected_bbox: bbox projected onto final layer
        # image_features: C x W x H tensor : output of conv net
        full_image_features = torch.from_numpy(image_features)
        x, y, x1, y1 = projected_bbox
        crop = full_image_features[:, x:x1, y:y1]
        return torch.cat([self.pool1(crop).view(-1), self.pool3(crop).view(-1),
                          self.pool6(crop).view(-1)], dim=0).data.numpy()


def get_bboxes(img, num_rects=2000):
    try:
        ss.setBaseImage(img)
        ss.switchToSelectiveSearchQuality()  # good quality search
        # ss.switchToSelectiveSearchFast() # fast search
        rects = ss.process()
        return rects[:num_rects]
    except KeyboardInterrupt:
        print('keyboard interrupt')
        sys.exit()
    except:
        return None


def iou(rect1, rect2):  # rect = [x, y, w, h]
    x1, y1, w1, h1 = rect1
    X1, Y1 = x1 + w1, y1 + h1
    x2, y2, w2, h2 = rect2
    X2, Y2 = x2 + w2, y2 + h2
    a1 = (X1 - x1 + 1) * (Y1 - y1 + 1)
    a2 = (X2 - x2 + 1) * (Y2 - y2 + 1)
    x_int = max(x1, x2)
    X_int = min(X1, X2)
    y_int = max(y1, y2)
    Y_int = min(Y1, Y2)
    a_int = (X_int - x_int + 1) * (Y_int - y_int + 1) * 1.0
    if x_int > X_int or y_int > Y_int:
        a_int = 0.0
    return a_int / (a1 + a2 - a_int)


# nearest neighbor in 1-based indexing
def _nnb_1(x):
    x1 = int(floor((x + 8) / 16.0))
    x1 = max(1, min(x1, 13))
    return x1


def project_onto_feature_space(rect, image_dims):
    # project bounding box onto conv net
    # @param rect: (x, y, w, h)
    # @param image_dims: (imgx, imgy), the size of the image
    # output bbox: (x, y, x'+1, y'+1) where the box is x:x', y:y'

    # For conv 5, center of receptive field of i is i_0 = 16 i for 1-based
    # indexing
    imgx, imgy = image_dims
    x, y, w, h = rect
    # scale to 224 x 224, standard input size.
    x1, y1 = ceil((x + w) * 224 / imgx), ceil((y + h) * 224 / imgy)
    x, y = floor(x * 224 / imgx), floor(y * 224 / imgy)
    px = _nnb_1(x + 1) - 1  # inclusive
    py = _nnb_1(y + 1) - 1  # inclusive
    px1 = _nnb_1(x1 + 1)  # exclusive
    py1 = _nnb_1(y1 + 1)  # exclusive
    return [px, py, px1, py1]


def project_from_feature_space_to_image(img_dict):
    pass


def get_positive_and_easy_negative_bounding_boxes(
        img, correct_bbox, num_rects=2000, min_iou=.5, neg_to_pos_ratio=100):
    proposals = get_bboxes(img)
    correct_bboxes = []
    incorrect_bboxes = []
    if hasattr(proposals, '__iter__'):
        for proposal in proposals:
            if iou(proposal, correct_bbox) > min_iou:
                correct_bboxes.append(proposal)
            else:
                incorrect_bboxes.append(proposal)
    else:
        return [], []
    n_to_sample = min(len(correct_bboxes) * neg_to_pos_ratio,
                      len(incorrect_bboxes))
    incorrect_bboxes = random.sample(incorrect_bboxes, n_to_sample)
    return correct_bboxes, incorrect_bboxes


def get_positive_and_negative_projected_bboxes(
        img, correct_bbox, num_rects=2000, min_iou=0.5, neg_to_pos_ratio=100):
    positive_bboxes, negative_bboxes = \
        get_positive_and_easy_negative_bounding_boxes(
            img, correct_bbox, num_rects, min_iou, neg_to_pos_ratio
        )
    positive_projs = [
        project_onto_feature_space(bbox, (img.shape[1], img.shape[0]))
        for bbox in positive_bboxes
    ]
    negative_projs = [
        project_onto_feature_space(bbox, (img.shape[1], img.shape[0]))
        for bbox in negative_bboxes
    ]
    return positive_projs, negative_projs


def load_images_cv2(img_ids, data_type='train2014'):
    coco = get_coco(data_type)
    images = coco.loadImgs(img_ids)
    images = {
        img['id']:
        cv2.imread('%s/%s/%s' % (dataDir, data_type, img['file_name']))
        for img in images
    }
    return {key: val for key, val in images.items() if val is not None}


def load_images_cv2_large(img_ids, data_type='train2014'):
    coco = get_coco(data_type)
    data_type = data_type + '_2'
    images = coco.loadImgs(img_ids)
    images = {
        img['id']:
        cv2.imread('%s/%s/%s' % (dataDir, data_type, img['file_name']))
        for img in images
    }
    return {key: val for key, val in images.items() if val is not None}


def load_data_small(data_type='train2014'):
    file_location = os.path.join(
        dataDir, 'features_small', '{}.p'.format(data_type)
    )
    with open(file_location, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        [img_list, feats] = u.load()
    return img_list, feats


def load_data_large(data_type='train2014'):
    file_location = os.path.join(
        dataDir, 'features2_small', '{}.p'.format(data_type)
    )
    with open(file_location, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        [img_list, feats] = u.load()
    return img_list, feats


def load_bboxes_small(data_type='train2014'):
    file_location = os.path.join(
        dataDir, 'bboxes', '{}_bboxes.p'.format(data_type)
    )
    with open(file_location, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        [img_list, feats] = u.load()
    return img_list, feats


def load_bboxes_large(data_type='train2014'):
    file_location = os.path.join(
        dataDir, 'bboxes2', '{}_bboxes.p'.format(data_type)
    )
    with open(file_location, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        [img_list, feats] = u.load()
    return img_list, feats


@cache.cached(timeout=60 * 60 * 24 * 60)
def features_from_img_id(data_type='train2014'):
    img_ids, feats = load_data_small(data_type)
    return {
        img_id: feature for img_id, feature in zip(img_ids, feats)
    }


@cache.cached(timeout=60 * 60 * 24 * 60)
def features_from_img_id_large(data_type='train2014'):
    img_ids, feats = load_data_large(data_type)
    return {
        img_id: feature for img_id, feature in zip(img_ids, feats)
    }


def load_data_small_supercategory(data_type):
    img_list, feats = load_data_small(data_type)
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


def enocode_feature_set(y, unique_features):
    unique_features.sort()
    feature_to_idx = {
        feature: idx for idx, feature in enumerate(unique_features)
    }
    y_new = np.zeros((len(y), len(unique_features)))
    for row, y_label in zip(y_new, y):
        for category in y_label:
            row[feature_to_idx[category]] = 1
    return y_new


@cache.cached(timeout=60 * 60 * 24 * 60)
def get_hw3_categories(size, data_type):
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
    categories = coco.loadCats(coco.getCatIds())
    cat_id_to_supercat = {
        cat['id']: cat['supercategory']
        for cat in categories
    }
    category_ids = []
    for annotation in annotations:
        if cat_id_to_supercat[annotation['category_id']] in [
                'animal', 'vehicle']:
            category_ids.append(annotation['category_id'])
    return sorted(list(set(category_ids)))


def load_category_level_data_hw2(size, data_type):
    file_location = os.path.join(
        dataDir, 'features2_{}'.format(size), '{}.p'.format(data_type)
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
    image_id_to_category = defaultdict(list)
    category_ids = []
    for annotation in annotations:
        if cat_id_to_supercat[annotation['category_id']] in [
                'animal', 'vehicle']:
            image_id_to_category[annotation['image_id']].append(
                annotation['category_id'])
            category_ids.append(annotation['category_id'])
    y = [image_id_to_category[image_id] for image_id in img_list]
    y = enocode_feature_set(y, list(set(category_ids)))
    x = np.array([
        point.ravel() for idx, point in enumerate(feats)
        if img_list[idx] in image_id_to_category]
    )
    return x, y


def get_feature_for_projected_bbox(projected_bbox, data_type='train2014'):
    featurizer = Featurizer()
    image_features_dict = features_from_img_id(data_type)
    return featurizer.featurize(
        projected_bbox['projected_bbox'],
        image_features_dict[projected_bbox['img_id']])


def get_features_for_projected_bboxes(
        projected_bboxes,
        data_type='train2014'):
    featurizer = Featurizer()
    image_features_dict = features_from_img_id(data_type)
    return np.array([
        featurizer.featurize(
            bbox['projected_bbox'],
            image_features_dict[bbox['img_id']])
        for bbox in projected_bboxes
    ])


def get_features_for_bboxes(bboxes, data_type):
    featurizer = Featurizer()
    image_features_dict = features_from_img_id(data_type)
    return np.array([
        featurizer.featurize(
            project_onto_feature_space(bbox['bbox'], bbox['image_shape']),
            image_features_dict[bbox['img_id']])
        for bbox in bboxes
    ])


def get_features_for_bboxes_large(bboxes, data_type):
    featurizer = Featurizer()
    image_features_dict = features_from_img_id_large(data_type)
    return np.array([
        featurizer.featurize(
            project_onto_feature_space(bbox['bbox'], bbox['image_shape']),
            image_features_dict[bbox['img_id']])
        for bbox in bboxes
    ])


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def plot_bbox(
        bbox,
        img_id,
        data_type,
        is_large=False,
        save_fig=False,
        save_prefix=''):
    coco = get_coco(data_type)
    img = coco.loadImgs([img_id])[0]
    if is_large:
        data_type = data_type + '_2'
    img_pil = Image.open('%s/%s/%s' % (dataDir, data_type, img['file_name']))
    draw = ImageDraw.Draw(img_pil)
    x, y, w, h = bbox
    draw.rectangle(((x, y, x + w, y + h)), fill=None, outline=(255, 0, 0))
    plt.imshow(img_pil)
    if save_fig:
        ensure_dir(dataDir + '/plots/')
        plt.savefig(dataDir + save_prefix + 'bbox_{}.png'.format(img_id))
    else:
        plt.show()
