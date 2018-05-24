from utils.coco_utils import (
    image_id_to_categories,
    load_images_cv2,
    load_images_cv2_large,
    get_annotations_for_images,
    dataDir,
    get_hw3_categories,
    load_bboxes_small,
    load_bboxes_large,
    iou,
)
import numpy as np
from utils.aws_utils import upload_to_s3
import os
import pickle

use_full_dataset = True
max_bytes = 2**31 - 1
if use_full_dataset:
    positive_training_data_save_location = dataDir + \
        '/small2_{}_feature_{}.pkl'
else:
    positive_training_data_save_location = dataDir + \
        '/small_{}_feature_{}.pkl'


def pickle_big_data(data, file_path):
    bytes_out = pickle.dumps(data, protocol=4)
    with open(file_path, 'wb') as f_out:
        for idx in range(0, len(bytes_out), max_bytes):
            f_out.write(bytes_out[idx:idx + max_bytes])


def load_data_for_category(
        category_id,
        img_id_to_bboxes,
        data_type):
    print('Loading training data for {}'.format(category_id))
    img_ids = list(image_id_to_categories(data_type)[category_id])
    if use_full_dataset:
        images = load_images_cv2_large(img_ids, data_type)
    else:
        images = load_images_cv2(img_ids, data_type)
    annotations = get_annotations_for_images(img_ids, data_type)
    positive_features = []
    negative_features = []
    idx = 0
    for img_id, img in images.items():
        print('{} of {} for {}'.format(idx, len(images), category_id))
        idx += 1
        positive_bounding_boxes = [
            annotation['bbox'] for annotation in
            annotations[img_id]
            if annotation['category_id'] == category_id
        ]
        for proposal in img_id_to_bboxes.get(img_id, []):
            is_a_positive_label = False
            i = 0
            while i < len(positive_bounding_boxes) and (
                    is_a_positive_label is False):
                if iou(positive_bounding_boxes[i], proposal) > 0.5:
                    positive_features.append(
                        {
                            'bbox': proposal,
                            'img_id': img_id,
                            'image_shape': (img.shape[1], img.shape[0])
                        }
                    )
                    is_a_positive_label = True
                else:
                    i += 1
            if not is_a_positive_label:
                negative_features.append(
                    {
                        'bbox': proposal,
                        'img_id': img_id,
                        'image_shape': (img.shape[1], img.shape[0])
                    }
                )
    file_path = positive_training_data_save_location.format(
        data_type, category_id)
    data_to_save = {
        'positive_features': positive_features,
        'negative_features': negative_features
    }
    pickle_big_data(data_to_save, file_path)
    upload_to_s3(
        'data/training_features/' +
        positive_training_data_save_location.format(
            data_type, category_id).split('/')[-1], file_path)
    os.remove(file_path)


def get_img_id_to_bboxes(data_type):
    if use_full_dataset:
        img_ids, bboxes = load_bboxes_large(data_type=data_type)
    else:
        img_ids, bboxes = load_bboxes_small(data_type=data_type)
    img_id_to_bboxes = {}
    for img_id, bboxes_for_img in zip(img_ids, bboxes):
        if bboxes_for_img is not None:
            if img_id in img_id_to_bboxes:
                img_id_to_bboxes[img_id] = np.concatenate(
                    [img_id_to_bboxes[img_id], bboxes_for_img]
                )
            else:
                img_id_to_bboxes[img_id] = bboxes_for_img
    return img_id_to_bboxes


if __name__ == '__main__':
    for data_type in ['train2014', 'val2014', 'test2014']:
        category_ids = get_hw3_categories('small', data_type)
        img_id_to_bboxes = get_img_id_to_bboxes(data_type)
        for category_id in category_ids:
            load_data_for_category(category_id, img_id_to_bboxes, data_type)
