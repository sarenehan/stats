from utils.coco_utils import (
    image_id_to_categories,
    load_images_cv2,
    get_annotations_for_images,
    get_positive_and_negative_projected_bboxes,
    dataDir,
    get_hw3_categories,
)
from utils.aws_utils import upload_to_s3
import os
import pickle
import random


max_bytes = 2**31 - 1
positive_training_data_save_location = dataDir + \
    '/small_{}_feature_{}.pkl'


def pickle_big_data(data, file_path):
    bytes_out = pickle.dumps(data, protocol=4)
    with open(file_path, 'wb') as f_out:
        for idx in range(0, len(bytes_out), max_bytes):
            f_out.write(bytes_out[idx:idx + max_bytes])


def unpickle_big_data(file_path):
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except:
        bytes_in = bytearray(0)
        input_size = os.path.getsize(file_path)
        with open(file_path, 'rb') as f_in:
            for _ in range(0, input_size, max_bytes):
                bytes_in += f_in.read(max_bytes)
        return pickle.loads(bytes_in)


def load_data_for_category(
        category_id,
        img_id_to_bboxes,
        data_type):
    print('Loading training data for {}'.format(category_id))
    img_ids = list(image_id_to_categories(data_type)[category_id])
    images = load_images_cv2(img_ids, data_type)
    annotations = get_annotations_for_images(img_ids, data_type)
    positive_features = []
    negative_features = []
    idx = 0
    for img_id, img in images:
        print('{} of {} for {}'.format(idx, len(images), category_id))
        idx += 1
        positive_bounding_boxes = [
            annotation['bbox'] for annotation in
            annotations[img_id]
            if annotation['category_id'] == category_id
        ]
        for proposal in img_id_to_bboxes[img_id]:
            is_a_positive_label = False
            idx = 0
            while idx < len(positive_bounding_boxes) and (
                is_a_positive_label is False):
                if iou(positive_bounding_boxes[idx], proposal) > 0.5:
                    positive_features.append(
                        {'bbox': proposal, 'img_id': img_id}
                    )
                    is_a_positive_label = True
                else:
                    idx += 1
            if not is_a_positive_label:
                negative_features.append(
                    {'bbox': proposal, 'img_id': img_id}
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
    img_ids, bboxes = load_bboxes_small(data_type=data_type)
    img_id_to_bboxes = {}
    for img_id, bboxes_for_img in zip(img_ids, bboxes):
        if img_id in img_id_to_bboxes:
            img_id_to_bboxes[img_id] = np.concatenate(
                [img_id_to_bboxes[img_id], bboxes_for_img]
            )
        else:
            img_id_to_bboxes[img_id] = bboxes_for_img
    return img_id_to_bboxes



if __name__ == '__main__':
    data_type = 'train2014'
    # data_type = 'val2014'
    # data_type = 'test2014'
    # idxes_for_node = list(range(4))
    # idxes_for_node = list(range(4, 8))
    # idxes_for_node = list(range(8, 13))
    # idxes_for_node = list(range(13, 18))
    # category_ids = get_hw3_categories('small', data_type)
    # img_id_to_bboxes = get_img_id_to_bboxes(data_type)
    # category_ids = [
    #     cat for idx, cat in enumerate(category_ids) if idx in idxes_for_node
    # ]
    category_ids = [23]
    for category_id in category_ids:
        load_data_for_category(category_id, img_id_to_bboxes, data_type)
