from utils.coco_utils import (
    get_image_ids_for_category,
    load_images_cv2,
    get_annotations_for_images,
    get_positive_and_easy_negative_features_for_image,
    dataDir,
    get_hw3_categories,
)
from utils.aws_utils import upload_to_s3
import os
import pickle
import random


max_bytes = 2**31 - 1
# data_type = 'train2014'
data_type = 'test2014'
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


def load_data_for_category(category_id):
    print('Loading training data for {}'.format(category_id))
    img_ids = get_image_ids_for_category(category_id, data_type)
    images = load_images_cv2(img_ids, data_type)
    annotations = get_annotations_for_images(img_ids, data_type)
    positive_features = []
    negative_features = []
    idx = 0
    if len(images) > 50:
        images = random.sample(list(images.items()), 50)
    else:
        images = list(images.items())
    for img_id, img in images:
        print('{} of {} for {}'.format(idx, len(images), category_id))
        idx += 1
        for annotation in annotations[img_id]:
            correct_bbox = annotation['bbox']
            pos_features_to_add, neg_pos_features_to_add = \
                get_positive_and_easy_negative_features_for_image(
                    img_id, img, correct_bbox,
                    data_type=data_type, num_rects=1000)
            positive_features.extend(pos_features_to_add)
            negative_features.extend(neg_pos_features_to_add)
    file_path = positive_training_data_save_location.format(
        data_type, category_id)
    if len(positive_features):
        data_to_save = {
            'positive_features': positive_features,
            'negative_features': negative_features
        }
        pickle_big_data(data_to_save, file_path)
        upload_to_s3(
            'data/training_features/' +
            positive_training_data_save_location.format(
                data_type, category_id), file_path)
        os.remove(file_path)


if __name__ == '__main__':
    # idxes_for_node = list(range(4))
    # idxes_for_node = list(range(4, 8))
    # idxes_for_node = list(range(8, 13))
    idxes_for_node = list(range(13, 18))
    category_ids = get_hw3_categories('small', data_type)
    category_ids = [
        cat for idx, cat in enumerate(category_ids) if idx in idxes_for_node
    ]
    for category_id in category_ids:
        load_data_for_category(category_id)
