from utils.coco_utils import (
    get_image_ids_for_category,
    load_images_cv2,
    get_annotations_for_images,
    get_positive_and_easy_negative_features_for_image,
    dataDir,
    get_all_categories,
)
import os
import pickle
import random
import boto
import boto.s3
from boto.s3.key import Key


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


def upload_to_s3(file_path):
    conn = boto.connect_s3(
        'AKIAIHIM4TAFUFPV2CMA',
        'yVEsdhCoizkiMr4/G/YO+nnhERzUF+/HcvEY5dJ3')
    bucket = conn.get_bucket('stat-548')
    k = Key(bucket)
    k.key = 'data/training_features/' + file_path.split('/')[-1]
    k.set_contents_from_filename(
        file_path,
        num_cb=10
    )


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
        upload_to_s3(file_path)
        os.remove(file_path)


if __name__ == '__main__':
    idxes_for_node = list(range(20))
    # idxes_for_node = list(range(20, 40))
    # idxes_for_node = list(range(40, 60))
    # idxes_for_node = list(range(60, 80))
    categories = get_all_categories(data_type)
    categories.sort(key=lambda x: x['id'])
    categories = [
        cat for idx, cat in enumerate(categories) if idx in idxes_for_node
    ]
    for category in categories:
        load_data_for_category(category['id'])
