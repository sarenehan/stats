import torch
import numpy as np
from utils.coco_utils import dataDir
import pickle


def load_data_for_category():
    with open(dataDir + '/small_training_features_cat_2.pkl', 'rb') as f:
        data = pickle.load(f)
    return data


def initialize_weights(d, k):
    return Variable((torch.rand(d, k) - .5) / 1000, requires_grad=True)


def compute_error_logistic_regression(w, lambda_, x, y):
    preds = 1 / 1 + torch.exp(-w.matmul(x))
    return (0.5 * (y - preds).pow(2)) + (lambda_ / 2) * w.norm()


def train_model(data, intercept_=True):
    positive_features = data['positive_features']
    negative_features = data['negative_features']
    y = np.array([1] * len(positive_features) + [0] * len(negative_features))
    x = np.array(positive_features + negative_features)
    if intercept_:
        x = np.hstack((np.ones((len(x), 1)), x))
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    # can use warm_start=True to reuse solution to previous call to fit.


if __name__ == '__main__':
    data = load_data_for_category()
