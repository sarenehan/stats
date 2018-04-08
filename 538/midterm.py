import numpy as np
from matplotlib import pyplot as plt
import math
import random
from sklearn.datasets import fetch_lfw_people

cacher = {}
mean_patch = None
var_patch = None


def get_mean_and_var_patch(images):
    sample = random.sample(images, 5)
    patches = [
        patch for image in sample
        for patch in generate_bag_of_patches_from_image(
            image, normalize=False)
    ]
    mean_patch = [
        sum(patch[idx] for patch in patches) / len(patches)
        for idx in range(len(patches[0]))
    ]
    var_patch = [
        math.sqrt(
            (sum((patch[idx] - mean_patch[idx]) ** 2 for patch in patches) / (
                len(patches) - 1))
        ) for idx in range(len(patches[0]))
    ]
    return mean_patch, var_patch


def get_images_by_target(lfw_people, target):
    target_idx = lfw_people.target_names.tolist().index(target)
    return [
        image for idx, image in enumerate(lfw_people.images)
        if lfw_people.target[idx] == target_idx
    ]


def preprocess_data():
    global mean_patch
    global var_patch
    lfw_people = fetch_lfw_people(min_faces_per_person=10, resize=0.4)
    positive_class = 'Rubens Barrichello'
    negative_class = 'Michael Jackson'
    x_positive = get_images_by_target(lfw_people, positive_class)
    x_negative = get_images_by_target(lfw_people, negative_class)
    x = x_positive + x_negative
    y = [1] * len(x_positive) + [-1] * len(x_negative)
    mean_patch, var_patch = get_mean_and_var_patch(x)
    return x, y


def plot_images(images, n_row=4, n_col=6, h=50, w=37):
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.xticks(())
        plt.yticks(())
    plt.show()


def gaussian_rbf_kernel(x1, x2, sigma):
    return math.exp(
        -sum((xi - xj)**2 for xi, xj in zip(x1, x2)) / (sigma ** 2))


def normalize_patch(patch):
    patch = [
        (patch_elem - mean_patch[idx]) / var_patch[idx]
        for idx, patch_elem in enumerate(patch)
    ]
    magnitude = math.sqrt(sum(elem ** 2 for elem in patch))
    return [elem / magnitude for elem in patch]


def generate_bag_of_patches_from_image(image, p=5, normalize=True):
    global cacher
    key_ = str(image)
    if key_ in cacher:
        return cacher[key_]
    patches = []
    for row in range(0, len(image) - p, 2):
        for col in range(0, len(image[0]) - p, 2):
            patch = [
                image[i][j]
                for i in range(row, row + p)
                for j in range(col, col + p)
            ]
            if normalize:
                patch = normalize_patch(patch)
            patches.append(patch)
    cacher[key_] = patches
    return patches


def kernel_function(image_1, image_2, sigma):
    image_1_patches = generate_bag_of_patches_from_image(image_1)
    image_2_patches = generate_bag_of_patches_from_image(image_2)
    k = 0
    for patch_1 in image_1_patches:
        for patch_2 in image_2_patches:
            k += gaussian_rbf_kernel(
                patch_1,
                patch_2,
                sigma
            )
    return k


def get_gaussian_kernel_matrix(data, sigma):
    n = len(data)
    kernel_matrix = np.array([[None] * n] * n)
    for i in range(n):
        for j in range(n):
            kernel_matrix[i][j] = kernel_function(
                data[i], data[j], sigma)
    return kernel_matrix


def change_point_function(t, kernel_matrix):
    n = len(kernel_matrix)

    term1 = ((n - t) / (n * t)) * sum(
        kernel_matrix[i][j] for i in range(t) for j in range(t)
    )
    term2 = (t / (n * (n - t))) * sum(
        kernel_matrix[i][j] for i in range(t + 1, n) for j in range(t + 1, n)
    )
    term3 = (- 2 / n) * sum(
        kernel_matrix[i][j] for i in range(1, t) for j in range(t + 1, n)
    )
    return term1 + term2 + term3


def plot_data(x, y, title, xlabel, ylabel):
    plt.scatter(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    images, labels = preprocess_data()
    plot_images(images)
    kernel_matrix = get_gaussian_kernel_matrix(images, 1)
    print(kernel_matrix)
    t_star = 12
    abs_error_by_sigma = []
    for sigma in [0.01, 0.1, 1, 10, 100]:
        kernel_matrix = get_gaussian_kernel_matrix(images, sigma)
        change_point_function_vals = []
        for t in range(1, len(images)):
            change_point_func_val = change_point_function(t, kernel_matrix)
            change_point_function_vals.append(change_point_func_val)
        plot_data(
            list(range(1, len(images))),
            change_point_function_vals,
            title='Fn(t) for sigma = {}'.format(sigma),
            xlabel='t',
            ylabel='Fn(t)')
        best_change_point = change_point_function_vals.index(
            max(change_point_function_vals)) + 1
        print(sigma)
        print(change_point_function_vals)
        print('Best change point for sigma = {}: {}'.format(
            sigma, best_change_point))
        abs_error_by_sigma.append([sigma, abs(best_change_point - t_star)])
    print('\n\n')
    print(abs_error_by_sigma)
