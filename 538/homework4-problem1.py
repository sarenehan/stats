import random
from mpl_toolkits.mplot3d import Axes3D  # noqa
from matplotlib import pyplot as plt
import math
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split


cacher = {}
mean_patch = None
var_patch = None


def gaussian_rbf_kernel(x1, x2, sigma):
    return math.exp(
        -sum((xi - xj)**2 for xi, xj in zip(x1, x2)) / (sigma ** 2))


def get_images_by_target(lfw_people, target):
    target_idx = lfw_people.target_names.tolist().index(target)
    return [
        image for idx, image in enumerate(lfw_people.images)
        if lfw_people.target[idx] == target_idx
    ]


def get_mean_and_var_patch(images):
    sample = random.sample(images, 5)
    patches = [
        patch for image in sample
        for _, patch in generate_bag_of_patches_from_image(
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


def preprocess_data():
    global mean_patch
    global var_patch
    lfw_people = fetch_lfw_people(min_faces_per_person=10, resize=0.4)
    people = [
        'Keanu Reeves',
        'Rubens Barrichello',
        'Michael Jackson'
    ]
    images = [
        get_images_by_target(lfw_people, name)
        for name in people
    ]
    y = []
    for idx, images_of_person in enumerate(images):
        y.extend([idx] * len(images_of_person))
    x = []
    for images_of_person in images:
        x.extend(images_of_person)

    mean_patch, var_patch = get_mean_and_var_patch(x)
    return x, y


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
            new_patch = [
                image[i][j]
                for i in range(row, row + p)
                for j in range(col, col + p)
            ]
            center_of_patch = [row + ((p - 1) / 2), col + ((p - 2) / 2)]
            if normalize:
                new_patch = normalize_patch(new_patch)
            patches.append((center_of_patch, new_patch))
    cacher[key_] = patches
    return patches


def kernel_function(image_1, image_2, sigma1, sigma2):
    image_1_patches = generate_bag_of_patches_from_image(image_1)
    image_2_patches = generate_bag_of_patches_from_image(image_2)
    k = 0
    for idx_patch1, (z1, patch_1) in enumerate(image_1_patches, 1):
        for idx_patch2, (z2, patch_2) in enumerate(image_2_patches, 1):
            k += gaussian_rbf_kernel(
                patch_1,
                patch_2,
                sigma1
            ) * gaussian_rbf_kernel(z1, z2, sigma2)
    return k


def sum_cross_points(sigma1, sigma2, x_points):
    global cacher
    key_ = str(sigma1) + str(sigma2) + str(len(x_points))
    if key_ not in cacher:
        cacher[key_] = sum(
            sum(kernel_function(x_i, x_j, sigma1, sigma2) for x_i in x_points)
            for x_j in x_points
        )
    return cacher[key_]


def distance_to_centroid(x_image, x_images_in_class, sigma1, sigma2):
    n = len(x_images_in_class)
    return kernel_function(x_image, x_image, sigma1, sigma2) - (
        (2 / n) * sum(
            kernel_function(x_image, x_point, sigma1, sigma2)
            for x_point in x_images_in_class
        )) + (
        (1 / (n**2)) * sum_cross_points(sigma1, sigma2, x_images_in_class)
    )


def predict_nearest_means(x_new, x_data, y_data, sigma1, sigma2):
    clusters = {}
    for idx, x_row in enumerate(x_data):
        if y_data[idx] not in clusters:
            clusters[y_data[idx]] = [x_row]
        else:
            clusters[y_data[idx]].append(x_row)
    best_label = None
    closest_centroid = float('inf')
    for y_label, x_points in clusters.items():
        centroid_distance = distance_to_centroid(
            x_new, x_points, sigma1, sigma2)
        if centroid_distance < closest_centroid:
            best_label = y_label
            closest_centroid = centroid_distance
    return best_label


def compute_test_error(
        x_train, x_test, y_train, y_test, sigma1, sigma2):
    incorrect_pred_count = 0
    for x, y in zip(x_test, y_test):
        incorrect_pred_count += y != predict_nearest_means(
            x, x_train, y_train, sigma1, sigma2)
    return incorrect_pred_count / len(x_test)


def optimize_parameters(sigma1s, sigma2s):
    x_data, y_data = preprocess_data()
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.33)
    test_errors = []
    for sigma1 in sigma1s:
        for sigma2 in sigma2s:
            print('Sigma1: {}; Sigma2: {}'.format(sigma1, sigma2))
            test_errors.append(
                compute_test_error(
                    x_train, x_test, y_train, y_test, sigma1, sigma2
                )
            )
            print(test_errors)
    return test_errors


if __name__ == '__main__':
    sigma1s = [1, 10]
    sigma2s = [0.01, 0.1, 1, 10, 100, 1000]
    gaussian_rbf_errors = optimize_parameters(sigma1s, sigma2s)
    sigma1s_transformed = [math.log(sigma, 10) for sigma in sigma1s]
    sigmas_transformed = [math.log(sigma, 10) for sigma in sigma2s]
    x, y = zip(*[(a, b) for a in sigma1s for b in sigma2s])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, gaussian_rbf_errors)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_xlabel('Sigma p')
    ax.set_ylabel('Simga l')
    ax.set_zlabel('Error Rate')
    ax.set_title('Error Rate by Sigma_p and Simga_l')
    plt.show()
