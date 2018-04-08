from matplotlib import pyplot as plt
import math
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split


cacher = {}


def gaussian_rbf_kernel(x1, x2, sigma):
    return math.exp(
        -sum((xi - xj)**2 for xi, xj in zip(x1, x2)) / (sigma ** 2))


def get_images_by_target(lfw_people, target):
    target_idx = lfw_people.target_names.tolist().index(target)
    return [
        image for idx, image in enumerate(lfw_people.images)
        if lfw_people.target[idx] == target_idx
    ]


def preprocess_data():
    lfw_people = fetch_lfw_people(min_faces_per_person=10, resize=0.4)
    positive_class = 'Rubens Barrichello'
    negative_class = 'Michael Jackson'
    x_positive = get_images_by_target(lfw_people, positive_class)
    x_negative = get_images_by_target(lfw_people, negative_class)
    y = [1] * len(x_positive) + [-1] * len(x_negative)
    return x_positive + x_negative, y


def normalize_patch(patch):
    magnitude = math.sqrt(sum(elem ** 2 for elem in patch))
    return [elem / magnitude for elem in patch]


def generate_bag_of_patches_from_image(image, p=5):
    global cacher
    key_ = str(image)
    if key_ in cacher:
        return cacher[key_]
    patches = []
    for row in range(0, len(image) - p, 2):
        for col in range(0, len(image[0]) - p, 2):
            patches.append(
                normalize_patch([
                    image[i][j]
                    for i in range(row, row + p)
                    for j in range(col, col + p)
                ])
            )
    cacher[key_] = patches
    return patches


def kernel_function(image_1, image_2, sigma):
    image_1_patches = generate_bag_of_patches_from_image(image_1)
    image_2_patches = generate_bag_of_patches_from_image(image_2)
    k = 0
    for idx, patch_1 in enumerate(image_1_patches):
        for patch_2 in image_2_patches:
            k += gaussian_rbf_kernel(
                patch_1,
                patch_2,
                sigma
            )
    return k


def sum_cross_points(kernel_param, x_points):
    global cacher
    key_ = str(kernel_param) + str(len(x_points))
    if key_ not in cacher:
        cacher[key_] = sum(
            sum(kernel_function(x_i, x_j, kernel_param) for x_i in x_points)
            for x_j in x_points
        )
    return cacher[key_]


def distance_to_centroid(x_image, x_images_in_class, sigma):
    n = len(x_images_in_class)
    return kernel_function(x_image, x_image, sigma) - (
        (2 / n) * sum(
            kernel_function(x_image, x_point, sigma)
            for x_point in x_images_in_class
        )) + (
        (1 / (n**2)) * sum_cross_points(sigma, x_images_in_class)
    )


def predict_nearest_means(x_new, x_data, y_data, sigma):
    clusters = {}
    for idx, x_row in enumerate(x_data):
        if y_data[idx] not in clusters:
            clusters[y_data[idx]] = [x_row]
        else:
            clusters[y_data[idx]].append(x_row)
    best_label = None
    closest_centroid = float('inf')
    for y_label, x_points in clusters.items():
        centroid_distance = distance_to_centroid(x_new, x_points, sigma)
        if centroid_distance < closest_centroid:
            best_label = y_label
            closest_centroid = centroid_distance
    return best_label


def compute_test_error(
        x_train, x_test, y_train, y_test, sigma):
    incorrect_pred_count = 0
    for x, y in zip(x_test, y_test):
        incorrect_pred_count += y != predict_nearest_means(
            x, x_train, y_train, sigma)
    return incorrect_pred_count / len(x_test)


def optimize_parameters(params_to_try):
    x_data, y_data = preprocess_data()
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.33)
    test_errors = []
    for param in params_to_try:
        print(param)
        test_errors.append(
            compute_test_error(
                x_train, x_test, y_train, y_test, param
            )
        )
        print(test_errors)
    return test_errors


def plot_gallery(images, titles, sigma, h=50, w=37, n_row=12, n_col=1):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())
        plt.title('Sigma = {}'.format(sigma))
    plt.show()


def plot_descending_distance_to_centroid(sigma):
    x_data, y_data = preprocess_data()
    clusters = {}
    for idx, x_row in enumerate(x_data):
        if y_data[idx] not in clusters:
            clusters[y_data[idx]] = [x_row]
        else:
            clusters[y_data[idx]].append(x_row)
    images = clusters[-1]
    image_distances = []
    for image in images:
        dist = distance_to_centroid(image, images, sigma)
        image_distances.append([image, dist])
    image_distances.sort(key=lambda x: x[1])
    titles = ['' for idx in range(12)]
    plot_gallery([image[0] for image in image_distances], titles, sigma)


if __name__ == '__main__':
    sigmas = [1, 10, 0.01, 0.1, 100, 1000]
    for sigma in sigmas:
        plot_descending_distance_to_centroid(sigma)
    gaussian_rbf_errors = optimize_parameters(sigmas)
    sigmas_transformed = [math.log(sigma, 10) for sigma in sigmas]
    plt.scatter(sigmas_transformed, gaussian_rbf_errors)
    plt.xlabel("log10(Sigma)")
    plt.ylabel("Misclassification Error Rate")
    plt.title('Error Rate by Sigma')
    plt.show()
