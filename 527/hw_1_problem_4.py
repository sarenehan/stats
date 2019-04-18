import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt


dir_ = '/Users/stewart/Desktop/'
test = dir_ + 'zip.test'
train = dir_ + 'zip.train'
numbers_to_consider = [2, 3]


def load_data():
    test_df = pd.read_csv(test, delimiter=' ', header=None)
    train_df = pd.read_csv(train, delimiter=' ', header=None)
    test_y = test_df.iloc[:, 0]
    train_y = train_df.iloc[:, 0]
    test_x = test_df.iloc[:, 1:]
    train_x = train_df.iloc[:, 1:-1]

    test_x = test_x[test_y.isin(numbers_to_consider)]
    test_y = test_y[test_y.isin(numbers_to_consider)]
    train_x = train_x[train_y.isin(numbers_to_consider)]
    train_y = train_y[train_y.isin(numbers_to_consider)]

    train_y[train_y == 3] = 1
    test_y[test_y == 3] = 1
    train_y[train_y == 2] = 0
    test_y[test_y == 2] = 0
    return train_x, train_y, test_x, test_y


def get_regressor_classification_error(preds, true):
    preds = (preds > 0.5).astype(int)
    return (preds != true).sum() / len(true)


def get_linear_model_error():
    train_x, train_y, test_x, test_y = load_data()
    linear_model = LinearRegression()
    linear_model.fit(train_x, train_y)
    train_error = get_regressor_classification_error(
        linear_model.predict(train_x), train_y)
    test_error = get_regressor_classification_error(
        linear_model.predict(test_x), test_y)
    return train_error, test_error


def get_knn_error(k_):
    train_x, train_y, test_x, test_y = load_data()
    model = KNeighborsClassifier(n_neighbors=k_)
    model.fit(train_x, train_y)
    test_error = (model.predict(test_x) != test_y).sum() / len(test_y)
    train_error = (model.predict(train_x) != train_y).sum() / len(train_y)
    return train_error, test_error


lm_train_error, lm_test_error = get_linear_model_error()
knn_train_errors = []
knn_test_errors = []
ks_to_test = [1, 3, 5, 7, 15]
for k_ in ks_to_test:
    train_error, test_error = get_knn_error(k_)
    knn_train_errors.append(train_error)
    knn_test_errors.append(test_error)


plt.scatter(ks_to_test, knn_train_errors, label='Knn Train')
plt.scatter(ks_to_test, knn_test_errors, label='Knn Test')
plt.xticks(ks_to_test)
plt.hlines(lm_train_error, 1, 15, label='Linear Model Train')
plt.hlines(lm_test_error, 1, 15, label='Linear Model Test')
plt.xlabel('K Neighbors')
plt.ylabel('Misclassification Rate')
plt.title('Linear Regression vs KNN on Zipcode Dataset')
plt.legend(loc='best')
plt.savefig('/Users/stewart/Desktop/hw_1_problem_5.png')
