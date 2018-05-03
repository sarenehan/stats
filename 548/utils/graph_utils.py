from matplotlib import pyplot as plt


def draw_error_over_time(
        train_errors, test_errors, val_errors, title, y_label):
    plt.plot(train_errors, label="Train")
    plt.plot(test_errors, label="Test")
    plt.plot(val_errors, label="Validation")
    plt.legend(loc="best")
    plt.ylabel(y_label)
    plt.title(title)


def draw_train_and_dev_errors_over_time(
        train_errors, dev_errors, title, y_label):
    plt.plot(train_errors, label="Train")
    plt.plot(dev_errors, label="Development")
    plt.legend(loc="best")
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()
