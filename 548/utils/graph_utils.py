from matplotlib import pyplot as plt


def draw_error_over_time(
        train_errors, test_errors, val_errors, title, y_label):
    plt.plot(train_errors, label="Train")
    plt.plot(test_errors, label="Test")
    plt.plot(val_errors, label="Validation")
    plt.legend(loc="best")
    plt.ylabel(y_label)
    plt.title(title)
