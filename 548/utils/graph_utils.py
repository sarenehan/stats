from matplotlib import pyplot as plt


default_graph_save_location_dir = '/Users/stewart/Desktop/'


def draw_error_over_time(
        train_errors, test_errors, val_errors, title, y_label):
    plt.plot(train_errors, label="Train")
    plt.plot(test_errors, label="Test")
    plt.plot(val_errors, label="Validation")
    plt.legend(loc="best")
    plt.ylabel(y_label)
    plt.title(title)


def draw_train_and_dev_errors_over_time(
        train_errors, dev_errors, title, y_label,
        save_fig=True, save_location_dir=default_graph_save_location_dir):
    plt.figure()
    plt.plot(train_errors, label="Train")
    plt.plot(dev_errors, label="Development")
    plt.legend(loc="best")
    plt.ylabel(y_label)
    plt.title(title)
    if save_fig:
        plt.savefig(default_graph_save_location_dir + title + '.png')
    else:
        plt.show()


def draw_scatterplot(
        x,
        y,
        xlabel='',
        ylabel='',
        title='',
        color='b',
        save_location=None):
    plt.scatter(x, y, color=color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if save_location:
        plt.savefig(save_location)
        plt.close("all")
    else:
        plt.show()
