import numpy as np
from data.MGDA_dataLoaders_utils import MGDA_Data, MultiMnist_dataset, Cifar10Mnist_dataset


def one_hot_encode_data(array):
    """One hot encodes the target labels of a list of data-labels"""
    res = []
    for index in range(len(array)):
        res.append([MGDA_Data.one_hot_encode(array[index][0]),
                         MGDA_Data.one_hot_encode(array[index][1])]
                   )
    return res

def load_Cifar10Mnist_mgda():

    print("Retrieving data...")

    cifar_labels = {
        0: "Airplane", 1: "Automobile", 2: "Bird", 3: "Cat", 4: "Deer",
        5: "Dog", 6: "Frog", 7: "Horse", 8: "Ship", 9: "Truck"
    }

    data_train, data_test = Cifar10Mnist_dataset()

    X_train, y_train = zip(*data_train)
    X_test, y_test = zip(*data_test)

    X_train = np.array(X_train)
    X_test = np.array(X_test)

    # plt.imshow(data_train[0][0])
    # plt.title(f'{(cifar_labels[y_train[0][0]], y_train[0][1])}')
    # plt.show()

    y_train = np.array(one_hot_encode_data(y_train))
    y_test = np.array(one_hot_encode_data(y_test))

    print("Data is loaded")

    return X_train, X_test, y_train, y_test


def load_MultiMnist_mgda():

    print("Retrieving data...")

    data_train, data_test = MultiMnist_dataset()

    X_train, y_train = zip(*data_train)
    X_test, y_test = zip(*data_test)

    X_train = np.array(X_train)
    X_test = np.array(X_test)

    # plt.imshow(data_train[0][0])
    # plt.title(f'{(y_train[0][0].item(), y_train[0][1].item())}')
    # plt.show()

    y_train = np.array(one_hot_encode_data(y_train))
    y_test = np.array(one_hot_encode_data(y_test))

    print("Data is loaded")

    return X_train, X_test, y_train, y_test