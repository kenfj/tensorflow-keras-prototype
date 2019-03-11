from keras.datasets import mnist
from keras.utils import np_utils


def mnist_data():
    NB_CLASSES = 10
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # reshape (60000, 28, 28) -> (10000, 784)
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")

    # normalize data
    X_train /= 255
    X_test /= 255

    Y_train = np_utils.to_categorical(y_train, NB_CLASSES)
    Y_test = np_utils.to_categorical(y_test, NB_CLASSES)

    return X_train, Y_train, X_test, Y_test
