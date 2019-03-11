from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.regularizers import l2


def build_model(
        input_shape,
        output_classes,
        n_hidden=128,
        dropout=0.3,
        optimizer=SGD()):

    kernel_regularizer = l2(0.0001)

    # (hidden -> relu -> dropout) is one set
    model = Sequential([
        Dense(
            n_hidden,
            input_shape=input_shape,
            kernel_regularizer=kernel_regularizer
        ),
        Activation('relu'),
        Dropout(dropout),
        Dense(
            n_hidden,
            kernel_regularizer=kernel_regularizer
        ),
        Activation('relu'),
        Dropout(dropout),
        Dense(output_classes),
        Activation("softmax")
    ])

    model.summary()

    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"]
    )

    return model
