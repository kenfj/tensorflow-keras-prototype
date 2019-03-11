# %%
import numpy as np
from keras.optimizers import RMSprop
from keras.backend import clear_session

from lib.callbacks import tensorboard, early_stopping
from lib.prep_data import mnist_data
from lib.build_model import build_model

np.random.seed(1671)    # for reproducibility

# %% [markdown]
# ## Setup Data

# %%
X_train, Y_train, X_test, Y_test = mnist_data()
# shape: (10000, 784)

print(X_train.shape[0], "train samples")
print(X_test.shape[0], "test samples")

# %% [markdown]
# ## Run Model


# %%

def build_and_fit_model(dropout):
    run_name = "dropout_" + str(dropout)

    optimizer = RMSprop()
    # optimizer = Adam()

    model = build_model(
        input_shape=(X_train.shape[1],),
        output_classes=10,
        dropout=dropout,
        optimizer=optimizer
    )

    batch_size = 128
    epochs = 20
    validation_split = 0.2
    callbacks = [
        tensorboard(run_name=run_name),
        early_stopping(5)
    ]

    model.fit(
        X_train,
        Y_train,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=2,
        validation_split=validation_split,
    )

    return model

# %% [markdown]
# ## Compare Models


# %%

dropouts = [0.1, 0.2, 0.3]
scores = []

for dropout in dropouts:
    clear_session()
    model = build_and_fit_model(dropout=dropout)

    score = model.evaluate(X_test, Y_test, verbose=1)
    scores.append(score)

# %%

for score in scores:
    print("Test score: %.4f Test accuracy: %.4f" %
          (score[0], score[1])
          )
