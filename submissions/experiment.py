# Imports and Environment Settings
import re

import numpy as np
import pandas as pd
import seaborn as sns  # for nicer plots
from matplotlib import pyplot as plt
from sympy import limit

sns.set(style="darkgrid")  # default style

import tensorflow as tf
from tensorflow.keras.datasets import imdb

tf.get_logger().setLevel("INFO")

# Dataset

(X_train, Y_train), (X_test, Y_test) = imdb.load_data(
    path="imdb.npz",
    num_words=None,
    skip_top=0,
    maxlen=None,
    seed=113,
    start_char=1,
    oov_char=2,
    index_from=3,
)

DEFAULT_SEQUENCE_SIZE = 20
DEFAULT_MAX_TOKEN_ID = 1000
DEFAULT_HIDDEN_LAYER_UNITS = 10
DEFAULT_EMBEDDINGS = 2

# Preprocess Features to Pad and Reduce Sequences and Limit Vocabulary


def preprocess(
    train_sequences=X_train,
    test_sequences=X_test,
    max_length=DEFAULT_SEQUENCE_SIZE,
    max_token_id=DEFAULT_MAX_TOKEN_ID,
    oov_id=2,
):
    sequence_sets = [train_sequences, test_sequences]

    result_sets = [
        np.array(
            list(
                tf.keras.preprocessing.sequence.pad_sequences(
                    sequence_set, maxlen=max_length, padding="post", value=0
                )
            )
        )
        for sequence_set in sequence_sets
    ]

    for result_set in result_sets:
        result_set[result_set >= max_token_id] = oov_id

    return result_sets[0], result_sets[1]


def experiment(
    sequence_length=DEFAULT_SEQUENCE_SIZE,
    vocab_size=DEFAULT_MAX_TOKEN_ID,
    hidden_units=DEFAULT_HIDDEN_LAYER_UNITS,
    embedding_dim=DEFAULT_EMBEDDINGS,
    verbose=1,
):

    train_features_reduced, test_features_reduced = preprocess(
        max_length=sequence_length, max_token_id=vocab_size,
    )

    model = build_experiment_model(
        vocab_size=vocab_size,
        sequence_length=sequence_length,
        hidden_units=hidden_units,
        embedding_dim=embedding_dim,
    )

    history = model.fit(
        x=train_features_reduced,
        y=Y_train,
        epochs=5,
        batch_size=64,
        validation_split=0.1,
        verbose=verbose,
    )

    test_accuracy = model.evaluate(test_features_reduced, Y_test)[1]

    if verbose == 1:
        history_report(model, history)

    return model, history, test_accuracy


def build_experiment_model(
    vocab_size=DEFAULT_MAX_TOKEN_ID,
    sequence_length=DEFAULT_SEQUENCE_SIZE,
    hidden_units=DEFAULT_HIDDEN_LAYER_UNITS,
    embedding_dim=DEFAULT_EMBEDDINGS,
):
    """Build a tf.keras model using embeddings."""
    # Clear session and remove randomness.
    tf.keras.backend.clear_session()
    tf.random.set_seed(0)

    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.Embedding(
            input_dim=vocab_size, output_dim=embedding_dim, input_length=sequence_length
        )
    )

    # This layer averages over the first dimension of the input by default.
    model.add(tf.keras.layers.GlobalAveragePooling1D())

    # Hidden layer
    model.add(tf.keras.layers.Dense(units=hidden_units, activation="relu"))

    model.add(
        tf.keras.layers.Dense(
            units=1,  # output dim (for binary classification)
            activation="sigmoid",  # apply the sigmoid function!
        )
    )

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model


def history_report(model, history, caption=None):
    if caption:
        print(caption)
    history = pd.DataFrame(history.history)
    plot_loss_history(history)
    plot_accuracy_history(history)

    final_training_accuracy, final_validation_accuracy = get_final_accuracy(history)

    print(f"Final training accuracy: {final_training_accuracy:.4f}")
    print(f"Final validation accuracy: {final_validation_accuracy:.4f}")
    print(f"Model parameter count: {get_total_parameters(model):,}")


def plot_loss_history(history):
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.xticks(range(0, len(history["loss"] + 1)))
    plt.plot(history["loss"], label="training", marker="o")
    plt.plot(history["val_loss"], label="validation", marker="o")
    plt.legend()
    plt.show()


def plot_accuracy_history(history):
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.xticks(range(0, len(history["accuracy"] + 1)))
    plt.plot(history["accuracy"], label="training", marker="o")
    plt.plot(history["val_accuracy"], label="validation", marker="o")
    plt.legend()
    plt.show()


def get_final_accuracy(history):
    return history["accuracy"].values[-1], history["val_accuracy"].values[-1]


def get_total_parameters(model):
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    summarystring = "\n".join(stringlist)
    total_parameter_string = re.search("Total params: (.*)\n", summarystring).group(1)
    return int(total_parameter_string.replace(",", ""))
