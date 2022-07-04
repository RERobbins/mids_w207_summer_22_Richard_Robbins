# Imports and Environment Settings
import itertools
import math
import re
import pickle

import numpy as np
import pandas as pd
import seaborn as sns  # for nicer plots
from matplotlib import pyplot as plt
from sympy import limit

sns.set(style="darkgrid")  # default style

import tensorflow as tf
from tensorflow.keras.datasets import imdb

tf.config.set_visible_devices([], "GPU")
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
DEFAULT_HIDDEN_LAYERS = []
DEFAULT_EMBEDDING_DIM = 2
DEFAULT_DROPOUT_RATE = 0.5

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
    hidden_layers=DEFAULT_HIDDEN_LAYERS,
    embedding_dim=DEFAULT_EMBEDDING_DIM,
    dropout_rate=DEFAULT_DROPOUT_RATE,
    epochs=5,
    verbose=1,
):

    train_features_reduced, test_features_reduced = preprocess(
        max_length=sequence_length, max_token_id=vocab_size
    )

    model = build_experiment_model(
        vocab_size=vocab_size,
        sequence_length=sequence_length,
        hidden_layers=hidden_layers,
        embedding_dim=embedding_dim,
        dropout_rate=dropout_rate,
    )

    history = model.fit(
        x=train_features_reduced,
        y=Y_train,
        epochs=epochs,
        batch_size=64,
        validation_split=0.1,
        verbose=verbose,
    )

    train_accuracy, validation_accuracy = get_final_accuracy(history)
    test_accuracy = model.evaluate(test_features_reduced, Y_test, verbose=verbose)[1]

    print(
        f"Final accuracy: train {train_accuracy:.4f} validation {validation_accuracy:.4f} test {test_accuracy:.4f}"
    )

    if verbose == 1:
        history_report(model, history)

    return model, history, validation_accuracy, test_accuracy


def next_step(
    sequence_length=DEFAULT_SEQUENCE_SIZE,
    vocab_size=DEFAULT_MAX_TOKEN_ID,
    hidden_layers=DEFAULT_HIDDEN_LAYERS,
    embedding_dim=DEFAULT_EMBEDDING_DIM,
    dropout_rate=DEFAULT_DROPOUT_RATE,
    epochs=5,
    verbose=1,
    exploration_rate=0.1,
    test_run=True,
):

    trial_hidden_layer_count = len(hidden_layers) + 1

    trial_vocab_multiple = 2 ** trial_hidden_layer_count
    preliminary_trial_vocab_size = vocab_size * (1 + exploration_rate)
    trial_vocab_size = round_to_next_multiple(
        preliminary_trial_vocab_size, trial_vocab_multiple
    )

    trial_sequence_length = round(sequence_length * (1 + exploration_rate))

    trial_hidden_layers = [
        int(trial_vocab_size / (2 ** idx)) for idx in range(trial_hidden_layer_count)
    ]

    trial_vocab_layer_pairs = [(trial_vocab_size, trial_hidden_layers[:-2]),
                               (trial_vocab_size, trial_hidden_layers[:-1]),
                               (trial_vocab_size, trial_hidden_layers),
                               (trial_vocab_size, [100]),
                               (trial_vocab_size, [100, 20]),
                              ]

    if len(hidden_layers) == 0:
        hidden_layers_extended = [math.ceil(vocab_size/2)]
    else:
        hidden_layers_extended = hidden_layers + [math.ceil(hidden_layers[-1] / 2)
                                                            ]
    base_vocab_layer_pairs = [
        (vocab_size, hidden_layers[:-1]),
        (vocab_size, hidden_layers),
        (vocab_size, hidden_layers_extended),
        (vocab_size, [100]),
        (vocab_size, [100, 20]),
    ]

    vocab_layer_pairs = [base_vocab_layer_pairs, trial_vocab_layer_pairs]
    sequences = [sequence_length, trial_sequence_length]
  
    embedding_dims = [
        dim
        for dim in [embedding_dim - 1, embedding_dim, embedding_dim + 1]
        if dim > 0
    ]

    dropout_rates = [
        rate
        for rate in [
            dropout_rate - 0.1,
            dropout_rate,
            dropout_rate + 0.1,
        ]
        if 0 < rate < 1
    ]
    
    parameters = itertools.product(
        vocab_layer_pairs, sequences, embedding_dims, dropout_rates
    )

    results = {}

    for vocab_layer_pairs, sequence_length, embedding_dim, dropout_rate in parameters:
        for vocab_size, hidden_layers in vocab_layer_pairs:
            print(
                f"{vocab_size=} {sequence_length=} {hidden_layers=} {embedding_dim=} {dropout_rate=}"
            )
            
            if test_run:
                continue
            
            _, _, validation_accuracy, _ = experiment(
                vocab_size=vocab_size,
                sequence_length=sequence_length,
                hidden_layers=hidden_layers,
                embedding_dim=embedding_dim,
                dropout_rate=dropout_rate,
                epochs=epochs,
                verbose=verbose,
            )
            results[
                (
                    vocab_size,
                    sequence_length,
                    tuple(hidden_layers),
                    embedding_dim,
                    dropout_rate,
                )
            ] = validation_accuracy

            with open("results_in_play.pickle", "wb") as f:
                pickle.dump(results, f)

    max_value_key = None
    if not test_run:
        max_value_key = max(results, key=results.get)
        max_value_accuracy = results[max_value_key]
        print(
            f"Maximum Validation Accuracy from Step: {max_value_accuracy} with Key: {max_value_key}"
        )

    return results, max_value_key


def round_to_next_multiple(number, multiple):
    return math.ceil(number / multiple) * multiple


def build_experiment_model(
    vocab_size=DEFAULT_MAX_TOKEN_ID,
    sequence_length=DEFAULT_SEQUENCE_SIZE,
    hidden_layers=DEFAULT_HIDDEN_LAYERS,
    embedding_dim=DEFAULT_EMBEDDING_DIM,
    dropout_rate=DEFAULT_DROPOUT_RATE,
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

    model.add(tf.keras.layers.GlobalAveragePooling1D())

    for hidden_layer in hidden_layers:
        model.add(tf.keras.layers.Dropout(rate=dropout_rate))
        model.add(tf.keras.layers.Dense(units=hidden_layer, activation="relu"))

    model.add(tf.keras.layers.Dropout(rate=0.5))

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
    if not isinstance(history, pd.DataFrame):
        history = pd.DataFrame(history.history)
    plot_loss_history(history)
    plot_accuracy_history(history)

    final_training_accuracy, final_validation_accuracy = get_final_accuracy(history)

    print(f"Final training accuracy: {final_training_accuracy:.4f}")
    print(f"Final validation accuracy: {final_validation_accuracy:.4f}")
    print(f"Model parameter count: {get_total_parameters(model):,}")


def plot_loss_history(history):
    if not isinstance(history, pd.DataFrame):
        history = pd.DataFrame(history.history)
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.xticks(range(0, len(history["loss"] + 1)))
    plt.plot(history["loss"], label="training", marker="o")
    plt.plot(history["val_loss"], label="validation", marker="o")
    plt.legend()
    plt.show()


def plot_accuracy_history(history):
    if not isinstance(history, pd.DataFrame):
        history = pd.DataFrame(history.history)
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.xticks(range(0, len(history["accuracy"] + 1)))
    plt.plot(history["accuracy"], label="training", marker="o")
    plt.plot(history["val_accuracy"], label="validation", marker="o")
    plt.legend()
    plt.show()


def get_final_accuracy(history):
    if not isinstance(history, pd.DataFrame):
        history = pd.DataFrame(history.history)
    return history["accuracy"].values[-1], history["val_accuracy"].values[-1]


def get_total_parameters(model):
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    summarystring = "\n".join(stringlist)
    total_parameter_string = re.search("Total params: (.*)\n", summarystring).group(1)
    return int(total_parameter_string.replace(",", ""))
