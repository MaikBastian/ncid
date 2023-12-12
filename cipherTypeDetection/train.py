import multiprocessing
from pathlib import Path

import numpy as np
import argparse
import sys
import time
import shutil
import csv
import random
from sklearn.model_selection import train_test_split
import os
import math
import pickle
import functools
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from datetime import datetime
# This environ variable must be set before all tensorflow imports!
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.metrics import SparseTopKCategoricalAccuracy
from tensorflow.keras.optimizers import Adam  # , Adamax
import tensorflow_datasets as tfds
sys.path.append("../")
import cipherTypeDetection.config as config
from cipherImplementations.cipher import OUTPUT_ALPHABET
from cipherTypeDetection.textLine2CipherStatisticsDataset import RotorCiphertextsDatasetParameters, PlaintextPathsDatasetParameters, CipherStatisticsDataset, TrainingBatch
from cipherTypeDetection.miniBatchEarlyStoppingCallback import MiniBatchEarlyStopping
from cipherTypeDetection.transformer import TransformerBlock, TokenAndPositionEmbedding, MultiHeadSelfAttention
from cipherTypeDetection.learningRateSchedulers import TimeBasedDecayLearningRateScheduler, CustomStepDecayLearningRateScheduler
tf.debugging.set_log_device_placement(enabled=False)
# always flush after print as some architectures like RF need very long time before printing anything.
print = functools.partial(print, flush=True)
# for device in tf.config.list_physical_devices('GPU'):
#    tf.config.experimental.set_memory_growth(device, True)

# Used for meta-classifier pipeline
import cipherTypeDetection.eval as cipherEval
from cipherTypeDetection.ensembleModel import EnsembleModel
from cipherTypeDetection.rotorCipherEnsemble import RotorCipherEnsemble
from types import SimpleNamespace
import pandas as pd

architecture = None


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def create_model(extend_model, cipher_types):
    global architecture
    optimizer = Adam(
        learning_rate=config.learning_rate, beta_1=config.beta_1, beta_2=config.beta_2, 
        epsilon=config.epsilon, amsgrad=config.amsgrad)
    # optimizer = Adamax()
    model_ = None

    # sizes for layers
    total_frequencies_size = 0
    for j in range(1, 3):
        total_frequencies_size += math.pow(len(OUTPUT_ALPHABET), j)
    total_frequencies_size = int(total_frequencies_size)

    # total_ny_gram_frequencies_size = int(math.pow(len(OUTPUT_ALPHABET), 2)) * 6

    # old feature length: 1505
    rotor_features = 802
    magic = 5

    input_layer_size = 18 + total_frequencies_size + rotor_features + magic
    output_layer_size = len(cipher_types) + len(config.ROTOR_CIPHER_TYPES)
    hidden_layer_size = int(2 * (input_layer_size / 3) + output_layer_size)

    # logistic regression baseline
    # model_ = tf.keras.Sequential()
    # model_.add(tf.keras.layers.Dense(output_layer_size, input_dim=input_layer_size, activation='softmax', use_bias=True))
    # model_.compile(optimizer='sgd', loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # extend model
    if extend_model is not None:
        # remove the last layer
        model_ = tf.keras.Sequential()
        for layer in extend_model.layers[:-1]:
            model_.add(layer)
        model_.add(tf.keras.layers.Dense(output_layer_size, activation='softmax', name="output"))
        model_.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy",
                       metrics=["accuracy", SparseTopKCategoricalAccuracy(k=3, name="k3_accuracy")])
        return model_

    # FFNN
    if architecture == 'FFNN':
        model_ = tf.keras.Sequential()
        model_.add(tf.keras.layers.Input(shape=(input_layer_size,)))
        for _ in range(config.hidden_layers):
            model_.add(tf.keras.layers.Dense(hidden_layer_size, activation='relu', use_bias=True))
        model_.add(tf.keras.layers.Dense(output_layer_size, activation='softmax'))
        model_.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", 
                       metrics=["accuracy", SparseTopKCategoricalAccuracy(k=3, name="k3_accuracy")])

    # CNN
    if architecture == 'CNN':
        config.FEATURE_ENGINEERING = False
        config.PAD_INPUT = True
        model_ = tf.keras.Sequential()
        model_.add(tf.keras.layers.Conv1D(
                   filters=config.filters, kernel_size=config.kernel_size, 
                   input_shape=(args.max_train_len, 1), activation='relu'))
        for _ in range(config.layers - 1):
            model_.add(tf.keras.layers.Conv1D(filters=config.filters, kernel_size=config.kernel_size, activation='relu'))
        # model_.add(tf.keras.layers.Dropout(0.2))
        model_.add(tf.keras.layers.MaxPooling1D(pool_size=2))
        model_.add(tf.keras.layers.Flatten())
        model_.add(tf.keras.layers.Dense(output_layer_size, activation='softmax'))
        model_.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", 
                       metrics=["accuracy", SparseTopKCategoricalAccuracy(k=3, name="k3_accuracy")])

    # LSTM
    if architecture == 'LSTM':
        config.FEATURE_ENGINEERING = False
        config.PAD_INPUT = True
        model_ = tf.keras.Sequential()
        model_.add(tf.keras.layers.Embedding(56, 64, input_length=args.max_train_len))
        # model_.add(tf.keras.layers.Dropout(0.2))
        model_.add(tf.keras.layers.LSTM(config.lstm_units))
        # model_.add(tf.keras.layers.Dropout(0.2))
        model_.add(tf.keras.layers.Flatten())
        model_.add(tf.keras.layers.Dense(output_layer_size, activation='softmax'))
        model_.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", 
                       metrics=["accuracy", SparseTopKCategoricalAccuracy(k=3, name="k3_accuracy")])

    # Decision Tree
    if architecture == 'DT':
        model_ = DecisionTreeClassifier(criterion=config.criterion, ccp_alpha=config.ccp_alpha)

    # Naive Bayes
    if architecture == 'NB':
        model_ = MultinomialNB(alpha=config.alpha, fit_prior=config.fit_prior)

    # Random Forest
    if architecture == 'RF':
        model_ = RandomForestClassifier(n_estimators=config.n_estimators, criterion=config.criterion, 
                                        bootstrap=config.bootstrap, n_jobs=30,
                                        max_features=config.max_features, max_depth=30, 
                                        min_samples_split=config.min_samples_split,
                                        min_samples_leaf=config.min_samples_leaf)

    # Extra Trees
    if architecture == 'ET':
        model_ = ExtraTreesClassifier(n_estimators=config.n_estimators, criterion=config.criterion, 
                                      bootstrap=config.bootstrap, n_jobs=30,
                                      max_features=config.max_features, max_depth=30, 
                                      min_samples_split=config.min_samples_split,
                                      min_samples_leaf=config.min_samples_leaf)

    # Transformer
    if architecture == "Transformer":
        config.FEATURE_ENGINEERING = False
        config.PAD_INPUT = True
        vocab_size = config.vocab_size
        maxlen = args.max_train_len
        embed_dim = config.embed_dim  # Embedding size for each token
        num_heads = config.num_heads  # Number of attention heads
        ff_dim = config.ff_dim  # Hidden layer size in feed forward network inside transformer

        inputs = tf.keras.layers.Input(shape=(maxlen,))
        embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
        x = embedding_layer(inputs)
        transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
        x = transformer_block(x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        outputs = tf.keras.layers.Dense(output_layer_size, activation="softmax")(x)

        model_ = tf.keras.Model(inputs=inputs, outputs=outputs)
        model_.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", 
                       metrics=["accuracy", SparseTopKCategoricalAccuracy(k=3, name="k3_accuracy")])

    # FFNN, NB
    if architecture == "[FFNN,NB]":
        model_ffnn = tf.keras.Sequential()
        model_ffnn.add(tf.keras.layers.Input(shape=(input_layer_size,)))
        for _ in range(config.hidden_layers):
            model_ffnn.add(tf.keras.layers.Dense(hidden_layer_size, activation='relu', use_bias=True))
        model_ffnn.add(tf.keras.layers.Dense(output_layer_size, activation='softmax'))
        model_ffnn.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy",
                           metrics=["accuracy", SparseTopKCategoricalAccuracy(k=3, name="k3_accuracy")])
        model_nb = MultinomialNB(alpha=config.alpha, fit_prior=config.fit_prior)
        return [model_ffnn, model_nb]
    
    if architecture == "SVM-Rotor":
        model_ = SVC(probability=True, C=1, gamma=0.0001, kernel="rbf")

    if architecture == "RF-Rotor":
        print("UNIMPLEMENTED ARCHITECTURE!")
        sys.exit(1)

    if architecture == "kNN-Rotor":
        print("UNIMPLEMENTED ARCHITECTURE!")
        sys.exit(1)

    if architecture == "LSTM-Rotor":
        print("UNIMPLEMENTED ARCHITECTURE!")
        sys.exit(1)

    if architecture == "MLP-Rotor":
        print("UNIMPLEMENTED ARCHITECTURE!")
        sys.exit(1)

    if architecture == "CNN-Rotor":
        print("UNIMPLEMENTED ARCHITECTURE!")
        sys.exit(1)

    if architecture == "ELM-Rotor":
        print("UNIMPLEMENTED ARCHITECTURE!")
        sys.exit(1)

    return model_

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='CANN Ciphertype Detection Neuronal Network Training Script', 
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size for training.')
    parser.add_argument('--train_dataset_size', default=16000, type=int,
                        help='Dataset size per fit. This argument should be dividable \n'
                             'by the amount of --ciphers.')
    parser.add_argument('--dataset_workers', default=1, type=int,
                        help='The number of parallel workers for reading the \ninput files.')
    parser.add_argument('--epochs', default=1, type=int,
                        help='Defines how many times the same data is used to fit the model.')
    parser.add_argument('--input_directory', default='../data/gutenberg_en', type=str,
                        help='Input directory of the plaintexts.')
    parser.add_argument('--download_dataset', default=True, type=str2bool,
                        help='Download the dataset automatically.')
    parser.add_argument('--save_directory', default='../data/models/',
                        help='Directory for saving generated models. \n'
                             'When interrupting, the current model is \n'
                             'saved as interrupted_...')
    parser.add_argument('--model_name', default='m.h5', type=str,
                        help='Name of the output model file. The file must \nhave the .h5 extension.')
    parser.add_argument('--ciphers', default='aca', type=str,
                        help='A comma seperated list of the ciphers to be created.\n'
                             'Be careful to not use spaces or use \' to define the string.\n'
                             'Possible values are:\n'
                             '- mtc3 (contains the ciphers Monoalphabetic Substitution, Vigenere,\n'
                             '        Columnar Transposition, Plaifair and Hill)\n'
                             '- aca (contains all currently implemented ciphers from \n'
                             '       https://www.cryptogram.org/resource-area/cipher-types/)\n'
                             '- all aca ciphers in lower case'
                             '- simple_substitution\n'
                             '- vigenere\n'
                             '- columnar_transposition\n'
                             '- playfair\n'
                             '- hill\n')
    parser.add_argument('--keep_unknown_symbols', default=False, type=str2bool,
                        help='Keep unknown symbols in the plaintexts. Known \n'
                             'symbols are defined in the alphabet of the cipher.')
    parser.add_argument('--max_iter', default=1000000, type=int,
                        help='the maximal number of iterations before stopping training.')
    parser.add_argument('--min_train_len', default=50, type=int,
                        help='The minimum length of a plaintext to be encrypted in training. \n'
                             'If this argument is set to -1 no lower limit is used.')
    parser.add_argument('--min_test_len', default=50, type=int,
                        help='The minimum length of a plaintext to be encrypted in testing. \n'
                             'If this argument is set to -1 no lower limit is used.')
    parser.add_argument('--max_train_len', default=-1, type=int,
                        help='The maximum length of a plaintext to be encrypted in training. \n'
                             'If this argument is set to -1 no upper limit is used.')
    parser.add_argument('--max_test_len', default=-1, type=int,
                        help='The maximum length of a plaintext to be encrypted in testing. \n'
                             'If this argument is set to -1 no upper limit is used.')
    parser.add_argument('--architecture', default='FFNN', type=str, 
                        choices=['FFNN', 'CNN', 'LSTM', 'DT', 'NB', 'RF', 'ET', 'Transformer',
                                 '[FFNN,NB]'],
                        help='The architecture to be used for training. \n'
                             'Possible values are:\n'
                             '- FFNN\n'
                             '- CNN\n'
                             '- LSTM\n'
                             '- DT\n'
                             '- NB\n'
                             '- RF\n'
                             '- ET\n'
                             '- Transformer\n'
                             '- [FFNN,NB]')
    parser.add_argument('--extend_model', default=None, type=str,
                        help='Load a trained model from a file and use it as basis for the new training.')

    return parser.parse_args()
    
def should_download_datasets(args):
    """Determines if the plaintext datasets should be loaded"""
    return (args.download_dataset and 
            not os.path.exists(args.input_directory) and 
            args.input_directory == os.path.abspath('../data/gutenberg_en'))

def download_datasets(args):
    """Downloads plaintexts and saves them in the input_directory"""
    print("Downloading Datsets...")
    tfds.download.add_checksums_dir('../data/checksums/')
    download_manager = tfds.download.download_manager.DownloadManager(download_dir='../data/', 
                                                       extract_dir=args.input_directory)
    data_url = ('https://drive.google.com/uc?id=1bF5sSVjxTxa3DB-P5wxn87nxWndRhK_V&export=download' +
        '&confirm=t&uuid=afbc362d-9d52-472a-832b-c2af331a8d5b')
    try:
        download_manager.download_and_extract(data_url)
    except Exception as e:
        print("Download of datasets failed. If this issues persists, try downloading the dataset yourself "
              "from: https://drive.google.com/file/d/1bF5sSVjxTxa3DB-P5wxn87nxWndRhK_V/view."
              "(For more information see the README.md of this project.)")
        print("Underlying error:")
        print(e)
        sys.exit(1)

    path = os.path.join(args.input_directory, 
                        'ZIP.ucid_1bF5sSVjxTx-P5wxn87nxWn_V_export_downloadR9Cwhunev5CvJ-ic__'
                        'HawxhTtGOlSdcCrro4fxfEI8A', 
                        os.path.basename(args.input_directory))
    dir_name = os.listdir(path)
    for name in dir_name:
        p = Path(os.path.join(path, name))
        parent_dir = p.parents[2]
        p.rename(parent_dir / p.name)
    os.rmdir(path)
    os.rmdir(os.path.dirname(path))
    print("Datasets Downloaded.")

def load_datasets_from_disk(args, cipher_types):
    """Loads datasets from the file system. 
    In case of the ACA ciphers the datasets are plaintext files that need to be
    encrypted before the features can be extracted.
    In case of the rotor ciphers there are already encrypted ciphertext files 
    that can directly be used to extract the features.
    """
    print("Loading Datasets...")
    plaintext_files = []
    dir_name = os.listdir(args.input_directory)
    max = 400
    for name in dir_name:
        path = os.path.join(args.input_directory, name)
        if os.path.isfile(path):
            max -= 1
            if max == 0:
                break
            plaintext_files.append(path)

    def validate_ciphertext_path(ciphertext_path, cipher_types):
        file_name = Path(ciphertext_path).stem.lower()
        if not file_name in cipher_types:
            raise Exception(f"Filename must equal one of the expected cipher types. Expected cipher types are: {cipher_types}. Current filename is '{file_name}'.")
    
    # TODO: Do not hard code folder!?
    rotor_cipher_dir = Path(args.input_directory).parent / "rotor_ciphertexts"
    rotor_ciphertexts = []
    dir_name = os.listdir(rotor_cipher_dir)
    max = 10000
    for name in dir_name:
        path = os.path.join(rotor_cipher_dir, name)
        validate_ciphertext_path(path, config.ROTOR_CIPHER_TYPES)
        if os.path.isfile(path):
            with open(path, "r") as f:
                label = Path(path).stem.lower()
                lines = f.readlines()
                for line in lines:
                    max -= 1
                    if max < 0:
                        break
                    rotor_ciphertexts.append((line.rstrip(), label))
                    
    train_plaintexts, test_plaintexts = train_test_split(plaintext_files, test_size=0.05, 
                                                         random_state=42, shuffle=True)
    train_rotor_ciphertexts, test_rotor_ciphertexts = train_test_split(rotor_ciphertexts, test_size=0.05, 
                                                                       random_state=42, shuffle=True)

    train_plaintext_parameters = PlaintextPathsDatasetParameters(train_plaintexts, cipher_types, args.train_dataset_size, 
                                                args.min_train_len, args.max_train_len,
                                                args.keep_unknown_symbols, args.dataset_workers)
    test_plaintext_parameters = PlaintextPathsDatasetParameters(test_plaintexts, cipher_types, args.train_dataset_size,     
                                               args.min_test_len, args.max_test_len,
                                               args.keep_unknown_symbols, args.dataset_workers)
    
    # Calculate a different batch size for rotor ciphertexts. There are fewer samples
    # that should not be exhausted too quickly, but should occur often enough that the
    # ML architecture recognizes them.
    rotor_train_dataset_size = (args.train_dataset_size // len(config.CIPHER_TYPES)) * 4

    # TODO: Move into calling function! (ROTOR_CIPHER_TYPES)
    train_rotor_ciphertexts_parameters = RotorCiphertextsDatasetParameters(train_rotor_ciphertexts, 
                                                            config.ROTOR_CIPHER_TYPES, 
                                                            rotor_train_dataset_size,
                                                            args.dataset_workers, 
                                                            args.min_train_len, 
                                                            args.max_train_len)
    test_rotor_ciphertexts_parameters = RotorCiphertextsDatasetParameters(test_rotor_ciphertexts, 
                                                            config.ROTOR_CIPHER_TYPES,
                                                            rotor_train_dataset_size,
                                                            args.dataset_workers, 
                                                            args.min_test_len, 
                                                            args.max_test_len)
    
    train_ds = CipherStatisticsDataset(train_plaintext_parameters, train_rotor_ciphertexts_parameters)
    test_ds = CipherStatisticsDataset(test_plaintext_parameters, test_rotor_ciphertexts_parameters)

    if args.train_dataset_size % train_ds.key_lengths_count != 0:
        print("WARNING: the --train_dataset_size parameter must be dividable by the amount of --ciphers  and the length configured "
              "KEY_LENGTHS in config.py. The current key_lengths_count is %d" % 
                  train_ds.key_lengths_count, file=sys.stderr)
    print("Datasets loaded.\n")
    return train_ds, test_ds
    
# TODO: Better name!
def create_model_for_hardware_config(extend_model, cipher_types):
    """Creates models depending on the GPU count and on extend_model"""
    print('Creating model...')

    gpu_count = (len(tf.config.list_physical_devices('GPU')) +
        len(tf.config.list_physical_devices('XLA_GPU')))
    if gpu_count > 1:
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            if extend_model is not None:
                extend_model = tf.keras.models.load_model(extend_model, compile=False)
            model = create_model(extend_model, cipher_types)
        if architecture in ("FFNN", "CNN", "LSTM", "Transformer") and extend_model is None:
            model.summary()
    else:
        if extend_model is not None:
            extend_model = tf.keras.models.load_model(extend_model, compile=False)
        model = create_model(extend_model, cipher_types)
        if architecture in ("FFNN", "CNN", "LSTM", "Transformer") and extend_model is None:
            model.summary()

    print('Model created.\n')
    return model
    
def train_model(model, args, train_ds):
    """Trains the model"""

    def sample_weights(class_labels):
        """Rotor ciphers will get a weight of 2, aca ciphers the default 1. Since the training
        dataset for rotor ciphers has less entries, this instructs the ML algorithms to remember
        them more."""
        weights = np.ones(len(class_labels))
        first_rotor_cipher_class = len(config.CIPHER_TYPES) - len(config.ROTOR_CIPHER_TYPES)
        weights[np.where(class_labels.numpy() > first_rotor_cipher_class)] = 2
        return weights

    print('Training model...')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='../data/logs', update_freq='epoch')
    early_stopping_callback = MiniBatchEarlyStopping(
        min_delta=1e-5, patience=250, monitor='accuracy', mode='max', restore_best_weights=True)
    # time_based_decay_lrate_callback = TimeBasedDecayLearningRateScheduler(args.train_dataset_size)
    custom_step_decay_lrate_callback = CustomStepDecayLearningRateScheduler(early_stopping_callback)
    start_time = time.time()
    train_iter = 0
    train_epoch = 0
    val_data = None
    val_labels = None
    training_batches = None
    combined_batch = TrainingBatch("mixed", [], [])
    classes = list(range(len(config.CIPHER_TYPES)))
    while train_ds.iteration < args.max_iter:
        training_batches = next(train_ds)

        # DTs, RFs and ETs only support one fit call: Sample all batches into one large batch.
        if architecture in ("DT", "RF", "ET"):
            for training_batch in training_batches:
                combined_batch.extend(training_batch)
            if train_ds.iteration < args.max_iter:
                print("Loaded %d ciphertexts." % train_ds.iteration)
                continue
            print("Loaded %d ciphertexts." % train_ds.iteration)
            training_batches = [combined_batch]

        for index, training_batch in enumerate(training_batches):
            statistics, labels = training_batch.tuple()
            train_iter = train_ds.iteration - len(training_batch) * (len(training_batches) - index - 1)

            # Create small validation dataset on first iteration
            if index == 0:
                statistics, val_data, labels, val_labels = train_test_split(statistics.numpy(), 
                                                                            labels.numpy(), 
                                                                            test_size=0.3)
                statistics = tf.convert_to_tensor(statistics)
                val_data = tf.convert_to_tensor(val_data)
                labels = tf.convert_to_tensor(labels)
                val_labels = tf.convert_to_tensor(val_labels)
            train_iter -= len(training_batch) * 0.3

            # Decision Tree training
            if architecture in ("DT", "RF", "ET"):
                train_iter = len(labels) * 0.7
                print("Start training the decision tree.")
                history = model.fit(statistics, labels, sample_weight=sample_weights(labels))
                if architecture == "DT":
                    plt.gcf().set_size_inches(25, 25 / math.sqrt(2))
                    print("Plotting tree.")
                    plot_tree(model, max_depth=3, fontsize=6, filled=True)
                    plt.savefig(args.model_name.split('.')[0] + '_decision_tree.svg', 
                                dpi=200, bbox_inches='tight', pad_inches=0.1)

            # Naive Bayes training
            elif architecture == "NB":
                history = model.partial_fit(statistics, labels, classes=classes,
                                            sample_weight=sample_weights(labels))

            # FFNN, NB
            elif architecture == "[FFNN,NB]":
                history = model[0].fit(statistics, labels, batch_size=args.batch_size, 
                                       validation_data=(val_data, val_labels), epochs=args.epochs,
                                       sample_weight=sample_weights(labels),
                                       callbacks=[early_stopping_callback, tensorboard_callback,    
                                                  custom_step_decay_lrate_callback])
                # time_based_decay_lrate_callback.iteration = train_iter
                history = model[1].partial_fit(statistics, labels, classes=classes, 
                                               sample_weight=sample_weights(labels))

            else:
                history = model.fit(statistics, labels, batch_size=args.batch_size, 
                                    validation_data=(val_data, val_labels), epochs=args.epochs,
                                    sample_weight=sample_weights(labels),
                                    callbacks=[early_stopping_callback, tensorboard_callback, 
                                               custom_step_decay_lrate_callback])
                # time_based_decay_lrate_callback.iteration = train_iter

            # print for Decision Tree, Naive Bayes and Random Forests
            if architecture in ("DT", "NB", "RF", "ET"):
                val_score = model.score(val_data, val_labels)
                train_score = model.score(statistics, labels)
                print("train accuracy: %f, validation accuracy: %f" % (train_score, val_score))

            if architecture == "[FFNN,NB]":
                val_score = model[1].score(val_data, val_labels)
                train_score = model[1].score(statistics, labels)
                print("train accuracy: %f, validation accuracy: %f" % (train_score, val_score))

            if train_ds.epoch > 0:
                train_epoch = train_ds.iteration // ((train_iter + train_ds.batch_size * train_ds.dataset_workers) // train_ds.epoch)
                
            print("Epoch: %d, Iteration: %d" % (train_epoch, train_iter))
            if train_iter >= args.max_iter or early_stopping_callback.stop_training:
                break
            
        if train_ds.iteration >= args.max_iter or early_stopping_callback.stop_training:
            train_ds.stop_outstanding_tasks()
            break

    elapsed_training_time = datetime.fromtimestamp(time.time()) - datetime.fromtimestamp(start_time)
    training_stats = 'Finished training in %d days %d hours %d minutes %d seconds with %d iterations and %d epochs.\n' % (
        elapsed_training_time.days, elapsed_training_time.seconds // 3600, 
        (elapsed_training_time.seconds // 60) % 60,
        elapsed_training_time.seconds % 60, train_iter, train_epoch)
    print(training_stats)
    return early_stopping_callback, train_iter, training_stats
        
def save_model(model, args):
    """Saves the model"""
    print('Saving model...')
    if not os.path.exists(args.save_directory):
        os.mkdir(args.save_directory)
    if args.model_name == 'm.h5':
        i = 1
        while os.path.exists(os.path.join(args.save_directory, args.model_name.split('.')[0] + str(i) + '.h5')):
            i += 1
        model_name = args.model_name.split('.')[0] + str(i) + '.h5'
    else:
        model_name = args.model_name
    model_path = os.path.join(args.save_directory, model_name)
    if architecture in ("FFNN", "CNN", "LSTM", "Transformer"):
        model.save(model_path)
    elif architecture in ("DT", "NB", "RF", "ET"):
        with open(model_path, "wb") as f:
            # this gets very large
            pickle.dump(model, f)
    elif architecture == "[FFNN,NB]":
        model[0].save('../data/models/' + model_path.split('.')[0] + "_ffnn.h5")
        with open('../data/models/' + model_path.split('.')[0] + "_nb.h5", "wb") as f:
            # this gets very large
            pickle.dump(model[1], f)
    with open('../data/' + model_path.split('.')[0] + '_parameters.txt', 'w') as f:
        for arg in vars(args):
            f.write("{:23s}= {:s}\n".format(arg, str(getattr(args, arg))))
    if architecture in ("FFNN", "CNN", "LSTM", "Transformer"):
        logs_destination = '../data/' + model_name.split('.')[0] + '_tensorboard_logs'
        try:
            if os.path.exists(logs_destination):
                shutil.rmtree(logs_destination)
            shutil.move('../data/logs', logs_destination)
        except Exception:
            print(f"Could not remove logs of previous run. Move of current logs "
                  f"from '../data/logs' to '{logs_destination}' failed.")
    print('Model saved.\n')
    
def predict_test_data(test_ds, model, args, early_stopping_callback, train_iter):
    """Testing the predictions of the model"""
    print('Predicting test data...\n')
    start_time = time.time()
    correct = [0]*len(config.CIPHER_TYPES)
    total = [0]*len(config.CIPHER_TYPES)
    correct_all = 0
    total_len_prediction = 0
    incorrect = []
    for i in range(len(config.CIPHER_TYPES)):
        incorrect += [[0]*len(config.CIPHER_TYPES)]

    prediction_dataset_factor = 10
    if early_stopping_callback.stop_training:
        while test_ds.dataset_workers * test_ds.batch_size > train_iter / prediction_dataset_factor and prediction_dataset_factor > 1:
            prediction_dataset_factor -= 1
        args.max_iter = int(train_iter / prediction_dataset_factor)
    else:
        while test_ds.dataset_workers * test_ds.batch_size > args.max_iter / prediction_dataset_factor:
            prediction_dataset_factor -= 1
        args.max_iter /= prediction_dataset_factor
    cntr = 0
    test_iter = 0
    test_epoch = 0

    while test_ds.iteration < args.max_iter:
        training_batches = next(test_ds)

        for training_batch in training_batches:
            statistics, labels = training_batch.tuple()
            
            # Decision Tree, Naive Bayes prediction
            if architecture in ("DT", "NB", "RF", "ET"):
                prediction = model.predict_proba(statistics)
            elif architecture == "[FFNN,NB]":
                prediction = model[0].predict(statistics, batch_size=args.batch_size, verbose=1)
            else:
                prediction = model.predict(statistics, batch_size=args.batch_size, verbose=1)
            for i in range(len(prediction)):
                if labels[i] == np.argmax(prediction[i]):
                    correct_all += 1
                    correct[labels[i]] += 1
                else:
                    incorrect[labels[i]][np.argmax(prediction[i])] += 1
                total[labels[i]] += 1
            total_len_prediction += len(prediction)
            cntr += 1
            test_iter = args.train_dataset_size * cntr
            test_epoch = test_ds.epoch
            if test_epoch > 0:
                test_epoch = test_iter // ((test_ds.iteration + test_ds.batch_size * test_ds.dataset_workers) // test_ds.epoch)
            print("Prediction Epoch: %d, Iteration: %d / %d" % (test_epoch, test_iter, args.max_iter))
            if test_iter >= args.max_iter:
                break
        if test_ds.iteration >= args.max_iter:
            break

    elapsed_prediction_time = datetime.fromtimestamp(time.time()) - datetime.fromtimestamp(start_time)

    if total_len_prediction > args.train_dataset_size:
        total_len_prediction -= total_len_prediction % args.train_dataset_size
    print('\ntest data predicted: %d ciphertexts' % total_len_prediction)
    for i in range(0, len(total)):
        if total[i] == 0:
            continue
        print('%s correct: %d/%d = %f' % (config.CIPHER_TYPES[i], correct[i], total[i], correct[i] / total[i]))
    if total_len_prediction == 0:
        t = 'N/A'
    else:
        t = str(correct_all / total_len_prediction)
    print('Total: %s\n' % t)

    prediction_stats = 'Prediction time: %d days %d hours %d minutes %d seconds with %d iterations and %d epochs.' % (
        elapsed_prediction_time.days, elapsed_prediction_time.seconds // 3600, 
        (elapsed_prediction_time.seconds // 60) % 60,
        elapsed_prediction_time.seconds % 60, test_iter, test_epoch)
    
    return prediction_stats, incorrect

def expand_cipher_groups(cipher_types):
    """Turn cipher group identifiers (ACA, MTC3) into a list of their ciphers"""
    expanded = cipher_types
    if config.MTC3 in expanded:
        del expanded[expanded.index(config.MTC3)]
        for i in range(5):
            expanded.append(config.CIPHER_TYPES[i])
    elif config.ACA in expanded:
        del expanded[expanded.index(config.ACA)]
        for i in range(56):
            expanded.append(config.CIPHER_TYPES[i])
    return expanded

def aca_pipeline(cipher_types):
    extend_model = args.extend_model
    if extend_model is not None:
        if architecture not in ('FFNN', 'CNN', 'LSTM'):
            print('ERROR: Models with the architecture %s can not be extended!' % architecture,
                  file=sys.stderr)
            sys.exit(1)
        if len(os.path.splitext(extend_model)) != 2 or os.path.splitext(extend_model)[1] != '.h5':
            print('ERROR: The extended model name must have the ".h5" extension!', file=sys.stderr)
            sys.exit(1)

    cipher_types = expand_cipher_groups(cipher_types)
    # config.CIPHER_TYPES = cipher_types

    if args.train_dataset_size * args.dataset_workers > args.max_iter:
        print("ERROR: --train_dataset_size * --dataset_workers must not be bigger than --max_iter. "
              "In this case it was %d > %d" % 
                  (args.train_dataset_size * args.dataset_workers, args.max_iter), 
              file=sys.stderr)
        sys.exit(1)

    if should_download_datasets(args):
        download_datasets(args)

    train_ds, test_ds = load_datasets_from_disk(args, cipher_types)
    
    # print("DONE!")
    # sys.exit(0)

    # ACA ciphers
    model = create_model_for_hardware_config(extend_model, cipher_types)
    early_stopping_callback, train_iter, training_stats = train_model(model, args, train_ds)
    save_model(model, args)
    prediction_stats, incorrect = predict_test_data(test_ds, model, args, early_stopping_callback, train_iter)
    
    print(training_stats)
    print(prediction_stats)

    print("Incorrect prediction counts: %s" % incorrect)

def rotor_pipeline():
    # create rotor models

    # save rotor models

    pass

# class PredictionResult:
#     def __init__(self, prediction, cipher):
#         self.prediction = prediction
#         self.cipher = cipher

def meta_classifier_pipeline():
    # TODO: ensure all architectures are trained!


    # breakpoint()

    # load aca classifiers
    model_path = Path("../data/models")
    try:
        transformer = tf.keras.models.load_model(
                os.path.join(model_path, "t96_transformer_final_100.h5"),
                custom_objects={
                    'TokenAndPositionEmbedding': TokenAndPositionEmbedding,
                    'MultiHeadSelfAttention': MultiHeadSelfAttention,
                    'TransformerBlock': TransformerBlock})
        ffnn = tf.keras.models.load_model(
                os.path.join(model_path, "t128_ffnn_final_100.h5"))
        lstm = tf.keras.models.load_model(
                os.path.join(model_path, "t129_lstm_final_100.h5"))
        optimizer = Adam(
            learning_rate=config.learning_rate,
            beta_1=config.beta_1,
            beta_2=config.beta_2,
            epsilon=config.epsilon,
            amsgrad=config.amsgrad)
        for model in [transformer, ffnn, lstm]:
            model.compile(
                optimizer=optimizer,
                loss="sparse_categorical_crossentropy",
                metrics=[
                    "accuracy",
                    SparseTopKCategoricalAccuracy(k=3, name="k3_accuracy")])
        with open(os.path.join(model_path, "t99_rf_final_100.h5"), "rb") as f:
            rfc = pickle.load(f)
        with open(os.path.join(model_path, "t128_nb_final_100.h5"), "rb") as f:
            nbc = pickle.load(f)
        aca_classifiers = [transformer, ffnn, lstm, rfc, nbc]
    except OSError as error:
        print("ERROR: \n"
              "At least one of the expected models for recognition of ACA ciphers for "
              "training of the meta classifier is missing. \n"
              "Expected filenames of models are: 't96_transformer_final_100.h5', "
              "'t128_ffnn_final_100.h5', 't129_lstm_final_100.h5', 't99_rf_final_100.h5' "
              "and 't128_nb_final_100.h5'. These models should be saved in 'data/models'.")
        sys.exit(1)

    # create rotor classifiers
    try:
        with open(os.path.join(model_path, "svm-combined"), "rb") as f:
            svm = pickle.load(f)
        with open(os.path.join(model_path, "knn-combined"), "rb") as f:
            knn = pickle.load(f)
        # with open(os.path.join(model_path, "rf-combined"), "rb") as f:
        #     rf2 = pickle.load(f)
        # with open(os.path.join(model_path, "mlp-combined"), "rb") as f:
        #     mlp = pickle.load(f)
        # TODO: Also the rest!
        rotor_classifiers = [svm, knn]
    except FileNotFoundError as error:
        # TODO: Extend error message with model names!
        print("ERROR: \n"
              "At least one of the expected models for recognition of Rotor ciphers for "
              "training of the meta classifier is missing. \n"
              "Expected filenames of models are: 'svm-combined' and "
              "'knn-combined'. These models should be saved in 'data/models'.")
        sys.exit(1)

    try:
        with open(os.path.join(model_path, "scaler"), "rb") as f:
            scaler = pickle.load(f)
    except FileNotFoundError as error:
        print("ERROR: \n. "
              "Could not load the scikit-learn StandardScaler. It is expected at the "
              "path 'data/models/scaler")
        sys.exit(1)


    # TODO: generate ciphertexts for aca architectures

    def load_ciphertext_lines(dir, *, upto_length):
        """Loads ciphertexts (with file ending .txt) in directory dir and converts them to
        a list of lines of max length upto_length. The resulting lines are randomized.
        upto_length should be at least 1000.
        """
        upto_length = max(10, upto_length) # TODO: Back to 1000

        files = [x for x in dir.iterdir() if x.is_file() and x.suffix == ".txt"]
        result = []

        # use some upper limites to limit memory usage
        max_files = math.floor(upto_length / 10)
        max_lines = 1000

        random_files = random.sample(files, max_files) if len(files) > max_files else files
        for file in random_files:
            with open(file, "r") as f:
                lines = f.readlines()
                random_lines = random.sample(lines, max_lines) if len(lines) > max_lines else lines
                for line in random_lines:
                    processed_line = line.lower() # TODO: Also remove invalid chars, etc.!
                    result.append(processed_line)

        return random.sample(result, upto_length) if len(result) > upto_length else result
    
    aca_cipher_types = [config.CIPHER_TYPES[cipher_index]
                        for cipher_index in range(56)]
    aca_cipher_indices = [config.CIPHER_TYPES.index(cipher_type) 
                          for cipher_type in aca_cipher_types]
    rotor_cipher_types = ["Enigma", "M209", "Purple", "Sigaba", "Typex"]
    all_cipher_types = aca_cipher_types + rotor_cipher_types
    
    def predict_with_aca_architectures(ciphertext_lines):
        cipherEval.architecture = "Ensemble"
        aca_ensemble = EnsembleModel(aca_classifiers,
                            ["Transformer", "FFNN", "LSTM", "RF", "NB"],
                            "weighted",
                            aca_cipher_indices)
        
        aca_architecture_predictions = []
        for index, value in enumerate(ciphertext_lines):
            line, cipher_type = value[0], value[1]
            prediction = cipherEval.predict_single_line(SimpleNamespace(
                ciphertext=line,
                # todo: needs fileupload first (either set ciphertext OR file, never both)
                file=None,
                ciphers=aca_cipher_types,
                batch_size=128,
                verbose=False
            ), aca_ensemble)
            prediction = prediction
            for cipher in rotor_cipher_types:
                prediction[cipher] = 0
            aca_architecture_predictions.append([prediction, cipher_type])
            if index % 100 == 0:
                print(f"Predicted {index} / {len(ciphertext_lines)} ciphers with the ACA architectures.")
        
        return aca_architecture_predictions
    
    def predict_with_rotor_architectures(ciphertext_lines):
        rotor_architecture_predictions = []

        rotor_ensemble = RotorCipherEnsemble(rotor_classifiers, scaler)
        for value in ciphertext_lines:
            line, cipher_type = value[0], value[1]
            prediction = rotor_ensemble.predict_single_line(line.upper())
            for cipher in aca_cipher_types:
                prediction[cipher] = 0
            rotor_architecture_predictions.append([prediction, cipher_type])

        return rotor_architecture_predictions
    
    def write_predictions_to_disk(predictions, file_path):
        with open(file_path, "w") as f:
            writer = csv.writer(f, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            keys = ["actual_cipher"] + list(predictions[0][0].keys())
            writer.writerow(keys)
            for prediction in predictions:
                writer.writerow([prediction[1]] + list(prediction[0].values()))

    # load ciphertexts
    ciphertext_dir = Path("../encrypted_samples")
    aca_ciphertext_lines = load_ciphertext_lines(ciphertext_dir / "aca-ciphertexts", upto_length=5000) # 5000!
    rotor_ciphertext_lines = load_ciphertext_lines(ciphertext_dir / "rotor-ciphertexts", upto_length=5000)

    # combine both ciphertext types together with labels for each type
    aca_df = pd.DataFrame(aca_ciphertext_lines)
    aca_df["cipher_type"] = "ACA"
    rotor_df = pd.DataFrame(rotor_ciphertext_lines)
    rotor_df["cipher_type"] = "Rotor"
    ciphertext_lines = pd.concat([aca_df, rotor_df]).to_numpy() # TODO: Remove limit!

    # get predictions from classifiers for all ciphertexts
    aca_architecture_predictions = predict_with_aca_architectures(ciphertext_lines)
    # rotor_architecture_predictions = predict_with_rotor_architectures(ciphertext_lines)

    # create and train meta classifier
    aca_architecture_features = list(map(lambda x: features_from_prediction(x[0], all_cipher_types),
                                     aca_architecture_predictions))
    # rotor_architecture_features = list(map(lambda x: features_from_prediction(x[0], all_cipher_types), 
    #                                    rotor_architecture_predictions))

    feature_names = ["percentage_of_probabilities_above_10_percent", 
                     "percentage_of_probabilities_below_2_percent", 
                     "max_probability", "index_of_max_prediction"]
    
    feature_dict = {}
    for feature_name in feature_names:
        feature_dict[feature_name] = [value[feature_name] for value in aca_architecture_features]
        # feature_dict[feature_name] = ([value[feature_name] for value in aca_architecture_features] + 
        #                         [value[feature_name] for value in rotor_architecture_features])
        
    aca_key = 0
    rotor_key = 1

    # feature_dict["classifier"] = ([aca_key] * len(aca_architecture_features) + 
    #                               [rotor_key] * len(rotor_architecture_features))
    
    feature_dict["cipher"] = [aca_key if value[1] == "ACA" else rotor_key 
                               for value in aca_architecture_predictions]
    # feature_dict["cipher"] = ([aca_key if value[1] == "ACA" else rotor_key 
    #                            for value in aca_architecture_predictions] + 
    #                           [aca_key if value[1] == "ACA" else rotor_key 
    #                            for value in rotor_architecture_predictions])
    
    predictions_df = pd.DataFrame(feature_dict)
    # write dataframe to disk
    predictions_df.to_csv(ciphertext_dir / "feature_predictions.csv")
 
    # predictions_df = pd.read_csv(ciphertext_dir / "feature_predictions.csv", index_col=[0])
    # # predictions_df = predictions_df.drop("percentage_of_probabilities_above_10_percent", axis=1)
    # predictions_df = predictions_df.drop("median_probability", axis=1)
    # predictions_df = predictions_df.iloc[:400]

    y = predictions_df["cipher"].to_numpy()
    X = predictions_df.drop("cipher", axis=1).to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True, stratify=y)

    meta_scaler = StandardScaler()

    X_train_scaled = meta_scaler.fit_transform(X_train)
    X_test_scaled = meta_scaler.transform(X_test)

    def train_knn():
        cv_method = StratifiedKFold(n_splits=10, shuffle=True)
        meta_classifier = GridSearchCV(KNeighborsClassifier(), 
                                    param_grid={"n_neighbors": [1, 2, 3, 5, 8, 10, 15], 
                                                "weights": ("distance", "uniform"),
                                                "metric": ('euclidean', 'manhattan')},
                                    cv=cv_method,
                                    scoring="accuracy",
                                    return_train_score=False,
                                    verbose=1)

        meta_classifier.fit(X_train_scaled, y_train)
        score = meta_classifier.score(X_test_scaled, y_test)
        print(f"Score of kNN meta classifier: {score}")
        print(f"Best kNN params: {meta_classifier.best_params_}")
        print(f"Best kNN score: {meta_classifier.best_score_}")

        with open(model_path / "kNN-meta-classifier", "wb") as f:
            pickle.dump(meta_classifier, f)
        with open(model_path / "kNN-meta-classifier-scaler", "wb") as f:
            pickle.dump(meta_scaler, f)

    def train_svm():
        cv_method = StratifiedKFold(n_splits=10, shuffle=True)
        meta_classifier = GridSearchCV(SVC(), # probability=True
                                    param_grid=[
        {"kernel": ["rbf"], "gamma": [1e-3, 1e-4], "C": [1, 10, 100]}
    ],
                                    cv=cv_method,
                                    scoring="accuracy",
                                    refit=True,
                                    verbose=1)

        meta_classifier.fit(X_train_scaled, y_train)
        score = meta_classifier.score(X_test_scaled, y_test)
        print(f"Score of SVM meta classifier: {score}")
        print(f"Best SVM params: {meta_classifier.best_params_}")
        print(f"Best SVM score: {meta_classifier.best_score_}")

        with open(model_path / "svm-meta-classifier", "wb") as f:
            pickle.dump(meta_classifier, f)
        with open(model_path / "svm-meta-classifier-scaler", "wb") as f:
            pickle.dump(meta_scaler, f)
    
    # Fitting 10 folds for each of 28 candidates, totalling 280 fits
    # [Parallel(n_jobs=1)]: Done  49 tasks      | elapsed:    2.3s
    # [Parallel(n_jobs=1)]: Done 199 tasks      | elapsed:   10.5s
    # Score of kNN meta classifier: 0.9544
    # Best kNN params: {'metric': 'manhattan', 'n_neighbors': 15, 'weights': 'uniform'}
    # Best kNN score: 0.952
    # Fitting 10 folds for each of 6 candidates, totalling 60 fits
    # [Parallel(n_jobs=1)]: Done  49 tasks      | elapsed:   29.7s
    # Score of SVM meta classifier: 0.9252
    # Best SVM params: {'C': 100, 'gamma': 0.001, 'kernel': 'rbf'}
    # Best SVM score: 0.9210666666666667
    train_knn()
    train_svm()

def features_from_prediction(prediction, cipher_types):
        probabilities_above_10_percent = 0
        probabilities_below_2_percent = 0

        max_probability = 0
        max_prediction_index = -1

        for cipher, probability in prediction.items():
            if probability > 10:
                probabilities_above_10_percent += 1
            elif probability < 2 and probability > 0:
                probabilities_below_2_percent += 1
            if probability > max_probability:
                max_probability = probability
                max_prediction_index = cipher_types.index(cipher)
            

        percentage_of_probabilities_above_10_percent = probabilities_above_10_percent / len(prediction)
        percentage_of_probabilities_below_2_percent = probabilities_below_2_percent / len(prediction)

        return {"percentage_of_probabilities_above_10_percent": percentage_of_probabilities_above_10_percent, 
                "percentage_of_probabilities_below_2_percent": percentage_of_probabilities_below_2_percent, 
                "max_probability": max_probability, "index_of_max_prediction": max_prediction_index}

if __name__ == "__main__":
    # Don't fork processes to keep memory footprint low. 
    multiprocessing.set_start_method("spawn")

    args = parse_arguments()

    cpu_count = os.cpu_count()
    if cpu_count and cpu_count < args.dataset_workers:
        print("WARNING: More dataset_workers set than CPUs available.")

    for arg in vars(args):
        print("{:23s}= {:s}".format(arg, str(getattr(args, arg))))

    m = os.path.splitext(args.model_name)
    if len(os.path.splitext(args.model_name)) != 2 or os.path.splitext(args.model_name)[1] != '.h5':
        print('ERROR: The model name must have the ".h5" extension!', file=sys.stderr)
        sys.exit(1)

    create_meta_classifier = False
    if create_meta_classifier:
        meta_classifier_pipeline()
    else:
        args.input_directory = os.path.abspath(args.input_directory)
        args.ciphers = args.ciphers.lower()
        architecture = args.architecture
        cipher_types = args.ciphers.split(',')

        if architecture in ("LSTM", "FFNN", "CNN", "RF", "ET", "DT", "NB", "Transformer", "[FFNN,NB]"):
            aca_pipeline(cipher_types)
        elif architecture in ("SVM-Rotor", "kNN-Rotor", "RF-Rotor"):
            rotor_pipeline()
        else:
            print("Unknown architecture")
            sys.exit(1)

    
