import multiprocessing
from pathlib import Path

import numpy as np
import argparse
import sys
import time
import shutil
from sklearn.model_selection import train_test_split
import os
import math
import pickle
import functools
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from datetime import datetime
# This environ variable must be set before all tensorflow imports!
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.metrics import SparseTopKCategoricalAccuracy
from tensorflow.keras.optimizers import Adam  # , Adamax
import tensorflow_datasets as tfds
sys.path.append("../")
from cipherTypeDetection.featureCalculations import calculate_histogram
import cipherTypeDetection.config as config
from cipherImplementations.cipher import OUTPUT_ALPHABET
from cipherTypeDetection.cipherStatisticsDataset import RotorCiphertextsDatasetParameters, PlaintextPathsDatasetParameters, CipherStatisticsDataset, TrainingBatch
from cipherTypeDetection.predictionPerformanceMetrics import PredictionPerformanceMetrics
from cipherTypeDetection.miniBatchEarlyStoppingCallback import MiniBatchEarlyStopping
from cipherTypeDetection.transformer import TransformerBlock, TokenAndPositionEmbedding, MultiHeadSelfAttention
from cipherTypeDetection.learningRateSchedulers import TimeBasedDecayLearningRateScheduler, CustomStepDecayLearningRateScheduler
tf.debugging.set_log_device_placement(enabled=False)
# always flush after print as some architectures like RF need very long time before printing anything.
print = functools.partial(print, flush=True)
# for device in tf.config.list_physical_devices('GPU'):
#    tf.config.experimental.set_memory_growth(device, True)


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def create_model(architecture, extend_model, cipher_types, max_train_len):
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
    total_frequencies_size = 38

    # total_ny_gram_frequencies_size = int(math.pow(len(OUTPUT_ALPHABET), 2)) * 6

    # old feature length: 1505
    rotor_features = 26 + 676 + 500 + 1 + 26 + 26 + 676
    ldi_stats = 0

    input_layer_size = 18 - 2 + total_frequencies_size + rotor_features + ldi_stats
    output_layer_size = len(cipher_types) + len(config.ROTOR_CIPHER_TYPES)
    hidden_layer_size = int(2 * (input_layer_size / 3) + output_layer_size)

    # logistic regression baseline
    # model_ = tf.keras.Sequential()
    # model_.add(tf.keras.layers.Dense(output_layer_size, input_dim=input_layer_size, activation='softmax', use_bias=True))
    # model_.compile(optimizer='sgd', loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # Create a model based on an existing one for further trainings
    if extend_model is not None:
        # remove the last layer
        model = tf.keras.Sequential()
        for layer in extend_model.layers[:-1]:
            model.add(layer)
        model.add(tf.keras.layers.Dense(output_layer_size, activation='softmax', name="output"))
        model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy",
                       metrics=["accuracy", SparseTopKCategoricalAccuracy(k=3, name="k3_accuracy")])
        return model
    
    # Create new model based on architecture
    if architecture == "FFNN":
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(input_layer_size,)))
        for _ in range(config.hidden_layers):
            model.add(tf.keras.layers.Dense(hidden_layer_size, activation='relu', use_bias=True))
        model.add(tf.keras.layers.Dense(output_layer_size, activation='softmax'))
        model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", 
                    metrics=["accuracy", SparseTopKCategoricalAccuracy(k=3, name="k3_accuracy")])
        return model
    
    elif architecture == "CNN":
        config.FEATURE_ENGINEERING = False
        config.PAD_INPUT = True
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Conv1D(
                filters=config.filters, kernel_size=config.kernel_size, 
                input_shape=(max_train_len, 1), activation='relu'))
        for _ in range(config.layers - 1):
            model.add(tf.keras.layers.Conv1D(filters=config.filters, kernel_size=config.kernel_size, activation='relu'))
        # model_.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(output_layer_size, activation='softmax'))
        model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", 
                    metrics=["accuracy", SparseTopKCategoricalAccuracy(k=3, name="k3_accuracy")])
        return model
    
    elif architecture == "LSTM":
        config.FEATURE_ENGINEERING = False
        config.PAD_INPUT = True
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Embedding(56, 64, input_length=max_train_len))
        # model_.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.LSTM(config.lstm_units))
        # model_.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(output_layer_size, activation='softmax'))
        model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", 
                    metrics=["accuracy", SparseTopKCategoricalAccuracy(k=3, name="k3_accuracy")])
        return model
    
    elif architecture == "DT":
        return DecisionTreeClassifier(criterion=config.criterion, ccp_alpha=config.ccp_alpha)
    
    elif architecture == "NB":
        return MultinomialNB(alpha=config.alpha, fit_prior=config.fit_prior)
    
    elif architecture == "RF":
        return RandomForestClassifier(n_estimators=config.n_estimators, criterion=config.criterion, 
                                      bootstrap=config.bootstrap, n_jobs=30,
                                      max_features=config.max_features, max_depth=30, 
                                      min_samples_split=config.min_samples_split,
                                      min_samples_leaf=config.min_samples_leaf)
    
    elif architecture == "ET":
        return ExtraTreesClassifier(n_estimators=config.n_estimators, criterion=config.criterion, 
                                    bootstrap=config.bootstrap, n_jobs=30,
                                    max_features=config.max_features, max_depth=30, 
                                    min_samples_split=config.min_samples_split,
                                    min_samples_leaf=config.min_samples_leaf)
    
    elif architecture == "Transformer":
        config.FEATURE_ENGINEERING = False
        config.PAD_INPUT = True
        vocab_size = config.vocab_size
        maxlen = max_train_len
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

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", 
                        metrics=["accuracy", SparseTopKCategoricalAccuracy(k=3, name="k3_accuracy")])
        return model
    
    elif architecture == "SVM":
        # cv_method = StratifiedKFold(n_splits=2, shuffle=True)
        # grid_search = GridSearchCV(SVC(probability=True), 
        #                             param_grid=[
        # {"kernel": ["rbf", "linear"], "gamma": [1e-3, 1e-4], "C": [0.1, 1]}
        # ],
        #                             cv=cv_method,
        #                             scoring="accuracy",
        #                             refit=True,
        #                             verbose=2)
        
        # # Best SVM parameters:  {'C': 1, 'gamma': 0.001, 'kernel': 'linear'}
        # model_ = grid_search
        return SVC(probability=True, C=1, gamma=0.001, kernel="linear")
    
    elif architecture == "SVM-Rotor":
        pipe = Pipeline([
            ('scale', StandardScaler()),
            ('clf', SVC(probability=True, C=1, gamma=0.001, kernel="linear"))])
        return SVC(probability=True, C=1, gamma=0.001, kernel="linear")
    
    elif architecture == "kNN":
        return KNeighborsClassifier(90, weights="distance", metric="euclidean")
    
    elif architecture == "[FFNN,NB]":
        model_ffnn = tf.keras.Sequential()
        model_ffnn.add(tf.keras.layers.Input(shape=(input_layer_size,)))
        for _ in range(config.hidden_layers):
            model_ffnn.add(tf.keras.layers.Dense(hidden_layer_size, activation='relu', use_bias=True))
        model_ffnn.add(tf.keras.layers.Dense(output_layer_size, activation='softmax'))
        model_ffnn.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy",
                        metrics=["accuracy", SparseTopKCategoricalAccuracy(k=3, name="k3_accuracy")])
        model_nb = MultinomialNB(alpha=config.alpha, fit_prior=config.fit_prior)
        return [model_ffnn, model_nb]
    
    elif architecture == "[DT,ET,RF,SVM,kNN]":
        dt = DecisionTreeClassifier(criterion=config.criterion, ccp_alpha=config.ccp_alpha)
        et = ExtraTreesClassifier(n_estimators=config.n_estimators, criterion=config.criterion, 
                                  bootstrap=config.bootstrap, n_jobs=30,
                                  max_features=config.max_features, max_depth=30, 
                                  min_samples_split=config.min_samples_split,
                                  min_samples_leaf=config.min_samples_leaf)
        rf = RandomForestClassifier(n_estimators=config.n_estimators, criterion=config.criterion, 
                                    bootstrap=config.bootstrap, n_jobs=30,
                                    max_features=config.max_features, max_depth=30, 
                                    min_samples_split=config.min_samples_split,
                                    min_samples_leaf=config.min_samples_leaf)
        svm = SVC(probability=True, C=1, gamma=0.001, kernel="linear")
        knn = KNeighborsClassifier(90, weights="distance", metric="euclidean")
        return [dt, et, rf, svm, knn]
    
    else:
        raise Exception(f"Could not create model. Unknown architecture '{architecture}'.")

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
    parser.add_argument('--plaintext_input_directory', default='../data/gutenberg_en', type=str,
                        help='Input directory of the plaintexts for training the aca ciphers.')
    parser.add_argument('--rotor_input_directory', default='../data/rotor_ciphertexts', type=str,
                        help='Input directory of the rotor ciphertexts.')
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
                                 'SVM', 'kNN', '[FFNN,NB]', '[DT,ET,RF,SVM,kNN]', 'SVM-Rotor'],
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
                             '- SVM\n'
                             '- kNN\n'
                             '- [FFNN,NB]\n'
                             '- [DT,ET,RF,SVM,kNN]'
                             '- SVM-Rotor'
                             )
    parser.add_argument('--extend_model', default=None, type=str,
                        help='Load a trained model from a file and use it as basis for the new training.')

    return parser.parse_args()
    
def should_download_datasets(args):
    """Determines if the plaintext datasets should be loaded"""
    return (args.download_dataset and 
            not os.path.exists(args.plaintext_input_directory) and 
            args.plaintext_input_directory == os.path.abspath('../data/gutenberg_en'))

def download_datasets(args):
    """Downloads plaintexts and saves them in the plaintext_input_directory"""
    print("Downloading Datsets...")
    checksums_dir = '../data/checksums/'
    if not Path(checksums_dir).exists():
        os.mkdir(checksums_dir)
    tfds.download.add_checksums_dir(checksums_dir)

    download_manager = tfds.download.download_manager.DownloadManager(download_dir='../data/', 
                                                       extract_dir=args.plaintext_input_directory)
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

    path = os.path.join(args.plaintext_input_directory, 
                        'ZIP.ucid_1bF5sSVjxTx-P5wxn87nxWn_V_export_downloadR9Cwhunev5CvJ-ic__'
                        'HawxhTtGOlSdcCrro4fxfEI8A', 
                        os.path.basename(args.plaintext_input_directory))
    dir_name = os.listdir(path)
    for name in dir_name:
        p = Path(os.path.join(path, name))
        parent_dir = p.parents[2]
        p.rename(parent_dir / p.name)
    os.rmdir(path)
    os.rmdir(os.path.dirname(path))
    print("Datasets Downloaded.")

def load_rotor_ciphertext_datasets_from_disk(args, requested_cipher_types, *, max_lines_per_cipher):
    def validate_ciphertext_path(ciphertext_path, cipher_types):
        file_name = Path(ciphertext_path).stem.lower()
        if not file_name in cipher_types:
            raise Exception(f"Filename must equal one of the expected cipher types. "
                            "Expected cipher types are: {cipher_types}. Current "
                            "filename is '{file_name}'.")
    
    # Check if all rotor ciphers are in the requested cipher_types. Otherwise return empty lists.
    # TODO: Probably should change requested_cipher_type from indices to enums (aca, mct3, rotor, etc.)
    rotor_cipher_types = [config.CIPHER_TYPES[i] for i in range(56, 61)]
    for rotor_type in rotor_cipher_types:
        if not rotor_type in requested_cipher_types:
            empty_params = RotorCiphertextsDatasetParameters(config.ROTOR_CIPHER_TYPES, 
                                                            0,
                                                            args.dataset_workers, 
                                                            args.min_train_len, 
                                                            args.max_train_len,
                                                            generate_evalutation_data=False)
            return ([], empty_params, [], empty_params)
    
    rotor_cipher_dir = args.rotor_input_directory
    rotor_ciphertexts = []
    dir_name = os.listdir(rotor_cipher_dir)
    for name in dir_name:
        path = os.path.join(rotor_cipher_dir, name)
        if os.path.isfile(path):
            validate_ciphertext_path(path, config.ROTOR_CIPHER_TYPES)
            with open(path, "r") as f:
                label = Path(path).stem.lower()
                lines = f.readlines()
                max = max_lines_per_cipher
                for line in lines:
                    max -= 1
                    if max <= 0:
                        continue
                    rotor_ciphertexts.append((line.rstrip(), label))

    train_rotor_ciphertexts, test_rotor_ciphertexts = train_test_split(rotor_ciphertexts, test_size=0.2, 
                                                                       random_state=42, shuffle=True)

    # Calculate batch size for rotor ciphers. If both aca and rotor ciphers are requested, 
    # the amount of samples of each rotor cipher per batch should be equal to the 
    # amount of samples of each aca cipher per loaded batch.
    number_of_rotor_ciphers = len(config.ROTOR_CIPHER_TYPES)
    number_of_aca_ciphers = len(requested_cipher_types) - number_of_rotor_ciphers
    if number_of_aca_ciphers == 0:
        rotor_dataset_batch_size = args.train_dataset_size
    else:
        amount_of_samples_per_cipher = args.train_dataset_size // number_of_aca_ciphers
        rotor_dataset_batch_size = amount_of_samples_per_cipher * number_of_rotor_ciphers

    train_rotor_ciphertexts_parameters = RotorCiphertextsDatasetParameters(config.ROTOR_CIPHER_TYPES, 
                                                            rotor_dataset_batch_size,
                                                            args.dataset_workers, 
                                                            args.min_train_len, 
                                                            args.max_train_len,
                                                            generate_evalutation_data=False)
    test_rotor_ciphertexts_parameters = RotorCiphertextsDatasetParameters(config.ROTOR_CIPHER_TYPES,
                                                            rotor_dataset_batch_size,
                                                            args.dataset_workers, 
                                                            args.min_test_len, 
                                                            args.max_test_len,
                                                            generate_evalutation_data=False)
    
    return (train_rotor_ciphertexts, train_rotor_ciphertexts_parameters, 
            test_rotor_ciphertexts, test_rotor_ciphertexts_parameters)

def load_plaintext_datasets_from_disk(args, cipher_types):
    # Check if all aca ciphers are in the required cipher_types. Otherwise return empty lists.
    aca_cipher_types = [config.CIPHER_TYPES[i] for i in range(56)]
    for aca_type in aca_cipher_types:
        if not aca_type in cipher_types:
            empty_params = PlaintextPathsDatasetParameters([], 
                                                           args.train_dataset_size, 
                                                           args.min_train_len, 
                                                           args.max_train_len,
                                                           args.keep_unknown_symbols, 
                                                           args.dataset_workers)
            return ([], empty_params, [], empty_params)

    plaintext_files = []
    dir_name = os.listdir(args.plaintext_input_directory)
    for name in dir_name:
        path = os.path.join(args.plaintext_input_directory, name)
        if os.path.isfile(path):
            plaintext_files.append(path)
    
    train_plaintexts, test_plaintexts = train_test_split(plaintext_files, test_size=0.05, 
                                                         random_state=42, shuffle=True)
    
    train_plaintext_parameters = PlaintextPathsDatasetParameters(cipher_types, args.train_dataset_size, 
                                                args.min_train_len, args.max_train_len,
                                                args.keep_unknown_symbols, args.dataset_workers)
    test_plaintext_parameters = PlaintextPathsDatasetParameters(cipher_types, args.train_dataset_size,     
                                               args.min_test_len, args.max_test_len,
                                               args.keep_unknown_symbols, args.dataset_workers)
    
    return (train_plaintexts, train_plaintext_parameters, test_plaintexts, test_plaintext_parameters)

def load_datasets_from_disk(args, cipher_types, max_rotor_lines):
    """Loads datasets from the file system. 
    In case of the ACA ciphers the datasets are plaintext files that need to be
    encrypted before the features can be extracted.
    In case of the rotor ciphers there are already encrypted ciphertext files 
    that can directly be used to extract the features.
    """
    print("Loading Datasets...")
    
    (train_plaintexts, 
     train_plaintext_parameters, 
     test_plaintexts, 
     test_plaintext_parameters) = load_plaintext_datasets_from_disk(args, cipher_types)
    (train_rotor_ciphertexts, 
     train_rotor_ciphertexts_parameters, 
     test_rotor_ciphertexts, 
     test_rotor_ciphertexts_parameters) = load_rotor_ciphertext_datasets_from_disk(args, cipher_types, max_lines_per_cipher=max_rotor_lines)

    train_ds = CipherStatisticsDataset(train_plaintexts, train_plaintext_parameters, train_rotor_ciphertexts, train_rotor_ciphertexts_parameters)
    test_ds = CipherStatisticsDataset(test_plaintexts, test_plaintext_parameters, test_rotor_ciphertexts, test_rotor_ciphertexts_parameters)

    if train_ds.key_lengths_count > 0 and args.train_dataset_size % train_ds.key_lengths_count != 0:
        print("WARNING: the --train_dataset_size parameter must be dividable by the amount of --ciphers  and the length configured "
              "KEY_LENGTHS in config.py. The current key_lengths_count is %d" % 
                  train_ds.key_lengths_count, file=sys.stderr)
    print("Datasets loaded.\n")
    return train_ds, test_ds
    
def create_model_with_distribution_strategy(architecture, extend_model, cipher_types, max_train_len):
    """Creates models depending on the GPU count and on extend_model"""
    print('Creating model...')

    gpu_count = (len(tf.config.list_physical_devices('GPU')) +
        len(tf.config.list_physical_devices('XLA_GPU')))
    if gpu_count > 1:
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            if extend_model is not None:
                extend_model = tf.keras.models.load_model(extend_model, compile=False)
            model = create_model(architecture, extend_model, cipher_types, max_train_len)
        if architecture in ("FFNN", "CNN", "LSTM", "Transformer") and extend_model is None:
            model.summary()
    else:
        if extend_model is not None:
            extend_model = tf.keras.models.load_model(extend_model, compile=False)
        model = create_model(architecture, extend_model, cipher_types, max_train_len)
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
        weights[np.where(class_labels.numpy() >= first_rotor_cipher_class)] = 4
        return weights
    
    checkpoints_dir = Path('../data/checkpoints')
    def delete_previous_checkpoints():
        shutil.rmtree(checkpoints_dir)

    def create_checkpoint_callback():
        if not checkpoints_dir.exists():
            os.mkdir(checkpoints_dir)
        checkpoint_file_path = os.path.join(checkpoints_dir, "epoch_{epoch:02d}-accuracy_{accuracy:.2f}.h5")
        # checkpoint_file_path = checkpoints_dir / "epoch_{epoch:02d}-accuracy_{accuracy:.2f}.h5"

        return tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_file_path,
            save_weights_only=False,
            monitor='val_accuracy',
            mode='max',
            save_best_only=False,
            save_freq=100)

    print('Training model...')
    delete_previous_checkpoints()

    architecture = args.architecture

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='../data/logs', update_freq='epoch')
    early_stopping_callback = MiniBatchEarlyStopping(
        min_delta=1e-5, patience=250, monitor='accuracy', mode='max', restore_best_weights=True)
    # time_based_decay_lrate_callback = TimeBasedDecayLearningRateScheduler(args.train_dataset_size)
    custom_step_decay_lrate_callback = CustomStepDecayLearningRateScheduler(early_stopping_callback)
    checkpoint_callback = create_checkpoint_callback()
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

        # For architectures that only support one fit call: Sample all batches into one large batch.
        if architecture in ("DT", "RF", "ET", "SVM", "kNN", "SVM-Rotor", "[DT,ET,RF,SVM,kNN]"):
            for training_batch in training_batches:
                combined_batch.extend(training_batch)
            if train_ds.iteration < args.max_iter:
                print("Loaded %d ciphertexts." % train_ds.iteration)
                continue
            train_ds.stop_outstanding_tasks()
            print("Loaded %d ciphertexts." % train_ds.iteration)
            training_batches = [combined_batch]

        for index, training_batch in enumerate(training_batches):
            statistics, labels = training_batch.tuple()
            train_iter = train_ds.iteration - len(training_batch) * (len(training_batches) - index - 1)

            # Create small validation dataset on first iteration
            if index == 0:
                # TODO: This does not work in case of DT, RF, etc. where on first iteration
                # all trainings statistics and labels are combined into one large set.
                # The validation data leaks into the training data.
                statistics, val_data, labels, val_labels = train_test_split(statistics.numpy(), 
                                                                            labels.numpy(), 
                                                                            test_size=0.3)
                statistics = tf.convert_to_tensor(statistics)
                val_data = tf.convert_to_tensor(val_data)
                labels = tf.convert_to_tensor(labels)
                val_labels = tf.convert_to_tensor(val_labels)
            train_iter -= len(training_batch) * 0.3

            # Decision Tree training
            if architecture in ("DT", "RF", "ET", "SVM", "kNN", "SVM-Rotor"):
                train_iter = len(labels) * 0.7
                print(f"Start training the {architecture}.")
                if architecture == "kNN":
                    # TODO: Since scikit's kNN does not support sample weights: Provide manually?
                    history = model.fit(statistics, labels)
                else:
                    # history = model.fit(statistics, labels)
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
                                                  custom_step_decay_lrate_callback, checkpoint_callback])
                # time_based_decay_lrate_callback.iteration = train_iter
                history = model[1].partial_fit(statistics, labels, classes=classes, 
                                               sample_weight=sample_weights(labels))
            
            # DT, ET, RF, SVM, kNN
            elif architecture == "[DT,ET,RF,SVM,kNN]":
                train_iter = len(labels) * 0.7
                print(f"Start training the {architecture}.")
                dt, et, rf, svm, knn = model
                for index, m in enumerate([dt, et, rf, svm]):
                    m.fit(statistics, labels, sample_weight=sample_weights(labels))
                    print(f"Trained model {index + 1} of {len(model)}")
                knn.fit(statistics, labels)
                print(f"Trained model {len(model)} of {len(model)}")

            else:
                history = model.fit(statistics, labels, batch_size=args.batch_size, 
                                    validation_data=(val_data, val_labels), epochs=args.epochs,
                                    sample_weight=sample_weights(labels),
                                    callbacks=[early_stopping_callback, tensorboard_callback, 
                                               custom_step_decay_lrate_callback, checkpoint_callback])
                # time_based_decay_lrate_callback.iteration = train_iter

            # print for Decision Tree, Naive Bayes and Random Forests
            if architecture in ("DT", "NB", "RF", "ET", "SVM", "kNN", "SVM-Rotor"):
                val_score = model.score(val_data, val_labels)
                train_score = model.score(statistics, labels)
                print("train accuracy: %f, validation accuracy: %f" % (train_score, val_score))

            if architecture == "[FFNN,NB]":
                val_score = model[1].score(val_data, val_labels)
                train_score = model[1].score(statistics, labels)
                print("train accuracy: %f, validation accuracy: %f" % (train_score, val_score))

            if architecture == "[DT,ET,RF,SVM,kNN]":
                for m in model:
                    val_score = m.score(val_data, val_labels)
                    train_score = m.score(statistics, labels)
                    print(f"{type(m).__name__}: train accuracy: {train_score}, validation accuracy: {val_score}")

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
    architecture = args.architecture
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
    elif architecture in ("DT", "NB", "RF", "ET", "SVM", "kNN", "SVM-Rotor"):
        with open(model_path, "wb") as f:
            # this gets very large
            pickle.dump(model, f)
    elif architecture == "[FFNN,NB]":
        model[0].save('../data/models/' + model_path.split('.')[0] + "_ffnn.h5")
        with open('../data/models/' + model_path.split('.')[0] + "_nb.h5", "wb") as f:
            # this gets very large
            pickle.dump(model[1], f)
    elif architecture == "[DT,ET,RF,SVM,kNN]":
        for index, name in enumerate(["dt","et","rf","svm","knn"]):
            with open('../data/models/' + model_path.split('.')[0] + f"_{name}.h5", "wb") as f:
                # this gets very large
                pickle.dump(model[index], f)
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
    architecture = args.architecture
    start_time = time.time()
    total_len_prediction = 0

    prediction_dataset_factor = 10
    if early_stopping_callback.stop_training:
        while test_ds.dataset_workers * test_ds.batch_size > train_iter / prediction_dataset_factor and prediction_dataset_factor > 1:
            prediction_dataset_factor -= 1
        args.max_iter = int(train_iter / prediction_dataset_factor)
    else:
        while test_ds.dataset_workers * test_ds.batch_size > args.max_iter / prediction_dataset_factor and prediction_dataset_factor > 1:
            prediction_dataset_factor -= 1
        args.max_iter /= prediction_dataset_factor
    cntr = 0
    test_iter = 0
    test_epoch = 0

    # sample all predictions and labels for later use
    prediction_metrics = {}
    if architecture == "[FFNN,NB]":
        prediction_metrics = {"FFNN": PredictionPerformanceMetrics(model_name="FFNN"),
                              "NB": PredictionPerformanceMetrics(model_name="NB")}
    elif architecture == "[DT,ET,RF,SVM,kNN]":
        prediction_metrics = {"DT": PredictionPerformanceMetrics(model_name="DT"),
                              "ET": PredictionPerformanceMetrics(model_name="ET"),
                              "RF": PredictionPerformanceMetrics(model_name="RF"),
                              "SVM": PredictionPerformanceMetrics(model_name="SVM"),
                              "kNN": PredictionPerformanceMetrics(model_name="kNN"),}
    else:
        prediction_metrics = {architecture: PredictionPerformanceMetrics(model_name=architecture)}

    combined_batch = TrainingBatch("mixed", [], [])
    while test_ds.iteration < args.max_iter:
        testing_batches = next(test_ds)

        # For architectures that only support one fit call: Sample all batches into one large batch.
        if architecture in ("DT", "RF", "ET", "SVM", "kNN", "SVM-Rotor", "[DT,ET,RF,SVM,kNN]"):
            for testing_batch in testing_batches:
                combined_batch.extend(testing_batch)
            if test_ds.iteration < args.max_iter:
                print("Loaded %d ciphertexts." % test_ds.iteration)
                continue
            test_ds.stop_outstanding_tasks()
            print("Loaded %d ciphertexts." % test_ds.iteration)
            testing_batches = [combined_batch]

        for testing_batch in testing_batches:
            statistics, labels = testing_batch.tuple()
            
            # Decision Tree, Naive Bayes prediction
            if architecture in ("DT", "NB", "RF", "ET", "SVM", "kNN"):
                prediction = model.predict_proba(statistics)
                prediction_metrics[architecture].add_predictions(labels, prediction)
            if architecture == "SVM-Rotor":
                prediction = model.predict_proba(statistics)
                # add probability 0 to all aca labels that are missing in the prediction
                padded_prediction = []
                for p in list(prediction):
                    padded = [0] * 56 + list(p)
                    padded_prediction.append(padded)
                prediction_metrics[architecture].add_predictions(labels, padded_prediction)
            elif architecture == "[FFNN,NB]":
                prediction = model[0].predict(statistics, batch_size=args.batch_size, verbose=1)
                nb_prediction = model[1].predict_proba(statistics)
                prediction_metrics["FFNN"].add_predictions(labels, prediction)
                prediction_metrics["NB"].add_predictions(labels, nb_prediction)
            elif architecture == "[DT,ET,RF,SVM,kNN]":
                prediction = model[0].predict_proba(statistics)
                prediction_metrics["DT"].add_predictions(labels, prediction)
                prediction_metrics["ET"].add_predictions(labels, model[1].predict_proba(statistics))
                prediction_metrics["RF"].add_predictions(labels, model[2].predict_proba(statistics))
                prediction_metrics["SVM"].add_predictions(labels, model[3].predict_proba(statistics))
                prediction_metrics["kNN"].add_predictions(labels, model[4].predict_proba(statistics))
            else:
                prediction = model.predict(statistics, batch_size=args.batch_size, verbose=1)
                prediction_metrics[architecture].add_predictions(labels, prediction)

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
    
    test_ds.stop_outstanding_tasks()
    elapsed_prediction_time = datetime.fromtimestamp(time.time()) - datetime.fromtimestamp(start_time)

    if total_len_prediction > args.train_dataset_size:
        total_len_prediction -= total_len_prediction % args.train_dataset_size
    print('\ntest data predicted: %d ciphertexts' % total_len_prediction)

    # print prediction metrics
    for metrics in prediction_metrics.values():
        metrics.print_evaluation()

    prediction_stats = 'Prediction time: %d days %d hours %d minutes %d seconds with %d iterations and %d epochs.' % (
        elapsed_prediction_time.days, elapsed_prediction_time.seconds // 3600, 
        (elapsed_prediction_time.seconds // 60) % 60,
        elapsed_prediction_time.seconds % 60, test_iter, test_epoch)
    
    return prediction_stats

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

def main():
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

    args.plaintext_input_directory = os.path.abspath(args.plaintext_input_directory)
    args.rotor_input_directory = os.path.abspath(args.rotor_input_directory)
    args.ciphers = args.ciphers.lower()
    architecture = args.architecture
    cipher_types = args.ciphers.split(',')

    extend_model = args.extend_model
    architecture = args.architecture
    if extend_model is not None:
        if architecture not in ('FFNN', 'CNN', 'LSTM'):
            # TODO: Also holds for SVM, kNN?
            print('ERROR: Models with the architecture %s can not be extended!' % architecture,
                  file=sys.stderr)
            sys.exit(1)
        if len(os.path.splitext(extend_model)) != 2 or os.path.splitext(extend_model)[1] != '.h5':
            print('ERROR: The extended model name must have the ".h5" extension!', file=sys.stderr)
            sys.exit(1)

    cipher_types = expand_cipher_groups(cipher_types)
    if architecture == "SVM-Rotor":
        cipher_types = [config.CIPHER_TYPES[i] for i in range(56, 61)]

    if args.train_dataset_size * args.dataset_workers > args.max_iter:
        print("ERROR: --train_dataset_size * --dataset_workers must not be bigger than --max_iter. "
              "In this case it was %d > %d" % 
                  (args.train_dataset_size * args.dataset_workers, args.max_iter), 
              file=sys.stderr)
        sys.exit(1)

    if should_download_datasets(args):
        download_datasets(args)

    # For the rotor-only model, load balanced set of great performing samples from the Paper
    # 'Classifying World War II Era Ciphers with Machine Learning' from Dalton and Stamp (in
    # the first 900 lines of each rotor ciphertext file) and a generated set of possibly more
    # representative ciphertexts.
    max_rotor_lines = 2000 if architecture == "SVM-Rotor" else 10000

    train_ds, test_ds = load_datasets_from_disk(args, cipher_types, max_rotor_lines)

    # ACA ciphers
    model = create_model_with_distribution_strategy(architecture, extend_model, 
                                                    cipher_types, args.max_train_len)
    early_stopping_callback, train_iter, training_stats = train_model(model, args, train_ds)
    save_model(model, args)
    prediction_stats = predict_test_data(test_ds, model, args, early_stopping_callback, train_iter)
    
    print(training_stats)
    print(prediction_stats)

if __name__ == "__main__":
    main()    
