from itertools import groupby
import tensorflow as tf
import cipherTypeDetection.config as config
from cipherImplementations.simpleSubstitution import SimpleSubstitution
import sys
from util.utils import map_text_into_numberspace
import copy
import random
import multiprocessing
from multiprocessing import pool as multiprocessing_pool
import logging
import numpy as np
from collections import deque
from featureCalculations import calculate_statistics
sys.path.append("../")


def encrypt(plaintext, label, key_length, keep_unknown_symbols, return_key=False):
    cipher = config.CIPHER_IMPLEMENTATIONS[label]
    plaintext = cipher.filter(plaintext, keep_unknown_symbols)
    if cipher.needs_plaintext_of_specific_length:
        plaintext = cipher.truncate_plaintext(plaintext, key_length)
    key = cipher.generate_random_key(key_length)
    if return_key:
        orig_key = copy.deepcopy(key)
    plaintext_numberspace = map_text_into_numberspace(plaintext, cipher.alphabet, cipher.unknown_symbol_number)
    if isinstance(key, bytes):
        key = map_text_into_numberspace(key, cipher.alphabet, cipher.unknown_symbol_number)
    elif isinstance(key, list) and len(key) == 2 and isinstance(key[0], bytes) and isinstance(key[1], bytes):
        key[0] = map_text_into_numberspace(key[0], cipher.alphabet, cipher.unknown_symbol_number)
        key[1] = map_text_into_numberspace(key[1], cipher.alphabet, cipher.unknown_symbol_number)
    elif isinstance(key, list) and len(key) == 3 and isinstance(key[0], bytes) and isinstance(key[1], bytes) and isinstance(key[2], bytes):
        key[0] = map_text_into_numberspace(key[0], cipher.alphabet, cipher.unknown_symbol_number)
        key[1] = map_text_into_numberspace(key[1], cipher.alphabet, cipher.unknown_symbol_number)
        key[2] = map_text_into_numberspace(key[2], cipher.alphabet, cipher.unknown_symbol_number)
    elif isinstance(key, list) and len(key) == 2 and isinstance(key[0], bytes) and isinstance(key[1], int):
        key[0] = map_text_into_numberspace(key[0], cipher.alphabet, cipher.unknown_symbol_number)
    elif isinstance(key, list) and len(key) == 3 and isinstance(key[0], int) and isinstance(key[1], bytes) and isinstance(key[2], bytes):
        key[1] = map_text_into_numberspace(key[1], cipher.alphabet, cipher.unknown_symbol_number)
        key[2] = map_text_into_numberspace(key[2], cipher.alphabet, cipher.unknown_symbol_number)
    elif isinstance(key, list) and len(key) == 2 and isinstance(key[0], (list, np.ndarray)) and (len(key[0]) == 5 or len(
            key[0]) == 10) and isinstance(key[1], bytes):
        key[1] = map_text_into_numberspace(key[1], cipher.alphabet, cipher.unknown_symbol_number)
    elif isinstance(key, list) and len(key) == 3 and isinstance(key[0], list) and isinstance(key[1], np.ndarray) and isinstance(
            key[2], bytes):
        key[2] = map_text_into_numberspace(key[2], cipher.alphabet, cipher.unknown_symbol_number)
    elif isinstance(key, dict):
        new_key_dict = {}
        for k in key:
            new_key_dict[cipher.alphabet.index(k)] = key[k]
        key = new_key_dict

    ciphertext = cipher.encrypt(plaintext_numberspace, key)
    if b'j' not in cipher.alphabet and config.CIPHER_TYPES[label] != 'homophonic':
        ciphertext = normalize_text(ciphertext, 9)
    if b'x' not in cipher.alphabet:
        ciphertext = normalize_text(ciphertext, 23)
    if return_key:
        return ciphertext, orig_key
    return ciphertext


def normalize_text(text, pos):
    for i in range(len(text)):
        if 26 >= text[i] >= pos:
            text[i] += 1
    return text

def pad_sequences(sequences, maxlen):
    """Pad sequences with data from itself."""
    ret_sequences = []
    for sequence in sequences:
        length = len(sequence)
        sequence = sequence * (maxlen // length) + sequence[:maxlen % length]
        ret_sequences.append(sequence)
    return np.array(ret_sequences)


multiprocessing_logger = multiprocessing.log_to_stderr(logging.INFO) # TODO: Use get_logger instead!?

class CipherStatisticsDataset:
    """This classes takes inputs for the composed `PlaintextPathsDataset` and 
    `RotorCiphertextsDataset`, does some processing (e.g. filtering of the characters and
    encryption) and finally calculate the statistics for the inputs. 
    The class provides a iterator interface which returns the calculated statistics and their 
    labels as `TrainingBatch`es. The number of training batches that are returned by each iteration 
    depend on `dataset_workers`. Each `TrainingBatch` contains labels and statistics for both kinds
    of underlying datasets (RotorCiphertextsDataset and PlaintextPathsDataset).
    To process the inputs, `dataset_workers` processes are used to calculate the statistics
    in parallel. The processes are organized in process pools to keep the overhead for small
    `batch_size`es small. To limit the time each iteration takes, brefore returning the current
    iterations results, new processes for the next iteration are already started and moved into
    the `_processing_queue`.
    """

    def __init__(self, plaintext_paths, plaintext_dataset_params, rotor_ciphertexts_with_labels, 
                 rotor_ciphertext_dataset_params, *, generate_evaluation_data=False):
        assert plaintext_dataset_params.dataset_workers == rotor_ciphertext_dataset_params.dataset_workers

        dataset_workers = plaintext_dataset_params.dataset_workers

        self._iteration = 0
        self._epoch = 0
        self._dataset_workers = dataset_workers
        
        self._pool = multiprocessing_pool.Pool(self._dataset_workers)
        # double ended queue for storing asynchronously processing functions
        self._processing_queue = deque() 
        self._logger = multiprocessing_logger

        self._plaintext_paths = plaintext_paths
        self._plaintext_dataset_params = plaintext_dataset_params
        self._rotor_ciphertexts_with_labels = rotor_ciphertexts_with_labels
        self._rotor_ciphertext_dataset_params = rotor_ciphertext_dataset_params

        self._initialize_datasets()

        self._generate_evaluation_data = generate_evaluation_data
    
    def _initialize_datasets(self):
        self._plaintext_dataset = PlaintextPathsDataset(self._plaintext_paths, self._plaintext_dataset_params, 
                                                        self._logger)
        self._ciphertext_dataset = RotorCiphertextsDataset(self._rotor_ciphertexts_with_labels, 
                                                           self._rotor_ciphertext_dataset_params, 
                                                           self._logger)
    
    @property
    def iteration(self):
        """The iteration corresponds to the number of lines processed by the dataset."""
        return self._iteration
    
    @property
    def epoch(self):
        """Each epoch represents the processing of all available inputs. If the epoch is 
        increased, the dataset will restart iterating it's inputs from the beginning."""
        return self._epoch
    
    @property
    def dataset_workers(self):
        """The number of parallel workers to use when calculating the statistics of the input. 
        This number also equals the number of `TrainingBatch`es returned by `__next__`."""
        return self._dataset_workers
    
    @property
    def batch_size(self):
        """The amount of statistics and labels in the returned `TrainingBatch`es of 
        iterator method `__next__`."""
        return (self._plaintext_dataset_params.batch_size + 
                self._rotor_ciphertext_dataset_params.batch_size)
    
    @property
    def key_lengths_count(self):
        """Returns the combined count of all key length values for all supported ciphers
        of the plaintext path dataset. See also `config.KEY_LENGTHS`.
        """
        return self._plaintext_dataset.key_lengths_count
    
    def stop_outstanding_tasks(self):
        self._pool.terminate()
        self._processing_queue = deque()
    
    def __iter__(self):
        return self
    
    def __next__(self):
        config_params = ConfigParams(config.CIPHER_TYPES, config.KEY_LENGTHS, 
                                     config.FEATURE_ENGINEERING, config.PAD_INPUT)
        
        # init epoch on first iteration
        if self._epoch == 0:
            self._epoch = 1

        ciphertext_inputs_exhausted = False
        plaintext_inputs_exhausted = False

        # process rotor cipher datasets
        ciphertext_worker = CiphertextLine2CipherStatisticsWorker(
            self._rotor_ciphertext_dataset_params, config_params)
        plaintext_worker = PlaintextLine2CipherStatisticsWorker(
            self._plaintext_dataset_params, config_params)

        # Number of workers to start to yield a single combined batch of rotor and aca 
        # ciphers.
        combined_process_count = self._dataset_workers * 2

        # Process both kinds of ciphers at once. Queue `combined_process_count * 2` processes, 
        # to ensure that the next `TrainingBatch`es are prepared while the models in `train.py` 
        # are trained. This ensures that the wait times for each iteration of the dataset are 
        # shorter then if we only start preprocessing at the begining of __next__ calls.
        # (Since the pool is initialized with `_dataset_workers`, the number of processes is
        # still kept at `_dataset_workers`.)
        ciphertext_inputs_exhausted, plaintext_inputs_exhausted = self._dispatch_concurrent(
            ciphertext_worker=ciphertext_worker, ciphertext_dataset=self._ciphertext_dataset, 
            plaintext_worker=plaintext_worker, plaintext_dataset=self._plaintext_dataset,
            number_of_processes=combined_process_count * 2)
        
        # Inputs exhausted: Increase epoch, re-initialize datasets and therefore begin 
        # iteration from the start.
        if ciphertext_inputs_exhausted or plaintext_inputs_exhausted:
            print(f"CipherStatisticsDataset: Ciphertexts or plaintexts of epoch {self._epoch} exhausted! Resetting iterators!")
            self._epoch += 1
            self._initialize_datasets()

        # Get all results of this iteration
        all_results = self._wait_for_results(number_of_processes=combined_process_count)

        # Wait until the workers of both cipher types have finished. Otherwise the returned
        # batch could only contain aca or rotor ciphers. 
        while TrainingBatch.represent_equal_cipher_type(all_results):
            assert len(self._processing_queue) > 0, "Expected different cipher type in queue!"
            next_results = self._wait_for_results(number_of_processes=combined_process_count)
            all_results.extend(next_results)

        # Combine ACA and rotor cipher training batches. Each batch should contain some 
        # statistics and labels of both.
        if self._generate_evaluation_data:
            paired_cipher_types = EvaluationBatch.paired_cipher_types(all_results)
            return [EvaluationBatch.combined(pair) for pair in paired_cipher_types] 
        else:
            paired_cipher_types = TrainingBatch.paired_cipher_types(all_results)
            return [TrainingBatch.combined(pair) for pair in paired_cipher_types] 

    def _dispatch_concurrent(self, *, ciphertext_worker, ciphertext_dataset, plaintext_worker, 
                             plaintext_dataset, number_of_processes):
        error_callback = lambda error: print(f"ERROR in ParallelIterator: {error}")

        ciphertext_inputs_exhausted = False
        plaintext_inputs_exhausted = False

        for index in range(number_of_processes):
            # start rotor and plaintext worker alternatly after one another
            if index % 2 == 0:
                try:
                    worker = ciphertext_worker
                    input_batch = next(ciphertext_dataset)
                except StopIteration:
                    ciphertext_inputs_exhausted = True
                    continue
            else:
                try:
                    worker = plaintext_worker
                    input_batch = next(plaintext_dataset)
                except StopIteration:
                    plaintext_inputs_exhausted = True
                    continue

            batch = self._pool.apply_async(worker.perform, 
                                            (input_batch, ),
                                            error_callback=error_callback)
            self._processing_queue.append(batch)
        
        return (ciphertext_inputs_exhausted, plaintext_inputs_exhausted)

    def _wait_for_results(self, number_of_processes):
        training_batches = []
        for _ in range(number_of_processes):
            try:
                result = self._processing_queue.popleft()
            except IndexError:
                break
            training_batch = result.get()
            self._iteration += len(training_batch)
            training_batches.append(training_batch)

        return training_batches

class RotorCiphertextsDatasetParameters:
    """Encapsulates the parameters of `RotorCiphertextsDataset`. These parameters are used
    for the initialization of the dataset itself, as well as for the worker processes and
    the main statistics dataset."""

    def __init__(self, cipher_types, batch_size, dataset_workers, 
                 min_text_len, max_text_len, generate_evalutation_data):
        self.cipher_types = cipher_types
        self.batch_size = batch_size
        self.dataset_workers = dataset_workers
        self.min_text_len = min_text_len
        self.max_text_len = max_text_len
        self.generate_evaluation_data = generate_evalutation_data
    
class RotorCiphertextsDataset:
    """Takes rotor ciphertext (with their labels) as input and returns batched
    lists of the ciphertext lines (and their labels) as iteration result.
    The inputs are converted to match the `max_text_len` parameter of 
    `RotorCiphertextsDatasetParameters` and rearranged to alternate
    the samples of the different cipher types.
    The batching allows for splitting the input in workable chunks 
    (that fit in RAM) and can be distributed to multiple worker processes."""
    def __init__(self, ciphertexts_with_labels, dataset_params, logger):
        self._ciphertexts_with_labels = ciphertexts_with_labels
        self._batch_size = dataset_params.batch_size
        self._index = 0
        self._convert_lines_to_length(self._ciphertexts_with_labels, 
                                      dataset_params.min_text_len,
                                      dataset_params.max_text_len)
        self._rearrange_ciphertexts()
        self._logger = logger

    def _convert_lines_to_length(self, ciphertexts_with_labels, min_length, max_length):
        """Convert lines so that their length is between `min_length` and `max_length`. 
        The corresponding labels (cipher types) of the ciphertext lines are kept after 
        combining / truncating."""
        ciphertexts_with_labels = sorted(ciphertexts_with_labels, key=lambda elem: elem[1])

        # Samples stores a long string of all concatenated ciphertexts of the same label.
        # The length of samples is therefore the same as the number of labels (cipher names).
        samples = []
        # Memorize labels in the order as they appear in `ciphertexts_with_labels`.
        labels = []

        # Concatenate all ciphertexts into long samples
        previous_label = None
        for ciphertext, label in ciphertexts_with_labels:
            if label != previous_label:
                samples.append(ciphertext)
                labels.append(label)
                previous_label = label
            else:
                samples[-1] = samples[-1] + ciphertext
        
        # Split the samples at `max_length` and append them into `results` with their labels
        results = []
        for index, sample in enumerate(samples):
            split_index = 0
            while split_index < len(sample):
                split_length = random.randint(min_length, max_length)
                splitted = sample[split_index:split_index + split_length]
                split_index += split_length
                results.append((splitted, labels[index]))

        self._ciphertexts_with_labels = results

    def _rearrange_ciphertexts(self):
        get_label = lambda element: element[1]
        # Ensure _ciphertexts_with_labels is sorted
        self._ciphertexts_with_labels = sorted(self._ciphertexts_with_labels, key=get_label)
        # Group by label
        grouped = []
        for _, group in groupby(self._ciphertexts_with_labels, key=get_label):
            grouped.append(list(group))

        rearranged = []
        for group in zip(*grouped):
            for element in group:
                rearranged.append(element)

        self._ciphertexts_with_labels = rearranged

    def __iter__(self):
        return self
    
    def __next__(self):
        input_length = len(self._ciphertexts_with_labels)
        if self._index >= input_length:
            raise StopIteration()

        end_index = self._index + self._batch_size
        result = self._ciphertexts_with_labels[self._index:end_index] 

        self._logger.info(f"RotorCiphertextDataset: Returning batch {self._index // self._batch_size}")

        self._index += self._batch_size
        return result
    
class PlaintextPathsDatasetParameters:
    """Encapsulates the parameters of `PlaintextPathsDataset`. These parameters are used
    for the initialization of the dataset itself, as well as for the worker processes and
    the main statistics dataset."""

    def __init__(self, cipher_types, batch_size, min_text_len, max_text_len, 
                 keep_unknown_symbols=False, dataset_workers=None,
                 generate_evaluation_data=False):
        self.cipher_types = cipher_types
        self.batch_size = batch_size
        self.min_text_len = min_text_len
        self.max_text_len = max_text_len
        self.keep_unknown_symbols = keep_unknown_symbols
        self.dataset_workers = dataset_workers
        self.generate_evaluation_data = generate_evaluation_data
    
class PlaintextPathsDataset:
    """Takes paths to plaintexts and returns list of size `batch_size` with lines
    from the plaintext files."""

    def __init__(self, plaintext_paths, dataset_params, logger):
        self._batch_size = dataset_params.batch_size
        self._min_text_len = dataset_params.min_text_len
        self._max_text_len = dataset_params.max_text_len
        self._keep_unknown_symbols = dataset_params.keep_unknown_symbols
        self._logger = logger

        cipher_types = dataset_params.cipher_types

        key_lengths_count = 0
        for cipher_t in cipher_types:
            index = cipher_types.index(cipher_t)
            if isinstance(config.KEY_LENGTHS[index], list):
                key_lengths_count += len(config.KEY_LENGTHS[index])
            else:
                key_lengths_count += 1
        self._key_lengths_count = key_lengths_count

        self._plaintext_dataset = tf.data.TextLineDataset(plaintext_paths)
        self._dataset_iter = self._plaintext_dataset.__iter__()

        self._index = 0
    
    @property
    def key_lengths_count(self):
        return self._key_lengths_count

    def __iter__(self):
        return self

    def __next__(self):
        """todo"""
        c = SimpleSubstitution(config.INPUT_ALPHABET, config.UNKNOWN_SYMBOL, config.UNKNOWN_SYMBOL_NUMBER)

        result = []
        number_of_lines = self._batch_size // self._key_lengths_count
        if number_of_lines == 0:
            print(f"ERROR: Batch size is too small to calculate the features for all cipher and key"
                  f"length combinations! Current batch size: {self._batch_size}. Minimum batch size: "
                  f"{self._key_lengths_count}.")
            raise StopIteration

        for _ in range(number_of_lines):
            # use the basic prefilter to get the most accurate text length
            filtered_data = c.filter(next(self._dataset_iter).numpy(), self._keep_unknown_symbols)
            # Select a random min length of the filtered_data to provide more variance in the 
            # resulting text length
            random_min_length = random.randint(self._min_text_len, self._max_text_len)
            while len(filtered_data) < random_min_length:
                # add the new data to the existing to speed up the searching process.
                filtered_data += c.filter(next(self._dataset_iter).numpy(), self._keep_unknown_symbols)
            if len(filtered_data) > self._max_text_len:
                result.append(filtered_data[:self._max_text_len-(self._max_text_len % 2)])
            else:
                result.append(filtered_data[:len(filtered_data)-(len(filtered_data) % 2)])
        
        self._logger.info(f"PlaintextPathsDataset: Returning batch {self._index}")
        self._index += 1

        return result

class ConfigParams:
    """Encapsulates some entries of the `config.py`. This removes the calls to global state
    from the workers. Should help to reason about the code, especially since the workers
    are typically executed in a concurrent process"""
    def __init__(self, cipher_types, key_lengths, feature_engineering, pad_input):
        # corresponds to config.CIPHER_TYPES
        self.cipher_types = copy.deepcopy(cipher_types)
        # corresponds to config.KEY_LENGTHS
        self.key_lengths = copy.deepcopy(key_lengths)
        # corresponds to config.FEATURE_ENGINEERING
        self.feature_engineering = copy.deepcopy(feature_engineering)
        #corresponds to config.PAD_INPUT
        self.pad_input = copy.deepcopy(pad_input)

class CiphertextLine2CipherStatisticsWorker:
    """This class provides an iterator that returns `TrainingBatch`es.
    It takes ciphertext lines and their corresponding labels (cipher names) and 
    calculates the statistics (features) for those lines.
    The size of the batches depends on the `batch_size` and `dataset_workers`. 
    Each `dataset_worker` will calculate the statistics for `batch_size` lines of the
    input. Therefore the output of the __next__ method will return lists of length 
    `dataset_workers`."""

    def __init__(self, dataset_params, config_params):
        self._max_text_len = dataset_params.max_text_len
        self._generate_evaluation_data = dataset_params.generate_evaluation_data
        self._config_params = config_params

    def perform(self, ciphertexts_with_labels):
        features = []
        labels = []

        config = self._config_params
        test_data = []

        for ciphertext_line, label in ciphertexts_with_labels:
            processed_line = self._preprocess_ciphertext_line(ciphertext_line)
            label_index = config.cipher_types.index(label)
            labels.append(label_index)
            if config.feature_engineering:
                feature = calculate_statistics(processed_line)
                features.append(feature)
            else:
                features.append(processed_line)
            if self._generate_evaluation_data:
                test_data.append(processed_line)
            
        if config.pad_input:
            features = pad_sequences(features, maxlen=self._max_text_len)
            features = features.reshape(features.shape[0], features.shape[1], 1)
        
        if self._generate_evaluation_data:
            return EvaluationBatch("rotor", features, labels, test_data)
        else:
            return TrainingBatch("rotor", features, labels)
    
    def _preprocess_ciphertext_line(self, ciphertext_line):
        cleaned = ciphertext_line.strip().replace(' ', '').replace('\n', '')
        mapped = self._map_text_into_numberspace(cleaned.lower())
        return mapped
    
    def _map_text_into_numberspace(self, text):
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        result = []
        for index in range(len(text)):
            try:
                result.append(alphabet.index(text[index]))
            except ValueError:
                raise Exception(f"Ciphertext contains unknown character '{text[index]}'. "
                                f"Known characters are: '{alphabet}'.")
        return result

class PlaintextLine2CipherStatisticsWorker:
    """This class takes paths to plaintext files and provides an iterator 
    interface, which will return batches of statistics and labels for training
    of a classifier. Internally it will encrypt the lines of the plaintext files
    with the given `cipher_types` and calculate the features for the encrypted 
    lines.
    Each `dataset_worker` will calculate the statistics for `batch_size` lines of the
    encrypted lines. Therefore the output of the __next__ method will return lists of length 
    `dataset_workers`. Each item in the list is of type `TrainingBatch`.
    """
    def __init__(self, dataset_params, config_params):
        self._keep_unknown_symbols = dataset_params.keep_unknown_symbols
        self._cipher_types = dataset_params.cipher_types
        self._max_text_len = dataset_params.max_text_len
        self._generate_evaluation_data = dataset_params.generate_evaluation_data
        self._config_params = config_params

    def perform(self, plaintexts):
        batch = []
        labels = []

        config = self._config_params
        ciphertexts = []

        for line in plaintexts:
            for cipher_type in self._cipher_types:
                index = config.cipher_types.index(cipher_type)
                label = self._cipher_types.index(cipher_type)
                if isinstance(config.key_lengths[label], list):
                    key_lengths = config.key_lengths[label]
                else:
                    key_lengths = [config.key_lengths[label]]
                for key_length in key_lengths:
                    try:
                        ciphertext = encrypt(line, index, key_length, self._keep_unknown_symbols)
                    except:
                        multiprocessing_logger.error(f"Could not encrypt line with cipher "
                                                     f"'{cipher_type}'. and key length {key_length}. "
                                                     f"Skipping line...")
                        continue
                    if config.feature_engineering:
                        statistics = calculate_statistics(ciphertext)
                        batch.append(statistics)
                    else:
                        batch.append(list(ciphertext))
                    if self._generate_evaluation_data:
                        ciphertexts.append(ciphertext)
                    labels.append(label)

        if config.pad_input:
            batch = pad_sequences(batch, maxlen=self._max_text_len)
            batch = batch.reshape(batch.shape[0], batch.shape[1], 1)

        # multiprocessing_logger.info(f"Batch: '{batch}'; labels: '{labels}'.")
        if self._generate_evaluation_data:
            return EvaluationBatch("aca", batch, labels, ciphertexts)
        else:
            return TrainingBatch("aca", batch, labels)

class TrainingBatch:
    """Encapsulates the calculates statistics (features) and their labels
    for inputs."""

    def __init__(self, cipher_type, statistics, labels):
        assert len(statistics) == len(labels), "Number of statistics (features) and labels must match!"

        self.cipher_type = cipher_type
        if isinstance(statistics, tf.Tensor):
            self.statistics = statistics
        else:
            self.statistics = tf.convert_to_tensor(statistics)
        if isinstance(labels, tf.Tensor):
            self.labels = labels
        else:
            self.labels = tf.convert_to_tensor(labels)

    def __len__(self):
        return len(self.statistics)
    
    def extend(self, other):
        if not isinstance(other, TrainingBatch):
            raise Exception("Can only extend TrainingBatch with other TrainingBatch instances")
        if len(self.statistics) == 0:
            self.statistics = other.statistics
        else:
            self.statistics = tf.concat([self.statistics, other.statistics], 0)
        if len(self.labels) == 0:
            self.labels = other.labels
        else:
            self.labels = tf.concat([self.labels, other.labels], 0)

    def tuple(self):
        return (self.statistics, self.labels)
    
    @staticmethod
    def represent_equal_cipher_type(training_batches):
        """Checks whether all batches in the training_batches list have the same cipher type"""
        if len(training_batches) <= 1:
            return True
        first_batch = training_batches[0]
        for training_batch in training_batches[1:]:
            if training_batch.cipher_type != first_batch.cipher_type:
                return False
        return True
    
    @staticmethod
    def combined(training_batches):
        """Takes lists of `TrainingBatch`es and combines them into one large `TrainingBatch`."""
        result = TrainingBatch("mixed", [], [])

        for training_batch in training_batches:
            result.extend(training_batch)

        return result

    @staticmethod
    def paired_cipher_types(training_batches):
        """Takes a list of `TrainingBatch`es and pairs batches with aca ciphers and
        rotor ciphers. The resulting list therefore contains lists with two elements each."""
        result = []

        aca_batches = filter(lambda batch: batch.cipher_type == "aca", training_batches)
        rotor_batches = filter(lambda batch: batch.cipher_type == "rotor", training_batches)

        for aca_batch, rotor_batch in zip(aca_batches, rotor_batches):
            result.append([aca_batch, rotor_batch])

        return result
    
class EvaluationBatch(TrainingBatch):
    """Subclass of `TrainingBatch` adding `ciphertexts` as a property. This property is 
    used in eval.py with architectures that take a feature-learning approach."""

    def __init__(self, cipher_type, statistics, labels, ciphertexts):
        super().__init__(cipher_type, statistics, labels)
        assert len(statistics) == len(ciphertexts), "Number of ciphertexts must match length of labels and statistics!"
        self.ciphertexts = ciphertexts

    def extend(self, other):
        if not isinstance(other, EvaluationBatch):
            raise Exception("Can only extend EvalTrainingBatch with other EvalTrainingBatch instances")
        
        super().extend(other)

        if len(self.ciphertexts) == 0:
            self.ciphertexts = other.ciphertexts
        else:
            self.ciphertexts = self.ciphertexts + other.ciphertexts
    
    def tuple(self):
        return (self.statistics, self.labels, self.ciphertexts)
    
    @staticmethod
    def combined(training_batches):
        """Takes lists of `TrainingBatch`es and combines them into one large `TrainingBatch`."""
        result = EvaluationBatch("mixed", [], [], [])

        for training_batch in training_batches:
            result.extend(training_batch)

        return result