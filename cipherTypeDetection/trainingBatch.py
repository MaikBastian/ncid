import tensorflow as tf


class TrainingBatch:
    """
    Encapsulates statistics (features) and their labels.

    A training batch provides the statistics and labels via it's
    `items()` method. The size of the training batches should 
    match the user's input in the commandline arguments.
    """

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
        """Returns the number of entries (labels and statistics pairs) in this batch."""
        
        return len(self.statistics)

    def extend(self, other):
        """Adds the statistics and labels of `other` to self."""

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

    def items(self):
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
    used in `eval.py` with architectures that take a feature-learning approach."""

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

    def items(self):
        return (self.statistics, self.labels, self.ciphertexts)

    @staticmethod
    def combined(training_batches):
        """Takes lists of `TrainingBatch`es and combines them into one large `TrainingBatch`."""
        result = EvaluationBatch("mixed", [], [], [])

        for training_batch in training_batches:
            result.extend(training_batch)

        return result