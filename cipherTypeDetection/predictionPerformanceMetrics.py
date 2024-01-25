import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, classification_report, confusion_matrix
import cipherTypeDetection.config as config

class PredictionPerformanceMetrics:
    """Helper class that allows for collection and evaluation of prediction metrics
    for an machine learning architecture. It allows for the batched evaluation by
    calling add_prediction multiple times."""

    def __init__(self, *, model_name):
        self.model_name = model_name
        self.true_labels = []
        self.predicted_labels = []

        self.correct = [0]*len(config.CIPHER_TYPES)
        self.correct_k3 = [0] * len(config.CIPHER_TYPES)
        self.total = [0]*len(config.CIPHER_TYPES)
        self.correct_all = 0
        self.correct_all_k3 = 0
        self.total_len_prediction = 0
        self.incorrect = []
        for i in range(len(config.CIPHER_TYPES)):
            self.incorrect += [[0]*len(config.CIPHER_TYPES)]
        
    def add_predictions(self, true_labels, predicted_probabilities):
        """Evaluates the predicted labels of the architecture in comparison with the
        true_labels. The predicted labels should be provided as an array of probabilities
        for each sample."""
        # get the max predicted label from predicted_probabilites
        predicted_labels = []
        for probabilities in predicted_probabilities:
            predicted_labels.append(np.argmax(probabilities))

        # save correct and incorrect predictions per label for evaluation
        for i in range(len(predicted_labels)):
            if true_labels[i] == predicted_labels[i]:
                self.correct_all += 1
                self.correct[true_labels[i]] += 1
            else:
                self.incorrect[true_labels[i]][predicted_labels[i]] += 1
            self.total[true_labels[i]] += 1

            max_3_predictions = np.flip(np.argsort(predicted_probabilities[i]))[:3]
            if true_labels[i] in max_3_predictions:
                self.correct_all_k3 += 1
                self.correct_k3[true_labels[i]] += 1
        
        self.total_len_prediction += len(predicted_probabilities)

        # also save true and predicted labels for evaluation
        self.true_labels.extend(true_labels)
        self.predicted_labels.extend(predicted_labels)

    def print_evaluation(self):
        """Prints several evaluations of the collected metrics."""
        accuracy = accuracy_score(self.true_labels, self.predicted_labels)
        cm = confusion_matrix(self.true_labels, self.predicted_labels, 
                              labels=range(len(config.CIPHER_TYPES)))

        accuracy_per_class = [0] * len(config.CIPHER_TYPES)

        # accuracy_per_class based upon: https://stackoverflow.com/a/65673016
        #
        # Calculate the accuracy for each one of our classes
        for idx in range(len(config.CIPHER_TYPES)):
            # True negatives are all the samples that are not our current GT class (not the current row) 
            # and were not predicted as the current class (not the current column)
            true_negatives = np.sum(np.delete(np.delete(cm, idx, axis=0), idx, axis=1))
            
            # True positives are all the samples of our current GT class that were predicted as such
            true_positives = cm[idx, idx]
            
            # The accuracy for the current class is the ratio between correct predictions to all predictions
            accuracy_per_class[idx] = (true_positives + true_negatives) / np.sum(cm)

        precision = precision_score(self.true_labels, self.predicted_labels, average=None)
        recall = recall_score(self.true_labels, self.predicted_labels, average=None)
        f1 = f1_score(self.true_labels, self.predicted_labels, average=None)
        mcc = matthews_corrcoef(self.true_labels, self.predicted_labels)

        print(f"Metrics of {self.model_name}:\n")

        self._print_correct_predictions_per_cipher()
        self._print_correct_k3_predictions_per_cipher()

        print(f"accuracy: {accuracy}", 
              f"accuracy per class: {accuracy_per_class}", 
              f"precision: {precision}", 
              f"recall: {recall}", 
              f"f1: {f1}", 
              f"mcc: {mcc}", 
              sep="\n")
        
        print(classification_report(self.true_labels, self.predicted_labels, digits=3))

        print(f"Confusion matrix: \n{list(cm)}")
        print("\n\n")
    
    def _print_correct_predictions_per_cipher(self):
        print("Correct predictions per cipher:")
        for i in range(0, len(self.total)):
            if self.total[i] == 0:
                continue
            print('%s correct: %d/%d = %f' % (config.CIPHER_TYPES[i], self.correct[i], self.total[i], self.correct[i] / self.total[i]))

        if self.total_len_prediction == 0:
            t = 'N/A'
        else:
            t = str(self.correct_all / self.total_len_prediction)
        print('Total: %s\n' % t)
    
    def _print_correct_k3_predictions_per_cipher(self):
        print("Correct prediction in top 3:")
        for i in range(0, len(self.total)):
            if self.total[i] == 0:
                continue
            print('%s correct: %d/%d = %f' % (config.CIPHER_TYPES[i], self.correct_k3[i], self.total[i], self.correct_k3[i] / self.total[i]))
        
        if self.total_len_prediction == 0:
            t = 'N/A'
        else:
            t = str(self.correct_all_k3 / self.total_len_prediction)
        print('Total k3: %s\n' % t)
