import numpy as np
from cipherTypeDetection.textLine2CipherStatisticsDataset import calculate_histogram
from cipherTypeDetection.textLine2CipherStatisticsDataset import calculate_digrams
from cipherTypeDetection.textLine2CipherStatisticsDataset import calculate_cipher_sequence

# TODO: Do not hard code here!
rotor_classes = ["Enigma", "M209", "Purple", "Sigaba", "Typex"]
number_of_rotor_classes = 5

class RotorCipherEnsemble:
    def __init__(self, models):
        self.models = models
        
    def predict_single_line(self, ciphertext_line):
        features = [calculate_histogram(ciphertext_line) +
                    calculate_digrams(ciphertext_line) + 
                    calculate_cipher_sequence(ciphertext_line)]
        prediction = [0] * number_of_rotor_classes
        for model in self.models:
            model_prediction = model.predict_proba(features)
            prediction = np.add(prediction, model_prediction[0])
            
        # divide the combined predictions by the number of models
        prediction = np.divide(prediction, np.full(number_of_rotor_classes, len(self.models)))
        # multiply by 100 
        prediction = np.multiply(prediction, np.full(number_of_rotor_classes, 100))
        
        # map the predictions to a dictionary containing the probability for each
        # rotor cipher type
        return { rotor_classes[index]: probability for index, probability in enumerate(prediction) }