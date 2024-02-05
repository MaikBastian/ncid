import numpy as np
import cipherTypeDetection.config as config
from cipherTypeDetection.featureCalculations import calculate_statistics

class RotorDifferentiationEnsemble:
    """This ensemble helps differentiating the rotor ciphers. The best training
    results of the general models are able to differentiate most types of ciphers
    (ACA and rotor ciphers). But rotor ciphers are often mistaken for one of the other
    rotor ciphers.
    This ensemble combines a general model (possible an ensemble as well) with an 
    architecture that is trained on rotor ciphers only. If the general model predicted 
    a rotor cipher with high probability, the rotor only architecture is used to 
    differentiate between the ciphers. The result is than scaled back to the ratio
    originally predicted by the general model.
    """

    def __init__(self, general_model_architecture, general_model, rotor_only_model):
        self._general_architecture = general_model_architecture
        self._general_model = general_model
        self._rotor_only_model = rotor_only_model

    def predict(self, statistics, ciphertexts, batch_size, verbose=0):
        # Get rotor cipher labels from config
        first_rotor_cipher_index = config.CIPHER_TYPES.index(config.ROTOR_CIPHER_TYPES[0])
        rotor_cipher_labels = range(first_rotor_cipher_index, 
                                    first_rotor_cipher_index + len(config.ROTOR_CIPHER_TYPES))
        
        # Perform full prediction for all ciphers
        architecture = self._general_architecture
        if architecture in ("DT", "NB", "RF", "ET", "SVM", "kNN"):
            predictions = self._general_model.predict_proba(statistics)
        elif architecture == "Ensemble":
            predictions = self._general_model.predict(statistics, ciphertexts, 
                                                      batch_size=batch_size, verbose=verbose)
        else:
            predictions = self._general_model.predict(statistics, 
                                                      batch_size=batch_size, verbose=verbose)

        result = []
        for prediction, ciphertext in zip(predictions, ciphertexts):
            max_prediction = np.argmax(prediction)

            if max_prediction in rotor_cipher_labels:
                # Use _rotor_only_model to correctly differentiate between the different rotor ciphers
                rotor_cipher_statistics = calculate_statistics(ciphertext)
                rotor_predictions = self._rotor_only_model.predict_proba([rotor_cipher_statistics])[0]

                # Calculate scale factor for the rotor cipher predictions. Since the general models 
                # should be quite accurate in the differentiation between aca and rotor ciphers
                # as a whole, use the ratio of the rotor cipher percentages as scale factor.
                scale_factor = sum(prediction[first_rotor_cipher_index:])

                # Take the aca predictions of the _general_model and the rotor predictions
                # from the _rotor_only_model and scale the latter to match the original
                # ratio from the _general_model.
                combined_prediction = []
                for aca_prediction_index in range(first_rotor_cipher_index): 
                    combined_prediction.append(prediction[aca_prediction_index])
                for rotor_prediction_index in range(len(rotor_cipher_labels)):
                    combined_prediction.append(rotor_predictions[rotor_prediction_index] * scale_factor)

                result.append(combined_prediction)
            else:
                # Ciphertext is probably of a ACA cipher, return the prediction as-is
                result.append(prediction)
        
        return result

    def evaluate(self, statistics, ciphertexts, labels, batch_size, verbose=0):
        correct_all = 0
        prediction = self.predict(statistics, ciphertexts, batch_size, verbose=verbose)
        for i in range(0, len(prediction)):
            if labels[i] == np.argmax(prediction[i]):
                correct_all += 1
        if verbose == 1:
            print("Accuracy: %f" % (correct_all / len(prediction)))
        return correct_all / len(prediction)
