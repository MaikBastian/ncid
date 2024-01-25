import numpy as np
from cipherTypeDetection.featureCalculations import calculate_histogram, calculate_digrams, calculate_cipher_sequence

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
        architecture = self._general_architecture
        if architecture in ("DT", "NB", "RF", "ET", "SVM", "kNN"):
            predictions = self._general_model.predict_proba(statistics)
        else:
            predictions = self._general_model.predict(statistics, batch_size=batch_size, verbose=verbose)
        
        rotor_cipher_labels = range(56, 61) # TODO: Use information of config

        result = []
        for prediction, ciphertext in zip(predictions, ciphertexts):
            max_prediction = np.argmax(prediction)

            if max_prediction in rotor_cipher_labels:
                # Use _rotor_only_model to correctly differentiate between the different rotor ciphers
                rotor_cipher_statistics = calculate_histogram(ciphertext) + calculate_digrams(ciphertext) + calculate_cipher_sequence(ciphertext)
                rotor_predictions = self._rotor_only_model.predict_proba(rotor_cipher_statistics)

                # Calculate scale factor for the rotor cipher predictions. Since the general models 
                # should be quite accurate in the differentiation between aca and rotor ciphers
                # as a whole, use the ratio of these general models as scale factor.
                combined_percentage_of_aca_predictions = sum(prediction[:56]) # TODO: Use information of config
                combined_percentage_of_rotor_predictions = sum(prediction[56:])
                scale_factor = 1
                if combined_percentage_of_rotor_predictions != 0:
                    scale_factor = combined_percentage_of_aca_predictions / combined_percentage_of_rotor_predictions

                # Take the aca predictions of the _general_model and the rotor predictions
                # from the _rotor_only_model and scale the latter to match the original
                # ratio from the _general_model.
                combined_prediction = []
                for aca_prediction_index in range(56): # TODO: Use information of config
                    combined_prediction.append(prediction[aca_prediction_index])
                for rotor_prediction_index in range(5): # TODO: Use information of config
                    combined_prediction.append(rotor_predictions[rotor_prediction_index] * scale_factor)

                result.append(combined_prediction)
            else:
                result.append(predictions)
        
        return result

        

