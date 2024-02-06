import numpy as np
import cipherTypeDetection.config as config
from cipherTypeDetection.featureCalculations import calculate_statistics

class RotorDifferentiationEnsemble:
    """
    This ensemble helps differentiating the rotor ciphers. The best training
    results of the general models are able to differentiate most types of ciphers
    (ACA and rotor ciphers). But rotor ciphers are often mistaken for one of the other
    rotor ciphers.
    This ensemble combines a general model (possible an ensemble as well) with an 
    architecture that is trained on rotor ciphers only. If the general model predicted 
    a rotor cipher with high probability, the rotor only architecture is used to 
    differentiate between the ciphers. The result is than scaled back to the ratio
    originally predicted by the general model, leaving the probabilities of the
    predicted ACA ciphers intact.
    """

    def __init__(self, general_model_architecture, general_model, rotor_only_model):
        """
        Initializes the rotor differentiation ensemble.

        Parameters
        ----------
        general_model_architecture : str
            The architecture of the general model used to predict all supported 
            cipher types.
        general_model :
            The model used to predict all supported cipher types.
        rotor_only_model :
            The model used to differentiate between the rotor ciphers. This model
            should only be trained on the rotor ciphers.
        """
        self._general_architecture = general_model_architecture
        self._general_model = general_model
        self._rotor_only_model = rotor_only_model

    def predict(self, statistics, ciphertexts, batch_size, verbose=0):
        """
        Takes statistics (or ciphertexts) and performs an initial prediction 
        via `self._general_models`. Furthermore improves the recognition of
        the rotor ciphers by using `self._rotor_only_model`.

        Parameters
        ----------
        statistics : list
            List of statistics to predict via the different models.
        ciphertexts : list
            For models that do not use features, ciphertexts is used to predict
            the cipher classes.
        batch_size : int
            Parameter provided to the underlying models.
        verbose : int
            Controls the logging output of the underlying models.
        
        Returns
        -------
        list
            A list of predictions for each provided statistic (or ciphertext).
        """
        # Get rotor cipher labels from config
        first_rotor_cipher_index = config.CIPHER_TYPES.index(config.ROTOR_CIPHER_TYPES[0])
        rotor_cipher_labels = range(first_rotor_cipher_index, 
                                    first_rotor_cipher_index + len(config.ROTOR_CIPHER_TYPES))
        
        # Perform full prediction for all ciphers
        architecture = self._general_architecture
        if architecture in ("DT", "NB", "RF", "ET", "SVM", "kNN"):
            predictions = self._general_model.predict_proba(statistics)
        elif architecture == "Ensemble":
            predictions = self._general_model.predict(statistics, 
                                                      ciphertexts, 
                                                      batch_size=batch_size, 
                                                      verbose=verbose)
        else:
            predictions = self._general_model.predict(statistics, 
                                                      batch_size=batch_size, 
                                                      verbose=verbose)

        # Adapt each prediction with the more specific predictions of the
        # rotor ciphers provided by the `rotor_only_model`. 
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
        """
        Evaluate the given statistics or ciphertexts with labels and return the
        accuracy as well as the k3-accuracy of the predictions of this ensemble.

        Parameters
        ----------
        statistics : list
            The statistics to predict by this ensemble. 
        ciphertexts : list
            The ciphertexts for prediction of a general model, that does 
            not use a feature engineering approach. Unused if `self._general_model`
            uses a feature engineering approach.
        labels : list
            The labels matching the statistics (or ciphertexts). Used to evaluate
            the predictions of this ensemble.
        batch_size : int
            Parameter for the underlying model(s). 
        verbose : int
            If > 0, logs the resulting accuracy and k3_accuracy to stdout.

        Returns
        -------
        tuple :
            The evaluated accuracy and k3_accuracy.
        """

        correct_all = 0
        correct_k3 = 0
        correct_rotor = 0
        correct_rotor_k3 = 0
        prediction = self.predict(statistics, ciphertexts, batch_size, verbose=verbose)

        for i in range(0, len(prediction)):
            max_3_predictions = np.flip(np.argsort(prediction[i]))[:3]
            if labels[i] == np.argmax(prediction[i]):
                correct_all += 1
            if labels[i] in max_3_predictions:
                correct_k3 += 1
            if labels[i] >= 56 and labels[i] == np.argmax(prediction[i]):
                correct_rotor += 1
            if labels[i] >= 56 and labels[i] in max_3_predictions:
                correct_rotor_k3 += 1

        if verbose >= 1:
            print("Accuracy: %f" % (correct_all / len(prediction)))
            print("k3-accuracy: %f" % (correct_k3 / len(prediction)))
            print(f"Rotor accuracy: {correct_rotor / len(list(filter(lambda label: label >= 56, labels)))}")
            print(f"Rotor k3-accuracy: {correct_rotor_k3 / len(list(filter(lambda label: label >= 56, labels)))}")

        return (correct_all / len(prediction), correct_k3 / len(prediction))
