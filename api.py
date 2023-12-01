#!/usr/bin/env python
import os
import tensorflow as tf
import pickle
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import SparseTopKCategoricalAccuracy

from fastapi import FastAPI, Query, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
from typing import Optional, Any, List
from types import SimpleNamespace

import cipherTypeDetection.eval as cipherEval
import cipherTypeDetection.config as config
from cipherTypeDetection.transformer import MultiHeadSelfAttention, TransformerBlock, TokenAndPositionEmbedding
from cipherTypeDetection.ensembleModel import EnsembleModel
from cipherTypeDetection.rotorCipherEnsemble import RotorCipherEnsemble

import pandas as pd


# init fast api
app = FastAPI()
models = {}
meta_models = {}

# allow cors
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)  # todo: remove later


@app.on_event("startup")
async def startup_event():
    """The models are loaded with hardcoded names. Change in future if multiple models are available."""
    model_path = "data/models"
    models["Transformer"] = (
        tf.keras.models.load_model(
            os.path.join(model_path, "t96_transformer_final_100.h5"),
            custom_objects={
                'TokenAndPositionEmbedding': TokenAndPositionEmbedding,
                'MultiHeadSelfAttention': MultiHeadSelfAttention,
                'TransformerBlock': TransformerBlock}),
        False,
        True)
    models["FFNN"] = (
        tf.keras.models.load_model(
            os.path.join(model_path, "t128_ffnn_final_100.h5")),
        True,
        False)
    models["LSTM"] = (
        tf.keras.models.load_model(
            os.path.join(model_path, "t129_lstm_final_100.h5")),
        False,
        True)
    optimizer = Adam(
        learning_rate=config.learning_rate,
        beta_1=config.beta_1,
        beta_2=config.beta_2,
        epsilon=config.epsilon,
        amsgrad=config.amsgrad)
    for _, item in models.items():
        item[0].compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=[
                "accuracy",
                SparseTopKCategoricalAccuracy(k=3, name="k3_accuracy")])
    # TODO add this in production when having at least 32 GB RAM
    # with open(os.path.join(model_path, "t99_rf_final_100.h5"), "rb") as f:
    #     models["RF"] = (pickle.load(f), True, False)
    with open(os.path.join(model_path, "t128_nb_final_100.h5"), "rb") as f:
        models["NB"] = (pickle.load(f), True, False)
    with open(os.path.join(model_path, "svm-combined"), "rb") as f:
        models["SVM-Rotor"] = (pickle.load(f), True, False)
    with open(os.path.join(model_path, "knn-combined"), "rb") as f:
        models["kNN-Rotor"] = (pickle.load(f), True, False)
    # with open(os.path.join(model_path, "rf-combined"), "rb") as f:
    #     models["RF-Rotor"] = (pickle.load(f), True, False)
    # with open(os.path.join(model_path, "mlp-combined"), "rb") as f:
    #     models["MLP-Rotor"] = (pickle.load(f), True, False)

    with open(os.path.join(model_path, "meta-classifier"), "rb") as f:
        meta_models["Classifier"] = pickle.load(f)
    with open(os.path.join(model_path, "meta-classifier-scaler"), "rb") as f:
        meta_models["Scaler"] = pickle.load(f)

class ArchitectureError(Exception):
    def __init__(self, response):
        Exception.__init__(self)
        self.response = response
        
def validate_architectures(architectures):
    max_architectures = len(models.keys())
    # Warn if the provided number of architectures is out of the expected bounds
    if not 0 < len(architectures) <= max_architectures:
        response = JSONResponse(
            {
                "success": False,
                "payload": None,
                "error_msg": "The number of architectures must be between 1 and %d." %
                max_architectures},
            status_code=status.HTTP_400_BAD_REQUEST)
        raise ArchitectureError(response)
            
    # Warn about duplicate architectures
    if len(set(architectures)) != len(architectures):
        response = JSONResponse({"success": False,
                             "payload": None,
                             "error_msg": "Multiple architectures of the same type are not "
                             "allowed!"},
                            status_code=status.HTTP_400_BAD_REQUEST)
        raise ArchitectureError(response)
    
    # Check if the provided architectures are known
    for architecture in architectures:
        if architecture not in models.keys():
            response = JSONResponse(
                {
                    "success": False,
                    "payload": None,
                    "error_msg": "The architecture '%s' does not exist!" %
                    architecture},
                status_code=status.HTTP_400_BAD_REQUEST)
            raise ArchitectureError(response)

class APIResponse(BaseModel):
    """Define api response model."""
    success: bool = True
    payload: Optional[Any] = {}
    error_msg: Optional[str] = None


@app.exception_handler(Exception)
async def exception_handler(request, exc):
    """Define exception response format."""
    return JSONResponse({"success": False, "payload": None, "error_msg": str(
        exc)}, status_code=status.HTTP_400_BAD_REQUEST)
# TODO: does not work (exceptions are still thrown), specific exceptions
# work -- todo: FIX ;D

# todo: custom error handling for unified error responses and no closing on error
# https://fastapi.tiangolo.com/tutorial/handling-errors/#override-the-default-exception-handlers


@app.get("/get_available_architectures", response_model=APIResponse)
async def get_available_architectures():
    return {"success": True, "payload": list(models.keys())}
    
def predict_with_aca_architectures(ciphertext, architectures):
    # only use architectures that can predict aca ciphers
    architectures = [architecture 
        for architecture in architectures 
        if architecture in ("Transformer", "FFNN", "LSTM", "RF", "NB")]
    
    aca_cipher_types = get_aca_cipher_types_to_use()
    
    if len(architectures) == 0:
        return {}
    elif len(architectures) == 1:
        model, feature_engineering, pad_input = models[architectures[0]]
        cipherEval.architecture = architectures[0]
        config.FEATURE_ENGINEERING = feature_engineering
        config.PAD_INPUT = pad_input
    else:
        cipher_indices = [config.CIPHER_TYPES.index(cipher_type) 
            for cipher_type in aca_cipher_types]
        model_list = []
        architecture_list = []
        for architecture in architectures:
            model_list.append(models[architecture][0])
            architecture_list.append(architecture)
        cipherEval.architecture = "Ensemble"
        model = EnsembleModel(
            model_list,
            architecture_list,
            "weighted",
            cipher_indices)
    
    return cipherEval.predict_single_line(SimpleNamespace(
        ciphertext=ciphertext,
        # todo: needs fileupload first (either set ciphertext OR file, never both)
        file=None,
        ciphers=aca_cipher_types,
        batch_size=128,
        verbose=False
    ), model)
    
def predict_with_rotor_architectures(ciphertext, architectures):
    # only use architectures that can predict rotor machine ciphers
    architectures = [architecture 
        for architecture in architectures 
        if architecture in ("SVM-Rotor", "RF-Rotor", "kNN-Rotor", "MLP-Rotor")]
    if len(architectures) == 0:
        return {}
    rotor_models = [models[architecture][0] for architecture in architectures]
    model_path = "data/models"
    with open(os.path.join(model_path, "scaler"), "rb") as f:
        scaler = pickle.load(f)
    return RotorCipherEnsemble(rotor_models, scaler).predict_single_line(ciphertext)

@app.get("/evaluate/single_line/ciphertext", response_model=APIResponse)
async def evaluate_single_line_ciphertext(ciphertext: str, architecture: List[str] = Query([])):
    # use plural inside function
    architectures = architecture
    try:
        validate_architectures(architectures)
    except ArchitectureError as error:
        return error.response
    
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
    
    try:
        aca_architecture_prediction = predict_with_aca_architectures(ciphertext, architectures)
        rotor_architecture_prediction = predict_with_rotor_architectures(ciphertext, architectures)

        aca_cipher_types = [config.CIPHER_TYPES[cipher_index]
                        for cipher_index in range(56)]
        rotor_cipher_types = ["Enigma", "M209", "Purple", "Sigaba", "Typex"]
        all_cipher_types = aca_cipher_types + rotor_cipher_types

        for cipher in rotor_cipher_types:
            aca_architecture_prediction[cipher] = 0
        for cipher in aca_cipher_types:
            rotor_architecture_prediction[cipher] = 0

        aca_key = 0
        rotor_key = 1

        # Only use aca predictions for meta-classifier, since those results seem to better indicate
        # what type of 'cipher group' (aca or rotor) was actually entered
        aca_feature_dict = features_from_prediction(aca_architecture_prediction, all_cipher_types)
        # rotor_feature_dict = features_from_prediction(rotor_architecture_prediction, all_cipher_types)

        predictions_data_frame = pd.DataFrame([aca_feature_dict])
        # predictions_data_frame = pd.DataFrame([aca_feature_dict, rotor_feature_dict])

        # TODO: Skip meta classifier if only one type of architecture is selected!
        meta_classifier = meta_models["Classifier"]
        meta_scaler = meta_models["Scaler"]

        scaled_prediction = meta_scaler.transform(predictions_data_frame.to_numpy())
        meta_predictions = meta_classifier.predict_proba(scaled_prediction)

        # TODO: Seem to be currently always 100% for either class: Why? Should be ensured, otherwise perfectly good
        # predictions get scaled down in the loop below!
        print(f"Meta predictions: {meta_predictions}")

        aca_cipher_probability = 0
        rotor_cipher_probability = 0
        for prediction in meta_predictions:
            aca_cipher_probability += prediction[0]
            rotor_cipher_probability += prediction[1]
        aca_cipher_probability = aca_cipher_probability / len(meta_predictions)
        rotor_cipher_probability = rotor_cipher_probability / len(meta_predictions)

        print(f"ACA probability: {aca_cipher_probability}, rotor probability: {rotor_cipher_probability}")

        # TODO: Improve: No or only one arch. selected
        # combined_prediction = {}
        # for cipher, probability in aca_architecture_prediction.items():
        #     combined_prediction[cipher] = probability * aca_cipher_probability
        # for cipher, probability in rotor_architecture_prediction.items():
        #     # TODO: Only works because the predicted ciphers are distinct
        #     if cipher in combined_prediction:
        #         existing_prediction = combined_prediction[cipher]
        #         combined_prediction[cipher] = existing_prediction + probability * rotor_cipher_probability
        #     else:
        #         combined_prediction[cipher] = probability * rotor_cipher_probability

        # print(f"Probability of aca cipher: {aca_cipher_probability}, probability of rotor cipher: {rotor_cipher_probability}.")

        new_prediction = {}
        for cipher, probability in aca_architecture_prediction.items():
            if aca_cipher_probability > 0:
                new_prediction[cipher] = probability * aca_cipher_probability
        for cipher, probability in rotor_architecture_prediction.items():
            if rotor_cipher_probability > 0:
                new_prediction[cipher] = probability * rotor_cipher_probability

        return {"success": True, "payload": new_prediction}
        # return {"success": True, "payload": combined_prediction}
    except BaseException as e:
        # only use these lines for debugging. Never in production environment due
        # to security reasons!
        import traceback
        traceback.print_exc()
        return JSONResponse({"success": False, "payload": None,
                            "error_msg": repr(e)}, status_code=500)
        # return JSONResponse(None, status_code=500)


###########################################################

def get_aca_cipher_types_to_use():
    cipher_types = []
    for cipher_index in range(56):
        cipher_types.append(config.CIPHER_TYPES[cipher_index])
    return cipher_types
