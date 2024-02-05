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
from cipherTypeDetection.rotorDifferentiationEnsemble import RotorDifferentiationEnsemble
from cipherTypeDetection.transformer import MultiHeadSelfAttention, TransformerBlock, TokenAndPositionEmbedding
from cipherTypeDetection.ensembleModel import EnsembleModel

import pandas as pd


# init fast api
app = FastAPI()
models = {}

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
    # models["Transformer"] = (
    #     tf.keras.models.load_model(
    #         os.path.join(model_path, "t96_transformer_final_100.h5"),
    #         custom_objects={
    #             'TokenAndPositionEmbedding': TokenAndPositionEmbedding,
    #             'MultiHeadSelfAttention': MultiHeadSelfAttention,
    #             'TransformerBlock': TransformerBlock}),
    #     False,
    #     True)
    models["FFNN"] = (
        tf.keras.models.load_model(
            os.path.join(model_path, "ffnn_1000_4000000.h5")
            # os.path.join(model_path, "t128_ffnn_final_100.h5")
            ),
        True,
        False)
    # models["LSTM"] = (
    #     tf.keras.models.load_model(
    #         os.path.join(model_path, "t129_lstm_final_100.h5")),
    #     False,
    #     True)
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
    # # TODO add this in production when having at least 32 GB RAM
    # # with open(os.path.join(model_path, "t99_rf_final_100.h5"), "rb") as f:
    # #     models["RF"] = (pickle.load(f), True, False)
    # with open(os.path.join(model_path, "t128_nb_final_100.h5"), "rb") as f:
    #     models["NB"] = (pickle.load(f), True, False)
    # with open(os.path.join(model_path, "svm-combined"), "rb") as f:
    #     models["SVM-Rotor"] = (pickle.load(f), True, False)
    # with open(os.path.join(model_path, "knn-combined"), "rb") as f:
    #     models["kNN-Rotor"] = (pickle.load(f), True, False)
    # # with open(os.path.join(model_path, "rf-combined"), "rb") as f:
    # #     models["RF-Rotor"] = (pickle.load(f), True, False)
    # # with open(os.path.join(model_path, "mlp-combined"), "rb") as f:
    # #     models["MLP-Rotor"] = (pickle.load(f), True, False)
    with open(os.path.join(model_path, "svm_rotor_only.h5"), "rb") as f:
        models["SVM-Rotor"] = pickle.load(f)

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
    
@app.get("/evaluate/single_line/ciphertext", response_model=APIResponse)
async def evaluate_single_line_ciphertext(ciphertext: str, architecture: List[str] = Query([])):
    # use plural inside function
    architectures = architecture
    try:
        validate_architectures(architectures)
    except ArchitectureError as error:
        return error.response
    
    try:
        # only use architectures that can predict aca ciphers
        architectures = [architecture 
            for architecture in architectures 
            if architecture in ("Transformer", "FFNN", "LSTM", "RF", "NB")]
        
        aca_cipher_types = list(range(56))
        rotor_cipher_types = list(range(56, 61))
        
        if len(architectures) == 0:
            return {}
        elif len(architectures) == 1:
            architecture = architectures[0]
            model, feature_engineering, pad_input = models[architecture]
            config.FEATURE_ENGINEERING = feature_engineering
            config.PAD_INPUT = pad_input
        else:
            model_list = []
            architecture_list = []
            for architecture in architectures:
                model_list.append(models[architecture][0])
                architecture_list.append(architecture)
            architecture = "Ensemble"
            model = EnsembleModel(
                model_list,
                architecture_list,
                "weighted",
                aca_cipher_types) # TODO: Use cipher_indices once newest models are used!
        
        # Embed all models in RotorDifferentiationEnsemble to improve recognition of rotor ciphers
        rotor_only_model = models["SVM-Rotor"]
        model = RotorDifferentiationEnsemble(architecture, model, rotor_only_model)
        architecture = "Ensemble"
        
        all_cipher_types = aca_cipher_types + rotor_cipher_types
        cipher_names = [config.CIPHER_TYPES[cipher_index] for cipher_index in all_cipher_types]

        eval_args = SimpleNamespace(
            ciphertext=ciphertext,
            # todo: needs fileupload first (either set ciphertext OR file, never both)
            file=None,
            ciphers=cipher_names,
            batch_size=128,
            verbose=False
        )
        prediction = cipherEval.predict_single_line(eval_args, model, architecture)
    
        return {"success": True, "payload": prediction}
    except BaseException as e:
        # only use these lines for debugging. Never in production environment due
        # to security reasons!
        import traceback
        traceback.print_exc()
        return JSONResponse({"success": False, "payload": None,
                            "error_msg": repr(e)}, status_code=500)
        # return JSONResponse(None, status_code=500)

