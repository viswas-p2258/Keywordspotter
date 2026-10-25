#!/usr/bin/env python3

"""
@author: Jithin Sasikumar

Script for inferencing with new or test data. It implements functionality
to make predictions on the real data from trained model artifact.
"""

import os
import hydra
import mlflow
import numpy as np
from typing import Any, Dict, Optional, Tuple
from tensorflow import keras
from config_dir.configType import KWSConfig
from src import data
from src.exception_handler import NotFoundError, ValueError

class KeywordSpotter:
    def __init__(self, audio_file: str, model_artifactory_dir: str,
                n_mfcc: int, mfcc_length: int, sampling_rate: int,
                display_labels: Optional[Dict[str, str]] = None) -> None:
        """
        Parameters
        ----------
        audio_file: str
            Name of the input audio file.
        model_artifactory_dir: str
            Directory that holds trained model artifacts.
        n_mfcc: int
            Number of MFCCs to return.
        mfcc_length: int
            Length of MFCC features for each audio input.
        sampling_rate: int
            Target sampling rate
        """
        self.audio_file = audio_file
        self.model_artifactory_dir = model_artifactory_dir
        self.n_mfcc = n_mfcc
        self.mfcc_length = mfcc_length
        self.sampling_rate = sampling_rate
        self.display_labels = display_labels or {}

    def predict(self) -> Tuple[str, float]:
        """
        Method to make predictions based on probabilities from the model on the given
        audio file.

        Parameters
        ----------
            None.

        Result
        ------
        predicted_keyword: str
            Predicted keyword from the model as text.
        label_probability: float
            Probability of the predicted keyword.

        Raises
        ------
        DirectoryError: Exception
            If self.model_artifactory_dir does not exist.
        ValueError: Exception
            If predicted_keyword or label_probability is none.
        NotFoundError: Exception
            When an exception is caught by the `try` block.
        """

        try:
            normalized_model_path = os.path.abspath(self.model_artifactory_dir)
            saved_model_file = os.path.join(normalized_model_path, "saved_model.pb")
            if not os.path.isfile(saved_model_file):
                raise NotFoundError(
                    f"TensorFlow SavedModel file not found: {saved_model_file}"
                )

            model: Any = keras.models.load_model(normalized_model_path)

            audio_mfcc: np.ndarray = data.convert_audio_to_mfcc(self.audio_file,
                                                    self.n_mfcc,
                                                    self.mfcc_length,
                                                    self.sampling_rate)
            reshaped_audio_mfcc: np.ndarray = audio_mfcc.reshape(1, self.n_mfcc, self.mfcc_length)
            labels = data.Preprocess().wrap_labels()
            model_output = model.predict(reshaped_audio_mfcc, verbose=0)
            predicted_label: str = labels[np.argmax(model_output)]
            predicted_keyword = self.display_labels.get(predicted_label, predicted_label)
            label_probability: float = float(np.max(model_output))

            if predicted_keyword is None or label_probability is None:
                raise ValueError("Model returned empty predictions!!!")

            return predicted_keyword, round(label_probability, 4)

        except NotFoundError as exc:
            raise NotFoundError(f"Cannot infer from model. Please check the paths and try it again !!! {exc}") from exc

@hydra.main(config_path="../config_dir", config_name="config", version_base="1.2")
def main(cfg: KWSConfig) -> None:
    inference_ = KeywordSpotter(cfg.names.audio_file,
                                cfg.paths.model_artifactory_dir,
                                cfg.params.n_mfcc, 
                                cfg.params.mfcc_length, cfg.params.sampling_rate,
                                dict(cfg.get("display_labels", {})))
    predicted_keyword, label_probability = inference_.predict()
    print(f"Predicted keyword: {predicted_keyword} \n Keyword probability: {label_probability}")

if __name__ == "__main__":
    main()