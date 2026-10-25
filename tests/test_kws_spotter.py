#!/usr/bin/env python3

"""
@author: Jithin Sasikumar

Script to perform unit testing using `pytest`.
"""

import warnings
import pytest
import numpy as np
from omegaconf import OmegaConf
from src import data
from src.inference import KeywordSpotter
warnings.filterwarnings('ignore')

cfg = OmegaConf.load('./config_dir/config.yaml')

@pytest.fixture
def mfcc() -> np.ndarray:
    """
    Fixture function to convert audio file to MFCC features to be
    used as a global variable in multiple tests.

    Parameters
    ----------
        None.

    Returns
    -------
    mfcc: np.ndarray
        Computed MFCC features
    """
    mfcc_features = data.convert_audio_to_mfcc(cfg.names.audio_file,
                                               cfg.params.n_mfcc,
                                               cfg.params.mfcc_length,
                                               cfg.params.sampling_rate)
    return mfcc_features

def test_label_type() -> None:
    """Function to test the datatype of labels which should be
    `str` in order to be used for training and inference.

    Parameters
    ----------
        None.

    Returns
    -------
        None.
    """
    labels = data.Preprocess().wrap_labels()
    assert all(isinstance(n, str) for n in labels)

def test_mfcc_shape(mfcc: pytest.fixture) -> None:
    """Function to test the shape of computed MFCC features
    from audio files. It is an ndarray whose shape should
    match the parameters(n_mfcc, mfcc_length) from the config.

    Parameters
    ----------
    mfcc: pytest.fixture
        Computed MFCC features

    Returns
    -------
        None.
    """
    assert mfcc.shape == (cfg.params.n_mfcc, cfg.params.mfcc_length)

def test_mfcc_dimension(mfcc: pytest.fixture) -> None:
    """Function to test the dimension of mfcc features array.

    Parameters
    ----------
    mfcc: pytest.fixture
        Computed MFCC features

    Returns
    -------
        None.
    """
    assert len(mfcc.shape) == 2

def test_preprocessed_data_matches_model_input_shape() -> None:
    preprocess = data.Preprocess(
        train_dir=cfg.paths.train_dir,
        n_mfcc=cfg.params.n_mfcc,
        mfcc_length=cfg.params.mfcc_length,
        sampling_rate=cfg.params.sampling_rate,
    )
    dataset = preprocess.preprocess_dataset(
        preprocess.labels,
        cfg.params.test_data_split_percent,
    )

    assert dataset.x_train.ndim == 3
    assert dataset.x_train.shape[1:] == (cfg.params.n_mfcc, cfg.params.mfcc_length)
    assert dataset.x_test.shape[1:] == (cfg.params.n_mfcc, cfg.params.mfcc_length)
    assert dataset.y_train.shape[0] == dataset.x_train.shape[0]
    assert dataset.y_test.shape[0] == dataset.x_test.shape[0]

def test_keyword_spotter_maps_class_to_spoken_word(monkeypatch) -> None:
    class FakeModel:
        def predict(self, features, verbose=0):
            return np.array([[0.1, 0.1, 0.8, 0, 0, 0]])

    monkeypatch.setattr("src.inference.os.path.isfile", lambda path: True)
    monkeypatch.setattr(
        "src.inference.keras.models.load_model",
        lambda path: FakeModel(),
    )
    monkeypatch.setattr(
        "src.inference.data.convert_audio_to_mfcc",
        lambda *args: np.zeros((49, 40)),
    )
    monkeypatch.setattr(
        "src.inference.data.Preprocess.wrap_labels",
        lambda self: ["audio_5", "audio1", "audio2", "audio3", "audio4", "audio6"],
    )

    spotter = KeywordSpotter(
        "sample.wav",
        "model",
        49,
        40,
        16000,
        display_labels={"audio2": "stop"},
    )

    assert spotter.predict() == ("stop", 0.8)