import tempfile
from os import path

import numpy as np
from numpy.random import RandomState

from mlflow import log_artifact


def split_data(data, fraction: float, random_state: RandomState):
    """
    Splits the data two sets of unequal proportions, in a deterministic manner.

    :param data: The data to split into two sets.
    :param fraction: The proportion of the data split. Must be a number between 0 and 1.
    :param random_state: The random state that determines how the data will be split.

    :return: Returns two sets of data, the first one being of the size of the given proportions, and the other one being
    the remaining data.
    """
    mask = random_state.random(len(data['image_data'])) < fraction

    return _apply_mask(data, mask), _apply_mask(data, ~mask)


def _apply_mask(data, mask):
    """
    A utility function that filters the data object based on the given mask. The mask and the data object need to be of
    equal lengths.

    :param data: The data object which to filter. Needs to have the following dictionary keys: 'image_data', 'class_',
    'category'.
    :param mask: The mask to apply to the data object.

    :return: Returns the data filtered by the given mask.
    """
    data = data.copy()
    data['image_data'] = data['image_data'][mask]
    data['class_'] = data['class_'][mask]
    data['category'] = data['category'][mask]

    return data


def save_data_object(data_object: dict, artifact_path: str):
    """
    Utility method for the emotion recognition module to save data as an artifact to the MLFlow artifact server. Splits
    data into the prediction data (image), labels (emotions), and encoding for the labels (numeric identification of
    emotion).

    :param data_object: A dictionary with data partials (predicted data, labels, encoded labels) to save into the MLFlow
    artifact server.
    :param artifact_path: The path at which to store the data partials as artifacts.
    """
    for data_entry_key in data_object.keys():
        _save_data_item(data_object[data_entry_key], artifact_filename=data_entry_key, artifact_path=artifact_path)


def _save_data_item(data_item: np.ndarray, artifact_filename, artifact_path):
    """
    Utility method for the emotion recognition module to save individual data partials (images, labels, etc.) into the
    MLFlow artifact server as a npy file.

    :param data_item: The individual data partial to store to the MLFlow server.
    :param artifact_filename: The file name with which to store the data partial artifact.
    :param artifact_path: The path at which to store the data partial artifact.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        local_file = path.join(tmpdir, f"{artifact_filename}.npy")

        np.save(file=local_file, arr=data_item)
        log_artifact(local_file, artifact_path)
