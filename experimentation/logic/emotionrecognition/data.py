"""This module contains all helper classes, interfaces and methods that assist a pipeline object in:

* Loading data
* Processing (new) data
* Selecting and process data features"""

from logic.data import DataProcessorABC
from logic.emotionrecognition.__utilities import save_data_object, split_data

from os import path, listdir
from copy import deepcopy

import cv2
import numpy as np
from numpy.random import RandomState

from mlflow.models import get_model_info
from mlflow.artifacts import download_artifacts


class EmotionDataProcessor(DataProcessorABC):
    """A custom data processing unit for emotion recognition data.

    ...

    Implements the three parent methods '_load_data()', '_process_data()', and '_process_features()' to load and process
    facial emotion data.


    Attributes
    ----------
    _config: dict
        the configuration variables passed down from the pipeline. For use in custom implementation of this component to
        dynamically adapt the data processor to the needs of the pipeline.

    Methods
    -------
    _load_data() -> Union[pd.DataFrame, np.ndarray, Any]
        Loads the emotion image data from the data folder into memory.
    _process_data() -> Union[pd.DataFrame, np.ndarray, Any]
        Processes the data by reshaping the 'image_data' into a numpy array of
        (-1, <image height>, <image width>, <colours>), where image height and image width is stored in the
        configuration file, and the amount of colours is assumed to be 1 (gray-scale).
    _process_features() -> Union[pd.DataFrame, np.ndarray, Any]
        Processes the data features by normalising the individual pixels of each image.
    """

    def _load_data(self):
        """
        Loads the emotion image data from the data folder into memory.

        :return: Returns a dictionary of the loaded data with the keys 'image_data', 'class_', 'category' containing the
        prediction data, encoded label, and label respectively.
        """

        # Store data items in list of independent variables (the image) and target variables (the emotion label;
        #   coded into numerical id)
        data = {
            'image_data': [],
            'class_': [],
            'category': []
        }

        # defined to add new data items to the data dictionary
        def add_data_item(image_data_, class__, category_):
            data['image_data'].append(image_data_)
            data['class_'].append(class__)
            data['category'].append(category_)

        # Iterate each folder named to one of the categories in the data directory and add each data item to the list.
        #   Chosen instead of a build-in image data generator from tensorflow's keras to make it new data less easy to
        #   access as an image for privacy reasons.
        for category in self._config['categories']:
            class_ = self._config['categories'].index(category)
            directory = path.join(self._config['data_directory'], category)

            for image in listdir(directory):
                image_data = cv2.imread(path.join(directory, image), cv2.IMREAD_GRAYSCALE)
                image_data = cv2.resize(image_data, (self._config['image_height'], self._config['image_width']))
                add_data_item(image_data, class_, category)

        return data

    def _process_data(self):
        """
        Processes the data by reshaping the 'image_data' into a numpy array of
        (-1, <image height>, <image width>, <colours>), where image height and image width is stored in the
        configuration file, and the amount of colours is assumed to be 1 (gray-scale).

        :return: Returns a dictionary of the processed data with the keys 'image_data', 'class_', 'category' containing
        the transformed prediction data, encoded label, and label respectively.
        """

        # Reshape data (greyscale shape), transform data into numpy arrays for neural network
        data = self.data
        reshaped_data = np.array(data['image_data'])\
            .reshape((-1, self._config['image_height'], self._config['image_width'], 1))

        data['image_data'] = reshaped_data
        data['class_'] = np.array(data['class_'])
        data['category'] = np.array(data['category'])

        return data

    def _process_features(self):
        """
        Processes the data features by normalising the individual pixels of each image.

        :return: Returns a dictionary of the finalised data with the keys 'image_data', 'class_', 'category' containing
        the normalised prediction data, encoded label, and label respectively.
        """

        # Normalise the data so that each pixel is a value between 0 and 1.
        data = self.data
        data['image_data'] = data['image_data'] / 255

        return data


class MLFlowEmotionDataProcessor(EmotionDataProcessor):
    """A custom data processing unit for saving emotion recognition data.

    ...

    Implements the '_process_features()' method over the base emotion recognition data processor, as to save the
    resulting data into the MLFlow artifact server.

    Methods
    -------
    _process_features() -> Union[pd.DataFrame, np.ndarray, Any]
        Processes the finalised data by uploading it to the MLFlow artifact server.
    """
    def _process_features(self):
        """
        Processes the finalised data by uploading it to the MLFlow artifact server.

        :return: Returns the unmodified dictionary of the finalised data with the keys 'image_data', 'class_',
        'category' containing the normalised prediction data, encoded label, and label respectively.
        """

        data = super(MLFlowEmotionDataProcessor, self)._process_features()

        # Record complete dataset in npy format for parent run
        save_data_object(data, artifact_path='data/complete')

        return data


class EmotionCalibrationDataProcessor(DataProcessorABC):
    """A custom data processing unit for emotion recognition data for model calibration.

    ...

    Implements the three parent methods '_load_data()', '_process_data()', and '_process_features()' to load and process
    facial emotion data.


    Attributes
    ----------
    _config: dict
        the configuration variables passed down from the pipeline. For use in custom implementation of this component to
        dynamically adapt the data processor to the needs of the pipeline.

    Methods
    -------
    _load_data() -> Union[pd.DataFrame, np.ndarray, Any]
        Loads the emotion image data from the data folder into memory, and loads data from the model currently running
        in production based on the configuration files.
    _process_data() -> Union[pd.DataFrame, np.ndarray, Any]
        Processes the data by reshaping the 'image_data' into a numpy array of
        (-1, <image height>, <image width>, <colours>), where image height and image width is stored in the
        configuration file, and the amount of colours is assumed to be 1 (gray-scale).
    _process_features() -> Union[pd.DataFrame, np.ndarray, Any]
        Processes the data features by normalising the individual pixels of each image.
    """

    def _load_data(self):
        """
        Loads the emotion image data from the data folder into memory.

        :return: Returns a dictionary of the loaded data with the keys 'image_data', 'class_', 'category' containing the
        prediction data, encoded label, and label respectively.
        """

        # Store data items in list of independent variables (the image) and target variables (the emotion label;
        #   coded into numerical id)
        data = {
            'image_data': [],
            'class_': [],
            'category': []
        }

        # defined to add new data items to the data dictionary
        def add_data_item(image_data_, class__, category_):
            data['image_data'].append(image_data_)
            data['class_'].append(class__)
            data['category'].append(category_)

        # Iterate each folder named to one of the categories in the data directory and add each data item to the list.
        #   Chosen instead of a build-in image data generator from tensorflow's keras to make it new data less easy to
        #   access as an image for privacy reasons.
        for category in self._config['categories']:
            class_ = self._config['categories'].index(category)
            directory = path.join(self._config['data_directory'], category)

            for image in listdir(directory):
                image_data = cv2.imread(path.join(directory, image), cv2.IMREAD_GRAYSCALE)
                image_data = cv2.resize(image_data, (self._config['image_height'], self._config['image_width']))
                add_data_item(image_data, class_, category)

        # Save how many records are in the training
        self._config['calibration_data_entries'] = len(data['image_data'])

        production_model_info = get_model_info(f'models:/{self._config["model_name"]}/Production')
        artifacts_path = download_artifacts(run_id=production_model_info.run_id,
                                            artifact_path='data',
                                            dst_path=path.join(self._config['data_directory'], 'old_data'))

        def read_data_from_file(data_folder: str):
            return {
                'image_data': np.load(path.join(data_folder, 'image_data.npy'))
                              .reshape((-1, self._config['image_height'], self._config['image_width'])) * 255,
                'class_': np.load(path.join(data_folder, 'class_.npy')),
                'category': np.load(path.join(data_folder, 'category.npy'))
            }

        old_validation_data = read_data_from_file(path.join(artifacts_path, 'validation'))
        old_test_data = read_data_from_file(path.join(artifacts_path, 'test'))

        # Get new data entries for training, validation and test calibration data to new data ratio
        def create_dataset(old_data, ratio, random_state: RandomState, calibration_data: dict = None):
            if calibration_data is None:
                dataset = {
                    'image_data': [],
                    'class_': [],
                    'category': []
                }
            else:
                dataset = deepcopy(calibration_data)

            # Choose random data selection from old data
            data_count = self._config['calibration_data_entries'] * ratio
            data_fraction = data_count / len(old_data['image_data'])
            old_data, _ = split_data(old_data, data_fraction, random_state)

            # Add selected data to calibration data
            dataset['image_data'].extend(old_data['image_data'][:int(data_count)])
            dataset['class_'].extend(old_data['class_'][:int(data_count)])
            dataset['category'].extend(old_data['category'][:int(data_count)])

            return dataset

        np_random_state = RandomState(self._config['random_seed'])

        training_data = create_dataset(old_validation_data, self._config['calibration_data_train_ratio'],
                                       calibration_data=data, random_state=np_random_state)
        validation_data = create_dataset(old_test_data, self._config['calibration_data_validation_ratio'],
                                         calibration_data=data, random_state=np_random_state)
        test_data = create_dataset(old_test_data, self._config['calibration_data_validation_ratio'],
                                   random_state=np_random_state)

        # Combine data partitions and return data
        self._config['calibration_data_entries'] = [0, ]
        self._config['calibration_data_entries'].append(self._config['calibration_data_entries'][-1]
                                                        + len(training_data['image_data']))
        self._config['calibration_data_entries'].append(self._config['calibration_data_entries'][-1]
                                                        + len(validation_data['image_data']))
        self._config['calibration_data_entries'].append(self._config['calibration_data_entries'][-1]
                                                        + len(test_data['image_data']))

        def extend_data_object(data_extension):
            data['image_data'].extend(data_extension['image_data'])
            data['class_'].extend(data_extension['class_'])
            data['category'].extend(data_extension['category'])

        data = {
            'image_data': [],
            'class_': [],
            'category': []
        }

        extend_data_object(training_data)
        extend_data_object(validation_data)
        extend_data_object(test_data)

        return data

    def _process_data(self):
        """
        Processes the data by reshaping the 'image_data' into a numpy array of
        (-1, <image height>, <image width>, <colours>), where image height and image width is stored in the
        configuration file, and the amount of colours is assumed to be 1 (gray-scale).

        :return: Returns a dictionary of the processed data with the keys 'image_data', 'class_', 'category' containing
        the transformed prediction data, encoded label, and label respectively.
        """

        # Reshape data (greyscale shape), transform data into numpy arrays for neural network
        data = self.data
        reshaped_data = np.array(data['image_data'])\
            .reshape((-1, self._config['image_height'], self._config['image_width'], 1))

        data['image_data'] = reshaped_data
        data['class_'] = np.array(data['class_'])
        data['category'] = np.array(data['category'])

        return data

    def _process_features(self):
        """
        Processes the data features by normalising the individual pixels of each image, then save data to mlflow.

        :return: Returns a dictionary of the finalised data with the keys 'image_data', 'class_', 'category' containing
        the normalised prediction data, encoded label, and label respectively.
        """

        # Normalise the data so that each pixel is a value between 0 and 1.
        data = self.data
        data['image_data'] = data['image_data'] / 255

        # Send data to mlflow
        def save_data_partition(artifact_path, dataset, start, stop):
            dataset = {
                'image_data': dataset['image_data'][start:stop],
                'class_': dataset['class_'][start:stop],
                'category': dataset['category'][start:stop]
            }

            save_data_object(dataset, artifact_path=artifact_path)

        partition_splits = self._config['calibration_data_entries']
        save_data_partition('data/training', data, partition_splits[0], partition_splits[1])
        save_data_partition('data/validation', data, partition_splits[1], partition_splits[2])
        save_data_partition('data/test', data, partition_splits[2], partition_splits[3])

        return data
