"""This module contains different machine learning experimentation pipelines, each based on the use of all three
pipeline components:

* DataProcessorABC
* ModelGeneratorABC
* ModelPublisherABC"""

# region Imports
from __future__ import annotations
from typing import Type, Union, Any
from os import environ

from logic.data import DataProcessorABC
from logic.model import ModelGeneratorABC
from logic.publisher import ModelPublisherABC

import pandas as pd

from mlflow import set_tracking_uri, set_experiment, start_run
# endregion


class BasicPipeline:
    """A basic implementation for a machine learning pipeline using a data processor, model processor and model
    publisher component. This implementation is not meant as a production ready pipeline system.

    ...

    Attributes
    ----------
    _data_processor_factory: Type[DataProcessorABC]
        The type used to create a data processor object. Must be an implementation of the 'DataProcessorABC' class.
    _model_processor_factory: Type[ModelGeneratorABC]
        The type used to create a model generator object. Must be an implementation of the 'ModelGeneratorABC' class.
    _model_publisher_factory: Type[ModelPublisherABC]
        The type used to create a model publisher object. Must be an implementation of the 'ModelPublisherABC' class.

    Methods
    -------
    run(data_in: Union[pd.DataFrame, Any] = None) -> list[Any]
        executes the pipeline with the input data, or runs it with dynamically loaded data if the input data is None.
    _log(data_processor: DataProcessorABC, model_generator: ModelGeneratorABC, model_publisher: ModelPublisherABC)
        logs the results of the pipeline components (if run in debug mode)
    """

    def __init__(self, data_processor: Type[DataProcessorABC] = None, model_generator: Type[ModelGeneratorABC] = None,
                 model_publisher: Type[ModelPublisherABC] = None, config: dict = None):
        self._data_processor_factory = data_processor if data_processor is not None else DataProcessorABC
        self._model_processor_factory = model_generator if model_generator is not None else ModelGeneratorABC
        self._model_publisher_factory = model_publisher if model_publisher is not None else ModelPublisherABC
        self._config = config if config is not None else {}

    # region Functions
    def run(self, data_in: Union[pd.DataFrame, Any] = None):
        """Executes a choreographed pipeline, and returns the published model objects.

        :param data_in: the data to be used in the pipeline. Can be None if the data processor dynamically loads in
            data. Type should match what type of data the data processor expects. The default is a dataframe, but can be
            different if (for example) using image data.
        :return: the models that are published according to the model publisher component.
        """

        data_processor = self._data_processor_factory(config=self._config, data=data_in)\
                             .process_data()\
                             .process_features()

        model_generator = self._model_processor_factory(config=self._config, data=data_processor.features)\
                              .generate()\
                              .validate()

        model_publisher = self._model_publisher_factory(config=self._config,
                                                        models=model_generator.models, data=data_processor.features)\
                              .test()\
                              .publish()

        return model_publisher.published_models
    # endregion

    # region Methods
    @staticmethod
    def _log(data_processor: DataProcessorABC, model_generator: ModelGeneratorABC, model_publisher: ModelPublisherABC) \
            -> None:
        """Logs the results of the pipeline components (if run in debug mode)

        :param data_processor: the data processor unit used to process the data.
        :param model_generator: the model generator used to generate models.
        :param model_publisher: the model publisher used to test and publish model.
        :return: None
        """

        if __debug__:
            print(f"Processed data:\n {data_processor.data} \n")
            print(f"Processed and selected features:\n {data_processor.features}\n")
            print(f"Models generated:\n {model_generator.models}\n")
            print(f"Models tested:\n {model_publisher.models}\n")
            print(f"Models published:\n {model_publisher.published_models}\n")
    # endregion


class MLFlowPipeline(BasicPipeline):
    def __init__(self, data_processor: Type[DataProcessorABC] = None, model_generator: Type[ModelGeneratorABC] = None,
                 model_publisher: Type[ModelPublisherABC] = None, config: dict = None):
        super(MLFlowPipeline, self).__init__(data_processor=data_processor, model_generator=model_generator,
                                             model_publisher=model_publisher, config=config)

        # Set MLFlow config
        self._mlflow_config = self._config['mlflow_config']
        self._mlflow_config['model_name'] = environ.get('MODEL_NAME', self._config['model_name']) \
                                            if environ.get('MODEL_NAME', '') != '' \
                                            else self._config['model_name']
        self._mlflow_config['sub_model_version'] = 0

        set_tracking_uri(environ['MLFLOW_TRACKING_URI'])
        set_experiment(self._config['experiment_name'])

    def run(self, data_in: Union[pd.DataFrame, Any] = None, run_name: str = ''):
        if run_name == '':
            run_name = f"v{self._mlflow_config['base_model_version']}.{self._mlflow_config['sub_model_version']}.0"

        with start_run(run_name=run_name) as run:
            published_models = super(MLFlowPipeline, self).run(data_in)

        self._mlflow_config['sub_model_version'] += 1

        return published_models
