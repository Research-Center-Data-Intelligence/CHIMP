"""This module contains all helper classes, interfaces and methods that assist a pipeline object in:

* Testing models
* Publishing models"""

# region Imports
from __future__ import annotations
from typing import Any, Union

from abc import abstractmethod, ABC

import pandas as pd


# endregion


class ModelPublisherABC(ABC):
    """This class is an interface for defining a custom model publishing unit.

    ...

    This abstract class is supposed to be run as part of a Pipeline object from the pipeline module, and should be used
    as part of a custom pipeline object definition. The two methods '_test()' and '_publish()' are expected to be
    explicitly defined, and can be implemented using the 'pass' keyword or by returning None. The models shall be
    provided by the pipeline via an implementation of the 'ModelGeneratorABC'.


    Attributes
    ----------
    _config: dict
        the configuration variables passed down from the pipeline. For use in custom implementation of this component to
        dynamically adapt the model publisher to the needs of the pipeline.
    _data: Union[pd.DataFrame, Any]
        a data container with which models can be tested. It can be expected that the data is already cleaned and
        processed. The assumption is that the data is given in the form of a dataframe, but this can be different
        depending on the implementation class used for 'DataProcessorABC' in the pipeline.
    _models: Any
        a model container in which a list of generated models are stored that can be tested and published. It can be
        expected that the models are already trained and pre-selected based on validation scores.
    _published_models: Any
        a model container in which a list of published models are stored. This can be used as a reference to keep the
        identities of published models in-memory.

    Methods
    -------
    test() -> self
        tests selected models using the abstract method '_test_models()' to test against outside data.
    publish() -> self
        publish selected models using the abstract method '_publish_models()' to publish one or more models.
    _test_models() -> list[Any]
        an abstract method triggered in the 'test()' function, which defines how models should be tested. A
        returned value from this method will overwrite in the 'models' attribute of this class.
    _publish_models() -> list[Any]
        an abstract method triggered in the 'publish()' function, which defines how models should be published (either
        locally or via other methods). A returned value from this method will be stored in the 'published_models'
        attribute of this class.
    """

    # region Attributes
    _data: Union[pd.DataFrame, Any]
    _models: list[Any]
    _published_models: list[Any]

    @property
    def models(self):
        return self._models

    @property
    def published_models(self):
        return self._models if len(self._published_models) == 0 else self._published_models
    # endregion

    def __init__(self, config: dict, models: list[Any], data: Union[pd.DataFrame, Any] = None):
        """Initialises the model publisher class with trained and preselected models, and test data.

        :param config: the configuration variables passed down from the pipeline, to be used to dynamically adapt the
            model publisher to the needs of the pipeline.
        :param models: a list of models which to test and publish.
        :param data: the input data from the Pipeline to test models with. -- Note: This can be of another type than
            dataframe, which might occur if the 'DataProcessorABC' is attuned to (for example) image data. Refer to the
            used 'DataProcessorABC' class if a type-related error seems to occur.
        """

        self._config = config

        self._data = data
        self._models = models
        self._published_models = []

    # region Functions
    def test(self) -> ModelPublisherABC:
        """Tests models using the abstract method '_test_models()'. Only returns results if there is usable test data.

        :return: an instance of itself.
        """
        if self._data is not None:
            results = self._test_models()

            if results is not None:
                self._models = results

        return self

    def publish(self) -> ModelPublisherABC:
        """Publishes new models using the abstract method '_publish_models()'.

        :return: an instance of itself.
        """

        published_models = self._publish_models()
        if published_models is not None:
            self._published_models = published_models

        return self
    # endregion

    # region Methods
    @abstractmethod
    def _test_models(self) -> Union[Any, list[Any]]:
        """An abstract method required for defining how to test models within the pipeline. Can be implemented as a
        'pass' method.

        :return: the models and their test results generated by the method, which will be store in the 'models'
        attribute.
        """

        pass

    @abstractmethod
    def _publish_models(self) -> Union[Any, list[Any]]:
        """An abstract method required for defining how to publish models within the pipeline (either locally or via
        some other method). A subselection can be made to limit the amount of models being published. Recommended is to
        publish only one or two models. Can be also be implemented as a 'pass' method instead.

        :return: the models used in the publishing process, which will be store in the 'published_models' attribute.
        """

        pass
    # endregion
