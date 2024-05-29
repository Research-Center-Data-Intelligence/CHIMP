"""This module contains all helper classes, interfaces and methods that assist a pipeline object in:

* Generating models
* Validating models"""

# region Imports
from __future__ import annotations
from typing import Any, Union

from abc import abstractmethod, ABC
import pandas as pd
import numpy as np
# endregion


class ModelGeneratorABC(ABC):
    """This class is an interface for defining a custom model generator unit.

    ...

    This abstract class is supposed to be run as part of a Pipeline object from the pipeline module, and should be used
    as part of a custom pipeline object definition. The two methods '_generate()' and '_validate()' are expected to be
    explicitly defined, but the '_validate()' method can be implemented using the 'pass' keyword or by returning None.
    The data shall be provided by the pipeline via an implementation of the 'DataProcessorABC'.


    Attributes
    ----------
    _config: dict
        the configuration variables passed down from the pipeline. For use in custom implementation of this component to
        dynamically adapt the model generator to the needs of the pipeline.
    _data: Union[pd.DataFrame, np.ndarray, Any]
        a data container with which the model generator can start generating models. It can be expected that the data is
        already cleaned and processed. The assumption is that the data is given in the form of a dataframe, but this can
        be different depending on the implementation class used for 'DataProcessorABC' in the pipeline.
    _models: list[Any]
        a model container in which the generated models are stored. If the '_validate()' method is implemented, the
        model selection can be further refined, and will be reflected like so.
    _selected_models: list[Any]
        a model container in which validated models are stored, and a model selection is given.

    Methods
    -------
    generate() -> self
        generates new models using the abstract method '_generate()'. Expects the abstract method to yield objects,
        or to return an iterable.
    validate() -> self
        validates the generated models using the abstract method '_validate()'. Expects the abstract method to yield
        objects or to return an iterable.
    _generate() -> list[Any]
        an abstract method triggered in the 'generate()' function, which defines how models should be generated. This
        method needs to be implemented to return at least a single model, and is required to either return an iterable
        or to be implemented as a generator function. The result will be stored in the 'models' attribute.
    _validate() -> list[Any]
        an abstract method triggered in the 'validate()' function, which defines how models should be validated and how
        the best models should be selected. A returned value from this method will override the 'models' unless
        implemented to return nothing.
    """

    # region Attributes
    _data: Union[pd.DataFrame, Any]
    _models: list
    _selected_models: list

    @property
    def data(self):
        return self._data

    @property
    def models(self):
        return self._models if len(self._selected_models) == 0 else self._selected_models
    # endregion

    def __init__(self, config: dict, data: Union[pd.DataFrame, np.ndarray, any]):
        """Initialises the model generator class with cleaned and processed data in a dataframe. Data can be of a
        different format if using a custom implementation of the 'DataProcessorABC' class.

        :param config: the configuration variables passed down from the pipeline, to be used to dynamically adapt the
            model generator to the needs of the pipeline.
        :param data: the input data from the Pipeline to generate models with. -- Note: This can be of another type than
            dataframe, which might occur if the 'DataProcessorABC' is attuned to (for example) image data. Refer to the
            used 'DataProcessorABC' class if a type-related error seems to occur.
        """
        assert data is not None, "No data was passed into the model generator."

        self._config = config

        self._data = data
        self._models = []
        self._selected_models = []

    # region Functions
    def generate(self) -> ModelGeneratorABC:
        """Generates new models using the abstract method '_generate()'. Expects the abstract method to yield objects,
        or to return an iterable.

        :return: an instance of itself.
        """

        generator = self._generate()
        assert generator is not None, "Implementation of the '_generate()' method does not return a value."

        for model in generator:
            self._models.append(model)

        return self

    def validate(self) -> ModelGeneratorABC:
        """Validates the generated models using the abstract method '_validate()'. Expects the abstract method to yield
        objects or to return an iterable.

        :return: an instance of itself.
        """

        validator = self._validate()

        if validator is not None:
            for model in validator:
                self._selected_models.append(model)

        return self
    # endregion

    # region Methods
    @abstractmethod
    def _generate(self) -> list[Any]:
        """An abstract method required for defining how to generate models within the pipeline. It has to yield at least
        a single model object. This function should either be implemented as a generator model, or return an iterable
        object.

        :return: the models generated by the method, which will be available via the 'models' property. Needs to return
            at least one model.
        """

        pass

    @abstractmethod
    def _validate(self) -> list[Any]:
        """An abstract method required for defining how to validate and select models within the pipeline. It doesn't
        need to return or yield anything and can be implemented as a 'pass' function or to return None. If implemented
        properly this function should either be implemented as a generator model, or return an iterable object.

        :return: the models selected and validated by the method, which will be available via the 'models' property. If
            returning None, all generated models will be available via the 'models' property.
        """

        pass
    # endregion
