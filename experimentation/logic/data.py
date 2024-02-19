"""This module contains all helper classes, interfaces and methods that assist a pipeline object in:

* Loading data
* Processing (new) data
* Selecting and process data features"""

# region Imports
from __future__ import annotations
from typing import Any, Union

from abc import abstractmethod, ABC
import pandas as pd
import numpy as np
# endregion


class DataProcessorABC(ABC):
    """This class is an interface for defining a custom data processing unit.

    ...

    This abstract class is supposed to be run as part of a Pipeline object from the pipeline module, and should be used
    as part of a custom pipeline object definition. The three methods '_load_data()', '_process_data()',
    and '_process_features()' are expected to be explicitly defined, but can be implemented using the 'pass' keyword or
    by returning None. If '_load_data()' returns no value, the data processor expects data to be given from outside the
    class definition.


    Attributes
    ----------
    _config: dict
        the configuration variables passed down from the pipeline. For use in custom implementation of this component to
        dynamically adapt the data processor to the needs of the pipeline.
    _data: Union[pd.DataFrame, np.ndarray, Any]
        a data container with which the data processor has been initialised or which has been loaded using the
        '_load_data()' method.
    _features: Union[pd.DataFrame, np.ndarray, Any]
        a feature container in which the processed features are stored.

    Methods
    -------
    process_data() -> self
        processes the data with using the abstract method '_process_data()'.
    process_features() -> self
        processes the features using the abstract method '_process_features()'
    _load_data() -> Union[pd.DataFrame, np.ndarray, Any]
        an abstract method triggered in the constructor of the class, which defines how data should be loaded if
        no data has been provided to the constructor. Should return a value if no data is provided to the constructor.
        The data returned by this method will be stored in the 'data' attribute of this class.
    _process_data() -> Union[pd.DataFrame, np.ndarray, Any]
        an abstract method triggered in the 'process_data()' function, which defines how data should be processed. A
        returned value from this method will be stored in the 'data' attribute of this class.
    _process_features() -> Union[pd.DataFrame, np.ndarray, Any]
        an abstract method triggered in the 'process_features()' function, which defines how features are extracted from
        the data, and which features should be used to create a model. A returned value from this method will be stored
        in the 'features' attribute of this class.
    """

    # region Attributes
    _data: Union[pd.DataFrame, np.ndarray, Any]
    _features: Union[pd.DataFrame, np.ndarray, Any]

    @property
    def data(self):
        return self._data

    @property
    def features(self):
        return self._features if self._features is not None else self._data
    # endregion

    def __init__(self, config: dict, data: Union[pd.DataFrame, np.ndarray, Any] = None):
        """Initialises the data processor class with either preloaded data in a dataframe or by using a loading scheme
        defined in the '_load_data()' method from the class implementing this abstract class. If no data is passed into
        the constructor, the '_load_data()' method must be defined to return usable data.

        :param config: the configuration variables passed down from the pipeline, to be used to dynamically adapt the
            data processor to the needs of the pipeline.
        :param data: the data to be processed within the pipeline. -- Note: This is assumed to be a dataframe, but can
            be implemented as any type. It will show a warning, as it deviates from the standard. This means image data
            may not be able to adhere to this contract and will show such warning.
        """

        self._config = config
        self._data = self._load_data() if data is None else data
        assert(self._data is not None, "'self.data' attribute is None: The data processor is initialised without any "
                                       "data, nor instructions on how to load data into memory.")

    # region Functions
    def process_data(self) -> DataProcessorABC:
        """Processes the data with using the abstract method '_process_data()'.

        :return: an instance of itself.
        """

        processed_data = self._process_data()
        if processed_data is not None:
            self._data = processed_data

        return self

    def process_features(self) -> DataProcessorABC:
        """Processes the features using the abstract method '_process_features()'

        :return: an instance of itself.
        """

        self._features = self._process_features()
        return self
    # endregion

    # region Methods
    @abstractmethod
    def _load_data(self) -> Union[pd.DataFrame, np.ndarray, Any]:
        """An abstract method required for defining how to dynamically load data into the pipeline. Can be implemented
        as a 'pass' method.

        :return: the data loaded in by the implementation class, which will be stored in the 'data' attribute. Can be
            defined to return None or nothing, if you do not need a way to dynamically load data into the pipeline.
        """

        pass

    @abstractmethod
    def _process_data(self) -> Union[pd.DataFrame, np.ndarray, Any]:
        """An abstract method required for defining how to clean and process data within the pipeline. Can be
        implemented as a 'pass' method.

        :return: the processed version of the data that is cleaned up and processed, which will be stored in the 'data'
            attribute. Can be defined to return None or nothing, in which case the 'data' attribute will not be
            overwritten.
        """

        pass

    @abstractmethod
    def _process_features(self) -> Union[pd.DataFrame, np.ndarray, Any]:
        """An abstract method required for refining features in the data and choosing which features to use. Can be
        implemented as a 'pass' method.

        :return: the chosen and processed feature-set, which will be stored in the 'features' attribute. Can be defined
            to return None or nothing, in which case the 'feature' attribute will refer back to the 'data' attribute.
        """

        pass
    # endregion
