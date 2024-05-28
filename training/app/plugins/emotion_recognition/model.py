import pandas as pd
from collections import Counter
from numpy.random import RandomState
from typing import Dict, List, Union
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
    SpatialDropout2D,
)
from tensorflow.keras.models import (
    model_from_json,
    Sequential,
)
from tensorflow.keras.optimizers import Adam, SGD


class EmotionModelGenerator:
    config: Dict
    data: Union[pd.DataFrame, any]
    train_data: Union[pd.DataFrame, any]
    validation_data: Union[pd.DataFrame, any]
    models: List
    selected_models: List
    np_random: RandomState

    def __init__(self, config: Dict, data: Union[pd.DataFrame, any]):
        self.config = config

        if not data:
            raise RuntimeError("No data provided or data could not be loaded correctly")
        self.data = data

        self.np_random = RandomState(self.config["random_seed"])
        train_data, _ = self._split_data(self.data, self.config["train_test_factor"])
        self.train_data, self.validation_data = self._split_data(
            self.data, self.config["train_validation_factor"]
        )

    def _split_data(self, data: Dict, fraction: float):
        """Splits the data into two sets of unequal proportions, in a deterministic manner."""

        def apply_mask(_data: Dict, _mask):
            _data = _data.copy()
            _data["image_data"] = _data["image_data"][_mask]
            _data["class_"] = _data["class_"][_mask]
            _data["category"] = _data["category"][_mask]
            return _data

        mask = self.np_random.random(len(self.data["image_data"])) < fraction
        return apply_mask(data, mask), apply_mask(data, ~mask)

    def generate(self):
        model = Sequential()

        # Define convolutional layers
        for i, conv_layer in enumerate(self.config["convolutional_layers"]):
            if i == 0:
                model.add(
                    Conv2D(
                        conv_layer["filters"],
                        tuple(conv_layer["kernel"]),
                        padding=conv_layer["padding"],
                        input_shape=(
                            self.config["image_height"],
                            self.config["image_width"],
                            1,
                        ),
                    )
                )
            else:
                model.add(
                    Conv2D(
                        conv_layer["filters"],
                        tuple(conv_layer["kernel"]),
                        padding=conv_layer["padding"],
                    )
                )
            model.add(BatchNormalization())
            model.add(Activation(conv_layer["activation"]))
            model.add(MaxPooling2D(pool_size=conv_layer["max_pooling"]))
            model.add(Dropout(conv_layer["dropout"]))

        # Flattening convolutional output for use in dense layers
        model.add(Flatten())

        # Define dense layers
        for dense_layer in self.config["dense_layers"]:
            model.add(Dense(dense_layer["nodes"]))
            model.add(BatchNormalization())
            model.add(Activation(dense_layer["activation"]))
            model.add(Dropout(dense_layer["dropout"]))

        # Define output using softmax for classification
        model.add(Dense(len(self.config["categories"]), activation="softmax"))

        # Compile model
        learning_rate = self.config["learning_rate"]
        optimizer = (
            Adam(learning_rate=learning_rate)
            if self.config["optimizer"].lower() == "adam"
            else (
                SGD(learning_rate=learning_rate)
                if self.config["optimizer"].lower() == "sgd"
                else Adam(learning_rate=learning_rate)
            )
        )
        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        # Define class weights for model training to account for under-sampled classes
        total_sample = len(self.train_data["class_"])
        class_weights = {
            key: value / total_sample
            for key, value in Counter(self.train_data["class_"]).items()
        }

        # Define model training procedure
        callbacks = [
            ReduceLROnPlateau(
                monitor="val_loss", factor=0.1, patience=2, min_lr=0.00001, mode="auto"
            ),
            EarlyStopping(
                monitor=self.config["early_stopping"]["metric"],
                min_delta=self.config["early_stopping"]["min_delta"],
                patience=self.config["early_stopping"]["patience"],
                mode=self.config["early_stopping"]["mode"],
            ),
        ]

        # Fit model
        history = model.fit(
            epochs=self.config["epochs"],
            x=self.train_data["image_data"],
            y=self.train_data["class_"],
            class_weight=class_weights,
            batch_size=self.config["batch_size"],
            shuffle=True,
            validation_data=(
                self.validation_data["image_data"],
                self.validation_data["class_"],
            ),
            callbacks=callbacks,
        )

        return ((model, history),)
