import os
import random
import shutil
import tensorflow as tf
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Dense,
    Flatten,
    MaxPool2D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from typing import List


class Training:
    original_pixel_dir: str
    original_other_dir: str
    train_dir: str
    test_dir: str
    valid_dir: str
    train_dir_pixel: str
    train_dir_other: str
    test_dir_pixel: str
    test_dir_other: str
    valid_dir_pixel: str
    valid_dir_other: str
    dirs: List[str]

    train_batches: ImageDataGenerator
    valid_batches: ImageDataGenerator
    test_batches: ImageDataGenerator
    model: Sequential

    def setup_directories(self, data_dir: str):
        self.original_pixel_dir = os.path.join(data_dir, "pixel")
        self.original_other_dir = os.path.join(data_dir, "other")

        self.train_dir = os.path.join(data_dir, "train")
        self.test_dir = os.path.join(data_dir, "test")
        self.valid_dir = os.path.join(data_dir, "valid")
        self.train_dir_pixel = os.path.join(self.train_dir, "pixel")
        self.train_dir_other = os.path.join(self.train_dir, "other")
        self.test_dir_pixel = os.path.join(self.test_dir, "pixel")
        self.test_dir_other = os.path.join(self.test_dir, "other")
        self.valid_dir_pixel = os.path.join(self.valid_dir, "pixel")
        self.valid_dir_other = os.path.join(self.valid_dir, "other")
        self.dirs = [
            self.train_dir,
            self.test_dir,
            self.valid_dir,
            self.train_dir_pixel,
            self.test_dir_pixel,
            self.valid_dir_pixel,
            self.train_dir_other,
            self.test_dir_other,
            self.valid_dir_other,
        ]

        # Calculate split
        pixel_image_count = len(os.listdir(self.original_pixel_dir))
        other_image_count = len(os.listdir(self.original_other_dir))

        pixel_train_count = int(pixel_image_count * 0.75)
        pixel_test_count = int((pixel_image_count - pixel_train_count) * 0.66)
        other_train_count = int(pixel_image_count * 0.75)
        other_test_count = int((other_image_count - other_train_count) * 0.66)

        # Create directories
        for directory in self.dirs:
            if not os.path.exists(directory):
                os.mkdir(directory)

        # Split dataset into train, test, and validation
        for c in random.sample(os.listdir(self.original_pixel_dir), pixel_train_count):
            shutil.move(
                os.path.join(self.original_pixel_dir, c),
                os.path.join(self.train_dir_pixel, c),
            )
        for c in random.sample(os.listdir(self.original_other_dir), other_train_count):
            shutil.move(
                os.path.join(self.original_other_dir, c),
                os.path.join(self.train_dir_other, c),
            )
        for c in random.sample(os.listdir(self.original_pixel_dir), pixel_test_count):
            shutil.move(
                os.path.join(self.original_pixel_dir, c),
                os.path.join(self.valid_dir_pixel, c),
            )
        for c in random.sample(os.listdir(self.original_other_dir), other_test_count):
            shutil.move(
                os.path.join(self.original_other_dir, c),
                os.path.join(self.valid_dir_other, c),
            )
        for c in random.sample(
            os.listdir(self.original_pixel_dir),
            len(os.listdir(self.original_pixel_dir)),
        ):
            shutil.move(
                os.path.join(self.original_pixel_dir, c),
                os.path.join(self.test_dir_pixel, c),
            )
        for c in random.sample(
            os.listdir(self.original_other_dir),
            len(os.listdir(self.original_other_dir)),
        ):
            shutil.move(
                os.path.join(self.original_other_dir, c),
                os.path.join(self.test_dir_other, c),
            )

    def create_batches(self):
        self.train_batches = ImageDataGenerator(
            preprocessing_function=tf.keras.applications.vgg16.preprocess_input
        ).flow_from_directory(
            directory=self.train_dir,
            target_size=(224, 224),
            classes=["pixel", "other"],
            batch_size=32,
        )
        self.valid_batches = ImageDataGenerator(
            preprocessing_function=tf.keras.applications.vgg16.preprocess_input
        ).flow_from_directory(
            directory=self.valid_dir,
            target_size=(224, 224),
            classes=["pixel", "other"],
            batch_size=32,
        )
        self.test_batches = ImageDataGenerator(
            preprocessing_function=tf.keras.applications.vgg16.preprocess_input
        ).flow_from_directory(
            directory=self.test_dir,
            target_size=(224, 224),
            classes=["pixel", "other"],
            batch_size=32,
        )

    def create_model(self):
        self.model = Sequential(
            [
                Conv2D(
                    filters=32,
                    kernel_size=(17, 17),
                    activation="relu",
                    padding="same",
                    input_shape=(224, 224, 3),
                ),
                MaxPool2D(pool_size=(2, 2), strides=2),
                Conv2D(
                    filters=64, kernel_size=(7, 7), activation="relu", padding="same"
                ),
                MaxPool2D(pool_size=(2, 2), strides=2),
                Conv2D(
                    filters=128, kernel_size=(5, 5), activation="relu", padding="same"
                ),
                Flatten(),
                Dense(units=2, activation="softmax"),
            ]
        )
        self.model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

    def train(self, data_dir: str):
        self.setup_directories(data_dir)
        self.create_batches()
        self.create_model()
        self.model.fit(
            x=self.train_batches,
            validation_data=self.valid_batches,
            epochs=1,
            verbose=2,
        )
        self.model.save(os.path.join(data_dir, "model.keras"))
        return self.model
