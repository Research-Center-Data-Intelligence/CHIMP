"""This module contains different machine learning experimentation pipelines for emotion recognition, each based on the
individual component needs for data processing (:class:`logic.data.DataProcessorABC`), model generation
(:class:`logic.model.ModelGeneratorABC`) and model publishing (:class:`logic.publisher.ModelPublisherABC`):

* DataProcessorABC -> :class:`EmotionDataProcessor`, :class:`MLFlowEmotionDataProcessor`,
* ModelGeneratorABC -> :class:`EmotionModelGenerator`, :class:`MLFlowEmotionModelGenerator`,
* ModelPublisherABC -> :class:`EmotionModelPublisher`, :class:`MLFlowEmotionModelPublisher`,"""

from logic.pipeline import BasicPipeline, MLFlowPipeline

from logic.emotionrecognition.data import EmotionDataProcessor, MLFlowEmotionDataProcessor
from logic.emotionrecognition.data import EmotionCalibrationDataProcessor
from logic.emotionrecognition.model import EmotionModelGenerator, MLFlowEmotionModelGenerator
from logic.emotionrecognition.model import EmotionCalibrationModelGenerator
from logic.emotionrecognition.publisher import EmotionModelPublisher, MLFlowEmotionModelPublisher
from logic.emotionrecognition.publisher import EmotionCalibrationModelPublisher


def build_emotion_recognition_pipeline(config: dict, do_calibrate_base_model: bool = False):
    """
    Build a validated emotion recognition pipeline. If 'do_calibrate_base_model' is set to True, instead of the base
    pipeline a calibration pipeline will be returned.

    :param config: the configuration with which to build the pipeline. The configuration allows for specification of
    certain aspects of the pipeline.
    :param do_calibrate_base_model: Set to True if you want to create a model more specifically attuned to new data
    based on a currently published model. Only applies when 'use_mlflow' is true in the configurations.

    :return: Returns a validated emotion recognition pipeline.
    """

    if config['use_mlflow']:
        if not do_calibrate_base_model:
            return MLFlowPipeline(config=config, data_processor=MLFlowEmotionDataProcessor,
                                  model_generator=MLFlowEmotionModelGenerator,
                                  model_publisher=MLFlowEmotionModelPublisher)
        else:
            return MLFlowPipeline(config=config, data_processor=EmotionCalibrationDataProcessor,
                                  model_generator=EmotionCalibrationModelGenerator,
                                  model_publisher=EmotionCalibrationModelPublisher)
    else:
        return BasicPipeline(config=config, data_processor=EmotionDataProcessor,
                             model_generator=EmotionModelGenerator, model_publisher=EmotionModelPublisher)
