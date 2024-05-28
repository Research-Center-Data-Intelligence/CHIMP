from typing import Optional

from app.plugin import BasePlugin, PluginInfo


class GameArtStyleDetectorPlugin(BasePlugin):
    def __init__(self):
        description = """Game Art Style Detector plugin trains a model to detect
the art style used in a game. Currently it only detects two styles, pixel and
other. For this a CNN is trained using Tensorflow.
        """
        self._info = PluginInfo(
            name="Game Art Style Detector",
            version="1.0",
            description=description,
            arguments={},
            model_return_type="tensorflow",
        )

    def init(self) -> PluginInfo:
        return self._info

    def run(self, *args, **kwargs) -> Optional[str]:
        from . import training

        print(f"Starting {self._info.name}")

        if "data_dir" not in kwargs:
            raise RuntimeError("No data_dir specified")

        trainer = training.Training()
        model = trainer.train(kwargs["data_dir"])
        run_name = self._connector.store_model(
            experiment_name="GameArtDetector",
            model=model,
            model_type="tensorflow",
        )
        return run_name
