from __future__ import annotations
from flask import Flask
from typing import Dict, Optional


class InferenceManager:
    """Integrates the connection with MLFlow for inference with Flask. The implementation
    of this class is inspired by the Flask-SQLAlchemy plugin, specifically the
    SQLAlchemy class.
    """

    _models = {}

    def __init__(self):
        pass

    def init_app(self, app: Flask) -> None:
        """Initialize a Flask application for use with this extension instance.

        Parameters
        ----------
        app : Flask
            The Flask application to initialize with this extension instance.

        Raises
        ------
        RuntimeError
            Raises a RuntimeError when an instance of this extension is already registered.
        """
        if "inference_manager" in app.extensions:
            raise RuntimeError(
                "A 'InferenceManager' instance has already been registered on this Flask app."
            )
        app.extensions["inference_manager"] = self

    def infer(
        self,
        model_name: str,
        data: Dict,
        stage: str = "production",
        id: Optional[str] = None,
    ) -> Dict:
        """ """
        return {"a": 0.9, "b": 0.1}
