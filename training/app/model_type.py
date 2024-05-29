from enum import Enum


class ModelType(Enum):
    SKLEARN = "sklearn"
    ONNX = "onnx"
    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"
    OTHER = "other"
    NONE = None

    @classmethod
    def get_model_type(cls, type_name: str):
        try:
            return cls[type_name.upper()]
        except KeyError:
            return cls.OTHER
