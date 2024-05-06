import pytest

from app.errors import InvalidDataFormatError
from app.model import BaseModel


class TestModel:
    """Tests for the model class."""

    def test_model_predict(self, model: BaseModel):
        """Test the model predict method."""
        with pytest.raises(InvalidDataFormatError):
            model.predict(1)

        result = model.predict([1, 2, 3])
        assert type(result) is dict
