from os import environ

from random import random
import numpy as np
import requests
from requests.exceptions import ConnectionError
from json import loads


class FacialEmotionInference:
    # TODO: Get emotions from tracking server. Added emotion list as parameters to the model.
    EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

    def __init__(self):
        # Simulate blue-green test; proper implementation would have validation for blue-green test
        self.stage = 'production' if random() < .9 else 'staging'

    def predict(self, image: np.ndarray, model_id: str = ''):
        # Reshape input, and preprocess pixel to value between 0 and 1
        image = np.array(image).reshape((-1, 48, 48, 1)) / 255

        # Post image to inference server
        url = environ['MODEL_INFERENCE_URL'] + f'?id={model_id}&stage={self.stage}'
        headers = {
            'Content-Type': 'application/json'
        }
        json_data = {
            'inputs': image.tolist()
        }
        try:
            response = requests.request('POST', headers=headers, url=url, json=json_data)

            # Unpack and return response ordered from most to least likely emotion
            if response.status_code == 200:
                text_response = loads(response.text)
                predictions = list(text_response['predictions'].values())[0][0]  # Unpack response into list of predictions
                class_responses = zip(self.EMOTIONS, predictions)
                return sorted(class_responses, key=lambda item: item[1], reverse=True)
        except ConnectionError:
            # Not possible to connect to inference server
            return [('', 0.0)]
        except TypeError:
            # Inference server did not have a model loaded, equipped to handle the request
            return [('', 0.0)]

        # Return empty list if no response was found
        return [('', 0.0)]
