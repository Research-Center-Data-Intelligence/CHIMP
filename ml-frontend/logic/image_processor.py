import cv2
import gevent
from gevent import Greenlet
import numpy as np
from os import path, getcwd
from time import time
from typing import Optional


from logic.model_inference import FacialEmotionInference


class ImageProcessor:
    font: int = cv2.FONT_HERSHEY_SIMPLEX
    image: np.array
    emotion_inference: FacialEmotionInference = FacialEmotionInference()
    _predictions: dict = {}
    _status: dict = {}

    @property
    def predictions(self):
        return list(self._predictions.values())
    
    @property
    def status_msg(self):
        return list(self._status.values())

    def __init__(self, inference_interval: int):
        self.is_processed = True
        self.inference_interval = inference_interval
        self._previous_inference_call = 0
        self._inference_thread: Optional[Greenlet] = None

        # Load facial recognition haar cascade
        cascade_file = path.join(getcwd(), 'static', 'cascades', 'frontalface_default_haarcascade.xml')
        self.face_cascade = cv2.CascadeClassifier(cascade_file)

    def load_image(self, binary_blob: str):
        self.image = cv2.imdecode(np.frombuffer(binary_blob, np.uint8), -1)
        self.is_processed = False

        return self

    def process(self, user_id: str = ''):
        if self.is_processed:
            return

        # Get gray-scale version of the image, detect each face, and get for each face an emotion prediction.
        grey_frame = cv2.cvtColor(self.image, cv2.COLOR_RGBA2GRAY)
        faces = self.face_cascade.detectMultiScale(grey_frame, 1.3, 5)

        # Flag for a new inference call if no inference call is active and the inference time-out interval has expired
        current_time = time()
        time_diff_over = current_time - self._previous_inference_call > self.inference_interval
        thread_alive = False
        if self._inference_thread:
            thread_alive = self._inference_thread.ready()
        do_new_inference_call = time_diff_over and not thread_alive

        for index, (x, y, width, height) in enumerate(faces):
            # Execute inference call if inference call has passed, else use previous results
            if do_new_inference_call:
                def execute_inference():
                    face = cv2.resize(grey_frame[y:y+height, x:x+width], (96, 96))
                    prediction, status = self.emotion_inference.predict(face, user_id)
                    self._predictions[index] = prediction
                    self._status[index] = status

                    # Set inference call time to current time
                    self._previous_inference_call = time()
                    return

                self._inference_thread = gevent.spawn(execute_inference)

            if index not in self._predictions:
                continue

            # For the current face, highlight it with a rectangle and write the most likely emotion above their face
            cv2.putText(self.image, self._predictions[index][0][0], (x, y-10), self.font, .85, (47, 47, 255), 2)
            cv2.rectangle(self.image, (x, y), (x+width, y+height), (192, 192, 0), 1)

        self.is_processed = True

    def get_image_blob(self) -> str:
        # Get a jpg blob of the image, in string format
        return cv2.imencode('.jpg', self.image)[1].tostring()
