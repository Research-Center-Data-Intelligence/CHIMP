from os import path, getcwd, makedirs
from shutil import rmtree as remove_directory
from threading import Thread
from zipfile import ZipFile
import json

from flask import Response, request, abort, jsonify
from werkzeug.utils import secure_filename

from logic.emotionrecognition.pipelines import build_emotion_recognition_pipeline


# region Training
PRINT_TENSORFLOW_INFO = True


class TrainingPipelineSingleton:
    _is_training: bool = False
    _thread = None
    pipeline = None

    def invoke(self):
        if self.pipeline is None:
            self._load_pipeline()

        if not self._is_training:
            self._thread = Thread(target=self._invoke_async, daemon=True)
            self._thread.start()

    def _load_pipeline(self):
        with open(path.join(getcwd(), 'config.json'), 'r') as f:
            self.pipeline = build_emotion_recognition_pipeline(config=json.load(f))

    def _invoke_async(self):
        self._is_training = True

        if PRINT_TENSORFLOW_INFO:
            self._print_tensorflow_info()
        self.pipeline.run()

        self._is_training = False

    @staticmethod
    def _print_tensorflow_info():
        import tensorflow as tf

        print("Version of Tensorflow: ", tf.__version__)
        print("Cuda Availability: ", tf.test.is_built_with_cuda())
        print("GPU  Availability: ", tf.config.list_physical_devices('GPU'))
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


_training_pipeline = TrainingPipelineSingleton()


def _train_model():
    # Make asynchronous pipeline call
    _training_pipeline.invoke()

    return Response(response='Pong!', status=200)
# endregion


# region Calibration
def _calibrate_model():
    def is_file_allowed(fname: str):
        return '.' in fname and fname.rsplit('.', 1)[1].lower() == 'zip'

    # Check if user id defined
    if 'user_id' not in request.args:
        return abort(400, 'No user specified.')

    # Check if files present in request
    if len(request.files) == 0:
        return abort(400, 'No files uploaded.')

    if 'zipfile' not in request.files:
        return abort(400, 'Different file expected.')

    # Check if file is a valid zip
    file = request.files['zipfile']

    if file.filename == '':
        return abort(400, 'No file selected.')
    if not is_file_allowed(file.filename):
        return abort(400, 'File type not allowed. Must be a zip.')

    # Save zip file
    file_name = secure_filename(file.filename)
    folder_path = path.join(getcwd(), 'uploads', request.args.get('user_id'))
    makedirs(folder_path, exist_ok=True)

    file_path = path.join(folder_path, file_name)
    file.save(file_path)

    # Unpack zip file
    with ZipFile(file_path, 'r') as zipfile:
        zipfile.extractall(folder_path)

    # Call calibration upon folder with the given user id
    with open(path.join(getcwd(), 'config.json'), 'r') as f:
        config = json.load(f)
        config['data_directory'] = folder_path

        pipeline = build_emotion_recognition_pipeline(config=config, do_calibrate_base_model=True)
        pipeline.run(run_name='calib'+request.args.get('user_id', '', str))

    # Remove data folder
    remove_directory(config['data_directory'], ignore_errors=True)

    return jsonify(success=True)
# endregion


def add_as_route_handler(app):
    global _train_model, _calibrate_model

    _train_model = app.route('/model/train', methods=['POST'])(_train_model)
    _calibrate_model = app.route('/model/calibrate', methods=['POST'])(_calibrate_model)

    return app
