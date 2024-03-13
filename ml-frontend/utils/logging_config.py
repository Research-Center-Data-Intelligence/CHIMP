from sys import stdout
from os import environ, path, getcwd, makedirs

import logging

from .messaging import MessagingLoggingHandler


def configure_logging(app):
    messaging_handler = MessagingLoggingHandler()

    module_logger = _create_module_logger()
    module_logger.setLevel(environ.get('logging-level', logging.DEBUG))
    module_logger.addHandler(messaging_handler)

    app.logger.setLevel(environ.get('logging-level', logging.DEBUG))
    app.logger.addHandler(module_logger)
    werkzeug_logger = logging.getLogger("werkzeug")
    werkzeug_logger.setLevel(logging.INFO)
    werkzeug_logger.addHandler(messaging_handler)
    socketio_logger = logging.getLogger("socketio")
    socketio_logger.setLevel(logging.INFO)
    socketio_logger.addHandler(messaging_handler)
    engineio_logger = logging.getLogger("engineio")
    engineio_logger.setLevel(logging.INFO)
    engineio_logger.addHandler(messaging_handler)


def _create_module_logger():
    logger = logging.getLogger(environ.get('logger-name', 'chimp-ml-frontend'))

    formatter = logging.Formatter(fmt='[%(levelname)s] %(asctime)s - %(message)s', datefmt='%d/%m/%Y %I:%M:%S %p')
    logger.addHandler(_get_stdout_handler(formatter))
    logger.addHandler(_get_file_handler(formatter))

    return logger


def _get_stdout_handler(formatter):
    handler = logging.StreamHandler(stdout)
    handler.setFormatter(formatter)
    handler.setLevel(logging.DEBUG)

    return handler


def _get_file_handler(formatter):
    directory = environ.get('logging-dir', path.join(getcwd(), 'logs'))
    makedirs(directory, exist_ok=True)

    handler = logging.FileHandler(filename=path.join(directory, 'ml-frontend.logs'), mode='a')
    handler.setFormatter(formatter)
    handler.setLevel(logging.DEBUG)

    return handler
