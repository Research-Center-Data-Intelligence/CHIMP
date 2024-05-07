from flask import Blueprint, jsonify, Response
from typing import Optional, Tuple
from werkzeug.exceptions import HTTPException
from werkzeug.http import HTTP_STATUS_CODES

bp = Blueprint("errors", __name__)


def error_response(
    status_code: int, message: Optional[str] = None
) -> Tuple[Response, int]:
    """Helper function for returning more meaningful error messages.

    :param status_code: HTTP status code
    :param message: Optional meaningful message explaining the error
    :return: Response object
    """
    if type(status_code) is not int:
        raise RuntimeError("status is not of type int")
    error = HTTP_STATUS_CODES.get(status_code, "Unknown Error")
    payload = {"status-code": status_code, "error": error}
    if message:
        payload["message"] = message
    return jsonify(payload), status_code


@bp.app_errorhandler(HTTPException)
def handle_exception(error) -> Tuple[Response, int]:
    return error_response(error.code, error.description)
