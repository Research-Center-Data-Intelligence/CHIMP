from flask import Blueprint, Response

bp = Blueprint("health", __name__)


@bp.route("/ping")
def ping() -> Response:
    return Response("pong", 200)
