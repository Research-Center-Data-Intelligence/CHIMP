import warnings
from flask import Blueprint, current_app, jsonify, Response, request
from werkzeug.exceptions import BadRequest

bp = Blueprint("inference", __name__)


@bp.route("/model/<model_name>/infer", methods=["POST"])
def infer_from_model(model_name: str) -> Response:
    """Get an inference from the given model.

    Parameters
    ----------
    model_name : str
        Name of the model to be used for inference from the request data.
    (optional query param) session_id : str
        Session ID that might denote a specific calibrated model.
    (optional query param) stage : str
        Select whether to use the production or staging model.

    Returns
    -------
    json:
        A json object containing the predictions from the model

    Raises
    ------
    NotFound
        When the specified model name is not found, a NotFound (404) error is raised


    Examples
    --------
    curl
        ```curl -X POST http://localhost:5254/model/example/infer -H 'Content-Type: application/json`\
         -d '{"inputs": ENCODED_DATA}'```
    """
    session_id = request.args.get("id", default="", type=str)
    stage = request.args.get("stage", default="production", type=str)

    if not request.is_json:
        raise BadRequest("The request data must be a json object.")
    elif not request.json.get("inputs"):
        raise BadRequest(
            f"The json requests must contain an 'inputs' field with an array of input data. Got '{request.json}'"
        )

    predictions = current_app.extensions["inference_manager"].infer(
        model_name, request.json, stage, session_id
    )

    return jsonify(
        {"status": f"inference from model {model_name} success", "data": predictions}
    )


@bp.route("/invocations", methods=["POST"])
def invocations():
    """Legacy route to support the old style of inference.

    WARNING: Calling the /invocations endpoint is deprecated, use the /model/<model_name>/infer endpoint instead

    Returns
    -------
    json:
        A json object containing the predictions from the model

    Examples
    --------
    curl
        ```curl -X POST http://localhost:5254/invocations -H 'Content-Type: application/json'\
        -d '{"inputs": ENCODED_DATA}'```

    """
    warnings.warn(
        "Calling the /invocations endpoint is deprecated, use the /model/<model_name>/infer endpoint instead"
    )
    return infer_from_model(current_app.config["LEGACY_MODEL_NAME"])
