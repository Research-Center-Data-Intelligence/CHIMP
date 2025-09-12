import warnings
from flask import Blueprint, current_app, jsonify, Response, request, Request
from werkzeug.exceptions import BadRequest, NotFound

from app.errors import ModelNotFoundError, InvalidDataFormatError, InvalidModelIdOrStage

bp = Blueprint("inference", __name__)


@bp.route("/model")
def get_models() -> Response:
    """Get a list of all available models.

    Parameters
    ----------
    (optional query param) reload_models : bool
        Whether or note to reload the models before generating the list of models.
        (this flag might be removed when a message queue based update mechanism is in place)

    Returns
    -------
    json:
        A json object containing a list of available models and a list of loaded models

    Examples
    --------
    curl
        `curl http://localhost:5254/model`
    curl with reloading models
        `curl http://localhost:5254/model?reload_models=true`
    """
    update_models = request.args.get("reload_models", default=False, type=bool)
    if update_models:
        current_app.extensions["inference_manager"].update_models(
            force=True, load_models=True
        )

    data = current_app.extensions["inference_manager"].get_models_list()

    return jsonify(
        {
            "status": "successfully retrieved models",
            "updated_models": update_models,
            "data": data,
        }
    )


@bp.route("/model/<model_name>/infer", methods=["POST"])
def infer_from_model(model_name: str, passed_request: Request = None) -> Response:
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
        ```curl -X POST http://localhost:5254/model/example/infer -H 'Content-Type: application/json'\
         -d '{"inputs": ENCODED_DATA}'```
    curl with model name containing spaces (these are encoded as "+" characters
        ```curl -X POST http://localhost:5254/model/some+model/infer -H 'Content-Type: application/json'\
        -d '{"inputs": ENCODED_DATA}'```
    """
    # This code is required to support the deprecated /invocations endpoint until it is
    # removed.
    current_request = request
    if passed_request:
        current_request = passed_request  # pragma: no cover

    model_name = model_name.replace(
        "+", " "
    )  # Replace URL encoded spaces with actual spaces
    session_id = current_request.args.get("id", default="", type=str)
    stage = current_request.args.get("stage", default="production", type=str)

    if not current_request.is_json:
        raise BadRequest("The request data must be a json object.")
    elif not type(current_request.json) is dict or not current_request.json.get(
        "inputs"
    ):
        raise BadRequest(
            "The json requests must contain an 'inputs' field with an array of input "
            f"data. Got '{current_request.json}'"
        )
    try:
        predictions, metadata = current_app.extensions["inference_manager"].infer(
            model_name, current_request.json.get("inputs"), stage, session_id
        )
    except ModelNotFoundError:
        raise NotFound(f"Could not find model with name {model_name}")
    except InvalidDataFormatError as ex:
        raise BadRequest(str(ex))
    except InvalidModelIdOrStage as ex:
        raise BadRequest(str(ex))

    return jsonify(
        {
            "metadata": metadata,
            "predictions": predictions,
        }
    )
