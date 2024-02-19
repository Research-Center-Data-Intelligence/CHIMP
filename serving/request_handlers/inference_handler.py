from logic.inference import InferenceManager

from flask import request, abort, jsonify

_inference_manager = InferenceManager()


def _infer_from_model():
    id = request.args.get('id', default='', type=str)
    stage = request.args.get('stage', default='production', type=str)

    if not request.is_json:
        return abort(400, "The request data must be a json object.")
    elif not request.json.get('inputs'):
        return abort(400, "The json request must contain an 'inputs' field with an array of input data.")

    try:
        if not id:
            predictions = _inference_manager.infer_from_global_model(request.json, stage)
            return jsonify(predictions=predictions)
        else:
            predictions = _inference_manager.infer_from_calibrated_model(id, request.json)
            return jsonify(predictions=predictions)
    except TypeError:
        return abort(400, "The data in the 'inputs' field of the json request is formatted incorrectly.")


def add_as_route_handler(app):
    global _infer_from_model

    _ping = app.route('/invocations', methods=['POST'])(_infer_from_model)

    return app
