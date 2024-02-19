from flask import Response


def _ping():
    return Response(response='Pong!', status=200)


def _health():
    return Response(response='Hello world!', status=200)


def add_as_route_handler(app):
    global _ping, _health

    _ping = app.route('/ping')(_ping)
    _health = app.route('/health')(_health)

    return app
