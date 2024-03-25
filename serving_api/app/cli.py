from flask.cli import FlaskGroup

from . import create_app

cli = FlaskGroup(create_app=create_app)


@cli.command()
def ping():
    print("pong")
