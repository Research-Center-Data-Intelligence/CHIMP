from app.cli import cli

if __name__ == "__main__":
    cli()
else:
    from app import create_app, create_celery_app

    app = create_app()
    celery_app = create_celery_app(app)
