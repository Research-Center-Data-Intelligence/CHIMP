"""The manage module provides a single entrypoint for both CLI and imports."""

from app.cli import cli

if __name__ == "__main__":
    cli()
else:
    from app import create_app

    app = create_app()
