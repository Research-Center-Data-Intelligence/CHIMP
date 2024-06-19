import os
import shutil
import pathlib
from flask import Flask
from flask.testing import FlaskClient
from io import BytesIO
from tempfile import mkdtemp
from zipfile import ZipFile

from app.datastore import BaseDatastore


class TestDatasetEndpoints:
    """Tests for the dataset endpoints."""

    def test_get_datasets(
        self, app: Flask, client: FlaskClient, datastore: BaseDatastore
    ):
        """Test the get datasets endpoint."""
        resp = client.get("/datasets")
        assert resp.status_code == 200
        assert resp.is_json
        data = resp.get_json()
        assert "status" in data and data["status"] == "successfully retrieved datasets"
        assert "datasets" in data and data["datasets"] == ["TestingDataset"]

        os.mkdir(os.path.join(app.config["DATA_DIRECTORY"], "test_dataset"))
        test_data = BytesIO("test".encode())
        datastore.store_object("test_dataset/test.txt", test_data, "test.txt")
        resp = client.get("/datasets")
        data = resp.get_json()
        assert "test_dataset" in data["datasets"]

    def test_upload_dataset(
        self, app: Flask, client: FlaskClient, datastore: BaseDatastore
    ):
        """Tests for the upload dataset endpoint."""
        # Test setup
        dataset_dir = app.config["DATA_DIRECTORY"]

        tmp_dir = mkdtemp(prefix="CHIMP_TESTING_")
        zip_path = os.path.join(tmp_dir, "test.zip")
        with ZipFile(zip_path, "w") as archive:
            for file_path in pathlib.Path(
                os.path.abspath(os.path.dirname(__file__))
            ).iterdir():
                archive.write(file_path, arcname=file_path.name)

        # No file
        resp = client.post("/datasets")
        assert resp.status_code == 400
        assert resp.get_json()["message"] == "No file in request"

        # File is not a zip
        with open(os.path.join(__file__), "rb") as f:
            resp = client.post(
                "/datasets",
                data={
                    "file": (f, os.path.basename(__file__)),
                    "dataset_name": "NoZipTest",
                },
            )
        assert resp.status_code == 400
        assert resp.get_json()["message"] == "File should be a zip"
        assert "NoZipTest" not in os.listdir(dataset_dir)

        # No dataset name
        with open(zip_path, "rb") as zip_file:
            resp = client.post(
                "/datasets",
                data={
                    "file": (zip_file, os.path.basename(zip_path)),
                },
            )
        assert resp.status_code == 400
        assert (
            resp.get_json()["message"] == "Dataset name ('dataset_name') field missing"
        )

        # Invalid dataset name
        with open(zip_path, "rb") as zip_file:
            resp = client.post(
                "/datasets",
                data={
                    "file": (zip_file, os.path.basename(zip_path)),
                    "dataset_name": "Invalid-@#$_Name",
                },
            )
        assert resp.status_code == 400
        assert (
            resp.get_json()["message"]
            == "Dataset name ('dataset_name') should only contain alphanumeric characters"
        )
        assert "Invalid-@#$_Name" not in os.listdir(dataset_dir)

        # Successful flow
        with open(zip_path, "rb") as zip_file:
            resp = client.post(
                "/datasets",
                data={
                    "file": (zip_file, os.path.basename(zip_path)),
                    "dataset_name": "TestingDataset2",
                },
            )
        assert resp.status_code == 200
        assert resp.is_json
        data = resp.get_json()
        assert "status" in data and data["status"] == "successfully uploaded dataset"
        assert "TestingDataset2" in [
            ds.replace("/", "")
            for ds in datastore.list_from_datastore("", recursive=False)
        ]

        # Duplicate dataset
        with open(zip_path, "rb") as zip_file:
            resp = client.post(
                "/datasets",
                data={
                    "file": (zip_file, os.path.basename(zip_path)),
                    "dataset_name": "TestingDataset",
                },
            )
        assert resp.status_code == 400
        assert (
            resp.get_json()["message"]
            == "Dataset with name 'TestingDataset' already exists"
        )

        # Invalid zip file
        invalid_zip_path = os.path.join(tmp_dir, "invalid.zip")
        with open(invalid_zip_path, "w") as f:
            f.write("This is an invalid zip file")
        with open(invalid_zip_path, "rb") as zip_file:
            resp = client.post(
                "/datasets",
                data={
                    "file": (zip_file, os.path.basename(invalid_zip_path)),
                    "dataset_name": "InvalidZipDataset",
                },
            )
        assert resp.status_code == 400
        assert resp.get_json()["message"] == "Invalid zip file"

        # Test cleanup
        shutil.rmtree(tmp_dir)
