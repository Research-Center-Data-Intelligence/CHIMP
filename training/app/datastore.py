import mimetypes
import os
from abc import ABC, abstractmethod
from flask import Flask
from io import BytesIO
from minio import Minio
from minio.error import S3Error
from typing import Dict, List, Optional


class BaseDatastore(ABC):
    """Base class for any Datastores. Datastores are responsible for managing the storage
    and retrieval of datasets."""

    _datastore_uri: str

    @abstractmethod
    def _init_datastore(self):  # pragma: no cover
        """Helper method for any datastore specific initialization."""
        pass

    @abstractmethod
    def list_from_datastore(
        self, target_path: str, recursive: bool = True
    ) -> List[str]:  # pragma: no cover
        """List all items from a given path.

        Parameters
        ----------
        target_path : str
            Path to the datastore to list the items from.
        recursive : bool
            Whether or not to list the items recursively.

        Returns
        -------
        A list of items from the datastore.
        """
        pass

    @abstractmethod
    def store_file_or_folder(self, target_path: str, src_path: str):  # pragma: no cover
        """Store a file or folder in the datastore

        Parameters
        ----------
        target_path : str
            The path to use in the datastore
        src_path : str
            The path of the file to store
        """
        pass

    @abstractmethod
    def store_object(
        self, target_path: str, data: BytesIO, file_name: str, mime_type: str = None
    ):  # pragma: no cover
        """Store a data object in the datastore.

        Parameters
        ----------
        target_path : str
            The path to use in the datastore
        data : BytesIO
            The binary data to store in the datastore
        file_name : str
            The name of the file that is uploaded
        mime_type : str
            The mimetype of the object to store
        """
        pass

    @abstractmethod
    def load_object_to_memory(
        self, object_path: str
    ) -> Optional[BytesIO]:  # pragma: no cover
        """Load an object from the datastore.

        Parameters
        ----------
        object_path: str
            The path to load from the datastore.

        Returns
        -------
        A bytes object, or None if the object could not be found.
        """
        pass

    @abstractmethod
    def load_object_to_file(
        self, object_path: str, save_path: str
    ) -> Optional[str]:  # pragma: no cover
        """Load an object from the datastore and save it to disk.

        Parameters
        ----------
        object_path : str
            The path to load from the datastore.
        save_path : str
            The path of the file to save the object to.

        Returns
        -------
        A string of the path where the file is stored or None if the object could not be found.
        """
        pass

    @abstractmethod
    def load_folder_to_filesystem(
        self, folder_path: str, save_path: str
    ) -> Optional[str]:  # pragma: no cover
        """Load a folder from the datastore and save it to disk.

        Parameters
        ----------
        folder_path : str
            The folder to retrieve from the datastore.
        save_path : str
            The path on the local file system to save the datastore to.

        Returns
        -------
        The path to the directory on the local filesystem, or None if
        the folder could not be found.
        """
        pass

    @abstractmethod
    def load_folder_to_memory(
        self, folder_path: str
    ) -> Optional[Dict[str, BytesIO]]:  # pragma: no cover
        """Load a folder from the datastore to memory.

        Parameters
        ----------
        folder_path : str
            Folder path to retrieve from the datastore.

        Returns
        -------
        A dictionary mapping the resource path to a bytes object, or None if
        the folder could not be found.
        """
        pass

    def init_app(self, app: Flask, datastore_uri: str):
        """Initialize a Flask application for use with this extension instance.

        Parameters
        ----------
        app : Flask
            The Flask application to initialize with this extension instance.
        datastore_uri : str
            The connection URI for the datastore

        Raises
        ------
        RuntimeError
            Raises a RuntimeError when an instance of this extension is already registered.
        """
        if "datastore" in app.extensions:
            raise RuntimeError(
                "A 'Datastore' instance has already been registered on this Flask app."
            )
        self._datastore_uri = datastore_uri
        self._init_datastore()
        app.extensions["datastore"] = self


class MinioDatastore(BaseDatastore):
    _client: Minio
    _access_key: str
    _secret_key: str

    def __init__(self, access_key: str, secret_key: str):
        self._access_key = access_key
        self._secret_key = secret_key

    def _init_datastore(self):
        self._client = Minio(
            self._datastore_uri,
            access_key=self._access_key,
            secret_key=self._secret_key,
            secure=False,
        )
        if not self._client.bucket_exists("datasets"):
            self._client.make_bucket("datasets")

    def list_from_datastore(
        self, target_path: str, recursive: bool = True
    ) -> List[str]:
        objects = self._client.list_objects(
            "datasets", prefix=target_path, recursive=recursive
        )
        return [obj.object_name for obj in objects]

    def store_file_or_folder(self, target_path: str, src_path: str):
        if os.path.isdir(src_path):
            for root, dirs, files in os.walk(src_path):
                for file_name in files:
                    file_path = os.path.join(root, file_name)
                    relative_path = os.path.relpath(file_path, src_path)
                    minio_target_path = os.path.join(
                        target_path, relative_path
                    ).replace("\\", "/")
                    self._client.fput_object("datasets", minio_target_path, file_path)
        else:
            self._client.fput_object("datasets", target_path, src_path)

    def store_object(
        self, target_path: str, data: BytesIO, file_name: str, mime_type: str = None
    ):
        if mime_type is None:
            mime_type = mimetypes.guess_type(file_name)
            if mime_type:
                mime_type = mime_type[0]
            else:
                mime_type = "application/octet-stream"

        self._client.put_object(
            "datasets", target_path, data, len(data.getbuffer()), content_type=mime_type
        )

    def load_object_to_memory(self, target_path: str) -> Optional[bytes]:
        try:
            response = self._client.get_object("datasets", target_path)
            data = BytesIO(response.read())
            response.close()
            response.release_conn()
        except S3Error as err:
            if err.code != "NoSuchKey":
                raise err
            data = None
        return data

    def load_object_to_file(self, object_path: str, save_path: str) -> Optional[str]:
        try:
            self._client.fget_object("datasets", object_path, save_path)
        except S3Error as err:
            if err.code != "NoSuchKey":
                raise err
            save_path = None

        return save_path

    def load_folder_to_filesystem(
        self, folder_path: str, save_path: str
    ) -> Optional[str]:
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        objects = self._client.list_objects(
            "datasets", prefix=folder_path, recursive=True
        )
        files_found = False
        for obj in objects:
            files_found = True
            relative_path = os.path.relpath(obj.object_name, folder_path)
            local_file_path = os.path.join(save_path, relative_path)
            local_file_dir = os.path.dirname(local_file_path)

            if not os.path.exists(local_file_dir):
                os.makedirs(local_file_dir)

            self._client.fget_object("datasets", obj.object_name, local_file_path)
        return save_path if files_found else None

    def load_folder_to_memory(self, folder_path: str) -> Optional[Dict[str, BytesIO]]:
        directory_contents = {}
        objects = self._client.list_objects(
            "datasets", prefix=folder_path, recursive=True
        )
        for obj in objects:
            response = self._client.get_object("datasets", obj.object_name)
            data = BytesIO(response.read())
            directory_contents[obj.object_name] = data
            response.close()
            response.release_conn()
        return directory_contents if directory_contents else None
