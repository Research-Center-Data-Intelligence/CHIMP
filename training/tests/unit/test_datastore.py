import os
from io import BytesIO

from app.datastore import BaseDatastore


class TestMinioDatastore:
    def test_list_from_datastore(self, datastore: BaseDatastore):
        """Test the list_from_datastore method."""
        # Single file and no prefix
        file = BytesIO("this is testfile1".encode())
        datastore._client.put_object(
            "datasets",
            "test/testfile1.txt",
            file,
            len(file.getbuffer()),
            content_type="text/plain",
        )
        result = datastore.list_from_datastore("")
        assert result == ["TestingDataset/test.txt", "test/testfile1.txt"]

        # Single file with prefix
        result = datastore.list_from_datastore("test")
        assert result == ["test/testfile1.txt"]

        # Non existing path
        result = datastore.list_from_datastore("does_not_exist")
        assert result == []

    def test_store_file_or_folder(self, datastore: BaseDatastore, tmpdir):
        """Test the store_file_or_folder method."""
        # Store folder
        tmpdir.join("file1.txt").write("this if file1")
        tmpdir.join("file2.txt").write("this if file2")
        datastore.store_file_or_folder("test_save_folder", tmpdir)
        result = datastore.list_from_datastore("test_save_folder")
        assert result == ["test_save_folder/file1.txt", "test_save_folder/file2.txt"]

        # Store file
        single_file = tmpdir.join("file3.txt")
        single_file.write("this if file3")
        datastore.store_file_or_folder("test_save_file/file3.txt", single_file)
        result = datastore.list_from_datastore("test_save_file")
        assert result == ["test_save_file/file3.txt"]

    def test_store_object(self, datastore: BaseDatastore):
        """Test the store_object method."""
        data = BytesIO("this is testfile1".encode())
        datastore.store_object("testfile1.txt", data, "testfile1.txt")
        result = datastore.list_from_datastore("")
        assert "testfile1.txt" in result

    def test_load_object_to_memory(self, datastore: BaseDatastore):
        """Test the load_object_to_memory method."""
        # Non-existing object
        assert not datastore.load_object_to_memory("testfile1.txt")

        # Load file
        data = BytesIO("this is file1".encode())
        datastore.store_object("file1.txt", data, "file1.txt")
        result = datastore.load_object_to_memory("file1.txt")
        assert data.read() == result.read()

    def test_load_object_to_file(self, datastore: BaseDatastore, tmpdir):
        """Test the load_object_to_file method."""
        # Non-existing object
        assert not datastore.load_object_to_file(
            "does-not-exist", tmpdir.join("does-not-exist.txt")
        )

        # Load file to filesystem
        data = BytesIO("this is file1".encode())
        datastore.store_object("file1.txt", data, "file1.txt")
        result = datastore.load_object_to_file("file1.txt", tmpdir.join("file1.txt"))
        assert result == tmpdir.join("file1.txt")
        assert os.path.exists(tmpdir.join("file1.txt"))

    def test_load_folder_to_filesystem(self, datastore: BaseDatastore, tmpdir):
        """Test the load_folder_to_filesystem method."""
        # Setup
        src_dir = tmpdir.mkdir("src")
        src_dir.mkdir("a")
        src_dir.join("file1.txt").write("this is file1")
        src_dir.join("file2.txt").write("this is file2")
        src_dir.join("a/file3.txt").write("this is file3")

        datastore.store_file_or_folder("src", src_dir)

        # Load folder to filesystem
        test1_dir = tmpdir.mkdir("test1")
        assert datastore.load_folder_to_filesystem("src", test1_dir) == test1_dir
        assert os.path.exists(test1_dir.join("file1.txt"))
        assert os.path.exists(test1_dir.join("file2.txt"))
        assert os.path.exists(test1_dir.join("a/file3.txt"))

        # Load non existing folder
        test2_dir = tmpdir.join("test2")
        assert not datastore.load_folder_to_filesystem("does-not-exist", test2_dir)

    def test_load_folder_to_memory(self, datastore: BaseDatastore, tmpdir):
        """Test the load_folder_to_memory method."""
        # Setup
        src_dir = tmpdir.mkdir("src")
        src_dir.mkdir("a")
        src_dir.join("file1.txt").write("this is file1")
        src_dir.join("file2.txt").write("this is file2")
        src_dir.join("a/file3.txt").write("this is file3")

        datastore.store_file_or_folder("src", src_dir)

        # Get directory that does not exist
        assert not datastore.load_folder_to_memory("does-not-exist")

        # Get directory
        result = datastore.load_folder_to_memory("src")
        assert "src/file1.txt" in result
        assert "src/file2.txt" in result
        assert "src/a/file3.txt" in result
