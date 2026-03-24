import json
from pathlib import Path

import pytest

import app.plugins.yolo_pose as yolo_pose_module
from app.plugins.yolo_pose import YoloPosePlugin


class _MockCompletedProcess:
    # Minimal subprocess.CompletedProcess-like object used by _export_onnx tests.
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class _MockConnector:
    def __init__(self):
        self.calls = []

    def store_model(self, **kwargs):
        # Capture call args so tests can assert exactly what run() forwards.
        self.calls.append(kwargs)
        return "stored-run", "run-id"


class TestYoloPosePlugin:
    """Tests for the YOLO pose plugin."""

    def test_init_returns_plugin_info(self):
        """Tests for plugin metadata initialization."""
        plugin = YoloPosePlugin()

        info = plugin.init()
        assert info.name == "YOLO Pose"
        assert info.model_return_type == "onnx"
        assert "experiment_name" in info.arguments
        assert "model_variant" in info.arguments

    def test_export_onnx_success_with_relative_path(self, tmp_path: Path, monkeypatch):
        """Tests for successful ONNX export parsing with a relative path."""
        exported_path = tmp_path / "model.onnx"
        exported_path.write_text("onnx")

        def mocked_run(cmd, cwd, capture_output, text):
            assert "--model-variant" in cmd
            assert cwd == str(tmp_path)
            assert capture_output is True
            assert text is True
            payload = json.dumps({"exported_path": "model.onnx"})
            return _MockCompletedProcess(
                returncode=0,
                stdout=f"log line\nCHIMP_EXPORT_RESULT:{payload}\n",
            )

        monkeypatch.setattr(yolo_pose_module.subprocess, "run", mocked_run)

        result = YoloPosePlugin._export_onnx(
            model_variant="yolo11n-pose.pt",
            imgsz=640,
            opset=13,
            device="cpu",
            export_dir=str(tmp_path),
        )
        assert result == str(exported_path)

    def test_export_onnx_raises_when_worker_fails(self, tmp_path: Path, monkeypatch):
        """Tests for export failure when the worker process returns non-zero."""
        monkeypatch.setattr(
            yolo_pose_module.subprocess,
            "run",
            lambda *args, **kwargs: _MockCompletedProcess(
                returncode=1,
                stderr="boom",
            ),
        )

        with pytest.raises(RuntimeError, match="YOLO export failed"):
            YoloPosePlugin._export_onnx("model.pt", 640, 13, "cpu", str(tmp_path))

    def test_export_onnx_raises_without_structured_line(self, tmp_path: Path, monkeypatch):
        """Tests for export failure when the structured result line is missing."""
        monkeypatch.setattr(
            yolo_pose_module.subprocess,
            "run",
            lambda *args, **kwargs: _MockCompletedProcess(
                returncode=0,
                stdout="just plain logs",
            ),
        )

        # The plugin expects one machine-parseable output line from the worker.
        with pytest.raises(RuntimeError, match="did not produce a structured result line"):
            YoloPosePlugin._export_onnx("model.pt", 640, 13, "cpu", str(tmp_path))

    def test_export_onnx_raises_with_invalid_structured_json(
        self,
        tmp_path: Path,
        monkeypatch,
    ):
        """Tests for export failure when the structured result payload is invalid JSON."""
        monkeypatch.setattr(
            yolo_pose_module.subprocess,
            "run",
            lambda *args, **kwargs: _MockCompletedProcess(
                returncode=0,
                stdout="CHIMP_EXPORT_RESULT:not-json",
            ),
        )

        with pytest.raises(RuntimeError, match="invalid structured result line"):
            YoloPosePlugin._export_onnx("model.pt", 640, 13, "cpu", str(tmp_path))

    def test_export_onnx_raises_when_path_missing_in_payload(
        self,
        tmp_path: Path,
        monkeypatch,
    ):
        """Tests for export failure when the structured payload has no ONNX path."""
        monkeypatch.setattr(
            yolo_pose_module.subprocess,
            "run",
            lambda *args, **kwargs: _MockCompletedProcess(
                returncode=0,
                stdout="CHIMP_EXPORT_RESULT:{}",
            ),
        )

        with pytest.raises(RuntimeError, match="did not produce an ONNX path"):
            YoloPosePlugin._export_onnx("model.pt", 640, 13, "cpu", str(tmp_path))

    def test_export_onnx_raises_when_exported_file_not_found(
        self,
        tmp_path: Path,
        monkeypatch,
    ):
        """Tests for export failure when the reported ONNX file does not exist."""
        payload = json.dumps({"exported_path": "missing.onnx"})
        monkeypatch.setattr(
            yolo_pose_module.subprocess,
            "run",
            lambda *args, **kwargs: _MockCompletedProcess(
                returncode=0,
                stdout=f"CHIMP_EXPORT_RESULT:{payload}",
            ),
        )

        with pytest.raises(RuntimeError, match="Exported ONNX file was not found"):
            YoloPosePlugin._export_onnx("model.pt", 640, 13, "cpu", str(tmp_path))

    def test_run_stores_model_and_copies_export(self, tmp_path: Path, monkeypatch):
        """Tests for run flow that copies export and stores model metadata."""
        plugin = YoloPosePlugin()
        plugin._connector = _MockConnector()

        export_dir = tmp_path / "export"
        export_dir.mkdir(parents=True, exist_ok=True)
        exported_path = export_dir / "exported.onnx"
        exported_path.write_text("onnx")

        copied = {}

        def mocked_copy2(src, dst):
            # Keep this test filesystem-only while still validating copy behavior.
            copied["src"] = src
            copied["dst"] = dst
            Path(dst).write_text(Path(src).read_text())

        monkeypatch.setattr(YoloPosePlugin, "_export_onnx", lambda *args, **kwargs: str(exported_path))
        monkeypatch.setattr(yolo_pose_module.onnx, "load", lambda path: {"loaded": path})
        monkeypatch.setattr(yolo_pose_module.shutil, "copy2", mocked_copy2)

        result = plugin.run(experiment_name="exp", temp_dir=str(tmp_path))

        assert result == "stored-run"
        assert plugin._connector.calls
        call = plugin._connector.calls[0]
        assert call["experiment_name"] == "exp"
        assert call["model_name"] == "exp"
        assert call["model_type"] == "onnx"
        assert call["hyperparameters"]["task"] == "pose"
        assert call["tags"]["framework"] == "ultralytics"
        assert "onnx_export" in call["artifacts"]
        assert copied["src"] == str(exported_path)

    def test_run_skips_copy_when_export_already_in_artifact_dir(self, tmp_path: Path, monkeypatch):
        """Tests for run flow that avoids copying when export already targets artifact dir."""
        plugin = YoloPosePlugin()
        plugin._connector = _MockConnector()

        artifact_dir = tmp_path / "yolo_pose"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        exported_path = artifact_dir / "kept.onnx"
        exported_path.write_text("onnx")

        monkeypatch.setattr(YoloPosePlugin, "_export_onnx", lambda *args, **kwargs: str(exported_path))
        monkeypatch.setattr(yolo_pose_module.onnx, "load", lambda path: {"loaded": path})
        monkeypatch.setattr(
            yolo_pose_module.shutil,
            "copy2",
            # If source equals destination target, run() should avoid redundant copying.
            lambda *args, **kwargs: pytest.fail("copy2 should not be called"),
        )

        result = plugin.run(
            experiment_name="exp",
            run_name="explicit-run",
            temp_dir=str(tmp_path),
            model_variant="my.pt",
            imgsz="320",
            opset="17",
            device="cuda:0",
        )

        assert result == "stored-run"
        call = plugin._connector.calls[0]
        assert call["run_name"] == "explicit-run"
        assert call["hyperparameters"]["model_variant"] == "my.pt"
        assert call["hyperparameters"]["imgsz"] == 320
        assert call["hyperparameters"]["opset"] == 17
        assert call["hyperparameters"]["device"] == "cuda:0"
