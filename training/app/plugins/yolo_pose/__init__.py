import os
import subprocess
import shutil
import sys
import json
from typing import Any, Optional

import onnx

from app.plugin import BasePlugin, PluginInfo


class YoloPosePlugin(BasePlugin):
    def __init__(self):
        self._info = PluginInfo(
            name="YOLO Pose",
            version="0.1.0",
            description="Exports a YOLO pose model to ONNX and registers it in MLflow.",
            arguments={
                "experiment_name": {
                    "name": "experiment_name",
                    "type": "str",
                    "description": "MLflow experiment name and registered model name.",
                    "optional": False,
                },
                "model_variant": {
                    "name": "model_variant",
                    "type": "str",
                    "description": "Ultralytics model identifier or local path (for example: yolo11n-pose.pt).",
                    "optional": True,
                },
                "imgsz": {
                    "name": "imgsz",
                    "type": "int",
                    "description": "Export image size used for ONNX graph creation.",
                    "optional": True,
                },
                "opset": {
                    "name": "opset",
                    "type": "int",
                    "description": "ONNX opset version.",
                    "optional": True,
                },
                "device": {
                    "name": "device",
                    "type": "str",
                    "description": "Export device for Ultralytics (cpu or cuda:0).",
                    "optional": True,
                },
            },
            model_return_type="onnx",
        )

    def init(self) -> PluginInfo:
        return self._info

    @staticmethod
    def _export_onnx(
        model_variant: str,
        imgsz: int,
        opset: int,
        device: str,
        export_dir: str,
    ) -> str:
        # Run export in a child process to isolate potential native-lib crashes.
        cmd = [
            sys.executable,
            "-m",
            "app.plugins.yolo_pose.export_onnx_worker",
            "--model-variant",
            model_variant,
            "--imgsz",
            str(imgsz),
            "--opset",
            str(opset),
            "--device",
            device,
        ]
        res = subprocess.run(
            cmd,
            cwd=export_dir,
            capture_output=True,
            text=True,
        )
        if res.returncode != 0:
            raise RuntimeError(
                "YOLO export failed with return code "
                f"{res.returncode}. stderr: {res.stderr.strip()}"
            )

        result_prefix = "CHIMP_EXPORT_RESULT:"
        lines = [line.strip() for line in res.stdout.splitlines() if line.strip()]
        result_line = next(
            (line for line in reversed(lines) if line.startswith(result_prefix)), None
        )
        if not result_line:
            raise RuntimeError(
                "YOLO export did not produce a structured result line. stdout: "
                f"{res.stdout.strip()}"
            )

        try:
            payload = json.loads(result_line[len(result_prefix) :].strip())
        except json.JSONDecodeError as ex:
            raise RuntimeError(
                "YOLO export produced an invalid structured result line."
            ) from ex

        exported_path = payload.get("exported_path")
        if not exported_path:
            raise RuntimeError("YOLO export did not produce an ONNX path.")

        if not os.path.isabs(exported_path):
            exported_path = os.path.abspath(os.path.join(export_dir, exported_path))
        if not os.path.exists(exported_path):
            raise RuntimeError(f"Exported ONNX file was not found at {exported_path}")
        return exported_path

    def run(self, *args, **kwargs) -> Optional[Any]:
        experiment_name = kwargs["experiment_name"]
        run_name = kwargs.get("run_name")
        temp_dir = kwargs["temp_dir"]

        model_variant = kwargs.get("model_variant") or "yolo11n-pose.pt"
        imgsz = int(kwargs.get("imgsz") or 640)
        opset = int(kwargs.get("opset") or 13)
        device = kwargs.get("device") or "cpu"

        print(
            f"[YOLO Pose] Loading model '{model_variant}' and exporting to ONNX "
            f"(imgsz={imgsz}, opset={opset}, device={device})"
        )

        export_dir = os.path.join(temp_dir, "export")
        os.makedirs(export_dir, exist_ok=True)
        exported_path = self._export_onnx(model_variant, imgsz, opset, device, export_dir)

        onnx_model = onnx.load(exported_path)

        artifact_dir = os.path.join(temp_dir, "yolo_pose")
        os.makedirs(artifact_dir, exist_ok=True)
        artifact_onnx_path = os.path.join(artifact_dir, os.path.basename(exported_path))
        if os.path.abspath(exported_path) != os.path.abspath(artifact_onnx_path):
            shutil.copy2(exported_path, artifact_onnx_path)

        hyperparameters = {
            "model_variant": model_variant,
            "imgsz": imgsz,
            "opset": opset,
            "device": device,
            "task": "pose",
        }
        tags = {
            "framework": "ultralytics",
            "task": "pose",
        }

        stored_run_name, _ = self._connector.store_model(
            experiment_name=experiment_name,
            model_name=experiment_name,
            run_name=run_name,
            model=onnx_model,
            model_type="onnx",
            hyperparameters=hyperparameters,
            tags=tags,
            artifacts={"onnx_export": artifact_dir},
        )

        return stored_run_name
