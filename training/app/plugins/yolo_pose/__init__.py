import os
import subprocess
import shutil
import sys
import json
from pathlib import Path
from typing import Optional

import onnx

from app.plugin import BasePlugin, PluginInfo
from .dataset import prepare_finetune_dataset


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
                "dataset_name": {
                    "name": "dataset_name",
                    "type": "str",
                    "description": "Managed dataset folder used for fine-tuning. If omitted, plugin exports baseline model.",
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
        worker_script = os.path.join(
            os.path.dirname(__file__),
            "export_onnx_worker.py",
        )
        cmd = [
            sys.executable,
            worker_script,
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

    def _prepare_finetune_dataset(self, dataset_name: str, temp_dir: str) -> str:
        return prepare_finetune_dataset(
            datastore=self._datastore,
            dataset_name=dataset_name,
            temp_dir=temp_dir,
        )

    @staticmethod
    def _fine_tune_model(
        model_variant: str,
        data_yaml_path: str,
        epochs: int,
    ) -> str:
        from .model import fine_tune_model

        return fine_tune_model(
            model_variant=model_variant,
            data_yaml_path=data_yaml_path,
            epochs=epochs,
        )

    def run(self, *args, **kwargs) -> Optional[str]:
        experiment_name = kwargs["experiment_name"]
        run_name = kwargs.get("run_name")
        temp_dir = kwargs["temp_dir"]

        # Hardcoded defaults keep POC invocation minimal and deterministic.
        model_variant = "yolo11n-pose.pt"
        imgsz = 640
        opset = 13
        device = "cpu"
        dataset_name = kwargs.get("dataset_name")
        fine_tune_enabled = bool(dataset_name)
        epochs = 10

        if fine_tune_enabled:
            # Stap 2: probeer het productiemodel op te halen uit MLflow.
            # Val terug op het basismodel als er nog geen productiemodel bestaat.
            pt_download_dir = os.path.join(temp_dir, "production_pt")
            os.makedirs(pt_download_dir, exist_ok=True)
            try:
                downloaded = self._connector.get_artifact(
                    save_to=pt_download_dir,
                    model_name=experiment_name,
                    experiment_name=experiment_name,
                    artifact_path="pt_checkpoint",
                )
                pt_files = list(Path(downloaded).rglob("*.pt"))
                if pt_files:
                    model_variant = str(pt_files[0])
                    print(f"[YOLO Pose] Using production checkpoint: {model_variant}")
                else:
                    print("[YOLO Pose] No .pt found in downloaded artifact, falling back to base model.")
            except Exception as exc:
                print(f"[YOLO Pose] Could not retrieve production model ({exc}), falling back to base model.")

        export_model_variant = model_variant
        metrics = {}
        if fine_tune_enabled:
            data_yaml_path = self._prepare_finetune_dataset(
                dataset_name=dataset_name,
                temp_dir=temp_dir,
            )
            print(
                f"[YOLO Pose] Fine-tuning model '{model_variant}' on dataset "
                f"'{dataset_name}' with minimal train call (epochs={epochs})."
            )
            export_model_variant, metrics = self._fine_tune_model(
                model_variant=model_variant,
                data_yaml_path=data_yaml_path,
                epochs=epochs,
            )

        print(
            f"[YOLO Pose] Loading model '{export_model_variant}' and exporting to ONNX "
            f"(imgsz={imgsz}, opset={opset}, device={device})"
        )

        export_dir = os.path.join(temp_dir, "export")
        os.makedirs(export_dir, exist_ok=True)
        exported_path = self._export_onnx(export_model_variant, imgsz, opset, device, export_dir)

        onnx_model = onnx.load(exported_path)

        artifact_onnx_dir = os.path.join(temp_dir, "yolo_pose_onnx")
        os.makedirs(artifact_onnx_dir, exist_ok=True)
        artifact_onnx_path = os.path.join(artifact_onnx_dir, os.path.basename(exported_path))
        if os.path.abspath(exported_path) != os.path.abspath(artifact_onnx_path):
            shutil.copy2(exported_path, artifact_onnx_path)

        artifact_pt_dir = os.path.join(temp_dir, "yolo_pose_pt")
        os.makedirs(artifact_pt_dir, exist_ok=True)
        pt_artifact_path = os.path.join(artifact_pt_dir, os.path.basename(export_model_variant))
        shutil.copy2(export_model_variant, pt_artifact_path)
        artifacts = {"onnx_export": artifact_onnx_dir, "pt_checkpoint": artifact_pt_dir}

        hyperparameters = {
            "model_variant": model_variant,
            "export_model_variant": export_model_variant,
            "imgsz": imgsz,
            "opset": opset,
            "device": device,
            "task": "pose",
            "fine_tune_enabled": fine_tune_enabled,
        }
        if fine_tune_enabled:
            hyperparameters.update(
                {
                    "dataset_name": dataset_name,
                    "epochs": epochs,
                }
            )
        tags = {
            "framework": "ultralytics",
            "task": "pose",
            "training_mode": "fine_tune" if fine_tune_enabled else "export_only",
        }

        stored_run_name, _ = self._connector.store_model(
            experiment_name=experiment_name,
            model_name=experiment_name,
            run_name=run_name,
            model=onnx_model,
            model_type="onnx",
            hyperparameters=hyperparameters,
            tags=tags,
            metrics=metrics,
            artifacts=artifacts,
        )

        return stored_run_name
