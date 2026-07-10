# YOLO Pose HPE in CHIMP: Engineering Deep-Dive

This document explains the implementation details of the YOLO Human Pose Estimation (HPE) path in CHIMP at engineering depth. It is intended for maintainers who need to debug, extend, or harden the pipeline.

## 1. Architecture boundaries

CHIMP splits YOLO HPE responsibilities into two layers:

- Training plugin layer: optionally fine-tunes Ultralytics YOLO pose checkpoints from managed datasets, then exports to ONNX and registers artifacts and metadata in MLflow.
- Serving layer: loads registered ONNX models, selects a YOLO-specific runtime wrapper, and decodes raw ONNX outputs into pose detections.

The plugin does not perform online inference decode. Decode belongs to serving.

## 2. Source map

Training side:

- training/app/plugins/yolo_pose/__init__.py
- training/app/plugins/yolo_pose/dataset.py
- training/app/plugins/yolo_pose/model.py
- training/app/plugin.py
- training/app/worker.py
- training/app/endpoints/training_endpoints.py
- training/app/connectors.py

Serving side:

- serving_api/app/connectors.py
- serving_api/app/inference.py
- serving_api/app/model.py
- serving_api/app/endpoints/inference_endpoints.py

Dependency declaration:

- training/plugin-requirements.txt

## 3. Execution call graph

The effective run path is:

1. HTTP POST /tasks/run/YOLO+Pose
2. training_endpoints.start_task()
3. WorkerManager.start_task()
4. WorkerManager._run_task() (Celery)
5. plugin_loader.load_plugins() and get_plugin()
6. YoloPosePlugin.run()
7. YoloPosePlugin._export_onnx() (subprocess)
8. connector.store_model() -> MLflow
9. serving_api loads model via connector.get_model()
10. serving model class chosen as YoloPoseOnnxModel
11. YoloPoseOnnxModel.predict() returns decoded detections

## 4. Training plugin API contract

### 4.1 PluginInfo contract

The plugin exposes these parameters through PluginInfo:

- experiment_name (required): MLflow experiment and model registration name.
- model_variant (optional): Ultralytics ID or local model path. Default yolo11n-pose.pt.
- imgsz (optional): export image size. Default 640.
- opset (optional): ONNX opset. Default 13.
- device (optional): export device. Default cpu.
- fine_tune_enabled (optional): enable dataset-driven training before export. Default false.
- dataset_name (optional): managed dataset to fine-tune on. Required when fine_tune_enabled=true.

Fine-tune hyperparameters are intentionally not exposed in the public plugin request contract.
For this proof of concept they are hardcoded in plugin code to keep the run API minimal.

Model return type is declared as onnx.

### 4.2 Request parsing behavior

The training endpoint reads values from form data first, then query parameters. All values arrive as strings from HTTP and are cast by plugin code for integer fields and fine_tune_enabled boolean coercion.

Fine-tuning defaults are currently hardcoded in training/app/plugins/yolo_pose/__init__.py.

Implication:

- imgsz and opset casting errors propagate as exceptions during plugin execution.
- No schema-level type coercion is performed in the endpoint itself.

## 5. Plugin implementation internals

### 5.1 YoloPosePlugin.__init__

Primary responsibility is static metadata declaration via PluginInfo. This is consumed by:

- /plugins response generation
- argument validation in /tasks/run/<plugin_name>
- plugin registration in PluginLoader

### 5.2 YoloPosePlugin.init

Returns self._info. This is lightweight and does not perform runtime initialization.

### 5.3 YoloPosePlugin._export_onnx

This is the core export primitive and is reused in both export-only and fine-tune flows.

Input contract:

- model_variant: str
- imgsz: int
- opset: int
- device: str
- export_dir: existing writable directory

Algorithm:

1. Build a short Python script string.
2. Script imports ultralytics.YOLO.
3. Script instantiates model = YOLO(model_variant).
4. Script calls model.export with:
  - format='onnx'
  - imgsz=<int>
  - opset=<int>
  - dynamic=False
  - simplify=False
  - device=<device>
5. Script prints exported path.
6. Parent process executes script with subprocess.run in export_dir.
7. Parent validates return code and parses stdout.
8. Parent resolves absolute path and verifies exported file exists.

Why subprocess is used:

- Native export stacks can fail in process-specific ways.
- Process isolation limits blast radius to the child process.

Failure surface:

- Return code non-zero: RuntimeError includes stderr text.
- Empty stdout: RuntimeError, no path emitted.
- Path does not exist: RuntimeError, stale or invalid output path.

### 5.4 Dataset and model helpers

Dataset conversion and fine-tuning internals are split from the plugin entrypoint:

- training/app/plugins/yolo_pose/dataset.py
  - Downloads managed dataset files from MinIO
  - Resolves COCO keypoints annotations from datapoints table
  - Builds YOLO pose train/val layout and data.yaml
- training/app/plugins/yolo_pose/model.py
  - Runs Ultralytics YOLO.train
  - Resolves best/last checkpoint path for subsequent ONNX export

The plugin still exposes compatibility wrapper methods (`_prepare_finetune_dataset`, `_fine_tune_model`) so existing tests and call sites remain stable.

### 5.5 YoloPosePlugin.run

This method orchestrates training (optional), export, and registry operations.

Step-by-step:

1. Read worker-injected values:
  - run_name
  - temp_dir
2. Read plugin args and apply defaults:
  - model_variant default yolo11n-pose.pt
  - imgsz default 640
  - opset default 13
  - device default cpu
3. If fine_tune_enabled=true:
  - validate dataset_name
  - download managed dataset files to temp workspace
  - load matching COCO keypoints annotations from datapoints table
  - convert data to YOLO pose directory layout plus data.yaml
  - run YOLO.train() using hardcoded defaults and resolve trained checkpoint path
4. Create temp_dir/export.
5. Call _export_onnx() using either the trained checkpoint (fine-tune mode) or model_variant (export-only mode).
6. Load ONNX graph via onnx.load() for connector logging.
7. Create temp_dir/yolo_pose artifact folder.
8. Copy ONNX file to artifact folder if source and target differ.
9. Build hyperparameters payload:
  - model_variant, export_model_variant, imgsz, opset, device, task=pose
  - fine_tune_enabled, and fine-tune metadata when enabled (dataset_name, epochs, train_batch_size, train_patience, sample_count, split_mode)
10. Build tags payload:
  - framework=ultralytics
  - task=pose
  - training_mode=export_only|fine_tune
11. Store model with connector.store_model(..., model_type='onnx').
12. Return stored run name.

Design note:

- The task tag and framework tag are not cosmetic. Serving uses them for runtime class selection.

## 6. Worker and lifecycle semantics

### 6.1 PluginLoader behavior

PluginLoader scans the configured plugin directory and imports modules that are either:

- package directories with __init__.py
- standalone .py files

It instantiates classes that subclass BasePlugin, then injects:

- self._connector
- self._datastore

### 6.2 Celery task wrapper behavior

WorkerManager._run_task:

1. Validates plugin_name exists.
2. Reloads plugins before execution.
3. Generates run_name using UTC timestamp + UUID.
4. Creates task temp directory with mkdtemp.
5. Invokes plugin.run.
6. Stores last successful result in Redis.
7. Deletes temp directory.

Operational impact:

- Export artifacts only persist if they are explicitly logged to MLflow.
- Local temp data is ephemeral by design.

## 7. MLflow persistence contract

The training connector performs these operations:

1. Set active experiment.
2. Start run (using generated run_name unless provided).
3. Log params, metrics, and tags.
4. Upload artifact directories.
5. Log model according to model_type.

For YOLO plugin, model_type resolves to ONNX and uses mlflow.onnx.log_model.

Expected persisted data for YOLO runs:

- Params: model_variant, imgsz, opset, device, task
- Tags: framework=ultralytics, task=pose, model_type=onnx
- Artifacts: onnx_export/<exported_model>.onnx
- Model registry entry: registered under experiment_name

## 8. Serving-time class selection

Serving connector chooses model wrapper class using tags first, then naming fallback.

Selection logic summary:

1. If tags indicate task=pose or framework=ultralytics: use YoloPoseOnnxModel.
2. Else if model name contains both yolo and pose: use YoloPoseOnnxModel.
3. Else use generic OnnxModel.

Consequence:

- Incorrect or missing tags may silently route inference through generic ONNX behavior without pose decode.

## 9. YoloPoseOnnxModel decode pipeline

### 9.1 Input expectations

Serving endpoint expects JSON with inputs as a list-compatible structure. Internally data is converted to numpy.

Typical preprocessing path for YOLO ONNX in CHIMP tests:

- load image
- resize to export size
- RGB ordering
- HWC -> NCHW
- float32 normalization to [0, 1]

### 9.2 Output normalization

Function _prediction_to_tensors handles multiple runtime output forms:

- dict outputs from pyfunc wrappers
- dataframe-like outputs with columns
- scalar/object wrappers

It emits a deterministic dictionary of numpy arrays.

### 9.3 Shape heuristic

Function _looks_like_yolo_pose_output checks whether tensor layout resembles YOLO pose output (single tensor, rank-3, sufficient channel width).

If true, decode path is activated.

### 9.4 Bounding box conversion

Function _xywh_to_xyxy transforms center-form boxes to corner-form boxes.

Given box [x, y, w, h], conversion is:

- x1 = x - w/2
- y1 = y - h/2
- x2 = x + w/2
- y2 = y + h/2

### 9.5 NMS implementation

Function _nms_indices performs greedy score-sorted NMS:

1. Sort by confidence descending.
2. Keep top candidate.
3. Compute IoU with remaining candidates.
4. Drop boxes above iou_threshold.
5. Repeat until exhausted.

Default thresholds used in decode:

- conf_threshold = 0.10
- iou_threshold = 0.45

### 9.6 Keypoint extraction

For each kept row:

- row[0:4] is box in xywh
- row[4] is object confidence
- row[5:] is reshaped to triplets [x, y, confidence]

Returned object per detection:

- confidence
- bbox_xywh
- bbox_xyxy
- keypoints: list of {x, y, confidence}

### 9.7 Returned prediction envelope

For YOLO-like outputs, predict returns:

- pose_detections: decoded objects
- raw: full tensor payload converted to JSON-compatible lists

Metadata from MLflow model object is also returned by endpoint.

## 10. Reliability and failure modes

### 10.1 Export/runtime mismatch

Symptom:

- subprocess export fails or produces incompatible graph.

Likely causes:

- wrong Python interpreter on PATH
- missing ultralytics in worker runtime
- unsupported device argument

Mitigation:

- pin interpreter path for subprocess command
- assert dependency versions in startup checks

### 10.2 Model not decoded as YOLO

Symptom:

- predictions only raw tensor dictionary, no pose_detections.

Likely causes:

- missing task/framework tags in run
- model name does not trigger fallback heuristic

Mitigation:

- ensure plugin tags are preserved
- verify run tags in MLflow UI/API

### 10.3 Empty detections

Symptom:

- pose_detections present but empty.

Likely causes:

- preprocessing shape mismatch
- too strict confidence threshold for current sample
- graph output layout different from decode assumptions

Mitigation:

- inspect raw output field
- test with known positive sample
- add debug logging around tensor shapes

## 11. Performance characteristics

- Export is offline and potentially expensive; it is intentionally asynchronous via Celery.
- Inference decoding is CPU numpy-based and linearithmic in detection count due to sorting in NMS.
- Returning raw outputs increases response size; useful for debugging, expensive for production payloads.

## 12. Extension guidance

High-value extension points:

1. Make conf_threshold and iou_threshold configurable per request or model tag.
2. Persist keypoint schema metadata (for example COCO keypoint index mapping) with model tags.
3. Add compatibility validation after export (shape signature check before MLflow log).
4. Add integration tests that assert:
  - model class selection based on tags
  - decode output schema
  - non-empty detections for golden sample
5. Replace subprocess "python" with configured interpreter from runtime settings.

## 13. Minimal operational checklist

Before running YOLO HPE pipeline, verify:

1. training/plugin-requirements.txt includes ultralytics in deployed image/environment.
2. MLflow tracking URI is reachable from training and serving components.
3. Worker has write permissions for temporary directories.
4. Registered run contains tags framework=ultralytics and task=pose.
5. Serving endpoint receives normalized NCHW float input tensors.
