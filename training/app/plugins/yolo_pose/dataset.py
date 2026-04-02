import json
import os
import shutil
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image


def _normalize_bbox(bbox: list, width: int, height: int) -> tuple[float, float, float, float]:
    if len(bbox) != 4:
        raise RuntimeError("COCO bbox must contain 4 values [x, y, w, h].")

    x, y, w, h = [float(v) for v in bbox]
    x_center = (x + (w / 2.0)) / width
    y_center = (y + (h / 2.0)) / height
    return x_center, y_center, w / width, h / height


def _normalize_keypoints(keypoints: list, width: int, height: int) -> list[float]:
    if len(keypoints) != 51:
        raise RuntimeError("COCO keypoints must contain 51 values for 17 joints.")

    normalized = []
    for index in range(0, len(keypoints), 3):
        x = float(keypoints[index]) / width
        y = float(keypoints[index + 1]) / height
        v = float(keypoints[index + 2])
        normalized.extend([x, y, v])
    return normalized


def _load_annotation_by_object_name(datastore, dataset_name: str) -> dict[str, dict]:
    query = """
        SELECT x, y, metadata
        FROM datapoints
        WHERE x LIKE %s
        ORDER BY id ASC
    """
    like_pattern = f"%/manageddataset/{dataset_name}/%"

    # image filename -> parsed annotation dict.
    annotation_by_object_name = {}
    with datastore._db_conn.cursor() as cursor:
        cursor.execute(query, (like_pattern,))
        rows = cursor.fetchall()

    for row in rows:
        x_value, y_value, metadata_value = row
        object_name = os.path.basename(urlparse(x_value).path)

        annotation = None
        # datapoints.y should contain full annotation JSON as a string
        if isinstance(y_value, str):
            try:
                y_parsed = json.loads(y_value)
                if isinstance(y_parsed, dict):
                    annotation = y_parsed
            except json.JSONDecodeError:
                annotation = None

            # create dict skip invalid rows.
        if isinstance(annotation, dict):
            annotation_by_object_name[object_name] = annotation

    return annotation_by_object_name


def _write_samples(samples: list[tuple[Path, dict]], images_dir: str, labels_dir: str):
    for source_image_path, annotation in samples:
        # Copy source image into YOLO dataset image folder
        target_image_path = os.path.join(images_dir, source_image_path.name)
        shutil.copy2(str(source_image_path), target_image_path)

        # normalize COCO pixel coords 
        with Image.open(source_image_path) as image:
            width, height = image.size

        x_center, y_center, bbox_w, bbox_h = _normalize_bbox(
            annotation["bbox"],
            width,
            height,
        )
        keypoints = _normalize_keypoints(annotation["keypoints"], width, height)

        # YOLO pose label row: class_id, bbox(xywh), 17 keypoint triplets (x, y, v)
        row_values = [
            0.0,
            x_center,
            y_center,
            bbox_w,
            bbox_h,
            *keypoints,
        ]

        # Label filename must match image stem so Ultralytics can pair them.
        label_path = os.path.join(labels_dir, f"{Path(source_image_path).stem}.txt")
        with open(label_path, "w", encoding="utf-8") as label_file:
            label_file.write(" ".join(str(v) for v in row_values))
            label_file.write("\n")


def prepare_finetune_dataset(datastore, dataset_name: str, temp_dir: str) -> str:
    # Pull all raw dataset files from managed storage into a local temp workspace.
    staging_root = os.path.join(temp_dir, "dataset_staging")
    os.makedirs(staging_root, exist_ok=True)

    downloaded_path = datastore.load_folder_to_filesystem(
        dataset_name,
        staging_root,
        bucket="manageddataset",
    )
    if downloaded_path is None:
        raise RuntimeError(f"No files found for dataset '{dataset_name}' in manageddataset bucket.")

    # Discover candidate image files recursively 
    image_paths = []
    for extension in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
        image_paths.extend(Path(downloaded_path).rglob(extension))
        image_paths.extend(Path(downloaded_path).rglob(extension.upper()))
    image_paths = sorted(set(image_paths))
    if not image_paths:
        raise RuntimeError(f"No image files found in dataset '{dataset_name}'.")

    # Map image filename -> annotation 
    annotation_by_object_name = _load_annotation_by_object_name(datastore, dataset_name)

    # Keep only images that have a valid COCO keypoints annotation 
    matched_samples = []
    for image_path in image_paths:
        object_name = image_path.name
        annotation = annotation_by_object_name.get(object_name)
        if annotation is None:
            continue

        if annotation.get("format") != "coco_keypoints":
            continue
        if len(annotation.get("bbox", [])) != 4:
            continue
        if len(annotation.get("keypoints", [])) != 51:
            continue
        matched_samples.append((image_path, annotation))

    if not matched_samples:
        raise RuntimeError(
            "No valid COCO keypoints annotations were found for downloaded dataset images."
        )

    # Build one YOLO pose sample set 
    yolo_root = os.path.join(temp_dir, "yolo_finetune_dataset")
    images_all_dir = os.path.join(yolo_root, "images", "all")
    labels_all_dir = os.path.join(yolo_root, "labels", "all")
    os.makedirs(images_all_dir, exist_ok=True)
    os.makedirs(labels_all_dir, exist_ok=True)

    _write_samples(matched_samples, images_all_dir, labels_all_dir)

    # Generate Ultralytics dataset config
    data_yaml_path = os.path.join(yolo_root, "data.yaml")
    data_yaml_content = "\n".join(
        [
            f"path: {yolo_root.replace(os.sep, '/')}",
            "train: images/all",
            "val: images/all",
            "kpt_shape: [17, 3]",
            "flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]",
            "names:",
            "  0: person",
        ]
    )
    with open(data_yaml_path, "w", encoding="utf-8") as data_yaml_file:
        data_yaml_file.write(data_yaml_content)
        data_yaml_file.write("\n")

    return data_yaml_path
