import os
import math
import re
from typing import Any, Dict, Optional, Tuple


def _to_float(value: Any) -> Optional[float]:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric_value):
        return None
    return numeric_value


def _sanitize_metric_name(name: str) -> str:
    sanitized = re.sub(r"[^0-9A-Za-z_./\- ]", "_", name.strip())
    sanitized = sanitized.replace(" ", "_")
    sanitized = re.sub(r"_+", "_", sanitized)
    return sanitized.strip("_")


def _merge_numeric_metrics(target: Dict[str, float], source: Dict[str, Any]) -> None:
    for key, value in source.items():
        metric_name = _sanitize_metric_name(str(key))
        if not metric_name:
            continue
        numeric_value = _to_float(value)
        if numeric_value is None:
            continue
        target[metric_name] = numeric_value


def collect_training_metrics(train_results: Any, trainer: Any) -> Dict[str, float]:
    metrics: Dict[str, float] = {}

    # Ultralytics train() for pose returns PoseMetrics; prefer its API directly.
    keys_fn = getattr(train_results, "keys", None)
    mean_results_fn = getattr(train_results, "mean_results", None)
    if callable(keys_fn) and callable(mean_results_fn):
        keys = list(keys_fn())
        mean_results = list(mean_results_fn())
        if len(keys) == len(mean_results):
            _merge_numeric_metrics(metrics, dict(zip(keys, mean_results)))

    results_dict = getattr(train_results, "results_dict", None)
    if isinstance(results_dict, dict):
        _merge_numeric_metrics(metrics, results_dict)

    if isinstance(train_results, dict):
        _merge_numeric_metrics(metrics, train_results)

    fitness_fn = getattr(train_results, "fitness", None)
    if callable(fitness_fn):
        fitness_value = _to_float(fitness_fn())
        if fitness_value is not None:
            metrics["fitness"] = fitness_value

    speed = getattr(train_results, "speed", None)
    if isinstance(speed, dict):
        _merge_numeric_metrics(metrics, {f"speed/{k}": v for k, v in speed.items()})

    trainer_metrics = getattr(trainer, "metrics", None)
    if isinstance(trainer_metrics, dict):
        _merge_numeric_metrics(metrics, trainer_metrics)

    for attr_name in ("best_fitness", "epoch"):
        attr_value = _to_float(getattr(trainer, attr_name, None))
        if attr_value is not None:
            metrics[attr_name] = attr_value

    return metrics


def fine_tune_model(
    model_variant: str,
    data_yaml_path: str,
    epochs: int,
) -> Tuple[str, Dict[str, float]]:
    # Import lazily so non-GPU services can still boot even when torch CUDA libs
    # are unavailable in those containers.
    from ultralytics import YOLO

    model = YOLO(model_variant)
    train_results = model.train(
        data=data_yaml_path,
        task="pose",
        epochs=epochs,
    )
    trainer = getattr(model, "trainer", None)
    metrics = collect_training_metrics(train_results, trainer)

    candidate_paths = []
    trainer = getattr(model, "trainer", None)
    best_path = getattr(trainer, "best", None)
    if best_path:
        candidate_paths.append(str(best_path))
    last_path = getattr(trainer, "last", None)
    if last_path:
        candidate_paths.append(str(last_path))

    for candidate in candidate_paths:
        if os.path.exists(candidate):
            return candidate, metrics

    raise RuntimeError(
        "YOLO fine-tuning completed but no checkpoint was found "
        f"(checked: {candidate_paths})."
    )
