import os

from ultralytics import YOLO


def fine_tune_model(
    model_variant: str,
    data_yaml_path: str,
    epochs: int,
) -> str:
    model = YOLO(model_variant)
    model.train(
        data=data_yaml_path,
        task="pose",
        epochs=epochs,
    )

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
            return candidate

    raise RuntimeError(
        "YOLO fine-tuning completed but no checkpoint was found "
        f"(checked: {candidate_paths})."
    )
