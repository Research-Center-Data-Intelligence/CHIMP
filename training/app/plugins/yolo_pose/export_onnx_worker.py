import argparse
import json

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a YOLO pose checkpoint to ONNX for CHIMP."
    )
    parser.add_argument("--model-variant", required=True, help="Model id or local path")
    parser.add_argument("--imgsz", type=int, required=True, help="Export image size")
    parser.add_argument("--opset", type=int, required=True, help="ONNX opset version")
    parser.add_argument("--device", required=True, help="Export device (cpu/cuda:0)")
    return parser.parse_args()


def main():
    args = parse_args()

    model = YOLO(args.model_variant)
    path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        dynamic=False,
        simplify=False,
        device=args.device,
    )

    # Emit one structured line so parent process can safely parse result.
    print("CHIMP_EXPORT_RESULT:" + json.dumps({"exported_path": str(path)}))


if __name__ == "__main__":
    main()
