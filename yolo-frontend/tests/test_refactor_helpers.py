import os
import sys
import unittest
from io import BytesIO
from unittest.mock import patch

import numpy as np
from PIL import Image

TEST_ROOT = os.path.dirname(os.path.dirname(__file__))
if TEST_ROOT not in sys.path:
    sys.path.insert(0, TEST_ROOT)

from coco_builder import build_coco_keypoints_predictions_from_frames
from utils import decode_yolo_pose_output, sample_evenly_spaced_indices
from video_utils import extract_png_frames_from_video


def _png_bytes(color: tuple[int, int, int] = (255, 0, 0), size: tuple[int, int] = (4, 4)) -> bytes:
    image = Image.new("RGB", size, color=color)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class SampleIndicesTests(unittest.TestCase):
    def test_sample_evenly_spaced_indices_handles_small_ranges(self) -> None:
        self.assertEqual(sample_evenly_spaced_indices(3, 5), [0, 1, 2])
        self.assertEqual(sample_evenly_spaced_indices(10, 4), [0, 3, 6, 9])

    def test_decode_yolo_pose_output_applies_nms(self) -> None:
        raw_output = np.zeros((1, 8, 10), dtype=np.float32)
        raw_output[0, 0:4, 0] = [100.0, 100.0, 20.0, 20.0]
        raw_output[0, 4, 0] = 0.95
        raw_output[0, 5:, 0] = [1.0, 2.0, 0.9]
        raw_output[0, 0:4, 1] = [101.0, 101.0, 20.0, 20.0]
        raw_output[0, 4, 1] = 0.80
        raw_output[0, 5:, 1] = [3.0, 4.0, 0.8]

        detections = decode_yolo_pose_output(raw_output, conf_threshold=0.1, iou_threshold=0.4)

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["bbox_xywh"], [100.0, 100.0, 20.0, 20.0])


class VideoAndCocoTests(unittest.TestCase):
    def test_extract_png_frames_from_video_uses_capture_frames(self) -> None:
        frames = [
            np.full((2, 2, 3), 10, dtype=np.uint8),
            np.full((2, 2, 3), 20, dtype=np.uint8),
            np.full((2, 2, 3), 30, dtype=np.uint8),
        ]

        class FakeCapture:
            def __init__(self, *_args, **_kwargs):
                self.index = 0

            def isOpened(self):
                return True

            def get(self, prop):
                return len(frames) if prop == 7 else 0

            def read(self):
                if self.index >= len(frames):
                    return False, None
                frame = frames[self.index]
                self.index += 1
                return True, frame

            def release(self):
                return None

        with patch("video_utils.cv2.VideoCapture", FakeCapture):
            extracted = extract_png_frames_from_video(b"dummy", frame_count=2)

        self.assertEqual(len(extracted), 2)
        self.assertTrue(extracted[0][0].endswith(".png"))
        self.assertTrue(extracted[0][1].startswith(b"\x89PNG"))

    def test_build_coco_keypoints_predictions_from_frames(self) -> None:
        frame_bytes = _png_bytes()

        def fake_infer_pose_fn(image_tensor, conf_threshold=0.0):
            self.assertEqual(image_tensor.shape, (1, 3, 640, 640))
            return [
                {
                    "confidence": 0.9,
                    "bbox_xywh": [320.0, 320.0, 100.0, 200.0],
                    "keypoints": [
                        {"x": 10.0, "y": 20.0, "confidence": 0.8}
                    ] + [{"x": 0.0, "y": 0.0, "confidence": 0.0}] * 16,
                }
            ]

        result = build_coco_keypoints_predictions_from_frames(
            [("frame_000000.png", frame_bytes)],
            infer_pose_fn=fake_infer_pose_fn,
        )

        self.assertEqual(len(result["images"]), 1)
        self.assertEqual(len(result["annotations"]), 1)
        self.assertEqual(result["annotations"][0]["category_id"], 1)


if __name__ == "__main__":
    unittest.main()