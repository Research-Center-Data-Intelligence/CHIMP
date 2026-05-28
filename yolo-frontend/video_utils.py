import os
import tempfile
import zipfile
from io import BytesIO

import cv2
import numpy as np
from PIL import Image

from config import DEFAULT_FRAME_COUNT
from utils import sample_evenly_spaced_indices


def _encode_frame_to_png_bytes(frame: np.ndarray) -> bytes:
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb_frame)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def extract_png_frames_from_video(video_bytes: bytes, frame_count: int = DEFAULT_FRAME_COUNT) -> list[tuple[str, bytes]]:
    if frame_count <= 0:
        raise ValueError("frame_count must be greater than zero")

    temp_path = None
    capture = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as temp_file:
            temp_file.write(video_bytes)
            temp_file.flush()
            temp_path = temp_file.name

        capture = cv2.VideoCapture(temp_path)
        if not capture.isOpened():
            raise ValueError("Could not open uploaded video")

        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_frames > 0:
            frame_indices = sample_evenly_spaced_indices(total_frames, frame_count)

            if not frame_indices:
                raise ValueError("Uploaded video did not contain any readable frames")

            extracted_frames: list[tuple[str, bytes]] = []
            current_index = 0
            target_positions = {index: position for position, index in enumerate(frame_indices)}

            while True:
                success, frame = capture.read()
                if not success:
                    break

                if current_index in target_positions:
                    frame_name = f"frame_{target_positions[current_index]:02d}_{current_index:06d}.png"
                    extracted_frames.append((frame_name, _encode_frame_to_png_bytes(frame)))

                    if len(extracted_frames) == len(frame_indices):
                        break

                current_index += 1

            if not extracted_frames:
                raise ValueError("No frames could be extracted from the uploaded video")

            return extracted_frames

        all_frames: list[np.ndarray] = []
        while True:
            success, frame = capture.read()
            if not success:
                break
            all_frames.append(frame)

        if not all_frames:
            raise ValueError("Uploaded video did not contain any readable frames")

        frame_indices = sample_evenly_spaced_indices(len(all_frames), frame_count)
        extracted_frames = []
        for output_position, frame_index in enumerate(frame_indices):
            frame_name = f"frame_{output_position:02d}_{frame_index:06d}.png"
            extracted_frames.append((frame_name, _encode_frame_to_png_bytes(all_frames[frame_index])))

        return extracted_frames
    finally:
        if capture is not None:
            capture.release()
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


def frames_to_zip_bytes(extracted_frames: list[tuple[str, bytes]]) -> bytes:
    output_zip = BytesIO()
    with zipfile.ZipFile(output_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for frame_name, frame_bytes in extracted_frames:
            archive.writestr(frame_name, frame_bytes)
    return output_zip.getvalue()