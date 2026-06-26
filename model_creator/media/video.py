from __future__ import annotations

import uuid
from pathlib import Path

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


def extract_frames(project_path: Path, video_path: Path, video_id: str, every_n_frames: int) -> list[dict]:
    if cv2 is None:
        raise RuntimeError("opencv-python is required to extract frames")
    if every_n_frames < 1:
        raise ValueError("every_n_frames must be >= 1")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {video_path}")

    images_dir = project_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    images: list[dict] = []
    frame_index = 0
    saved_index = 0
    stem = video_path.stem

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % every_n_frames == 0:
            image_id = str(uuid.uuid4())
            filename = f"{stem}_frame_{frame_index:08d}.jpg"
            rel_path = f"images/{filename}"
            output_path = project_path / rel_path
            cv2.imwrite(str(output_path), frame)
            height, width = frame.shape[:2]
            images.append(
                {
                    "id": image_id,
                    "file": rel_path,
                    "width": int(width),
                    "height": int(height),
                    "video_id": video_id,
                    "source_frame": frame_index,
                    "sequence": saved_index,
                }
            )
            saved_index += 1
        frame_index += 1

    capture.release()
    return images

