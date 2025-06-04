"""
Usage
-----
```bash
python face_filter.py [--cam 0] [--show-boxes]
```
Controls during runtime
-----------------------
```
n – toggle all masks on/off
r – reset assignments and redraw masks
q – quit
```

you can add your own filters to filters dir

Add `--show-boxes` if you want to see the debug rectangles and face IDs.

"""
from __future__ import annotations

import argparse
import itertools
import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


FILTERS_DIR = Path("filters")
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


def load_filter(path: Path) -> np.ndarray:
    """Load a single PNG with transparency as BGRA array."""
    if not path.exists():
        raise FileNotFoundError(path)
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"Cannot read: {path}")
    if img.ndim == 2:  # grayscale → BGRA
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:  # add full‑alpha channel
        a = np.full((*img.shape[:2], 1), 255, dtype=img.dtype)
        img = np.concatenate((img, a), axis=2)
    elif img.shape[2] != 4:
        raise ValueError(f"Unsupported channel count: {img.shape[2]}")
    return img


def overlay_bgra(frame: np.ndarray, overlay: np.ndarray, x: int, y: int, w: int, h: int) -> None:
    """Overlay *overlay* on *frame* ROI with alpha blending."""
    overlay_resized = cv2.resize(overlay, (w, h), interpolation=cv2.INTER_AREA)
    b, g, r, a = cv2.split(overlay_resized)
    overlay_rgb = cv2.merge((b, g, r))
    mask = cv2.merge((a, a, a))

    roi = frame[y : y + h, x : x + w]
    bg = cv2.bitwise_and(roi, cv2.bitwise_not(mask))
    fg = cv2.bitwise_and(overlay_rgb, mask)
    frame[y : y + h, x : x + w] = cv2.add(bg, fg)


# SIMPLE CENTROID TRACKER
class CentroidTracker:
    def __init__(self, max_missing: int = 30):
        self.next_id = 0
        self.objects: Dict[int, Tuple[int, int]] = {}
        self.missing: Dict[int, int] = {}
        self.max_missing = max_missing

    def update(self, rects: List[Tuple[int, int, int, int]]) -> Dict[int, Tuple[int, int, int, int]]:
        """Update tracker with a list of bounding boxes (x1, y1, x2, y2)."""
        updated: Dict[int, Tuple[int, int, int, int]] = {}
        centroids = np.array([(int((x1 + x2) / 2), int((y1 + y2) / 2)) for x1, y1, x2, y2 in rects])

        if len(self.objects) == 0:
            for rect in rects:
                self.objects[self.next_id] = ((rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2)
                self.missing[self.next_id] = 0
                updated[self.next_id] = rect
                self.next_id += 1
            return updated

        object_ids = list(self.objects.keys())
        object_centroids = np.array(list(self.objects.values()))

        if len(rects) > 0:
            D = np.linalg.norm(object_centroids[:, None] - centroids[None, :], axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            assigned_rows, assigned_cols = set(), set()

            for row, col in zip(rows, cols):
                if row in assigned_rows or col in assigned_cols:
                    continue
                if D[row, col] > 50:
                    continue
                obj_id = object_ids[row]
                self.objects[obj_id] = centroids[col]
                self.missing[obj_id] = 0
                updated[obj_id] = rects[col]
                assigned_rows.add(row)
                assigned_cols.add(col)

            for col, rect in enumerate(rects):
                if col not in assigned_cols:
                    obj_id = self.next_id
                    self.objects[obj_id] = centroids[col]
                    self.missing[obj_id] = 0
                    updated[obj_id] = rect
                    self.next_id += 1

        for obj_id in list(self.objects.keys()):
            if obj_id not in updated:
                self.missing[obj_id] += 1
                if self.missing[obj_id] > self.max_missing:
                    del self.objects[obj_id]
                    del self.missing[obj_id]
        return updated




def main() -> None:
    parser = argparse.ArgumentParser(description="AR face filter with per‑person masks")
    parser.add_argument("--cam", type=int, default=0, help="Camera index or video file path")
    parser.add_argument("--show-boxes", action="store_true", help="Draw debug boxes + IDs")
    args = parser.parse_args()

    filter_paths = sorted(p for p in FILTERS_DIR.glob("*.png"))
    if not filter_paths:
        raise RuntimeError("No PNG filters found in 'filters/' directory!")
    filters = [load_filter(p) for p in filter_paths]
    print(f"[INFO] Loaded {len(filters)} filter(s).")

    tracker = CentroidTracker()
    id_to_filter: Dict[int, int] = {}
    filter_cycle = itertools.cycle(range(len(filters)))
    masks_enabled = True

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise IOError(f"Cannot open camera: {args.cam}")

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    print("Controls: n = toggle filters | r = reset assignments | q = quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("end of video")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        rects: List[Tuple[int, int, int, int]] = []
        for (x, y, w, h) in detected:
            rects.append((x, y, x + w, y + h))

        assignments = tracker.update(rects)

        # draw & overlay
        for obj_id, (x1, y1, x2, y2) in assignments.items():
            if obj_id not in id_to_filter:
                id_to_filter[obj_id] = next(filter_cycle)
            if masks_enabled:
                w, h = x2 - x1, y2 - y1
                pad_y = int(h * 0.15)
                pad_x = int(w * 0.05)
                y_ = max(0, y1 - pad_y)
                x_ = max(0, x1 - pad_x)
                h_ = min(frame.shape[0] - y_, h + 2 * pad_y)
                w_ = min(frame.shape[1] - x_, w + 2 * pad_x)
                overlay_bgra(frame, filters[id_to_filter[obj_id]], x_, y_, w_, h_)
            if args.show_boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"#{obj_id}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

        cv2.imshow("Face Filter", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("n"):
            masks_enabled = not masks_enabled
        elif key == ord("r"):
            id_to_filter.clear()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
