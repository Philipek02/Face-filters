#!/usr/bin/env python3

from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np


FILTERS: Dict[str, Tuple[str, Path]] = {
    "g": ("Glasses", Path("filters/glasses.png")),
    "m": ("Mustache", Path("filters/mustache.png")),
}
DEFAULT_FILTER_KEY = "g"  # klawisz aktywowany na start; None = brak filtra
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


def load_filter(path: Path) -> np.ndarray:

    if not path.exists():
        raise FileNotFoundError(path)

    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"Nie udało się wczytać pliku: {path}")

    if len(img.shape) == 2:  # 1‑kanałowy (GRAY)
        print(f"[WARN] {path.name}: obraz 1‑kanałowy – konwertuję do BGRA.")
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:  # BGR bez alfy
        print(f"[WARN] {path.name}: brak kanału alfa – dodaję pełną alfę.")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    elif img.shape[2] != 4:
        raise ValueError(f"{path.name}: nieobsługiwany format (channels={img.shape[2]})")

    return img

def overlay_bgra(
    frame: np.ndarray,
    overlay: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
) -> None:
    """Nakłada PNG ‑> ROI o rozmiarze (w, h) z uwzględnieniem alfy."""
    overlay_resized = cv2.resize(overlay, (w, h), interpolation=cv2.INTER_AREA)

    b, g, r, a = cv2.split(overlay_resized)
    overlay_rgb = cv2.merge((b, g, r))
    mask = cv2.merge((a, a, a))

    roi = frame[y : y + h, x : x + w]

    bg = cv2.bitwise_and(roi, cv2.bitwise_not(mask))
    fg = cv2.bitwise_and(overlay_rgb, mask)

    frame[y : y + h, x : x + w] = cv2.add(bg, fg)



def main() -> None:
    parser = argparse.ArgumentParser(description="Prosty AR‑filtr na twarz (OpenCV)")
    parser.add_argument("--cam", type=int, default=0, help="Index kamery lub ścieżka do pliku video")
    parser.add_argument("--list-filters", action="store_true", help="Wyświetl dostępne filtry i wyjdź")
    args = parser.parse_args()

    if args.list_filters:
        print("Dostępne filtry:")
        for key, (label, path) in FILTERS.items():
            print(f"  {key} : {label} → {path}")
        return

    loaded_filters: Dict[str, np.ndarray] = {}
    for key, (_, path) in FILTERS.items():
        try:
            loaded_filters[key] = load_filter(path)
        except Exception as e:
            print(f"[ERROR] {e}. Filtr '{key}' będzie pominięty.")

    if not loaded_filters:
        raise RuntimeError("Brak poprawnie wczytanych filtrów!")

    active_key = DEFAULT_FILTER_KEY if DEFAULT_FILTER_KEY in loaded_filters else None

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise IOError(f"Nie można otworzyć kamery/pliku: {args.cam}")

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

    print("Sterowanie: g = okulary | m = wąsy | n = brak filtra | q = quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Brak klatki – koniec wideo?")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        if active_key in loaded_filters:
            filt = loaded_filters[active_key]
            for (x, y, w, h) in faces:
                pad_y = int(h * 0.15)
                pad_x = int(w * 0.05)
                y_ = max(0, y - pad_y)
                x_ = max(0, x - pad_x)
                h_ = min(frame.shape[0] - y_, h + 2 * pad_y)
                w_ = min(frame.shape[1] - x_, w + 2 * pad_x)
                overlay_bgra(frame, filt, x_, y_, w_, h_)

        cv2.imshow("Face Filter", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("n"):
            active_key = None
        elif chr(key) in loaded_filters:
            active_key = chr(key)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
