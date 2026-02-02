import os
import pickle
from typing import List

import cv2
import face_recognition  # type: ignore

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FACES_DIR = os.path.join(BASE_DIR, "models", "faces")
OUTPUT_PATH = os.path.join(BASE_DIR, "models", "encodings.pickle")


def iter_images() -> List[str]:
    images: List[str] = []
    if not os.path.isdir(FACES_DIR):
        return images
    for person in os.listdir(FACES_DIR):
        person_dir = os.path.join(FACES_DIR, person)
        if not os.path.isdir(person_dir):
            continue
        for file in os.listdir(person_dir):
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                images.append(os.path.join(person_dir, file))
    return images


def main() -> None:
    encodings = []
    names = []
    for img_path in iter_images():
        name = os.path.basename(os.path.dirname(img_path))
        image = cv2.imread(img_path)
        if image is None:
            continue
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, model="hog")
        if not boxes:
            continue
        encs = face_recognition.face_encodings(rgb, boxes)
        for enc in encs:
            encodings.append(enc)
            names.append(name)

    data = {"encodings": encodings, "names": names}
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved encodings: {OUTPUT_PATH} (count={len(names)})")


if __name__ == "__main__":
    main()
