"""
Build a real-image INT8 calibration set for the Hailo optimizer.
Per your conversion doc (B.6): raw uint8 [0,255], RGB order, letterboxed
to 640x640 to match training/export preprocessing exactly.

Aim for ~1024 images drawn from your ACTUAL train set (both D-Fire and
the merged FireAndSmoke portion) for representative coverage -- 100 is
a functional minimum but full-quality optimization wants closer to 1024.
"""
import cv2
import numpy as np
import glob
import random

def letterbox(img, new_shape=640, color=(114, 114, 114)):
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw, dh = new_shape - new_unpad[0], new_shape - new_unpad[1]
    dw, dh = dw / 2, dh / 2
    img = cv2.resize(img, new_unpad)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

# EDIT this glob to point at your merged train/images folder
image_paths = glob.glob("/mnt/d/6 semester/Fire and Smoke/archive/train/images/*.jpg")
random.seed(0)
random.shuffle(image_paths)
image_paths = image_paths[:1024]  # cap at ~1024 per Hailo's recommendation

batch = []
for path in image_paths:
    img = cv2.imread(path)
    if img is None:
        continue
    img = letterbox(img, 640)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    batch.append(img)

arr = np.stack(batch).astype(np.uint8)  # (N, 640, 640, 3)
np.save("calib_set.npy", arr)
print(f"Saved calib_set.npy with {arr.shape[0]} images, shape {arr.shape}")
