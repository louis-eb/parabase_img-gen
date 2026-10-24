#!/usr/bin/env python3
"""Turn a dataset folder from the data repo into an AI Toolkit dataset:
images (jpg/png/heic/heif) -> RGB JPEG, long edge <= MAX_PX, EXIF rotation applied;
the caption .txt with the same name is copied next to it.

usage: prep_dataset.py SRC_DIR DST_DIR [MAX_PX]"""
import os
import shutil
import sys

from PIL import Image, ImageOps

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pillow_heif = None

src, dst = sys.argv[1], sys.argv[2]
max_px = int(sys.argv[3]) if len(sys.argv) > 3 else 1600
os.makedirs(dst, exist_ok=True)
n, missing = 0, []
for f in sorted(os.listdir(src)):
    stem, ext = os.path.splitext(f)
    if ext.lower() not in (".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp"):
        continue
    cap = os.path.join(src, stem + ".txt")
    if not os.path.exists(cap):
        missing.append(f)
        continue
    out = os.path.join(dst, stem + ".jpg")
    if not os.path.exists(out):
        im = ImageOps.exif_transpose(Image.open(os.path.join(src, f))).convert("RGB")
        im.thumbnail((max_px, max_px), Image.LANCZOS)
        im.save(out, "JPEG", quality=94)
    shutil.copy2(cap, os.path.join(dst, stem + ".txt"))
    n += 1
print(f"{n} images ready in {dst}")
if missing:
    print("no caption, skipped:", missing)
