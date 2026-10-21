#!/usr/bin/env python3
"""Exercise native dependencies on the bundle interpreter, including cv2 I/O."""

import importlib.metadata
import platform

import blake3
import cv2
import kornia_rs  # noqa: F401 - loading the native extension is the check
import numpy as np

from build_opencv_arm64 import ALIASES, PROVIDER

assert platform.machine().upper() == "ARM64", platform.machine()
assert importlib.metadata.version("blake3") == "1.0.11"
assert importlib.metadata.version("kornia-rs") == "0.1.15"
assert blake3.blake3(b"abc").hexdigest() == (
    "6437b3ac38465133ffb63b75273a8db548c558465d79db03fd359c6cd5bd9d85"
)
assert cv2.__version__.startswith("4.14.0"), cv2.__version__
opencv_version = importlib.metadata.version(PROVIDER)
for alias in ALIASES:
    assert importlib.metadata.version(alias) == opencv_version, alias
image = np.zeros((16, 16, 3), dtype=np.uint8)
image[4:12, 4:12] = (0, 128, 255)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
assert gray.shape == (16, 16) and gray[8, 8] > gray[0, 0]
ok, encoded = cv2.imencode(".png", image)
assert ok
np.testing.assert_array_equal(cv2.imdecode(encoded, cv2.IMREAD_COLOR), image)
# ximgproc ships only in contrib.
assert cv2.ximgproc.guidedFilter(image, image, 2, 1e-2).shape == image.shape
print(cv2.getBuildInformation())
print("OpenCV contrib PNG round-trip and aliases, blake3 hashing, and kornia-rs"
      " ARM64 imports OK")
