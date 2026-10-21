#!/usr/bin/env python3
"""Build OpenCV from pinned public sources on native Windows ARM64.

Requires an ARM64 CPython >= 3.11, git, and Visual Studio 2026 C++ ARM64
build tools (or set CMAKE_GENERATOR for another installed VS version).
Build dependencies live in a separate venv, not the bundle.
The contrib wheel includes upstream's license and third-party notices.
"""

import argparse
import base64
import hashlib
import os
import platform
import subprocess
import sys
import zipfile
from pathlib import Path

# Same public source revision used for NVIDIA's WoA evaluation build.
OPENCV_REF = "c9cce3de2eae2039ebb0c22d13c9ed85a40e338f"

# The contrib build has every main module plus contrib's, and provides cv2.
PROVIDER = "opencv-contrib-python"
# All four OpenCV distributions install the same cv2 module, but installers
# match requirements by distribution name, and PyPI has no Windows ARM64
# wheel for any of them. Empty wheels under the other names depend on the
# provider, so custom nodes that ask for them resolve to this build.
ALIASES = ("opencv-python", "opencv-python-headless", "opencv-contrib-python-headless")


def wheel_name(distribution):
    return distribution.replace("-", "_")


def find_wheel(directory):
    wheels = list(directory.glob("*.whl"))
    if len(wheels) != 1 or not wheels[0].name.endswith("-cp37-abi3-win_arm64.whl"):
        raise RuntimeError(f"Expected one cp37-abi3-win_arm64 wheel, found {wheels}")
    if not wheels[0].name.startswith(f"{wheel_name(PROVIDER)}-"):
        raise RuntimeError(f"Expected the {PROVIDER} package, found {wheels[0].name}")
    return wheels[0]


def write_alias_wheel(directory, distribution, version):
    """Write a wheel with no files whose only metadata pins the provider."""
    dist_info = f"{wheel_name(distribution)}-{version}.dist-info"
    files = {
        f"{dist_info}/METADATA": (
            "Metadata-Version: 2.1\n"
            f"Name: {distribution}\n"
            f"Version: {version}\n"
            f"Summary: Resolves to {PROVIDER} {version}, which provides cv2\n"
            f"Requires-Dist: {PROVIDER}=={version}\n"
        ),
        f"{dist_info}/WHEEL": (
            "Wheel-Version: 1.0\n"
            "Generator: build_opencv_arm64.py\n"
            "Root-Is-Purelib: true\n"
            "Tag: py3-none-any\n"
        ),
    }
    record = []
    for path, text in files.items():
        data = text.encode("utf-8")
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
        record.append(f"{path},sha256={digest.decode()},{len(data)}")
    record.append(f"{dist_info}/RECORD,,")
    files[f"{dist_info}/RECORD"] = "\n".join(record) + "\n"

    wheel = directory / f"{wheel_name(distribution)}-{version}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as zf:
        for path, text in files.items():
            # A fixed timestamp keeps the bytes identical across builds.
            info = zipfile.ZipInfo(path, date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o644 << 16
            zf.writestr(info, text)
    return wheel


def build(work):
    """Build the provider wheel plus its aliases; returns all wheel paths."""
    if sys.platform != "win32" or platform.machine().upper() != "ARM64":
        raise SystemExit("OpenCV must be built with native Windows ARM64 Python")
    if sys.version_info < (3, 11):
        raise SystemExit("Building OpenCV requires Python 3.11 or later")
    # Keep importing the local tooling compatible with older Python versions
    # on platforms that never build this wheel.
    import tomllib

    work = work.resolve()
    work.mkdir(parents=True, exist_ok=False)
    source = work / "source"
    output = work / "wheels"
    output.mkdir()

    def run(*args, **kwargs):
        subprocess.run([str(arg) for arg in args], check=True, **kwargs)

    # Fetch a commit, not a moving branch or the internal NVIDIA binary.
    run("git", "init", source)
    run("git", "-C", source, "remote", "add", "origin",
        "https://github.com/opencv/opencv-python.git")
    run("git", "-C", source, "fetch", "--depth=1", "origin", OPENCV_REF)
    run("git", "-C", source, "checkout", "--detach", "FETCH_HEAD")
    # find_version.py uses this suffix for the version pinned in requirements.
    run("git", "-C", source, "config", "core.abbrev", "7")
    # Shallow, at the commits the superproject records. setup.py's own
    # submodule update then has nothing left to fetch.
    run("git", "-C", source, "submodule", "update", "--init", "--depth=1",
        "opencv", "opencv_contrib")
    run(sys.executable, "-m", "venv", work / "venv")
    python = work / "venv" / "Scripts" / "python.exe"
    requirements = tomllib.loads((source / "pyproject.toml").read_text())["build-system"]["requires"]
    run(python, "-m", "pip", "install", *requirements,
        "cmake==4.4.3", "scikit-build==0.19.1", "wheel==0.45.1")
    env = {
        **os.environ,
        "CI_BUILD": "1",
        "ENABLE_CONTRIB": "1",
        "ENABLE_HEADLESS": "0",
        "ENABLE_ROLLING": "0",
        "CMAKE_BUILD_PARALLEL_LEVEL": "4",
    }
    # Upstream hard-codes VS2022; the native hosted runner now uses VS2026.
    # scikit-build gives these command-line CMake args precedence over setup.py.
    generator = os.environ.get("CMAKE_GENERATOR", "Visual Studio 18 2026")
    run(python, "setup.py", "bdist_wheel", "--py-limited-api=cp37",
        "--dist-dir", output, "--", "-G", generator, cwd=source, env=env)
    wheel = find_wheel(output)
    version = wheel.name.split("-")[1]
    aliases = [write_alias_wheel(output, name, version) for name in ALIASES]
    print(f"Built {wheel.name} from {OPENCV_REF}, plus aliases "
          f"{', '.join(ALIASES)}", flush=True)
    return [wheel, *aliases]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    build(parser.parse_args().work_dir)
