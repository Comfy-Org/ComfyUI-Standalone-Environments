#!/usr/bin/env python3
"""Build OpenCV from pinned public sources on native Windows ARM64.

Requires an ARM64 CPython >= 3.11, git, and Visual Studio 2026 C++ ARM64
build tools (or set CMAKE_GENERATOR for another installed VS version).
Build dependencies live in a separate venv, not the bundle.
The returned wheel includes upstream's license and third-party notices.
"""

import argparse
import os
import platform
import subprocess
import sys
import tomllib
from pathlib import Path

# Same public source revision used for NVIDIA's WoA evaluation build.
OPENCV_REF = "c9cce3de2eae2039ebb0c22d13c9ed85a40e338f"


def find_wheel(directory):
    wheels = list(directory.glob("*.whl"))
    if len(wheels) != 1 or not wheels[0].name.endswith("-cp37-abi3-win_arm64.whl"):
        raise RuntimeError(f"Expected one cp37-abi3-win_arm64 wheel, found {wheels}")
    if not wheels[0].name.startswith("opencv_python-"):
        raise RuntimeError(f"Expected the main OpenCV package, found {wheels[0].name}")
    return wheels[0]


def build(work):
    if sys.platform != "win32" or platform.machine().upper() != "ARM64":
        raise SystemExit("OpenCV must be built with native Windows ARM64 Python")
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
    run("git", "-C", source, "submodule", "update", "--init", "--depth=1", "opencv")
    run(sys.executable, "-m", "venv", work / "venv")
    python = work / "venv" / "Scripts" / "python.exe"
    requirements = tomllib.loads((source / "pyproject.toml").read_text())["build-system"]["requires"]
    run(python, "-m", "pip", "install", *requirements,
        "cmake==4.4.3", "scikit-build==0.19.1", "wheel==0.45.1")
    env = {
        **os.environ,
        "CI_BUILD": "1",
        "ENABLE_CONTRIB": "0",
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
    print(f"Built {wheel.name} from {OPENCV_REF}", flush=True)
    return wheel


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    build(parser.parse_args().work_dir)
