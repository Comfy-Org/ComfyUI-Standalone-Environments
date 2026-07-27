#!/usr/bin/env python3
"""Build one standalone-environment archive locally, without publishing.

Reproduces the steps of .github/workflows/build-standalone-env.yml for a
single variant on the host platform: download python-build-standalone,
clone ComfyUI, pip-install the vendor + ComfyUI + manager requirements,
bundle uv, strip, smoke-test, generate manifest.json, and package the
archive with the exact 7z parameters CI uses.

The output layout matches what serve_local.py ingests:

    <out>/<variant>/<tag>/comfyui-standalone-<variant>-<tag>.7z
    <out>/<variant>/<tag>/manifest.json

Nothing is uploaded anywhere. Keep the steps here in sync with the
workflow when it changes.

Cross-building is not possible (pip resolves wheels for the running
platform), so only variants matching the host OS are accepted; use a
workflow_dispatch on a fork (no R2 secrets -> publish steps skip) to get
artifacts for other platforms.

Usage:
    python scripts/build_local.py win-amd --comfyui-ref v0.4.61
    python scripts/build_local.py win-nvidia          # latest stable ComfyUI
"""

import argparse
import json
import platform
import re
import shutil
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Mirror the workflow's top-level env + matrix. Update together.
PYTHON_VERSION = "3.13.12"
PBS_RELEASE = "20260211"

# Per-variant python_version overrides from the matrix (currently none;
# e.g. win-amd was pinned to 3.12 while on the universal ROCm 7.2.1 wheels).
VARIANT_PYTHON_VERSIONS = {}

PYTHON_PLATFORMS = {
    "win": "x86_64-pc-windows-msvc",
    "linux": "x86_64-unknown-linux-gnu",
    "mac": "aarch64-apple-darwin",
}

VARIANTS = {
    "win-nvidia": "requirements-nvidia.txt",
    "win-intel-xpu": "requirements-intel.txt",
    "win-amd": "requirements-amd-win.txt",
    "win-cpu": "requirements-cpu.txt",
    "linux-nvidia": "requirements-nvidia.txt",
    "linux-amd": "requirements-amd.txt",
    "linux-intel-xpu": "requirements-intel.txt",
    "mac-mps": "requirements-mac.txt",
}

HOST_OS = {"win32": "win", "linux": "linux", "darwin": "mac"}.get(sys.platform)

# PYTHON_PLATFORMS above pins the CPU architecture per OS; a host with a
# different machine type would silently get a foreign-arch environment.
HOST_ARCHES = {"win": {"AMD64"}, "linux": {"x86_64"}, "mac": {"arm64"}}

# Keep in sync with SAFE_SEGMENT in serve_local.py / r2Catalog.ts: the tag
# becomes a path segment in the served URL and the archive filename.
SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

# Exact CI compression parameters - keeps local sizes representative.
SEVENZ_ARGS = ["-t7z", "-m0=lzma2", "-mx=3", "-mfb=32", "-md=16m", "-ms=on", "-mf=off"]


def variant_os(variant):
    return variant.split("-", 1)[0]


def variant_python_version(variant):
    return VARIANT_PYTHON_VERSIONS.get(variant, PYTHON_VERSION)


def is_safe_tag(tag):
    return bool(SAFE_SEGMENT.match(tag)) and ".." not in tag


def build_number_from_tag(tag):
    """CI extracts the numeric -env<N> suffix; anything else is build 0."""
    m = re.search(r"-env(\d+)$", tag)
    return int(m.group(1)) if m else 0


def archive_filename(variant, tag):
    ext = ".tar.gz" if variant_os(variant) == "mac" else ".7z"
    return f"comfyui-standalone-{variant}-{tag}{ext}"


def build_manifest(variant, tag, comfyui_ref, comfyui_commit, versions, vendor_req_name, req_contents):
    """The manifest.json shape the workflow's 'Generate build manifest' step
    emits; serve_local.py derives R2-style release entries from it."""
    return {
        "id": variant,
        "version": tag,
        "build": build_number_from_tag(tag),
        "archive_base": f"comfyui-standalone-{variant}-{tag}",
        # CI archives the manifest with an empty files list (the list is only
        # populated later for the GitHub-release copy, after volume-splitting).
        "files": [],
        "comfyui_ref": comfyui_ref,
        "comfyui_commit": comfyui_commit,
        "python_version": variant_python_version(variant),
        "torch_version": versions["torch"],
        "torchvision_version": versions["torchvision"],
        "torchaudio_version": versions["torchaudio"],
        "pbs_release": PBS_RELEASE,
        "uv_version": versions["uv"],
        "vendor_requirements": vendor_req_name,
        "vendor_requirements_content": req_contents["vendor"],
        "comfyui_requirements_content": req_contents["comfyui"],
        "manager_requirements_content": req_contents["manager"],
        "build_date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def run(cmd, **kwargs):
    print(f"+ {' '.join(str(c) for c in cmd)}", flush=True)
    return subprocess.run(cmd, check=True, **kwargs)


def download(url, dest):
    print(f"Downloading {url}", flush=True)
    with urllib.request.urlopen(url) as resp, open(dest, "wb") as fh:
        shutil.copyfileobj(resp, fh)


def resolve_latest_comfyui_ref():
    url = "https://api.github.com/repos/Comfy-Org/ComfyUI/releases/latest"
    with urllib.request.urlopen(url) as resp:
        tag = json.load(resp).get("tag_name")
    if not tag:
        raise SystemExit("Failed to resolve latest ComfyUI release tag; pass --comfyui-ref")
    print(f"Auto-detected latest stable ComfyUI release: {tag}")
    return tag


def env_python(env_dir):
    return env_dir / ("python.exe" if HOST_OS == "win" else "bin/python3")


def fetch_python_standalone(work, python_version):
    platform = PYTHON_PLATFORMS[HOST_OS]
    name = f"cpython-{python_version}+{PBS_RELEASE}-{platform}-install_only.tar.gz"
    url = f"https://github.com/astral-sh/python-build-standalone/releases/download/{PBS_RELEASE}/{name}"
    tarball = work / "python-standalone.tar.gz"
    download(url, tarball)
    extract = work / "python-extract"
    extract.mkdir()
    run(["tar", "-xzf", str(tarball), "-C", str(extract)])
    (extract / "python").rename(work / "standalone-env")
    tarball.unlink()
    shutil.rmtree(extract)


def clone_comfyui(work, ref):
    run(["git", "clone", "--depth", "1", "--branch", ref,
         "https://github.com/Comfy-Org/ComfyUI.git", str(work / "ComfyUI")])


def install_requirements(work, vendor_req_name):
    final = work / "requirements_final.txt"
    vendor = (REPO_ROOT / vendor_req_name).read_text(encoding="utf-8")
    final.write_text(
        vendor.rstrip("\n")
        + "\n-r ComfyUI/requirements.txt\n-r ComfyUI/manager_requirements.txt\n",
        encoding="utf-8",
    )
    python = env_python(work / "standalone-env")
    if HOST_OS != "win":
        python.chmod(python.stat().st_mode | 0o111)
    subprocess.run([str(python), "-m", "ensurepip", "--upgrade"], check=False,
                   capture_output=True)
    run([str(python), "-m", "pip", "install", "--upgrade", "pip"], cwd=work)
    run([str(python), "-m", "pip", "install", "--no-cache-dir",
         "-r", "requirements_final.txt", "pygit2"], cwd=work)


def bundle_uv(work):
    env = work / "standalone-env"
    base = "https://github.com/astral-sh/uv/releases/latest/download"
    if HOST_OS == "win":
        download(f"{base}/uv-x86_64-pc-windows-msvc.zip", work / "uv.zip")
        import zipfile
        with zipfile.ZipFile(work / "uv.zip") as zf:
            with zf.open("uv.exe") as src, open(env / "uv.exe", "wb") as dst:
                shutil.copyfileobj(src, dst)
        (work / "uv.zip").unlink()
        return env / "uv.exe"
    archive = "uv-aarch64-apple-darwin.tar.gz" if HOST_OS == "mac" else "uv-x86_64-unknown-linux-gnu.tar.gz"
    download(f"{base}/{archive}", work / "uv.tar.gz")
    extract = work / "uv-extract"
    extract.mkdir()
    run(["tar", "-xzf", str(work / "uv.tar.gz"), "-C", str(extract), "--strip-components=1"])
    shutil.copy2(extract / "uv", env / "bin" / "uv")
    (work / "uv.tar.gz").unlink()
    shutil.rmtree(extract)
    return env / "bin" / "uv"


def site_packages_dir(env):
    if HOST_OS == "win":
        return env / "Lib" / "site-packages"
    return next((env / "lib").glob("python*/site-packages"))


def strip_environment(work):
    """The workflow's 'Strip unnecessary files' step: dev-only payload that
    the app never needs at runtime."""
    env = work / "standalone-env"
    site = site_packages_dir(env)
    for pattern in ["torch/lib/*.lib"]:
        for f in site.glob(pattern):
            f.unlink()
    for f in site.rglob("*.a"):
        f.unlink()
    for sub in ["torch/include", "torch/share", "caffe2",
                "torch/_inductor/autoheuristic/datasets", "torch/test"]:
        shutil.rmtree(site / sub, ignore_errors=True)
    for cache in list(env.rglob("__pycache__")):
        shutil.rmtree(cache, ignore_errors=True)
    for pyc in env.rglob("*.pyc"):
        pyc.unlink(missing_ok=True)
    # Debug-symbol stripping (CI does this on linux/mac) only trims size;
    # skip silently when binutils' strip is unavailable.
    if HOST_OS != "win" and shutil.which("strip"):
        subprocess.run(f'find "{env}" -name "*.so" -exec strip --strip-debug {{}} + 2>/dev/null',
                       shell=True, check=False)
        if HOST_OS == "mac":
            subprocess.run(f'find "{env}" -name "*.dylib" -exec strip -x {{}} + 2>/dev/null',
                           shell=True, check=False)


def smoke_test(work):
    python = env_python(work / "standalone-env")
    run([str(python), "-c",
         "import torch; print('torch', torch.__version__);"
         "import tqdm; print('tqdm', tqdm.__version__);"
         "import transformers; print('transformers', transformers.__version__);"
         "import pygit2; print('pygit2', pygit2.__version__);"
         "print('All core imports OK')"], cwd=work)


def introspect_versions(work, uv_bin):
    python = env_python(work / "standalone-env")

    def pkg_version(name):
        r = subprocess.run([str(python), "-c", f"import {name}; print({name}.__version__)"],
                           capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else "N/A"

    uv_out = subprocess.run([str(uv_bin), "--version"], capture_output=True, text=True)
    return {
        "torch": pkg_version("torch"),
        "torchvision": pkg_version("torchvision"),
        "torchaudio": pkg_version("torchaudio"),
        "uv": uv_out.stdout.split()[1] if uv_out.returncode == 0 else "N/A",
    }


def find_7z():
    for candidate in ["7z", "7za", r"C:\Program Files\7-Zip\7z.exe"]:
        found = shutil.which(candidate) or (candidate if Path(candidate).is_file() else None)
        if found:
            return found
    raise SystemExit("7z not found on PATH (install 7-Zip / p7zip)")


def package(work, variant, tag, vendor_req_name, out_dir):
    filename = archive_filename(variant, tag)
    # The vendor requirements file rides inside the archive like in CI.
    shutil.copy2(REPO_ROOT / vendor_req_name, work / vendor_req_name)
    contents = ["standalone-env", "ComfyUI", "manifest.json", vendor_req_name]
    if variant_os(variant) == "mac":
        run(["tar", "-czf", filename, *contents], cwd=work)
    else:
        run([find_7z(), "a", *SEVENZ_ARGS, filename, *contents], cwd=work)
    dest = out_dir / variant / tag
    dest.mkdir(parents=True, exist_ok=True)
    shutil.move(str(work / filename), dest / filename)
    shutil.copy2(work / "manifest.json", dest / "manifest.json")
    return dest / filename


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("variant", choices=sorted(VARIANTS))
    parser.add_argument("--comfyui-ref", default="",
                        help="ComfyUI branch/tag (default: latest stable release)")
    parser.add_argument("--tag", default="",
                        help="Release tag (default: <comfyui-ref>-local1; use a"
                             " -local suffix so it can never shadow a real tag)")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "dist",
                        help="Output root, consumed by serve_local.py (default: dist/)")
    parser.add_argument("--work-dir", type=Path, default=None,
                        help="Scratch dir (default: build/<variant>-<tag>; kept for inspection)")
    args = parser.parse_args()

    if HOST_OS is None or variant_os(args.variant) != HOST_OS:
        raise SystemExit(
            f"{args.variant} cannot be built on this host (pip resolves wheels for the"
            " running platform). Use workflow_dispatch on a fork for other platforms.")
    if platform.machine() not in HOST_ARCHES[HOST_OS]:
        raise SystemExit(
            f"{args.variant} targets {PYTHON_PLATFORMS[HOST_OS]} but this machine is"
            f" {platform.machine()}")

    ref = args.comfyui_ref or resolve_latest_comfyui_ref()
    tag = args.tag or f"{ref}-local1"
    # Fail on unsafe tags now, not after a multi-GB build: the tag becomes a
    # path segment and serve_local.py would reject it anyway.
    if not is_safe_tag(tag):
        raise SystemExit(
            f"Tag {tag!r} is not a safe path segment (letters, digits, '.', '_', '-')."
            " Branch refs with '/' need an explicit --tag, e.g. --tag mybranch-local1")
    work = (args.work_dir or (REPO_ROOT / "build" / f"{args.variant}-{tag}")).resolve()
    args.out_dir = args.out_dir.resolve()
    if work.exists():
        raise SystemExit(f"Work dir already exists, remove it first: {work}")
    work.mkdir(parents=True)

    vendor_req_name = VARIANTS[args.variant]
    fetch_python_standalone(work, variant_python_version(args.variant))
    clone_comfyui(work, ref)
    install_requirements(work, vendor_req_name)
    uv_bin = bundle_uv(work)
    strip_environment(work)
    smoke_test(work)

    commit = run(["git", "-C", str(work / "ComfyUI"), "rev-parse", "HEAD"],
                 capture_output=True, text=True).stdout.strip()
    versions = introspect_versions(work, uv_bin)
    req_contents = {
        "vendor": (REPO_ROOT / vendor_req_name).read_text(encoding="utf-8").strip(),
        "comfyui": (work / "ComfyUI" / "requirements.txt").read_text(encoding="utf-8").strip(),
        "manager": (work / "ComfyUI" / "manager_requirements.txt").read_text(encoding="utf-8").strip(),
    }
    manifest = build_manifest(args.variant, tag, ref, commit, versions,
                              vendor_req_name, req_contents)
    (work / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    archive = package(work, args.variant, tag, vendor_req_name, args.out_dir)
    print(f"\nBuilt: {archive} ({archive.stat().st_size} bytes)")
    print(f"Work dir kept for inspection: {work}")
    print(f"Serve it: python scripts/serve_local.py --root {args.out_dir}")


if __name__ == "__main__":
    main()
