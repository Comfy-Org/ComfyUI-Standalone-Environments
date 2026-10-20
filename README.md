# ComfyUI-Launcher-Environments

Standalone environment definitions consumed by [ComfyUI Desktop 2.0](https://github.com/Comfy-Org/ComfyUI-Desktop-2.0-Beta) to provision Python environments for ComfyUI installations.

## Torch index stacks manifest

`torch-index-stacks.json` lists the index-served PyTorch stacks (pip-applied
tuples with no bundle artifact) that the desktop app offers in its PyTorch
picker, e.g. legacy CUDA builds for GPUs the current bundles dropped. The
desktop app fetches it from R2 on every update check, so new stacks ship by
editing this file - no app release needed.

- Edit via pull request; CI validates the file with
  `scripts/validate_torch_index_stacks.py` (the app silently drops invalid
  entries, so validation runs here where a mistake fails loudly).
- On merge to `main`, the `Publish Torch Index Stacks` workflow uploads it to
  `standalone-environments/torch-index-stacks.json` on R2. The desktop app
  also has a GCS fallback mirror that is synced outside this repository -
  until that sync is automated
  ([#10](https://github.com/Comfy-Org/ComfyUI-Standalone-Environments/issues/10)),
  fallback clients can serve a stale manifest after a withdrawal, so update
  the mirror alongside urgent changes.
- The desktop app validates entries default-deny at runtime; this file cannot
  point installs at arbitrary indexes.
- An explicit empty `"stacks": []` withdraws every index-served stack from
  all apps; deleting entries selectively withdraws just those.
- Stable entries never carry PEP 440 dev (nightly) versions - validation
  rejects them. Nightlies use `"kind": "pytorch-nightly-index"` and are
  managed exclusively by the scheduled `Refresh Nightly Torch Stacks`
  workflow (`scripts/refresh_nightly_stacks.py`), which re-resolves exact
  dated tuples from the live nightly indexes daily, validates, commits,
  and publishes. Do not hand-edit nightly entries: dated nightly wheels
  are purged from pytorch.org's index after roughly 60 days, so they are
  pins that decay - validation rejects any nightly entry older than 7
  days, and the desktop app stops offering entries ~45 days after their
  wheel date. The tag allowlist (`NIGHTLY_TAGS`, currently `cu132`) lives
  in the refresh script; old app versions ignore the nightly kind
  entirely.

## Building and testing environments locally (no publishing)

`scripts/build_local.py` reproduces the `Build Standalone Environment`
workflow for one variant on your machine, and `scripts/serve_local.py`
serves the result with the same URL layout as R2 so a dev desktop can
ingest it. Nothing is uploaded anywhere.

Prerequisites: Python 3, `git`, 7-Zip (`7z` on PATH, or the default
Windows install location), network access, and tens of GB of free disk.
Only variants matching the host OS can be built (pip resolves wheels for
the running platform) - for other platforms, run the workflow via
`workflow_dispatch` on a fork without R2 secrets, which skips every
publish step and leaves the archives on the draft GitHub release.

```powershell
# 1. Build a variant (output lands in dist/<variant>/<tag>/)
python scripts/build_local.py win-nvidia --comfyui-ref v0.4.61

# 2. Generate R2-style metadata and serve dist/ on 127.0.0.1:8000.
#    Also serves the working-tree torch-index-stacks.json, so local
#    manifest edits are testable the same way.
python scripts/serve_local.py

# 3. Point an unpackaged desktop at it (packaged builds ignore this)
cd ..\ComfyUI-Launcher
$env:COMFY_STANDALONE_BASE_URL = 'http://127.0.0.1:8000'; pnpm dev
```

While the override is active the desktop's GCS mirror fallback is
disabled, so a failure against the local server fails loudly instead of
silently falling back to production metadata. Local builds default to a
`<ref>-local1` tag; keep a `-local` suffix so a local artifact can never
shadow a real release tag. Keep `build_local.py`'s constants in sync
with the workflow matrix when it changes; `scripts/test_local_tooling.py`
covers the metadata generation
(`python -m unittest scripts.test_local_tooling`).


## Windows ARM64 native dependencies

The `beta-win-nvidia-arm64` environment includes OpenCV so custom nodes can
import `cv2` without compiling it locally. The environment workflow builds
`opencv-python` from the pinned public commit in `scripts/build_opencv_arm64.py`,
then consumes that run's wheel artifact. It does not redistribute NVIDIA's
internal evaluation archive or rely on an unpublished release URL.

The reusable `Build Windows ARM64 OpenCV` workflow also runs on relevant PRs.
It installs the wheel alongside the upstream PyPI ARM64 wheels for `blake3`
and `kornia-rs`, then checks native imports, a known BLAKE3 digest, color
conversion, and a lossless PNG encode/decode round-trip on Python 3.13.
The completed standalone bundle runs the same checks after stripping.
Older releases can still download the historical `win-arm64-wheels-v1` assets;
new builds no longer use those custom Rust wheels.

Local ARM64 builds use the same OpenCV build script and require native ARM64
Python 3.11 or later, git, and Visual Studio 2026 C++ ARM64 build tools (or
`CMAKE_GENERATOR` set to the installed Visual Studio version). Build
dependencies are installed into a separate temporary venv. A fresh OpenCV
source build adds build time; the resulting desktop environment contains the
installed wheel, not the compiler or the build venv.

This is the main, GUI-enabled OpenCV package. It has no `opencv_contrib`
modules and no FFmpeg video backend on Windows ARM64. Image round-trip checks
do not establish MP4/custom-node workflow compatibility; RTX Spark workflow
QA is still required before an environment release. Native wheel CI publishes
only a GitHub Actions artifact; publishing environments remains the existing
separate release workflow.
