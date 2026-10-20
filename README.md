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
