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
