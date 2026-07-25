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
  `standalone-environments/torch-index-stacks.json` on R2.
- The desktop app validates entries default-deny at runtime; this file cannot
  point installs at arbitrary indexes.
- An explicit empty `"stacks": []` withdraws every index-served stack from
  all apps; deleting entries selectively withdraws just those.