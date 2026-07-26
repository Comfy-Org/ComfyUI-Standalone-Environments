#!/usr/bin/env python3
"""Serve locally built standalone-environment archives to a dev desktop.

Scans a directory of build_local.py outputs (<root>/<variant>/<tag>/
containing the archive + manifest.json), generates the R2-layout metadata
the desktop app fetches (latest.json, <variant>/releases.json), copies the
working-tree torch-index-stacks.json alongside, and serves the whole tree
over plain HTTP.

Point an unpackaged desktop at it (packaged builds ignore the override):

    $env:COMFY_STANDALONE_BASE_URL = 'http://127.0.0.1:8000'; pnpm dev

The generated entries mirror the 'Update R2 release metadata' step of
.github/workflows/build-standalone-env.yml, and are checked against the
same allowlists the app's r2Catalog validator enforces so a bad tag fails
here instead of silently vanishing from the install wizard.

Usage:
    python scripts/serve_local.py                # serve dist/ on :8000
    python scripts/serve_local.py --no-serve    # just (re)generate metadata
"""

import argparse
import http.server
import json
import re
import shutil
import sys
from functools import partial
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Keep in sync with SAFE_SEGMENT / isSafeFilename in the desktop's
# r2Catalog.ts - entries failing these are silently dropped by the app.
SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def release_entry(manifest, filename, size):
    """One releases.json entry, shaped exactly like the CI publish step
    builds from a per-variant manifest.json."""
    return {
        "tag": manifest["version"],
        "comfyui_version": manifest["comfyui_ref"],
        "comfyui_commit": manifest["comfyui_commit"],
        "build": manifest.get("build", 0),
        "date": manifest["build_date"],
        "file": filename,
        "size": size,
        "python_version": manifest["python_version"],
        "torch_version": manifest["torch_version"],
        "torchvision_version": manifest["torchvision_version"],
        "torchaudio_version": manifest["torchaudio_version"],
    }


def check_entry_servable(variant, entry):
    """Fail loudly on anything the desktop's validator would silently drop."""
    problems = []
    for field in ["tag", "file"]:
        if not SAFE_SEGMENT.match(entry[field]) or ".." in entry[field]:
            problems.append(f"{field} {entry[field]!r} fails the app's path allowlist")
    if not SAFE_SEGMENT.match(variant):
        problems.append(f"variant dir {variant!r} fails the app's path allowlist")
    if entry["size"] <= 0:
        problems.append("archive size must be positive")
    return problems


def discover(root):
    """Find (variant, entry) pairs for every <root>/<variant>/<tag>/manifest.json
    that has its archive sitting next to it."""
    found = []
    for manifest_path in sorted(root.glob("*/*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        variant = manifest["id"]
        tag_dir = manifest_path.parent
        archives = [p for p in tag_dir.iterdir()
                    if p.name.startswith(manifest["archive_base"]) and p.name != "manifest.json"]
        if not archives:
            print(f"WARNING: no archive next to {manifest_path}, skipping")
            continue
        if len(archives) > 1:
            raise SystemExit(f"Multiple archives next to {manifest_path}: {archives}")
        archive = archives[0]
        entry = release_entry(manifest, archive.name, archive.stat().st_size)
        problems = check_entry_servable(variant, entry)
        if problems:
            raise SystemExit(f"{manifest_path}: " + "; ".join(problems))
        if manifest_path.parent.parent.name != variant or manifest_path.parent.name != entry["tag"]:
            raise SystemExit(
                f"{manifest_path}: directory layout must be <root>/{variant}/{entry['tag']}/"
                " to match the URL the generated metadata points at")
        found.append((variant, entry))
    return found


def build_metadata(found):
    """(latest.json dict, {variant: releases.json dict}) from discovered
    entries - newest per vendor first, like the CI merge step."""
    vendors = {}
    for variant, entry in found:
        vendors.setdefault(variant, []).append(entry)
    releases = {}
    latest = {}
    for variant, entries in vendors.items():
        entries.sort(key=lambda e: e["date"], reverse=True)
        releases[variant] = {"releases": entries}
        latest[variant] = entries[0]
    return latest, releases


def generate(root, stacks_manifest):
    found = discover(root)
    if not found:
        raise SystemExit(f"No built environments under {root} - run build_local.py first")
    latest, releases = build_metadata(found)
    for variant, doc in releases.items():
        (root / variant / "releases.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
    (root / "latest.json").write_text(json.dumps(latest, indent=2), encoding="utf-8")
    if stacks_manifest.is_file():
        shutil.copy2(stacks_manifest, root / "torch-index-stacks.json")
    else:
        print(f"WARNING: {stacks_manifest} not found; the PyTorch picker will have no entries")
    for variant, entry in sorted(latest.items()):
        print(f"  {variant}: {entry['tag']} ({entry['file']}, {entry['size']} bytes)")
    return latest


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "dist",
                        help="Directory of build_local.py outputs (default: dist/)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--stacks-manifest", type=Path,
                        default=REPO_ROOT / "torch-index-stacks.json",
                        help="torch-index-stacks.json to serve (default: working tree copy,"
                             " so local manifest edits are testable too)")
    parser.add_argument("--no-serve", action="store_true",
                        help="Only (re)generate latest.json / releases.json")
    args = parser.parse_args()

    print(f"Generating metadata under {args.root}:")
    generate(args.root, args.stacks_manifest)
    if args.no_serve:
        return

    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(args.root))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    # Advertise the literal bound address: 'localhost' can resolve to ::1
    # first on Windows while this server is IPv4-only.
    print(f"\nServing {args.root} at http://127.0.0.1:{args.port}")
    print("Point an unpackaged desktop at it:")
    print(f"  $env:COMFY_STANDALONE_BASE_URL = 'http://127.0.0.1:{args.port}'; pnpm dev")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
