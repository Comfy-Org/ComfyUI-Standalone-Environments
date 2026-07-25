#!/usr/bin/env python3
"""Validate torch-index-stacks.json before publishing to R2.

Mirrors the desktop app's default-deny remote-manifest validator
(src/main/sources/standalone/torchIndexManifest.ts in Comfy-Desktop).
The app silently DROPS any entry that fails validation, so a typo would
just make the stack vanish from the picker with no error anywhere; this
script turns that into a failing check at authoring time instead.

Strictness differs from the app on purpose: the app drops invalid entries
one by one (forward compat with future entry types), while this script
fails the whole file on any problem - nothing invalid should ever be
committed for the CURRENT schema.

Keep the rules here in sync with the app validator when the schema evolves.
"""

import json
import re
import sys

SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
# Package versions end up in pip `pkg==version` arguments - allowlist.
SAFE_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+]*$")
# PEP 440 dev release (any accepted spelling: .dev20260720, bare .dev,
# compact dev1), checked against the public version (local tag stripped).
# Keep in sync with isDevVersion() in the app's torchStackTypes.ts.
DEV_RELEASE = re.compile(r"(\d|[._-])dev\d*$", re.IGNORECASE)
PYTHON_ABI = re.compile(r"^\d+\.\d+$")
ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}")
CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")
ACCELS = {"nvidia", "amd", "intel-xpu", "cpu", "mps"}
PLATFORMS = {"win32", "linux", "darwin"}
NOTE_MAX_LENGTH = 300


def local_tag(version):
    """Local version tag after `+` (`2.11.0+cu126` -> `cu126`), '' if none."""
    if not isinstance(version, str) or "+" not in version:
        return ""
    return version.split("+", 1)[1].lower()


def public_version(version):
    return version.split("+", 1)[0]


def check_entry(i, r, errors):
    def fail(msg):
        errors.append(f"stacks[{i}]: {msg}")

    if not isinstance(r, dict):
        fail("entry is not an object")
        return

    index_tag = r.get("indexTag")
    if not isinstance(index_tag, str) or not SAFE_SEGMENT.match(index_tag):
        fail(f"indexTag missing or unsafe: {index_tag!r}")
        return

    accel = r.get("accel")
    if accel not in ACCELS:
        fail(f"accel must be one of {sorted(ACCELS)}, got {accel!r}")
        return

    # `kind` must match the mechanism the app derives from the accelerator.
    expected_kind = "pypi" if accel == "mps" else "pytorch-index"
    if "kind" in r and r["kind"] != expected_kind:
        fail(f"kind must be {expected_kind!r} for accel {accel!r} (or omitted), got {r['kind']!r}")

    platforms = r.get("platforms")
    if not isinstance(platforms, list) or not platforms:
        fail("platforms must be a non-empty array")
        return
    for p in platforms:
        if p not in PLATFORMS:
            fail(f"unknown platform {p!r} (must be one of {sorted(PLATFORMS)})")

    pkgs = r.get("packages")
    if not isinstance(pkgs, dict):
        fail("packages missing or not an object")
        return
    torch = pkgs.get("torch")
    if not isinstance(torch, str) or not SAFE_VERSION.match(torch):
        fail(f"packages.torch missing or unsafe: {torch!r}")
        return
    for opt in ("torchvision", "torchaudio"):
        v = pkgs.get(opt)
        if v is not None and (not isinstance(v, str) or not SAFE_VERSION.match(v)):
            fail(f"packages.{opt} unsafe: {v!r}")

    # Nightly (dev) builds live in a separate index namespace with ~60-day
    # retention - a decaying promise schema 1 cannot express: older app
    # versions would derive the STABLE index from the local tag, and even
    # on apps that know the nightly namespace the entry rots as the wheel
    # is purged. Exposing nightlies needs a future schema/kind with refresh
    # automation behind it; until then reject them outright.
    for name in ("torch", "torchvision", "torchaudio"):
        v = pkgs.get(name)
        if isinstance(v, str) and DEV_RELEASE.search(public_version(v)):
            fail(
                f"packages.{name} is a PEP 440 dev (nightly) release: {v!r} - "
                "schema 1 is stable index stacks only; nightlies need a future "
                "manifest kind with automated refresh"
            )

    # One coherent source per accelerator: the accel must name an index tag
    # it can actually be served from, and the torch local tag must agree
    # with it - pip installs from whatever index the LOCAL TAG derives, so a
    # disagreeing entry would lie about its install source.
    tag_ok = (
        re.match(r"^cu\d+$", index_tag) if accel == "nvidia"
        else re.match(r"^rocm[\d.]+$", index_tag) if accel == "amd"
        else index_tag == "xpu" if accel == "intel-xpu"
        else index_tag == "cpu" if accel == "cpu"
        else index_tag == "pypi"
    )
    if not tag_ok:
        fail(f"indexTag {index_tag!r} is not valid for accel {accel!r}")

    torch_tag = local_tag(torch)
    if accel == "mps":
        # The only PyPI-served accel: untagged tuple, mac-only.
        if torch_tag != "":
            fail(f"mps tuples must be untagged, got torch {torch!r}")
        if any(p != "darwin" for p in platforms):
            fail("mps stacks must be darwin-only")
    elif torch_tag != index_tag:
        fail(f"torch local tag {torch_tag!r} does not match indexTag {index_tag!r}")

    # pytorch.org publishes no Windows ROCm wheels; AMD's SDK channel is not
    # a mechanism schema 1 can express.
    if accel == "amd" and "win32" in platforms:
        fail("amd stacks cannot target win32 in schema 1 (no Windows ROCm wheels on pytorch.org)")

    # Companion packages install from the same index - same tag (or none).
    for opt in ("torchvision", "torchaudio"):
        companion_tag = local_tag(pkgs.get(opt))
        if companion_tag != "" and companion_tag != torch_tag:
            fail(f"packages.{opt} local tag {companion_tag!r} does not match torch tag {torch_tag!r}")

    date = r.get("date")
    if not isinstance(date, str) or not ISO_DATE.match(date):
        fail(f"date missing or not ISO (YYYY-MM-DD): {date!r}")

    cap = r.get("computeCap")
    if cap is not None:
        if (
            not isinstance(cap, dict)
            or not isinstance(cap.get("min"), (int, float))
            or not isinstance(cap.get("max"), (int, float))
            or isinstance(cap.get("min"), bool)
            or isinstance(cap.get("max"), bool)
        ):
            fail(f"computeCap must be {{min: number, max: number}}: {cap!r}")
        elif cap["min"] > cap["max"]:
            fail(f"computeCap.min {cap['min']} > max {cap['max']}")

    abis = r.get("pythonAbis")
    if abis is not None:
        # Present-but-empty is ambiguous (the app treats empty as
        # unrestricted) - reject rather than silently widen.
        if not isinstance(abis, list) or not abis:
            fail("pythonAbis must be a non-empty array when present")
        elif not all(isinstance(a, str) and PYTHON_ABI.match(a) for a in abis):
            fail(f"pythonAbis entries must be 'major.minor' strings: {abis!r}")

    note_key = r.get("noteKey")
    if note_key is not None and (not isinstance(note_key, str) or not SAFE_SEGMENT.match(note_key)):
        fail(f"noteKey unsafe: {note_key!r}")

    note = r.get("note")
    if note is not None and (
        not isinstance(note, str) or len(note) > NOTE_MAX_LENGTH or CONTROL_CHARS.search(note)
    ):
        fail("note must be a plain string of at most 300 chars with no control characters")


def main(path):
    with open(path, encoding="utf-8") as fh:
        try:
            doc = json.load(fh)
        except json.JSONDecodeError as e:
            print(f"ERROR: {path} is not valid JSON: {e}", file=sys.stderr)
            return 1

    errors = []
    if not isinstance(doc, dict):
        errors.append("document must be a JSON object")
    else:
        if doc.get("schemaVersion") != 1:
            errors.append(f"schemaVersion must be 1, got {doc.get('schemaVersion')!r}")
        stacks = doc.get("stacks")
        if not isinstance(stacks, list):
            errors.append("stacks must be an array")
        else:
            for i, entry in enumerate(stacks):
                check_entry(i, entry, errors)

            # The renderer round-trips only the stack ID, so duplicates could
            # display one tuple and install another. The app drops ALL
            # colliding entries; here they fail the file.
            seen = {}
            for i, entry in enumerate(stacks):
                if not isinstance(entry, dict):
                    continue
                pkgs = entry.get("packages")
                torch = pkgs.get("torch") if isinstance(pkgs, dict) else None
                if not isinstance(torch, str) or not isinstance(entry.get("indexTag"), str):
                    continue
                stack_id = f"pytorch-index:{entry['indexTag']}:{public_version(torch)}"
                if stack_id in seen:
                    errors.append(
                        f"stacks[{i}]: duplicate stack ID {stack_id!r} (also stacks[{seen[stack_id]}])"
                    )
                else:
                    seen[stack_id] = i

            if not stacks and not errors:
                print(
                    "WARNING: stacks is empty - publishing this WITHDRAWS every "
                    "index-served stack from all desktop apps.",
                    file=sys.stderr,
                )

    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        print(f"\n{len(errors)} error(s) in {path}", file=sys.stderr)
        return 1

    n = len(doc["stacks"])
    print(f"OK: {path} is valid ({n} stack{'s' if n != 1 else ''})")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: validate_torch_index_stacks.py <torch-index-stacks.json>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
