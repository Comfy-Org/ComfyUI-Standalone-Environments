#!/usr/bin/env python3
"""Refresh the nightly entries in torch-index-stacks.json.

For each allowlisted index tag, scrapes the PEP 503 simple indexes under
https://download.pytorch.org/whl/nightly/<tag>/ and resolves the newest
wheel date where torch, torchvision, and torchaudio all published wheels
for at least one shared platform and Python ABI. The resulting exact dev
tuples replace the manifest's `pytorch-nightly-index` entries; stable
entries are never touched.

Nightly wheels are purged from the index after roughly 60 days, so these
entries are pins that must be re-resolved continuously - this script is
meant to run from the scheduled refresh workflow, which validates and
publishes the result. It exits 0 with no changes when the manifest is
already current, so the workflow can skip the commit.

Run: python3 scripts/refresh_nightly_stacks.py torch-index-stacks.json
"""

import json
import re
import sys
import urllib.request
from datetime import date, datetime, timezone

NIGHTLY_INDEX_BASE = "https://download.pytorch.org/whl/nightly"
PACKAGES = ("torch", "torchvision", "torchaudio")

# Index tags to offer nightlies for. NVIDIA covers the overwhelming
# majority of users; add tags here deliberately (AMD Windows stays out -
# pytorch.org publishes no Windows ROCm wheels, and mps has no dev builds
# on PyPI). Every tag must be one the desktop app's runtime index gate
# accepts (cu*/rocm*/xpu/cpu).
NIGHTLY_TAGS = ("cu132",)

# Refuse to publish a triple older than this: the publish-side freshness
# gate in validate_torch_index_stacks.py enforces the same bound, and a
# nightly index that has not produced a coherent triple for a week is a
# problem a human should look at, not something to silently republish.
MAX_AGE_DAYS = 7

# name-version-pythontag-abitag-platformtag.whl (PEP 427); normalized
# versions contain no dashes, so a plain split is unambiguous.
WHEEL_RE = re.compile(
    r"^(?P<name>[A-Za-z0-9_]+)-(?P<version>[A-Za-z0-9.!+]+)"
    r"-(?P<pytag>[a-z0-9]+)-(?P<abitag>[a-z0-9]+)-(?P<plattag>[a-z0-9_.]+)\.whl$"
)
DEV_DATE_RE = re.compile(r"\.dev(\d{8})\+")
ANCHOR_TEXT_RE = re.compile(r">([^<>]+\.whl)</a>")


def accel_for_tag(tag):
    if tag.startswith("cu"):
        return "nvidia"
    if tag.startswith("rocm"):
        return "amd"
    if tag == "xpu":
        return "intel-xpu"
    if tag == "cpu":
        return "cpu"
    raise ValueError(f"no accel mapping for index tag {tag!r}")


def platform_for_wheel(plattag):
    """Electron-style platform for a wheel platform tag; None = unsupported."""
    if plattag == "win_amd64":
        return "win32"
    if "manylinux" in plattag and plattag.endswith("x86_64"):
        return "linux"
    return None  # aarch64, macos, musllinux, ... - not desktop targets


def abi_for_wheel(pytag, abitag):
    """`major.minor` for a standard CPython wheel; None for anything else
    (freethreaded cp31Xt, abi3, pypy) - the desktop app's venvs are
    standard CPython."""
    m = re.fullmatch(r"cp3(\d+)", pytag)
    if not m or abitag != pytag:
        return None
    return f"3.{m.group(1)}"


def fetch_index(tag, package):
    """(version, platform, abi) triples for one package's dated nightly
    wheels on desktop-relevant platforms."""
    url = f"{NIGHTLY_INDEX_BASE}/{tag}/{package}/"
    with urllib.request.urlopen(url, timeout=60) as resp:
        html = resp.read().decode("utf-8", errors="replace")
    return parse_index(package, tag, html)


def parse_index(package, tag, html):
    """Parse a PEP 503 simple index page (see fetch_index)."""
    wheels = []
    for filename in ANCHOR_TEXT_RE.findall(html):
        m = WHEEL_RE.match(filename)
        if not m:
            continue
        # A malformed index page must not smuggle another distribution's
        # version under this package's name (PEP 503 normalization).
        if m.group("name").lower().replace("_", "-") != package:
            continue
        version = m.group("version")
        if not DEV_DATE_RE.search(version) or not version.endswith(f"+{tag}"):
            continue
        platform = platform_for_wheel(m.group("plattag"))
        abi = abi_for_wheel(m.group("pytag"), m.group("abitag"))
        if platform is None or abi is None:
            continue
        wheels.append((version, platform, abi))
    return wheels


def resolve_tag(tag, today):
    """The newest coherent nightly entry for one tag, or an error string."""
    # per package: date -> {"versions": set, "plat_abis": {platform: set(abi)}}
    by_pkg = {}
    for pkg in PACKAGES:
        dates = {}
        for version, platform, abi in fetch_index(tag, pkg):
            day = DEV_DATE_RE.search(version).group(1)
            slot = dates.setdefault(day, {"versions": set(), "plat_abis": {}})
            slot["versions"].add(version)
            slot["plat_abis"].setdefault(platform, set()).add(abi)
        if not dates:
            return None, f"{tag}: no usable {pkg} nightly wheels"
        by_pkg[pkg] = dates

    shared_dates = set.intersection(*(set(d) for d in by_pkg.values()))
    for day in sorted(shared_dates, reverse=True):
        iso = f"{day[0:4]}-{day[4:6]}-{day[6:8]}"
        if (today - date.fromisoformat(iso)).days > MAX_AGE_DAYS:
            break  # newest shared date is already too old - give up
        slots = {pkg: by_pkg[pkg][day] for pkg in PACKAGES}
        # exactly one version per package per date, or the date is ambiguous
        if any(len(s["versions"]) != 1 for s in slots.values()):
            continue
        # Per platform: ABIs every package publishes there. A platform with
        # no fully-covered ABI is dropped. The manifest's pythonAbis is
        # GLOBAL (not per-platform), so advertise only ABIs complete on
        # every kept platform - a union would promise wheel-less combos
        # (e.g. torchaudio skipping linux cp312 while win32 has it). If the
        # kept platforms share no ABI at all, keep win32 alone (the
        # overwhelming majority) rather than publish nothing or overstate.
        abis_by_platform = {}
        for p in ("win32", "linux"):
            if not all(p in s["plat_abis"] for s in slots.values()):
                continue
            shared = set.intersection(*(s["plat_abis"][p] for s in slots.values()))
            if shared:
                abis_by_platform[p] = shared
        if not abis_by_platform:
            continue
        common = set.intersection(*abis_by_platform.values())
        if not common:
            keep = "win32" if "win32" in abis_by_platform else next(iter(abis_by_platform))
            abis_by_platform = {keep: abis_by_platform[keep]}
            common = abis_by_platform[keep]
        return {
            "kind": "pytorch-nightly-index",
            "indexTag": tag,
            "accel": accel_for_tag(tag),
            "platforms": list(abis_by_platform),
            "packages": {pkg: next(iter(slots[pkg]["versions"])) for pkg in PACKAGES},
            "date": iso,
            "pythonAbis": sorted(common, key=lambda a: int(a.split(".")[1])),
        }, None
    return None, f"{tag}: no coherent torch/torchvision/torchaudio triple within {MAX_AGE_DAYS} days"


def main(path):
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    stacks = doc.get("stacks")
    if not isinstance(stacks, list):
        print(f"ERROR: {path} has no stacks array", file=sys.stderr)
        return 1

    today = datetime.now(timezone.utc).date()
    fresh, errors = [], []
    for tag in NIGHTLY_TAGS:
        entry, err = resolve_tag(tag, today)
        if err:
            errors.append(err)
        else:
            fresh.append(entry)
    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1

    # Stable entries stay untouched and keep their order; ALL nightly
    # entries are replaced (a tag dropped from the allowlist is withdrawn).
    stable = [s for s in stacks if not (isinstance(s, dict) and s.get("kind") == "pytorch-nightly-index")]
    updated = stable + fresh
    if updated == stacks:
        print("No changes - manifest is current.")
        return 0
    doc["stacks"] = updated
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, indent=2)
        fh.write("\n")
    for entry in fresh:
        print(f"Updated {entry['indexTag']}: torch {entry['packages']['torch']} ({entry['date']})")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: refresh_nightly_stacks.py <torch-index-stacks.json>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
