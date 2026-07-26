#!/usr/bin/env python3
"""Deterministic tests for validate_torch_index_stacks.py and
refresh_nightly_stacks.py - no network, no live-index dependence.

Run: python3 -m unittest scripts.test_torch_index_scripts -v
 (or from scripts/: python3 -m unittest test_torch_index_scripts)
"""

import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import refresh_nightly_stacks as refresher  # noqa: E402
import validate_torch_index_stacks as validator  # noqa: E402


def utc_today():
    return datetime.now(timezone.utc).date()


def nightly_entry(days_old=1, **overrides):
    """A valid pytorch-nightly-index entry, dated relative to today."""
    day = utc_today() - timedelta(days=days_old)
    d = day.strftime("%Y%m%d")
    entry = {
        "kind": "pytorch-nightly-index",
        "indexTag": "cu132",
        "accel": "nvidia",
        "platforms": ["win32", "linux"],
        "packages": {
            "torch": f"2.14.0.dev{d}+cu132",
            "torchvision": f"0.29.0.dev{d}+cu132",
            "torchaudio": f"2.11.0.dev{d}+cu132",
        },
        "date": day.isoformat(),
        "pythonAbis": ["3.10", "3.11", "3.13"],
    }
    entry.update(overrides)
    return entry


def stable_entry(**overrides):
    entry = {
        "indexTag": "cu128",
        "accel": "nvidia",
        "platforms": ["win32", "linux"],
        "packages": {
            "torch": "2.11.0+cu128",
            "torchvision": "0.26.0+cu128",
            "torchaudio": "2.11.0+cu128",
        },
        "date": "2026-03-25",
    }
    entry.update(overrides)
    return entry


def entry_errors(entry):
    errors = []
    validator.check_entry(0, entry, errors)
    return errors


class ValidatorNightlyRules(unittest.TestCase):
    def test_valid_fresh_nightly_passes(self):
        self.assertEqual(entry_errors(nightly_entry()), [])

    def test_today_and_one_day_future_pass(self):
        # UTC "tomorrow" can be legitimate for a machine west of UTC.
        self.assertEqual(entry_errors(nightly_entry(days_old=0)), [])
        self.assertEqual(entry_errors(nightly_entry(days_old=-1)), [])

    def test_stale_nightly_fails(self):
        errs = entry_errors(nightly_entry(days_old=validator.NIGHTLY_MAX_AGE_DAYS + 1))
        self.assertTrue(any("publish window" in e for e in errs), errs)

    def test_freshness_boundary_passes(self):
        self.assertEqual(entry_errors(nightly_entry(days_old=validator.NIGHTLY_MAX_AGE_DAYS)), [])

    def test_far_future_nightly_fails(self):
        errs = entry_errors(nightly_entry(days_old=-2))
        self.assertTrue(any("publish window" in e for e in errs), errs)

    def test_mixed_wheel_dates_fail(self):
        e = nightly_entry()
        other = (utc_today() - timedelta(days=2)).strftime("%Y%m%d")
        e["packages"]["torchaudio"] = f"2.11.0.dev{other}+cu132"
        errs = entry_errors(e)
        self.assertTrue(any("share one wheel date" in x for x in errs), errs)

    def test_date_field_must_match_wheel_date(self):
        e = nightly_entry(date=(utc_today() - timedelta(days=2)).isoformat())
        errs = entry_errors(e)
        self.assertTrue(any("must equal the wheel date" in x for x in errs), errs)

    def test_stable_version_under_nightly_kind_fails(self):
        e = nightly_entry()
        e["packages"]["torchvision"] = "0.29.0+cu132"
        errs = entry_errors(e)
        self.assertTrue(any(".devYYYYMMDD" in x for x in errs), errs)

    def test_undated_dev_under_nightly_kind_fails(self):
        e = nightly_entry()
        e["packages"]["torch"] = "2.14.0.dev1+cu132"
        errs = entry_errors(e)
        self.assertTrue(any(".devYYYYMMDD" in x for x in errs), errs)

    def test_dev_version_under_stable_kind_fails(self):
        day = (utc_today() - timedelta(days=1)).strftime("%Y%m%d")
        e = stable_entry()
        e["packages"]["torch"] = f"2.14.0.dev{day}+cu128"
        errs = entry_errors(e)
        self.assertTrue(any("dev (nightly) release" in x for x in errs), errs)

    def test_mps_nightly_fails(self):
        e = nightly_entry(
            indexTag="pypi",
            accel="mps",
            platforms=["darwin"],
        )
        for pkg in e["packages"]:
            e["packages"][pkg] = e["packages"][pkg].split("+", 1)[0]
        errs = entry_errors(e)
        self.assertTrue(any("mps entries cannot be nightly" in x for x in errs), errs)

    def test_nightly_tag_mismatch_fails(self):
        e = nightly_entry()
        e["packages"]["torch"] = e["packages"]["torch"].replace("+cu132", "+cu130")
        errs = entry_errors(e)
        self.assertTrue(any("does not match indexTag" in x for x in errs), errs)

    def test_nightly_tag_outside_allowlist_fails(self):
        e = nightly_entry(indexTag="cu130")
        for pkg in e["packages"]:
            e["packages"][pkg] = e["packages"][pkg].replace("+cu132", "+cu130")
        errs = entry_errors(e)
        self.assertTrue(any("is not offered" in x for x in errs), errs)


class ValidatorExistingRules(unittest.TestCase):
    def test_stable_entry_passes(self):
        self.assertEqual(entry_errors(stable_entry()), [])

    def test_amd_win32_fails(self):
        e = stable_entry(
            indexTag="rocm7.2.1",
            accel="amd",
            packages={"torch": "2.11.0+rocm7.2.1"},
        )
        errs = entry_errors(e)
        self.assertTrue(any("cannot target win32" in x for x in errs), errs)

    def test_empty_python_abis_fails(self):
        errs = entry_errors(stable_entry(pythonAbis=[]))
        self.assertTrue(any("pythonAbis" in x for x in errs), errs)

    def test_repo_manifest_is_valid(self):
        manifest = Path(__file__).resolve().parent.parent / "torch-index-stacks.json"
        self.assertEqual(validator.main(str(manifest)), 0)


def wheel_anchor(name):
    return f'<a href="/whl/nightly/x/{name}">{name}</a><br/>'


def index_html(names):
    return "<html><body>" + "".join(wheel_anchor(n) for n in names) + "</body></html>"


class RefresherParsing(unittest.TestCase):
    def test_parse_index_filters_wheels(self):
        html = index_html(
            [
                # kept: dated dev, right tag, desktop platform, standard
                # CPython; percent-encoded anchor text must decode
                "torch-2.14.0.dev20260724%2Bcu132-cp313-cp313-win_amd64.whl",
                "torch-2.14.0.dev20260724+cu132-cp312-cp312-manylinux_2_28_x86_64.whl",
                # dropped: wrong index tag
                "torch-2.14.0.dev20260724%2Bcu130-cp313-cp313-win_amd64.whl",
                # dropped: not a dated dev build
                "torch-2.13.0%2Bcu132-cp313-cp313-win_amd64.whl",
                # dropped: non-desktop platforms
                "torch-2.14.0.dev20260724%2Bcu132-cp313-cp313-manylinux_2_28_aarch64.whl",
                "torch-2.14.0.dev20260724%2Bcu132-cp313-cp313-macosx_11_0_arm64.whl",
                # dropped: freethreaded ABI
                "torch-2.14.0.dev20260724%2Bcu132-cp313-cp313t-win_amd64.whl",
                # dropped: another distribution smuggled under torch's index
                "notorch-2.99.0.dev20260724%2Bcu132-cp313-cp313-win_amd64.whl",
                # dropped: not a wheel filename
                "torch-2.14.0.dev20260724.tar.gz",
            ]
        )
        wheels = refresher.parse_index("torch", "cu132", html)
        self.assertEqual(
            sorted(wheels),
            [
                ("2.14.0.dev20260724+cu132", "linux", "3.12"),
                ("2.14.0.dev20260724+cu132", "win32", "3.13"),
            ],
        )


def fake_indexes(data):
    """fetch_index replacement: data[package] = list of wheel triples."""

    def fetch(tag, package):
        return list(data.get(package, []))

    return fetch


def triples(pkg_version, day, plat_abis):
    """[(version, platform, abi)] for one package/date across platforms."""
    out = []
    for platform, abis in plat_abis.items():
        for abi in abis:
            out.append((f"{pkg_version}.dev{day}+cu132", platform, abi))
    return out


class RefresherResolution(unittest.TestCase):
    def resolve(self, data, days_old=1):
        today = utc_today()
        day = (today - timedelta(days=days_old)).strftime("%Y%m%d")
        with mock.patch.object(refresher, "fetch_index", fake_indexes(data)):
            entry, err = refresher.resolve_tag("cu132", today)
        return entry, err, day

    def test_full_coverage_keeps_both_platforms(self):
        both = {"win32": ["3.12", "3.13"], "linux": ["3.12", "3.13"]}
        today = utc_today()
        day = (today - timedelta(days=1)).strftime("%Y%m%d")
        data = {
            "torch": triples("2.14.0", day, both),
            "torchvision": triples("0.29.0", day, both),
            "torchaudio": triples("2.11.0", day, both),
        }
        entry, err, _ = self.resolve(data)
        self.assertIsNone(err)
        self.assertEqual(sorted(entry["platforms"]), ["linux", "win32"])
        self.assertEqual(entry["pythonAbis"], ["3.12", "3.13"])
        self.assertEqual(entry["packages"]["torch"], f"2.14.0.dev{day}+cu132")

    def test_one_platform_abi_gap_shrinks_global_abis(self):
        # torchaudio missing linux 3.12: 3.12 must NOT be advertised
        # globally, but both platforms stay listed via the shared 3.13.
        today = utc_today()
        day = (today - timedelta(days=1)).strftime("%Y%m%d")
        both = {"win32": ["3.12", "3.13"], "linux": ["3.12", "3.13"]}
        data = {
            "torch": triples("2.14.0", day, both),
            "torchvision": triples("0.29.0", day, both),
            "torchaudio": triples("2.11.0", day, {"win32": ["3.12", "3.13"], "linux": ["3.13"]}),
        }
        entry, err, _ = self.resolve(data)
        self.assertIsNone(err)
        self.assertEqual(sorted(entry["platforms"]), ["linux", "win32"])
        self.assertEqual(entry["pythonAbis"], ["3.13"])

    def test_no_shared_abi_keeps_win32_only(self):
        today = utc_today()
        day = (today - timedelta(days=1)).strftime("%Y%m%d")
        data = {
            "torch": triples("2.14.0", day, {"win32": ["3.13"], "linux": ["3.12"]}),
            "torchvision": triples("0.29.0", day, {"win32": ["3.13"], "linux": ["3.12"]}),
            "torchaudio": triples("2.11.0", day, {"win32": ["3.13"], "linux": ["3.12"]}),
        }
        entry, err, _ = self.resolve(data)
        self.assertIsNone(err)
        self.assertEqual(entry["platforms"], ["win32"])
        self.assertEqual(entry["pythonAbis"], ["3.13"])

    def test_platform_missing_a_package_is_dropped(self):
        today = utc_today()
        day = (today - timedelta(days=1)).strftime("%Y%m%d")
        both = {"win32": ["3.13"], "linux": ["3.13"]}
        data = {
            "torch": triples("2.14.0", day, both),
            "torchvision": triples("0.29.0", day, both),
            "torchaudio": triples("2.11.0", day, {"win32": ["3.13"]}),
        }
        entry, err, _ = self.resolve(data)
        self.assertIsNone(err)
        self.assertEqual(entry["platforms"], ["win32"])

    def test_ambiguous_version_falls_back_to_older_date(self):
        today = utc_today()
        newest = (today - timedelta(days=1)).strftime("%Y%m%d")
        older = (today - timedelta(days=2)).strftime("%Y%m%d")
        both = {"win32": ["3.13"], "linux": ["3.13"]}
        data = {
            # two torch versions on the newest date -> ambiguous, skipped
            "torch": triples("2.14.0", newest, both)
            + triples("2.15.0", newest, both)
            + triples("2.14.0", older, both),
            "torchvision": triples("0.29.0", newest, both) + triples("0.29.0", older, both),
            "torchaudio": triples("2.11.0", newest, both) + triples("2.11.0", older, both),
        }
        entry, err, _ = self.resolve(data)
        self.assertIsNone(err)
        self.assertEqual(entry["packages"]["torch"], f"2.14.0.dev{older}+cu132")

    def test_too_old_shared_date_is_rejected(self):
        entry, err, _ = self.resolve(
            {
                "torch": triples(
                    "2.14.0",
                    (utc_today() - timedelta(days=refresher.MAX_AGE_DAYS + 1)).strftime("%Y%m%d"),
                    {"win32": ["3.13"]},
                ),
                "torchvision": triples(
                    "0.29.0",
                    (utc_today() - timedelta(days=refresher.MAX_AGE_DAYS + 1)).strftime("%Y%m%d"),
                    {"win32": ["3.13"]},
                ),
                "torchaudio": triples(
                    "2.11.0",
                    (utc_today() - timedelta(days=refresher.MAX_AGE_DAYS + 1)).strftime("%Y%m%d"),
                    {"win32": ["3.13"]},
                ),
            }
        )
        self.assertIsNone(entry)
        self.assertIn("no coherent", err)

    def test_missing_package_is_an_error(self):
        entry, err, _ = self.resolve(
            {
                "torch": triples(
                    "2.14.0", (utc_today() - timedelta(days=1)).strftime("%Y%m%d"), {"win32": ["3.13"]}
                ),
                "torchvision": [],
                "torchaudio": [],
            }
        )
        self.assertIsNone(entry)
        self.assertIn("no usable", err)

    def test_resolved_entry_passes_validator(self):
        today = utc_today()
        day = (today - timedelta(days=1)).strftime("%Y%m%d")
        both = {"win32": ["3.13"], "linux": ["3.13"]}
        data = {
            "torch": triples("2.14.0", day, both),
            "torchvision": triples("0.29.0", day, both),
            "torchaudio": triples("2.11.0", day, both),
        }
        entry, err, _ = self.resolve(data)
        self.assertIsNone(err)
        self.assertEqual(entry_errors(entry), [])


class RefresherMain(unittest.TestCase):
    def run_main(self, doc, data):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text(json.dumps(doc), encoding="utf-8")
            with mock.patch.object(refresher, "fetch_index", fake_indexes(data)):
                code = refresher.main(str(path))
            return code, json.loads(path.read_text(encoding="utf-8"))

    def full_data(self, day):
        both = {"win32": ["3.13"], "linux": ["3.13"]}
        return {
            "torch": triples("2.14.0", day, both),
            "torchvision": triples("0.29.0", day, both),
            "torchaudio": triples("2.11.0", day, both),
        }

    def test_replaces_nightlies_and_keeps_stable_untouched(self):
        day = (utc_today() - timedelta(days=1)).strftime("%Y%m%d")
        doc = {
            "schemaVersion": 1,
            "stacks": [stable_entry(), nightly_entry(days_old=5)],
        }
        code, updated = self.run_main(doc, self.full_data(day))
        self.assertEqual(code, 0)
        self.assertEqual(updated["stacks"][0], stable_entry())
        nightlies = [s for s in updated["stacks"] if s.get("kind") == "pytorch-nightly-index"]
        self.assertEqual(len(nightlies), 1)
        self.assertEqual(nightlies[0]["packages"]["torch"], f"2.14.0.dev{day}+cu132")

    def test_no_rewrite_when_current(self):
        day = (utc_today() - timedelta(days=1)).strftime("%Y%m%d")
        data = self.full_data(day)
        doc = {"schemaVersion": 1, "stacks": [stable_entry()]}
        code, updated = self.run_main(doc, data)
        self.assertEqual(code, 0)
        code2, updated2 = self.run_main(updated, data)
        self.assertEqual(code2, 0)
        self.assertEqual(updated, updated2)

    def test_refresh_failure_leaves_manifest_untouched(self):
        doc = {"schemaVersion": 1, "stacks": [stable_entry(), nightly_entry()]}
        code, updated = self.run_main(doc, {"torch": [], "torchvision": [], "torchaudio": []})
        self.assertEqual(code, 1)
        self.assertEqual(updated, doc)


if __name__ == "__main__":
    unittest.main()
