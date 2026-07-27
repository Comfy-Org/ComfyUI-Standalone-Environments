#!/usr/bin/env python3
"""Deterministic tests for build_local.py and serve_local.py metadata
generation - no network, no actual environment builds.

Run: python3 -m unittest scripts.test_local_tooling -v
 (or from scripts/: python3 -m unittest test_local_tooling)
"""

import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_local  # noqa: E402
import serve_local  # noqa: E402


def manifest(**overrides):
    """A build_local-shaped manifest.json for the win-amd variant."""
    m = {
        "id": "win-amd",
        "version": "v0.4.61-local1",
        "build": 0,
        "archive_base": "comfyui-standalone-win-amd-v0.4.61-local1",
        "files": [],
        "comfyui_ref": "v0.4.61",
        "comfyui_commit": "a" * 40,
        "python_version": "3.13.12",
        "torch_version": "2.10.0+rocm7.14.0",
        "torchvision_version": "0.25.0+rocm7.14.0",
        "torchaudio_version": "2.10.0+rocm7.14.0",
        "pbs_release": "20260211",
        "uv_version": "0.9.5",
        "vendor_requirements": "requirements-amd-win.txt",
        "vendor_requirements_content": "torch",
        "comfyui_requirements_content": "x",
        "manager_requirements_content": "y",
        "build_date": "2026-07-26T12:00:00Z",
    }
    m.update(overrides)
    return m


class BuildNumberFromTag(unittest.TestCase):
    def test_env_suffix_extracted(self):
        self.assertEqual(build_local.build_number_from_tag("v0.4.61-env3"), 3)

    def test_non_env_tags_are_build_zero(self):
        for tag in ["v0.4.61-local1", "v0.4.61", "test"]:
            self.assertEqual(build_local.build_number_from_tag(tag), 0)


class SafeTag(unittest.TestCase):
    def test_accepts_release_and_local_tags(self):
        for tag in ["v0.4.61", "v0.4.61-env3", "v0.4.61-local1", "mybranch-local1"]:
            self.assertTrue(build_local.is_safe_tag(tag), tag)

    def test_rejects_tags_that_break_paths_or_urls(self):
        # Branch refs with '/' are the common footgun (default tag is
        # derived from --comfyui-ref); traversal must never reach disk.
        for tag in ["feature/foo-local1", "../evil", "a..b", "", ".hidden", "a b"]:
            self.assertFalse(build_local.is_safe_tag(tag), tag)


class VariantPythonVersion(unittest.TestCase):
    def test_overrides_win_over_default(self):
        try:
            build_local.VARIANT_PYTHON_VERSIONS["win-amd"] = "3.12.12"
            self.assertEqual(build_local.variant_python_version("win-amd"), "3.12.12")
        finally:
            del build_local.VARIANT_PYTHON_VERSIONS["win-amd"]

    def test_variants_without_override_use_workflow_default(self):
        for v in ["win-nvidia", "win-amd", "linux-amd", "mac-mps"]:
            self.assertEqual(build_local.variant_python_version(v),
                             build_local.PYTHON_VERSION)


class WorkflowParity(unittest.TestCase):
    """build_local.py mirrors the CI workflow's env + matrix; these fail
    when build-standalone-env.yml changes without updating the tooling."""

    @classmethod
    def setUpClass(cls):
        workflow = (Path(__file__).resolve().parent.parent
                    / ".github" / "workflows" / "build-standalone-env.yml")
        cls.text = workflow.read_text(encoding="utf-8")
        # Matrix entries end where the job body begins. Commented-out
        # entries don't match the anchored regex.
        matrix = cls.text[:cls.text.index("runs-on: ${{ matrix.os }}")]
        starts = [(m.start(), m.group(1))
                  for m in re.finditer(r"^\s+- id: ([\w-]+)$", matrix, re.M)]
        ends = [s for s, _ in starts[1:]] + [len(matrix)]
        cls.entries = {vid: matrix[start:end]
                       for (start, vid), end in zip(starts, ends)}

    def test_python_and_pbs_defaults_match(self):
        self.assertIn(f'PYTHON_VERSION: "{build_local.PYTHON_VERSION}"', self.text)
        self.assertIn(f'PBS_RELEASE: "{build_local.PBS_RELEASE}"', self.text)

    def test_variant_ids_match_matrix(self):
        self.assertEqual(sorted(self.entries), sorted(build_local.VARIANTS))

    def test_vendor_requirements_match_matrix(self):
        for vid, block in self.entries.items():
            m = re.search(r"^\s+vendor_requirements: (\S+)$", block, re.M)
            self.assertIsNotNone(m, vid)
            self.assertEqual(m.group(1), build_local.VARIANTS[vid], vid)

    def test_python_version_overrides_match_matrix(self):
        overrides = {}
        for vid, block in self.entries.items():
            m = re.search(r'^\s+python_version: "([\d.]+)"$', block, re.M)
            if m:
                overrides[vid] = m.group(1)
        self.assertEqual(overrides, build_local.VARIANT_PYTHON_VERSIONS)

    def test_7z_parameters_match(self):
        self.assertIn(" ".join(build_local.SEVENZ_ARGS), self.text)


class ArchiveFilename(unittest.TestCase):
    def test_mac_uses_tar_gz_others_7z(self):
        self.assertEqual(build_local.archive_filename("mac-mps", "v1-local1"),
                         "comfyui-standalone-mac-mps-v1-local1.tar.gz")
        self.assertEqual(build_local.archive_filename("win-amd", "v1-local1"),
                         "comfyui-standalone-win-amd-v1-local1.7z")
        self.assertEqual(build_local.archive_filename("linux-nvidia", "v1-local1"),
                         "comfyui-standalone-linux-nvidia-v1-local1.7z")


class BuildManifest(unittest.TestCase):
    def test_shape_matches_ci_manifest(self):
        m = build_local.build_manifest(
            "win-amd", "v0.4.61-env2", "v0.4.61", "b" * 40,
            {"torch": "2.10.0+rocm7.14.0", "torchvision": "0.25.0+rocm7.14.0",
             "torchaudio": "2.10.0+rocm7.14.0", "uv": "0.9.5"},
            "requirements-amd-win.txt",
            {"vendor": "torch", "comfyui": "x", "manager": "y"},
        )
        # Field-for-field the CI 'Generate build manifest' step's output.
        self.assertEqual(m["id"], "win-amd")
        self.assertEqual(m["build"], 2)
        self.assertEqual(m["archive_base"], "comfyui-standalone-win-amd-v0.4.61-env2")
        # CI archives the manifest before populating the files list.
        self.assertEqual(m["files"], [])
        self.assertEqual(m["python_version"], build_local.PYTHON_VERSION)
        self.assertEqual(m["vendor_requirements_content"], "torch")
        # ISO-8601 Zulu, same format string as the workflow's `date -u`.
        self.assertRegex(m["build_date"], r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


class ReleaseEntry(unittest.TestCase):
    def test_maps_manifest_to_r2_variant_shape(self):
        entry = serve_local.release_entry(manifest(), "archive.7z", 123)
        self.assertEqual(entry, {
            "tag": "v0.4.61-local1",
            "comfyui_version": "v0.4.61",
            "comfyui_commit": "a" * 40,
            "build": 0,
            "date": "2026-07-26T12:00:00Z",
            "file": "archive.7z",
            "size": 123,
            "python_version": "3.13.12",
            "torch_version": "2.10.0+rocm7.14.0",
            "torchvision_version": "0.25.0+rocm7.14.0",
            "torchaudio_version": "2.10.0+rocm7.14.0",
        })


class CheckEntryServable(unittest.TestCase):
    def entry(self, **overrides):
        e = serve_local.release_entry(manifest(), "ok-file.7z", 1)
        e.update(overrides)
        return e

    def test_valid_entry_has_no_problems(self):
        self.assertEqual(serve_local.check_entry_servable("win-amd", self.entry()), [])

    def test_rejects_what_the_app_validator_drops(self):
        # Mirrors SAFE_SEGMENT / isSafeFilename / size checks in r2Catalog.ts.
        self.assertTrue(serve_local.check_entry_servable("win-amd", self.entry(tag="../evil")))
        self.assertTrue(serve_local.check_entry_servable("win-amd", self.entry(file="a/b.7z")))
        self.assertTrue(serve_local.check_entry_servable("win-amd", self.entry(size=0)))
        self.assertTrue(serve_local.check_entry_servable("bad/variant", self.entry()))


class BuildMetadata(unittest.TestCase):
    def test_newest_per_vendor_wins_latest(self):
        older = serve_local.release_entry(
            manifest(version="v0.4.60-local1", build_date="2026-07-01T00:00:00Z"), "old.7z", 1)
        newer = serve_local.release_entry(manifest(), "new.7z", 2)
        latest, releases = serve_local.build_metadata(
            [("win-amd", older), ("win-amd", newer)])
        self.assertEqual(latest["win-amd"]["file"], "new.7z")
        self.assertEqual([e["file"] for e in releases["win-amd"]["releases"]],
                         ["new.7z", "old.7z"])


class GenerateEndToEnd(unittest.TestCase):
    def make_tree(self, root, m, archive_name=None):
        d = root / m["id"] / m["version"]
        d.mkdir(parents=True)
        (d / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
        (d / (archive_name or m["archive_base"] + ".7z")).write_bytes(b"payload")

    def test_generates_r2_layout_from_built_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dist"
            self.make_tree(root, manifest())
            stacks = Path(tmp) / "torch-index-stacks.json"
            stacks.write_text("{}", encoding="utf-8")

            serve_local.generate(root, stacks)

            latest = json.loads((root / "latest.json").read_text(encoding="utf-8"))
            self.assertEqual(latest["win-amd"]["tag"], "v0.4.61-local1")
            self.assertEqual(latest["win-amd"]["size"], len(b"payload"))
            releases = json.loads(
                (root / "win-amd" / "releases.json").read_text(encoding="utf-8"))
            self.assertEqual(len(releases["releases"]), 1)
            self.assertTrue((root / "torch-index-stacks.json").is_file())

    def test_rejects_layout_that_would_404(self):
        # Metadata points at <root>/<id>/<tag>/<file>; a tree whose dirs
        # disagree with the manifest would generate URLs that 404.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dist"
            m = manifest()
            d = root / "wrong-dir" / m["version"]
            d.mkdir(parents=True)
            (d / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
            (d / (m["archive_base"] + ".7z")).write_bytes(b"x")
            with self.assertRaises(SystemExit):
                serve_local.generate(root, Path(tmp) / "missing.json")

    def test_empty_tree_fails_instead_of_serving_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dist"
            root.mkdir()
            with self.assertRaises(SystemExit):
                serve_local.generate(root, Path(tmp) / "missing.json")


if __name__ == "__main__":
    unittest.main()
