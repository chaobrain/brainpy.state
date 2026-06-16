# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# -*- coding: utf-8 -*-

"""Tests for the brainpy compatibility guard in :mod:`brainpy_state._compat`.

The guard's whole job is to fail correctly, so these tests pin every branch:
the release-segment parser, the absent/ok/stale/unparseable paths of the check,
the boundary versions, the user-facing message, and a drift guard that keeps
``MIN_BRAINPY`` in lock-step with the ``brainpy>=...`` pin in ``pyproject.toml``.
"""

import re
import tomllib
import unittest
import warnings
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from unittest import mock

from brainpy_state import _compat
from brainpy_state._compat import (
    MIN_BRAINPY,
    _parse_release,
    check_brainpy_compatibility,
)


class TestParseRelease(unittest.TestCase):
    """The pure release-segment parser -- no mocking required."""

    def test_full_triplet(self):
        self.assertEqual(_parse_release("2.7.6"), (2, 7, 6))

    def test_two_part_defaults_patch_to_zero(self):
        self.assertEqual(_parse_release("2.7"), (2, 7, 0))

    def test_post_suffix_strips_to_release(self):
        self.assertEqual(_parse_release("2.7.6.post1"), (2, 7, 6))

    def test_rc_suffix_strips_to_release(self):
        self.assertEqual(_parse_release("2.7.6rc1"), (2, 7, 6))

    def test_dev_suffix_strips_to_release(self):
        self.assertEqual(_parse_release("2.7.6.dev0"), (2, 7, 6))

    def test_local_version_strips_to_release(self):
        self.assertEqual(_parse_release("2.7.6+cuda12"), (2, 7, 6))

    def test_unparseable_returns_none(self):
        self.assertIsNone(_parse_release("garbage"))

    def test_empty_returns_none(self):
        self.assertIsNone(_parse_release(""))


class TestCheckBrainpyCompatibility(unittest.TestCase):
    """The guard, with ``importlib.metadata.version`` mocked for determinism."""

    @mock.patch("importlib.metadata.version")
    def test_brainpy_absent_is_tolerated_silently(self, mock_version):
        # brainpy is a declared hard dependency; its absence implies a deliberate
        # --no-deps environment, tolerated with neither raise nor warning.
        mock_version.side_effect = PackageNotFoundError
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.assertIsNone(check_brainpy_compatibility())

    @mock.patch("importlib.metadata.version")
    def test_exact_threshold_is_ok(self, mock_version):
        mock_version.return_value = "2.7.6"
        self.assertIsNone(check_brainpy_compatibility())

    @mock.patch("importlib.metadata.version")
    def test_above_threshold_is_ok(self, mock_version):
        mock_version.return_value = "2.7.8"
        self.assertIsNone(check_brainpy_compatibility())

    @mock.patch("importlib.metadata.version")
    def test_dev_build_of_target_is_allowed(self, mock_version):
        # release-segment compare: a .dev/rc of the target is not blocked.
        mock_version.return_value = "2.7.6.dev0"
        self.assertIsNone(check_brainpy_compatibility())

    @mock.patch("importlib.metadata.version")
    def test_one_patch_below_is_blocked(self, mock_version):
        mock_version.return_value = "2.7.5"
        with self.assertRaises(ImportError):
            check_brainpy_compatibility()

    @mock.patch("importlib.metadata.version")
    def test_minor_below_is_blocked(self, mock_version):
        mock_version.return_value = "2.6.9"
        with self.assertRaises(ImportError):
            check_brainpy_compatibility()

    @mock.patch("importlib.metadata.version")
    def test_two_part_below_is_blocked(self, mock_version):
        # "2.7" parses to (2, 7, 0) which is below (2, 7, 6).
        mock_version.return_value = "2.7"
        with self.assertRaises(ImportError):
            check_brainpy_compatibility()

    @mock.patch("importlib.metadata.version")
    def test_unparseable_version_warns_and_allows(self, mock_version):
        mock_version.return_value = "garbage"
        with self.assertWarns(UserWarning):
            self.assertIsNone(check_brainpy_compatibility())

    @mock.patch("importlib.metadata.version")
    def test_error_message_names_both_versions(self, mock_version):
        mock_version.return_value = "2.7.5"
        with self.assertRaises(ImportError) as ctx:
            check_brainpy_compatibility()
        msg = str(ctx.exception)
        self.assertIn("2.7.5", msg)  # the offending installed version
        self.assertIn("2.7.6", msg)  # the required threshold

    @mock.patch("importlib.metadata.version")
    def test_min_version_is_injectable(self, mock_version):
        # Drives the stale branch deterministically without string-mocking down.
        mock_version.return_value = "2.7.8"
        with self.assertRaises(ImportError):
            check_brainpy_compatibility(min_version=(99, 0, 0))


class TestPyprojectConsistency(unittest.TestCase):
    """Drift guard: the runtime constant must match the packaging pin."""

    def test_min_brainpy_matches_pyproject_pin(self):
        pyproject = Path(_compat.__file__).resolve().parent.parent / "pyproject.toml"
        data = tomllib.loads(pyproject.read_text())
        deps = data["project"]["dependencies"]

        brainpy_spec = next(
            (d for d in deps if re.match(r"brainpy\s*>=", d)), None
        )
        self.assertIsNotNone(
            brainpy_spec, "no 'brainpy>=...' entry found in pyproject dependencies"
        )

        m = re.search(r">=\s*(\d+)\.(\d+)(?:\.(\d+))?", brainpy_spec)
        self.assertIsNotNone(m, f"could not parse version from {brainpy_spec!r}")
        pinned = (int(m.group(1)), int(m.group(2)), int(m.group(3) or 0))

        self.assertEqual(
            pinned,
            MIN_BRAINPY,
            f"pyproject pins brainpy>={pinned} but MIN_BRAINPY={MIN_BRAINPY}",
        )


if __name__ == "__main__":
    unittest.main()
