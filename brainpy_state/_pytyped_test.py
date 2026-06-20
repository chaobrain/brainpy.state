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

"""Tests that ``brainpy_state`` ships its PEP 561 ``py.typed`` marker.

The marker is what tells downstream type checkers (mypy, pyright) to read this
package's inline annotations instead of treating it as untyped. A single marker
at the top-level package root makes the whole subpackage tree typed. These tests
guard against the marker being deleted or dropped from the build (``package-data``
in ``pyproject.toml``), which would silently strip type information from users.
"""

import importlib.resources
import unittest
from pathlib import Path

import brainpy_state


class TestPyTypedMarker(unittest.TestCase):
    def test_marker_exists_next_to_package(self):
        # The marker sits at the importable package root, alongside __init__.py.
        pkg_dir = Path(brainpy_state.__file__).parent
        marker = pkg_dir / 'py.typed'
        self.assertTrue(
            marker.is_file(),
            f'PEP 561 marker missing: expected {marker} to exist.',
        )

    def test_marker_discoverable_as_package_data(self):
        # importlib.resources resolves the marker the way an installed wheel
        # exposes it, so this fails if py.typed is not packaged as data.
        marker = importlib.resources.files('brainpy_state') / 'py.typed'
        self.assertTrue(
            marker.is_file(),
            'py.typed is not discoverable via importlib.resources; check '
            '[tool.setuptools.package-data] in pyproject.toml.',
        )

    def test_marker_is_complete_not_partial(self):
        # A non-empty marker containing "partial" signals PEP 561 partial typing.
        # brainpy_state ships full inline types, so the marker must be empty.
        marker = importlib.resources.files('brainpy_state') / 'py.typed'
        self.assertEqual(
            marker.read_text().strip(),
            '',
            'py.typed should be empty (full typing), not a "partial" marker.',
        )


if __name__ == '__main__':
    unittest.main()
