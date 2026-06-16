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

"""Tests that ``__version_info__`` stays well-formed and in sync with ``__version__``.

Robustness of the parse against PEP 440 suffixes is covered by the
``_parse_release`` tests in :mod:`brainpy_state._compat_test`; here we only pin
that the package's own version metadata is consistent and never crashes import.
"""

import unittest

from brainpy_state._compat import _parse_release
from brainpy_state._version import __version__, __version_info__


class TestVersionInfo(unittest.TestCase):
    def test_is_three_int_tuple(self):
        self.assertIsInstance(__version_info__, tuple)
        self.assertEqual(len(__version_info__), 3)
        self.assertTrue(all(isinstance(p, int) for p in __version_info__))

    def test_matches_parsed_version_string(self):
        self.assertEqual(__version_info__, _parse_release(__version__))

    def test_reflects_release_prefix_of_version(self):
        # The first two components must mirror the declared version string.
        major_minor = tuple(int(p) for p in __version__.split(".")[:2])
        self.assertEqual(__version_info__[:2], major_minor)


if __name__ == "__main__":
    unittest.main()
