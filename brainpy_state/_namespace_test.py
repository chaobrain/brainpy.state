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

"""Tests for the namespace guard in :mod:`brainpy_state._namespace`.

The guard's whole job is to reject a direct ``import brainpy_state`` while letting
the blessed ``brainpy.state`` path, the test suite, doc builds, and an explicit
override through. These tests pin the pure decision matrix, the enforcer's raise
path and message, and -- in real subprocesses, the only place the live import
machinery actually runs -- the external-blocked / override / blessed end-to-end
behaviour.
"""

import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

from brainpy_state import _namespace
from brainpy_state._namespace import (
    ALLOW_DIRECT_IMPORT_ENV,
    _is_internal_access,
    enforce_namespace_access,
)

_REPO_ROOT = Path(_namespace.__file__).resolve().parent.parent


def _run_import(code, *, allow_override=False):
    """Import ``brainpy_state`` in a clean child interpreter and report the result.

    Runs from the repo root (so the local package is importable via ``sys.path[0]``)
    with ``pytest``/``sphinx`` absent and -- unless ``allow_override`` -- the override
    env var stripped, isolating the genuine external-import code path.
    """
    env = {k: v for k, v in os.environ.items() if k != ALLOW_DIRECT_IMPORT_ENV}
    if allow_override:
        env[ALLOW_DIRECT_IMPORT_ENV] = "1"
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )


class TestIsInternalAccess(unittest.TestCase):
    """The pure allow/deny decision -- no live interpreter state."""

    def test_blessed_shim_frame_allows(self):
        self.assertTrue(_is_internal_access(["brainpy.state", "brainpy"], {}, {}))

    def test_shim_frame_anywhere_on_stack_allows(self):
        # The shim frame need not be the immediate caller.
        names = ["brainpy_state._foo", "brainpy.state", "brainpy", "__main__"]
        self.assertTrue(_is_internal_access(names, {}, {}))

    def test_pytest_loaded_allows(self):
        self.assertTrue(_is_internal_access(["__main__"], {"pytest": object()}, {}))

    def test_sphinx_loaded_allows(self):
        self.assertTrue(_is_internal_access(["__main__"], {"sphinx": object()}, {}))

    def test_override_env_allows(self):
        self.assertTrue(
            _is_internal_access(["__main__"], {}, {ALLOW_DIRECT_IMPORT_ENV: "1"})
        )

    def test_external_import_denied(self):
        self.assertFalse(_is_internal_access(["__main__", "user.app"], {}, {}))

    def test_empty_override_value_is_not_an_allowance(self):
        # An empty string is falsy -> treated as "not set".
        self.assertFalse(
            _is_internal_access(["__main__"], {}, {ALLOW_DIRECT_IMPORT_ENV: ""})
        )

    def test_brainpy_state_frame_is_not_the_shim(self):
        # The private package's own frames must not count as the public shim.
        self.assertFalse(_is_internal_access(["brainpy_state", "__main__"], {}, {}))


class TestEnforceNamespaceAccess(unittest.TestCase):
    """The enforcer, with the decision mocked for determinism."""

    def test_allowed_returns_silently(self):
        with mock.patch.object(_namespace, "_is_internal_access", return_value=True):
            self.assertIsNone(enforce_namespace_access())

    def test_denied_raises_importerror(self):
        with mock.patch.object(_namespace, "_is_internal_access", return_value=False):
            with self.assertRaises(ImportError):
                enforce_namespace_access()

    def test_message_points_at_brainpy_state_and_override(self):
        with mock.patch.object(_namespace, "_is_internal_access", return_value=False):
            with self.assertRaises(ImportError) as ctx:
                enforce_namespace_access()
        msg = str(ctx.exception)
        self.assertIn("brainpy.state", msg)       # the public namespace to use
        self.assertIn("brainpy.state.LIF", msg)   # a concrete example
        self.assertIn(ALLOW_DIRECT_IMPORT_ENV, msg)  # the documented escape hatch


class TestEndToEndImport(unittest.TestCase):
    """Real subprocesses exercising the live import machinery."""

    def test_direct_import_is_blocked(self):
        result = _run_import("import brainpy_state")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("brainpy.state", result.stderr)
        self.assertIn("ImportError", result.stderr)

    def test_direct_from_import_is_blocked(self):
        result = _run_import("from brainpy_state import LIF")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("brainpy.state", result.stderr)

    def test_direct_private_submodule_import_is_blocked(self):
        # Reaching into a private submodule still runs the package __init__ first.
        result = _run_import("from brainpy_state._base import Neuron")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("brainpy.state", result.stderr)

    def test_override_env_allows_direct_import(self):
        result = _run_import(
            "import brainpy_state; print('ok', brainpy_state.__version__)",
            allow_override=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ok", result.stdout)

    def test_public_namespace_path_succeeds(self):
        # The blessed path: brainpy.state -> shim -> brainpy_state, plus the
        # re-exported version dunder.
        result = _run_import(
            "import brainpy.state as s; print('ok', s.LIF.__name__, s.__version__)"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ok LIF", result.stdout)


if __name__ == "__main__":
    unittest.main()
