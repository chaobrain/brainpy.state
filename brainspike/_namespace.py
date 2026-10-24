# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
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

"""Guard ensuring the package is reached through the public ``brainspike``.

``brainspike`` is the private implementation package; the supported public API
is the ``brainspike`` namespace, served by a thin shim in ``brainpy`` that does
``from brainspike import *``. Importing ``brainspike`` directly bypasses that
namespace, leaks the internal layout, and is unsupported. This module fails such an
import fast and loudly with a single actionable instruction.

The guard runs once, at the very top of ``brainspike/__init__.py`` -- so it gates
only the *entry* import. Internal ``from brainspike._xxx import ...`` statements
issued *during* initialisation never re-run it (the package is already in
``sys.modules``), which is exactly what the circular-import constraint in
``CLAUDE.md`` requires.
"""

import os
import sys

__all__ = ["ALLOW_DIRECT_IMPORT_ENV", "enforce_namespace_access"]

# Environment variable that, when truthy, lets ``brainspike`` be imported directly
# (escape hatch for tooling that legitimately needs the private package).
ALLOW_DIRECT_IMPORT_ENV = "BRAINSPIKE_ALLOW_DIRECT_IMPORT"

# ``__name__`` of the public shim module. When ``brainspike`` (or ``import brainpy``,
# which auto-loads it) triggers the import, this frame sits on the import call stack.
_PUBLIC_SHIM_NAME = "brainspike"

_MESSAGE = (
    "`brainspike` is the private implementation package and must not be "
    "imported directly. Use the public `brainspike` namespace instead:\n"
    "\n"
    "    import brainpy\n"
    "    neuron = brainspike.LIF(...)\n"
    "\n"
    "    # or\n"
    "    from brainpy import state\n"
    "    neuron = state.LIF(...)\n"
    "\n"
    "(Running the brainspike test suite or building its docs is allowed "
    f"automatically; set {ALLOW_DIRECT_IMPORT_ENV}=1 to override.)"
)


def _is_internal_access(stack_names, modules, environ):
    """Return whether a direct ``brainspike`` import should be permitted.

    Pure helper -- every input is injected so the decision is testable without
    touching the live interpreter state.

    Parameters
    ----------
    stack_names : iterable of str
        The ``__name__`` of each frame on the import call stack (innermost first).
    modules : container of str
        The loaded-module registry to probe, typically :data:`sys.modules`. Tested
        with the ``in`` operator, so a ``dict`` or ``set`` both work.
    environ : mapping
        The process environment, typically :data:`os.environ`.

    Returns
    -------
    bool
        ``True`` if access is permitted -- i.e. the import was triggered through the
        ``brainspike`` shim, the code is running under ``pytest`` or ``sphinx``, or
        the :data:`ALLOW_DIRECT_IMPORT_ENV` override is set to a truthy value.
        ``False`` for a genuine external ``import brainspike``.

    Examples
    --------
    .. code-block:: python

        >>> from brainspike._namespace import _is_internal_access
        >>> _is_internal_access(["brainspike", "brainpy"], {}, {})
        True
        >>> _is_internal_access(["__main__"], {"pytest": object()}, {})
        True
        >>> _is_internal_access(["__main__"], {}, {})
        False
    """
    if any(name == _PUBLIC_SHIM_NAME for name in stack_names):
        return True
    if "pytest" in modules or "sphinx" in modules:
        return True
    if environ.get(ALLOW_DIRECT_IMPORT_ENV):
        return True
    return False


def _iter_stack_names():
    """Yield the ``__name__`` of every frame currently on the stack, innermost first.

    Yields
    ------
    str
        The ``__name__`` global of each frame, or ``""`` when a frame has none.
    """
    frame = sys._getframe()
    while frame is not None:
        yield frame.f_globals.get("__name__", "")
        frame = frame.f_back


def enforce_namespace_access():
    """Abort a direct ``brainspike`` import, pointing the user at ``brainspike``.

    Inspects the live import call stack, :data:`sys.modules`, and
    :data:`os.environ` via :func:`_is_internal_access`. Allowed imports return
    silently; a genuine external ``import brainspike`` raises.

    Raises
    ------
    ImportError
        If ``brainspike`` was imported directly rather than through the
        ``brainspike`` namespace, and no maintenance allowance applies.

    See Also
    --------
    brainspike._compat.check_brainpy_compatibility : the sibling import-time guard.
    """
    if not _is_internal_access(_iter_stack_names(), sys.modules, os.environ):
        raise ImportError(_MESSAGE)
