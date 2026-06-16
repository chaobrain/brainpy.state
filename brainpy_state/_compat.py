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

"""Compatibility guard ensuring no stale ``brainpy`` shadows ``brainpy.state``."""

import re
import warnings

__all__ = ["MIN_BRAINPY", "check_brainpy_compatibility"]

# Minimum compatible ``brainpy`` release. Keep in sync with the ``brainpy>=...``
# pin in ``pyproject.toml`` -- the consistency is enforced by a test, not at the
# user's runtime, so this stays a dead-simple literal.
MIN_BRAINPY = (2, 7, 6)


def _parse_release(version_str):
    """Extract the PEP 440 release segment as an integer tuple.

    Parameters
    ----------
    version_str : str
        A version string such as ``"2.7.6"``, ``"2.7"``, ``"2.7.6.post1"``,
        ``"2.7.6rc1"``, ``"2.7.6.dev0"`` or ``"2.7.6+cuda12"``.

    Returns
    -------
    tuple of int or None
        ``(major, minor, patch)`` with a missing patch defaulting to ``0``, or
        ``None`` if the string does not begin with a ``major.minor`` release
        segment.

    Notes
    -----
    Only the leading release segment is compared, so pre-/post-/dev-/local
    suffixes of a given ``major.minor.patch`` are treated as equal to it. This is
    intentional: a ``2.7.6.dev0`` build of the target version is *not* blocked.

    Examples
    --------
    .. code-block:: python

        >>> from brainpy_state._compat import _parse_release
        >>> _parse_release("2.7.6")
        (2, 7, 6)
        >>> _parse_release("2.7")
        (2, 7, 0)
        >>> _parse_release("2.7.6.dev0")
        (2, 7, 6)
        >>> _parse_release("garbage") is None
        True
    """
    m = re.match(r"(\d+)\.(\d+)(?:\.(\d+))?", version_str)
    if m is None:
        return None
    return (int(m.group(1)), int(m.group(2)), int(m.group(3) or 0))


def check_brainpy_compatibility(min_version=MIN_BRAINPY):
    """Abort import if an incompatible ``brainpy`` is installed.

    ``brainpy_state`` is published as the ``brainpy.state`` namespace and requires
    a recent ``brainpy``. A stale ``brainpy`` (e.g. left behind by a pinned
    lockfile or a ``pip install --no-deps``) would shadow the namespace and
    produce confusing downstream errors, so this guard fails fast and loudly at
    import time with a single actionable instruction.

    Parameters
    ----------
    min_version : tuple of int, optional
        The minimum acceptable ``(major, minor, patch)``. Defaults to
        :data:`MIN_BRAINPY`. Injectable to keep the comparison testable.

    Raises
    ------
    ImportError
        If ``brainpy`` is installed but older than ``min_version``.

    Warns
    -----
    UserWarning
        If ``brainpy`` is installed but its version string cannot be parsed; the
        check is skipped rather than guessing.

    Notes
    -----
    A missing ``brainpy`` (``PackageNotFoundError``) is tolerated silently: it is
    a declared hard dependency, so its absence implies a deliberate ``--no-deps``
    environment that this guard does not fight. Every other failure path is left
    to propagate -- there is no catch-all, so the swallowed-exception bug this
    function once had cannot recur (the ``raise`` lives outside the ``try``).
    """
    from importlib.metadata import version, PackageNotFoundError

    try:
        brainpy_version = version("brainpy")
    except PackageNotFoundError:
        # brainpy not installed -- tolerated (it is a declared hard dependency).
        return

    release = _parse_release(brainpy_version)
    if release is None:
        warnings.warn(
            f"brainpy.state could not parse the installed brainpy version "
            f"{brainpy_version!r}; skipping the compatibility check.",
            stacklevel=2,
        )
        return

    if release < tuple(min_version):
        req = ".".join(map(str, min_version))
        raise ImportError(
            f"brainpy.state requires brainpy >= {req}, but found brainpy "
            f"{brainpy_version}. Upgrade with: pip install -U \"brainpy>={req}\""
        )
