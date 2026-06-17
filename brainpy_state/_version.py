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

from ._compat import _parse_release

__version__ = "0.1.0"
# Robust release-segment parse: never crash on PEP 440 suffixes (".post1", "rc1",
# ".dev0", "+local"). Falls back to (0, 0, 0) only if __version__ is malformed.
__version_info__ = _parse_release(__version__) or (0, 0, 0)

del _parse_release

