"""Rename the ``brainpy_state`` package to ``brainspike``.

The rename is purely mechanical so that branches cut before it can be rebased
by re-running this script on them:

    python scripts/rename_to_brainspike.py

It moves the package directory and rewrites every spelling of the old name
(``brainpy_state``, ``brainpy.state``, ``brainpy-state``, ``BRAINPY_STATE``)
in tracked text files. ``docs/specs/`` and ``changelog.md`` are left untouched
because they are historical records. These rename rules run only while
``brainpy_state/`` still exists; on every run, ``import brainspike as X`` is
rewritten to a plain ``import brainspike`` (with ``X.`` -> ``brainspike.``).
"""

import pathlib
import re
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent.parent
SKIP_PREFIXES = ('docs/specs/',)
SKIP_FILES = {'changelog.md', 'scripts/rename_to_brainspike.py'}
TEXT_SUFFIXES = {
    '.py', '.md', '.rst', '.toml', '.txt', '.yml', '.yaml', '.ipynb', '.cfg',
    '.ini', '.json', '.html', '.css', '.js', '.sh', '.in', '.bat', '',
}

RULES = [
    (re.compile(r'\bfrom brainpy import state as (\w+)'), r'import brainspike as \1'),
    (re.compile(r'\bimport brainpy\.state as (\w+)'), r'import brainspike as \1'),
    (re.compile(r'\bbrainpy\.state\b'), 'brainspike'),
    (re.compile(r'\bbrainpy_state'), 'brainspike'),
    (re.compile(r'\bbrainpy-state\b'), 'brainspike'),
    (re.compile(r'\bBRAINPY_STATE'), 'BRAINSPIKE'),
]
IMPORT_BRAINPY = re.compile(r'^(\s*)import brainpy\s*$', re.M)


def fix_bare_brainpy_import(text: str) -> str:
    """``import brainpy`` + ``brainpy.state.X`` became ``brainspike.X``; import it."""
    if 'brainspike.' not in text or re.search(r'^\s*import brainspike\b', text, re.M):
        return text
    if re.search(r'\bbrainpy\.', text):
        return IMPORT_BRAINPY.sub(r'\1import brainpy\n\1import brainspike', text, count=1)
    return IMPORT_BRAINPY.sub(r'\1import brainspike', text)


_ALIAS_IMPORT = re.compile(r'\bimport brainspike as (\w+)')


def drop_brainspike_aliases(text: str) -> str:
    """Rewrite ``import brainspike as X`` to ``import brainspike`` and ``X.`` to ``brainspike.``."""
    for alias in set(_ALIAS_IMPORT.findall(text)):
        text = re.sub(rf'(?<![\w.]){alias}\b', 'brainspike', text)
    return _ALIAS_IMPORT.sub('import brainspike', text)


def main():
    files = subprocess.run(['git', 'ls-files'], cwd=ROOT, check=True,
                           capture_output=True, text=True).stdout.split('\n')
    # After the rename, ``brainpy.state`` means BrainPy's own module (the kernel parity
    # tests import from it), so the rename rules only run on a tree that still has
    # ``brainpy_state/``. Dropping ``brainspike`` aliases is always safe.
    renaming = (ROOT / 'brainpy_state').is_dir()
    changed = 0
    for name in files:
        if not name or name in SKIP_FILES or name.startswith(SKIP_PREFIXES):
            continue
        path = ROOT / name
        if path.suffix not in TEXT_SUFFIXES or not path.is_file():
            continue
        old = path.read_text(encoding='utf-8')
        new = old
        if renaming:
            for pattern, repl in RULES:
                new = pattern.sub(repl, new)
        if path.suffix in {'.py', '.md', '.rst', '.ipynb'}:
            if renaming:
                new = fix_bare_brainpy_import(new)
            new = drop_brainspike_aliases(new)
        if new != old:
            path.write_text(new, encoding='utf-8')
            changed += 1
    if (ROOT / 'brainpy_state').is_dir():
        subprocess.run(['git', 'mv', 'brainpy_state', 'brainspike'], cwd=ROOT, check=True)
    print(f'rewrote {changed} files')


if __name__ == '__main__':
    main()
