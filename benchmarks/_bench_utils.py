# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Pure helpers shared by the benchmark scripts.

Nothing here imports JAX, so the helpers are cheap to test
(``python -m pytest benchmarks/check_bench_utils.py``) and safe to use in the
sweep driver process, which must stay free of JAX state so that each measured
configuration starts in a fresh subprocess.
"""
import json
import resource
import statistics
import sys

#: Line prefix used by a single-configuration run to hand its record to the
#: sweep driver (and to whoever captures the remote stdout).
RESULT_PREFIX = 'RESULT_JSON: '

#: Bumped whenever the record layout changes incompatibly.
SCHEMA_VERSION = 1

#: Keys every record carries, successful or failed.
REQUIRED_KEYS = ('schema_version', 'benchmark', 'status', 'config', 'meta')

#: Extra keys a successful record carries.
OK_KEYS = ('build', 'compile', 'steady', 'memory', 'sanity')

#: Allowed values of ``record['status']``.
STATUSES = ('ok', 'failed')


def summarize(samples):
    """Return ``{'median', 'min', 'max', 'n', 'samples'}`` of a non-empty list.

    Parameters
    ----------
    samples : sequence of float
        Repeated measurements.

    Returns
    -------
    dict
        Summary statistics; the samples are kept for later re-analysis.

    Raises
    ------
    ValueError
        If ``samples`` is empty.
    """
    samples = [float(s) for s in samples]
    if not samples:
        raise ValueError('summarize() needs at least one sample')
    return {
        'median': statistics.median(samples),
        'min': min(samples),
        'max': max(samples),
        'n': len(samples),
        'samples': samples,
    }


def ru_maxrss_bytes(ru_maxrss, platform=None):
    """Convert ``getrusage().ru_maxrss`` to bytes.

    Linux reports kibibytes, macOS reports bytes.

    Parameters
    ----------
    ru_maxrss : int
        Raw ``ru_maxrss`` value.
    platform : str, optional
        ``sys.platform`` value; defaults to the running platform.

    Returns
    -------
    int
        Peak resident set size in bytes.
    """
    platform = sys.platform if platform is None else platform
    if platform == 'darwin':
        return int(ru_maxrss)
    return int(ru_maxrss) * 1024


def peak_rss_bytes():
    """Peak resident set size of the current process, in bytes."""
    return ru_maxrss_bytes(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def current_rss_bytes(status_text=None):
    """Current resident set size (``VmRSS``) in bytes, or ``None`` if unknown.

    Parameters
    ----------
    status_text : str, optional
        Content of ``/proc/self/status``; read from disk when omitted.

    Returns
    -------
    int or None
        ``VmRSS`` in bytes, or ``None`` when the field is absent (non-Linux).
    """
    if status_text is None:
        try:
            with open('/proc/self/status') as f:
                status_text = f.read()
        except OSError:
            return None
    for line in status_text.splitlines():
        if line.startswith('VmRSS:'):
            parts = line.split()
            value = int(parts[1])
            unit = parts[2].lower() if len(parts) > 2 else 'kb'
            scale = {'kb': 1024, 'mb': 1024 ** 2, 'gb': 1024 ** 3, 'b': 1}[unit]
            return value * scale
    return None


def cpu_model(cpuinfo_text=None):
    """CPU model name from ``/proc/cpuinfo`` (``None`` if unavailable)."""
    if cpuinfo_text is None:
        try:
            with open('/proc/cpuinfo') as f:
                cpuinfo_text = f.read()
        except OSError:
            return None
    for line in cpuinfo_text.splitlines():
        if line.startswith('model name'):
            return line.split(':', 1)[1].strip()
    return None


def affinity_label(cpus, n_total):
    """Compact label of a CPU affinity set: ``'all'`` or ranges like ``'0-7,12'``.

    Parameters
    ----------
    cpus : iterable of int
        CPUs the process may run on (``os.sched_getaffinity(0)``).
    n_total : int
        Number of CPUs of the host.

    Returns
    -------
    str
        ``'all'`` when ``cpus`` covers every CPU, otherwise a range list.
    """
    cpus = sorted(set(int(c) for c in cpus))
    if len(cpus) >= n_total:
        return 'all'
    parts = []
    start = prev = cpus[0]
    for c in cpus[1:] + [None]:
        if c is not None and c == prev + 1:
            prev = c
            continue
        parts.append(str(start) if start == prev else f'{start}-{prev}')
        if c is not None:
            start = prev = c
    return ','.join(parts)


def config_key(record):
    """Identity of a configuration.

    ``(device, order, precision, brainevent backend, cpu affinity)``; the backend
    override is ``None`` for brainevent's default, the affinity ``'all'`` when
    the process was not pinned.
    """
    cfg = record['config']
    return (cfg['device'], int(cfg['order']), int(cfg['precision']),
            cfg.get('brainevent_backend'), cfg.get('cpu_affinity', 'all'))


def validate_record(record):
    """Check that ``record`` follows the result schema.

    Parameters
    ----------
    record : dict
        A single-configuration result.

    Raises
    ------
    ValueError
        Describing the first problem found.
    """
    missing = [k for k in REQUIRED_KEYS if k not in record]
    if missing:
        raise ValueError(f'record misses keys {missing}')
    if record['schema_version'] != SCHEMA_VERSION:
        raise ValueError(f"schema_version {record['schema_version']} != {SCHEMA_VERSION}")
    if record['status'] not in STATUSES:
        raise ValueError(f"unknown status {record['status']!r}")
    for k in ('device', 'order', 'precision'):
        if k not in record['config']:
            raise ValueError(f'config misses {k!r}')
    if int(record['config']['precision']) not in (32, 64):
        raise ValueError(f"precision must be 32 or 64, got {record['config']['precision']}")
    if record['status'] == 'ok':
        missing = [k for k in OK_KEYS if k not in record]
        if missing:
            raise ValueError(f'ok record misses keys {missing}')
        if 'median' not in record['steady'].get('s_per_sim_second', {}):
            raise ValueError('ok record misses steady.s_per_sim_second.median')
    elif 'error' not in record:
        raise ValueError("failed record needs an 'error' field")


def failure_record(config, meta, error, *, returncode=None, stderr_tail=None):
    """Build a schema-conforming record for a configuration that did not finish.

    Parameters
    ----------
    config : dict
        Must hold ``device``, ``order``, ``precision``.
    meta : dict
        Run metadata known to the caller.
    error : str
        Short reason (e.g. ``'OOM'`` or the exception text).
    returncode : int, optional
        Exit code of the subprocess.
    stderr_tail : str, optional
        Last lines of the subprocess stderr.

    Returns
    -------
    dict
        A record with ``status == 'failed'``.
    """
    rec = {
        'schema_version': SCHEMA_VERSION,
        'benchmark': 'brunel_simulator_baseline',
        'status': 'failed',
        'config': dict(config),
        'meta': dict(meta),
        'error': str(error),
        'returncode': returncode,
        'stderr_tail': stderr_tail,
    }
    validate_record(rec)
    return rec


def classify_failure(text, returncode=None):
    """Short failure label from a subprocess's return code and stderr tail.

    Parameters
    ----------
    text : str
        Error message or stderr tail.
    returncode : int, optional
        Subprocess return code (negative = killed by that signal).

    Returns
    -------
    str
        ``'OOM'`` for out-of-memory signatures, ``'killed (SIGKILL)'`` (usually
        the kernel OOM killer), ``'abort (SIGABRT)'`` (e.g. a failed check in a
        native kernel), ``'signal N'`` for other signals, otherwise ``'error'``.
    """
    t = (text or '').lower()
    if ('resource_exhausted' in t or 'out of memory' in t or 'memoryerror' in t
            or 'failed to allocate' in t):
        return 'OOM'
    if returncode == -9:
        return 'killed (SIGKILL)'
    if returncode == -6:
        return 'abort (SIGABRT)'
    if returncode is not None and returncode < 0:
        return f'signal {-returncode}'
    return 'error'


def parse_result_lines(text):
    """Extract every record printed with :data:`RESULT_PREFIX` from ``text``.

    Lines may carry a remote-shell prefix before the marker; everything up to
    and including the marker is dropped.
    """
    records = []
    for line in text.splitlines():
        pos = line.find(RESULT_PREFIX)
        if pos < 0:
            continue
        records.append(json.loads(line[pos + len(RESULT_PREFIX):]))
    return records


def merge_records(existing, new):
    """Merge record lists; a new record replaces an old one with the same key.

    Order: existing keys keep their position, new keys are appended.
    """
    out = list(existing)
    index = {config_key(r): i for i, r in enumerate(out)}
    for r in new:
        k = config_key(r)
        if k in index:
            out[index[k]] = r
        else:
            index[k] = len(out)
            out.append(r)
    return out


def sort_records(records):
    """Sort by device (cpu first), order, precision."""
    dev_rank = {'cpu': 0, 'gpu': 1}
    return sorted(records, key=lambda r: (dev_rank.get(r['config']['device'], 9),
                                          r['config'].get('brainevent_backend') or '',
                                          r['config'].get('cpu_affinity', 'all') != 'all',
                                          int(r['config']['order']),
                                          int(r['config']['precision'])))


def _fmt(x, nd=3):
    return '-' if x is None else f'{x:.{nd}f}'


def markdown_table(records):
    """Render records as a GitHub markdown table (one row per configuration)."""
    cols = ('device', 'cpus', 'backend', 'order', 'N', 'prec', 'status', 'build s',
            'compile s', 'recompile s', 's / sim-s', 'ms / step',
            'peak mem GiB', 'E rate Hz', 'load')
    rows = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    for r in sort_records(records):
        c = r['config']
        lead = [c['device'], c.get('cpu_affinity', 'all'),
                c.get('brainevent_backend') or 'default', str(c['order']),
                str(5 * int(c['order'])), f"f{c['precision']}"]
        load = _fmt(r['meta'].get('loadavg_1min'), 2)
        if r['status'] != 'ok':
            rest = [f"FAILED ({r['error']})"] + ['-'] * 7 + [load]
        else:
            mem = r['memory'].get('peak_bytes')
            rest = ['ok',
                    _fmt(r['build']['build_s'], 1),
                    _fmt(r['compile']['compile_s'], 2),
                    _fmt(r['compile'].get('warmup_xla_events_s'), 2),
                    _fmt(r['steady']['s_per_sim_second']['median'], 3),
                    _fmt(r['steady']['ms_per_step']['median'], 4),
                    _fmt(None if mem is None else mem / 2 ** 30, 3),
                    _fmt(r['sanity']['e_rate_hz']['median'], 2),
                    load]
        rows.append('| ' + ' | '.join(lead + rest) + ' |')
    return '\n'.join(rows)

