"""Datasets: where every input of the benchmark comes from, whether it is on this machine, and how to get it.

config/datasets.tsv has one line per file or file set:

  sample  kind  url  status  dest  citation  notes

  kind        pod5 | reference | ground_truth | model | code
  url         public location (s3://..., https://...), list:<file> (one URL per line, file relative to test/data),
              repo:<path> (shipped in this repository), derived (computed from the reference), or
              unavailable (no public copy; see notes)
  status      verified (the URL was checked to resolve) | in_repo | derived | unavailable
  dest        where test/data/download_*.sh (and `rawmod_bench.py fetch`) put it, relative to $BENCH_DATA
              (a directory for pod5 sets)

The sample sheet the steps use (sample, basecall BAM, read-subset pod5, reference) is built from it: the pod5 subset
and basecall live under $BENCH_WORK/<sample>/input/, the reference under $BENCH_DATA/<dest>. Setting BENCH_SAMPLES
uses an existing sheet instead.
"""
from __future__ import annotations

import shlex
from pathlib import Path

from . import settings as ST
from .settings import DATA_DIR, HERE

MANIFEST = HERE / 'config' / 'datasets.tsv'
READ_IDS = DATA_DIR / 'read_ids'


def manifest():
    cols = ['sample', 'kind', 'url', 'status', 'dest', 'citation', 'notes']
    out = []
    for line in open(MANIFEST):
        if line.strip() and not line.startswith('#'):
            v = line.rstrip('\n').split('\t')
            out.append(dict(zip(cols, v + [''] * (len(cols) - len(v)))))
    return out


def benchmark_samples():
    """Samples with a read-id list, i.e. the ones the benchmark scores (data/read_ids/MANIFEST.tsv order)."""
    return [l.split('\t')[0] for l in open(READ_IDS / 'MANIFEST.tsv') if l.strip() and not l.startswith('#')]


def expected_reads(sample):
    for l in open(READ_IDS / 'MANIFEST.tsv'):
        if l.startswith(sample + '\t'):
            return int(l.split('\t')[1])
    raise SystemExit(f'{sample}: no read-id list in {READ_IDS}')


def _data():
    return Path(ST.need('BENCH_DATA', 'datasets'))


def entries(sample, kind):
    return [m for m in manifest() if kind == m['kind'] and sample in m['sample'].split(',')]


def local(m):
    """Path of a manifest entry on this machine: $BENCH_DATA/<dest>."""
    return _data() / m['dest'] if m['dest'] else None


def source_pod5(sample):
    """pod5 files the sample's reads are cut from (every .pod5 under each pod5 entry's dest)."""
    out = []
    for m in entries(sample, 'pod5'):
        p = local(m)
        if p is None or not p.exists():
            raise SystemExit(f"{sample}: pod5 {m['url']} not found at {p}; run the dataset's test/data/download_*.sh"
                             + (' (no public copy: see config/datasets.tsv notes)' if m['status'] == 'unavailable' else ''))
        if p.is_dir():          # only the files this entry lists (a run directory can hold more)
            names = {u.rsplit('/', 1)[-1] for u in urls(m)}
            out += sorted(f for f in p.rglob('*.pod5') if f.name in names or len(urls(m)) == 1 and urls(m)[0].endswith('/'))
        else:
            out.append(p)
    if not out:
        raise SystemExit(f'{sample}: no pod5 entry in {MANIFEST}')
    return out


def reference(sample):
    e = entries(sample, 'reference')
    if not e:
        raise SystemExit(f'{sample}: no reference entry in {MANIFEST}')
    return local(e[0])


def samples_tsv():
    """Path of the sample sheet for bench.common (SAMPLES_TSV); built in $BENCH_WORK unless BENCH_SAMPLES is set."""
    if ST.get('BENCH_SAMPLES'):
        return Path(ST.get('BENCH_SAMPLES'))
    work = Path(ST.need('BENCH_WORK', 'outputs'))
    work.mkdir(parents=True, exist_ok=True)
    p = work / 'samples.tsv'
    lines = ['# sample\tbasecall bam\tread-subset pod5\treference   (built by bench.datasets.samples_tsv)']
    for s in benchmark_samples():
        ref = reference(s)
        lines.append(f'{s}\t{work}/{s}/input/base.bam\t{work}/{s}/input/sub.pod5\t{ref}')
    p.write_text('\n'.join(lines) + '\n')
    return p


def urls(m):
    """The entry's URLs (a list:<file> url expands to that file's lines)."""
    u = m['url']
    if u.startswith('list:'):
        return [l.strip() for l in open(DATA_DIR / u[5:]) if l.strip()]
    return u.split()


def fetchable(m):
    return (m['status'] == 'verified' and bool(m['dest']) and m['kind'] in ('pod5', 'reference', 'ground_truth')
            and not (m['kind'] != 'pod5' and m['url'].endswith('/')))      # PacBio run folders: test/data/download_*.sh


def fetch_commands(m):
    """Shell command that downloads one manifest entry to $BENCH_DATA/<dest>, or None if there is nothing to fetch
    (no public URL, derived, shipped in the repository, or provenance only).
    pod5 entries: dest is a directory; other kinds: dest is the file (a .gz URL for an uncompressed dest is decompressed)."""
    if not fetchable(m):
        return None
    dst, Q = local(m), shlex.quote
    if m['kind'] == 'pod5':
        cmds = [f'mkdir -p {Q(str(dst))}']
        for u in urls(m):
            if u.startswith('s3://'):
                cmds.append(f"aws s3 cp --no-sign-request {'--recursive ' if u.endswith('/') else ''}{Q(u)} {Q(str(dst))}/")
            else:
                cmds.append(f'wget -c -P {Q(str(dst))} {Q(u)}')
        return ' && '.join(cmds)
    u = urls(m)[0]
    gz = u.split('?')[0].endswith('.gz') and dst.suffix != '.gz'
    tgt = Q(str(dst) + ('.gz' if gz else ''))
    get = f'aws s3 cp --no-sign-request {Q(u)} {tgt}' if u.startswith('s3://') else f'wget -c -O {tgt} {Q(u)}'
    return f'mkdir -p {Q(str(dst.parent))} && {get}' + (f' && gunzip -f {tgt}' if gz else '')


def status_rows(samples=None):
    """(sample, kind, status, here?, url, note) for the `datasets` command."""
    out = []
    data = ST.get('BENCH_DATA')
    for m in manifest():
        if samples and not set(m['sample'].split(',')) & set(samples):
            continue
        here = '-'
        if data and fetchable(m):
            p = Path(data) / m['dest']
            if m['kind'] == 'pod5' and m['url'].startswith('list:'):
                names = {u.rsplit('/', 1)[-1] for u in urls(m)}
                n = len({f.name for f in p.rglob('*.pod5')} & names) if p.is_dir() else 0
                here = 'yes' if n == len(names) else f'{n}/{len(names)}'
            else:
                here = 'yes' if p.exists() else 'no'
        out.append((m['sample'], m['kind'], m['status'], here, m['url'], m['notes']))
    return out
