"""Plan the benchmark's steps as a dependency graph and run it locally (in order) or as SLURM jobs (one job per task,
chained with --dependency=afterok). Every task re-enters rawmod_bench.py as `task <step> [--sample] [--subtool]`, so
local and cluster runs execute the same code."""
from __future__ import annotations

import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, field

from . import settings as ST
from .settings import HERE

ORDER = ['subset', 'basecall', 'readsel', 'rows', 'refine', 'tool', 'aggregate', 'score']
CPU, GPU = 'cpu', 'gpu'


@dataclass
class Task:
    step: str
    sample: str | None = None
    subtool: str | None = None
    deps: list = field(default_factory=list)
    res: str = CPU
    extra: list = field(default_factory=list)       # extra CLI args (score options)

    @property
    def key(self):
        return ':'.join(x for x in (self.step, self.sample, self.subtool) if x)

    def argv(self):
        a = ['task', self.step]
        if self.sample:
            a += ['--sample', self.sample]
        if self.subtool:
            a += ['--subtool', self.subtool]
        return a + self.extra


def plan(steps, samples, subtools, score_args=()):
    """Tasks for `steps` (subset of ORDER) over `samples` x `subtools`, deps only on tasks that are in the plan
    (anything not in the plan is assumed done; each task checks its own inputs)."""
    from . import common as C
    T = C.subtools()
    want = set(steps)
    tasks, keys = [], set()

    def add(t):
        if is_done(t):
            return
        t.deps = [d for d in t.deps if d in keys]
        tasks.append(t); keys.add(t.key)

    for s in samples:
        if 'subset' in want:
            add(Task('subset', s))
        if 'basecall' in want:
            add(Task('basecall', s, deps=[f'subset:{s}'], res=GPU))
        if 'readsel' in want:
            add(Task('readsel', s, deps=[f'basecall:{s}']))
    if 'rows' in want:
        add(Task('rows'))
    allsel = [f'readsel:{s}' for s in samples]
    for s in samples:
        if 'refine' in want and any(T[t]['family'] == 'rawmod' for t in subtools):
            add(Task('refine', s, deps=[f'basecall:{s}']))
        for t in subtools:
            if 'tool' in want:
                d = [f'basecall:{s}', f'subset:{s}']
                if T[t]['family'] == 'rawmod':
                    d += [f'refine:{s}', 'rows'] + allsel       # candidates = every row position on this sample
                add(Task('tool', s, t, deps=d, res=GPU))
            if 'aggregate' in want:
                add(Task('aggregate', s, t, deps=[f'tool:{s}:{t}', f'readsel:{s}']))
    if 'score' in want:
        add(Task('score', deps=sorted(keys), extra=list(score_args)))
    return tasks


def is_done(t):
    """Has this task's output been produced already? (the task would skip itself; don't schedule it)"""
    from pathlib import Path
    from . import common as C, steps
    if t.step == 'score':
        return False
    if t.step == 'rows':
        man = [l.split('\t') for l in open(steps.DATA / 'rows' / 'MANIFEST.tsv') if not l.startswith('#')]
        return all((C.WORK / 'rows' / m[0] / m[1]).exists() for m in man)   # partial: the step reruns and fills in
    r = C.samples()[t.sample]
    p = {'subset': Path(r['sub_pod5']), 'basecall': Path(r['sub_bam'] + '.bai'), 'readsel': C.readsel_path(t.sample),
         'refine': C.WORK / t.sample / 'refine' / C.INPUTS / 'peaks_refined.tsv'}.get(t.step)
    if t.step == 'tool':
        p = steps.raw_output(t.sample, t.subtool)
    elif t.step == 'aggregate':
        p = C.sites_path(t.sample, t.subtool)
    return p.exists()


def _cli():
    return [ST.get('RAWMOD_PYTHON') or sys.executable, str(HERE / 'rawmod_bench.py')]


def run_local(tasks, dry=False):
    for i, t in enumerate(tasks, 1):
        print(f'\n=== [{i}/{len(tasks)}] {t.key}', flush=True)
        if dry:
            print('  ' + shlex.join(_cli() + t.argv()))
            continue
        r = subprocess.run(_cli() + t.argv(), env=os.environ.copy())
        if r.returncode:
            raise SystemExit(f'{t.key} failed; fix it and rerun the same command (finished steps are skipped)')


def run_slurm(tasks, dry=False):
    from . import common as C
    logs = C.WORK / 'logs' / 'bench'
    logs.mkdir(parents=True, exist_ok=True)
    jid = {}
    for t in tasks:
        res = ST.get('BENCH_SLURM_GPU' if t.res == GPU else 'BENCH_SLURM_CPU', '')
        dep = [jid[d] for d in t.deps if d in jid]
        cmd = ['sbatch', '--parsable', '--export=ALL', f'--job-name=rb_{t.key.replace(":", "_")}'[:60],
               f'--output={logs}/{t.key.replace(":", "_")}_%j.out'] + shlex.split(res)
        if dep:
            cmd.append('--dependency=afterok:' + ':'.join(dep))
        cmd += ['--wrap', shlex.join(_cli() + t.argv())]
        if dry:
            print(shlex.join(cmd)); jid[t.key] = f'<{t.key}>'
            continue
        out = subprocess.run(cmd, capture_output=True, text=True)
        if out.returncode:
            raise SystemExit(f'sbatch failed for {t.key}: {out.stderr.strip()}')
        jid[t.key] = out.stdout.strip().split(';')[0]
        print(f'{jid[t.key]}\t{t.key}', flush=True)
    if not dry:
        print(f'{len(jid)} jobs submitted; logs in {logs}')
