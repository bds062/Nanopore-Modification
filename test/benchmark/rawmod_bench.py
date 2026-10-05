#!/usr/bin/env python3
"""RawMod site-level benchmark: recreate the paper's tool comparison, or add any RawMod checkpoint as a column.

  python test/benchmark/rawmod_bench.py config                      what is configured, what is missing
  python test/benchmark/rawmod_bench.py datasets                    every input, its public link, and whether it is here
  python test/benchmark/rawmod_bench.py fetch [SAMPLE ...] [--execute]   print (or run) the download commands
  python test/benchmark/rawmod_bench.py run [--slurm] [--steps ...] [--samples ...] [--tools ...]
  python test/benchmark/rawmod_bench.py rawmod --checkpoint my.pt --name mine [--slurm]
  python test/benchmark/rawmod_bench.py score [--holdout r81] [--rawmod rawmod,mine]
  python test/benchmark/rawmod_bench.py leaveout [--holdout r81]   RawMod in distribution vs. organism/library left out
  python test/benchmark/rawmod_bench.py status

Steps (run in this order; every step skips work whose output already exists, so any command can be rerun):
  subset     cut each sample's exact reads (data/read_ids/) out of the downloaded pod5
  basecall   Dorado 1.4.0 sup@v5.0.0 + move tables, aligned (one basecall shared by every tool)
  readsel    the K=10 reads each site is scored from, identical for every tool
  rows       ground-truth positions (motif rows rebuilt from the reference, the rest from data/rows/)
  refine     Remora signal refinement (RawMod input)
  tool       each tool / model on each sample
  aggregate  per-read calls -> per-site scores
  score      sequence-only nulls, tables (LaTeX + TSV), figures, summary.md

See test/README.md.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from bench import settings as ST  # noqa: E402

if ST.get('BENCH_WORK'):
    os.environ['BENCH_WORK'] = ST.get('BENCH_WORK')
if ST.get('SAMTOOLS'):
    os.environ['SAMTOOLS'] = ST.get('SAMTOOLS')

DEFAULT_FAMILIES = ['rawmod', 'dorado', 'unimeth', 'deepmod2', 'rockfish', 'methynano']
FAMILY_NEEDS = {'dorado': ['DORADO', 'DORADO_MODELS'], 'unimeth': ['UNIMETH', 'UNIMETH_CHECKPOINTS'],
                'unimeth_r81': ['UNIMETH', 'UNIMETH_FT_PYTHONPATH', 'UNIMETH_FT_CHECKPOINTS'], 'deepmod2': ['DEEPMOD2'], 'rockfish': ['ROCKFISH', 'ROCKFISH_MODEL'],
                'methynano': ['METHYNANO_DIR', 'DORADO_092_DIR'], 'rawmod': ['RAWHASH2_DIR', 'REFINE_PYTHON']}


def _samples_env():
    """Point bench.common at the sample sheet (built from config/datasets.tsv unless BENCH_SAMPLES is set)."""
    if 'SAMPLES_TSV' not in os.environ:
        from bench import datasets
        os.environ['SAMPLES_TSV'] = str(datasets.samples_tsv())


def _samples(a):
    from bench import datasets
    allowed = datasets.benchmark_samples()
    if not a.samples:
        return allowed
    bad = [s for s in a.samples if s not in allowed]
    if bad:
        raise SystemExit(f'unknown sample(s) {bad}; benchmark samples: {", ".join(allowed)}')
    return a.samples


def _subtools(spec, quiet=False):
    """--tools: 'default' | 'all' | comma list of families and/or subtool names."""
    from bench import common as C
    T = C.subtools()
    items = DEFAULT_FAMILIES if spec == 'default' else sorted({t['family'] for t in T.values()}) if spec == 'all' else spec.split(',')
    out = []
    for it in items:
        if it in T:
            out.append(it); continue
        members = [t for t in T if T[t]['family'] == it]
        if not members:
            raise SystemExit(f'--tools: {it!r} is neither a tool family nor a subtool ({", ".join(sorted(T))})')
        if it == 'rawmod' and spec == 'default':
            members = ['rawmod']                           # registered extra checkpoints: `rawmod --name` or --tools rawmod_<name>
        missing = [k for k in FAMILY_NEEDS.get(it, []) if not ST.get(k)]
        if missing and spec in ('default', 'all'):
            if not quiet:
                print(f'[skip] {it}: {", ".join(missing)} not set (rawmod_bench.py config)')
            continue
        out += members
    return out


def cmd_config(a):
    print(f"settings: environment > {os.environ.get('RAWMOD_PATHS_FILE', '~/.config/rawmod/paths.env')} > "
          f"{os.environ.get('BENCH_SITE_FILE', HERE / 'config' / 'site.env')}\n")
    for k, v, d in ST.report():
        print(f"  {'ok ' if v else '-- '} {k:24s} {v or '(not set)':60s}  {d}")
    print('\nRawMod models (config/rawmod_models.tsv):')
    from bench import models
    for st, m in models.load().items():
        print(f"  {st:24s} {m['checkpoint']}{'' if m['checkpoint'].exists() else '   [not found]'}")


def cmd_datasets(a):
    from bench import datasets
    rows = datasets.status_rows(a.samples)
    w = [max(len(str(r[i])) for r in rows) for i in range(4)]
    print(f"{'sample':{w[0]}}  {'kind':{w[1]}}  {'link':{w[2]}}  {'here':{w[3]}}  url")
    for r in rows:
        print(f'{r[0]:{w[0]}}  {r[1]:{w[1]}}  {r[2]:{w[2]}}  {r[3]:{w[3]}}  {r[4]}')
    miss = [r for r in rows if r[2] == 'unavailable']
    if miss:
        print(f'\n{len(miss)} input(s) have no public copy:')
        for r in miss:
            print(f'  {r[0]} {r[1]}: {r[5]}')
    if not ST.get('BENCH_DATA'):
        print('\nBENCH_DATA is not set, so local availability is not checked.')


def cmd_fetch(a):
    from bench import datasets
    todo = [m for m in datasets.manifest() if (not a.samples or set(m['sample'].split(',')) & set(a.samples))
            and (not a.kind or m['kind'] in a.kind.split(','))]
    for m in todo:
        c = datasets.fetch_commands(m)
        if c is None:
            if m['status'] == 'unavailable':
                print(f"# {m['sample']} {m['kind']}: no public copy ({m['notes'][:160]})")
            continue
        if datasets.local(m).exists() and not a.force:
            print(f"# here already: {m['sample']} {m['kind']} -> {datasets.local(m)}")
            continue
        print(c)
        if a.execute:
            from bench.steps import sh
            sh(c)
    if not a.execute:
        print('\n# printed only; rerun with --execute to download (pod5 sets are tens to hundreds of GB).')


def _plan_and_run(a, steps, subtools, score_args=()):
    from bench import runner
    tasks = runner.plan(steps, _samples(a), subtools, score_args)
    print(f'{len(tasks)} tasks: ' + ', '.join(f'{s} x{sum(t.step == s for t in tasks)}' for s in runner.ORDER
                                              if any(t.step == s for t in tasks)))
    (runner.run_slurm if a.slurm else runner.run_local)(tasks, dry=a.dry_run)


def _score_args(a):
    out = []
    if a.holdout:
        out += ['--holdout', a.holdout]
    if a.rawmod:
        out += ['--rawmod', a.rawmod]
    if getattr(a, 'out', None):
        out += ['--out', a.out]
    return out


def cmd_run(a):
    _samples_env()
    from bench import runner
    steps = a.steps.split(',') if a.steps else runner.ORDER
    bad = set(steps) - set(runner.ORDER)
    if bad:
        raise SystemExit(f'--steps: unknown {sorted(bad)}; choose from {runner.ORDER}')
    _plan_and_run(a, steps, _subtools(a.tools), _score_args(a))


def cmd_rawmod(a):
    """Register a checkpoint as a column and run everything it needs (prerequisite steps are skipped if done)."""
    _samples_env()
    from bench import models
    st = models.register(a.name, a.checkpoint, a.mask_bases, a.legacy_ch9, a.marks)
    print(f'registered {st} -> {a.checkpoint} (config/rawmod_models.tsv)')
    a.rawmod = a.rawmod or ','.join(dict.fromkeys(['rawmod', a.name]))
    _plan_and_run(a, ['subset', 'basecall', 'readsel', 'rows', 'refine', 'tool', 'aggregate', 'score'], [st], _score_args(a))


def cmd_score(a):
    _samples_env()
    _task_score(a)


def cmd_status(a):
    _samples_env()
    from bench import common as C, steps
    samples, subs = _samples(a), _subtools(a.tools, quiet=True)
    S = C.samples()
    head = ['subset', 'basecall', 'readsel', 'refine'] + subs
    print('sample'.ljust(18) + ' '.join(h[:14].ljust(14) for h in head))
    for s in samples:
        r = S[s]
        cells = [Path(r['sub_pod5']).exists(), Path(r['sub_bam'] + '.bai').exists(), C.readsel_path(s).exists(),
                 (C.WORK / s / 'refine' / C.INPUTS / 'peaks_refined.tsv').exists()]
        marks = ['done' if c else '.' for c in cells]
        for t in subs:
            marks.append('sites' if C.sites_path(s, t).exists() else 'ran' if steps.raw_output(s, t).exists() else '.')
        print(s.ljust(18) + ' '.join(m.ljust(14) for m in marks))
    print("\n'ran' = tool output present, 'sites' = aggregated (ready to score)")


def _task_score(a):
    """nulls -> tables -> curves -> summary.md for the chosen RawMod columns and holdout."""
    from bench import steps
    env = {'BENCH_HOLDOUT': a.holdout or '', 'BENCH_RAWMOD': a.rawmod or 'rawmod'}
    os.environ.update(env)
    from bench import common as C
    tag = C.TAG + (f'_holdout-{a.holdout}' if a.holdout else '')
    out = _score_dir(a)
    out.mkdir(parents=True, exist_ok=True)
    py = steps.py()
    if not (C.WORK / f'nulls_{tag}.tsv').exists():
        steps.sh(f'{py} -m bench.nulls', env=env, cwd=HERE)
    steps.sh(f'{py} -m bench.paper_table --out {steps.q(out)} | tee {steps.q(out / "paper_table.log")}', env=env, cwd=HERE)
    for rows, name in (('all', 'fig_benchmark_roc_all'), (None, 'fig_benchmark_roc_main')):
        steps.sh(f"{py} -m bench.paper_curves --out {steps.q(out)} --name {name}" + (f' --rows {rows}' if rows else ''),
                 env=env, cwd=HERE)
    steps.sh(f"{py} -m bench.paper_fig3 --dir {steps.q(out)} --name fig_benchmark_main", env=env, cwd=HERE)
    _summary(out)


def _score_dir(a):
    from bench import common as C
    tag = C.TAG + (f'_holdout-{a.holdout}' if a.holdout else '')
    return Path(a.out) if a.out else C.WORK / f'results_{tag}' / (a.rawmod or 'rawmod').replace(',', '+')


def cmd_leaveout(a):
    """Supplementary leave-out table: RawMod in distribution vs. RawMod with the row's organism or library withheld.
    Needs `score` (same --holdout) and the leave-out checkpoints run on every sample (evaluation/3_leaveout.sh)."""
    _samples_env()
    from bench import steps
    os.environ['BENCH_HOLDOUT'] = a.holdout or ''
    out = _score_dir(a)
    if not (out / 'paper_models.tsv').exists():
        raise SystemExit(f'{out}/paper_models.tsv not found: run `rawmod_bench.py score` with the same --holdout first')
    steps.sh(f'{steps.py()} -m bench.paper_leaveout --dir {steps.q(out)}', env={'BENCH_HOLDOUT': a.holdout or ''}, cwd=HERE)


def _md(df):
    """DataFrame -> GitHub markdown table (no tabulate dependency)."""
    cols = [df.index.name or ''] + [str(c) for c in df.columns]
    fmt = lambda v: '' if v != v else f'{v:.3f}' if isinstance(v, float) else str(v)
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '|'.join('---' for _ in cols) + '|']
    lines += ['| ' + ' | '.join([str(i)] + [fmt(v) for v in r]) + ' |' for i, r in zip(df.index, df.itertuples(index=False))]
    return '\n'.join(lines)


def _summary(out):
    import pandas as pd
    m = pd.read_csv(out / 'paper_models.tsv', sep='\t')
    m['auroc'] = pd.to_numeric(m['auroc'], errors='coerce')
    wide = m.pivot_table(index='row', columns='model', values='auroc', sort=False)
    t = pd.read_csv(out / 'paper_tables.tsv', sep='\t')
    ag = t.pivot_table(index='row', columns='tool', values='agnostic', sort=False, aggfunc='first')
    ag = ag.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')     # tools that were not run
    wide = wide.dropna(axis=1, how='all')
    log = (out / 'paper_table.log').read_text().splitlines() if (out / 'paper_table.log').exists() else []
    foot = [l for l in log if l.startswith(('Mean AUROC', 'Pooled AUROC'))]
    md = ['# RawMod benchmark results', '', f'Output directory: `{out}`', '',
          '## Modification-agnostic AUROC (one score per site per tool)', '', _md(ag), '']
    if foot:
        md += ['```'] + foot + ['```', '']
    md += ['## Per-model AUROC', '', _md(wide), '',
           'LaTeX tables: `tables/`; figures: `fig_benchmark_main.pdf` (Fig. 3), `fig_benchmark_roc_all.pdf`, '
           '`fig_benchmark_roc_main.pdf`; raw numbers: `paper_models.tsv`, `paper_tables.tsv`.']
    (out / 'summary.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md[:7]) + '\n' + '\n'.join(foot))
    print(f'\nfull summary: {out / "summary.md"}')


def cmd_task(a):
    """Internal: run one task (what local runs and SLURM jobs execute)."""
    _samples_env()
    from bench import steps
    if a.step == 'score':
        return _task_score(a)
    fn = getattr(steps, a.step)
    if a.step in ('tool', 'aggregate'):
        fn(a.sample, a.subtool)
    elif a.step == 'rows':
        fn()
    else:
        fn(a.sample)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    def common_run(p, tools=True):
        p.add_argument('--samples', nargs='+', help='default: every benchmark sample')
        if tools:
            p.add_argument('--tools', default='default',
                           help="'default' (RawMod + released baselines), 'all', or a comma list of families/subtools")
        p.add_argument('--slurm', action='store_true', help='submit one SLURM job per task (BENCH_SLURM_CPU/GPU)')
        p.add_argument('--dry-run', action='store_true', help='print what would run')

    def score_opts(p):
        p.add_argument('--holdout', default='', help="'r81' = only positions the released RawMod model never trained on")
        p.add_argument('--rawmod', default='', help='RawMod columns: comma list of config/rawmod_models.tsv names (default rawmod)')
        p.add_argument('--out', default='', help='output directory for tables/curves (default $BENCH_WORK/results_<tag>/<models>)')

    sub.add_parser('config', help='show settings').set_defaults(fn=cmd_config)
    p = sub.add_parser('datasets', help='list inputs, public links, local availability'); p.add_argument('samples', nargs='*')
    p.set_defaults(fn=cmd_datasets)
    p = sub.add_parser('fetch', help='print or run download commands'); p.add_argument('samples', nargs='*')
    p.add_argument('--kind', default='', help='pod5,reference,ground_truth,model'); p.add_argument('--execute', action='store_true')
    p.add_argument('--force', action='store_true'); p.set_defaults(fn=cmd_fetch)
    p = sub.add_parser('run', help='run the benchmark (all or some steps)'); common_run(p)
    p.add_argument('--steps', default='', help='comma list (default all): subset,basecall,readsel,rows,refine,tool,aggregate,score')
    score_opts(p); p.set_defaults(fn=cmd_run)
    p = sub.add_parser('rawmod', help='benchmark any RawMod checkpoint as its own column'); common_run(p, tools=False)
    p.add_argument('--checkpoint', required=True); p.add_argument('--name', required=True, help='column name (letters, digits, _)')
    p.add_argument('--mask-bases', action='store_true', help='checkpoint was trained with base identity blanked')
    p.add_argument('--legacy-ch9', action='store_true', help='checkpoint predates the ch9 fix')
    p.add_argument('--marks', default='5mC,5hmC,6mA,4mC,5hmU', help='modifications it is trained on (bold cells)')
    score_opts(p); p.set_defaults(fn=cmd_rawmod)
    p = sub.add_parser('score', help='tables, curves and summary from what has run'); score_opts(p); p.set_defaults(fn=cmd_score)
    p = sub.add_parser('leaveout', help='supplementary table: RawMod in distribution vs. leave-out checkpoints'); score_opts(p)
    p.set_defaults(fn=cmd_leaveout)
    p = sub.add_parser('status', help='what has run, per sample and tool'); p.add_argument('--samples', nargs='+')
    p.add_argument('--tools', default='default'); p.set_defaults(fn=cmd_status)
    p = sub.add_parser('task', help=argparse.SUPPRESS); p.add_argument('step'); p.add_argument('--sample'); p.add_argument('--subtool')
    score_opts(p); p.set_defaults(fn=cmd_task)
    a = ap.parse_args()
    a.fn(a)


if __name__ == '__main__':
    main()
