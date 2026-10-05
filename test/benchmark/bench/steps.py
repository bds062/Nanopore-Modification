"""The benchmark as small, idempotent steps. Each step is one function of (sample[, subtool]) that skips itself when
its output exists, so any run can be repeated or resumed; rawmod_bench.py runs them locally or as SLURM jobs.

  per sample   subset -> basecall -> readsel -> refine (RawMod only)
  global       rows (after every readsel)
  per tool     tool:<subtool>  (dorado / unimeth / unimeth_r81 / deepmod2 / rockfish / methynano / rawmod)
  per tool     aggregate:<subtool>  (per-read calls -> per-site scores over the sample's K selected reads)
  global       score  (nulls, tables, ROC curves, plain-text summary)

Output layout: $BENCH_WORK/<sample>/{input,refine,<subtool>,rawmod_features}/...
"""
from __future__ import annotations

import gzip
import hashlib
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from . import settings as ST
from .settings import DATA_DIR as DATA, HERE, REPO

PER_READ_FAMILIES = ('dorado', 'unimeth', 'unimeth_r81', 'deepmod2', 'rockfish', 'methynano')
GPU_STEPS = {'basecall', 'tool'}


# ----------------------------------------------------------------------------------------------- helpers
def C():
    from . import common
    return common


def sh(cmd, env=None, cwd=None):
    """Run a shell command, echoing it; stop on failure."""
    print(f'+ {cmd}', flush=True)
    e = dict(os.environ, **(env or {}))
    r = subprocess.run(['bash', '-o', 'pipefail', '-c', cmd], env=e, cwd=cwd)
    if r.returncode:
        raise SystemExit(f'command failed ({r.returncode}): {cmd}')


def q(x):
    return shlex.quote(str(x))


def py():
    return ST.get('RAWMOD_PYTHON') or sys.executable


def sample_row(sample):
    S = C().samples()
    if sample not in S:
        raise SystemExit(f'unknown sample {sample!r}; known: {", ".join(S)}')
    return S[sample]


def done(path, what):
    if Path(path).exists() and Path(path).stat().st_size > 0:
        print(f'[skip] {what}: {path} exists')
        return True
    return False


def provenance(path, **kv):
    Path(path).write_text(''.join(f'{k}\t{v}\n' for k, v in kv.items()))


# ----------------------------------------------------------------------------------------------- per sample
def subset(sample):
    """Cut the sample's exact read set (test/data/read_ids/<sample>.txt.gz) out of the downloaded pod5 files."""
    from . import datasets
    r = sample_row(sample)
    out = Path(r['sub_pod5'])
    if done(out, f'{sample} read subset'):
        return
    src = datasets.source_pod5(sample)
    ids = DATA / 'read_ids' / f'{sample}.txt.gz'
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp_ids = out.with_suffix('.ids.txt')
    with gzip.open(ids, 'rt') as f:
        tmp_ids.write_text(f.read())
    sh(f"{q(ST.need('POD5', 'read subsets'))} filter {' '.join(map(q, src))} --ids {q(tmp_ids)} --missing-ok "
       f"--output {q(out)}.tmp && mv {q(out)}.tmp {q(out)}")
    import pod5
    got = sum(1 for _ in pod5.Reader(out).read_ids)
    want = datasets.expected_reads(sample)
    provenance(out.parent / 'SUBSET.txt', read_ids=ids, source=' '.join(map(str, src)), reads=got, expected=want)
    if got != want:
        print(f'WARNING {sample}: {got:,} of {want:,} listed reads found in the source pod5 files; the benchmark will '
              f'run but its numbers can differ from the paper', flush=True)


def basecall(sample):
    """Dorado 1.4.0 sup@v5.0.0 with move tables, aligned to the sample's reference (one basecall for every tool)."""
    r = sample_row(sample)
    bam = Path(r['sub_bam'])
    if done(f'{bam}.bai', f'{sample} basecall'):
        return
    bam.parent.mkdir(parents=True, exist_ok=True)
    dorado, models = ST.need('DORADO', 'basecalling'), ST.need('DORADO_MODELS', 'basecalling')
    sam = ST.get('SAMTOOLS', 'samtools')
    sh(f"{q(dorado)} basecaller {q(models)}/dna_r10.4.1_e8.2_400bps_sup@v5.0.0 {q(r['sub_pod5'])} --emit-moves "
       f"--reference {q(r['ref'])} --device cuda:all > {q(bam)}.unsorted.bam && "
       f"{q(sam)} sort -@ 8 -o {q(bam)} {q(bam)}.unsorted.bam && {q(sam)} index {q(bam)} && rm {q(bam)}.unsorted.bam")
    provenance(bam.parent / 'BASECALL.txt', dorado=dorado, model='dna_r10.4.1_e8.2_400bps_sup@v5.0.0',
               flags='--emit-moves, no qscore filter (readsel applies MIN_QS)', pod5=r['sub_pod5'], reference=r['ref'])


def readsel(sample):
    """The K reads every tool is scored from, per site (bench.readsel)."""
    if done(C().readsel_path(sample), f'{sample} read selection'):
        return
    sh(f'{q(py())} -m bench.readsel --sample {q(sample)}', cwd=HERE)


def refine(sample):
    """RawMod input: Remora reference-anchored refinement of the sample's basecalls (RawHash2 refine_moves_remora.py)."""
    r = sample_row(sample)
    out = C().WORK / sample / 'refine' / C().INPUTS
    if done(out / 'peaks_refined.tsv', f'{sample} refinement'):
        return
    out.mkdir(parents=True, exist_ok=True)
    script = f"{ST.need('RAWHASH2_DIR', 'RawMod refinement')}/test/scripts/refine_moves_remora.py"
    sh(f"{q(ST.need('REFINE_PYTHON', 'RawMod refinement'))} {q(script)} --pod5 {q(r['sub_pod5'])} --bam {q(r['sub_bam'])} "
       f"--level-table {q(ST.level_table())} --output {q(out / 'moves_refined.tsv')} --output-bam {q(out / 'reads_refined.bam')} "
       f"--ref-mapping --output-peaks {q(out / 'peaks_refined.tsv')} --min-mapq 0")
    provenance(out / 'PROVENANCE.txt', script=script, level_table=ST.level_table(), flags='--ref-mapping --min-mapq 0')


# ----------------------------------------------------------------------------------------------- global: rows
def rows(_=None):
    """Ground-truth positions of every row: motif rows rebuilt from the reference (bench.motif_rows), every other row
    installed from test/data/rows/. Every file is checked against test/data/rows/MANIFEST.tsv."""
    W = C().WORK / 'rows'
    man = [l.rstrip('\n').split('\t') for l in open(DATA / 'rows' / 'MANIFEST.tsv') if not l.startswith('#')]
    S = C().samples()
    ref_of = {c[0]: c[1] for c in C()._rows(HERE / 'config' / 'motifs.tsv')}
    motif = sorted({m[0] for m in man if m[4] == 'motif_rows'})
    todo = [r for r in motif if not (W / r / 'gt.bed').exists()]
    absent = [r for r in todo if not Path(S[ref_of[r]]['ref']).exists()]
    for r in absent:
        print(f'[skip] row {r}: reference of {ref_of[r]} not downloaded yet ({S[ref_of[r]]["ref"]})')
    if set(todo) - set(absent):
        sh(f"{q(py())} -m bench.motif_rows --rows {','.join(r for r in todo if r not in absent)}", cwd=HERE)
    bad = []
    for row, fname, n, md5, how in man:
        dst = W / row / fname
        if row in absent:
            continue
        if how == 'shipped' and not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(DATA / 'rows' / row / f'{fname}.gz', 'rb') as f:
                dst.write_bytes(f.read())
            prov = DATA / 'rows' / row / 'PROVENANCE.txt'
            if prov.exists():
                shutil.copy(prov, dst.parent / 'PROVENANCE.txt')
        if hashlib.md5(dst.read_bytes()).hexdigest() != md5:
            bad.append(str(dst))
    if bad:
        raise SystemExit('row files differ from test/data/rows/MANIFEST.tsv (delete them and rerun `rows`):\n  ' + '\n  '.join(bad))
    n = len({m[0] for m in man}) - len(absent)
    print(f'rows: {n} rows in {W} match the manifest' + (f'; {len(absent)} skipped (rerun after fetching)' if absent else ''))


# ----------------------------------------------------------------------------------------------- per tool
def _unimeth_cmd(r, t, out, code_path=None):
    """Released models: the `unimeth infer` command. Fine-tunes: `python -m unimeth.inference` from the patched source
    on PYTHONPATH (patches/unimeth-11215d4-finetune.patch adds the 5hmU/4mC/5hmC vocabulary), with the same python as
    the released install."""
    ck, flags, freq = (ST.expand(x) for x in t['arg'].split(';'))
    uni = ST.need('UNIMETH', 'UniMeth')
    if code_path:
        head = f"PYTHONNOUSERSITE=1 PYTHONPATH={q(code_path)} {q(Path(uni).parent / 'python')} -m unimeth.inference"
    else:
        head = f'PYTHONNOUSERSITE=1 {q(uni)} infer'
    return (f"{head} --pod5 {q(r['sub_pod5'])} --bam {q(r['sub_bam'])} --model {q(ck)} {flags} --pore_type R10.4.1 "
            f"--frequency {freq} --output_format tsv --out {q(out)}/calls.tsv.tmp --batch_size 128 "
            f"--signal_index {q(out)}/signal-index.sqlite && mv {q(out)}/calls.tsv.tmp {q(out)}/calls.tsv")


def tool(sample, subtool):
    """Run one subtool (config/subtools.tsv or a registered RawMod checkpoint) on one sample."""
    Cm = C()
    T = Cm.subtools()
    if subtool not in T:
        raise SystemExit(f'unknown subtool {subtool!r}; known: {", ".join(T)}')
    t, r, fam = T[subtool], sample_row(sample), T[subtool]['family']
    W = Cm.WORK / sample
    sam = ST.get('SAMTOOLS', 'samtools')
    if fam == 'rawmod':
        return rawmod(sample, subtool)
    if fam == 'dorado':
        out = W / subtool
        if done(out / 'calls.bam.bai', f'{sample} {subtool}'):
            return
        out.mkdir(parents=True, exist_ok=True)
        models = ST.need('DORADO_MODELS', 'Dorado modification calls')
        sh(f"{q(ST.need('DORADO', 'Dorado'))} basecaller {q(models)}/dna_r10.4.1_e8.2_400bps_sup@v5.0.0 {q(r['sub_pod5'])} "
           f"--modified-bases-models {q(models)}/{t['arg']} --reference {q(r['ref'])} --device cuda:0 > {q(out)}/calls.unsorted.bam && "
           f"{q(sam)} sort -@ 8 -o {q(out)}/calls.bam {q(out)}/calls.unsorted.bam && {q(sam)} index {q(out)}/calls.bam && "
           f"rm {q(out)}/calls.unsorted.bam")
        provenance(out / 'PROVENANCE.txt', dorado='1.4.0', base='dna_r10.4.1_e8.2_400bps_sup@v5.0.0', mods=t['arg'])
    elif fam in ('unimeth', 'unimeth_r81'):
        out = W / subtool / Cm.INPUTS
        if done(out / 'calls.tsv', f'{sample} {subtool}'):
            return
        out.mkdir(parents=True, exist_ok=True)
        code = None if fam == 'unimeth' else ST.need('UNIMETH_FT_PYTHONPATH', f'{subtool} (patched UniMeth)')
        sh(_unimeth_cmd(r, t, out, code))
        provenance(out / 'PROVENANCE.txt', subtool=subtool, arg=t['arg'], code=code or 'released unimeth')
    elif fam == 'deepmod2':
        out = W / 'deepmod2' / Cm.INPUTS
        if done(out / 'output.per_read', f'{sample} deepmod2'):
            return
        out.mkdir(parents=True, exist_ok=True)
        sh(f"{q(ST.get('DEEPMOD2_PYTHON') or py())} {q(ST.need('DEEPMOD2', 'DeepMod2'))} detect --bam {q(r['sub_bam'])} "
           f"--input {q(r['sub_pod5'])} --file_type pod5 --model bilstm_r10.4.1_5khz_v5.0 --ref {q(r['ref'])} --seq_type dna "
           f"--threads 8 --bam_threads 4 --batch_size 1024 --device cuda --output {q(out)} --prefix output --skip_unmapped")
    elif fam == 'rockfish':
        out = W / 'rockfish' / Cm.INPUTS
        if done(out / 'predictions.tsv', f'{sample} rockfish'):
            return
        out.mkdir(parents=True, exist_ok=True)
        sh(f"export TMPDIR=$(mktemp -d); {q(ST.need('ROCKFISH', 'Rockfish'))} inference -i {q(r['sub_pod5'])} "
           f"--bam_path {q(r['sub_bam'])} --model_path {q(ST.need('ROCKFISH_MODEL', 'Rockfish'))} -t 8 -b 4096 -d 0 "
           f"-o {q(out)}/predictions.tsv.tmp && mv {q(out)}/predictions.tsv.tmp {q(out)}/predictions.tsv")
    elif fam == 'methynano':
        out = W / subtool / 'mrb50'
        if done(out / 'calls.csv', f'{sample} {subtool}'):
            return
        out.mkdir(parents=True, exist_ok=True)
        d9, mn = ST.need('DORADO_092_DIR', 'MethyNano basecalls'), ST.need('METHYNANO_DIR', 'MethyNano')
        bam = W / 'hac42.moves.bam'
        # MethyNano's own basecall: dorado 0.9.2 hac@v4.2.0 with moves, primary only. Shared by the three MethyNano
        # subtools, which run as parallel jobs: the first takes the lock and basecalls, the others wait and reuse it.
        import fcntl
        with open(f'{bam}.lock', 'w') as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            if not Path(f'{bam}.bai').exists():
                sh(f"{q(d9)}/dorado basecaller {q(d9)}/dna_r10.4.1_e8.2_400bps_hac@v4.2.0 {q(r['sub_pod5'])} --emit-moves "
                   f"--reference {q(r['ref'])} --device cuda:0 > {q(bam)}.tmp.bam && {q(sam)} view -u -F 0x900 {q(bam)}.tmp.bam | "
                   f"{q(sam)} sort -@ 8 -o {q(bam)} && {q(sam)} index {q(bam)} && rm {q(bam)}.tmp.bam")
        ck = next((Path(mn) / 'models' / t['arg']).glob('*.pth'))
        # --min-read-bases 50: MethyNano's default 500 drops every ~112 bp oligo read; it is a read filter, not the model
        sh(f"cd {q(mn)} && PYTHONNOUSERSITE=1 {q(ST.get('METHYNANO_PYTHON') or py())} predict_aln.py --pod5 {q(r['sub_pod5'])} "
           f"--bam {q(bam)} --reference {q(r['ref'])} --ckpt {q(ck)} --output {q(out)}/calls.csv.tmp --min-read-bases 50 "
           f"--workers 7 --device auto --fp16 && mv {q(out)}/calls.csv.tmp {q(out)}/calls.csv")
    else:
        raise SystemExit(f'no runner for family {fam!r}')


def rawmod(sample, subtool='rawmod'):
    """One RawMod checkpoint on one sample: candidate positions (every row position scored on this sample) ->
    pileup images from exactly the benchmark's selected reads (cached per featurization setting, shared by every
    checkpoint with the same reads-per-image) -> scores_{plus,minus}.tsv in $BENCH_WORK/<sample>/<subtool>/<TAG>/."""
    from . import models, rawmod_cands
    Cm = C()
    reg = models.load()
    if subtool not in reg:
        raise SystemExit(f'{subtool} is not in config/rawmod_models.tsv (rawmod_bench.py rawmod --checkpoint ... --name ...)')
    m, r = reg[subtool], sample_row(sample)
    out = Cm.WORK / sample / subtool / Cm.TAG
    if done(out / 'scores_minus.tsv', f'{sample} {subtool}'):
        return
    out.mkdir(parents=True, exist_ok=True)
    nread = models.reads_per_image(m['checkpoint'])
    feat = Cm.WORK / sample / 'rawmod_features' / f'{Cm.TAG}_r{nread}'
    feat.mkdir(parents=True, exist_ok=True)
    rawmod_cands.build(sample, out=feat)
    key = hashlib.md5(b''.join((feat / f'cand_{s}.bed').read_bytes() for s in ('plus', 'minus'))).hexdigest()
    stamp = feat / 'CANDS_MD5'
    if stamp.exists() and stamp.read_text().strip() != key:      # rows changed since the images were built
        for s in ('plus', 'minus'):
            (feat / f'features_{s}.h5').unlink(missing_ok=True)
    stamp.write_text(key + '\n')
    ref_dir = Cm.WORK / sample / 'refine' / Cm.INPUTS
    for s, flag in (('plus', '--strand +'), ('minus', '--strand - --orient read')):
        h5 = feat / f'features_{s}.h5'
        if not (feat / f'cand_{s}.bed').stat().st_size:
            (out / f'scores_{s}.tsv').write_text('contig\tpos\tscore\tn_reads\n')
            continue
        prior = Cm.WORK / sample / 'rawmod' / Cm.TAG       # images built next to the scores by an earlier run
        if not h5.exists() and nread == 15 and (prior / f'features_{s}.h5').exists() and \
                (prior / f'cand_{s}.bed').read_bytes() == (feat / f'cand_{s}.bed').read_bytes():
            h5.symlink_to(prior / f'features_{s}.h5')                  # same positions, reads and settings: reuse
            print(f'reusing {prior / f"features_{s}.h5"}')
        if not h5.exists():
            sh(f"{q(py())} {q(REPO / 'rawmod' / 'featurization.py')} --pod5 {q(r['sub_pod5'])} --bam {q(ref_dir / 'reads_refined.bam')} "
               f"--peaks {q(ref_dir / 'peaks_refined.tsv')} --output {q(h5)}.tmp --level-table {q(ST.level_table())} --gt "
               f"--candidate-bed {q(feat / f'cand_{s}.bed')} --read-select {q(Cm.readsel_path(sample))} --min-reads 1 "
               f"--max-reads {nread} --max-images-per-base 1 --min-mapq 0 --normalize --half-window 10 --L 10 {flag} "
               f"&& mv {q(h5)}.tmp {q(h5)}")
        sh(f"{q(py())} {q(REPO / 'scripts' / 'test' / 'score_genome.py')} --h5 {q(h5)} --dataset {q(sample)}_{s} "
           f"--checkpoint {q(m['checkpoint'])} --out-dir {q(out)}{' --legacy-ch9' if m['legacy_ch9'] else ''} && "
           f"zcat {q(out)}/{q(sample)}_{s}_scores.tsv.gz | cut -f1,2,3,5 > {q(out)}/scores_{s}.tsv",
           env={'PILEUP_MASK_BASES': '1' if m['mask_bases'] else '0'})
    provenance(out / 'PROVENANCE.txt', checkpoint=m['checkpoint'], mask_bases=int(m['mask_bases']),
               legacy_ch9=int(m['legacy_ch9']), reads_per_image=nread, features=feat)


def aggregate(sample, subtool):
    """Per-read calls (or RawMod's per-site scores) -> the common per-site format over the K selected reads."""
    if done(C().sites_path(sample, subtool), f'{sample} {subtool} site scores'):
        return
    sh(f'{q(py())} -m bench.aggregate --sample {q(sample)} --subtool {q(subtool)}', cwd=HERE)


def raw_output(sample, subtool):
    """Path whose existence means the subtool has run on the sample (input of `aggregate`)."""
    Cm = C()
    t = Cm.subtools()[subtool]
    W, fam = Cm.WORK / sample, t['family']
    return {'dorado': W / subtool / 'calls.bam.bai', 'deepmod2': W / 'deepmod2' / Cm.INPUTS / 'output.per_read',
            'rockfish': W / 'rockfish' / Cm.INPUTS / 'predictions.tsv', 'methynano': W / subtool / 'mrb50' / 'calls.csv',
            'rawmod': W / subtool / Cm.TAG / 'scores_minus.tsv'}.get(fam, W / subtool / Cm.INPUTS / 'calls.tsv')
