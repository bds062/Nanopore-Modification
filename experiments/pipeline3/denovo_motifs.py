#!/usr/bin/env python3
"""
De novo motif re-discovery from RawMod site scores — Figure 1.

Implements nanodisco's published discovery protocol (Tourancheau et al., Nat Methods
2021) verbatim, substituting RawMod's per-site modification probability for their
native-vs-WGA current-difference p-value. Everything downstream is unchanged, so the
recovered-motif count is comparable in kind to their 45/46:

  1. per-position score            (ours: RawMod P(modified); theirs: current-diff p)
  2. Fisher's method over a 5 bp sliding window
  3. rank peaks by combined p, keep the top N (nanodisco: 2000)
  4. extract 22 bp around each peak
  5. MEME  -dna -mod zoops -nmotifs 5 -minw 4 -maxw 14,  keep E <= 1e-30
  6. remove peaks explained by a discovered motif, iterate to surface rarer motifs

The claim only holds if the scored positions were chosen WITHOUT reference to the
motifs — use a genome-wide features.h5 built by featurize_genomewide.sh (no
--candidate-bed), and a checkpoint that never saw the organism (see score_genome.py).

Also runs the NULL: the identical procedure on an unmodified control (WGA for
H. pylori, PCR amplicon for SPO1) must recover nothing.

Usage:
  denovo_motifs.py --scores X_scores.tsv.gz --ref genome.fa --out-dir motifs/X \
      [--top-peaks 2000] [--expected-preset hpylori_26695] [--null]
"""
import argparse
import gzip
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.stats import chi2, rankdata

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'pipeline'))

MEME_BIN = os.environ.get('MEME_BIN', '/fs/cbcb-scratch/bds062/envs/meme/bin/meme')

# Known (REBASE-derived) motifs per organism, reused from the featurisation presets so
# there is a single source of truth for "what we should recover".
try:
    from motif_gt import _PRESETS as MOTIF_PRESETS       # noqa: E402
except Exception:                                        # pragma: no cover
    MOTIF_PRESETS = {}

_IUPAC = {'A': 'A', 'C': 'C', 'G': 'G', 'T': 'T', 'R': '[AG]', 'Y': '[CT]',
          'S': '[GC]', 'W': '[AT]', 'K': '[GT]', 'M': '[AC]', 'B': '[CGT]',
          'D': '[AGT]', 'H': '[ACT]', 'V': '[ACG]', 'N': '[ACGT]'}
_COMP = str.maketrans('ACGTN', 'TGCAN')


def read_fasta(path):
    seqs, name, parts = {}, None, []
    op = gzip.open if str(path).endswith('.gz') else open
    with op(path, 'rt') as fh:
        for line in fh:
            if line.startswith('>'):
                if name is not None:
                    seqs[name] = ''.join(parts).upper()
                name, parts = line[1:].split()[0], []
            else:
                parts.append(line.strip())
    if name is not None:
        seqs[name] = ''.join(parts).upper()
    return seqs


def load_scores(path):
    contig, pos, score = [], [], []
    op = gzip.open if str(path).endswith('.gz') else open
    with op(path, 'rt') as fh:
        header = fh.readline()
        for line in fh:
            c = line.rstrip('\n').split('\t')
            contig.append(c[0]); pos.append(int(c[1])); score.append(float(c[2]))
    return np.array(contig), np.array(pos, dtype=np.int64), np.array(score, dtype=np.float64)


def scores_to_pvalues(score):
    """Convert P(modified) to a right-tailed p-value by empirical rank.

    We do not have a parametric null, so use the empirical distribution over all scored
    positions: a site scoring above 1-q of the genome gets p = q. This is deliberately
    conservative and, importantly, makes the peak ranking depend only on the ORDER of
    scores, matching nanodisco's use of a ranked statistic.
    """
    n = len(score)
    r = rankdata(-score, method='average')        # rank 1 = highest score
    return np.clip(r / (n + 1.0), 1.0 / (n + 1.0), 1.0)


def fisher_smooth(contig, pos, pval, window=5):
    """nanodisco step 2: combine consecutive p-values with Fisher's method in a
    `window`-bp sliding window, per contig, respecting genomic coordinates.

    Vectorised. Fisher's method is X = -2 * sum(ln p_i) with df = 2k, so the whole
    window can be accumulated with searchsorted lookups per offset instead of a
    per-site scipy call (which is ~100x slower over 500k sites).
    """
    out = np.full(len(pval), 1.0)
    half = window // 2
    logp = np.log(np.clip(pval, 1e-300, 1.0))
    for c in np.unique(contig):
        m = np.nonzero(contig == c)[0]
        order = m[np.argsort(pos[m])]
        p = pos[order].astype(np.int64)
        lv = logp[order]
        acc = np.zeros(len(p)); cnt = np.zeros(len(p), dtype=np.int64)
        for d in range(-half, half + 1):
            tgt = p + d
            j = np.searchsorted(p, tgt)
            ok = (j < len(p))
            j_ok = np.clip(j, 0, len(p) - 1)
            hit = ok & (p[j_ok] == tgt)          # that genomic position was scored
            acc[hit] += lv[j_ok][hit]
            cnt[hit] += 1
        X = -2.0 * acc
        out[order] = chi2.sf(X, 2 * np.maximum(cnt, 1))
    return out


def pick_peaks(contig, pos, comb_p, top_n, min_sep=10):
    """nanodisco step 3: rank by combined p, take the top N, enforcing a minimum
    separation so one strong region does not consume the whole budget."""
    order = np.argsort(comb_p)
    chosen, taken = [], {}
    for i in order:
        c, p = contig[i], int(pos[i])
        bucket = taken.setdefault(c, [])
        if any(abs(p - q) < min_sep for q in bucket[-50:]):
            continue
        bucket.append(p); chosen.append(i)
        if len(chosen) >= top_n:
            break
    return np.array(chosen, dtype=np.int64)


def write_peak_fasta(seqs, contig, pos, idx, out_fa, flank=11):
    """nanodisco step 4: 22 bp around each peak."""
    n = 0
    with open(out_fa, 'w') as fh:
        for i in idx:
            c, p = contig[i], int(pos[i])
            s = seqs.get(c)
            if s is None:
                continue
            a, b = p - flank, p + flank
            if a < 0 or b > len(s):
                continue
            sub = s[a:b]
            if 'N' in sub:
                continue
            fh.write(f">{c}_{p}\n{sub}\n")
            n += 1
    return n


def run_meme(fa, outdir, nmotifs=5, minw=4, maxw=14, evalue=1e-30):
    """nanodisco step 5."""
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    cmd = [MEME_BIN, str(fa), '-dna', '-mod', 'zoops', '-nmotifs', str(nmotifs),
           '-minw', str(minw), '-maxw', str(maxw), '-oc', str(outdir), '-nostatus']
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=7200)
    except FileNotFoundError:
        raise SystemExit(f"MEME not found at {MEME_BIN}; set MEME_BIN")
    except subprocess.CalledProcessError as e:
        print(f"  MEME failed: {e.stderr.decode()[:400]}", file=sys.stderr)
        return []
    return parse_meme(outdir / 'meme.txt', evalue)


def parse_meme(meme_txt, evalue):
    """Return [(consensus, E-value, n_sites)] for motifs passing the E cutoff."""
    if not Path(meme_txt).exists():
        return []
    found = []
    txt = Path(meme_txt).read_text(errors='replace')
    # MEME 5.x: "MOTIF <consensus> MEME-1 width = W  sites = S  llr = L  E-value = E"
    for m in re.finditer(
            r'MOTIF\s+(\S+)\s+MEME-\d+\s+width\s*=\s*(\d+)\s+sites\s*=\s*(\d+).*?'
            r'E-value\s*=\s*(\S+)', txt):
        cons, _w, sites, ev = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
        try:
            evf = float(ev)
        except ValueError:
            continue
        if evf <= evalue:
            found.append((cons, evf, sites))
    return found


def motif_matches(discovered, expected):
    """Does a discovered consensus correspond to an expected REBASE motif?

    Accepts a hit if either string (or its reverse complement) contains the other as a
    IUPAC-compatible substring — MEME often reports a longer or shifted window than the
    canonical recognition site (e.g. `GGATCC` for `GATC`).
    """
    def rc(s):
        return s.translate(_COMP)[::-1]

    def compat(a, b):
        """b (IUPAC) occurs inside a (IUPAC)."""
        if len(b) > len(a):
            return False
        rx = ''.join(_IUPAC.get(ch, ch) for ch in b)
        return re.search(rx, a) is not None

    d = discovered.upper()
    for e in expected:
        e = e.upper()
        if compat(d, e) or compat(rc(d), e) or compat(e, d) or compat(rc(e), d):
            return e
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scores', required=True)
    ap.add_argument('--ref', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--top-peaks', type=int, default=2000)
    ap.add_argument('--rounds', type=int, default=3,
                    help='iterative rounds (nanodisco step 6)')
    ap.add_argument('--expected-preset', default=None,
                    help='motif_gt preset naming the REBASE motifs we should recover')
    ap.add_argument('--null', action='store_true',
                    help='label output as a negative control (nothing should be found)')
    a = ap.parse_args()

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    seqs = read_fasta(a.ref)
    contig, pos, score = load_scores(a.scores)
    print(f"loaded {len(score):,} scored sites across {len(set(contig))} contig(s)",
          flush=True)

    expected = []
    if a.expected_preset and a.expected_preset in MOTIF_PRESETS:
        expected = sorted({m for m, _o, _s in MOTIF_PRESETS[a.expected_preset]})
        print(f"expected (REBASE) motifs: {expected}", flush=True)

    pval = scores_to_pvalues(score)
    print("combining p-values (Fisher, 5 bp window) ...", flush=True)
    comb = fisher_smooth(contig, pos, pval, window=5)

    alive = np.ones(len(comb), dtype=bool)
    all_found, recovered = [], {}
    for rnd in range(1, a.rounds + 1):
        idx_alive = np.nonzero(alive)[0]
        if len(idx_alive) < 100:
            break
        sub_idx = pick_peaks(contig[idx_alive], pos[idx_alive], comb[idx_alive],
                             a.top_peaks)
        idx = idx_alive[sub_idx]
        fa = out / f'peaks_round{rnd}.fa'
        n = write_peak_fasta(seqs, contig, pos, idx, fa)
        print(f"round {rnd}: {n} peak sequences -> MEME", flush=True)
        if n < 50:
            break
        motifs = run_meme(fa, out / f'meme_round{rnd}')
        if not motifs:
            print(f"  round {rnd}: no motif passed E<=1e-30", flush=True)
            break
        for cons, ev, sites in motifs:
            hit = motif_matches(cons, expected) if expected else None
            all_found.append((rnd, cons, ev, sites, hit or ''))
            if hit:
                recovered.setdefault(hit, (cons, ev))
            print(f"  MOTIF {cons:16} E={ev:.2e} sites={sites:5}"
                  f"{'  <== matches ' + hit if hit else ''}", flush=True)
        # step 6: retire peaks explained by a discovered motif
        pats = [re.compile(''.join(_IUPAC.get(ch, ch) for ch in c)) for c, _, _ in motifs]
        for i in idx:
            c, p = contig[i], int(pos[i])
            s = seqs.get(c, '')
            w = s[max(0, p - 11):p + 11]
            if any(rx.search(w) for rx in pats):
                alive[i] = False

    tsv = out / 'discovered_motifs.tsv'
    with open(tsv, 'w') as fh:
        fh.write("round\tconsensus\tE_value\tn_sites\tmatches_expected\n")
        for r, c, e, s, h in all_found:
            fh.write(f"{r}\t{c}\t{e:.3e}\t{s}\t{h}\n")

    summary = out / 'summary.txt'
    with open(summary, 'w') as fh:
        kind = "NULL CONTROL (expect nothing)" if a.null else "de novo discovery"
        fh.write(f"{kind}\nscores: {a.scores}\n")
        fh.write(f"motifs passing E<=1e-30: {len(all_found)}\n")
        if expected:
            fh.write(f"expected: {len(expected)}  recovered: {len(recovered)}\n")
            for e in expected:
                got = recovered.get(e)
                fh.write(f"  {e:14} {'RECOVERED as ' + got[0] if got else 'missed'}\n")
    print(f"\n{'NULL: ' if a.null else ''}{len(all_found)} motif(s) passed E<=1e-30")
    if expected:
        print(f"recovered {len(recovered)}/{len(expected)} expected motifs")
    print(f"wrote {tsv}\nwrote {summary}")


if __name__ == '__main__':
    main()
