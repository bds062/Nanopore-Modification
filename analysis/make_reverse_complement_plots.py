#!/usr/bin/env python3
"""
Reverse-complement / strand-handling study for the DeepMod featurization.

Produces read-level "cartoon" pileups (like results9/5mC_cartoon.png) plus
large-scale metric plots that show how reverse-strand reads are mapped into the
pileup tensor, and the strand-collapse artifact it produces at palindromic
motif sites.

Mechanism (get_ref_info_from_bam, featurization.py:203-205): for a reverse read
the reference span is reverse-complemented and its positions reversed, but sites
are keyed by (contig, pos) with NO strand. At a palindromic 6mA GATC site the
reference ROW of an image therefore inherits A or its complement T depending on
which strand's reads dominate the image — even though every READ row (and the
true forward reference) is A. Empirically the reference row is 50%% A / 50%% T at
6mA positives, and for the flipped (T) images matches_ref = 0 for ALL reads.

Outputs (to --out-dir): 3 cartoons + 6 metric plots + 1 second-dataset metric.
"""
import argparse
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ── fixed, CVD-safe encodings (assigned by role, used consistently everywhere) ─
FWD = '#2563eb'      # forward strand (blue)
REV = '#e6820e'      # reverse strand (orange)  — blue/orange is CVD-safe
REFINK = '#111827'   # reference trace / ink
MUTED = '#6b7280'
BASE_COLORS = {'A': '#2ca25f', 'C': '#3182bd', 'G': '#f59e0b', 'T': '#de2d26', 'N': '#9ca3af'}
BASES = np.array(['A', 'C', 'G', 'T'])

plt.rcParams.update({
    'figure.dpi': 130, 'savefig.dpi': 130, 'font.size': 11,
    'axes.spawn' if False else 'axes.edgecolor': '#d1d5db',
    'axes.linewidth': 0.8, 'axes.grid': True, 'grid.color': '#eef2f7',
    'grid.linewidth': 0.8, 'axes.axisbelow': True,
})


def load(h5_path):
    h = h5py.File(h5_path, 'r'); a = dict(h.attrs)
    L = int(a['L']); hw = int(a['half_window']); W = int(a['W'])
    cs = hw * L
    return h, L, hw, W, cs


def img_block(h, idx):
    return h['tensors'][np.asarray(idx).tolist()].astype(np.float32)


# ── read-level cartoon (matplotlib, strand-coloured) ──────────────────────────
def cartoon(h, cs, L, W, idx, out_png, title, half_win=5, max_reads=10):
    t = h['tensors'][int(idx)].astype(np.float32)
    nr = int(h['n_reads'][int(idx)])
    rp = int(h['ref_pos'][int(idx)]); lab = int(h['labels'][int(idx)])
    center = W // 2
    b0, b1 = center - half_win, center + half_win + 1
    c0, c1 = b0 * L, b1 * L
    # reads are stored sorted by strand; interleave forward/reverse so the display
    # shows BOTH strands rather than the first block of one.
    all_strand = t[1:1 + nr, cs, 6]
    fwd_i = list(np.nonzero(all_strand > 0)[0]); rev_i = list(np.nonzero(all_strand < 0)[0])
    order = []
    while fwd_i or rev_i:
        if fwd_i:
            order.append(fwd_i.pop(0))
        if rev_i:
            order.append(rev_i.pop(0))
    order = np.array(order, dtype=int)
    raw_reads = t[1:1 + nr, c0:c1, 0][order]
    raw = np.vstack([t[0:1, c0:c1, 0], raw_reads])
    strand = all_strand[order]
    matches = t[1:1 + nr, cs, 8][order]
    # reference-row base per window position (from one-hot)
    ref_bases = [BASES[np.argmax(t[0, (b) * L:(b + 1) * L, 2:6].mean(0))]
                 if t[0, (b) * L:(b + 1) * L, 2:6].mean() > 0.05 else 'N'
                 for b in range(b0, b1)]
    show = min(nr, max_reads)
    clip = max(np.nanpercentile(np.abs(raw[np.isfinite(raw)]), 98), 1.0)
    ncols = c1 - c0
    cand_local = (center - b0)

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(12, 0.55 * (show + 1) + 1.8),
        gridspec_kw={'width_ratios': [3, 1], 'wspace': 0.12})
    # candidate shaded band
    for ax in (axL,):
        ax.axvspan(cand_local * L, (cand_local + 1) * L, color='#fde68a', alpha=0.55, zorder=0)

    def trace(ax, row, y, color, lw, alpha):
        v = np.clip(raw[row], -clip, clip) / clip
        good = np.isfinite(raw[row])
        ax.plot(np.arange(ncols)[good], y + 0.38 * v[good], color=color, lw=lw, alpha=alpha)

    # reference row on top
    trace(axL, 0, show, REFINK, 2.4, 1.0)
    for r in range(show):
        y = show - 1 - r
        col = FWD if strand[r] > 0 else REV
        trace(axL, 1 + r, y, col, 1.3, 0.85)
    axL.set_xlim(0, ncols); axL.set_ylim(-0.7, show + 0.8)
    axL.set_yticks([show] + list(range(show)))
    axL.set_yticklabels(['REF'] + [f"{'+' if strand[show-1-i]>0 else '−'}" for i in range(show)],
                        fontsize=9)
    for b, base in enumerate(ref_bases):
        axL.text((b + 0.5) * L, -0.5, base, ha='center', va='center',
                 color=BASE_COLORS[base], fontweight='bold', fontsize=12)
    for b in range(b1 - b0 + 1):
        axL.axvline(b * L, color='#e5e7eb', lw=0.6, zorder=0)
    axL.set_xticks([])
    axL.set_ylabel('reads (by strand)     ', fontsize=10)
    axL.set_title(title + f"\ncontig-pos {rp}  |  reference-row base at candidate = "
                  f"'{ref_bases[cand_local]}'  |  {nr} reads", fontsize=10, loc='left')

    # Right panel: per-read dip AT the candidate relative to THAT READ'S OWN
    # flanking bases. Deliberately NOT (read - reference row): the reference row
    # is the k-mer model's *expected* current while read rows are MAD-normalized
    # *measured* current, so their difference is dominated by a large constant
    # baseline offset (~2 units here) that swamps the per-read variation. Each
    # read minus its own flanks is baseline-free and is the quantity that
    # actually reflects a modification at the candidate.
    cand_sl = slice(cand_local * L, (cand_local + 1) * L)
    flank_mask = np.ones(raw.shape[1], dtype=bool)
    flank_mask[cand_sl] = False
    devs = np.array([np.nanmean(raw[1 + r, cand_sl]) - np.nanmean(raw[1 + r, flank_mask])
                     for r in range(show)])
    ys = np.arange(show)[::-1]
    cols = [FWD if strand[r] > 0 else REV for r in range(show)]
    axR.barh(ys, devs, color=cols, height=0.6)
    axR.axvline(0, color='#374151', lw=1.2, ls='--')
    axR.set_ylim(-0.7, show + 0.8); axR.set_yticks([])
    axR.set_xlabel('Δ current at candidate\n(read − its own flanks)', fontsize=10)
    frac_match = float(np.mean(matches[:show]))
    nfwd = int(np.sum(strand[:show] > 0)); nrev = int(np.sum(strand[:show] < 0))
    axR.set_title(f'per-read deviation\n{nfwd} fwd / {nrev} rev shown · '
                  f'{frac_match:.0%} match ref row', fontsize=9)

    handles = [Line2D([0], [0], color=REFINK, lw=2.4, label='reference row'),
               Line2D([0], [0], color=FWD, lw=1.6, label='forward read (+)'),
               Line2D([0], [0], color=REV, lw=1.6, label='reverse read (−)')]
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.02), fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_png, bbox_inches='tight'); plt.close(fig)
    print(f"  wrote {out_png}")


def gather_positive_stats(h, cs, nmax=8000):
    lab = h['labels'][:]; nr = h['n_reads'][:]; rp = h['ref_pos'][:]
    idx = np.nonzero(lab > 0)[0]
    if len(idx) > nmax:
        idx = np.random.default_rng(0).choice(idx, nmax, replace=False); idx.sort()
    blk = h['tensors'][idx.tolist()].astype(np.float32)
    refbase = BASES[np.argmax(blk[:, 0, cs, 2:6], axis=1)]
    rec = {'refbase': refbase, 'frac_rev': [], 'match_fwd': [], 'match_rev': [],
           'sig_fwd': [], 'sig_rev': [], 'read_base_A_frac': []}
    for i, ii in enumerate(idx):
        k = int(nr[ii]); reads = blk[i, 1:1 + k]
        s = reads[:, cs, 6]; m = reads[:, cs, 8]; sig = reads[:, cs, 0]
        rb = np.argmax(reads[:, cs, 2:6], axis=1)
        rec['frac_rev'].append(float(np.mean(s < 0)) if k else np.nan)
        rec['match_fwd'] += m[s > 0].tolist(); rec['match_rev'] += m[s < 0].tolist()
        rec['sig_fwd'] += sig[s > 0].tolist(); rec['sig_rev'] += sig[s < 0].tolist()
        rec['read_base_A_frac'].append(float(np.mean(rb == 0)) if k else np.nan)
    rec['frac_rev'] = np.array(rec['frac_rev'])
    return rec


def bar_two(ax, cats, vals, colors, ylabel, title, fmt='{:.0%}'):
    b = ax.bar(cats, vals, color=colors, width=0.62)
    for r, v in zip(b, vals):
        ax.text(r.get_x() + r.get_width() / 2, v, fmt.format(v), ha='center',
                va='bottom', fontsize=11, fontweight='bold')
    ax.set_ylabel(ylabel); ax.set_title(title, fontsize=12, loc='left')
    ax.set_ylim(0, max(vals) * 1.25 + 1e-9)


def metric_plots(h, cs, out, tag='Ecoli_DM (6mA, GATC)'):
    rec = gather_positive_stats(h, cs)
    rb = rec['refbase']

    # m1: reference-row base distribution at 6mA positives
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    u, c = np.unique(rb, return_counts=True)
    order = ['A', 'C', 'G', 'T']
    vals = [c[list(u).index(b)] / len(rb) if b in u else 0 for b in order]
    bar_two(ax, order, vals, [BASE_COLORS[b] for b in order],
            'fraction of positive sites', f'M1  Reference-row base at 6mA sites — {tag}')
    ax.axhline(1.0, color=MUTED, ls=':', lw=1)
    ax.text(0.02, 0.94, "biology: every 6mA is on an A → should be 100% A.\n"
            "observed: 50% flipped to complement T (strand collapse).",
            transform=ax.transAxes, fontsize=9.5, color=REFINK, va='top')
    fig.tight_layout(); fig.savefig(out / 'M1_refbase_distribution.png'); plt.close(fig)

    # m2: matches_ref by reference-row base
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    res = {}
    for base in ['A', 'T']:
        sel = rb == base
        # recompute matches for those images from the stored per-read matches is
        # aggregated; instead compute mean matches per group via read arrays:
        res[base] = None
    # simpler: aggregate matches per group directly
    lab = h['labels'][:]; nr = h['n_reads'][:]
    idxA, idxT, mA, mT = [], [], [], []
    posidx = np.nonzero(lab > 0)[0]
    posidx = np.random.default_rng(1).choice(posidx, min(4000, len(posidx)), replace=False)
    blk = h['tensors'][sorted(posidx.tolist())].astype(np.float32)
    base_row = BASES[np.argmax(blk[:, 0, cs, 2:6], axis=1)]
    for i, ii in enumerate(sorted(posidx.tolist())):
        k = int(nr[ii]); m = blk[i, 1:1 + k, cs, 8]
        (mA if base_row[i] == 'A' else mT if base_row[i] == 'T' else []).extend(m.tolist())
    vals = [np.mean(mA) if mA else 0, np.mean(mT) if mT else 0]
    bar_two(ax, ['ref row = A', 'ref row = T'], vals, [BASE_COLORS['A'], BASE_COLORS['T']],
            'mean matches_ref over reads', f'M2  Do reads match the reference row? — {tag}')
    ax.text(0.02, 0.5, "when the reference row is flipped to T,\n"
            "0% of reads match it — every read still reads A.",
            transform=ax.transAxes, fontsize=9.5, color=REFINK, va='center')
    fig.tight_layout(); fig.savefig(out / 'M2_matches_ref_by_refbase.png'); plt.close(fig)

    # m3: flip probability vs reverse-read fraction (binned)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    fr = rec['frac_rev']; flipped = (rb == 'T').astype(float)
    bins = np.linspace(0, 1, 11); bc = 0.5 * (bins[:-1] + bins[1:])
    which = np.digitize(fr, bins) - 1; which = np.clip(which, 0, len(bc) - 1)
    p = np.array([flipped[which == j].mean() if (which == j).any() else np.nan
                  for j in range(len(bc))])
    ax.plot(bc, p, '-o', color='#7c3aed', lw=2, ms=7)
    ax.set_xlabel('fraction of reverse-strand reads in the image')
    ax.set_ylabel('P(reference row flipped to complement)')
    ax.set_title(f'M3  Strand majority drives the flip — {tag}', fontsize=12, loc='left')
    ax.set_ylim(-0.03, 1.03)
    fig.tight_layout(); fig.savefig(out / 'M3_flip_vs_reverse_fraction.png'); plt.close(fig)

    # m4: strand composition histogram — perfectly bimodal (each image strand-PURE)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.hist(fr[np.isfinite(fr)], bins=25, color='#0ea5a4', edgecolor='white')
    ax.axvline(0.5, color=REFINK, ls='--', lw=1.2)
    ax.set_xlabel('fraction of reverse-strand reads per site')
    ax.set_ylabel('number of sites')
    ax.set_title(f'M4  Every candidate image is strand-PURE — {tag}', fontsize=12, loc='left')
    ax.text(0.30, 0.75, "no site is mixed:\n0% or 100% reverse, never between",
            transform=ax.transAxes, fontsize=10, color=REFINK, ha='center')
    fig.tight_layout(); fig.savefig(out / 'M4_strand_composition_hist.png'); plt.close(fig)

    # m5: per-read base is always A; reference row is 50/50 (grouped)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    read_A = np.nanmean(rec['read_base_A_frac'])
    groups = ['per-read base\n= A', 'reference row\n= A', 'reference row\n= T']
    vals = [read_A, np.mean(rb == 'A'), np.mean(rb == 'T')]
    cols = [FWD, BASE_COLORS['A'], BASE_COLORS['T']]
    bar_two(ax, groups, vals, cols, 'fraction', f'M5  Reads agree (A); the reference row does not — {tag}')
    fig.tight_layout(); fig.savefig(out / 'M5_perread_vs_refrow_base.png'); plt.close(fig)

    # m6: signal at candidate, forward vs reverse reads
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    sf = np.array(rec['sig_fwd']); sr = np.array(rec['sig_rev'])
    sf = sf[np.isfinite(sf)]; sr = sr[np.isfinite(sr)]
    rng = np.percentile(np.concatenate([sf, sr]), [1, 99])
    ax.hist(sf, bins=60, range=tuple(rng), color=FWD, alpha=0.6, density=True, label='forward (+)')
    ax.hist(sr, bins=60, range=tuple(rng), color=REV, alpha=0.6, density=True, label='reverse (−)')
    ax.set_xlabel('normalized current at candidate (channel 0)')
    ax.set_ylabel('density')
    ax.set_title(f'M6  Reverse reads carry a different signal — {tag}', fontsize=12, loc='left')
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out / 'M6_signal_by_strand.png'); plt.close(fig)
    print("  wrote M1..M6")
    return rec


def m8_base_determines_strand(h, cs, ref_fasta, out, tag):
    """The mechanism plot: the TRUE forward reference base fully determines which
    strand's reads survive --target-bases AC, because the filter is applied to
    each read's own strand-oriented (reverse-complemented) reference base."""
    seq = []
    for line in open(ref_fasta):
        if not line.startswith('>'):
            seq.append(line.strip())
    seq = ''.join(seq).upper()
    nr = h['n_reads'][:]; rp = h['ref_pos'][:]; lab = h['labels'][:]
    idx = np.random.default_rng(0).choice(len(lab), min(4000, len(lab)), replace=False)
    idx.sort()
    blk = h['tensors'][idx.tolist()].astype(np.float32)
    counts = {b: {'fwd': 0, 'rev': 0, 'mixed': 0} for b in 'ACGT'}
    for i, ii in enumerate(idx):
        k = int(nr[ii])
        if k == 0:
            continue
        p = int(rp[ii])
        if p >= len(seq):
            continue
        b = seq[p]
        if b not in counts:
            continue
        fr = float(np.mean(blk[i, 1:1 + k, cs, 6] < 0))
        counts[b]['fwd' if fr == 0 else 'rev' if fr == 1 else 'mixed'] += 1

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    x = np.arange(4); wdt = 0.26
    fwd = [counts[b]['fwd'] for b in 'ACGT']
    rev = [counts[b]['rev'] for b in 'ACGT']
    mix = [counts[b]['mixed'] for b in 'ACGT']
    ax.bar(x - wdt, fwd, wdt, label='all forward (+)', color=FWD)
    ax.bar(x, rev, wdt, label='all reverse (−)', color=REV)
    ax.bar(x + wdt, mix, wdt, label='mixed', color='#9ca3af')
    for xi, v in zip(x - wdt, fwd):
        if v: ax.text(xi, v, str(v), ha='center', va='bottom', fontsize=9, fontweight='bold')
    for xi, v in zip(x, rev):
        if v: ax.text(xi, v, str(v), ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(list('ACGT'))
    ax.set_xlabel('TRUE forward reference base at the candidate')
    ax.set_ylabel('number of sites')
    ax.set_title(f'M8  The forward base decides the strand — {tag}', fontsize=12, loc='left')
    ax.legend(frameon=False, fontsize=9)
    ax.text(0.02, 0.72,
            "--target-bases AC is applied to each read's OWN strand-oriented\n"
            "reference base. A reverse read reports the COMPLEMENT, so:\n"
            "   forward A/C  ->  only forward reads survive\n"
            "   forward T/G  ->  only reverse reads survive (complement is A/C)\n"
            "Mixed sites are therefore impossible by construction.",
            transform=ax.transAxes, fontsize=9, va='top', color=REFINK)
    fig.tight_layout(); fig.savefig(out / 'M8_forward_base_determines_strand.png')
    plt.close(fig)
    print("  wrote M8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default='/fs/cbcb-scratch/bds062/results/reverse_complement')
    ap.add_argument('--h5', default='/fs/cbcb-scratch/bds062/results/benchmark_results/Ecoli_DM_5kHz/features.h5')
    ap.add_argument('--h5b', default='/fs/cbcb-scratch/bds062/results/benchmark_results/arabidopsis/features.h5')
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    h, L, hw, W, cs = load(args.h5)
    lab = h['labels'][:]; nr = h['n_reads'][:]
    pos = np.nonzero(lab > 0)[0]
    blkc = h['tensors'][sorted(pos[:6000].tolist())].astype(np.float32)
    rowbase = BASES[np.argmax(blkc[:, 0, cs, 2:6], axis=1)]
    # Every candidate image is strand-PURE (the --target-bases AC filter is applied
    # to each read's own strand-oriented reference base, so forward-A/C positions
    # keep only forward reads and forward-T/G positions keep only reverse reads).
    # So select by strand purity, NOT by a strand mix — a mixed site does not exist.
    def pick_strand(want_rev, min_reads=15):
        for i, ii in enumerate(sorted(pos[:6000].tolist())):
            k = int(nr[ii])
            if k < min_reads:
                continue
            s = blkc[i, 1:1 + k, cs, 6]
            frac_rev = float(np.mean(s < 0))
            if (frac_rev == 1.0) == want_rev:
                return ii, rowbase[i]
        return sorted(pos[:6000].tolist())[0], '?'

    idx_fwd, base_fwd = pick_strand(want_rev=False)
    idx_rev, base_rev = pick_strand(want_rev=True)

    cartoon(h, cs, L, W, idx_fwd, out / 'C1_strand_positive_site.png',
            f"C1  STRAND-POSITIVE site — all reads are forward (+). Reference row = '{base_fwd}'",
            max_reads=10)
    cartoon(h, cs, L, W, idx_rev, out / 'C2_strand_negative_site.png',
            f"C2  STRAND-NEGATIVE site — all reads are reverse (−). Reference row = '{base_rev}'",
            max_reads=10)
    cartoon(h, cs, L, W, idx_rev, out / 'C3_strand_negative_wide.png',
            f"C3  Same strand-negative site, wider window — reverse reads carry the 6mA on their own strand",
            half_win=8, max_reads=14)

    metric_plots(h, cs, out, tag='Ecoli_DM (6mA, GATC)')
    m8_base_determines_strand(h, cs, '/fs/cbcb-scratch/bds062/ref_prepped/ecoli.fa',
                              out, tag='Ecoli_DM (6mA, GATC)')
    h.close()

    # second dataset: arabidopsis 5mC CpG (C/G palindrome) — same signature on C/G
    try:
        hb, Lb, hwb, Wb, csb = load(args.h5b)
        recb = gather_positive_stats(hb, csb, nmax=6000)
        rbb = recb['refbase']
        fig, ax = plt.subplots(figsize=(6.4, 4.6))
        order = ['A', 'C', 'G', 'T']
        vals = [np.mean(rbb == b) for b in order]
        bar_two(ax, order, vals, [BASE_COLORS[b] for b in order],
                'fraction of positive sites',
                'M7  Reference-row base at 5mC sites — arabidopsis (CpG)')
        ax.text(0.02, 0.9, "5mC is on C → should be ~100% C.\n"
                "CpG is a C/G palindrome → C collapses to G the same way.",
                transform=ax.transAxes, fontsize=9.5, va='top')
        fig.tight_layout(); fig.savefig(out / 'M7_arabidopsis_refbase.png'); plt.close(fig)
        hb.close()
        print("  wrote M7 (arabidopsis)")
    except Exception as e:
        print(f"  (skipped arabidopsis M7: {e})")

    print(f"\nAll plots in {out}")


if __name__ == '__main__':
    main()
