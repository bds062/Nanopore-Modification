#!/usr/bin/env python3
"""
orca_remake — causal-validation figure (DNA analog of ORCA's Mettl3/NSUN2
knockout validation).

ORCA validates that its modification calls are real, not artifacts, by
showing they collapse under a gene knockout that removes the modifying
enzyme. We have no knockout strain, but we have two causal-negative controls
for DNA, one per data source (both ONT and SPO1, nothing outside that
scope): the SPO1 PCR amplicon (bc01-05) is chemically stripped of every
modification (PCR erases all base modifications), and ONT's own `control.h5`
is an entirely unmodified synthetic construct. Both should show a low
modified-call rate. The native SPO1 whole-genome barcodes (bc06/07) carry
the true 5mC/5hmC/6mA/5hmU signal and should show a high one. A method whose
calls track real chemistry, not an artifact of the pileup image, should call
"modified" far more often on the native reads than on either negative
control.

Reads deepmod_umces/results5/eval/metrics.tsv (already computed by
eval_umces5.py on the SupCon model -- trained on bc02-07, i.e. SPO1 only,
with ONT scored zero-shot at eval time only) and turns it into a single
modified-call-rate bar figure. No new inference is run here (eval_umces5.py
was re-run separately to add the ont_control row).
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METRICS_TSV = '/fs/cbcb-scratch/bds062/results/deepmod_umces/results5/eval/metrics.tsv'
OUT_DIR = Path('/fs/cbcb-scratch/bds062/results/orca_remake/figures')

# dataset -> (display label, is_causal_negative)
# call rate = fraction of sites called "modified" by RawMod:
#   for a true-unmod control that is rec_unmod's complement (1 - rec_unmod = FPR)
#   for a true-modified set it is rec_mod itself (recall = TPR)
PANELS = {
    'barcode01':   ('SPO1 PCR amplicon\n(bc01, chemistry-stripped)', True),
    'ont_control': ('ONT control\n(synthetic, all unmod)', True),
    'barcode06':   ('SPO1 native WGS (bc06)\ntrue mods present', False),
    'barcode07':   ('SPO1 native WGS (bc07)\ntrue mods present', False),
}

COLORS = ['#D65F5F', '#D65F5F', '#4878CF', '#4878CF']  # red = causal-negative control, blue = true-positive


def main():
    rows = {}
    with open(METRICS_TSV) as fh:
        for row in csv.DictReader(fh, delimiter='\t'):
            rows[row['dataset']] = row

    labels, call_rates, colors, annotations = [], [], [], []
    for i, (name, (label, is_neg)) in enumerate(PANELS.items()):
        r = rows[name]
        if is_neg:
            rate = 1.0 - float(r['rec_unmod'])
            ann = f"false-positive rate\n{rate*100:.1f}%"
        else:
            rate = float(r['rec_mod'])
            ann = f"recall of true\nmodified sites\n{rate*100:.1f}%"
        labels.append(label)
        call_rates.append(rate)
        colors.append(COLORS[i])
        annotations.append(ann)

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    x = range(len(labels))
    bars = ax.bar(x, call_rates, color=colors, width=0.6)
    for b, rate, ann in zip(bars, call_rates, annotations):
        ax.text(b.get_x() + b.get_width() / 2, rate + 0.02, ann,
                ha='center', va='bottom', fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9.5, rotation=12, ha='right')
    ax.set_ylabel("RawMod fraction of sites called 'modified'", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_title(
        "Causal validation: modification calls track real chemistry,\n"
        "not the pileup image alone (DNA analog of ORCA's Mettl3/NSUN2 knockout check)",
        fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    out = OUT_DIR / 'fig_causal_validation.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == '__main__':
    main()
