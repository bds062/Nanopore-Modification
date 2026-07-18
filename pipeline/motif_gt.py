#!/usr/bin/env python3
"""
Generate motif-based ground-truth BED files from a reference FASTA.

For well-characterised bacterial methyltransferases, every occurrence of the
recognition motif is methylated with near-100% efficiency in vivo.  These
motif-based labels are a valid independent ground truth (they are derived from
the genome sequence, not from the nanopore signal being modelled).

Supported presets (--preset flag):
  ecoli_dam      GATC → 6mA at the A (Dam methylase, E. coli / Anabaena)
  ecoli_dcm      CCWGG → 5mC at inner C (Dcm methylase, W = A or T)
  ecoli_msssi    CG → 5mC at C (MSssI, all CpG dinucleotides)
  hpylori_26695  H. pylori 26695 known methylation motifs (4mC/6mA)
  hpylori_j99    H. pylori J99 known methylation motifs
  tdenticola     T. denticola ATCC35405 known methylation motifs
  anabaena       Anabaena PCC7120 known methylation motifs

Custom motifs can be specified with --motif (IUPAC regex) and --mod-base.

Output:
  gt_modified.bed   — tab: ref_name, 0-based position of the modified base
  candidate.bed     — same as gt_modified.bed (all occurrences are modified;
                      complement / unmodified strand positions are excluded)

Usage:
  python motif_gt.py --ref REF.fa.gz --preset ecoli_dam \\
                     --outdir /path/to/gt/ecoli_dm
  python motif_gt.py --ref REF.fa.gz --motif GATC --mod-base A \\
                     --mod-offset 1 --outdir /path/to/gt/
"""

import argparse
import gzip
import os
import re
import sys

# IUPAC ambiguity codes
_IUPAC = {
    'A': 'A', 'C': 'C', 'G': 'G', 'T': 'T',
    'R': '[AG]', 'Y': '[CT]', 'S': '[GC]', 'W': '[AT]',
    'K': '[GT]', 'M': '[AC]', 'B': '[CGT]', 'D': '[AGT]',
    'H': '[ACT]', 'V': '[ACG]', 'N': '[ACGT]',
}


def iupac_to_regex(motif: str) -> str:
    return ''.join(_IUPAC.get(c.upper(), c) for c in motif)


# Each entry: (motif_iupac, offset_of_modified_base_in_motif, strand_convention)
# strand_convention: '+' = forward strand only, 'both' = both strands
_PRESETS = {
    # Dam methylase: GATC, adenine at offset 1 (0-based), both strands
    'ecoli_dam': [('GATC', 1, 'both')],

    # Dcm methylase: CCWGG (W=A/T), inner C at offset 1, both strands
    'ecoli_dcm': [('CCWGG', 1, 'both')],

    # MSssI: all CG dinucleotides, C at offset 0, both strands
    'ecoli_msssi': [('CG', 0, 'both')],

    # Anabaena PCC7120: Dam-like 6mA at GATC
    'anabaena': [('GATC', 1, 'both')],

    # H. pylori 26695 restriction-modification systems (REBASE):
    #   HpyIM    m4C at GCATG position 1
    #   HpyIIM   m6A at CTTCAAG position 6
    #   HpyIIIM  m4C at TCTTC position 3 (and complement GAAGA pos 1)
    'hpylori_26695': [
        ('GCATG',   1, '+'),    # HpyIM  m4C
        ('CATGC',   0, '+'),    # HpyIM  m4C complement strand
        ('CTTCAAG', 6, '+'),    # HpyIIM 6mA
        ('CTTTGAAG', 7, '+'),   # HpyIIM 6mA complement (note: palindrome variant)
        ('TCTTC',   3, '+'),    # HpyIIIM m4C
        ('GAAGA',   1, '+'),    # HpyIIIM complement
    ],

    # H. pylori J99 restriction-modification systems (REBASE):
    #   HpyAIII  m6A at GTNNNNNNAC  (non-palindromic N6-methyl)
    #   HpyAIV   m4C at TCNNNNNNNGC
    'hpylori_j99': [
        ('GTNNNNNNAC', 1, '+'),  # HpyAIII 6mA
        ('TCNNNNNNNGC', 1, '+'), # HpyAIV  4mC
    ],

    # T. denticola ATCC35405 (REBASE):
    #   TdeI  m6A at TATAC  position 1 (approx)
    #   Dam   m6A at GATC
    'tdenticola': [
        ('GATC', 1, 'both'),    # Dam-like 6mA
        ('TATAC', 1, '+'),      # TdeI 6mA (tentative)
        ('GTATA', 3, '+'),      # TdeI complement
    ],
}

# Complement map
_COMP = str.maketrans('ACGT', 'TGCA')


def revcomp(seq: str) -> str:
    return seq.translate(_COMP)[::-1]


def open_fasta(path: str):
    if path.endswith('.gz') or path.endswith('.bgz'):
        return gzip.open(path, 'rt')
    return open(path, 'r')


def find_motif_positions(seq: str, motif_regex: str, offset: int, strand: str,
                         chrom: str, chrom_start: int):
    """
    Yield (chrom, 0-based position of modified base, strand_char) for every
    occurrence of motif_regex in seq.
    chrom_start: 0-based offset of seq within the full chromosome.
    """
    pattern = re.compile(motif_regex, re.IGNORECASE)

    # Forward strand
    for m in pattern.finditer(seq):
        yield chrom, chrom_start + m.start() + offset, '+'

    # Reverse strand (if requested)
    if strand == 'both':
        rc_seq = revcomp(seq)
        rc_len = len(rc_seq)
        for m in pattern.finditer(rc_seq):
            # Convert rc position back to forward-strand coordinate
            rc_pos = m.start() + offset
            fwd_pos = chrom_start + (rc_len - rc_pos - 1)
            yield chrom, fwd_pos, '-'


def main():
    parser = argparse.ArgumentParser(
        description='Generate motif-based ground-truth BED from reference FASTA')
    parser.add_argument('--ref', required=True,
                        help='Reference FASTA (.fa, .fa.gz, .fna.bgz)')
    parser.add_argument('--outdir', required=True,
                        help='Output directory for BED files')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--preset', choices=list(_PRESETS.keys()),
                       help='Named methylation preset')
    group.add_argument('--motif', default=None,
                       help='IUPAC motif string (e.g. GATC)')
    parser.add_argument('--mod-base', default=None,
                        help='Modified base within motif (A or C); '
                             'used with --motif only')
    parser.add_argument('--mod-offset', type=int, default=None,
                        help='0-based offset of modified base in motif; '
                             'used with --motif only')
    parser.add_argument('--strand', choices=['+', 'both'], default='both',
                        help='Strand to search (used with --motif)')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    gt_path   = os.path.join(args.outdir, 'gt_modified.bed')
    cand_path = os.path.join(args.outdir, 'candidate.bed')

    if args.preset:
        motif_list = _PRESETS[args.preset]
    else:
        if args.mod_offset is None:
            parser.error('--mod-offset required with --motif')
        motif_list = [(args.motif, args.mod_offset, args.strand)]

    print(f"Reference  : {args.ref}", file=sys.stderr)
    print(f"Output dir : {args.outdir}", file=sys.stderr)
    if args.preset:
        print(f"Preset     : {args.preset} ({len(motif_list)} motif(s))",
              file=sys.stderr)
    for motif, offset, strand in motif_list:
        print(f"  motif={motif}  offset={offset}  strand={strand}", file=sys.stderr)

    n_written = 0
    seen = set()  # deduplicate (chrom, pos) across motifs / strands

    with open_fasta(args.ref) as fh, \
         open(gt_path, 'w') as fgt, \
         open(cand_path, 'w') as fcand:

        chrom = None
        seq_parts = []

        def flush_chrom():
            nonlocal n_written
            if chrom is None:
                return
            seq = ''.join(seq_parts).upper()
            for motif_str, offset, strand in motif_list:
                rx = iupac_to_regex(motif_str)
                for c, pos, s in find_motif_positions(seq, rx, offset, strand,
                                                      chrom, 0):
                    key = (c, pos)
                    if key in seen:
                        continue
                    seen.add(key)
                    line = f"{c}\t{pos}\n"
                    fgt.write(line)
                    fcand.write(line)
                    n_written += 1

        for line in fh:
            line = line.rstrip('\n')
            if line.startswith('>'):
                flush_chrom()
                chrom = line[1:].split()[0]
                seq_parts = []
                seen.clear()  # positions are per-chromosome
            else:
                seq_parts.append(line)

        flush_chrom()

    print(f"\nModified positions written: {n_written:,}", file=sys.stderr)
    print(f"  gt_modified.bed  → {gt_path}", file=sys.stderr)
    print(f"  candidate.bed    → {cand_path}", file=sys.stderr)


if __name__ == '__main__':
    main()
