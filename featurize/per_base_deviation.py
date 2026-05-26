#!/usr/bin/env python3
"""Per-base signal deviation analysis.

For each reference position, aggregates the deviation between the observed
segment mean signal and the expected k-mer level from the pore model, across
all reads covering that position.

Output TSV columns:
  ref_name  ref_pos  ref_base  strand  n_reads
  mean_dev  std_dev  diff1  t_stat  mean_dwell  dwell_var  kmer  [gt]

Metrics:
  mean_dev    – mean(observed_segment_mean − expected_kmer_level) across reads
  std_dev     – std of per-read deviations at this position
  diff1       – mean_dev[i] − mean_dev[i-1]  (NaN for the first position on
                each chromosome/strand; positions are ordered by ref_pos)
  t_stat      – Welch's t-statistic comparing the pooled deviations from the
                w=3 positions immediately to the left vs the w=3 positions
                immediately to the right of each position in sorted order
                (NaN when fewer than w neighbours exist on either side, or
                when either window has fewer than 2 observations)
  mean_dwell  – mean number of raw signal samples per segmentation event
                across all reads at this position  (peaks[i+1] − peaks[i])
  gt          – (optional) True/False ground-truth label; written only when
                --gt is supplied. True if (ref_name, ref_pos) is present in
                the BED file passed to --gt, False otherwise.
"""

import os
import sys
import argparse
import collections
import pod5
import pysam
import numpy as np
from scipy import stats
from tqdm import tqdm


# ── helpers ───────────────────────────────────────────────────────────────────

def get_pod5_readers(pod5_path):
    read_reader_map = {}
    if pod5_path.endswith('.pod5'):
        reader = pod5.Reader(pod5_path)
        for rid in reader.read_ids:
            read_reader_map[str(rid)] = reader
        return read_reader_map
    for fname in os.listdir(pod5_path):
        if fname.endswith('.pod5'):
            reader = pod5.Reader(os.path.join(pod5_path, fname))
            for rid in reader.read_ids:
                read_reader_map[str(rid)] = reader
    return read_reader_map


def load_level_table(path):
    kmer_levels = {}
    with open(path) as f:
        first_line = f.readline().strip()
        parts = first_line.split('\t')
        if parts[0] != 'kmer':
            kmer_levels[parts[0]] = float(parts[1])
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            kmer_levels[parts[0]] = float(parts[1])
    return kmer_levels


def normalize_level_table(kmer_levels):
    """Return a copy of kmer_levels with values z-score normalized across all
    k-mers (subtract global mean, divide by global std).  The same transform
    must be applied to each read's raw signal before computing deviations so
    that observed and expected values live in the same space."""
    values = np.array(list(kmer_levels.values()), dtype=np.float64)
    mu  = float(values.mean())
    sig = float(values.std())
    if sig < 1e-10:
        raise ValueError("K-mer level table has (near-)zero variance; "
                         "cannot normalize.")
    return {k: (v - mu) / sig for k, v in kmer_levels.items()}, mu, sig


def load_peaks_file(path):
    peaks = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            peaks[parts[0]] = np.array([int(p) for p in parts[1:]], dtype=np.int64)
    return peaks


def load_moves_file(path):
    peaks = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            read_id = parts[0]
            mv_str  = parts[1]
            ts_str  = parts[2]
            mv_parts = mv_str.split(',')
            stride = int(mv_parts[1])
            moves  = [int(x) for x in mv_parts[2:]]
            ts_offset = int(ts_str.split(':')[2])
            peak_positions = [ts_offset + i * stride
                              for i, m in enumerate(moves) if m == 1]
            peak_positions.append(ts_offset + len(moves) * stride)
            peaks[read_id] = np.array(peak_positions, dtype=np.int64)
    return peaks


def load_gt(path: str) -> set:
    """Load a BED file and return a set of (ref_name, ref_pos) tuples.

    Only the first two columns (chrom, chromStart) are used, matching the
    convention in find_mods.py.  ref_pos values are stored as int.
    """
    gt_set = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) < 2:
                continue
            gt_set.add((parts[0], int(parts[1])))
    return gt_set


_COMP = str.maketrans('ACGT', 'TGCA')


def _revcomp(seq):
    return seq.translate(_COMP)[::-1]


# ── per-base deviation extraction ─────────────────────────────────────────────

def get_ref_info_from_bam(bam_read):
    """Return (ref_seq, ref_positions) where ref_positions[i] is the absolute
    reference coordinate for local index i in ref_seq.

    For reverse-strand reads, ref_seq is reverse-complemented so that index 0
    corresponds to the 5' end of the read (matching signal direction).
    ref_positions is also reversed accordingly.
    """
    pairs = bam_read.get_aligned_pairs(with_seq=True)
    ref_bases = {}
    for qpos, rpos, rbase in pairs:
        if rpos is not None and rbase is not None:
            ref_bases[rpos] = rbase.upper()

    if not ref_bases:
        return None, None

    min_rpos = min(ref_bases.keys())
    max_rpos = max(ref_bases.keys())
    ref_seq = ''.join(ref_bases.get(p, 'N') for p in range(min_rpos, max_rpos + 1))
    ref_positions = list(range(min_rpos, max_rpos + 1))

    if bam_read.is_reverse:
        ref_seq = _revcomp(ref_seq)
        ref_positions = ref_positions[::-1]

    return ref_seq, ref_positions


def zscore_signal(signal):
    """Return a per-read z-score normalized copy of signal."""
    mu  = float(signal.mean())
    sig = float(signal.std())
    if sig < 1e-10:
        raise ValueError("Read signal has (near-)zero variance; cannot z-score.")
    return (signal - mu) / sig


def extract_per_base_deviations(signal, peaks, ref_seq, ref_positions,
                                 kmer_levels, kmer_size, center_idx):
    """Yield (ref_pos, deviation, obs_mean, exp_level, dwell, kmer) for each valid base.

    deviation  = observed_segment_mean − expected_kmer_level
    dwell      = number of raw signal samples in the segment  (peaks[i+1] − peaks[i])
    kmer       = the k-mer string used to look up the expected level
    """
    n_segments = len(peaks) - 1
    n_bases    = len(ref_seq)
    n_usable   = min(n_segments, n_bases)

    for i in range(n_usable):
        start = int(peaks[i])
        end   = int(peaks[i + 1])

        if start >= end or start < 0 or end > len(signal) or (end - start) < 2:
            continue

        k_start = i - center_idx
        k_end   = k_start + kmer_size
        if k_start < 0 or k_end > n_bases:
            continue

        kmer = ref_seq[k_start:k_end]
        if 'N' in kmer or kmer not in kmer_levels:
            continue

        obs_mean  = float(np.mean(signal[start:end]))
        exp_level = kmer_levels[kmer]
        deviation = obs_mean - exp_level
        dwell     = int(end - start)

        yield ref_positions[i], deviation, obs_mean, exp_level, dwell, kmer


def auto_detect_center_idx(kmer_size, sample_data, kmer_levels):
    """Try all center_idx values on sample reads; return best by mean Pearson r."""
    best_center = kmer_size // 2
    best_r = -2.0

    for candidate in range(kmer_size):
        rs = []
        for signal, peaks, ref_seq, ref_positions in sample_data:
            devs = list(extract_per_base_deviations(
                signal, peaks, ref_seq, ref_positions,
                kmer_levels, kmer_size, candidate))
            if len(devs) < 10:
                continue
            obs = np.array([d[2] for d in devs])
            exp = np.array([d[3] for d in devs])
            if np.std(obs) < 1e-10 or np.std(exp) < 1e-10:
                continue
            r, _ = stats.pearsonr(obs, exp)
            rs.append(r)
        if rs:
            mean_r = float(np.mean(rs))
            if mean_r > best_r:
                best_r = mean_r
                best_center = candidate

    return best_center, best_r


# ── downstream position-level metrics ────────────────────────────────────────

def welch_t(devs_a, devs_b):
    """Welch's t-statistic between two groups of deviations.

    Returns NaN if either group has fewer than 2 observations or if the
    pooled standard error is effectively zero.
    """
    n1, n2 = len(devs_a), len(devs_b)
    if n1 < 2 or n2 < 2:
        return float('nan')
    m1, m2 = np.mean(devs_a), np.mean(devs_b)
    v1 = np.var(devs_a, ddof=1)
    v2 = np.var(devs_b, ddof=1)
    se = np.sqrt(v1 / n1 + v2 / n2)
    if se < 1e-10:
        return float('nan')
    return float((m1 - m2) / se)


def compute_position_metrics(pos_data, min_reads, window=3):
    """Compute diff1 and t_stat for every eligible position.

    Positions are processed independently per (ref_name, strand) group,
    sorted by ref_pos, so that diff1 and t_stat only compare positions on
    the same chromosome and strand.

    Parameters
    ----------
    pos_data : dict
        Keys are (ref_name, ref_pos, strand); values are dicts with
        'ref_base', 'devs' (list of floats), 'dwells' (list of ints).
    min_reads : int
        Minimum number of reads required to include a position.
    window : int
        Half-width for the rolling Welch's t-test  (w positions on each side).

    Returns
    -------
    dict : same keys as pos_data, each value is a dict with:
        mean_dev, std_dev, diff1, t_stat, mean_dwell, dwell_var
    """
    # Filter to positions that pass the coverage threshold
    eligible = {k: v for k, v in pos_data.items() if len(v['devs']) >= min_reads}
    if not eligible:
        return {}

    # Group keys by (ref_name, strand)
    groups = collections.defaultdict(list)
    for key in eligible:
        ref_name, ref_pos, strand = key
        groups[(ref_name, strand)].append(key)

    results = {}

    for group_keys in groups.values():
        # Sort positions along the reference
        group_keys.sort(key=lambda k: k[1])

        # Pre-compute per-position summary stats
        pos_stats = {}
        for key in group_keys:
            arr  = np.array(eligible[key]['devs'], dtype=np.float64)
            n    = len(arr)
            dw   = np.array(eligible[key]['dwells'], dtype=np.float64)
            pos_stats[key] = dict(
                mean_dev   = float(np.mean(arr)),
                std_dev    = float(np.std(arr, ddof=1)) if n > 1 else 0.0,
                mean_dwell = float(np.mean(dw)),
                dwell_var  = float(np.var(dw, ddof=1)) if len(dw) > 1 else 0.0,
                devs       = arr,   # kept for windowed t-test
            )

        # diff1: difference in mean_dev between consecutive positions
        for idx, key in enumerate(group_keys):
            if idx == 0:
                pos_stats[key]['diff1'] = float('nan')
            else:
                prev_key = group_keys[idx - 1]
                pos_stats[key]['diff1'] = (pos_stats[key]['mean_dev']
                                           - pos_stats[prev_key]['mean_dev'])

        # Rolling Welch's t-stat: left window [i-w .. i-1] vs right [i+1 .. i+w]
        n_pos = len(group_keys)
        for idx, key in enumerate(group_keys):
            left_start  = max(0,       idx - window)
            right_end   = min(n_pos,   idx + window + 1)

            left_devs  = np.concatenate(
                [pos_stats[group_keys[j]]['devs']
                 for j in range(left_start, idx)]
            ) if idx > 0 else np.array([])

            right_devs = np.concatenate(
                [pos_stats[group_keys[j]]['devs']
                 for j in range(idx + 1, right_end)]
            ) if idx < n_pos - 1 else np.array([])

            pos_stats[key]['t_stat'] = welch_t(left_devs, right_devs)

        # Collect into results (drop the raw devs array)
        for key in group_keys:
            s = pos_stats[key]
            results[key] = dict(
                mean_dev   = s['mean_dev'],
                std_dev    = s['std_dev'],
                diff1      = s['diff1'],
                t_stat     = s['t_stat'],
                mean_dwell = s['mean_dwell'],
                dwell_var  = s['dwell_var'],
            )

    return results


# ── pileup chunking ───────────────────────────────────────────────────────────

def pileup_chunks(devs, dwells, max_pileup, rng):
    """Split devs/dwells into non-overlapping random chunks of size max_pileup.

    The arrays are shuffled once with a shared permutation so that devs[i] and
    dwells[i] stay paired.  Only complete chunks of exactly max_pileup reads
    are returned; any remainder (< max_pileup reads) is discarded.

    Parameters
    ----------
    devs : np.ndarray  shape (n,)
    dwells : np.ndarray  shape (n,)
    max_pileup : int
    rng : np.random.Generator

    Yields
    ------
    (chunk_devs, chunk_dwells) – each a 1-D np.ndarray of length max_pileup
    """
    n = len(devs)
    if n < max_pileup:
        # Not enough reads to fill even one chunk — yield the single group
        # so positions with fewer reads than max_pileup are still reported.
        yield devs, dwells
        return

    idx = rng.permutation(n)
    n_full_chunks = n // max_pileup
    for c in range(n_full_chunks):
        chunk_idx = idx[c * max_pileup: (c + 1) * max_pileup]
        yield devs[chunk_idx], dwells[chunk_idx]


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Per-base signal deviation analysis for modification detection.')
    parser.add_argument('--pod5', required=True,
                        help='Pod5 file or directory')
    parser.add_argument('--bam', required=True,
                        help='Aligned BAM file')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--peaks', help='Peaks file')
    group.add_argument('--moves', help='Moves TSV file')
    parser.add_argument('--level-table', required=True,
                        help='K-mer level table (pore model)')
    parser.add_argument('--output', required=True,
                        help='Output per-position TSV')
    parser.add_argument('--min-mapq', type=int, default=60,
                        help='Minimum MAPQ filter (default: 60)')
    parser.add_argument('--min-reads', type=int, default=5,
                        help='Minimum reads covering a position to report it '
                             '(default: 5)')
    parser.add_argument('--center-idx', type=int, default=None,
                        help='K-mer center index (auto-detected if not set)')
    parser.add_argument('--window', type=int, default=3,
                        help='Number of neighbouring positions on each side '
                             'to pool for the rolling Welch t-stat (default: 3)')
    parser.add_argument('--normalize', action='store_true',
                        help='Compare z-score values instead of raw current '
                             '(pA). Each read\'s signal is z-score normalized '
                             'per-read (mean=0, std=1) and the k-mer level '
                             'table is normalized by its own global mean and '
                             'std, so deviations are dimensionless and '
                             'comparable across reads with different baseline '
                             'currents.')
    parser.add_argument('--max-pileup', type=int, default=None,
                        metavar='N',
                        help='If set, positions covered by more than N reads '
                             'are split into multiple output rows of exactly N '
                             'reads each (randomly sampled without replacement, '
                             'one shuffle per position).  Any remainder reads '
                             'that do not fill a complete chunk are discarded.  '
                             'Positions with fewer than N reads produce a single '
                             'row as normal.  diff1 and t_stat are computed from '
                             'the full read pool at each position and are shared '
                             'across all split rows.')
    parser.add_argument(
        '--gt',
        nargs='?',
        const='__EMPTY__',
        default=None,
        metavar='BED',
        help='Path to a ground-truth BED file (columns: ref_name, ref_pos). '
             'When provided, appends a boolean "gt" column to the output TSV: '
             'True if the position is in the BED file, False otherwise. '
             'Supply the flag without a path (--gt alone) to write all-False '
             'gt labels (useful for negative-control runs).'
    )
    args = parser.parse_args()

    # ── validate --max-pileup ─────────────────────────────────────────────────
    if args.max_pileup is not None and args.max_pileup < 1:
        parser.error("--max-pileup must be a positive integer.")

    # ── load ground truth (optional) ─────────────────────────────────────────
    if args.gt is None:
        gt_set  = None          # --gt not given; column omitted entirely
        write_gt = False
    elif args.gt == '__EMPTY__':
        gt_set   = set()        # --gt given without a file; all False
        write_gt = True
        print("Ground truth: --gt supplied without a file; "
              "all positions labelled False.", file=sys.stderr)
    else:
        gt_set   = load_gt(args.gt)
        write_gt = True
        print(f"Ground truth: loaded {len(gt_set):,} positions from {args.gt}",
              file=sys.stderr)

    # ── load inputs ──────────────────────────────────────────────────────────
    print(f"Loading level table: {args.level_table}", file=sys.stderr)
    kmer_levels = load_level_table(args.level_table)
    kmer_size = len(next(iter(kmer_levels)))
    print(f"  {len(kmer_levels)} k-mers, k={kmer_size}", file=sys.stderr)

    if args.normalize:
        kmer_levels_norm, levels_mu, levels_sig = normalize_level_table(kmer_levels)
        print(f"  Normalization ON  (level table: mu={levels_mu:.4f} "
              f"sig={levels_sig:.4f})", file=sys.stderr)
        active_levels = kmer_levels_norm
    else:
        active_levels = kmer_levels

    if args.peaks:
        print(f"Loading peaks: {args.peaks}", file=sys.stderr)
        seg_borders = load_peaks_file(args.peaks)
    else:
        print(f"Loading moves: {args.moves}", file=sys.stderr)
        seg_borders = load_moves_file(args.moves)
    print(f"  {len(seg_borders)} reads", file=sys.stderr)

    print(f"Loading pod5: {args.pod5}", file=sys.stderr)
    read_reader_map = get_pod5_readers(args.pod5)
    print(f"  {len(read_reader_map)} reads in pod5", file=sys.stderr)

    bam_fh = pysam.AlignmentFile(args.bam, 'rb', check_sq=False)

    # ── auto-detect center_idx ────────────────────────────────────────────────
    center_idx = args.center_idx
    if center_idx is None:
        print(f"Auto-detecting center_idx over {kmer_size} candidates ...",
              file=sys.stderr)
        sample_data = []
        for bam_read in bam_fh:
            if (bam_read.is_supplementary or bam_read.is_secondary
                    or bam_read.is_unmapped):
                continue
            if bam_read.mapping_quality < args.min_mapq:
                continue
            read_id = bam_read.query_name
            if read_id not in seg_borders or read_id not in read_reader_map:
                continue
            try:
                pod5_read = next(read_reader_map[read_id].reads(
                    selection=[read_id]))
                signal = pod5_read.signal.astype(np.float64)
                if args.normalize:
                    signal = zscore_signal(signal)
                peaks = seg_borders[read_id]
                ref_seq, ref_positions = get_ref_info_from_bam(bam_read)
                if ref_seq is not None:
                    sample_data.append((signal, peaks, ref_seq, ref_positions))
            except Exception:
                continue
            if len(sample_data) >= 50:
                break

        center_idx, best_r = auto_detect_center_idx(
            kmer_size, sample_data, active_levels)
        print(f"  center_idx={center_idx}  (mean r={best_r:.4f} on "
              f"{len(sample_data)} reads)", file=sys.stderr)
        bam_fh.close()
        bam_fh = pysam.AlignmentFile(args.bam, 'rb', check_sq=False)
    else:
        print(f"Using center_idx={center_idx}", file=sys.stderr)

    # ── main pass: accumulate deviations and dwells per position ─────────────
    # Key: (ref_name, ref_pos, strand)
    # Value: {'ref_base': str, 'kmer': str, 'devs': list[float], 'dwells': list[int]}
    pos_data = collections.defaultdict(
        lambda: {'ref_base': 'N', 'kmer': '', 'devs': [], 'dwells': []})

    n_total = n_evaluated = n_skipped = 0
    ref_name_map = {}  # tid -> ref_name from BAM header

    for bam_read in tqdm(bam_fh, desc="Processing reads", file=sys.stderr):
        n_total += 1

        if (bam_read.is_supplementary or bam_read.is_secondary
                or bam_read.is_unmapped):
            n_skipped += 1
            continue
        if bam_read.mapping_quality < args.min_mapq:
            n_skipped += 1
            continue

        read_id = bam_read.query_name
        if read_id not in seg_borders or read_id not in read_reader_map:
            n_skipped += 1
            continue

        tid = bam_read.reference_id
        if tid not in ref_name_map:
            ref_name_map[tid] = bam_read.reference_name
        ref_name = ref_name_map[tid]
        strand = '-' if bam_read.is_reverse else '+'

        try:
            pod5_read = next(read_reader_map[read_id].reads(
                selection=[read_id]))
            signal = pod5_read.signal.astype(np.float64)
            if args.normalize:
                signal = zscore_signal(signal)

            peaks   = seg_borders[read_id]
            ref_seq, ref_positions = get_ref_info_from_bam(bam_read)
            if ref_seq is None:
                n_skipped += 1
                continue

            n_yielded = 0
            for ref_pos, deviation, obs_mean, exp_level, dwell, kmer in \
                    extract_per_base_deviations(
                        signal, peaks, ref_seq, ref_positions,
                        active_levels, kmer_size, center_idx):

                key = (ref_name, ref_pos, strand)
                entry = pos_data[key]
                entry['devs'].append(deviation)
                entry['dwells'].append(dwell)

                if entry['ref_base'] == 'N':
                    try:
                        local_i = ref_positions.index(ref_pos)
                        entry['ref_base'] = ref_seq[local_i]
                    except ValueError:
                        pass

                if not entry['kmer']:
                    entry['kmer'] = kmer

                n_yielded += 1

            if n_yielded > 0:
                n_evaluated += 1
            else:
                n_skipped += 1

        except Exception as e:
            n_skipped += 1
            if n_skipped <= 5:
                print(f"  Warning: skipped {read_id}: {e}", file=sys.stderr)
            continue

    # ── compute diff1, t_stat, dwell, and write output ────────────────────────
    print(f"\nComputing position-level metrics (window={args.window}) ...",
          file=sys.stderr)
    results = compute_position_metrics(pos_data,
                                       min_reads=args.min_reads,
                                       window=args.window)

    print(f"Writing per-position stats to {args.output} ...", file=sys.stderr)

    # ── dedup: one row per (ref_name, ref_pos) ───────────────────────────────
    # When the same reference position is covered by reads on both strands
    # (producing different ref_base characters), keep the entry with the
    # higher read count and discard the other.
    best_key_for_pos = {}   # (ref_name, ref_pos) -> winning (ref_name, ref_pos, strand) key
    for key in results:
        ref_name, ref_pos, strand = key
        pos_id = (ref_name, ref_pos)
        if pos_id not in best_key_for_pos:
            best_key_for_pos[pos_id] = key
        else:
            current_best = best_key_for_pos[pos_id]
            if len(pos_data[key]['devs']) > len(pos_data[current_best]['devs']):
                best_key_for_pos[pos_id] = key

    n_dropped = len(results) - len(best_key_for_pos)
    if n_dropped:
        print(f"  Dropped {n_dropped} duplicate-position entries "
              f"(kept higher-coverage strand)", file=sys.stderr)

    sorted_keys = sorted(best_key_for_pos.values(), key=lambda k: (k[0], k[1], k[2]))

    def fmt(v):
        """Format a float, replacing NaN with 0."""
        return '0' if np.isnan(v) else f"{v:.4f}"

    # ── RNG for --max-pileup shuffling ────────────────────────────────────────
    rng = np.random.default_rng()

    # ── write output TSV ─────────────────────────────────────────────────────
    gt_header = "\tgt" if write_gt else ""

    # Track expanded row count for the summary line
    n_rows_written = 0

    with open(args.output, 'w') as out_fh:
        out_fh.write(
            "ref_name\tref_pos\tref_base\tstrand\tn_reads\t"
            "mean_dev\tstd_dev\tdiff1\tt_stat\tmean_dwell\tdwell_var\tkmer"
            f"{gt_header}\n"
        )
        for key in sorted_keys:
            ref_name, ref_pos, strand = key
            entry = pos_data[key]
            s     = results[key]

            gt_col = ""
            if write_gt:
                in_gt = (ref_name, ref_pos) in gt_set
                gt_col = f"\t{in_gt}"

            # diff1 and t_stat are position-level metrics (computed from the
            # full read pool) and are shared across all pileup-split rows.
            diff1_str  = fmt(s['diff1'])
            t_stat_str = fmt(s['t_stat'])

            devs   = np.array(entry['devs'],   dtype=np.float64)
            dwells = np.array(entry['dwells'], dtype=np.float64)

            # Iterate over pileup chunks (one chunk when --max-pileup is unset)
            for chunk_devs, chunk_dwells in pileup_chunks(
                    devs, dwells, args.max_pileup, rng) \
                    if args.max_pileup is not None \
                    else [(devs, dwells)]:

                n_chunk = len(chunk_devs)
                chunk_mean_dev   = float(np.mean(chunk_devs))
                chunk_std_dev    = float(np.std(chunk_devs, ddof=1)) \
                                   if n_chunk > 1 else 0.0
                chunk_mean_dwell = float(np.mean(chunk_dwells))
                chunk_dwell_var  = float(np.var(chunk_dwells, ddof=1)) \
                                   if n_chunk > 1 else 0.0

                out_fh.write(
                    f"{ref_name}\t{ref_pos}\t{entry['ref_base']}\t{strand}\t"
                    f"{n_chunk}\t"
                    f"{fmt(chunk_mean_dev)}\t{fmt(chunk_std_dev)}\t"
                    f"{diff1_str}\t{t_stat_str}\t"
                    f"{chunk_mean_dwell:.2f}\t{chunk_dwell_var:.2f}\t"
                    f"{entry['kmer']}{gt_col}\n"
                )
                n_rows_written += 1

    # ── ground-truth summary (mirrors find_mods.py output style) ─────────────
    if write_gt and gt_set:
        n_gt = sum(
            1 for key in sorted_keys
            if (key[0], key[1]) in gt_set
        )
        print(f"\nGT positives in output: {n_gt:,}", file=sys.stderr)

    # ── summary ───────────────────────────────────────────────────────────────
    print(f"\n=== Summary ===", file=sys.stderr)
    print(f"Total BAM records:  {n_total}", file=sys.stderr)
    print(f"Reads evaluated:    {n_evaluated}", file=sys.stderr)
    print(f"Reads skipped:      {n_skipped}", file=sys.stderr)
    print(f"Positions reported: {len(results):,} "
          f"(>= {args.min_reads} reads)", file=sys.stderr)
    if args.max_pileup is not None:
        print(f"Max pileup:         {args.max_pileup} reads/row  "
              f"→ {n_rows_written:,} total rows written", file=sys.stderr)
    print(f"center_idx:         {center_idx}", file=sys.stderr)
    print(f"Window (t_stat):    {args.window} positions each side", file=sys.stderr)
    print(f"Normalization:      "
          f"{'z-score (per-read signal + level table)' if args.normalize else 'raw pA'}",
          file=sys.stderr)
    if write_gt:
        gt_desc = args.gt if args.gt != '__EMPTY__' else '(none — all False)'
        print(f"Ground truth BED:   {gt_desc}", file=sys.stderr)
    print(f"Output: {args.output}", file=sys.stderr)


if __name__ == '__main__':
    main()