#!/usr/bin/env python3
"""
featurize_pileup.py
===================
Builds a DeepVariant-style pileup tensor for each candidate reference position
and saves the featurization + labels to an HDF5 file.

Multi-image support
-------------------
When a position has more than max_reads reads, read membership is randomly
partitioned into floor(n_reads / max_reads) non-overlapping chunks of exactly
max_reads reads each.  Use --max-images-per-base to cap how many chunks/images
are emitted per reference base.  Within each chunk, rows are ordered in the
DeepVariant/DeepSomatic style: haplotype tag (HP; untagged = 0), then
alignment start.  The per-row ``image_idx`` dataset records which chunk a row
came from (0-based), so downstream code can group images back by position.

Positions with fewer than max_reads reads still produce exactly one image
(using all available reads, zero-padded to max_reads as before) provided they
meet --min-reads.

Tensor layout
-------------
Shape per image: (max_reads + 1, window_positions * L, n_channels)

  Row 0          : REFERENCE ROW (like DeepVariant's reference track)
  Rows 1..max_reads : individual reads (padded with zeros if coverage < max_reads)

  window_positions = 2*half_window + 1
  L                = fixed number of signal samples per base (resampled)
  n_channels       = 9:

    Ch 0  raw signal          resampled to L samples, z-scored or MAD per read
                              (reference row: expected kmer level, broadcast flat)
    Ch 1  dwell time          log1p(dwell), broadcast across L columns
                              (reference row: 0)
    Ch 2  is_A                one-hot base identity
    Ch 3  is_C                one-hot base identity
    Ch 4  is_G                one-hot base identity
    Ch 5  is_T                one-hot base identity
    Ch 6  strand              +1 forward / -1 reverse, broadcast
                              (reference row: 0)
    Ch 7  mapping quality     MAPQ / 60, clipped to [0, 1], broadcast
                              (reference row: 0)
    Ch 8  matches reference   1 if read base == ref base else 0, broadcast
                              (reference row: 0)

HDF5 layout
-----------
  /tensors    float32  (N_images, max_reads+1, window_positions * L, n_channels)
  /labels     int8     (N_images,)   1 = modified, 0 = unmodified
  /ref_names  bytes    (N_images,)   reference chromosome name
  /ref_pos    int64    (N_images,)   reference position
  /n_reads    int16    (N_images,)   reads in this image (always == max_reads
                                     except possibly for the only image of a
                                     position with < max_reads coverage)
  /image_idx  int16    (N_images,)   0-based index of this image within its
                                     position (useful for grouping back)

  N_images >= N_positions; positions with high coverage contribute multiple rows.

Usage
-----
  python featurize_pileup.py \
      --pod5      reads.pod5 \
      --bam       aligned.bam \
      --moves     moves.tsv \
      --output    pileup_features.h5 \
      --gt        modifications.bed \
      --half-window  10 \
      --L            10 \
      --max-reads    30 \
      --max-images-per-base 3 \
      --min-reads    5  \
      --min-mapq     60
"""

import os
import sys
import argparse
import collections

import numpy as np
import pod5
import pysam
import h5py
from tqdm import tqdm


# ── constants ─────────────────────────────────────────────────────────────────

N_CHANNELS = 9

# One-hot channel indices for each base
BASE_ONEHOT = {
    'A': (1, 0, 0, 0),
    'C': (0, 1, 0, 0),
    'G': (0, 0, 1, 0),
    'T': (0, 0, 0, 1),
    'N': (0, 0, 0, 0),
}
# Channel layout (for reference):
# 0 raw_signal / kmer_level
# 1 dwell (log1p)
# 2 is_A
# 3 is_C
# 4 is_G
# 5 is_T
# 6 strand
# 7 mapq_norm
# 8 matches_ref


# ── I/O helpers ───────────────────────────────────────────────────────────────

def get_pod5_readers(pod5_path: str) -> dict:
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


def load_moves_file(path: str) -> dict:
    peaks = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts    = line.split('\t')
            read_id  = parts[0]
            mv_str   = parts[1]
            ts_str   = parts[2]
            mv_parts = mv_str.split(',')
            stride   = int(mv_parts[1])
            moves    = [int(x) for x in mv_parts[2:]]
            ts_offset = int(ts_str.split(':')[2])
            peak_positions = [ts_offset + i * stride
                              for i, m in enumerate(moves) if m == 1]
            peak_positions.append(ts_offset + len(moves) * stride)
            peaks[read_id] = np.array(peak_positions, dtype=np.int64)
    return peaks


def load_peaks_file(path: str) -> dict:
    peaks = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            peaks[parts[0]] = np.array([int(p) for p in parts[1:]], dtype=np.int64)
    return peaks


def load_gt(path: str) -> set:
    gt_set = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                gt_set.add((parts[0], int(parts[1])))
    return gt_set


_COMP = str.maketrans('ACGT', 'TGCA')


def _revcomp(seq: str) -> str:
    return seq.translate(_COMP)[::-1]


# ── BAM helpers ───────────────────────────────────────────────────────────────

def get_ref_info_from_bam(bam_read):
    """Return (ref_seq, ref_positions) aligned to read direction (5'→3')."""
    pairs = bam_read.get_aligned_pairs(with_seq=True)
    ref_bases = {}
    for qpos, rpos, rbase in pairs:
        if rpos is not None and rbase is not None:
            ref_bases[rpos] = rbase.upper()
    if not ref_bases:
        return None, None

    min_rpos = min(ref_bases.keys())
    max_rpos = max(ref_bases.keys())
    ref_seq  = ''.join(ref_bases.get(p, 'N') for p in range(min_rpos, max_rpos + 1))
    ref_pos  = list(range(min_rpos, max_rpos + 1))

    if bam_read.is_reverse:
        ref_seq = _revcomp(ref_seq)
        ref_pos = ref_pos[::-1]

    return ref_seq, ref_pos


# ── kmer level table helpers ──────────────────────────────────────────────────

def load_level_table(path: str) -> dict:
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


def normalize_level_table(kmer_levels: dict) -> tuple:
    """Z-score normalize kmer expected levels across all kmers."""
    values = np.array(list(kmer_levels.values()), dtype=np.float64)
    mu  = float(values.mean())
    sig = float(values.std())
    if sig < 1e-10:
        raise ValueError("K-mer level table has near-zero variance; cannot normalize.")
    return {k: (v - mu) / sig for k, v in kmer_levels.items()}, mu, sig


def auto_detect_center_idx(kmer_size: int, sample_data: list,
                            kmer_levels: dict) -> tuple:
    """Try every center_idx; return the one with highest mean Pearson r."""
    best_center, best_r = kmer_size // 2, -2.0
    for candidate in range(kmer_size):
        rs = []
        for signal, peaks, ref_seq, ref_positions in sample_data:
            n_usable = min(len(peaks) - 1, len(ref_seq))
            obs, exp = [], []
            for i in range(n_usable):
                start, end = int(peaks[i]), int(peaks[i + 1])
                if start >= end or end > len(signal):
                    continue
                k_start = i - candidate
                k_end   = k_start + kmer_size
                if k_start < 0 or k_end > len(ref_seq):
                    continue
                kmer = ref_seq[k_start:k_end]
                if 'N' in kmer or kmer not in kmer_levels:
                    continue
                obs.append(float(np.mean(signal[start:end])))
                exp.append(kmer_levels[kmer])
            if len(obs) < 10:
                continue
            obs_a, exp_a = np.array(obs), np.array(exp)
            if np.std(obs_a) < 1e-10 or np.std(exp_a) < 1e-10:
                continue
            r = float(np.corrcoef(obs_a, exp_a)[0, 1])
            rs.append(r)
        if rs:
            mean_r = float(np.mean(rs))
            if mean_r > best_r:
                best_r, best_center = mean_r, candidate
    return best_center, best_r


# ── signal helpers ────────────────────────────────────────────────────────────

def zscore(signal: np.ndarray) -> np.ndarray:
    mu  = signal.mean()
    sig = signal.std()
    if sig < 1e-10:
        return signal - mu
    return (signal - mu) / sig


def mad_normalize(signal: np.ndarray) -> np.ndarray:
    """Median / MAD normalization — standard nanopore signal scaling."""
    med = float(np.median(signal))
    mad = float(np.median(np.abs(signal - med))) * 1.4826
    if mad < 1e-10:
        return signal - med
    return (signal - med) / mad


def resample_segment(samples: np.ndarray, L: int) -> np.ndarray:
    """Resample a variable-length segment to exactly L points via linear interp."""
    n = len(samples)
    if n == L:
        return samples.astype(np.float32)
    if n == 1:
        return np.full(L, samples[0], dtype=np.float32)
    x_old = np.linspace(0, 1, n)
    x_new = np.linspace(0, 1, L)
    return np.interp(x_new, x_old, samples).astype(np.float32)


# ── per-read, per-position segment extraction ─────────────────────────────────

def extract_base_segments(signal: np.ndarray,
                           peaks:  np.ndarray,
                           ref_seq: str,
                           ref_positions: list,
                           L: int,
                           mapq: int,
                           strand: int) -> dict:
    """
    For each reference position covered by this read, extract and resample
    the raw signal segment to L samples.

    Returns dict: ref_pos -> {
        'samples':      np.ndarray shape (L,)   raw resampled signal
        'dwell':        int                      raw sample count
        'base':         str                      called base at this position
        'mapq':         int                      mapping quality of the read
        'strand':       int                      +1 or -1
    }
    """
    n_segments = len(peaks) - 1
    n_bases    = len(ref_seq)
    n_usable   = min(n_segments, n_bases)

    result = {}
    for i in range(n_usable):
        start = int(peaks[i])
        end   = int(peaks[i + 1])
        if start >= end or start < 0 or end > len(signal) or (end - start) < 1:
            continue

        seg     = signal[start:end]
        dwell   = int(end - start)
        resampl = resample_segment(seg, L)

        rpos = ref_positions[i]
        base = ref_seq[i] if i < len(ref_seq) else 'N'
        result[rpos] = {
            'samples': resampl,
            'dwell':   dwell,
            'base':    base,
            'mapq':    mapq,
            'strand':  strand,
        }

    return result


# ── reference row builder ─────────────────────────────────────────────────────

def build_reference_row(center_pos: int,
                         half_window: int,
                         L: int,
                         ref_base_map: dict,
                         kmer_levels: dict,
                         kmer_size: int,
                         center_idx: int) -> np.ndarray:
    """
    Build the single reference row of shape (W*L, N_CHANNELS).

    Like DeepVariant's reference track:
      Ch 0 : expected kmer level for each base (flat across L columns)
              If no kmer level available, 0.
      Ch 1 : 0 (no dwell for reference)
      Ch 2-5: one-hot base identity
      Ch 6 : 0 (no strand for reference)
      Ch 7 : 0 (no MAPQ for reference)
      Ch 8 : 0 (matches_ref not applicable for reference row)

    ref_base_map : dict of ref_pos -> base char (all positions in window)
    kmer_levels  : dict of kmer_str -> float (may be None)
    """
    W   = 2 * half_window + 1
    row = np.zeros((W * L, N_CHANNELS), dtype=np.float32)

    for w_idx, rpos in enumerate(range(center_pos - half_window,
                                        center_pos + half_window + 1)):
        col_start = w_idx * L
        col_end   = col_start + L

        base = ref_base_map.get(rpos, 'N').upper()

        # Ch 0: expected kmer level (broadcast flat)
        exp_level = 0.0
        if kmer_levels is not None and base != 'N':
            # We need the full kmer context; look it up from ref_base_map
            k_start_pos = rpos - center_idx
            kmer = ''.join(
                ref_base_map.get(k_start_pos + j, 'N').upper()
                for j in range(kmer_size)
            )
            if 'N' not in kmer and kmer in kmer_levels:
                exp_level = float(kmer_levels[kmer])
        row[col_start:col_end, 0] = np.float32(exp_level)

        # Ch 2-5: one-hot base identity
        oh = BASE_ONEHOT.get(base, (0, 0, 0, 0))
        row[col_start:col_end, 2] = oh[0]   # is_A
        row[col_start:col_end, 3] = oh[1]   # is_C
        row[col_start:col_end, 4] = oh[2]   # is_G
        row[col_start:col_end, 5] = oh[3]   # is_T

        # Channels 1, 6, 7, 8 remain 0 for reference row

    return row


# ── pileup builder ────────────────────────────────────────────────────────────

def build_pileup_tensor(read_data:    list,
                         ref_row:      np.ndarray,
                         center_pos:   int,
                         half_window:  int,
                         L:            int,
                         max_reads:    int) -> np.ndarray:
    """
    Build the (max_reads+1, W*L, N_CHANNELS) pileup tensor for one image.

    Row 0 is the reference row (pre-built).
    Rows 1..max_reads are individual reads (zero-padded if fewer reads).

    read_data : list of read records (already the exact subset for this image)
                record['segments'] maps ref_pos -> {
                    'samples', 'dwell', 'base', 'mapq', 'strand'
                }

    Callers are responsible for subsetting read_data to at most max_reads
    entries before calling this function.  No further subsampling is done here.

    Channels per read row:
      0  raw signal (L values)
      1  dwell: log1p(dwell), broadcast
      2  is_A, broadcast
      3  is_C, broadcast
      4  is_G, broadcast
      5  is_T, broadcast
      6  strand (+1/-1), broadcast
      7  MAPQ / 60, clipped [0,1], broadcast
      8  matches_ref (1 if read base == ref base), broadcast
    """
    W      = 2 * half_window + 1
    height = max_reads + 1  # +1 for reference row

    tensor = np.zeros((height, W * L, N_CHANNELS), dtype=np.float32)

    # Row 0: reference
    tensor[0] = ref_row

    # Rows 1..: reads
    for r_idx, read_record in enumerate(read_data):
        seg_dict = read_record.get('segments', read_record)
        row_idx = r_idx + 1   # offset by 1 for reference row
        for w_idx, rpos in enumerate(range(center_pos - half_window,
                                           center_pos + half_window + 1)):
            col_start = w_idx * L
            col_end   = col_start + L

            if rpos not in seg_dict:
                continue   # missing coverage → zero columns

            entry     = seg_dict[rpos]
            read_base = entry['base'].upper()

            # Ch 0: raw resampled signal
            tensor[row_idx, col_start:col_end, 0] = entry['samples']

            # Ch 1: dwell (log1p normalized, broadcast)
            tensor[row_idx, col_start:col_end, 1] = float(np.log1p(entry['dwell']))

            # Ch 2-5: one-hot base identity
            oh = BASE_ONEHOT.get(read_base, (0, 0, 0, 0))
            tensor[row_idx, col_start:col_end, 2] = oh[0]   # is_A
            tensor[row_idx, col_start:col_end, 3] = oh[1]   # is_C
            tensor[row_idx, col_start:col_end, 4] = oh[2]   # is_G
            tensor[row_idx, col_start:col_end, 5] = oh[3]   # is_T

            # Ch 6: strand
            tensor[row_idx, col_start:col_end, 6] = float(entry['strand'])

            # Ch 7: MAPQ normalized to [0, 1] (clip at 60)
            tensor[row_idx, col_start:col_end, 7] = float(
                min(entry['mapq'], 60) / 60.0)

            # Ch 8: matches reference (compare to ref row one-hot)
            ref_oh_start = col_start
            ref_base_oh  = tensor[0, ref_oh_start, 2:6]   # [is_A, is_C, is_G, is_T]
            read_oh      = np.array(oh, dtype=np.float32)
            matches      = float(np.array_equal(ref_base_oh, read_oh) and read_base != 'N')
            tensor[row_idx, col_start:col_end, 8] = matches

    return tensor


# ── read ordering and multi-image partitioner ─────────────────────────────────

def get_haplotype_index(bam_read) -> int:
    """Return the DeepVariant-style haplotype sort key for a read."""
    if not bam_read.has_tag('HP'):
        return 0
    try:
        hp = int(bam_read.get_tag('HP'))
    except Exception:
        return 0
    return hp if hp > 0 else 0


def make_read_record(seg_dict: dict, bam_read) -> dict:
    """Bundle per-base segments with a stable DeepVariant-style sort key."""
    ref_start = bam_read.reference_start
    if ref_start is None or ref_start < 0:
        ref_start = 0
    return {
        'segments': seg_dict,
        'sort_key': (
            get_haplotype_index(bam_read),
            int(ref_start),
            bam_read.query_name,
        ),
    }


def sort_reads_for_pileup(read_list: list) -> list:
    """Sort read rows by haplotype, then alignment start."""
    return sorted(
        read_list,
        key=lambda r: r.get('sort_key', (0, 0, '')) if isinstance(r, dict)
        else (0, 0, ''),
    )

def partition_reads(read_list: list,
                    max_reads: int,
                    rng: np.random.Generator,
                    max_images: int = None) -> list[list]:
    """
    Split reads into chunks and order rows DeepVariant-style.

    For n > max_reads, chunk membership is sampled by shuffling indices,
    matching DeepVariant's random overflow read selection behavior. Rows within
    each emitted chunk are then sorted by haplotype and alignment start before
    stacking read rows. If max_images is set, only that many chunks are emitted.

    If n < max_reads, returns a single chunk containing all reads
    (length < max_reads; the tensor builder zero-pads the rest).
    """
    n = len(read_list)
    if n <= max_reads:
        # Only one image possible; return all reads as a single chunk.
        return [sort_reads_for_pileup(list(read_list))]

    indices = np.arange(n)
    rng.shuffle(indices)

    n_images = n // max_reads
    if max_images is not None:
        n_images = min(n_images, max_images)

    chunks = []
    for i in range(n_images):
        chunk_idx = indices[i * max_reads:(i + 1) * max_reads]
        chunk = [read_list[j] for j in chunk_idx]
        chunks.append(sort_reads_for_pileup(chunk))
    return chunks


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Build DeepVariant-style pileup tensors for nanopore '
                    'modification detection.')
    parser.add_argument('--pod5',        required=True,  help='Pod5 file or directory')
    parser.add_argument('--bam',         required=True,  help='Aligned BAM file')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--peaks',  help='Peaks file (segmentation boundaries)')
    group.add_argument('--moves',  help='Moves TSV file')
    parser.add_argument('--output',      required=True,  help='Output HDF5 file')
    parser.add_argument(
        '--gt',
        nargs='?',
        const='__EMPTY__',
        default=None,
        metavar='BED',
        help='Path to a ground-truth BED file (columns: ref_name, ref_pos). '
             'Supply --gt alone (no path) for all-False labels (unmodified control).'
    )
    parser.add_argument('--normalize', action='store_true',
                        help='Normalize each read\'s raw signal using MAD scaling. '
                             'Recommended when combining reads from different '
                             'flow cells or pore generations.')
    parser.add_argument('--half-window', type=int, default=10,
                        help='Number of bases on each side of candidate position '
                             '(default: 10)')
    parser.add_argument('--L',           type=int, default=10,
                        help='Resampled signal length per base (default: 10)')
    parser.add_argument('--max-reads',   type=int, default=30,
                        help='Reads per image — each image uses exactly this many '
                             'reads (zero-padded if fewer). Positions with more '
                             'reads produce multiple images: floor(n/max_reads). '
                             'Tensor height = max_reads + 1 (ref row). (default: 30)')
    parser.add_argument('--max-images-per-base', type=int, default=None,
                        help='Maximum number of images to emit for each reference '
                             'base after read partitioning. Use this to stop '
                             'high-coverage bases from dominating training. '
                             '(default: uncapped)')
    parser.add_argument('--min-reads',   type=int, default=5,
                        help='Minimum reads to emit a position (default: 5)')
    parser.add_argument('--min-mapq',    type=int, default=60,
                        help='Minimum MAPQ filter (default: 60)')
    parser.add_argument('--target-base', default=None,
                        help='Only featurize positions where ref_base matches '
                             'this character, e.g. A for m6A (default: all bases)')
    parser.add_argument('--level-table', default=None,
                        help='K-mer level table (TSV: kmer\\tmean). When supplied, '
                             'the reference row (row 0) encodes expected kmer levels '
                             'in channel 0. Read rows always use raw resampled signal.')
    parser.add_argument('--center-idx', type=int, default=None,
                        help='K-mer center index for level-table lookup '
                             '(auto-detected from 50 reads if not set).')
    parser.add_argument('--seed',        type=int, default=42,
                        help='RNG seed for read subsampling (default: 42)')
    args = parser.parse_args()

    if args.max_images_per_base is not None and args.max_images_per_base < 1:
        parser.error('--max-images-per-base must be >= 1 when set')

    rng = np.random.default_rng(args.seed)

    # ── load inputs ──────────────────────────────────────────────────────────
    print(f"Loading pod5: {args.pod5}", file=sys.stderr)
    read_reader_map = get_pod5_readers(args.pod5)
    print(f"  {len(read_reader_map):,} reads in pod5", file=sys.stderr)

    if args.peaks:
        print(f"Loading peaks: {args.peaks}", file=sys.stderr)
        seg_borders = load_peaks_file(args.peaks)
    else:
        print(f"Loading moves: {args.moves}", file=sys.stderr)
        seg_borders = load_moves_file(args.moves)
    print(f"  {len(seg_borders):,} segmented reads", file=sys.stderr)

    gt_set = None
    if args.gt is None:
        print("No --gt supplied; all labels will be 0 (inference mode).",
              file=sys.stderr)
    elif args.gt == '__EMPTY__':
        gt_set = set()
        print("--gt supplied without a file; all labels set to 0 (unmodified control).",
              file=sys.stderr)
    else:
        gt_set = load_gt(args.gt)
        print(f"Ground truth: {len(gt_set):,} modified positions from {args.gt}",
              file=sys.stderr)

    bam_fh = pysam.AlignmentFile(args.bam, 'rb', check_sq=False)

    # ── kmer level table (optional) ───────────────────────────────────────────
    kmer_levels = None
    kmer_size   = None
    center_idx  = None

    if args.level_table:
        print(f"Loading level table: {args.level_table}", file=sys.stderr)
        kmer_levels_raw = load_level_table(args.level_table)
        kmer_size = len(next(iter(kmer_levels_raw)))
        print(f"  {len(kmer_levels_raw):,} k-mers, k={kmer_size}", file=sys.stderr)

        if args.normalize:
            kmer_levels, lmu, lsig = normalize_level_table(kmer_levels_raw)
            print(f"  Level table z-score normalized "
                  f"(mu={lmu:.4f}, sigma={lsig:.4f})", file=sys.stderr)
        else:
            kmer_levels = kmer_levels_raw

        if args.center_idx is not None:
            center_idx = args.center_idx
            print(f"  center_idx={center_idx} (user-supplied)", file=sys.stderr)
        else:
            print(f"  Auto-detecting center_idx over {kmer_size} candidates "
                  f"on up to 50 reads ...", file=sys.stderr)
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
                    raw    = pod5_read.signal.astype(np.float64)
                    sig    = mad_normalize(raw) if args.normalize else zscore(raw)
                    peaks  = seg_borders[read_id]
                    ref_seq, ref_positions = get_ref_info_from_bam(bam_read)
                    if ref_seq is not None:
                        sample_data.append((sig, peaks, ref_seq, ref_positions))
                except Exception:
                    continue
                if len(sample_data) >= 50:
                    break

            center_idx, best_r = auto_detect_center_idx(
                kmer_size, sample_data, kmer_levels)
            print(f"  center_idx={center_idx}  "
                  f"(mean r={best_r:.4f} on {len(sample_data)} reads)",
                  file=sys.stderr)

            bam_fh.close()
            bam_fh = pysam.AlignmentFile(args.bam, 'rb', check_sq=False)
    else:
        print("No --level-table supplied; reference row channel 0 will be 0.",
              file=sys.stderr)

    # ── pass 1: collect per-read segment dicts per reference position ─────────
    # pos_reads[(ref_name, ref_pos)]    = list of seg_dicts
    # pos_ref_context[(ref_name)]       = dict of ref_pos -> base (all covered pos)
    pos_reads      = collections.defaultdict(list)
    pos_refbase    = {}    # (ref_name, ref_pos) -> base char
    pos_ref_context = collections.defaultdict(dict)  # ref_name -> {ref_pos: base}

    n_total = n_eval = n_skip = 0

    for bam_read in tqdm(bam_fh, desc="Processing reads", file=sys.stderr):
        n_total += 1
        if (bam_read.is_supplementary or bam_read.is_secondary
                or bam_read.is_unmapped):
            n_skip += 1
            continue
        if bam_read.mapping_quality < args.min_mapq:
            n_skip += 1
            continue

        read_id = bam_read.query_name
        if read_id not in seg_borders or read_id not in read_reader_map:
            n_skip += 1
            continue

        strand = -1 if bam_read.is_reverse else +1
        mapq   = bam_read.mapping_quality
        ref_name = bam_read.reference_name

        try:
            pod5_read = next(read_reader_map[read_id].reads(selection=[read_id]))
            raw       = pod5_read.signal.astype(np.float64)
            signal    = mad_normalize(raw) if args.normalize else zscore(raw)
            peaks     = seg_borders[read_id]
            ref_seq, ref_positions = get_ref_info_from_bam(bam_read)
            if ref_seq is None:
                n_skip += 1
                continue

            seg_dict = extract_base_segments(
                signal, peaks, ref_seq, ref_positions, args.L, mapq, strand)
            read_record = make_read_record(seg_dict, bam_read)

            for rpos, entry in seg_dict.items():
                key = (ref_name, rpos)
                pos_reads[key].append(read_record)
                if key not in pos_refbase:
                    pos_refbase[key] = entry['base']
                # Accumulate reference base context for reference row construction
                pos_ref_context[ref_name][rpos] = entry['base']

            n_eval += 1

        except Exception as e:
            n_skip += 1
            if n_skip <= 5:
                print(f"  Warning: skipped {read_id}: {e}", file=sys.stderr)

    bam_fh.close()
    print(f"\nReads evaluated: {n_eval:,}  skipped: {n_skip:,}", file=sys.stderr)

    # ── pass 2: build tensors for eligible positions ──────────────────────────
    eligible = {k: v for k, v in pos_reads.items() if len(v) >= args.min_reads}
    if args.target_base:
        tb = args.target_base.upper()
        eligible = {k: v for k, v in eligible.items()
                    if pos_refbase.get(k, 'N').upper() == tb}

    print(f"Eligible positions: {len(eligible):,}  "
          f"(>= {args.min_reads} reads"
          + (f", ref_base={args.target_base}" if args.target_base else "")
          + ")", file=sys.stderr)

    if not eligible:
        print("No eligible positions found — nothing to write.", file=sys.stderr)
        sys.exit(1)

    # Pre-compute image partitions for every position so we know total N_images
    # before allocating the output arrays.
    sorted_keys = sorted(eligible.keys(), key=lambda k: (k[0], k[1]))

    position_chunks: list[tuple] = []   # (key, chunk_list, image_idx)
    for key in sorted_keys:
        read_list = eligible[key]
        chunks = partition_reads(
            read_list, args.max_reads, rng,
            max_images=args.max_images_per_base,
        )
        for img_idx, chunk in enumerate(chunks):
            position_chunks.append((key, chunk, img_idx))

    n_images = len(position_chunks)
    n_pos    = len(sorted_keys)

    W          = 2 * args.half_window + 1
    height     = args.max_reads + 1   # +1 for reference row
    tensor_dim = (n_images, height, W * args.L, N_CHANNELS)

    print(f"\nPositions: {n_pos:,}  →  Images (rows): {n_images:,}  "
          f"(avg {n_images / n_pos:.1f} images/position)", file=sys.stderr)
    if args.max_images_per_base is not None:
        print(f"Images per position capped at: {args.max_images_per_base}",
              file=sys.stderr)
    print(f"Tensor shape per image: ({height}, {W * args.L}, {N_CHANNELS})",
          file=sys.stderr)
    print(f"  (row 0 = reference, rows 1-{height-1} = reads)", file=sys.stderr)
    print(f"Full dataset shape: {tensor_dim}", file=sys.stderr)
    print(f"Estimated memory (float32): "
          f"{np.prod(tensor_dim) * 4 / 1024**3:.2f} GB", file=sys.stderr)

    tensors    = np.zeros(tensor_dim, dtype=np.float32)
    labels_arr = np.zeros(n_images, dtype=np.int8)
    ref_names  = []
    ref_poss   = np.zeros(n_images, dtype=np.int64)
    n_reads_v  = np.zeros(n_images, dtype=np.int16)
    image_idxs = np.zeros(n_images, dtype=np.int16)

    # Cache reference rows — they are identical across all images of the same
    # position, so build once per position.
    ref_row_cache: dict = {}

    for i, (key, chunk, img_idx) in enumerate(
            tqdm(position_chunks, desc="Building tensors", file=sys.stderr)):
        ref_name, ref_pos = key

        # Build (or reuse) the reference row for this position.
        if key not in ref_row_cache:
            ref_base_map_local = pos_ref_context.get(ref_name, {})
            ref_row_cache[key] = build_reference_row(
                center_pos=ref_pos,
                half_window=args.half_window,
                L=args.L,
                ref_base_map=ref_base_map_local,
                kmer_levels=kmer_levels,
                kmer_size=kmer_size,
                center_idx=center_idx,
            )
        ref_row = ref_row_cache[key]

        tensors[i] = build_pileup_tensor(
            chunk, ref_row, ref_pos, args.half_window,
            args.L, args.max_reads
        )
        labels_arr[i]  = 1 if (gt_set and key in gt_set) else 0
        ref_names.append(ref_name.encode('utf-8'))
        ref_poss[i]    = ref_pos
        n_reads_v[i]   = min(len(chunk), 32767)
        image_idxs[i]  = min(img_idx, 32767)

    # Free the cache — can be large for high-coverage datasets.
    del ref_row_cache

    # ── write HDF5 ────────────────────────────────────────────────────────────
    print(f"\nWriting HDF5 to {args.output} ...", file=sys.stderr)
    with h5py.File(args.output, 'w') as hf:
        hf.create_dataset('tensors',   data=tensors,              compression='gzip',
                          compression_opts=4,
                          chunks=(min(64, n_images), height, W * args.L,
                                  N_CHANNELS))
        hf.create_dataset('labels',    data=labels_arr)
        hf.create_dataset('ref_names',
                          data=np.array(ref_names,
                                        dtype=h5py.special_dtype(vlen=bytes)))
        hf.create_dataset('ref_pos',   data=ref_poss)
        hf.create_dataset('n_reads',   data=n_reads_v)
        hf.create_dataset('image_idx', data=image_idxs,
                          compression='gzip', compression_opts=1)

        hf.attrs['half_window']   = args.half_window
        hf.attrs['L']             = args.L
        hf.attrs['max_reads']     = args.max_reads
        hf.attrs['W']             = W
        hf.attrs['n_channels']    = N_CHANNELS
        hf.attrs['height']        = height
        hf.attrs['n_positions']   = n_pos
        hf.attrs['n_images']      = n_images
        hf.attrs['max_images_per_base'] = (
            args.max_images_per_base if args.max_images_per_base is not None else 0
        )
        hf.attrs['channel_names'] = [
            'raw_signal', 'dwell_log1p',
            'is_A', 'is_C', 'is_G', 'is_T',
            'strand', 'mapq_norm', 'matches_ref',
        ]
        hf.attrs['normalization'] = 'MAD' if args.normalize else 'zscore'
        hf.attrs['ref_row']       = 'row_0'
        hf.attrs['label_semantics'] = 'binary_modified_vs_unmodified'
        hf.attrs['read_order']    = 'haplotype_then_alignment_start'
        hf.attrs['multi_image']   = True
        hf.attrs['partition']     = 'random_nonoverlapping_chunks_sorted_by_read_order'
        if args.level_table:
            hf.attrs['level_table'] = args.level_table
            hf.attrs['center_idx']  = center_idx
            hf.attrs['kmer_size']   = kmer_size

    n_mod   = int(labels_arr.sum())
    n_unmod = len(labels_arr) - n_mod
    print(f"\n=== Summary ===", file=sys.stderr)
    print(f"Positions written   : {n_pos:,}", file=sys.stderr)
    print(f"Images written      : {n_images:,}  "
          f"(avg {n_images / n_pos:.1f}/position)", file=sys.stderr)
    print(f"Modified (label=1)  : {n_mod:,}  images", file=sys.stderr)
    print(f"Unmodified (label=0): {n_unmod:,}  images", file=sys.stderr)
    print(f"Output HDF5         : {args.output}", file=sys.stderr)


if __name__ == '__main__':
    main()
