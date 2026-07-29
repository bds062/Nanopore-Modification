#!/usr/bin/env python3
"""
Score every site in a genome-wide features.h5 with a RawMod checkpoint, and emit a
per-site score table for downstream de novo motif discovery / clustering.

NON-CIRCULARITY (the point of this script):
For a de novo claim the scoring model must never have seen the organism it is scoring.
`--checkpoint` therefore defaults to the *leave-one-dataset-out* fold matching the
dataset, not the mixed model. Mapping (see CHECKPOINT_FOR):
    Ecoli_DM_5kHz        -> lodo_Ecoli_DM_5kHz
    UMCES / SPO1         -> lodo_UMCES
    HP26695_WT, HPJ99,
    Anabaena, Tdenticola -> mixed        (test-only organisms: never in ANY training
                                          pool, so the mixed model is already
                                          non-circular for them)
Note HP26695_WGA *is* in the training pool, so H. pylori 26695 WT is scored with the
mixed model only because the WT sample itself is held out; the WGA control of the same
organism was trained on. lodo_HP26695_WGA_5kHz is available via --checkpoint if the
stricter reading is wanted, and `--both` scores with both for comparison.

Outputs `<out>/<dataset>_scores.tsv.gz`:  contig, pos, score, label, n_reads
"""
import argparse
import gzip
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

# Resolve imports against THIS repo, not the scratch working copies. Only data and
# trained checkpoints live on scratch; all code comes from the repo.
REPO = Path(__file__).resolve().parents[2]          # .../Nanopore-Modification
for _p in (REPO / 'experiments' / 'pipeline2',
           REPO / 'experiments' / 'pipeline1',
           REPO / 'deepmod'):
    sys.path.insert(0, str(_p))

# Trained weights (data artefact, not code) — overridable for portability.
MODELS = Path(os.environ.get(
    'RAWMOD_MODELS',
    '/fs/cbcb-scratch/bds062/results/deepmod_full_pipeline2/results2/models'))

CHECKPOINT_FOR = {
    'Ecoli_DM_5kHz':        MODELS / 'lodo_Ecoli_DM_5kHz/lodo/best_model.pt',
    'Ecoli_DM_MSssI_5kHz':  MODELS / 'lodo_Ecoli_DM_MSssI_5kHz/lodo/best_model.pt',
    'Ecoli_WT_5kHz':        MODELS / 'lodo_Ecoli_WT_5kHz/lodo/best_model.pt',
    'arabidopsis':          MODELS / 'lodo_arabidopsis/lodo/best_model.pt',
    'UMCES':                MODELS / 'lodo_UMCES/lodo/best_model.pt',
    'ONT':                  MODELS / 'lodo_ONT/lodo/best_model.pt',
    'HP26695_WGA_5kHz':     MODELS / 'lodo_HP26695_WGA_5kHz/lodo/best_model.pt',
    # test-only organisms — never in any training pool
    'HP26695_WT_5kHz':      MODELS / 'mixed/base/best_model.pt',
    'HPJ99_WT_5kHz':        MODELS / 'mixed/base/best_model.pt',
    'Anabaena_WT_5kHz':     MODELS / 'mixed/base/best_model.pt',
    'Tdenticola_WT_5kHz':   MODELS / 'mixed/base/best_model.pt',
}


def load_model(ckpt, device):
    """Reuse the repo's canonical loader (experiments/pipeline1/test_model.py).

    Checkpoints are written by run_pipeline.py as
        {'model_state': ..., 'in_channels': 11, 'val_auprc': ..., 'epoch': ..., 'tag': ...}
    and test_model.load_model already handles that plus architecture auto-detection.

    Exception: a DANN checkpoint also carries the sequence-context adversary's
    weights ('adv.*'). The adversary is training-only (forward() returns the logit
    alone at eval), but the module must exist for a strict load, so rebuild it with
    dann_lambda>0 rather than dropping the keys silently.
    """
    raw = torch.load(str(ckpt), map_location=device, weights_only=False)
    sd = raw['model_state'] if isinstance(raw, dict) and 'model_state' in raw else raw
    # A DANN checkpoint carries 'adv.*'; a SupCon checkpoint carries 'proj.*'.
    # Both auxiliaries are training-only (forward() returns the logit alone at
    # eval), but the submodules must exist for a strict load, so rebuild them.
    has_adv = any(k.startswith('adv.') for k in sd)
    has_proj = any(k.startswith('proj.') for k in sd)
    has_sad = any(k.startswith('sad_head.') for k in sd)
    if has_adv or has_proj or has_sad:
        from run_convformer_v2 import ConvFormerV2
        # infer aux dims from the state dict
        proj_dim = int(sd['proj.net.2.weight'].shape[0]) if has_proj else 0
        sad_dim = int(sd['sad_head.weight'].shape[0]) if has_sad else 0
        model = ConvFormerV2(dropout=0.0,
                             dann_lambda=1.0 if has_adv else 0.0,  # value irrelevant at eval
                             supcon_dim=proj_dim, sad_dim=sad_dim)
        model.load_state_dict(sd)
        model.to(device).eval()
        aux = []
        if has_adv:
            aux.append('DANN adversary')
        if has_proj:
            aux.append(f'SupCon proj dim={proj_dim}')
        if has_sad:
            aux.append(f'DeepSAD dim={sad_dim}')
        print(f"  checkpoint: {' + '.join(aux)} present "
              f"epoch={raw.get('epoch')} val_auprc={raw.get('val_auprc')}", flush=True)
        return model, 'convformer_v2'

    from test_model import load_model as _load
    model, meta = _load(str(ckpt), device, arch='auto')
    print(f"  checkpoint: tag={meta.get('tag')} epoch={meta.get('epoch')} "
          f"val_auprc={meta.get('val_auprc')}", flush=True)
    model.eval()
    return model, meta['arch']


def score(h5_path, ckpt, device, batch=512, workers=8, want_embed=False,
          legacy_ch9=False):
    """Stream the whole file through the model. Returns (scores, embeds|None).

    legacy_ch9=True computes the ch9 delta at the k-mer centre (attrs['center_idx'])
    rather than the window centre, to MATCH models trained before the ch9 fix. Use
    it for pipeline1/results2 checkpoints; leave False for models trained after the
    fix (e.g. the pipeline3 ablation arms).
    """
    from model import PileupDataset
    with h5py.File(h5_path, 'r') as h:
        n = h['tensors'].shape[0]
    ds = PileupDataset([str(h5_path)], np.arange(n, dtype=np.int64), [n],
                       augment=False, seed=0, signal_noise_std=0.0,
                       delta_channels=True, preload=False,
                       legacy_ch9_center=legacy_ch9)
    from torch.utils.data import DataLoader
    from model import make_loader_kwargs, _worker_init_fn
    try:
        lk = make_loader_kwargs(batch, workers, device, _worker_init_fn)
        loader = DataLoader(ds, shuffle=False, **lk)
    except Exception:
        loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers)

    model, arch = load_model(ckpt, device)

    # Where the 512/96-dim penultimate embedding lives differs by architecture:
    #   - ConvFormerV2/ConvFormer: self.head has no internal pooling, so the
    #     INPUT to self.head (= self.norm(pooled)) is already the flat embedding.
    #   - PileupInceptionV3 ('inception'): self.head is
    #     Sequential(AdaptiveAvgPool2d, Flatten, Dropout, Linear) -- the pool
    #     is INSIDE head, so hooking head itself would capture the pre-pool
    #     (B, 512, 1, W) conv feature map, not an embedding. Hook head[3] (the
    #     final Linear) instead; its INPUT is the pooled+flattened+dropout
    #     (B, 512) embedding (see model.py's dual-output forward for SupCon,
    #     which taps the identical point via self.head[2](...)).
    cap = {}
    hook = None
    if want_embed:
        def _grab(_mod, inp, _out):
            cap['e'] = inp[0].detach()
        target = model.head[3] if arch == 'inception' else model.head
        hook = target.register_forward_hook(_grab)

    out = np.empty(n, dtype=np.float32)
    embeds = [] if want_embed else None
    i = 0
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device, non_blocking=True)
            logit = model(xb)
            if want_embed:
                embeds.append(cap['e'].float().cpu().numpy())
            p = torch.sigmoid(logit.squeeze(-1)).float().cpu().numpy()
            out[i:i + len(p)] = p
            i += len(p)
    if hook is not None:
        hook.remove()
    if want_embed and embeds:
        embeds = np.concatenate(embeds, 0)
    return out, embeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5', required=True)
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--checkpoint', default=None,
                    help='override; default is the non-circular fold for --dataset')
    ap.add_argument('--tag', default=None, help='suffix for the output file')
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--save-embeddings', action='store_true')
    ap.add_argument('--legacy-ch9', action='store_true',
                    help='compute ch9 at the k-mer centre, to match models trained '
                         'before the ch9 fix (pipeline1 / results2)')
    a = ap.parse_args()

    ckpt = Path(a.checkpoint) if a.checkpoint else CHECKPOINT_FOR.get(a.dataset)
    if ckpt is None or not Path(ckpt).exists():
        raise SystemExit(f"no checkpoint for {a.dataset!r} (looked at {ckpt})")
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"scoring {a.dataset}\n  h5   = {a.h5}\n  ckpt = {ckpt}\n  dev  = {device}",
          flush=True)

    if a.legacy_ch9:
        print("  ch9: LEGACY (k-mer centre) to match a pre-fix checkpoint", flush=True)
    s, emb = score(a.h5, ckpt, device, a.batch, a.workers, a.save_embeddings,
                   legacy_ch9=a.legacy_ch9)

    with h5py.File(a.h5, 'r') as h:
        contigs = [x.decode() if isinstance(x, bytes) else str(x)
                   for x in h['ref_names'][:]]
        pos = h['ref_pos'][:]
        lab = h['labels'][:]
        nrd = h['n_reads'][:] if 'n_reads' in h else np.zeros(len(pos), dtype=int)

    tag = f"_{a.tag}" if a.tag else ""
    fout = out / f"{a.dataset}{tag}_scores.tsv.gz"
    with gzip.open(fout, 'wt') as fh:
        fh.write("contig\tpos\tscore\tlabel\tn_reads\n")
        for c, p, sc, l, nr in zip(contigs, pos, s, lab, nrd):
            fh.write(f"{c}\t{int(p)}\t{sc:.6f}\t{int(l)}\t{int(nr)}\n")
    print(f"wrote {fout}  ({len(s):,} sites)  mean_score={s.mean():.4f}", flush=True)

    if emb is not None:
        np.save(out / f"{a.dataset}{tag}_embed.npy", emb.astype(np.float16))
        print(f"wrote embeddings {emb.shape}", flush=True)


if __name__ == '__main__':
    main()
