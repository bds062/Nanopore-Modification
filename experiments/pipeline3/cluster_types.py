#!/usr/bin/env python3
"""
Unsupervised modification-TYPE clustering — the novel contribution (step 2).

The literature is explicit that unsupervised detectors "detect only the modification
site and not the modification type" (Modena, NAR 2025), and that supervised typing
"precludes the discovery of novel modification types beyond their scope". This script
tests whether RawMod's representation separates modification chemistry with NO type
labels supplied.

Two feature spaces are clustered independently and then compared:

  A. SIGNAL SIGNATURE  — the per-site current profile over the pileup window, taken
     straight from the tensor (channel 0, reference row subtracted). This is the
     nanodisco-style feature ([-6,+7] bp current difference) and structurally cannot
     encode dataset provenance beyond what the chemistry implies.
  B. MODEL EMBEDDING   — RawMod's penultimate representation (norm(pooled)), captured
     by score_genome.py --save-embeddings.

THE CONFOUND, HANDLED EXPLICITLY:
4mC occurs only in H. pylori and 5hmU only in SPO1, so for those two, "type" and
"organism" are perfectly confounded — clusters could be tracking provenance, which is
exactly the dataset-identity shortcut we know this model has (lodo_WGA FPR 50.2%).
We therefore report BOTH:
    ARI(clusters, modification_type)   and   ARI(clusters, source_dataset)
and additionally restrict to 6mA and 5mC, which DO appear in multiple organisms
(6mA: E. coli / Anabaena / T. denticola; 5mC: E. coli / arabidopsis). If clusters track
type rather than organism on that unconfounded subset, the representation is encoding
chemistry. That subset is the honest test; the full four-type plot is the illustration.

Usage:
  cluster_types.py --spec spec.tsv --out-dir clusters/ [--max-per-group 4000]
where spec.tsv has columns:  dataset  h5_path  mod_type  [embed_npy]
"""
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'deepmod'))

from sklearn.mixture import GaussianMixture           # noqa: E402
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score  # noqa: E402
from sklearn.decomposition import PCA                 # noqa: E402
from sklearn.preprocessing import StandardScaler      # noqa: E402


def signal_signature(h5_path, idx, half_window=None, L=None):
    """Feature space A: current profile across the window, reference row removed.

    tensor[0] is the reference (expected k-mer level) row and tensor[1:] are reads, so
    mean(reads) - reference is a per-site current-difference profile directly analogous
    to nanodisco's native-minus-WGA signature.
    """
    with h5py.File(h5_path, 'r') as h:
        a = dict(h.attrs)
        L = L or int(a['L']); W = int(a['W'])
        nr = h['n_reads'][:]
        idx = np.asarray(sorted(idx), dtype=np.int64)
        blk = h['tensors'][idx.tolist()].astype(np.float32)
    out = np.zeros((len(idx), W), dtype=np.float32)
    for i in range(len(idx)):
        k = max(int(nr[idx[i]]), 1)
        reads = blk[i, 1:1 + k, :, 0]                  # (k, W*L)
        ref = blk[i, 0, :, 0]                          # (W*L,)
        diff = np.nanmean(reads, axis=0) - ref
        out[i] = np.nanmean(diff.reshape(W, L), axis=1)  # per-window-position mean
    return np.nan_to_num(out)


def load_spec(path):
    rows = []
    with open(path) as fh:
        header = fh.readline().rstrip('\n').split('\t')
        for line in fh:
            if not line.strip() or line.startswith('#'):
                continue
            c = line.rstrip('\n').split('\t')
            rows.append(dict(zip(header, c)))
    return rows


def gmm_cluster(X, n_components, seed=0):
    Xs = StandardScaler().fit_transform(X)
    if Xs.shape[1] > 32:
        Xs = PCA(n_components=32, random_state=seed).fit_transform(Xs)
    gm = GaussianMixture(n_components=n_components, covariance_type='full',
                         random_state=seed, n_init=3)
    return gm.fit_predict(Xs), gm


def bic_sweep(X, kmax=8, seed=0):
    """Unsupervised model selection: how many distinct states does the data support?
    Mirrors the GMM component-number approach (Bioinformatics 36(19):4928)."""
    Xs = StandardScaler().fit_transform(X)
    if Xs.shape[1] > 32:
        Xs = PCA(n_components=32, random_state=seed).fit_transform(Xs)
    out = []
    for k in range(1, kmax + 1):
        gm = GaussianMixture(k, covariance_type='full', random_state=seed, n_init=2)
        gm.fit(Xs)
        out.append((k, float(gm.bic(Xs))))
    return out


def evaluate(labels_pred, mod_type, dataset, tag, fh):
    ari_t = adjusted_rand_score(mod_type, labels_pred)
    ari_d = adjusted_rand_score(dataset, labels_pred)
    nmi_t = normalized_mutual_info_score(mod_type, labels_pred)
    nmi_d = normalized_mutual_info_score(dataset, labels_pred)
    msg = (f"{tag}\n"
           f"  ARI vs modification type : {ari_t:.3f}   NMI {nmi_t:.3f}\n"
           f"  ARI vs source dataset    : {ari_d:.3f}   NMI {nmi_d:.3f}\n"
           f"  -> clusters track {'TYPE' if ari_t > ari_d else 'DATASET'} "
           f"(margin {ari_t - ari_d:+.3f})\n")
    print(msg, end='', flush=True)
    fh.write(msg)
    return {'tag': tag, 'ari_type': ari_t, 'ari_dataset': ari_d,
            'nmi_type': nmi_t, 'nmi_dataset': nmi_d}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--max-per-group', type=int, default=4000)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    rows = load_spec(a.spec)
    sigs, embs, mtype, dset = [], [], [], []
    have_embed = True
    for r in rows:
        h5 = r['h5_path']; mt = r['mod_type']; ds = r['dataset']
        with h5py.File(h5, 'r') as h:
            lab = h['labels'][:]
        pool = np.nonzero(lab > 0)[0] if mt != 'unmod' else np.nonzero(lab == 0)[0]
        if len(pool) == 0:
            print(f"  [skip] {ds}/{mt}: no sites", file=sys.stderr); continue
        take = pool if len(pool) <= a.max_per_group else rng.choice(
            pool, a.max_per_group, replace=False)
        take = np.sort(take)
        sigs.append(signal_signature(h5, take))
        mtype += [mt] * len(take); dset += [ds] * len(take)
        ep = r.get('embed_npy', '')
        if ep and Path(ep).exists():
            e = np.load(ep)
            embs.append(e[take].astype(np.float32))
        else:
            have_embed = False
        print(f"  {ds:22} {mt:6} n={len(take):,}", flush=True)

    X_sig = np.vstack(sigs)
    mtype = np.array(mtype); dset = np.array(dset)
    n_types = len(set(mtype))
    print(f"\npooled: {X_sig.shape[0]:,} sites, {n_types} modification types, "
          f"{len(set(dset))} datasets", flush=True)

    report = out / 'clustering_report.txt'
    results = []
    with open(report, 'w') as fh:
        fh.write("Unsupervised modification-type clustering\n"
                 "(no type labels supplied to the clustering; used only to score it)\n\n")

        # --- how many states does the data support, unsupervised? ---
        sweep = bic_sweep(X_sig, kmax=min(8, max(2, n_types + 3)), seed=a.seed)
        best_k = min(sweep, key=lambda t: t[1])[0]
        fh.write(f"BIC sweep (signal signature): {sweep}\n"
                 f"BIC-selected components: {best_k}  (true #types = {n_types})\n\n")
        print(f"BIC-selected components: {best_k} (true types {n_types})", flush=True)

        # --- feature space A ---
        lab_sig, _ = gmm_cluster(X_sig, n_types, a.seed)
        results.append(evaluate(lab_sig, mtype, dset, "A. SIGNAL SIGNATURE (all types)", fh))

        # --- feature space B ---
        lab_emb = None
        if have_embed and embs:
            X_emb = np.vstack(embs)
            lab_emb, _ = gmm_cluster(X_emb, n_types, a.seed)
            results.append(evaluate(lab_emb, mtype, dset,
                                    "B. MODEL EMBEDDING (all types)", fh))
            agree = adjusted_rand_score(lab_sig, lab_emb)
            fh.write(f"\nAgreement between feature spaces (ARI): {agree:.3f}\n")
            print(f"\nfeature-space agreement ARI: {agree:.3f}", flush=True)
        else:
            fh.write("\n(model embeddings unavailable — run score_genome.py "
                     "--save-embeddings)\n")

        # --- the honest, unconfounded test ---
        # 6mA and 5mC each occur in >1 organism; 4mC (H. pylori) and 5hmU (SPO1) do not.
        multi = [t for t in set(mtype)
                 if len(set(dset[mtype == t])) > 1 and t != 'unmod']
        fh.write(f"\nUNCONFOUNDED SUBSET (types present in >1 organism): {multi}\n")
        if len(multi) >= 2:
            m = np.isin(mtype, multi)
            lab_m, _ = gmm_cluster(X_sig[m], len(multi), a.seed)
            results.append(evaluate(lab_m, mtype[m], dset[m],
                                    f"C. SIGNAL SIGNATURE, unconfounded {multi}", fh))
            if lab_emb is not None:
                lab_me, _ = gmm_cluster(np.vstack(embs)[m], len(multi), a.seed)
                results.append(evaluate(lab_me, mtype[m], dset[m],
                                        f"D. MODEL EMBEDDING, unconfounded {multi}", fh))
        else:
            fh.write("  not enough multi-organism types for the unconfounded test\n")
            print("  [warn] unconfounded test skipped: need >=2 types in >1 organism",
                  file=sys.stderr)

    np.savez_compressed(out / 'clustering_data.npz',
                        X_sig=X_sig, mod_type=mtype, dataset=dset,
                        labels_sig=lab_sig,
                        labels_emb=lab_emb if lab_emb is not None else np.array([]),
                        X_emb=np.vstack(embs) if (have_embed and embs) else np.array([]))
    with open(out / 'clustering_metrics.tsv', 'w') as fh:
        fh.write("tag\tari_type\tari_dataset\tnmi_type\tnmi_dataset\n")
        for r in results:
            fh.write(f"{r['tag']}\t{r['ari_type']:.4f}\t{r['ari_dataset']:.4f}\t"
                     f"{r['nmi_type']:.4f}\t{r['nmi_dataset']:.4f}\n")
    print(f"\nwrote {report}\nwrote {out/'clustering_metrics.tsv'}")


if __name__ == '__main__':
    main()
