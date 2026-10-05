# Datasets of the RawMod benchmark

We assume your current directory is this directory ([`data`](./)). Each script downloads one dataset with the links we
used and places the files where the benchmark expects them (`<dataset>/pod5_files/`, `<dataset>/ref.fa`,
`<dataset>/ground_truth/`). The scripts need the [AWS CLI](https://aws.amazon.com/cli/) (no account is needed), `wget`,
`gunzip` and, optionally, `samtools` (to index the references). A file that is already present is not downloaded again.

```bash
bash download_d1_ecoli_k12.sh
bash download_d2_anabaena_pcc7120.sh
bash download_d3_hpylori_26695.sh
bash download_d4_athaliana.sh
bash download_d5_osativa.sh
bash download_d6_mouse_brain.sh
bash download_d7_human_hg002.sh
bash download_d8_ont_oligos.sh
bash download_d9_spo1.sh
bash download_d10_hpylori_j99.sh       # WITH_PACBIO=1 also downloads the PacBio reads (only to rebuild the _pb rows)
bash download_d11_tdenticola.sh        # WITH_PACBIO=1 also downloads the PacBio reads
```

| Dataset | Samples | Benchmark rows | Download | Source |
|---|---|---|---|---|
| D1 *E. coli* K-12 MG1655 | `ecoli_wt_w1m`, `ecoli_dm_w1m`, `ecoli_mssi_w1m` | `ecoli_dam_6mA_ko`, `ecoli_dcm_5mC_ko`, `ecoli_mssi_5mC_ko` | 16.5 GB | [1] |
| D2 *Anabaena* sp. PCC 7120 | `anabaena` | `anabaena_6mA` | 8.4 GB | [1] |
| D3 *H. pylori* 26695 | `hp26695`, `hp26695_wga` | `hp26695_4mC`, `hp26695_6mA` | 4.0 GB | [1] |
| D4 *A. thaliana* Col-0 | `arabidopsis_chr1` | `arabidopsis_cpg`, `arabidopsis_noncpg` | 31 GB | [1] |
| D5 *O. sativa* Nipponbare | `rice_w` | `rice_cpg` | 60 GB | [1] |
| D6 Mouse brain | `mouse` | `mouse_cpg` | 938 GB | [1] |
| D7 Human HG002 | `hg002` | `hg002_cpg` | 286 GB | [2], [3] |
| D8 ONT all-5-mer oligonucleotides | `syn_{5mC,5hmC,6mA,control}`, `..._rep2` | `syn_5mC`, `syn_5hmC`, `syn_6mA`; `syn_*_rep2` | 3 GB | [4] |
| D9 Bacillus phage SPO1 | `spo1_bc07` (native), `spo1_bc01` (PCR) | `spo1_5hmU` | reference only | this study (not yet public) |
| D10 *H. pylori* J99 | `hpj99` | `hpj99_4mC_pb`, `hpj99_6mA_pb` | 2.1 GB (+0.7 GB PacBio) | [1], [5] |
| D11 *T. denticola* ATCC 35405 | `tdenticola` | `tden_4mC_pb`, `tden_6mA_pb` | 3.5 GB (+1.8 GB PacBio) | [1], [5] |

All nanopore data are R10.4.1. The 15 rows of D1-D9 form the paper's benchmark; the replicate-2 rows of D8 and the rows
of D10 and D11 are additional rows. The SPO1 nanopore libraries are not yet publicly available; place their pod5 files
in `d9_spo1/pod5_files/` to score the `spo1_5hmU` row. The mouse-brain benchmark reads are spread over all 452 files of
the two runs, so the whole dataset is needed although only 39,622 reads are used.

Every file, its link, and its citation are listed in [`../benchmark/config/datasets.tsv`](../benchmark/config/datasets.tsv);
`python ../benchmark/rawmod_bench.py datasets` shows which of them are on disk.

## Files distributed with the benchmark

These files define the benchmark and are part of the repository:

| Path | Contents |
|---|---|
| `read_ids/<sample>.txt.gz` | the reads of each sample that the benchmark uses; `MANIFEST.tsv` gives their number and a checksum. The `subset` step cuts exactly these reads out of the downloaded pod5 files. |
| `rows/<row>/{gt,cand}.bed.gz` | ground-truth positions (0-based) of every row that is not rebuilt from the reference: modified positions (`gt`) and scored positions (`cand`); `MANIFEST.tsv` gives checksums of all rows, including the motif rows the `rows` step rebuilds, and `PROVENANCE.txt` describes how each row was built |
| `holdout/rawmod_seen_results81_mixed.tsv.gz` | every image of the released RawMod model's training pool with its split (`train`, `test`, `unused`); `--holdout r81` removes the positions of the `train` images |
| `pod5_urls/*.txt` | the files of the multi-file runs (D4-D7) |
| `pacbio/run_ipdsummary.sh` | PacBio ipdSummary calls for D10 and D11 (SMRT Link 13.1); `bench.pacbio_rows` turns them into the `_pb` rows |
| `unimeth_ft/hp26695_4mC_labels.bed.gz` | the 4mC positions used to label the UniMeth-FT 4mC fine-tuning |

### How the read sets were chosen

* D1: primary alignments starting in NC_000913.3:0-1,000,000 with mean basecall quality >= 10, at most 15,000 reads per
  library (the same window in all three libraries).
* D4: primary alignments on chromosome 1 (NC_003070.9) with mean quality >= 10, at most 100,000 reads.
* D5: primary alignments starting in NC_089035.1:0-13,000,000 with mean quality >= 10, at most 15,000 reads.
* D6: reads overlapping candidate CpG sites on chromosome 1.
* D2, D3, D7, D10, D11: the reads of the first 15,000 primary alignments; the WGA control of D3 over the region
  covered by the native library.
* D8: replicate 1, a held-out 20% of each library; replicate 2, 32,000 random reads (seed 0).

When more reads qualified than the cap, the reads with the lowest seeded hash of their read ID were kept
(`bench.common.read_rank`). Every site is then scored from exactly 10 of these reads, the same for every tool.

## References

1. Kulkarni et al. Comprehensive benchmarking of tools for nanopore-based detection of DNA methylation. *Nat. Commun.*
   (2026), doi:10.1038/s41467-026-75183-6. Data: AWS Open Data `ont-basemod-benchmark-data`.
2. ONT Open Data, GIAB 2023.05: https://epi2me.nanoporetech.com/giab-2023.05/
3. ONT Open Data, GM24385 5mC (bisulfite): https://epi2me.nanoporetech.com/gm24385-5mc/
4. ONT Open Data, modification validation 2024.10: https://epi2me.nanoporetech.com/mod-validation-data/
5. PacBio 2021-11 Microbial 96-plex: https://downloads.pacbcloud.com/public/dataset/2021-11-Microbial-96plex/
