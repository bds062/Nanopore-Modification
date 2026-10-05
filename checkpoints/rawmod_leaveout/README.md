# Leave-out models

Each model was trained with the configuration of `../rawmod_final/mixed.pt` (scripts/train/run_matched_loco.py,
`--fold logo_<name>`), but without any sample of one organism or library. They are scored in the supplementary
leave-out table of the benchmark (`test/evaluation/3_leaveout.sh`).

| Model | Withheld from training | Benchmark rows it scores |
|---|---|---|
| `ecoli.pt` | *E. coli* K-12 (wild type, dam- dcm-, dam- dcm- + M.SssI) | `ecoli_dam_6mA_ko`, `ecoli_dcm_5mC_ko`, `ecoli_mssi_5mC_ko` |
| `anabaena.pt` | *Anabaena* PCC 7120 | `anabaena_6mA` |
| `hp26695.pt` | *H. pylori* 26695 (native and WGA) | `hp26695_4mC`, `hp26695_6mA` |
| `plant.pt` | *A. thaliana* | `arabidopsis_cpg`, `arabidopsis_noncpg` |
| `mammal.pt` | HG001 and HG002 | `hg002_cpg` |
| `ont5mC.pt` | ONT 5mC oligonucleotides and their control | `syn_5mC` |
| `ont5hmC.pt` | ONT 5hmC oligonucleotides and their control | `syn_5hmC` |
| `ont6mA.pt` | ONT 6mA oligonucleotides and their control | `syn_6mA` |
| `spo1.pt` | SPO1 (all barcodes) | `spo1_5hmU` |
| `j99tden.pt` | *H. pylori* J99 and *T. denticola* | `hpj99_*_pb`, `tden_*_pb` (additional rows) |

For 4mC (`hp26695.pt`), 5hmC (`ont5hmC.pt`) and 5hmU (`spo1.pt`) the withheld data are the only training source of the
modification, so these models detect a modification they have not seen.
