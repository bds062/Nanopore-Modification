#!/bin/bash
# PacBio per-site 6mA/4mC calls for the _pb rows (H. pylori J99, T. denticola), following Kulkarni et al. (2026), who
# used the same data and ipdSummary options. The four barcodes of a species are pooled and aligned to the nanopore
# reference of the dataset, so coordinates match the benchmark:
#   HiFi kinetics -> per-strand records (ccs-kinetics-bystrandify) -> pbmm2 -> ipdSummary (Sequel II kinetics models).
# Requires the SMRT Tools of SMRT Link 13.1 (Sequel II; ccs-kinetics-bystrandify, pbmm2, pbindex, ipdSummary, samtools)
# on PATH, or SMRT_BIN pointing at smrtcmds/bin. Download the reads first: WITH_PACBIO=1 bash download_<dataset>.sh
#
#   bash pacbio/run_ipdsummary.sh d10_hpylori_j99      (or d11_tdenticola)     [THREADS=16]
# Output: <dataset>/pacbio/ipd/{ref.fa,aligned.bam,modbase.gff,kinetics.csv.gz,PROVENANCE.txt}; then build the rows with
#   (cd ../benchmark && python -m bench.pacbio_rows)
set -euo pipefail
cd "$(dirname "$0")/.."
DS=${1:?dataset: d10_hpylori_j99 or d11_tdenticola}
T=${THREADS:-16}
B=${SMRT_BIN:+$SMRT_BIN/}
case $DS in
  d10_hpylori_j99) BCS="2009 2033 2057 2081" ;;
  d11_tdenticola)  BCS="2023 2047 2071 2095" ;;
  *) echo "unknown dataset $DS" >&2; exit 1 ;;
esac
O=$DS/pacbio/ipd; mkdir -p "$O"
cp "$DS/ref.fa" "$O/ref.fa"; ${B}samtools faidx "$O/ref.fa"
: > "$O/bystrand.fofn"
for bc in $BCS; do
  in=$DS/pacbio/reads/m64004_210929_143746.bc$bc.bam
  out=$O/bc$bc.bystrand.bam
  [ -s "$out" ] || ${B}ccs-kinetics-bystrandify "$in" "$out"
  echo "$out" >> "$O/bystrand.fofn"
done
[ -s "$O/aligned.bam" ] || ${B}pbmm2 align "$O/ref.fa" "$O/bystrand.fofn" "$O/aligned.bam" --sort -j "$T" --preset CCS
[ -s "$O/aligned.bam.pbi" ] || ${B}pbindex "$O/aligned.bam"
${B}ipdSummary -j "$T" "$O/aligned.bam" --reference "$O/ref.fa" --identify m6A,m4C --minCoverage 20 --methylFraction \
  --pvalue 0.01 --gff "$O/modbase.gff" --csv "$O/kinetics.csv"
gzip -f "$O/kinetics.csv"
{ echo "dataset	$DS"; echo "barcodes	$BCS"; echo "smrtlink	13.1.0.221970";
  echo "ipdSummary	--identify m6A,m4C --minCoverage 20 --methylFraction --pvalue 0.01"; } > "$O/PROVENANCE.txt"
grep -v '^#' "$O/modbase.gff" | cut -f3 | sort | uniq -c
