# RNA (RNA004) DeepMod outputs

Figures from the first end-to-end RNA run (m6A, m5C, pseudoU, inosine + control,
two replicates). Pipeline: Remora refinement -> RNA featurization (--rna) ->
PileupInceptionV3 training + LODO.

## Figures
- `lodo_comparison_macro_f1.png` - Base vs LODO vs ZeroR, macro F1
- `lodo_comparison_label1.png`   - modified-class precision/recall/F1 (the meaningful one)
- `lodo_comparison_label0.png`   - unmodified-class metrics
- `confusion_matrix.png`         - base model test confusion matrix
- `precision_recall.png`         - base model PR curve
- `channel_importance.png`       - permutation channel importance

## Headline
Working RNA baseline. Modified-class LODO F1: pseU is the standout (0.67 rep1,
0.90 rep2); m6A and inosine moderate (~0.3-0.44); m5C weak (~0.15). Base model is
weak overall (modified F1 ~0.12), consistent with the ~4.4% positive-class
imbalance and mixing distinct modification signatures in one binary head. The
pseU > m6A/inosine > m5C ordering matches an independent signal-fingerprinting
analysis.
