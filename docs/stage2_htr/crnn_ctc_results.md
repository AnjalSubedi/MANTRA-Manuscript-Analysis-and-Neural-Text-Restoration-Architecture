# CRNN–CTC results

MANTRA uses a convolutional encoder and bidirectional LSTM to recognize variable-width Devanagari lines. The recognizer takes 64-pixel-high grayscale images and decodes text with greedy CTC. CER here is total insertions, deletions, and substitutions divided by the total number of reference Unicode code points.

## Data and checkpoint selection

| Run | Training | Validation | Test | Selected epoch |
| --- | ---: | ---: | ---: | ---: |
| MANTRA-Synth100k `label_review_v1` | 79,243 lines | 12,649 lines | 8,108 lines | 10 |
| Manuscript fine-tuning pilot | 78 lines from 4 manuscripts | 16 lines from 1 manuscript | 23 lines from 3 manuscripts | 24 |

The synthetic checkpoint reached 2.02% validation CER. The manuscript run started from those weights and used a fresh AdamW optimizer with a learning rate of 0.0002, batch size 8, and seed 42. Its vocabulary appended three code points observed in the training transcriptions while preserving existing class IDs. The Kirtipur petition supplied the validation lines used to select epoch 24.

## Before manuscript fine-tuning

| Evaluation with the synthetic checkpoint | Lines scored | CER | Exact lines |
| --- | ---: | ---: | ---: |
| Held-out synthetic test | 8,108 | 0.79% | 5,773 |
| Royal land grant, full diagnostic | 17 | 75.70% | 0 |
| Seven other Nepali manuscripts, diagnostic | 103 | 96.15% | 0 |

The Royal grant diagnostic included its heading, 15 body lines, and reverse endorsement. The seven-manuscript diagnostic scored 103 of 116 numbered body lines. It excluded 13 crops with reference gaps, overlapping seals, or a watermark. Straightening the same 103 lines produced 96.36% CER.

## Matched fine-tuning comparison

| Manuscript | Split | Lines | Original CER | Fine-tuned CER | Reduction (percentage points) |
| --- | --- | ---: | ---: | ---: | ---: |
| Kirtipur petition | Validation | 16 | 95.54% | 62.73% | 32.81 |
| Royal land grant | Test | 14 | 70.89% | 62.12% | 8.77 |
| Summons to salt traders | Test | 7 | 91.50% | 86.23% | 5.26 |
| Ghee receipt | Test | 2 | 83.59% | 77.34% | 6.25 |
| **Pooled manuscript test** | **Test** | **23** | **76.51%** | **68.73%** | **7.78** |

Both checkpoints used the same test crops and references. The matched comparison used 14 numbered body lines from the Royal grant before its watermarked final line. Absolute reductions use unrounded CER values. The pilot reduced the total from 915 to 822 edits across 1,196 reference characters. Eighteen lines improved, four worsened, and one tied. Both checkpoints recorded zero exact test lines.

The model improved on all three held-out manuscripts in this pilot. The small test set and high remaining CER limit conclusions about other manuscripts and practical transcription accuracy.

[Aggregate metrics and epoch histories](../../results/crnn_ctc/metrics.json) · [Replay the manuscript pilot](../../experiments/manuscript_pilot/README.md) · [Training and evaluation guide](crnn_ctc_training.md)
