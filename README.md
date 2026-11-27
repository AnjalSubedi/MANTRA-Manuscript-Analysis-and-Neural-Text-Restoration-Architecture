# MANTRA

[![Tests](https://github.com/AnjalSubedi/MANTRA-Manuscript-Analysis-and-Neural-Text-Restoration-Architecture/actions/workflows/tests.yml/badge.svg)](https://github.com/AnjalSubedi/MANTRA-Manuscript-Analysis-and-Neural-Text-Restoration-Architecture/actions/workflows/tests.yml)

**Manuscript Analysis and Neural Text Restoration Architecture** is a research project for recognizing historical Nepali handwriting in Devanagari. MANTRA includes manuscript line segmentation, synthetic line generation, and a CRNN–CTC recognizer for variable-width line images.

## Results

| Checkpoint | Evaluation | Lines | Character error rate | Exact line matches |
| --- | --- | ---: | ---: | ---: |
| Synthetic baseline | Held-out MANTRA-Synth100k | 8,108 | **0.79%** | 5,773 |
| Synthetic baseline | Held-out manuscript pilot | 23 | 76.51% | 0 |
| Manuscript fine-tuned | Same manuscript pilot | 23 | **68.73%** | 0 |

Fine-tuning used 78 labeled lines from four manuscripts. The pilot held out three entire manuscripts for testing and used a separate manuscript for checkpoint selection. The remaining error rate makes this a research baseline for historical handwriting.

[Full results and evaluation protocol](docs/stage2_htr/crnn_ctc_results.md) · [Machine-readable metrics](results/crnn_ctc/metrics.json) · [Pilot reproduction](experiments/manuscript_pilot/README.md)

## Setup

Use Python 3.13 with the pinned dependencies. Run these commands from the repository root.

```bash
python -m pip install -r requirements-dev.txt
python -m pip install -e . --no-deps
python -m playwright install chromium
python -m pytest -q
```

## Train and evaluate

Training uses the reviewed `label_review_v1` data release configured in `configs/stage2_data_synth100k.yaml`. Prepare the local images and associated manifests using the [data workflows](docs/stage1_synthetic/data_workflows.md). The [charset and loader guide](data/stage2_htr/README.md) describes the matching inputs.

```bash
python src/stage2_htr/inspect_dataset.py --num-samples 32
python src/stage2_htr/train.py --device auto --output-dir outputs/crnn_ctc/baseline
python src/stage2_htr/evaluate.py --checkpoint outputs/crnn_ctc/baseline/best.pt --output-dir outputs/crnn_ctc/test
```

The trainer records validation CER and saves `best.pt`, `last.pt`, and `metrics.json`. The evaluator writes corpus CER and paired predictions and references. The [training guide](docs/stage2_htr/crnn_ctc_training.md) covers resumption, Kaggle packaging, manuscript fine-tuning, and external evaluation.

## Repository guide

| Location | Contents |
| --- | --- |
| [`src/manuscripts/`](src/manuscripts/) | [Line segmentation and crop review](docs/manuscript_segmentation.md) |
| [`src/stage1_dataset/`](src/stage1_dataset/) | [Synthetic Devanagari data generation and repair](docs/stage1_synthetic/data_workflows.md) |
| [`src/stage2_htr/`](src/stage2_htr/) | Recognition models, training, evaluation, and data loading |
| [`experiments/manuscript_pilot/`](experiments/manuscript_pilot/) | Document split, crop geometry, and pilot preparation |
| [`results/crnn_ctc/`](results/crnn_ctc/) | Aggregate CRNN–CTC results and epoch histories |
| [`tests/`](tests/) | CPU tests for data, segmentation, and recognition |

Source corpora, scans, transcriptions, and generated checkpoints are local run inputs and outputs. Dataset release documentation records the review policy, and the manuscript source notes describe access and provenance. [Synthetic release](data/stage1_synthetic/README.md) · [Manuscript sources](docs/research/manuscript_dataset_sources.md)
