# Stage 2.3: Image Dataset Loader & Preprocessing

This document specifies the PyTorch dataset pipeline that converts raw HTR manifest rows into preprocessed image tensors and encoded CTC labels for CRNN-CTC training.

---

## 1. Purpose

Phase 2.3 bridges the gap between the static Stage 2.1/2.2 artefacts (manifest CSVs + charset JSON) and the live training loop.  It provides:

1. A `torch.utils.data.Dataset` that loads line images and encodes labels.
2. Deterministic image preprocessing (grayscale, height resize, tensor conversion).
3. A collate function that pads variable-width images per batch.
4. An inspection script for offline QA without training.

> [!WARNING]
> **Do not modify Stage 1 or Stage 2.1/2.2 files.**
> The images, metadata, manifests, and charset are finalized.
> All preprocessing is applied *on-the-fly* during dataset loading — source files are read-only.

---

## 2. Input Files

| File | Description |
| --- | --- |
| `data/stage2_htr/manifests/htr_train.csv` | Training split manifest (79,243 rows) |
| `data/stage2_htr/manifests/htr_val.csv` | Validation split manifest (12,649 rows) |
| `data/stage2_htr/manifests/htr_test.csv` | Test split manifest (8,108 rows) |
| `data/stage2_htr/manifests/htr_all.csv` | Full manifest (100,000 rows) |
| `data/stage2_htr/vocab/charset_synth100k.json` | Character-to-ID mapping (98 classes) |

---

## 3. Module Reference

### `src/stage2_htr/transforms.py`
Pure-function image preprocessing:
- `load_image(path)` — Opens file, converts to grayscale mode `L`.
- `resize_to_height(img, target_height)` — Aspect-ratio-preserving resize via Lanczos resampling.
- `image_to_tensor(img, invert)` — Converts to `float32` tensor `[1, H, W]` in `[0, 1]`.
- `preprocess_image(path, target_height, max_width, invert)` — End-to-end pipeline.

### `src/stage2_htr/dataset.py`
- `HTRLineDataset` — Map-style `torch.utils.data.Dataset`.
- `encode_text(text, char_to_id)` — Encodes label text to class IDs. Raises `ValueError` on unknown characters. Never encodes `<BLANK>`.
- `decode_ids(ids, id_to_char, remove_blank)` — Decodes IDs back to text, stripping `<BLANK>`.

### `src/stage2_htr/collate.py`
- `htr_collate_fn(batch, pad_value)` — Right-pads images to max batch width, flat-concatenates labels for CTC loss.

### `src/stage2_htr/inspect_dataset.py`
- CLI tool for offline QA. Supports `--num-samples` quick check and `--check-all` full dry-run.

---

## 4. Dataset Output Dictionary

Each sample returned by `HTRLineDataset.__getitem__` is a dictionary:

| Key | Type | Description |
| --- | --- | --- |
| `image` | `Tensor [1, 64, W]` | Preprocessed grayscale float32 tensor |
| `label` | `Tensor [L]` | Encoded character IDs (no `<BLANK>`) |
| `label_length` | `int` | Number of characters in the label |
| `text` | `str` | Original text string (exact, unmodified) |
| `image_path` | `str` | Relative path to source PNG |
| `line_id` | `str` | Unique line identifier |
| `split` | `str` | `train`, `val`, or `test` |
| `width` | `int` | Original image width |
| `height` | `int` | Original image height |
| `processed_width` | `int` | Width after height-based resize |
| `processed_height` | `int` | Always 64 |
| `domain` | `str` | Text domain category |
| `difficulty_bucket` | `str` | `easy`, `medium`, or `hard` |
| `degradation_profile` | `str` | Applied degradation profile |
| `font_name` | `str` | Font used for rendering |

---

## 5. Image Preprocessing Policy

### Resize Strategy
1. **Fixed height = 64 px** for all images.
2. Width is scaled proportionally to preserve the original aspect ratio.
3. Lanczos resampling is used for high-quality downscaling.

### Width Handling
- **Default:** `max_width = null` — no width cap.
- **If `max_width` is set:** images wider than `max_width` after resize are **skipped** (never loaded). They are reported as load failures.
- **Never crop** text content.
- **Never stretch** or squeeze horizontally.
- **Aspect ratio is always preserved.**
- Future options (if memory is a concern): width-bucketed batching or proportional width resize with height re-pad.

### Pixel Convention
- Grayscale values in `[0.0, 1.0]`.
- **Default (`invert_image: false`):** Background ≈ 1.0 (white), text ≈ 0.0 (dark).
- **Inverted (`invert_image: true`):** Background ≈ 0.0 (black), text ≈ 1.0 (bright).
- No channel mean/std normalization in Phase 2.3.

### Batch Padding
- In `htr_collate_fn`, images are right-padded to the maximum width in the batch.
- Padding value matches the background:
  - `1.0` when `invert_image=false` (white padding).
  - `0.0` when `invert_image=true` (black padding).

---

## 6. Exact-Label Preservation Policy

> [!IMPORTANT]
> Text labels are loaded and encoded **exactly as they appear** in the manifest.
> No normalization, cleaning, or character removal is performed.

Specifically preserved:
- ZWNJ `U+200C` (zero-width non-joiner)
- Vedic accent marks `U+0951` (udatta), `U+0952` (anudatta)
- Avagraha `ऽ`, Visarga `ः`, Anusvara `ं`, Chandrabindu `ँ`
- Devanagari digits `०–९`
- Danda `।` and double danda `॥`
- Halant `्`

If any character is not in the charset, `encode_text` raises a hard `ValueError`. There is no `<UNK>` fallback.

---

## 7. CTC Label Preparation

For `torch.nn.CTCLoss`:
- `<BLANK>` (id 0) is the CTC blank token — it never appears in target labels.
- Labels are flat-concatenated in the collate function (not padded).
- `label_lengths` gives the per-sample label length.
- `processed_widths` gives the per-sample image width (before batch padding). The model's downsampling factor converts this to CTC `input_lengths` later.

---

## 8. Commands

### Quick sample inspection
```bash
python src/stage2_htr/inspect_dataset.py \
  --manifest data/stage2_htr/manifests/htr_train.csv \
  --charset data/stage2_htr/vocab/charset_synth100k.json \
  --num-samples 32 --batch-size 8
```

### Full dry-run (all 100k rows)
```bash
python src/stage2_htr/inspect_dataset.py \
  --manifest data/stage2_htr/manifests/htr_all.csv \
  --charset data/stage2_htr/vocab/charset_synth100k.json \
  --check-all
```

### Expected QA Outputs
- `data/stage2_htr/reports/dataset_loader_report.md`
- `data/stage2_htr/reports/dataset_loader_report.json`
- `data/stage2_htr/reports/preprocessed_samples_grid.png`
- `data/stage2_htr/reports/dataset_loader_check_all.json` (when `--check-all` is used)

---

## 9. Configuration

See `configs/stage2_data_synth100k.yaml` for centralized data-loading parameters:
```yaml
dataset:
  image_height: 64
  max_width: null
  invert_image: false
  augment: false

dataloader:
  batch_size: 16
  num_workers: 4
```
