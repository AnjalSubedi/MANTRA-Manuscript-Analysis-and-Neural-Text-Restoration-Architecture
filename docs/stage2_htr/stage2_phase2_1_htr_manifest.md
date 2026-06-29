# Stage 2.1: HTR Training Manifest Generation

This document outlines the design, execution, schema, and quality control metrics of Stage 2.1 HTR manifest generation for the MANTRA HTR-LLM framework.

---

## 1. Purpose of Stage 2.1

The purpose of Stage 2.1 is to build finalized HTR training manifests mapping **Devanagari line images** to their corresponding **transcription text labels**. These manifests serve as the direct inputs for training HTR models (Stage 2) and represent the consolidated outputs of Stage 1 synthetic generation.

> [!WARNING]
> **Do not modify Stage 1 dataset files.** The Stage 1 images, labels, and metadata files represent a finalized state and should never be manually edited, normalized, or cleaned during this process. Doing so will break the deterministic generation mapping and cause discrepancies in downstream HTR model evaluations.

---

## 2. Dataset Paths and Manifest Layout

### Input Source of Truth
- **Final Unified Metadata:** `data/stage1_synthetic/metadata/synthetic_image_metadata_stage1_all_final.csv`
- **Secondary Helper Labels:** `data/stage1_synthetic/labels/stage1_all_lines.csv`

### Output Manifest Files
All manifests are written as UTF-8 encoded CSV files under `data/stage2_htr/manifests/`:
- **Train Manifest:** `data/stage2_htr/manifests/htr_train.csv` (used for model optimization)
- **Validation Manifest:** `data/stage2_htr/manifests/htr_val.csv` (used for validation during training)
- **Test Manifest:** `data/stage2_htr/manifests/htr_test.csv` (used for final performance reporting)
- **Unified Manifest:** `data/stage2_htr/manifests/htr_all.csv` (contains all 100,000 records)

### QA Report Files
- **Markdown Summary:** `data/stage2_htr/reports/htr_manifest_report.md`
- **JSON Report:** `data/stage2_htr/reports/htr_manifest_report.json`

---

## 3. CSV Manifest Schema

Each manifest file contains exactly 20 columns:

| Column Name | Type | Description | Required / Optional |
| --- | --- | --- | --- |
| `image_path` | string | Relative path to the image file (starts with `data/stage1_synthetic/images/`) | **Required** |
| `text` | string | Exact ground-truth text transcription | **Required** |
| `split` | string | Target training split (`train`, `val`, or `test`) | **Required** |
| `line_id` | string | Unique identifier of the text line | **Required** |
| `source_id` | string | Identifier of the source book/manuscript | Optional |
| `source_file` | string | Name of the source text file | Optional |
| `domain` | string | Sub-domain category (e.g., historical, literary, sanskrit, numerals) | Optional |
| `font_name` | string | TTF font used to render the line image | Optional |
| `degradation_profile` | string | Degradation pattern applied (e.g., D1_light_scan, composed) | Optional |
| `degradation_family` | string | Degradation family (`single` or `composed`) | Optional |
| `difficulty_bucket` | string | Difficulty level (`easy`, `medium`, or `hard`) | Optional |
| `variant_index` | integer | Identifier of image variant (1, 2, or 3) | Optional |
| `generation_round` | string | Identification of synthetic generation round | Optional |
| `renderer` | string | Rendering engine utilized (`chromium`) | Optional |
| `width` | integer | Bounding box width of the image in pixels | Optional |
| `height` | integer | Bounding box height of the image in pixels | Optional |
| `has_digit` | string | Flag indicating if text contains Devanagari numerals (`true`/`false`) | Optional |
| `has_danda` | string | Flag indicating if text contains a single danda (`true`/`false`) | Optional |
| `has_double_danda` | string | Flag indicating if text contains a double danda (`true`/`false`) | Optional |
| `has_conjunct` | string | Flag indicating if text contains conjuncts/halant (`true`/`false`) | Optional |

---

## 4. Expected Counts

| Manifest File | Expected Count | Description |
| --- | --- | --- |
| `htr_all.csv` | **100,000** | Unified dataset rows |
| `htr_train.csv` | **79,243** | Training split |
| `htr_val.csv` | **12,649** | Validation split |
| `htr_test.csv` | **8,108** | Test split |

---

## 5. Quality Control & QA Validations

The manifest building pipeline automatically checks the integrity of the data against these strict quality gates:
1. **Physical Existence:** 100% of image paths must point to files that physically exist on disk (Target: `0` missing).
2. **Non-Blank Labels:** Every record must have a non-empty `text` transcription (Target: `0` blank).
3. **No Duplicate Paths:** Image paths must be unique globally and within splits to prevent leakage (Target: `0` duplicates).
4. **Digit Checks:** No ASCII digits (`0-9`) are allowed in the transcription. Devanagari numerals (`०-९`) are preserved. (Target: `0` ASCII digits).
5. **No Latin Letters:** No Latin characters (`a-zA-Z`) are allowed in the transcription (Target: `0` Latin letters).
6. **Symbol Filtering:** Transcriptions must not contain bad symbol noise (e.g. `_ & % € $ £ ₹ « » [ ]`). Visarga `ः`, Anusvara `ं`, Candrabindu `ँ`, and Avagraha `ऽ` are fully allowed and preserved.

---

## 6. How to Run / Regenerate Manifests

The HTR manifests can be regenerated at any time by executing the following command from the project root:

```bash
python src/stage2_htr/build_htr_manifest.py --metadata data/stage1_synthetic/metadata/synthetic_image_metadata_stage1_all_final.csv --output-dir data/stage2_htr/manifests --report-dir data/stage2_htr/reports
```

You can also run it with default paths:

```bash
python src/stage2_htr/build_htr_manifest.py
```
