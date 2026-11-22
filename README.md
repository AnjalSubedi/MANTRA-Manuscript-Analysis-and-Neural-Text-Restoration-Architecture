# MANTRA

Manuscript Analysis and Neural Text Restoration Architecture provides synthetic
Devanagari line generation and data preparation for handwritten text recognition.
The current pipeline covers Stage 1 image generation, Stage 2 manifests,
character vocabulary, preprocessing, PyTorch data loading, and draft manuscript
line segmentation.

## Folder structure

```text
MANTRA/
├── configs/                  # Shared generation and loader settings
├── src/
│   ├── stage1_dataset/        # Cleaning, rendering, degradation, font policy, repairs
│   ├── stage2_htr/            # Manifests, vocabulary, dataset, collation, inspection
│   └── manuscripts/           # Draft line crops, source coordinates, review previews
├── tests/
│   ├── stage1_dataset/        # Font isolation and rendering regression checks
│   ├── stage2_htr/            # Loader, configuration, and publication checks
│   └── manuscripts/           # Crop geometry and mark-preservation checks
├── docs/
│   ├── development/          # Maintenance, layout, and validation notes
│   ├── research/             # Methodology draft and dataset-source investigations
│   ├── stage1_synthetic/     # Dataset design and generation documentation
│   └── stage2_htr/           # Manifest, charset, and loader documentation
├── data/                     # Local corpora and generated data; large files ignored
│   ├── stage1_synthetic/
│   │   ├── fonts/
│   │   ├── images/            # Every new PNG output stays under images/
│   │   │   └── font_holdout_v2/{train,val,test}/
│   │   ├── labels/
│   │   ├── metadata/
│   │   ├── releases/          # Reviewed metadata, labels, quarantine and provenance
│   │   └── text/{raw,cleaned,generated}/
│   └── stage2_htr/
│       ├── releases/          # Matching manifests, vocabulary and QA per release
│       └── {manifests,vocab,reports}/ # Legacy artifacts
├── outputs/                  # Run logs, resumable repair journals, backups; ignored
├── pyproject.toml            # Editable package and test configuration
├── requirements.txt          # Verified runtime dependencies
└── requirements-dev.txt      # Runtime dependencies and pytest
```

Reusable code belongs in `src/`, regression tests in `tests/`, and explanatory
documents in `docs/`. Keep one-off reports and logs in a named `outputs/` run
folder. Generated data and checkpoints do not belong in Git.

## Setup

Use Python 3.10 or newer; the pinned requirements were verified with Python 3.13
on Windows. From the repository root:

```bash
python -m pip install -r requirements-dev.txt
python -m pip install -e . --no-deps
python -m playwright install chromium
python -m pytest -q
```

The raw corpora, generated label files, fonts, PNGs, metadata, and Stage 2
artifacts are local inputs/outputs excluded from Git. A fresh checkout needs the
corpus and label inputs described in [dataset design](docs/stage1_synthetic/stage1_dataset_design.md)
before image generation. The repository does not include a corpus downloader
for every source; copying this repository alone does not reproduce the dataset.

## Work with an existing dataset

The Stage 1 configuration selects the active metadata and labels through
`paths.final_metadata` and `paths.final_labels`. The Stage 2 configuration selects
the matching manifests and vocabulary. Use these configured paths for training;
older files under the flat `labels/`, `metadata/` and `manifests/` folders remain
historical snapshots. See the [label-review release](docs/development/label_review_release.md)
for quarantine decisions, source-verification limits and release checks.

First validate Stage 1 metadata and publish manifests. Failed QA writes reports,
returns a nonzero exit code, and leaves any existing manifests untouched.

```bash
python src/stage2_htr/build_htr_manifest.py
python src/stage2_htr/build_charset.py
python src/stage2_htr/inspect_dataset.py --num-samples 32
python src/stage2_htr/inspect_dataset.py --check-all
```

The full inspection decodes pixels, applies the configured preprocessing,
roundtrips labels, checks tensor shape and range, and collates every successful
sample. It does not accept image headers as proof that pixels are readable.
Failures produce JSON and Markdown diagnostics, including when no sample loads.

## Font holdouts and existing data repair

`configs/stage1_synth100k_config.yaml` assigns validation fonts to Biryani and
Jaldi, and test fonts to Mukta and Noto Serif Devanagari. The remaining fonts are
used for training. Font fallback stays within the sample's assigned split.
Manifest publication rejects violations of this policy.

The bundled Hind, Poppins, Rajdhani, Rozha One and Teko faces use Latin numeral
forms for Devanagari digit code points. They are allowed for lines without
Devanagari digits; digit-bearing lines use another font within their split.
Character-map coverage alone is insufficient to approve those combinations.

For a legacy dataset generated before font isolation was enforced:

```bash
python src/stage1_dataset/repair_font_splits.py --plan-only
python src/stage1_dataset/repair_font_splits.py --workers 3
python src/stage2_htr/build_htr_manifest.py --workers 3
python src/stage2_htr/inspect_dataset.py --check-all
```

The repair preserves text, line IDs, split membership, and row count. It keeps
original PNGs, writes replacement PNGs under `images/font_holdout_v2/`, and backs
up the old metadata and Stage 2 artifacts under `outputs/font_split_repair/`.
It uses Chromium's shaped canvas renderer, records the actual rendering and
degradation metadata, and publishes the new metadata only after all replacements
pass. Repeating an interrupted run resumes its journal.

## Generate new images

With the corpus-derived and generated label CSVs available, the merge and image
generation sequence is:

```bash
python src/stage1_dataset/merge_core_labels.py
python src/stage1_dataset/merge_final_labels.py
python src/stage1_dataset/generate_synthetic_images.py --labels data/stage1_synthetic/labels/stage1_core_merged_lines.csv --metadata-out data/stage1_synthetic/metadata/synthetic_image_metadata_core_merged.csv
python src/stage1_dataset/generate_final_dataset.py
python src/stage1_dataset/finalize_dataset.py
```

See the [Stage 1 documentation](docs/stage1_synthetic/phase4_synthetic_image_generation.md)
for cleaning, label preparation, and rendering details. Generation and
finalization enforce the configured font partition, and finalization preserves
an already valid 100,000-row dataset instead of replacing it with legacy parts.

## Loader configuration

`configs/stage2_data_synth100k.yaml` controls image height, width limits, inversion,
paths, batch size, and worker count. Relative configured paths are resolved from
the repository root. CLI `--manifest`, `--charset`, and `--batch-size` override
the respective settings.

```python
from stage2_htr.config import load_data_config, dataset_options, make_dataloader
from stage2_htr.dataset import HTRLineDataset

config = load_data_config()
dataset = HTRLineDataset(
    config["paths"]["manifest_train"], config["paths"]["charset_json"],
    **dataset_options(config),
)
loader = make_dataloader(dataset, config)
```

With a width limit, the dataset filters over-wide images before sampling and
exposes them in `dataset.skipped_rows`. An all-filtered inspection fails with a
report instead of attempting an empty batch. Inversion and padding agree:
white background uses `1.0`; inverted black background uses `0.0`.

On Windows, create worker-based loaders inside an `if __name__ == "__main__":`
guard. The inspection command already does this.

See [maintenance and validation](docs/development/review_fixes.md) and
[the loader reference](docs/stage2_htr/stage2_phase2_3_dataset_loader.md).

## Manuscript line segmentation

The manuscript command proposes line boundaries for a main text block with
roughly horizontal Devanagari writing. It exports line PNGs, source coordinates,
numbered page previews, contact sheets, and an index for visual review. Source
pixels inside each polygon retain their original resolution; the surrounding
area is filled with white.

The command accepts PNG, JPEG, TIFF, BMP, WebP, and JPEG 2000 files, including
uppercase extensions. It expands multi-page scans in frame order, converts
non-PNG pages automatically, and saves `normalized.png` in each page's output
folder. Existing ordinary PNG inputs are reused directly. Original file paths,
file hashes, frame numbers, and normalization details are recorded in `page.json`
and the line index. Same-name scans with different extensions receive distinct
page IDs.

EXIF orientation is applied before segmentation, and transparent areas are
composited on white. The segmentation image uses 8-bit RGB. For 16-bit grayscale
scans, values are scaled across the full 0–65535 range, with the conversion
recorded. Coordinates refer to the normalized PNG.

For locally acquired EAP272/1/2/11 scans named
`EAP272-1-2-11_0001.png` through `EAP272-1-2-11_0129.png`, run:

```bash
python src/manuscripts/segment_lines.py --input-dir outputs/manuscript_downloads/EAP272-1-2-11/images --output-dir outputs/manuscript_segmentation/EAP272-1-2-11/segmentation_v3 --overrides configs/manuscripts/eap272_1_2_11.json
```

Use `--pages 2-5,20,129` for a smaller selection in its own output directory.
Unique numeric filename suffixes supply page numbers. For other filename
patterns or conflicting suffixes, selection uses 1-based positions in the
expanded list, sorted by natural filename order and then by frame number.
Repeating a run verifies the source and output hashes before reusing completed
pages. Use a new output directory when changing the source selection, settings,
or segmentation code version.

All crops await visual review. Flags identify potential ink intersections and
irregular spacing. The collection configuration leaves the handwritten cover
for manual layout review and supplies the final page's single-line region.
Marginal notes and folio numbers need separate review. The source scans and
generated crops are local files excluded from Git. Consult the
[manuscript-source investigation](docs/research/manuscript_dataset_sources.md)
for provenance and access findings.
