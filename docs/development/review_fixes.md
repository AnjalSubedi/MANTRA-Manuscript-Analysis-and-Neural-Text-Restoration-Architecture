# Review fixes and maintenance

The 2026-10-27 review identified seven issues. They are covered by these changes:

| Issue | Resolution | Regression coverage |
| --- | --- | --- |
| Fonts crossed every split | Explicit font groups shared by generation, fallback, repairs, and manifest QA | Partition validation, restricted fallback, manifest violations |
| Failed manifest QA returned success | Validate all rows before publication; save reports and return failure without replacing existing manifests | Missing images, invalid labels/splits/IDs, bad symbols, cross-split text |
| Full inspection read headers only | Use the real dataset preprocessing and collate pipeline for every row | Truncated PNG with valid header; empty manifest |
| Inspection crashed on failed loads | Retain diagnostics, skip empty batches, smoke-test only valid indices | All-failed and mixed-success inspection; missing setup inputs |
| Width caps raised during iteration | Filter widths at dataset initialization using actual image dimensions | Stale metadata dimensions, all-filtered dataset, valid DataLoader batch |
| Inversion/configuration ignored | Shared YAML loader and matching image/padding settings | Nondefault height, inversion, and black batch padding |
| No installable dependency manifest | Runtime/dev requirements and an editable src-layout package | Import and regression suite |

## Repository boundaries

- `src/stage1_dataset/`: synthetic generation, rendering, font partition, repairs.
- `src/stage2_htr/`: training data preparation and inspection.
- `tests/`: deterministic regression checks using temporary tiny datasets.
- `docs/`: design, commands, and maintenance guidance.
- `data/`: corpora, PNGs, active metadata, manifests, vocabularies, QA reports.
- `outputs/`: ignored logs, backups, and journals for individual maintenance runs.

Do not place reusable audit scripts in an output folder. Move them into the
appropriate `src/` package when making them part of the supported pipeline.

## Data repair and provenance

The repair never relabels a PNG or simply moves its split to satisfy font QA.
It re-renders only samples with disallowed fonts, preserving labels, line IDs,
variants, and split membership. Original images remain available for provenance.
Replacement images are grouped under `data/stage1_synthetic/images/font_holdout_v2/`.
New repair runs use a separate run directory inside that folder, so later repairs
cannot overwrite images referenced by an earlier dataset. Initial migration
images may remain directly under the split folders.

`outputs/font_split_repair/original_metadata.csv` and `original_manifests/` /
`original_reports/` retain the previous active artifacts. `replacements.jsonl`
supports resume. The input fingerprint prevents reuse of a journal with different
metadata or configuration. New metadata is published atomically after every
replacement passes; rebuild the manifests afterward.

Chromium canvas uses the same browser text shaper with embedded fonts and
measured ink bounds, avoiding per-line page screenshots. Repair metadata records
`chromium_canvas`, font size, padding, actual dimensions, degradation parameters,
and the previous image path. Any clean fallback updates the profile metadata.

## Validation commands

```bash
python -m pytest -q
python src/stage2_htr/build_htr_manifest.py --workers 3
python src/stage2_htr/inspect_dataset.py --num-samples 192
python src/stage2_htr/inspect_dataset.py --check-all
```

Manifest QA checks pixel decoding, required fields, allowed split names,
duplicate paths, duplicate line/variant pairs, cross-split labels and line IDs,
invalid label characters, and font isolation. Loader QA additionally checks
configured preprocessing, finite tensor values, range, label roundtrips, batch
collation, and a DataLoader smoke test. Empty inputs cannot pass.

The repair keeps a 100,000-row active dataset. Because old PNGs are retained,
the total physical PNG count may exceed 100,000; the active metadata and
manifests determine which samples belong to the dataset.

## Earlier font-partition result: 2026-10-27

- 33 regression tests passed; the final Stage 1 subset also passed after the last
  metadata compatibility adjustment.
- 43,925 images were replaced to enforce the font partition.
- All 100,000 active images passed manifest decoding and publication gates.
- A 192-sample inspection and the full 100,000-sample loader inspection passed.
- Full inspection collated 6,250 batches with no load, encoding, shape, range,
  finite-value, or batching failures. No samples were excluded by a width cap.
- The active dataset uses 8 training, 2 validation, and 2 test fonts, with no font
  overlap. Split counts remain 79,243 / 12,649 / 8,108.
- Comparing every row against the backup found no changes to text, line IDs,
  source IDs/files, split membership, variant indices, or generation rounds.

The local run summary is `outputs/font_split_repair/validation_summary.json`.
Detailed QA reports for that snapshot are under `data/stage2_htr/reports/`; original metadata
and previous manifests/reports remain backed up under `outputs/font_split_repair/`.

The subsequent [label-review release](label_review_release.md) adds quarantine,
new authored replacement samples, degree-symbol training coverage and numeral
font restrictions. Its manifests, vocabulary and QA use release-specific paths
selected by the current configuration.
