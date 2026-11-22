# MANTRA-Synth100k

This folder contains the Stage 1 synthetic Devanagari dataset for MANTRA.
The reviewed release is `releases/label_review_v1/`.

Status: active release `label_review_v1`; all final release QA passed.

## Purpose

The dataset is used to pretrain the HTR model before transfer learning on printed and historical Devanagari manuscript data.

## Structure

```text
data/stage1_synthetic/
├── text/{raw,cleaned,generated}/
├── fonts/
├── images/                    # Original and replacement PNGs; originals retained
│   ├── {train,val,test}/
│   ├── font_holdout_v2/        # Resumable font-policy repairs
│   └── releases/label_review_v1/
├── releases/label_review_v1/
│   ├── metadata.csv           # Reviewed release image records
│   ├── labels.csv             # Reviewed release labels
│   ├── authored_source.csv    # Explicitly authored replacement text
│   ├── replacement_mapping.csv
│   ├── review/                # Decisions and exact-text approval receipts
│   ├── quarantine/            # Original warning records, still unverified
│   └── input_metadata.csv     # Preserved pre-review snapshot
├── labels/                    # Legacy generation inputs/snapshots
├── metadata/                  # Legacy generation metadata
├── reports/                   # Earlier Stage 1 reports
└── corpus_manifest.csv        # Original natural-text source catalogue
```

## Release counts

Total images: 100,000

- Train: 79,243
- Validation: 12,649
- Test: 8,108

## Selecting data

Use `paths.final_metadata` and `paths.final_labels` in
`configs/stage1_synth100k_config.yaml` and the matching paths in
`configs/stage2_data_synth100k.yaml`. Older flat files are preserved snapshots
and should not be substituted for the configured release.

382 warning records were quarantined and replaced with new authored examples;
they are not claimed to be corrected historical transcriptions. All 191 unique
original warning labels remain unverified until independent source evidence
is available. The original corpus catalogue is preserved; new synthetic text
and its split-specific source IDs are recorded in the release's authored bank
and metadata.

See [review decisions and validation](../../docs/development/label_review_release.md)
and the local run report `outputs/label_verification_2026-10-27/report.md`.
