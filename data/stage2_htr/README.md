# Stage 2 HTR data

The reviewed release is `releases/label_review_v1/`, containing matching
`manifests/`, `vocab/` and `reports/` folders.

Status: active release `label_review_v1`; all final release QA passed.

Use the paths in `configs/stage2_data_synth100k.yaml`. The flat `manifests/`,
`vocab/` and `reports/` folders contain earlier snapshots. Never mix a release's
manifests with a different character mapping.

The release contains 100,000 images (79,243 training, 12,649 validation and
8,108 test), 97 label characters plus CTC blank, and disjoint font holdouts.
Images are resized to height 64; width is unlimited and inversion is disabled
in the current configuration.

From the project root, default validation commands follow the configured paths:

```powershell
python src/stage2_htr/build_htr_manifest.py
python src/stage2_htr/build_charset.py
python src/stage2_htr/inspect_dataset.py --num-samples 32
python src/stage2_htr/inspect_dataset.py --check-all
```

The full release check also has four partition reports under
`outputs/label_verification_2026-10-27/loader_shards/`. Its aggregate binds the
100,000 checked records to the exact manifest and metadata hashes.

See [label verification and its limits](../../docs/development/label_review_release.md).
