# Prepare MANTRA-Synth100k data

The [dataset release](../../data/stage1_synthetic/README.md) and [data design](stage1_dataset_design.md) describe the local inputs. Use the active release selected by `paths.final_metadata` and `paths.final_labels` in the [Stage 1 config](../../configs/stage1_synth100k_config.yaml) and the matching manifests in the [Stage 2 config](../../configs/stage2_data_synth100k.yaml).

## Generate and publish

After preparing the corpus-derived and generated label CSVs, generate the synthetic release and build the matching manifests.

```bash
python src/stage1_dataset/merge_core_labels.py
python src/stage1_dataset/merge_final_labels.py
python src/stage1_dataset/generate_synthetic_images.py \
  --labels data/stage1_synthetic/labels/stage1_core_merged_lines.csv \
  --metadata-out data/stage1_synthetic/metadata/synthetic_image_metadata_core_merged.csv
python src/stage1_dataset/generate_final_dataset.py
python src/stage1_dataset/finalize_dataset.py
python src/stage2_htr/build_htr_manifest.py
python src/stage2_htr/build_charset.py
python src/stage2_htr/inspect_dataset.py --check-all
```

The full loader check decodes pixels and verifies preprocessing, label round-trips, and collation. Publication validates source metadata before replacing existing manifests. The [label-review release notes](../development/label_review_release.md) document quarantine decisions and verification limits.

## Font holdouts and legacy repair

Biryani and Jaldi supply validation fonts. Mukta and Noto Serif Devanagari supply test fonts. Fallback stays within the assigned split and manifest publication checks these boundaries. On the bundled Hind, Poppins, Rajdhani, Rozha One, and Teko faces, Devanagari digit code points use Latin glyphs. Digit-bearing lines use other fonts within their existing split.

For older releases generated before font isolation, inspect the repair plan, regenerate affected line images, and validate the resulting release.

```bash
python src/stage1_dataset/repair_font_splits.py --plan-only
python src/stage1_dataset/repair_font_splits.py --workers 3
python src/stage2_htr/build_htr_manifest.py --workers 3
python src/stage2_htr/inspect_dataset.py --check-all
```

The repair keeps text, line IDs, split membership, row count, and source PNGs. It records replacement PNGs under `images/font_holdout_v2/` and keeps the prior metadata and Stage 2 artifacts under `outputs/font_split_repair/`. It uses Chromium's shaped canvas renderer and publishes metadata after the replacements pass. An interrupted repair can continue from its saved journal. [Image generation guide](phase4_synthetic_image_generation.md)
