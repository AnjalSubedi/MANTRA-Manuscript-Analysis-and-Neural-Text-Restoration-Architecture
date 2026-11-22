# Manuscript dataset shortlist for MANTRA

Checked on 2026-10-27. This is a discovery and access assessment, not a certification of transcription quality. Existing MANTRA data and dependencies were not changed. One FLAME page/XML pair and one AnciDev transcription were retrieved as inspection samples; full datasets were not imported.

## Recommended acquisition order

1. Inspect and convert AnciDev for initial historical Devanagari recognition experiments.
2. Use FLAME's accessible Sanskrit image/PAGE-XML material for an independent pilot; preserve any final evaluation portion from training.
3. Consider the MIDF/eGangotri reviewed-page release if a larger Sanskrit training collection is needed and line-level alignment can be validated.
4. Establish a separate Nepali manuscript collection from the image sources in [the Nepali source report](nepali_dataset_sources.md), with independently checked transcriptions. Sarawgi's dataset is excluded: the user already requested it and was declined because of copyright. Documenta Nepalica remains contingent on separate image permission. Sanskrit results alone do not validate Old Nepali recognition.

The role assignments above are recommendations. Actual suitability depends on sampled transcription accuracy, language, manuscript identity, image resolution, crop quality, and available annotation expertise.

## 1. AnciDev: first dataset to inspect

- Official repository: https://github.com/vriti2003/AnciDev
- Author-provided download: https://drive.google.com/drive/folders/1wOqHgSUPieIoBwQ1ACn6ixpOxBuVo3T6?usp=sharing
- Paper: https://aclanthology.org/2025.bhasha-1.8/

The authors report 3,000 transcribed lines from 500 Devanagari manuscript pages. The live repository supplies the Drive link. Its public folder contains manuscript subfolders; one checked Sanskrit Ramayana folder contains page PNGs and matching Google transcription documents. A sample transcription exported successfully with five nonempty lines. This confirms access to source material, not a ready-made MANTRA line manifest. No clear dataset license was located in the inspected README. Confirm reuse terms and inspect line alignment before adopting the data.

**Next action:** inventory page/transcription pairs; recover manuscript IDs; extract text; inspect line correspondence and crop complete matras/conjuncts. Preserve author splits and additionally assess manuscript-disjoint partitions. Do not automatically modernize the transcription.

## 2. FLAME historical Sanskrit material: image and geometry pilot

- Project and data: https://github.com/flame-cai/gnn-synthetic-layout-historical
- Paper: https://arxiv.org/html/2608.18696v1
- Repository evaluation folder: `app/tests/eval_dataset/`

The paper describes 31 pages across three Sanskrit manuscripts with Unicode/layout annotations. The checked repository tree contains 15 evaluation JPG/XML pairs; broader paper coverage was not verified file by file. One downloaded pair decodes at 2500 by 940 pixels and contains 23 XML text lines with nonempty labels. The repository has a GPL-3.0 software license; source-image/data rights need separate confirmation. The paper retains a diacritic in ground truth even if segmentation misses it, so recropping is important for exact image-label training.

**Next action:** parse PAGE `TextLine`, `Coords`, and `TextEquiv/Unicode`; display overlays; preserve marginalia and catalog marks under an explicit annotation policy. Validate crop completeness before converting to line images. Avoid using the same pages for both verifier fitting and final testing.

Inspection revision: `5355016e8bbb7d9254ae17bf84d67c7f70939154`. Local sample: `outputs/manuscript_dataset_search_2026-10-27/flame_sample.jpg` and `flame_sample.xml`.

## 3. MIDF/eGangotri reviewed Sanskrit release: larger candidate

- Primary dataset: https://huggingface.co/datasets/tadad/midf-egangotri-sanskrit
- Relevant configuration: `pages_alignment_reviewed`
- Published reviewed release: `pages-reviewed-v0.5`

The dataset card reports 2,879 reviewed-alignment pages with manuscript-disjoint page splits. Scholarly transcriptions, human alignment review, and model-assisted repairs are distinguished by provenance fields; sampled records explicitly say `not_character_verified`. These are page targets, not automatically verified line labels. The much larger `smriti_ocr` configuration is machine-generated OCR. The top-level license is `other`; record-level transcription and image permissions must be checked separately. The API confirmed the repository is public and ungated, with released Parquet shards.

**Next action:** pin a revision; start with a small human-alignment-reviewed subset; inspect text/image correspondence and region coverage; derive and verify line labels before connecting it to the existing line loader. Treat OCR predictions as weak labels, not final-test ground truth. Check source overlap with other collections, since mirrors can contain the same scans.

Observed main revision: `6db1c00265dc00eb6510ac11499e67f4debb1369`. Main evolves; use a frozen release for an experiment. Do not download the entire scan/OCR repository just to inspect reviewed pages.

## 4. Documenta Nepalica: closest language match, requires preparation

- Platform and editorial policy: https://nepalica.hadw-bw.de/nepal/
- Example document: https://nepalica.hadw-bw.de/nepal/index.php/editions/show/51550
- Related recognition project: https://github.com/anjalisarawgi/nepOCR/

Documenta Nepalica provides diplomatic editions with original line breaks and TEI-XML. The checked example distinguishes licensed edition text from manuscript facsimiles that require prior written permission. It is an archive/edition source, not a confirmed bulk line-image benchmark. Sarawgi's repository explicitly marks the Old Nepali handwritten dataset restricted and not publicly available. The user already requested that dataset and reports a copyright-based refusal; do not include it as a future access option.

**Next action:** use the separately documented public-domain image candidates for the initial Nepali pilot. For any later Documenta Nepalica collection, establish permitted image access, obtain diplomatic XML, and annotate line geometry. Do not send inquiries on the user's behalf without explicit authorization. Preserve seals, damaged-text annotations, and historical forms according to the project annotation guide.

## Supporting resources, with narrower roles

- Printed Devanagari ground truth, DOI `10.11588/DATA/EGOKEI`: https://doi.org/10.11588/DATA/EGOKEI . An optional printed-transfer resource already cited in the proposal. It does not replace handwritten manuscript supervision. The dataset landing/API could not be verified during this pass; file access remains to be checked.
- Indiscapes/IHDIA: https://ihdia.iiit.ac.in/indiscapes/ . Useful for manuscript layout research; do not assume layout labels include recognition transcriptions.
- Nepalese Pracalit manuscript ground truth: https://doi.org/10.5281/zenodo.6967421 . A relevant neighboring research resource, but a different script from the current Devanagari recognizer.

## Acceptance checklist for an imported collection

This is the proposed MANTRA import policy, not a statement that any candidate already passes:

1. Preserve original files and source/manuscript/page identifiers.
2. Record access terms, dataset version, checksums, and annotation provenance.
3. Distinguish scholarly transcription, reviewed alignment, model-assisted repair, and unreviewed OCR.
4. Inspect image-text pairs and crop boundaries at the actual model input resolution.
5. Keep variants and pages from a document within the same experimental partition.
6. Record uncertain readings and scoring rules; do not replace source readings with a normalized edition.
7. Check charset coverage and CTC target feasibility before training.
8. Reserve an independently reviewed test subset for harmful-edit and preservation evaluation.

## Minimal folder arrangement when importing

```text
data/manuscripts/<collection>/raw/         Original images, XML, and transcripts
data/manuscripts/<collection>/lines/       Derived, verified line crops
data/manuscripts/<collection>/manifests/   Labels, source IDs, split assignments
data/manuscripts/<collection>/provenance/  Release, terms, checksums, review records
outputs/manuscript_dataset_search_2026-10-27/  Discovery evidence and pilot samples
```

These data directories are a suggested future layout; they have not been populated by this search. Keep the existing synthetic data separate and use an adapter to emit the HTR manifest schema.
