# Label-review release: label_review_v1

This release addresses the active dataset's label-review queue and the
validation-only `°` character. It preserves 100,000 samples and the split sizes
79,243 / 12,649 / 8,108.

## Review decisions and evidence limits

All 191 unique warning labels (382 image variants) have a disposition record in
`data/stage1_synthetic/releases/label_review_v1/review/decisions.csv`.

- 88 labels / 176 images had detached vowel/halant warnings.
- 103 labels / 206 images had other mark/accent warnings.
- 38 labels were found verbatim, after documented normalization, in available
  local OCR sources. This establishes lineage; it does not independently verify
  a printed or handwritten source.
- The source text files for the other 153 labels are absent locally. Original
  matching-edition scans were not located in the searched project/user folders.
- All 191 originals are therefore `quarantined_unverified`. None is declared
  incorrect or given a guessed correction. Their verification remains pending
  independent source evidence.

The original images, labels and provenance are preserved in the quarantine
metadata and input snapshot. Quarantine is a release-eligibility decision,
not a claim that every warned historical form is wrong.

## Replacement samples

The release replaces those 382 records with new, explicitly authored synthetic
examples. They have new line IDs and source IDs; they are not represented as
corrected historical transcriptions. The source bank is `authored_source.csv`.

Template banks and source IDs are disjoint between train, validation and test.
All replacement images use the assigned split's fonts and clean rendering, with
saved font, text and image hashes plus rendering parameters. Every replacement
is linked to its quarantined predecessor in `replacement_mapping.csv`.

The replacement count matches each split's quarantine count: train 308,
validation 44 and test 30. Other transcriptions and sample identities are
unchanged. An additional digit-font repair changes images as described below.

## Degree-symbol coverage

The 62 validation images containing literal U+00B0 `°` are preserved. Their
notation is not silently converted to U+0970 or another symbol.

Four explicitly authored angle statements create 32 training examples of `°`.
Their final fonts are restricted to Karma, Noto Sans Devanagari and Yatra One
after the numeral audit below. These are synthetic
coverage data, not purported reconstructions of the source book's notation.
No validation transcription is copied into training.

Both Vedic accent classes remain in the retained dataset: 5,233 U+0951 and
1,307 U+0952 occurrences. No blanket accent removal or historical spelling
normalization is applied.

## Devanagari numeral rendering

Reviewing the replacement contact sheets exposed a separate font issue. The
bundled Hind, Poppins, Rajdhani, Rozha One and Teko faces draw Devanagari digits
with Latin numeral forms. Poppins and Teko have exactly identical decomposed
outlines for all ten Latin/Devanagari digit pairs. The other three were confirmed
by side-by-side rendering; glyph names or cmap membership alone did not catch it.

The release repairs all 11,580 affected images, preserving text, line IDs,
variants, sources and split membership. Digit-bearing training lines use Karma,
Noto Sans Devanagari or Yatra One, subject to full text coverage. Validation and
test fonts already have distinct Devanagari numerals. Original PNGs, a metadata
snapshot and a resumable replacement journal are preserved.

The five affected faces remain usable for lines without Devanagari digits.
Font selection, rendering and manifest publication now reject the unsafe
font/digit combinations. This restriction describes the inspected bundled font
files; new font versions need their own visual digit audit.

## Preventing recurrence

`label_quality.py` supplies shared warning and character-coverage checks.
Warnings request source review. A source-supported exception can be retained by
recording an exact-text approval with a scan or authoritative-transcription
reference; a match to the same OCR text is insufficient.

Generation preflights require reviewed labels. Manifest publication rejects
unapproved warnings and evaluation characters absent from training, alongside
the existing image integrity, split and font gates. The charset and inspection
commands follow the configured active release.

## Artifacts and activation

Stage 1 release artifacts:
`data/stage1_synthetic/releases/label_review_v1/`.

Stage 2 manifests, vocabulary and QA reports:
`data/stage2_htr/releases/label_review_v1/`.

The active pointers are in `configs/stage1_synth100k_config.yaml` and
`configs/stage2_data_synth100k.yaml`. Activation follows manifest, vocabulary,
sample-loader and full-loader QA. Run logs and prior configuration backups are
under `outputs/label_verification_2026-10-27/`.

The release report distinguishes closure of warning cases for the active
release from independent transcription verification of the quarantined originals.
This is not a full human verification of the other source transcriptions.
