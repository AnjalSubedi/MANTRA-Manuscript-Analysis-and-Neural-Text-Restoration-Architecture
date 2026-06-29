# MANTRA-Synth100k Dataset Design

## Dataset Name

MANTRA-Synth100k

## Purpose

MANTRA-Synth100k is a synthetic Devanagari line-image dataset designed for pretraining handwritten text recognition models before transfer learning on printed and historical Devanagari manuscript data.

The dataset supports the MANTRA framework: an uncertainty-aware hybrid HTR-LLM system for recognition, restoration, and interpretation of historical Devanagari manuscripts.

## Core Idea

This dataset will not be a random text rendering dataset. It will be a coverage-aware, renderer-verified, degradation-profiled synthetic Devanagari line-image dataset.

Each image will contain one rendered Devanagari text line. Since the text is generated or collected before rendering, the ground-truth label will be automatically known.

## Target Size

Total images: 100,000

Train: 80,000  
Validation: 10,000  
Test: 10,000  

## Dataset Composition

1. Natural Devanagari text lines: 70,000
2. Rare conjunct and matra coverage lines: 15,000
3. Sanskrit verse and danda-heavy lines: 5,000
4. Historical Nepali / administrative-style lines: 5,000
5. Names, dates, places, numerals, and title-style lines: 5,000

## Corpus Categories

The raw text corpus should include:

1. Modern Nepali prose
2. Old Nepali or historical-style Nepali
3. Sanskrit prose
4. Sanskrit verse
5. Hindi Devanagari text for script diversity
6. Religious and philosophical texts
7. Historical, legal, or administrative-style text
8. Manually generated rare conjunct coverage text

## Important Devanagari Coverage

The dataset must cover:

- Basic vowels: अ आ इ ई उ ऊ ऋ ए ऐ ओ औ
- Basic consonants: क ख ग घ ङ च छ ज झ ञ ट ठ ड ढ ण त थ द ध न प फ ब भ म य र ल व श ष स ह
- Matras: ा ि ी ु ू ृ े ै ो ौ
- Anusvara: ं
- Chandrabindu: ँ
- Visarga: ः
- Halant: ्
- Danda: ।
- Double danda: ॥
- Devanagari digits: ० १ २ ३ ४ ५ ६ ७ ८ ९
- Common conjuncts: क्ष त्र ज्ञ श्र द्ध द्व द्य ह्म स्त्र क्त्य प्र ब्र भ्र ग्र क्र

## Rendering Requirement

The preferred renderer is Pango + Cairo + HarfBuzz because Devanagari requires proper text shaping.

Fallback rendering with PIL is allowed only if RAQM/libraqm support is available and visual inspection confirms that matras, conjuncts, and ligatures are rendered correctly.

## Font Requirement

Minimum fonts: 10  
Recommended fonts: 15–25  

Font groups:

1. Clean modern Devanagari fonts
2. Printed/book-style Devanagari fonts
3. Sanskrit-style fonts
4. Nepali publication-style fonts
5. Handwriting-like fonts, only if rendering is correct

Each dataset sample must store the font name and font size.

## Degradation Profiles

The dataset will use controlled degradation profiles:

- D0_clean_print
- D1_light_scan
- D2_low_contrast_faded_ink
- D3_uneven_background
- D4_broken_strokes
- D5_noisy_archive_scan
- D6_blurred_scan
- D7_old_paper_texture

Recommended distribution:

- Clean: 20%
- Light degradation: 30%
- Moderate degradation: 25%
- Faded or uneven background: 20%
- Harder synthetic damage: 5%

## Line Length Distribution

- Short lines, 10–25 characters: 10%
- Medium lines, 26–60 characters: 55%
- Long lines, 61–100 characters: 30%
- Very long lines, 101–130 characters: 5%

## Dataset Splitting

The split must be source-aware and font-aware.

This means:

- Some text sources should be used only in validation or test.
- Some fonts should be held out from training and used only in validation or test.
- Exact duplicate lines should not dominate the dataset.

## Quality Control

The dataset must pass these checks before being accepted:

1. All image paths exist.
2. All labels are non-empty.
3. Labels contain valid Devanagari text.
4. Train/validation/test split sizes are correct.
5. Character frequency is reported.
6. Conjunct frequency is reported.
7. Font distribution is reported.
8. Degradation distribution is reported.
9. Duplicate labels are reported.
10. Sample grids are manually inspected.

## Final Outputs

- images/train/
- images/val/
- images/test/
- labels/train.csv
- labels/val.csv
- labels/test.csv
- labels/metadata_full.csv
- reports/dataset_stats.json
- reports/char_frequency.json
- reports/conjunct_frequency.json
- reports/font_distribution.json
- reports/degradation_distribution.json
- reports/sample_grids/
- DATASET_CARD.md
