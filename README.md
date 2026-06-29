# MANTRA: Manuscript Analysis and Neural Text Restoration Architecture

MANTRA is an end-to-end framework designed for the recognition, restoration, and interpretation of historical Devanagari manuscripts. This repository hosts the pipeline and configurations for **Stage 1 (Synthetic Dataset Generation)** and **Stage 2 (HTR Manifest Generation)**.

## 📌 Project Overview

MANTRA-Synth100k generates **100,000 synthetic Devanagari line images** with corresponding transcription ground truths. The dataset is designed to pretrain handwritten text recognition (HTR) models before transfer learning on printed and historical Devanagari manuscript data.

### Core Stages

```mermaid
graph TD
    A[Raw Text Corpus] --> B[Text Cleaning & Normalization]
    B --> C[Stage 1: Chromium Text Renderer]
    C --> D[Controlled Degradation Profiles]
    D --> E[MANTRA-Synth100k Dataset (100k PNGs)]
    E --> F[Stage 2: HTR Manifest Generation]
    F --> G[QA Verification Gates]
    G --> H[Final Training Split Manifests]
```

1. **Stage 1: Synthetic Dataset Generation**
   - **Corpus Cleaning & Rare Glyph Targets:** Collects and cleans Sanskrit prose/verse, historical Nepali prose, and administrative styles. Guarantees frequency coverage for rare conjuncts, matras, dandas, and Devanagari numerals.
   - **Chromium Rendering:** Renders text lines using Chromium's text shaper, ensuring correct Devanagari conjunct rendering and glyph positioning.
   - **Degradation Engine:** Simulates historical manuscript conditions (faded ink, uneven background, scan noise, bleedthrough, warped lines) across 24 controlled degradation profiles.
2. **Stage 2: HTR Manifest Generation**
   - Consolidates the 100k synthetic dataset.
   - Computes physical image metadata (width, height) and filters syntax errors.
   - Splits data into `train` (79,243), `val` (12,649), and `test` (8,108) manifests, enforcing zero leakage.

---

## 📂 Repository Structure

```
MANTRA/
│
├── configs/
│   └── stage1_synth100k_config.yaml  # Main synthetic generation & degradation configuration
│
├── data/                             # Generated artifacts (ignored in Git except for README/manifest)
│   ├── stage1_synthetic/
│   │   ├── README.md                 # Description of the synthetic dataset
│   │   ├── corpus_manifest.csv       # Summary of corpus sources and cleaning status
│   │   ├── fonts/                    # Downloaded TTF font files
│   │   ├── images/                   # Rendered png line-images (split into train/val/test)
│   │   ├── labels/                   # CSV label lists of text lines per domain
│   │   ├── metadata/                 # Detailed generation metadata CSVs
│   │   └── text/                     # Text corpora files (raw, cleaned, generated)
│   │
│   └── stage2_htr/
│       ├── manifests/                # HTR train/val/test CSV splits
│       └── reports/                  # Pipeline QA markdown and JSON reports
│
├── docs/                             # Project documentation
│   ├── stage1_synthetic/             # Stage 1 detailed documentation
│   │   ├── dataset_schema.md         # Schema description of final metadata CSV
│   │   ├── phase3_cleaning_summary.md# Summary of corpus cleaning processes
│   │   ├── phase3_text_cleaning.md   # Guidelines and patterns for raw text cleaning
│   │   ├── phase4_synthetic_image_generation.md  # Setup, commands, and options for image generation
│   │   ├── render_quality_summary_stage1_all_final.md # Stage 1 image quality audit
│   │   ├── stage1_all_merge_report.md# Report on labels merging and final count targets
│   │   └── stage1_dataset_design.md  # Core goals and architectural design of Stage 1
│   │
│   └── stage2_htr/                   # Stage 2 detailed documentation
│       └── stage2_phase2_1_htr_manifest.md       # Stage 2.1 design, manifest schema, and QA gates
│
├── src/                              # Pipeline Source Code
│   ├── stage1_dataset/               # Scripts for dataset synthesis, rendering, and QA
│   │   ├── chromium_text_renderer.py # Chromium-based rendering interface
│   │   ├── clean_corpus.py           # Text cleaning and preprocessing pipeline
│   │   ├── degradation_profiles.py   # Degradation algorithms and profiles
│   │   ├── finalize_dataset.py       # Reuses core images, checks missing items, generates top-ups
│   │   ├── generate_coverage_lines.py# Generates rare matra and conjunct text lines
│   │   ├── generate_final_dataset.py # Orchestrates parallel synthesis execution
│   │   ├── generate_line_labels.py   # Splits corpus into line-level chunks
│   │   ├── generate_synthetic_images.py # Base script for generating synthetic images
│   │   ├── make_sample_grids.py      # Generates grid images for QA visual inspection
│   │   ├── merge_core_labels.py      # Merges core label files
│   │   ├── merge_final_labels.py     # Aggregates labels across all domains
│   │   ├── phase3_report.py          # Generates phase 3 report summaries
│   │   ├── qa_audit.py               # Generates reports on character frequency statistics
│   │   ├── render_quality_check.py   # Audits rendered images for clipping and readability
│   │   └── render_text_line.py       # Wrapper for rendering text lines
│   │
│   └── stage2_htr/                   # Scripts for HTR manifests
│       └── build_htr_manifest.py     # Verifies data integrity and exports training manifest splits
│
└── .gitignore                        # Git exclusion rules for large datasets
```

---

## ⚡ Quick Start

### 1. Prerequisites & Environment Setup
Ensure Python 3.10+ is installed. Clone the repository and install required packages:
```bash
pip install -r requirements.txt
playwright install chromium
```

### 2. Generate Synthetic Dataset (Stage 1)
To generate raw line labels, download fonts, render synthetic images, apply degradation, and assemble the final 100k dataset:
```bash
# Run raw text cleaning
python src/stage1_dataset/clean_corpus.py

# Generate text lines targeting specific coverages
python src/stage1_dataset/generate_coverage_lines.py

# Perform core and final labeling merges
python src/stage1_dataset/merge_final_labels.py

# Run parallel image rendering and degradation synthesis
python src/stage1_dataset/generate_final_dataset.py

# Finalize image counts and outputs to exactly 100k
python src/stage1_dataset/finalize_dataset.py
```

### 3. Generate HTR Training Manifests (Stage 2)
To inspect the 100k images on disk, run automated QA validations, and build the train, validation, and test splits:
```bash
python src/stage2_htr/build_htr_manifest.py
```

---

## 🔍 Quality Assurance (QA) Gates
The generation and partitioning pipelines enforce strict quality limits to guarantee clean training datasets:
- **0 Physical Image Path Failures:** Ensures every referenced image file physically exists on the disk.
- **0 Empty Labels:** Filters out empty transcriptions.
- **0 Duplicate Path Leakage:** Enforces global and split-specific uniqueness of paths.
- **0 ASCII/Latin Intrusion:** Verifies that no Western numerals (`0-9`) or letters (`a-z`) contaminate the Sanskrit and Devanagari labels.
- **0 Bad OCR Symbols:** Removes invalid symbols (e.g. `_ & % € [ ]`) while preserving correct Devanagari punctuation (Visarga, Dandda, etc.).
