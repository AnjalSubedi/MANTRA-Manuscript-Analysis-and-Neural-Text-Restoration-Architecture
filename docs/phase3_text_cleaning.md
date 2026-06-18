# Phase 3 Text Cleaning & Line Label Generation Documentation (Second-Pass Refinement)

## Purpose of Phase 3
Phase 3 involves cleaning raw Devanagari text files collected in Phase 2 and generating line-level labels. The primary goal is to curate a high-quality, verified text corpus from historical documents for subsequent synthetic Devanagari line-image rendering. The second-pass cleaning further refines this by discarding front matter, table-of-contents, index pages, bibliographies, and advanced OCR/scanner garbage.

## Directories
- **Raw Input Directory:** `data/stage1_synthetic/text/raw/` (contains files like [devmala_vamshavali.txt](file:///d:/MANTRA/data/stage1_synthetic/text/raw/devmala_vamshavali.txt))
- **Cleaned Output Directory:** `data/stage1_synthetic/text/cleaned/` (contains files like [devmala_vamshavali.clean.txt](file:///d:/MANTRA/data/stage1_synthetic/text/cleaned/devmala_vamshavali.clean.txt))
- **Labels Directory:** `data/stage1_synthetic/labels/` (contains [historical_nepali_lines.csv](file:///d:/MANTRA/data/stage1_synthetic/labels/historical_nepali_lines.csv))
- **Reports Directory:** `data/stage1_synthetic/reports/` (contains JSON and Markdown summaries)

## Cleaning & Filtering Rules (Second-Pass)

### Section Trimming (TOC & Bibliography Removal)
To preserve only the core body text, specific file-level start and end triggers are implemented:
- **[bhasha_vamshavali_1.txt](file:///d:/MANTRA/data/stage1_synthetic/text/raw/bhasha_vamshavali_1.txt):** Discards the noisy opening editorial sections and starts from the invocation `श्रीणणेशाय नमः` / `भाषावंशावली`.
- **[nepali_sanskriti_vishayak_agralekhharu.txt](file:///d:/MANTRA/data/stage1_synthetic/text/raw/nepali_sanskriti_vishayak_agralekhharu.txt):** Discards publisher headers and indices, starting from `नेपालमा पाशुपत सम्प्रदायः एक ग्रध्ययन`.
- **[devmala_vamshavali.txt](file:///d:/MANTRA/data/stage1_synthetic/text/raw/devmala_vamshavali.txt):** Discards introductory lists and table-of-contents pages, starting from `नमो गोरङ्गकालीम्याम्‌`.
- **[nepal_ko_itihas_peshal_dahal.txt](file:///d:/MANTRA/data/stage1_synthetic/text/raw/nepal_ko_itihas_peshal_dahal.txt):** 
  - Trims front matter: Starts after the table of contents entry `सहायक ग्रन्थस्‌ची`, at the beginning of the first chapter (`नेपालको इतिहासका स्रोतहरू`).
  - Trims bibliography: Discards all bibliography/references starting at `सहायक ग्रन्थहरूको सूची` near the end of the file.

### OCR & Scanner Garbage Filtering
- **Long Digit Sequences:** Discards lines containing 4 or more consecutive `0`s or `1`s (e.g. `0000`, `010100001000`) or 5 or more consecutive ASCII digits.
- **Mixed Codes:** Discards lines containing scanner codes like `:::::222::::`.
- **ASCII Density Check:** Discards short lines (`length < 25`) if ASCII digits make up more than 15% of the line and Devanagari letters count is less than 6. This eliminates standalone page numbers, publisher dates, or index items while keeping dates within full Devanagari sentences.
- **Punctuation & Symbols:** Discards lines where non-word scanner symbols (colons, dots, dashes, stars, etc.) make up more than 20% of the line.

## Execution Workflow

Run the cleaning scripts sequentially from the root of the workspace (`D:\MANTRA`):

1. **Clean Corpus:**
   ```powershell
   python src/stage1_dataset/clean_corpus.py
   ```
2. **Generate Line Labels:**
   ```powershell
   python src/stage1_dataset/generate_line_labels.py
   ```
3. **Compile Phase 3 Report:**
   ```powershell
   python src/stage1_dataset/phase3_report.py
   ```

## Git Tracking Note
> [!WARNING]
> Raw corpus texts and their cleaned variations should **not** be pushed to public Git repositories unless the source licenses explicitly allow public redistribution. 
> However, cleaning scripts, docs, configs, and reports can be safely tracked.
