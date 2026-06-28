# Phase 4: Synthetic Devanagari Line Image Generation (MANTRA-Synth)

This document describes Phase 4 of the MANTRA HTR project: a research-quality synthetic text line-image generation pipeline that produces realistic training datasets from Devanagari text corpora.

## 1. Phase 4 Goal

The goal of Phase 4 is to generate a reproducible, highly controlled, metadata-rich, and QA-verified synthetic Devanagari line-image dataset. This pre-training dataset will be used to initialize handwriting text recognition (HTR) models before fine-tuning them on historical printed and manuscript Nepali collections.

## 2. Sarawgi-Style Baseline Summary

A recent Old Nepali HTR paper by Sarawgi et al. established a synthetic baseline for training:
- **Source Material**: Text extracted from historical Nepali books/textbooks.
- **Rendering**: Grayscale line-level images using Python's Pillow library.
- **Diversity**: Used multiple Devanagari fonts.
- **Augmentations**: Applied simple random noise and distortion effects.
- **Size**: Approximately 105,000 images.

While successful, this baseline has limitations:
- Standard Pillow rendering does not correctly shape complex Devanagari ligatures (matras and conjuncts) without specific system libraries (`libraqm`).
- Simple random augmentations do not match the structured degradation found in historical scans (e.g. faded ink, uneven illumination, paper fiber textures).
- The pipeline was not metadata-rich, making QA and regression testing difficult.

## 3. MANTRA Enhancements

MANTRA improves upon the Sarawgi baseline in the following ways:

1. **Complex Shaping Prioritization**: Prefers Pango + Cairo + HarfBuzz for correct Devanagari shaping. If Pillow is used as a fallback, it detects the presence of `libraqm` and verifies shaping using a programmatic smoke test.
2. **Controlled Degradation Profiles**: Replaces purely random augmentations with 8 realistic, parameter-controlled profiles (`D0` to `D7`) representing clean print, scan blur, faded ink, paper texture, uneven backgrounds, and broken strokes.
3. **Reproducibility & Determinism**: All random operations (font assignment, parameters, rotations, distortions) are seeded deterministically.
4. **Rich Metadata Tracking**: Saves detailed generation statistics for every line image, including splits, font groups, dimensions, renderer, and orthographic properties (presence of conjuncts, dandas, digits, etc.).
5. **Comprehensive QA Checks**: Automatically identifies rendering failures such as blank pages, text clipping, tofu (missing glyphs verified via font cmap parsing), and unreadable outputs. It compiles visual sample grids for visual verification.
6. **Source-Aware Splitting**: Strictly respects split tags generated during Phase 3 text cleaning to prevent train/validation/test data leakage.

## 4. Rendering Method

The rendering pipeline (`src/stage1_dataset/render_text_line.py`) enforces correct script shaping using the following prioritized engine fallback:

1. **Pango + Cairo + HarfBuzz**: System layout engine (highly reliable, preferred on Linux/macOS).
2. **Chromium (Playwright)**: Web layout engine (extremely reliable fallback for Windows, handles all complex script shaping rules faithfully).
3. **Pillow with RAQM**: Pillow drawing API with complex text layout support (only if `libraqm` is installed and visual shaping QA passes).
4. **Abort**: Aborts dataset generation if no reliable renderer is available. Plain Pillow rendering without RAQM is **strictly blocked** for full generation.

### Setup Instructions for Chromium Renderer
To run the renderer using headless Chromium, install Playwright and retrieve the browser:
```powershell
pip install playwright
python -m playwright install chromium
```

- **Tofu Safeguard**: Uses `fontTools.ttLib` to dynamically parse the font's character mapping table (`cmap`), skipping lines that contain characters unsupported by the font instead of generating empty tofu box images.

## 5. Degradation Profiles

The pipeline applies realistic document degradation (`src/stage1_dataset/degradation_profiles.py`):
- **`D0_clean_print`**: Clean rendering with grayscale normalization and minimal rotation.
- **`D1_light_scan`**: Adds mild Gaussian blur and minor contrast shift.
- **`D2_low_contrast_faded_ink`**: Compresses the dynamic range to simulate faded gray ink.
- **`D3_uneven_background`**: Generates a spatial gradient across the background to simulate uneven scanner lighting.
- **`D4_broken_strokes`**: Combines morphological thinning (erosion) with pixel dropout inside the ink region.
- **`D5_noisy_archive_scan`**: Combines low contrast, uneven background, mild blur, and salt-and-pepper noise.
- **`D6_blurred_scan`**: Employs moderate Gaussian or directional motion blur.
- **`D7_old_paper_texture`**: Applies procedural low-frequency noise mimicking aged paper fibers.
- **`D8_old_scan_soft`**: Simulates an old grayscale book scan using Gaussian blur, downscale-then-upscale operations, off-white background adjustments, and sparse black dust speckles.
- **`D9_faded_gray_ink`**: Models faded ink or weak print by applying soft morphological thinning, a local Gaussian noise mask for opacity variance, gray text values, and mild blur.
- **`D10_noisy_scan_crop`**: Simulates page-edge scanned crops with horizontal gradient shadow bands at the margins, uneven illumination, speckles, and vertical jitter without cutting text.
- **`D11_mild_warped_line`**: Applies a sinusoidal warping transformation to emulate baseline waviness, along with downscale lens softening.
- **`D12_dark_thick_scan`**: Models a darker, thick printed scan using morphological dilation of the text ink, reduced background brightness, mild blur, and scanner noise.

In addition to profiles, small random rotations (max ±1.5°), translations (jitter), and minor affine shears are applied.

## 6. Metadata Schema

Every generated image is recorded in `data/stage1_synthetic/metadata/synthetic_image_metadata.csv`:

| Column | Description |
| --- | --- |
| `image_id` | Unique ID of the generated image |
| `line_id` | Source line reference ID from Phase 3 CSV |
| `image_path` | Relative path to the image file |
| `text` | The ground-truth text |
| `source_id` | Source corpus identifier |
| `source_file` | Source text filename |
| `split` | Dataset split (`train`, `val`, or `test`) |
| `font_name` | Family name of the font used |
| `font_path` | Path to the font file |
| `font_group` | Font category (e.g. Modern, Printed, Sanskrit) |
| `font_size` | Selected font size (28 to 52) |
| `renderer` | Engine used (`pango_cairo`, `chromium`, `pil_raqm`, or `PIL_NO_RAQM_WARNING`) |
| `degradation_profile` | Applied degradation profile |
| `degradation_level` | Intensity score (0 to 12) |
| `degradation_parameters_json` | Valid JSON dictionary containing precise randomization parameters used |
| `width` | Final image width |
| `height` | Final image height |
| `text_length` | Character length of text |
| `has_danda` | Boolean indicating if text contains `।` |
| `has_double_danda` | Boolean indicating if text contains `॥` |
| `has_digit` | Boolean indicating if text contains digits |
| `has_conjunct` | Boolean indicating if text contains conjuncts |
| `seed` | Parameter generation seed |
| `render_status` | Output status (`success` or reason for failure) |

## 7. QA Reporting & Grids

The pipeline features verification scripts and reports:
1. `src/stage1_dataset/render_quality_check.py`: Inspects images for physical defects (empty files, blank canvas, clipping, out-of-bound dimensions) and dumps stats.
2. `src/stage1_dataset/make_sample_grids.py`: Composes cells of lines into composite grids for split, font, and degradation visual checks, and outputs a specialized **Shaping QA Grid** (`shaping_qa_grid.png`) that showcases 8 core complex text scenarios side-by-side.
3. `renderer_capability_report.json`: Saves a detailed capability checklist under `data/stage1_synthetic/reports/` tracking availability of Pango, Cairo, HarfBuzz, Chromium, and Pillow RAQM, along with test status for Devanagari shaping strings like *इसापूर्व* and *प्राप्त*.

## 8. CLI Commands

### Run the Smoke Test
Executes a 300-image sample run (with automated font downloads, shaping diagnostics, quality reports, and sample grids):
```powershell
python src/stage1_dataset/generate_synthetic_images.py --smoke-test --limit 300 --split all --seed 42 --overwrite --renderer chromium
python src/stage1_dataset/render_quality_check.py
python src/stage1_dataset/make_sample_grids.py
```

### Full Generation Command
Generates the complete pretraining dataset. If shaping support is missing, the generator blocks this action:
```powershell
python src/stage1_dataset/generate_synthetic_images.py --split all --seed 42
```

## 9. Version Control Protection

To prevent bloated repository commits, the `.gitignore` protects generated files.
The following locations are excluded from git:
- `data/stage1_synthetic/fonts/` (contains downloaded TTF files)
- `data/stage1_synthetic/images/` (contains generated line images)
- `data/stage1_synthetic/metadata/` (contains large CSVs and manifests)
- `data/stage1_synthetic/reports/sample_grids/` (contains QA images)
