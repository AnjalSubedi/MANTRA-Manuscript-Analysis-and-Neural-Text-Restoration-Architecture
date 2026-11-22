# Nepali-language handwriting and manuscript sources

Checked: 2026-10-27. Scope: Nepali-language text in Devanagari. Sanskrit-only, Newar/Pracalit, synthetic images, isolated characters, and text-only corpora are not substitutes for the historical Nepali recognition benchmark.

## Finding

This search verified a downloadable modern Nepali word-recognition resource and concrete historical Nepali image sources. It did not verify an openly downloadable, ready-to-train historical Nepali line-image/transcription benchmark. The user already requested Sarawgi's data and reports that access was declined because of copyright; exclude that dataset from acquisition plans. These are access findings, not a proof that no other dataset exists.

## Historical Nepali: unavailable research dataset

### 1. Sarawgi / nepOCR research dataset

- Source: https://github.com/anjalisarawgi/nepOCR/
- Paper: https://arxiv.org/abs/2512.17111
- Fit: historical Old Nepali handwriting; closest to MANTRA's primary target.
- Access: the repository marks the handwritten Old Nepali stage as restricted / not publicly available. Public code and models do not include permission to use the source data.
- User-confirmed outcome: a request was already made and declined because of copyright. Do not recommend requesting it again or depend on it for MANTRA. Retain the publication as related work only.

## Historical Nepali: image sources for an independently annotated benchmark

### A. Penn Libraries: Pinas ko mantra katha

- Catalog and downloads: https://openn.library.upenn.edu/Data/0002/html/mscoll390_item803.html
- Identifier: Ms. Coll. 390 Item 803.
- The catalog explicitly states the language is Nepali, describes two leaves with 7-8 lines per leaf, and marks the manuscript images/content public domain. It provides three folio/spread image captures plus an additional composite image; these are not four independent pages.
- Technical metadata is in TEI, but this is a catalog description, not a verified full transcription. The item-specific metadata license is CC BY 4.0.
- Status: catalog, image links, and item rights statement checked. Image/TEI retrieval timed out in this pass, so pixel decoding and complete-file availability are not yet verified.
- Role: small genuine manuscript pilot requiring manual transcription, crop review, and explicit handling of corrections/marginalia. It cannot alone support broad manuscript-generalization claims.

### B. Bhanubhakta Ramayana: Sundara Kanda manuscript images

- File collection: https://commons.wikimedia.org/wiki/Category:Bhanubhakta_Ramayana_manuscripts
- Checked file: https://commons.wikimedia.org/wiki/File:Bhanubhakta_Ramayana_-_Sundara_Kanda_-_1.jpg
- The category lists 43 files, including a full/cropped derivative pair; do not call this 43 independent manuscript pages. The checked file identifies traditional Nepali text, dates it to 1850, credits British Library material, and carries a public-domain label.
- Original catalog reference recorded on Commons: https://eap.bl.uk/archive-file/EAP272-1-2-32
- Role: larger candidate for manual line transcription. Check every file's provenance/rights, recover highest-resolution originals, and detect derivative duplicates. A modern printed Ramayana may help locate passages but is not a diplomatic ground truth for this witness.
- The current synthetic corpus already includes literary Ramayana material. Audit textual overlap before assigning manuscript passages to final testing.

### C. Shree Swasthani Brata Katha manuscript images

- File collection: https://commons.wikimedia.org/wiki/Category:Shree_Swasthani_Brata_Katha_manuscripts
- Checked file: https://commons.wikimedia.org/wiki/File:Shree_Swasthani_Brata_Katha-_Nepali_language_manuscript_-_1.jpg
- The category lists 15 files. The checked first file states Nepali language, dates it to the 1900s, attributes the source to the British Library, and carries a public-domain label.
- Original catalog reference recorded on Commons: https://eap.bl.uk/archive-file/EAP272-1-2-169
- Role: another manuscript witness for a pilot. No checked line-level transcription release was located. Review per-file rights, image legibility, duplicates, and manuscript identity before importing.

Commons rights statements above are the checked file-page labels, not a blanket certification of every file in each category. Dataset size, independent writer count, and verified line count remain unmeasured.

## Historical Nepali: permission-dependent archive

### 2. Documenta Nepalica

- Platform and annotation policy: https://nepalica.hadw-bw.de/nepal/
- Fit: historical Nepali legal and administrative documents, with diplomatic editions and original line breaks.
- Access: the checked National Archives examples distinguish edition text under CC BY-SA 4.0 from facsimiles requiring prior written permission.
- Preparation: obtain permitted images, choose diplomatic rather than normalized display, extract original text/line breaks, and annotate line geometry. Exclude or mark editorial reconstruction and illegible content under an explicit scoring policy. Some Nepali documents include Sanskrit formulae; assign language at line/span level where necessary.

Concrete document candidates checked:

| Document | Subject and date | Edition |
| --- | --- | --- |
| K_0469_0009 | Prithvinarayan Shah appoints Bhagavantanath as overseer of jogis; VS 1827 | https://nepalica.hadw-bw.de/nepal/editions/show/1 |
| K_0019_0014A | Rajendra's order concerning relief work on guthi land; VS 1891 | https://nepalica.hadw-bw.de/nepal/index.php/editions/show/51550 |
| L_1200_0009G | Janga Bahadur's order concerning sirto tax collection; VS 1931 | https://nepalica.hadw-bw.de/nepal/editions/show/57001 |

These are candidate source documents, not three completed MANTRA datasets. The first also has a facsimile viewer at https://digi.hadw-bw.de/view/k_0469_0009 with IIIF metadata links. Viewing availability does not establish training or redistribution permission.

## Downloadable modern Nepali handwriting: intermediate training

### 3. Handwritten Labeled Dataset for Text Recognition

- Dataset: https://www.kaggle.com/datasets/sweekardahal/handwritten-labeled-dataset-for-text-recognition
- Verified file inventory: 4,358 JPG crops plus `labels.csv`.
- Kaggle lists CC0 for version 1. Treat this as the publisher's stated license and retain its provenance.
- Labels were actually downloaded: 4,358 records with unique filenames and nonempty targets. Examples include `घुमाउनु`, `मिलाएका`, `गृहकार्य`, and `नेपाली`.
- Actual format: UTF-16 text, filename followed by whitespace and transcription, no header. Despite the `.csv` suffix, do not assume comma-separated UTF-8.
- Three crops were downloaded and visually inspected. Their dimensions were 38x16, 75x33, and 75x25 pixels, so resolution and crop completeness need review. No full semantic label certification was performed.
- Role: possible intermediate Nepali handwriting adaptation and input-pipeline testing. It is not a historical manuscript benchmark and mostly supplies short words, limiting contextual-correction evaluation.

Use filename mapping, not row-position matching: `0.jpg` is the last label record, not the first. Keep the original source filenames and inspect duplicates/related writer pages before creating splits. Do not assume the words have independent writer identities.

### 4. Nepali Handwritten Images for Text Detection / NHD

- Dataset: https://www.kaggle.com/datasets/sweekardahal/nepali-handwritten-images-for-text-detection
- Author description: https://github.com/dahalsweekar/Nepali-Handwritten-Dataset-Major-Collection
- Kaggle version 2 inventory: 958 images and 958 XML files; listed license CC0.
- Inspected annotation: `test/125.xml` contains 134 bounding boxes, all with the class name `text`. No recognition transcripts were found in this sample.
- Role: detection/localization development, or a manually transcribed modern Nepali line collection. It is not ready for recognition merely because XML annotations exist.

The author's approximately 1,000-image description and the release's exact 958-image file inventory describe different levels of precision; use versioned file counts when importing.

## Research dataset worth requesting locally

### 5. Kathmandu University Nepali handwritten words

- Paper: https://link.springer.com/chapter/10.1007/978-3-031-97019-1_11
- Lab publication: https://ilprl.ku.edu.np/publications/42
- Lab contact: https://ilprl.ku.edu.np/contact/

The publisher's abstract reports 8,558 handwritten word samples from 28 individuals, by Regent Nanda Vaidya and Bal Krishna Bal. This is modern word-level handwriting, not historical manuscript lines. No public dataset download was verified. The lab lists `bal@ku.edu.np` as its lead researcher's contact. A research request is a reasonable next action; it has not been sent. Access and reuse terms remain to be established.

## Recommended MANTRA path

1. Build a small, independently transcribed historical Nepali pilot from the explicit public-domain Penn item and individually checked Ramayana/Swasthani images. Sarawgi's dataset is excluded following the user's confirmed refusal.
2. Meanwhile audit the small downloadable word-recognition resource as an optional bridge from synthetic fonts to modern handwriting.
3. Reserve manuscript-disjoint, independently checked historical Nepali lines for verifier calibration and final evaluation. Words without context cannot replace that benchmark.
4. Use NHD only for localization unless additional actual transcription annotations are obtained or created.
5. Keep Sanskrit as optional auxiliary data, explicitly labeled and evaluated separately.
6. Expand to a bounded Documenta Nepalica collection only if its separate image-use requirements are satisfied; diplomatic editions being readable online do not establish image reuse permission.

A further sentence-recognition paper was located at DOI `10.1109/ICOIICS67115.2025.11390446`. Its primary abstract/data could not be inspected because the publisher required a browser verification challenge. Dataset-size claims from secondary summaries are therefore not treated as a verified release here.

Inspection evidence is under `outputs/nepali_dataset_search_2026-10-27/`: a label file, three word crops, one detection XML, and `inspection_summary.json`. Active MANTRA manifests, synthetic images, and dependencies were not changed.
