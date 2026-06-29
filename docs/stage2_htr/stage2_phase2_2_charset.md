# Stage 2.2: Charset and Vocabulary Creation

This document outlines the design, execution, schemas, and quality control of the character vocabulary (charset) creation for Stage 2.2 of the MANTRA HTR-LLM framework.

---

## 1. Purpose of Charset Creation

For handwritten text recognition (HTR) model training using Connectionist Temporal Classification (CTC) loss, a fixed character mapping vocabulary is required. This mapping associates specific character tokens with target class IDs, enabling:
1. **Differentiable Label Encoding:** Encoding transcription text strings into numeric label sequences for baseline CRNN-CTC model loss computation.
2. **Deterministic Label Decoding:** Converting model probability output vectors back into Devanagari text line transcriptions.

> [!WARNING]
> **Do not modify Stage 1 or Stage 2.1 files.** The synthetic dataset, labels, and final splits are finalized. Manually editing image directories, metadata files, or split manifests will break the deterministic generation pipeline, invalidate test evaluation, and cause label-image mismatches during model loading.

---

## 2. Input and Output Mapping

### Single Source of Truth Input
- **Unified HTR Manifest:** `data/stage2_htr/manifests/htr_all.csv` (100,000 text labels)

### Output Vocabulary Artifacts
All vocabulary files are written under `data/stage2_htr/vocab/`:
- **Text Vocabulary:** `data/stage2_htr/vocab/charset_synth100k.txt` (a simple character list, one real character per line, starting from space)
- **JSON Structure:** `data/stage2_htr/vocab/charset_synth100k.json` (comprehensive mapping schema)

---

## 3. Special Token & CTC Blank Policy

To support the first CRNN-CTC baseline, we define the following special token rules:
- **`<BLANK>` (index 0):** The CTC blank token. It represents the null class during sequence decoding. In alignment with standard CTC frameworks, the blank token is placed at index 0 and is excluded from the list of real character strings.
- **No Padding/Unknown Tokens in Model Output:** To keep output classes minimal and highly clean:
  - `<PAD>` (padding token) is excluded from the model classes. If sequence batching requires label padding, it is handled dynamically within the PyTorch dataloader collate function and mapped to a ignore-index like `-100`, not inside the CTC output vocabulary.
  - `<UNK>` (unknown token) is excluded. Because the charset is built by scanning 100% of the labels in `htr_all.csv`, every character is guaranteed to be encodable. Any unknown character encountered during encoding will throw a strict ValueError and fail execution rather than masking issues via an unknown placeholder.
- **Real Characters start at index 1:** Real characters start mapping from index 1. Space (`" "`) is mapped to index 1.

---

## 4. Character Ordering and Sorting Rules

To ensure reproducibility across builds, unique characters extracted from the corpus labels are organized in a stable deterministic order:
1. **Space Character (`" "`)** is placed first among real characters (Class ID: 1).
2. **Devanagari Digits (`०-९`)** are placed next (Class IDs: 2–11).
3. **Devanagari Punctuation (`।` and `॥`)** follow the digits (Class IDs: 12–13).
4. **Common Punctuation** (any Unicode character marked as category `P` or standard ASCII punctuation, excluding Devanagari punctuation) is sorted by Unicode code point value.
5. **Devanagari Block characters** (`\u0900` to `\u097f`, excluding digits and punctuation) are sorted by Unicode code point value.
6. **Remaining symbols** are sorted by Unicode code point value.

---

## 5. JSON Vocabulary Schema

The output JSON mapping file `charset_synth100k.json` conforms to the following schema:

```json
{
  "blank_token": "<BLANK>",
  "blank_id": 0,
  "characters": [
    " ",
    "०",
    "१",
    "...(sorted real characters)"
  ],
  "char_to_id": {
    "<BLANK>": 0,
    " ": 1,
    "०": 2,
    "१": 3
  },
  "id_to_char": {
    "0": "<BLANK>",
    "1": " ",
    "2": "०",
    "3": "१"
  },
  "num_characters": 97,
  "num_classes": 98
}
```

---

## 6. QA Validation Gates

The charset building script verifies data integrity using strict QA checks:
1. **0 Blank Labels:** Verifies that no empty text transcription rows exist in the HTR manifest.
2. **0 Unencodable Characters:** Verifies that every character in the 100k labels successfully translates into a class ID.
3. **0 Encode-Decode Mismatches:** Ensures that for all 100,000 rows, `decode_ids(encode_text(text)) == text` (with blank token stripping applied during decoding).
4. **0 ASCII Digits:** Audits that no Western numbers (`0-9`) contaminate the corpus labels.
5. **0 Latin Letters:** Audits that no Western characters (`a-zA-Z`) contaminate the corpus labels.
6. **Sanskrit/Vedic Accents Preservation:** Checks for the presence of accents like `U+0951` and `U+0952` inside the vocabulary to ensure historical text layout accuracy.

---

## 7. How to Run / Regenerate Vocabulary

The vocabulary and QA reports can be regenerated by running:

```bash
python src/stage2_htr/build_charset.py --manifest data/stage2_htr/manifests/htr_all.csv --output-dir data/stage2_htr/vocab --report-dir data/stage2_htr/reports
```

You can also run it with default paths:

```bash
python src/stage2_htr/build_charset.py
```
