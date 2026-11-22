# Devanagari content-quality audit

The original audit below is preserved as historical evidence. Follow-up
quarantine decisions, explicitly authored replacements and the additional
digit-font repair are documented in [the reviewed release](label_review_release.md).
For a new audit, choose a new `--output-dir` and pass that same directory to
`audit_ocr_review.py --audit-dir` to preserve earlier review evidence.

File integrity and loader QA do not establish exact transcription correctness.
Use these additional, read-only checks before accepting a foundation dataset:

```powershell
python -m pip install -r requirements-audit.txt
python -u src/stage1_dataset/audit_rendering.py --screen-structural
python -u src/stage1_dataset/audit_ocr_review.py
```

`audit_rendering.py` checks the complete active metadata against source labels,
font character coverage, unsafe numeral forms, and image-quality rules. It creates clean conjunct
probes for all fonts and actual/clean/reference comparisons for `श्री` and `सुब्बा`.
Detached dependent marks are recorded separately from accent/notation review.
The optional screening output is separate from active manifests.

`audit_ocr_review.py` uses the installed optional EasyOCR Devanagari recognizer
locally. It compares actual samples with clean controls. OCR disagreement is
advisory and must never automatically rewrite a ground-truth label.

The 2026-10-27 run checked 100,000 images and found no failures under the pixel
rules or missing font codepoints. It identified 176 images / 88 labels with
detached vowel/halant sequences and 206 other mark/accent candidates requiring
source review. These were not detected by the earlier integrity/preprocessing
QA. The 48 targeted visual comparisons showed no obvious `श्री` / `सुब्बा`
substitution; this does not certify all transcriptions.

The detailed local report, source review queue, font panels and screening preview
are under `outputs/rendering_audit_2026-10-27/`. Preserve historical spellings and
legitimate accents, verify corrections against the source, then regenerate
affected image-label pairs and rebuild downstream artifacts.
