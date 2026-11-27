"""Independent, advisory OCR of the conjunct probes and actual comparison set.

OCR disagreement is never a reason to rewrite a ground-truth label. Comparing
the same recognizer on clean references helps expose its own font/language errors.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage1_dataset.chromium_text_renderer import get_chromium_renderer, shutdown_chromium_renderer
from stage2_htr.metrics import edit_distance


def main():
    import easyocr
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", default="outputs/rendering_audit_2026-10-27")
    args = parser.parse_args()
    output = Path(args.audit_dir)
    torch.set_num_threads(2)
    reader = easyocr.Reader(["ne", "en"], gpu=False, detector=False,
                            model_storage_directory=str(output / "models"), verbose=False)

    def recognize(image):
        results = reader.recognize(np.array(image.convert("L")), detail=1, batch_size=1, workers=0)
        return " ".join(result[1] for result in results)

    records = []
    for path in sorted(output.glob("probe_*.png")):
        stem, index = path.stem.rsplit("_", 1)
        probes = ["श्री", "सुब्बा", "क्ष त्र ज्ञ श्र", "कि की कु कू कृ", "र्क क्र स्त्र न्द्र"]
        text = probes[int(index) - 1]
        with Image.open(path) as image:
            predicted = recognize(image)
        records.append({"kind": "clean_probe", "font_name": stem.removeprefix("probe_"),
                        "text": text, "image_path": str(path), "actual_ocr": predicted,
                        "actual_edit_distance": edit_distance(text, predicted)})
    print(f"Independent OCR: {len(records)} clean font probes", flush=True)
    with (output / "visual_samples.csv").open(encoding="utf-8-sig", newline="") as stream:
        samples = list(csv.DictReader(stream))
    renderer = get_chromium_renderer()
    try:
        for index, row in enumerate(samples):
            with Image.open(row["image_path"]) as image:
                actual_ocr = recognize(image)
            clean = renderer.render_text_canvas(
                row["text"], f"data/stage1_synthetic/fonts/{row['font_name']}.ttf",
                int(row.get("font_size") or 40), 20, 20,
            )
            clean_ocr = recognize(clean)
            records.append({"kind": "dataset_sample", "font_name": row["font_name"],
                            "text": row["text"], "image_path": row["image_path"],
                            "actual_ocr": actual_ocr, "clean_reference_ocr": clean_ocr,
                            "actual_edit_distance": edit_distance(row["text"], actual_ocr),
                            "clean_edit_distance": edit_distance(row["text"], clean_ocr),
                            "comparison_image": row["comparison_image"]})
            if (index + 1) % 8 == 0:
                print(f"Independent OCR: {index + 1}/{len(samples)} real images and clean references", flush=True)
    finally:
        shutdown_chromium_renderer()
    fields = list(dict.fromkeys(key for record in records for key in record))
    with (output / "ocr_review.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    summary = {
        "advisory_only": True,
        "model": "EasyOCR 1.7.2 Devanagari recognizer; local CPU; detector disabled",
        "clean_probes": sum(record["kind"] == "clean_probe" for record in records),
        "clean_probes_exact_ocr": sum(record["kind"] == "clean_probe" and record["actual_edit_distance"] == 0 for record in records),
        "dataset_samples": len(samples),
        "samples_clean_exact_actual_differs": sum(record["kind"] == "dataset_sample" and record["clean_edit_distance"] == 0 and record["actual_edit_distance"] > 0 for record in records),
        "clean_probes_character_disagreements_ignoring_spaces": sum(
            record["kind"] == "clean_probe" and "".join(record["text"].split()) != "".join(record["actual_ocr"].split())
            for record in records
        ),
        "samples_clean_exact_actual_character_differs_ignoring_spaces": sum(
            record["kind"] == "dataset_sample" and record["clean_edit_distance"] == 0
            and "".join(record["text"].split()) != "".join(record["actual_ocr"].split())
            for record in records
        ),
        "limitation": "OCR errors on clean fonts are model limitations; none of these results alone proves a rendering substitution.",
    }
    (output / "ocr_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
