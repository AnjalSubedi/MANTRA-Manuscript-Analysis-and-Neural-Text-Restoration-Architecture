# -*- coding: utf-8 -*-
"""
MANTRA Stage 2.3 — Dataset Inspection & QA Script.

Verifies the full image-loading → preprocessing → encoding → batching
pipeline without training a model.  Produces QA reports and a sample
grid image.
"""
import os
import sys
import csv
import json
import argparse
import logging
import math
import time
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from PIL import Image, ImageDraw, ImageFont

# Ensure we can import sibling modules when run from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stage2_htr.dataset import HTRLineDataset, encode_text, decode_ids
from stage2_htr.collate import htr_collate_fn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ── helpers ────────────────────────────────────────────────────────────────

def _safe_getitem(ds: HTRLineDataset, idx: int) -> dict | None:
    """Attempt to load a sample; return None on failure."""
    try:
        return ds[idx]
    except Exception as e:
        logger.warning("Failed to load sample %d: %s", idx, e)
        return None


def _make_sample_grid(
    samples: list[dict],
    out_path: str,
    max_cols: int = 4,
    thumb_height: int = 48,
    label_height: int = 28,
    gap: int = 4,
) -> None:
    """Render a grid of preprocessed line images with numbered labels."""
    n = len(samples)
    cols = min(n, max_cols)
    rows = math.ceil(n / cols)

    # Determine per-cell width from actual sample widths
    max_thumb_w = max(s["image"].shape[2] for s in samples)
    # Cap thumbnail width for readability
    max_thumb_w = min(max_thumb_w, 800)

    cell_w = max_thumb_w + 2 * gap
    cell_h = thumb_height + label_height + 3 * gap
    grid_w = cols * cell_w
    grid_h = rows * cell_h

    canvas = Image.new("L", (grid_w, grid_h), 255)
    draw = ImageDraw.Draw(canvas)

    for i, s in enumerate(samples):
        col = i % cols
        row = i // cols
        x0 = col * cell_w + gap
        y0 = row * cell_h + gap

        # Thumbnail (resize width to fit cell)
        img_t = s["image"][0].numpy()  # [H, W]
        img_t = (img_t * 255).clip(0, 255).astype(np.uint8)
        thumb = Image.fromarray(img_t, mode="L")
        tw = min(thumb.width, max_thumb_w)
        th = thumb_height
        thumb = thumb.resize((tw, th), Image.LANCZOS)
        canvas.paste(thumb, (x0, y0))

        # Label index text below the image
        label_text = f"[{i+1}]"
        draw.text((x0, y0 + th + gap), label_text, fill=0)

    canvas.save(out_path)
    logger.info("Saved sample grid to %s", out_path)


# ── full dry-run ───────────────────────────────────────────────────────────

def _run_check_all(manifest_csv: str, charset_json: str) -> dict:
    """Iterate every row: open image, encode/decode roundtrip.

    Returns a summary dict.
    """
    logger.info("Running --check-all on %s …", manifest_csv)
    with open(charset_json, "r", encoding="utf-8") as f:
        charset = json.load(f)
    char_to_id = charset["char_to_id"]
    id_to_char = {int(k): v for k, v in charset["id_to_char"].items()}

    total = 0
    img_open_fail = 0
    encode_fail = 0
    decode_mismatch = 0
    blank_text = 0
    ascii_digit_rows = 0
    latin_rows = 0
    widths = []
    heights = []

    with open(manifest_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            text = row.get("text", "")
            path = row.get("image_path", "")

            if not text:
                blank_text += 1
                continue

            # Image open check
            try:
                img = Image.open(path)
                w, h = img.size
                widths.append(w)
                heights.append(h)
                img.close()
            except Exception:
                img_open_fail += 1

            # Encode-decode roundtrip
            try:
                ids = encode_text(text, char_to_id)
                decoded = decode_ids(ids, id_to_char)
                if decoded != text:
                    decode_mismatch += 1
            except Exception:
                encode_fail += 1

            if any("0" <= c <= "9" for c in text):
                ascii_digit_rows += 1
            if any(("a" <= c <= "z") or ("A" <= c <= "Z") for c in text):
                latin_rows += 1

            if total % 20000 == 0:
                logger.info("  … checked %d rows", total)

    return {
        "total_rows": total,
        "image_open_failures": img_open_fail,
        "encoding_errors": encode_fail,
        "decode_mismatches": decode_mismatch,
        "blank_text_rows": blank_text,
        "ascii_digit_rows": ascii_digit_rows,
        "latin_letter_rows": latin_rows,
        "original_width_min": min(widths) if widths else 0,
        "original_width_max": max(widths) if widths else 0,
        "original_width_mean": round(sum(widths) / len(widths), 2) if widths else 0,
        "original_height_min": min(heights) if heights else 0,
        "original_height_max": max(heights) if heights else 0,
        "original_height_mean": round(sum(heights) / len(heights), 2) if heights else 0,
    }


# ── main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MANTRA Stage 2.3 — Dataset Loader Inspection & QA"
    )
    parser.add_argument(
        "--manifest",
        default="data/stage2_htr/manifests/htr_train.csv",
        help="Path to HTR manifest CSV",
    )
    parser.add_argument(
        "--charset",
        default="data/stage2_htr/vocab/charset_synth100k.json",
        help="Path to charset JSON",
    )
    parser.add_argument(
        "--num-samples", type=int, default=32,
        help="Number of random samples to inspect",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Batch size for collate test",
    )
    parser.add_argument(
        "--check-all", action="store_true",
        help="Full dry-run: open every image & roundtrip every label",
    )
    parser.add_argument(
        "--report-dir",
        default="data/stage2_htr/reports",
        help="Directory for QA reports",
    )
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    os.makedirs(args.report_dir, exist_ok=True)

    print("=" * 80)
    print("MANTRA STAGE 2.3: DATASET LOADER INSPECTION")
    print("=" * 80)

    # ── full dry-run path ──────────────────────────────────────────────
    if args.check_all:
        t0 = time.time()
        check_all_result = _run_check_all(args.manifest, args.charset)
        elapsed = time.time() - t0

        all_pass = (
            check_all_result["image_open_failures"] == 0
            and check_all_result["encoding_errors"] == 0
            and check_all_result["decode_mismatches"] == 0
            and check_all_result["blank_text_rows"] == 0
            and check_all_result["ascii_digit_rows"] == 0
            and check_all_result["latin_letter_rows"] == 0
        )
        check_all_result["qa_status"] = "PASS" if all_pass else "FAIL"
        check_all_result["elapsed_seconds"] = round(elapsed, 2)

        print(f"\n--check-all completed in {elapsed:.1f}s")
        for k, v in check_all_result.items():
            print(f"  {k}: {v}")

        # Save JSON
        json_path = os.path.join(args.report_dir, "dataset_loader_check_all.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(check_all_result, f, ensure_ascii=False, indent=2)
        print(f"Saved check-all JSON to {json_path}")

        if not all_pass:
            sys.exit(1)
        return

    # ── sample inspection path ─────────────────────────────────────────
    print(f"Manifest : {args.manifest}")
    print(f"Charset  : {args.charset}")
    print(f"Samples  : {args.num_samples}")
    print(f"Batch    : {args.batch_size}")

    ds = HTRLineDataset(
        manifest_csv=args.manifest,
        charset_json=args.charset,
        image_height=64,
        max_width=None,
        augment=False,
        return_metadata=True,
    )
    total_rows = len(ds)
    print(f"Dataset size: {total_rows}")

    # Sample indices
    import random
    rng = random.Random(42)
    n = min(args.num_samples, total_rows)
    indices = rng.sample(range(total_rows), n)

    samples: list[dict] = []
    load_failures = 0
    encode_errors = 0
    decode_mismatches = 0

    for idx in indices:
        s = _safe_getitem(ds, idx)
        if s is None:
            load_failures += 1
            continue

        # Verify encode-decode roundtrip
        try:
            decoded = ds.decode(s["label"].tolist())
            if decoded != s["text"]:
                decode_mismatches += 1
        except Exception:
            encode_errors += 1

        samples.append(s)

    print(f"Loaded {len(samples)}/{n} samples  "
          f"(failures={load_failures}, enc_errors={encode_errors}, "
          f"decode_mismatches={decode_mismatches})")

    # ── shape & sanity checks ──────────────────────────────────────────
    shape_ok = True
    nan_inf_count = 0
    empty_label_count = 0
    processed_widths = []
    processed_heights = []
    orig_widths = []
    orig_heights = []
    label_lengths = []
    domain_counts = Counter()

    for s in samples:
        img = s["image"]
        c, h, w = img.shape
        if c != 1 or h != 64:
            shape_ok = False
        if not torch.isfinite(img).all():
            nan_inf_count += 1
        if s["label_length"] == 0:
            empty_label_count += 1

        processed_widths.append(s["processed_width"])
        processed_heights.append(s["processed_height"])
        orig_widths.append(s["width"])
        orig_heights.append(s["height"])
        label_lengths.append(s["label_length"])
        domain_counts[s.get("domain", "unknown")] += 1

    # ── batch test ─────────────────────────────────────────────────────
    bs = min(args.batch_size, len(samples))
    batch = htr_collate_fn(samples[:bs])
    batch_img_shape = list(batch["images"].shape)
    batch_labels_len = int(batch["labels"].shape[0])
    batch_ll = batch["label_lengths"].tolist()
    batch_pw = batch["processed_widths"].tolist()

    print(f"Batch images shape : {batch_img_shape}")
    print(f"Batch labels flat  : {batch_labels_len}")

    # ── DataLoader smoke test ──────────────────────────────────────────
    subset = Subset(ds, indices[:bs])
    loader = DataLoader(subset, batch_size=bs, collate_fn=htr_collate_fn)
    loader_batch = next(iter(loader))
    loader_ok = (loader_batch["images"].shape[0] == bs)
    print(f"DataLoader batch OK: {loader_ok}")

    # ── sample grid ────────────────────────────────────────────────────
    grid_path = os.path.join(args.report_dir, "preprocessed_samples_grid.png")
    grid_samples = samples[:min(16, len(samples))]
    _make_sample_grid(grid_samples, grid_path)

    # ── QA verdict ─────────────────────────────────────────────────────
    qa_passed = (
        load_failures == 0
        and encode_errors == 0
        and decode_mismatches == 0
        and empty_label_count == 0
        and nan_inf_count == 0
        and shape_ok
        and loader_ok
    )
    qa_status = "PASS" if qa_passed else "FAIL"
    print(f"\nQA Status: {qa_status}")

    # ── build report data ──────────────────────────────────────────────
    report = {
        "qa_status": qa_status,
        "manifest": args.manifest,
        "charset": args.charset,
        "total_manifest_rows": total_rows,
        "samples_requested": n,
        "samples_loaded": len(samples),
        "load_failures": load_failures,
        "encoding_errors": encode_errors,
        "decode_mismatches": decode_mismatches,
        "empty_labels": empty_label_count,
        "nan_inf_tensors": nan_inf_count,
        "shape_check_passed": shape_ok,
        "dataloader_ok": loader_ok,
        "fixed_processed_height": 64,
        "original_width_min": min(orig_widths) if orig_widths else 0,
        "original_width_max": max(orig_widths) if orig_widths else 0,
        "original_width_mean": round(sum(orig_widths) / len(orig_widths), 2) if orig_widths else 0,
        "original_height_min": min(orig_heights) if orig_heights else 0,
        "original_height_max": max(orig_heights) if orig_heights else 0,
        "original_height_mean": round(sum(orig_heights) / len(orig_heights), 2) if orig_heights else 0,
        "processed_width_min": min(processed_widths) if processed_widths else 0,
        "processed_width_max": max(processed_widths) if processed_widths else 0,
        "processed_width_mean": round(sum(processed_widths) / len(processed_widths), 2) if processed_widths else 0,
        "label_length_min": min(label_lengths) if label_lengths else 0,
        "label_length_max": max(label_lengths) if label_lengths else 0,
        "label_length_mean": round(sum(label_lengths) / len(label_lengths), 2) if label_lengths else 0,
        "batch_image_shape_example": batch_img_shape,
        "domain_sample_counts": dict(domain_counts),
        "sample_details": [],
    }

    # Warnings
    warnings = []
    if report["label_length_max"] > 120:
        warnings.append(f"Very long labels detected (max={report['label_length_max']})")
    if report["processed_width_max"] > 2000:
        warnings.append(f"Very wide processed images (max={report['processed_width_max']}px)")
    report["warnings"] = warnings

    for i, s in enumerate(samples[:8]):
        report["sample_details"].append({
            "index": i + 1,
            "image_path": s["image_path"],
            "text": s["text"],
            "label_length": s["label_length"],
            "original_size": f"{s['width']}x{s['height']}",
            "processed_size": f"{s['processed_width']}x{s['processed_height']}",
            "decoded": ds.decode(s["label"].tolist()),
        })

    # ── save JSON report ───────────────────────────────────────────────
    json_path = os.path.join(args.report_dir, "dataset_loader_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Saved JSON report to {json_path}")

    # ── save markdown report ───────────────────────────────────────────
    md_path = os.path.join(args.report_dir, "dataset_loader_report.md")
    with open(md_path, "w", encoding="utf-8-sig") as f:
        f.write("# MANTRA Stage 2.3 — Dataset Loader QA Report\n\n")

        f.write("## Executive Summary\n")
        f.write(f"- **QA Status:** {qa_status}\n")
        f.write(f"- **Manifest:** `{args.manifest}`\n")
        f.write(f"- **Total Rows:** {total_rows}\n")
        f.write(f"- **Samples Tested:** {len(samples)}\n\n")

        f.write("## QA Validation Table\n")
        f.write("| Check | Expected | Observed | Status |\n")
        f.write("| --- | --- | --- | --- |\n")
        f.write(f"| Load Failures | 0 | {load_failures} | {'PASS' if load_failures == 0 else 'FAIL'} |\n")
        f.write(f"| Encoding Errors | 0 | {encode_errors} | {'PASS' if encode_errors == 0 else 'FAIL'} |\n")
        f.write(f"| Decode Mismatches | 0 | {decode_mismatches} | {'PASS' if decode_mismatches == 0 else 'FAIL'} |\n")
        f.write(f"| Empty Labels | 0 | {empty_label_count} | {'PASS' if empty_label_count == 0 else 'FAIL'} |\n")
        f.write(f"| NaN/Inf Tensors | 0 | {nan_inf_count} | {'PASS' if nan_inf_count == 0 else 'FAIL'} |\n")
        f.write(f"| Shape [1, 64, W] | Yes | {'Yes' if shape_ok else 'No'} | {'PASS' if shape_ok else 'FAIL'} |\n")
        f.write(f"| DataLoader Batch | OK | {'OK' if loader_ok else 'FAIL'} | {'PASS' if loader_ok else 'FAIL'} |\n\n")

        f.write("## Image Dimension Statistics\n")
        f.write("| Metric | Original | Processed |\n")
        f.write("| --- | --- | --- |\n")
        f.write(f"| Width Min | {report['original_width_min']} | {report['processed_width_min']} |\n")
        f.write(f"| Width Max | {report['original_width_max']} | {report['processed_width_max']} |\n")
        f.write(f"| Width Mean | {report['original_width_mean']} | {report['processed_width_mean']} |\n")
        f.write(f"| Height Min | {report['original_height_min']} | 64 |\n")
        f.write(f"| Height Max | {report['original_height_max']} | 64 |\n")
        f.write(f"| Height Mean | {report['original_height_mean']} | 64 |\n\n")

        f.write("## Label Length Statistics\n")
        f.write(f"- Min: {report['label_length_min']}\n")
        f.write(f"- Max: {report['label_length_max']}\n")
        f.write(f"- Mean: {report['label_length_mean']}\n\n")

        f.write("## Batch Tensor Shape Example\n")
        f.write(f"- Images: `{batch_img_shape}`\n")
        f.write(f"- Labels flat length: `{batch_labels_len}`\n")
        f.write(f"- Label lengths: `{batch_ll}`\n")
        f.write(f"- Processed widths: `{batch_pw}`\n\n")

        f.write("## Domain Sample Counts\n")
        f.write("| Domain | Count |\n")
        f.write("| --- | --- |\n")
        for d, c in sorted(domain_counts.items()):
            f.write(f"| {d} | {c} |\n")
        f.write("\n")

        if warnings:
            f.write("## Warnings\n")
            for w in warnings:
                f.write(f"> [!WARNING]\n> {w}\n\n")

        f.write("## Sample Details\n")
        f.write("| # | Image Path | Text | Label Len | Original | Processed |\n")
        f.write("| --- | --- | --- | --- | --- | --- |\n")
        for sd in report["sample_details"]:
            short_path = sd["image_path"].split("/")[-1]
            short_text = sd["text"][:40] + ("…" if len(sd["text"]) > 40 else "")
            f.write(f"| {sd['index']} | {short_path} | {short_text} | {sd['label_length']} | {sd['original_size']} | {sd['processed_size']} |\n")
        f.write("\n")

    print(f"Saved markdown report to {md_path}")
    print("=" * 80)
    print("DATASET INSPECTION FINISHED")
    print("=" * 80)

    if not qa_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
