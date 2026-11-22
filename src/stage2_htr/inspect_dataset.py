"""Inspect the real HTR preprocessing, label, and batching pipeline."""
import argparse
import copy
import json
import logging
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage2_htr.collate import htr_collate_fn
from stage2_htr.config import DEFAULT_CONFIG, dataset_options, load_data_config, make_dataloader
from stage2_htr.dataset import HTRLineDataset, encode_text

logger = logging.getLogger(__name__)


def _make_sample_grid(samples, out_path, max_cols=4):
    if not samples:
        return
    cols = min(len(samples), max_cols)
    cell_w, cell_h = 808, 80
    canvas = Image.new("L", (cols * cell_w, math.ceil(len(samples) / cols) * cell_h), 255)
    draw = ImageDraw.Draw(canvas)
    for index, sample in enumerate(samples):
        pixels = (sample["image"][0].numpy() * 255).clip(0, 255).astype(np.uint8)
        thumb = Image.fromarray(pixels)
        thumb.thumbnail((800, 48), Image.Resampling.LANCZOS)
        x, y = (index % cols) * cell_w + 4, (index // cols) * cell_h + 4
        canvas.paste(thumb, (x, y))
        draw.text((x, y + 52), f"[{index + 1}]", fill=0)
    canvas.save(out_path)


def inspect_dataset(ds, indices, batch_size=8):
    """Check every selected row and batch; retain only a few preview tensors."""
    result = {
        "total_rows": ds.source_row_count, "rows_checked": len(indices),
        "samples_loaded": 0, "image_open_failures": 0, "encoding_errors": 0,
        "decode_mismatches": 0, "blank_text_rows": 0, "ascii_digit_rows": 0,
        "latin_letter_rows": 0, "nan_inf_tensors": 0, "pixel_range_errors": 0,
        "shape_check_passed": True, "batching_errors": 0, "batches_checked": 0,
        "skipped_width_rows": len(ds.skipped_rows),
        "skipped_width_samples": ds.skipped_rows[:50],
        "fixed_processed_height": ds.image_height, "invert_image": ds.invert_image,
        "errors": [],
    }
    dimensions = {name: [] for name in (
        "original_width", "original_height", "processed_width", "label_length"
    )}
    domain_counts = {}
    previews, valid_indices, pending = [], [], []
    pad_value = 0.0 if ds.invert_image else 1.0

    def record_error(index, kind, error):
        if len(result["errors"]) < 50:
            result["errors"].append({
                "index": index, "image_path": ds.rows[index].get("image_path", ""),
                "kind": kind, "message": str(error),
            })

    def check_batch():
        if not pending:
            return
        try:
            batch = htr_collate_fn(pending, pad_value=pad_value)
            if batch["labels"].numel() != int(batch["label_lengths"].sum()):
                raise ValueError("Concatenated label lengths do not match")
            result["batch_image_shape_example"] = list(batch["images"].shape)
            result["batches_checked"] += 1
        except Exception as error:
            result["batching_errors"] += 1
            record_error(pending[0]["_index"], "batch", error)
        pending.clear()

    for count, index in enumerate(indices, 1):
        text = ds.rows[index].get("text", "")
        result["blank_text_rows"] += int(not text.strip())
        result["ascii_digit_rows"] += int(any("0" <= c <= "9" for c in text))
        result["latin_letter_rows"] += int(any("a" <= c.lower() <= "z" for c in text))
        try:
            encode_text(text, ds.char_to_id)
        except ValueError as error:
            result["encoding_errors"] += 1
            record_error(index, "encoding", error)
            continue
        try:
            sample = ds[index]
        except Exception as error:
            result["image_open_failures"] += 1
            record_error(index, "image", error)
            continue
        sample["_index"] = index
        result["samples_loaded"] += 1
        image = sample["image"]
        result["shape_check_passed"] &= (
            image.ndim == 3 and image.shape[:2] == (1, ds.image_height)
            and image.shape[2] > 0
        )
        result["nan_inf_tensors"] += int(not torch.isfinite(image).all())
        result["pixel_range_errors"] += int(not ((image >= 0).all() and (image <= 1).all()))
        result["decode_mismatches"] += int(ds.decode(sample["label"].tolist()) != text)
        for name, key in (("original_width", "width"), ("original_height", "height"),
                          ("processed_width", "processed_width"), ("label_length", "label_length")):
            dimensions[name].append(sample[key])
        domain = sample.get("domain", "unknown")
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
        if len(previews) < 16:
            previews.append(sample)
        if len(valid_indices) < batch_size:
            valid_indices.append(index)
        pending.append(sample)
        if len(pending) == batch_size:
            check_batch()
        if count % 10000 == 0:
            logger.info("Checked %d/%d rows", count, len(indices))
    check_batch()
    for name, values in dimensions.items():
        result[name + "_min"] = min(values, default=0)
        result[name + "_max"] = max(values, default=0)
        result[name + "_mean"] = round(sum(values) / len(values), 2) if values else 0
    result["domain_sample_counts"] = domain_counts
    result["load_failures"] = result["image_open_failures"]
    result["empty_labels"] = result["blank_text_rows"]
    result["warnings"] = []
    if result["label_length_max"] > 120:
        result["warnings"].append("Labels longer than 120 codepoints were found")
    if result["processed_width_max"] > 2000:
        result["warnings"].append("Processed images wider than 2000 pixels were found")
    result["sample_details"] = [{
        "index": sample["_index"], "image_path": sample["image_path"],
        "text": sample["text"], "label_length": sample["label_length"],
        "original_size": f"{sample['width']}x{sample['height']}",
        "processed_size": f"{sample['processed_width']}x{sample['processed_height']}",
        "decoded": ds.decode(sample["label"].tolist()),
    } for sample in previews[:8]]
    return result, previews, valid_indices


def _passed(result):
    return (
        result["rows_checked"] > 0 and result["samples_loaded"] > 0
        and result["shape_check_passed"]
        and all(result[key] == 0 for key in (
            "image_open_failures", "encoding_errors", "decode_mismatches",
            "blank_text_rows", "ascii_digit_rows", "latin_letter_rows",
            "nan_inf_tensors", "pixel_range_errors", "batching_errors",
        ))
    )


def _run_check_all(manifest_csv, charset_json, **options):
    """Use the actual dataset and collate for the complete manifest."""
    ds = HTRLineDataset(manifest_csv, charset_json, **options)
    result, _, _ = inspect_dataset(ds, range(len(ds)))
    result["qa_status"] = "PASS" if _passed(result) else "FAIL"
    return result


def _write_reports(result, report_dir, check_all):
    report_dir.mkdir(parents=True, exist_ok=True)
    stem = "dataset_loader_check_all" if check_all else "dataset_loader_report"
    (report_dir / (stem + ".json")).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = ["# Dataset loader QA report", "", f"QA status: **{result['qa_status']}**", "",
             "| Check | Result |", "| --- | --- |"]
    for key, value in result.items():
        if isinstance(value, (str, int, float, bool)):
            lines.append(f"| {key} | {value} |")
    lines.extend(["", "## Failures", ""])
    lines.extend(f"- {error}" for error in result.get("errors", []))
    (report_dir / (stem + ".md")).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--manifest", help="Overrides configured train manifest (all for --check-all)")
    parser.add_argument("--charset", help="Overrides paths.charset_json")
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, help="Overrides dataloader.batch_size")
    parser.add_argument("--check-all", action="store_true")
    parser.add_argument("--report-dir", help="Defaults to the configured release's reports directory")
    args = parser.parse_args(argv)
    if args.num_samples <= 0 or (args.batch_size is not None and args.batch_size <= 0):
        parser.error("num-samples and batch-size must be positive")
    report_dir = Path(args.report_dir or "data/stage2_htr/reports")
    # Inspection is sequential per image; large thread pools add overhead for
    # these small CPU tensors and compete with DataLoader worker processes.
    torch.set_num_threads(1)
    t0 = time.time()
    result = {"qa_status": "FAIL", "errors": []}
    previews = []
    try:
        config = load_data_config(args.config)
        if args.report_dir is None:
            report_dir = Path(config["paths"]["manifest_all"]).parent.parent / "reports"
        manifest = args.manifest or config["paths"]["manifest_all" if args.check_all else "manifest_train"]
        charset = args.charset or config["paths"]["charset_json"]
        bs = args.batch_size or config["dataloader"]["batch_size"]
        ds = HTRLineDataset(manifest, charset, **dataset_options(config))
        indices = range(len(ds)) if args.check_all else random.Random(42).sample(
            range(len(ds)), min(args.num_samples, len(ds))
        )
        result, previews, valid_indices = inspect_dataset(ds, indices, batch_size=bs)
        result.update(manifest=str(manifest), charset=str(charset),
                      total_manifest_rows=ds.source_row_count, samples_requested=len(indices))
        result["dataloader_ok"] = False
        if valid_indices:
            try:
                # Windows spawns workers and pickles the dataset. Send only
                # the smoke-test rows rather than copying a 100k-row manifest
                # into every worker just to load one small batch.
                subset = copy.copy(ds)
                subset.rows = [ds.rows[index] for index in valid_indices]
                subset.source_row_count = len(subset.rows)
                subset.skipped_rows = []
                loader = make_dataloader(
                    subset, config, batch_size=min(bs, len(valid_indices)),
                )
                batch = next(iter(loader))
                result["dataloader_ok"] = batch["images"].shape[0] == len(valid_indices)
            except Exception as error:
                result["errors"].append({"kind": "dataloader", "message": str(error)})
        result["qa_status"] = "PASS" if _passed(result) and result["dataloader_ok"] else "FAIL"
    except Exception as error:
        result["errors"].append({"kind": "setup", "message": str(error)})
        logger.exception("Dataset inspection failed")
    result["elapsed_seconds"] = round(time.time() - t0, 2)
    _write_reports(result, report_dir, args.check_all)
    grid = report_dir / "preprocessed_samples_grid.png"
    if previews:
        _make_sample_grid(previews, grid)
    elif grid.exists():
        grid.unlink()
    print(f"QA status: {result['qa_status']}; reports: {report_dir}")
    return 0 if result["qa_status"] == "PASS" else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    raise SystemExit(main())
