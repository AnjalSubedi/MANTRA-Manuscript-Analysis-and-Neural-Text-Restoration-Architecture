"""Score a trained MANTRA line recognizer and save its transcriptions."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
from uuid import uuid4

import torch
from torch.utils.data import Subset

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stage2_htr.config import DEFAULT_CONFIG, PROJECT_ROOT, dataset_options, load_data_config, make_dataloader
from stage2_htr.ctc import greedy_decode
from stage2_htr.dataset import HTRLineDataset
from stage2_htr.metrics import edit_distance
from stage2_htr.model import CRNN
from stage2_htr.train import CHECKPOINT_VERSION, bucket_batches, file_sha256, resized_width_hint, write_metrics


@torch.inference_mode()
def predict(model, loader, id_to_char, device, stream):
    vocabulary = set(id_to_char.values())
    unsupported = Counter()
    lines = scored = edits = reference_characters = exact_matches = blank_predictions = 0
    next_update = 1000
    for batch in loader:
        images = batch["images"].to(device, non_blocking=device.type == "cuda")
        log_probs, lengths = model(images, batch["processed_widths"])
        predictions = greedy_decode(log_probs, lengths, id_to_char)
        for line_id, path, reference, prediction in zip(
            batch["line_ids"], batch["image_paths"], batch["texts"], predictions, strict=True,
        ):
            record = {"line_id": line_id, "image_path": path, "prediction": prediction,
                      "reference": None, "character_edits": None}
            lines += 1
            blank_predictions += not prediction
            if reference.strip():
                distance = edit_distance(prediction, reference)
                record.update(reference=reference, character_edits=distance)
                scored += 1
                edits += distance
                reference_characters += len(reference)
                exact_matches += distance == 0
                unsupported.update(char for char in reference if char not in vocabulary)
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        if lines >= next_update:
            print(f"Evaluated {lines:,}/{len(loader.dataset):,} lines", flush=True)
            next_update += 1000
    return {
        "lines": lines, "scored_lines": scored, "character_edits": edits,
        "reference_characters": reference_characters,
        "cer": edits / reference_characters if reference_characters else None,
        "exact_matches": exact_matches, "exact_line_rate": exact_matches / scored if scored else None,
        "blank_predictions": blank_predictions, "out_of_vocabulary_reference_characters": dict(unsupported),
    }


def evaluate(args):
    config = load_data_config(args.data_config, project_root=args.data_root)
    paths = config["paths"]
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != CHECKPOINT_VERSION or "id_to_char" not in checkpoint:
        raise ValueError("Expected a best.pt checkpoint from the CRNN–CTC trainer")
    options = dataset_options(config)
    if options != checkpoint["data_options"]:
        raise ValueError("Input preprocessing differs from the checkpoint")

    synthetic_test = args.manifest is None
    manifest = Path(paths["manifest_test"]) if synthetic_test else args.manifest
    checked_files = ("charset_json", "manifest_train", "manifest_val") if synthetic_test else ("charset_json",)
    for name in checked_files:
        if file_sha256(paths[name]) != checkpoint["data_sha256"].get(name):
            raise ValueError(f"{name} differs from the data recorded in the checkpoint")
    dataset = HTRLineDataset(manifest, paths["charset_json"], project_root=args.data_root,
                             encode_labels=False, **options)
    if dataset.id_to_char != checkpoint["id_to_char"] or dataset.num_classes != checkpoint["model_options"]["num_classes"]:
        raise ValueError("Dataset vocabulary differs from the checkpoint")
    manifest_lines = len(dataset)
    if not manifest_lines or dataset.skipped_rows:
        raise ValueError("Evaluation requires a nonempty manifest with all lines within the configured width limit")
    if synthetic_test and any(row.get("split") != "test" or not row["text"].strip() for row in dataset.rows):
        raise ValueError("The held-out test manifest must contain only test rows with transcriptions")

    batch_size = args.batch_size or config["dataloader"]["batch_size"]
    line_count = min(args.limit or manifest_lines, manifest_lines)
    widths = [resized_width_hint(row, dataset.image_height) for row in dataset.rows[:line_count]]
    evaluation_set = Subset(dataset, range(line_count)) if line_count < manifest_lines else dataset
    batches = bucket_batches(widths, batch_size, batch_size * 32)
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    model = CRNN(**checkpoint["model_options"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    loader_options = {} if args.workers is None else {"num_workers": args.workers}
    loader = make_dataloader(evaluation_set, config, batch_size=1, batch_sampler=batches,
                             pin_memory=device.type == "cuda", **loader_options)
    report = {
        "evaluation": "synthetic_test" if synthetic_test else "external",
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": file_sha256(args.checkpoint),
        "checkpoint_epoch": checkpoint["epoch"], "checkpoint_validation_cer": checkpoint["validation_cer"],
        "manifest": str(manifest.resolve()), "manifest_sha256": file_sha256(manifest),
        "manifest_lines": manifest_lines, "decoder": "greedy_ctc", "device": str(device),
        "torch_version": str(torch.__version__), "batch_size": batch_size,
    }
    print(f"Evaluating {line_count:,} lines from {manifest} using epoch {checkpoint['epoch']} on {device}", flush=True)
    started = time.monotonic()
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = args.output_dir.with_name(f".{args.output_dir.name}-{uuid4().hex}.tmp")
    staging.mkdir()
    try:
        with (staging / "predictions.jsonl").open("w", encoding="utf-8") as stream:
            report.update(predict(model, loader, checkpoint["id_to_char"], device, stream))
        report["seconds"] = round(time.monotonic() - started, 3)
        write_metrics(staging / "metrics.json", report)
        staging.rename(args.output_dir)
    finally:
        if staging.exists():
            for name in ("predictions.jsonl", "metrics.json", "metrics.json.part"):
                (staging / name).unlink(missing_ok=True)
            staging.rmdir()
    cer = report["cer"]
    summary = f"CER {cer:.2%} on {report['scored_lines']:,} scored lines" if cer is not None else "no transcribed lines"
    print(f"Evaluated {report['lines']:,} lines, {summary}. Outputs saved to {args.output_dir}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="Use a new directory for each evaluation")
    parser.add_argument("--data-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--manifest", type=Path, help="External CSV with image_path and text; defaults to the synthetic test split")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--limit", type=int, help="Use the first N lines for a smoke check")
    args = parser.parse_args(argv)
    if any(value is not None and value < 1 for value in (args.threads, args.batch_size, args.limit)):
        parser.error("threads, batch-size, and limit must be positive")
    if args.workers is not None and args.workers < 0:
        parser.error("workers must be nonnegative")
    if args.output_dir.exists():
        parser.error("output-dir already exists")
    torch.set_num_threads(args.threads)
    evaluate(args)


if __name__ == "__main__":
    main()
