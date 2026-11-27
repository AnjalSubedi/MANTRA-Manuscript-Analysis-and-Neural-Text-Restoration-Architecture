"""Fit a fixed batch on CPU and verify its predictions from a saved checkpoint."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import torch
from torch.utils.data import Subset

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stage2_htr.config import DEFAULT_CONFIG, dataset_options, load_data_config, make_dataloader
from stage2_htr.ctc import ctc_loss, greedy_decode
from stage2_htr.dataset import HTRLineDataset
from stage2_htr.metrics import character_error_rate
from stage2_htr.model import CRNN


def load_batch(config_path, indices):
    config = load_data_config(config_path)
    manifest = config["paths"]["manifest_train"]
    dataset = HTRLineDataset(manifest, config["paths"]["charset_json"], **dataset_options(config))
    if not indices or len(set(indices)) != len(indices) or not all(0 <= i < len(dataset) for i in indices):
        raise ValueError("indices must be distinct zero-based positions in the training dataset")
    subset = Subset(dataset, indices)
    batch = next(iter(make_dataloader(subset, config, batch_size=len(indices), num_workers=0)))
    identity = {
        "manifest_train": manifest,
        "indices": indices,
        "line_ids": batch["line_ids"],
        "texts": batch["texts"],
        "resized_widths": batch["processed_widths"].tolist(),
        "pixels_sha256": hashlib.sha256(batch["images"].numpy().tobytes()).hexdigest(),
        "id_to_char": dataset.id_to_char,
    }
    return dataset, batch, identity


@torch.no_grad()
def evaluate(model, batch, id_to_char):
    model.eval()
    log_probs, lengths = model(batch["images"], batch["processed_widths"])
    loss = ctc_loss(log_probs, batch["labels"], lengths, batch["label_lengths"])
    predictions = greedy_decode(log_probs, lengths, id_to_char)
    return loss.item(), character_error_rate(predictions, batch["texts"]), predictions


def train(args):
    torch.manual_seed(args.seed)
    dataset, batch, identity = load_batch(args.config, args.indices)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    model_options = {"num_classes": dataset.num_classes, "image_height": dataset.image_height}
    model = CRNN(**model_options)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    initial_loss, initial_cer, predictions = evaluate(model, batch, dataset.id_to_char)
    history = [{"step": 0, "ctc_loss": initial_loss, "cer": initial_cer}]
    print(f"Step 0: CTC loss {initial_loss:.4f}, CER {initial_cer:.2%}", flush=True)

    for step in range(1, args.max_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        log_probs, lengths = model(batch["images"], batch["processed_widths"])
        loss = ctc_loss(log_probs, batch["labels"], lengths, batch["label_lengths"])
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite CTC loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, error_if_nonfinite=True)
        optimizer.step()

        if step % args.eval_every == 0 or step == args.max_steps:
            eval_loss, cer, predictions = evaluate(model, batch, dataset.id_to_char)
            history.append({"step": step, "ctc_loss": eval_loss, "cer": cer})
            print(f"Step {step}: CTC loss {eval_loss:.4f}, CER {cer:.2%}", flush=True)
            if cer == 0 and eval_loss < initial_loss:
                break

    passed = history[-1]["cer"] == 0 and history[-1]["ctc_loss"] < initial_loss
    checkpoint = args.output_dir / "checkpoint.pt"
    temporary = checkpoint.with_suffix(".pt.part")
    torch.save({
        "model_options": model_options,
        "model_state_dict": model.state_dict(),
        "training_batch": identity,
        "predictions": predictions,
    }, temporary)
    temporary.replace(checkpoint)
    report = {
        "status": "PASS" if passed else "FAIL",
        "config": str(args.config.resolve()),
        "manifest_train": identity["manifest_train"],
        "seed": args.seed,
        "cpu_threads": args.threads,
        "learning_rate": args.learning_rate,
        "max_steps": args.max_steps,
        "eval_every": args.eval_every,
        "sample_indices": args.indices,
        "resized_widths": identity["resized_widths"],
        "history": history,
        "samples": [
            {"line_id": line_id, "reference": text, "prediction": prediction}
            for line_id, text, prediction in zip(batch["line_ids"], batch["texts"], predictions)
        ],
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(f"{report['status']}: saved {checkpoint}", flush=True)
    return 0 if passed else 1


def verify(args):
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    expected = checkpoint["training_batch"]
    dataset, batch, identity = load_batch(args.config, expected["indices"])
    if identity != expected:
        raise ValueError("The current batch or charset differs from the checkpoint")
    model = CRNN(**checkpoint["model_options"])
    model.load_state_dict(checkpoint["model_state_dict"])
    loss, cer, predictions = evaluate(model, batch, dataset.id_to_char)
    if predictions != checkpoint["predictions"]:
        raise ValueError("Reloaded model predictions differ from the saved predictions")
    print(f"Reload verified: {len(predictions)} matching predictions, CTC loss {loss:.4f}, CER {cer:.2%}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--threads", type=int, default=2, help="Number of CPU threads")
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("train", help="Fit a new model on one fixed training batch")
    fit.add_argument("--indices", type=int, nargs="+", required=True)
    fit.add_argument("--output-dir", type=Path, required=True)
    fit.add_argument("--max-steps", type=int, default=500)
    fit.add_argument("--eval-every", type=int, default=25)
    fit.add_argument("--learning-rate", type=float, default=0.002)
    fit.add_argument("--seed", type=int, default=42)
    reload = commands.add_parser("verify", help="Reload a checkpoint and compare its predictions")
    reload.add_argument("checkpoint", type=Path)
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.command == "train" and (
        min(args.max_steps, args.eval_every) < 1
        or not math.isfinite(args.learning_rate) or args.learning_rate <= 0
    ):
        parser.error("steps and learning rate must be finite and positive")
    torch.set_num_threads(args.threads)
    try:
        return train(args) if args.command == "train" else verify(args)
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"Tiny-batch run failed: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
