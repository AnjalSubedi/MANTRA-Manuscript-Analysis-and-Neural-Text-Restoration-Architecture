"""Train and resume the MANTRA CRNN–CTC baseline on synthetic line images."""

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import torch
from torch.utils.data import Subset
import yaml

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stage2_htr.config import DEFAULT_CONFIG, PROJECT_ROOT, dataset_options, load_data_config, make_dataloader
from stage2_htr.ctc import ctc_loss, greedy_decode
from stage2_htr.dataset import HTRLineDataset
from stage2_htr.metrics import character_error_counts
from stage2_htr.model import CRNN

TRAIN_CONFIG = PROJECT_ROOT / "configs/stage2_crnn_ctc.yaml"
CHECKPOINT_VERSION = 1


@dataclass
class Progress:
    epoch: int = 1
    batches_done: int = 0
    step: int = 0
    lines_seen: int = 0
    loss_sum: float = 0.0


def training_settings(path, epochs=None):
    settings = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(settings, dict):
        raise ValueError("Training config must contain a YAML mapping")
    if epochs is not None:
        settings["epochs"] = epochs
    for name in ("epochs", "hidden_size", "width_bucket_batches", "checkpoint_every_steps", "log_every_steps"):
        if type(settings.get(name)) is not int or settings[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("learning_rate", "weight_decay", "gradient_clip"):
        value = settings.get(name)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (name != "weight_decay" and value == 0):
            raise ValueError(f"{name} must be finite and {'nonnegative' if name == 'weight_decay' else 'positive'}")
    if type(settings.get("seed")) is not int or not 0 <= settings["seed"] < 2 ** 32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    return settings


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resized_width_hint(row, height):
    try:
        return max(1, round(int(row["width"]) * (height / int(row["height"]))))
    except (KeyError, ValueError, TypeError, ZeroDivisionError):
        # Manifest dimensions guide grouping only. The loader measures the actual image.
        return 0


def bucket_batches(widths, batch_size, bucket_size, *, seed=None):
    """Group similar widths while keeping epoch ordering reproducible."""
    indices = list(range(len(widths)))
    rng = random.Random(seed)
    if seed is not None:
        rng.shuffle(indices)
    batches = []
    for start in range(0, len(indices), bucket_size):
        bucket = sorted(indices[start:start + bucket_size], key=widths.__getitem__)
        batches.extend(bucket[offset:offset + batch_size] for offset in range(0, len(bucket), batch_size))
    if seed is not None:
        rng.shuffle(batches)
    return batches


@contextmanager
def run_lock(directory):
    """Hold an OS lock so an interrupted process can release it automatically."""
    with (directory / ".train.lock").open("a+b") as stream:
        try:
            if os.name == "nt":
                import msvcrt

                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError(f"Another trainer is writing to {directory}") from error
        yield


def save_torch(path, payload):
    temporary = path.with_suffix(path.suffix + ".part")
    torch.save(payload, temporary)
    temporary.replace(path)


def write_metrics(path, payload):
    temporary = path.with_suffix(".json.part")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def forward_loss(model, batch, device):
    images = batch["images"].to(device, non_blocking=device.type == "cuda")
    log_probs, lengths = model(images, batch["processed_widths"])
    return ctc_loss(log_probs, batch["labels"], lengths, batch["label_lengths"]), log_probs, lengths


@torch.inference_mode()
def validate(model, loader, id_to_char, device):
    model.eval()
    loss_sum = lines = edits = reference_chars = 0
    for batch in loader:
        loss, log_probs, lengths = forward_loss(model, batch, device)
        predictions = greedy_decode(log_probs, lengths, id_to_char)
        batch_edits, batch_chars = character_error_counts(predictions, batch["texts"])
        edits += batch_edits
        reference_chars += batch_chars
        count = len(predictions)
        loss_sum += loss.item() * count
        lines += count
    if not (lines and reference_chars and math.isfinite(loss_sum)):
        raise RuntimeError("Validation requires readable labels and finite loss")
    return {"loss": loss_sum / lines, "cer": edits / reference_chars,
            "edits": edits, "reference_characters": reference_chars, "lines": lines}


def run_training(args, settings, data_config, device):
    paths = data_config["paths"]
    dataset_args = dataset_options(data_config)
    datasets = {
        split: HTRLineDataset(paths[f"manifest_{split}"], paths["charset_json"],
                              project_root=str(args.data_root), **dataset_args)
        for split in ("train", "val")
    }
    train_data = datasets["train"]
    val_data = datasets["val"]
    line_counts = {"train": min(len(train_data), args.train_limit or len(train_data)),
                   "val": min(len(val_data), args.val_limit or len(val_data))}
    if any(count < 1 for count in line_counts.values()):
        raise ValueError("Both training and validation splits must contain at least one line")
    batch_size = args.batch_size or data_config["dataloader"]["batch_size"]
    workers = args.workers if args.workers is not None else data_config["dataloader"]["num_workers"]
    bucket_size = settings["width_bucket_batches"] * batch_size
    widths = {
        split: [resized_width_hint(row, dataset.image_height) for row in dataset.rows[:line_counts[split]]]
        for split, dataset in datasets.items()
    }
    hashes = {
        name: file_sha256(paths[name])
        for name in ("manifest_train", "manifest_val", "charset_json")
    }
    model_options = {"num_classes": train_data.num_classes, "image_height": train_data.image_height,
                     "hidden_size": settings["hidden_size"]}
    contract = {
        "data_options": data_config["dataset"], "data_sha256": hashes, "line_counts": line_counts,
        "model_options": model_options, "batch_size": batch_size,
        **{key: settings[key] for key in ("seed", "learning_rate", "weight_decay", "gradient_clip", "width_bucket_batches")},
    }
    torch.manual_seed(settings["seed"])
    model = CRNN(**model_options).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
                                  weight_decay=settings["weight_decay"])
    progress, history, best = Progress(), [], None
    output = args.output_dir
    if args.resume:
        latest = output / "last.pt"
        if latest.exists() and not latest.samefile(args.resume):
            raise ValueError("Resume from this directory's last.pt or choose a new output directory")
        saved = torch.load(args.resume, map_location="cpu", weights_only=True)
        if saved["format_version"] != CHECKPOINT_VERSION or saved["contract"] != contract:
            raise ValueError("Checkpoint training settings or data fingerprints differ from this run")
        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        progress, history, best = Progress(**saved["progress"]), saved["history"], saved["best"]
        torch.set_rng_state(saved["torch_rng_state"])
        if device.type == "cuda" and saved["cuda_rng_state"] is not None:
            torch.cuda.set_rng_state(saved["cuda_rng_state"], device)
        print(f"Resumed epoch {progress.epoch}, batch {progress.batches_done}, step {progress.step}", flush=True)
    elif any((output / name).exists() for name in ("last.pt", "best.pt", "metrics.json")):
        raise FileExistsError("Run outputs already exist. Use --resume or a new output directory.")

    descriptor = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"Device: {descriptor}; train={line_counts['train']:,}; val={line_counts['val']:,}; batch={batch_size}",
          flush=True)
    started = time.monotonic()
    initial_step = progress.step
    val_batches = bucket_batches(widths["val"], batch_size, bucket_size)

    def loader(dataset, batches, seed):
        # Loader worker seeding must not change the model's dropout RNG on resume.
        return make_dataloader(dataset, data_config, batch_size=1, batch_sampler=batches,
                               num_workers=workers, pin_memory=device.type == "cuda",
                               generator=torch.Generator().manual_seed(seed))

    val_loader = loader(val_data, val_batches, settings["seed"])

    def time_exhausted():
        return args.time_limit_minutes is not None and time.monotonic() - started >= 60 * args.time_limit_minutes

    def budget_exhausted():
        return (args.max_steps is not None and progress.step >= args.max_steps) or time_exhausted()

    def publish_best():
        if best is not None:
            save_torch(output / "best.pt", {
                "format_version": CHECKPOINT_VERSION, "model_options": model_options,
                "data_options": dataset_args, "id_to_char": train_data.id_to_char,
                "data_sha256": hashes, **best,
            })

    def checkpoint(status, *, new_best=False):
        save_torch(output / "last.pt", {
            "format_version": CHECKPOINT_VERSION, "contract": contract,
            "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
            "progress": asdict(progress), "history": history, "best": best,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        })
        if new_best:
            publish_best()
        write_metrics(output / "metrics.json", {
            "status": status, "training": settings, "contract": contract,
            "data_paths": {name: paths[name] for name in hashes},
            "device": descriptor, "torch_version": str(torch.__version__),
            "code_revision": os.environ.get("MANTRA_CODE_REVISION"),
            "num_workers": workers, "progress": asdict(progress), "epochs": history,
            "best": {key: best[key] for key in ("epoch", "validation_cer", "validation_loss")} if best else None,
            "invocation_seconds": round(time.monotonic() - started, 3),
            "invocation_steps": progress.step - initial_step,
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        })

    if args.resume:
        publish_best()  # Recover best.pt even when only last.pt was carried to a new runtime.
    checkpoint("running")
    while progress.epoch <= settings["epochs"] and not budget_exhausted():
        epoch = progress.epoch
        seed = settings["seed"] + epoch
        batches = bucket_batches(widths["train"], batch_size, bucket_size, seed=seed)
        remaining = batches[progress.batches_done:]
        model.train()
        for batch in loader(train_data, remaining, seed):
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = forward_loss(model, batch, device)
            loss_value = loss.item()
            if not math.isfinite(loss_value):
                raise RuntimeError(f"Non-finite training loss before step {progress.step + 1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            count = batch["images"].shape[0]
            progress.step += 1
            progress.batches_done += 1
            progress.lines_seen += count
            progress.loss_sum += count * loss_value
            if progress.step % settings["log_every_steps"] == 0:
                elapsed = max(time.monotonic() - started, 1e-9)
                rate = (progress.step - initial_step) / elapsed
                print(f"Epoch {epoch}, batch {progress.batches_done}/{len(batches)}, step {progress.step}: "
                      f"CTC {loss_value:.4f}, {rate:.2f} steps/s", flush=True)
            if progress.step % settings["checkpoint_every_steps"] == 0:
                checkpoint("running")
            if budget_exhausted():
                break

        if progress.batches_done < len(batches) or time_exhausted():
            break
        result = validate(model, val_loader, val_data.id_to_char, device)
        training_loss = progress.loss_sum / progress.lines_seen
        history.append({"epoch": epoch, "step": progress.step, "train_loss": training_loss,
                        "train_lines": progress.lines_seen, "validation": result})
        improved = best is None or (result["cer"], result["loss"]) < (best["validation_cer"], best["validation_loss"])
        if improved:
            best = {"epoch": epoch, "validation_cer": result["cer"], "validation_loss": result["loss"],
                    "model_state_dict": {name: tensor.detach().to(device="cpu", copy=True)
                                         for name, tensor in model.state_dict().items()}}
        print(f"Epoch {epoch} complete: train CTC {training_loss:.4f}, "
              f"validation CTC {result['loss']:.4f}, validation CER {result['cer']:.2%}", flush=True)
        progress = Progress(epoch=epoch + 1, step=progress.step)
        checkpoint("complete" if progress.epoch > settings["epochs"] else "running", new_best=improved)

    status = "complete" if progress.epoch > settings["epochs"] else "paused"
    checkpoint(status)
    print(f"{status.title()}: step {progress.step}; checkpoint {output / 'last.pt'}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=TRAIN_CONFIG)
    parser.add_argument("--data-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-steps", type=int, help="Pause at this total optimizer step")
    parser.add_argument("--time-limit-minutes", type=float, help="Pause training after this invocation budget")
    parser.add_argument("--train-limit", type=int, help="Use only the first N training lines for a smoke run")
    parser.add_argument("--val-limit", type=int, help="Use only the first N validation lines for a smoke run")
    args = parser.parse_args(argv)
    for name in ("threads", "batch_size", "epochs", "max_steps", "train_limit", "val_limit", "time_limit_minutes"):
        value = getattr(args, name)
        if value is not None and (value <= 0 or not math.isfinite(value)):
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if args.workers is not None and args.workers < 0:
        parser.error("--workers must be nonnegative")
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable in the active runtime")
    torch.set_num_threads(args.threads)
    args.data_root = args.data_root.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        settings = training_settings(args.config, args.epochs)
        data_config = load_data_config(args.data_config, project_root=args.data_root)
        with run_lock(args.output_dir):
            run_training(args, settings, data_config, device)
    except (KeyError, OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"Training failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
