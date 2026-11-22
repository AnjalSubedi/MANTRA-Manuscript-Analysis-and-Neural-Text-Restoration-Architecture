"""Re-render fonts that violate the configured holdouts, preserving sample splits.

Original PNGs are retained. A resumable journal and metadata/report backups live
under outputs/font_split_repair/. The active metadata changes only after every
replacement succeeds and the complete font partition passes validation.
"""
import argparse
import atexit
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import random
import shutil
import sys
import time

import cv2
from PIL import Image
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage1_dataset.chromium_text_renderer import shutdown_chromium_renderer
from stage1_dataset.degradation_profiles import apply_degradation
from stage1_dataset.font_policy import configured_font_splits, font_is_allowed
from stage1_dataset.generate_synthetic_images import calculate_readability_metrics
from stage1_dataset.render_text_line import check_font_supports_text, render_text_line

_config = None
_fonts = None


def _run_image_dir(config, work_dir, fingerprint):
    # A new repair must never overwrite PNGs referenced by an earlier active
    # dataset, even when it changes policy or uses a different work directory.
    identity = f"{Path(work_dir).resolve()}:{fingerprint}"
    run_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    return Path(config["paths"]["image_dir"]) / "font_holdout_v2" / run_id


def _read_journal(path):
    """Recover completed replacements, including after a partial last write."""
    done = {}
    if not path.exists():
        return done
    with path.open("rb+") as stream:
        while True:
            start = stream.tell()
            line = stream.readline()
            if not line:
                break
            try:
                record = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                if not line.endswith(b"\n"):
                    stream.seek(start)
                    stream.truncate()
                    break
                continue
            try:
                with Image.open(record["row"]["image_path"]) as image:
                    image.load()
                done[record["index"]] = record["row"]
            except (OSError, ValueError, KeyError):
                pass
            if not line.endswith(b"\n"):
                stream.seek(0, os.SEEK_END)
                stream.write(b"\n")
                break
    return done


def _initialize(config):
    global _config, _fonts
    _config = config
    _fonts = configured_font_splits(config)
    cv2.setNumThreads(1)
    atexit.register(shutdown_chromium_renderer)


def _repair_chunk(items):
    repaired = []
    rendering = _config["rendering"]
    for index, row in items:
        rng = random.Random(_config["dataset"]["random_seed"] + 9000000 + index)
        supported = [p for p in _fonts[row["split"]] if check_font_supports_text(p, row["text"])]
        if not supported:
            raise ValueError(f"No allowed font covers sample {index}: {row['line_id']}")
        font = rng.choice(supported)
        size = rng.randint(rendering["font_size_min"], rendering["font_size_max"])
        padding = rng.randint(rendering["padding_min"], rendering["padding_max"])
        clean, renderer, status, reason = render_text_line(
            row["text"], font, size, padding, padding, renderer_preference="chromium_canvas"
        )
        if status != "success":
            raise RuntimeError(f"Sample {index} could not render: {reason}")
        profile = row["degradation_profile"]
        for attempt in range(6):
            # Record a clean fallback truthfully when all degradation attempts
            # fail the same readability and clipping gate used by generation.
            if attempt == 5:
                profile = "D0_clean_print"
            image, params = apply_degradation(
                clean, profile, target_difficulty=row["difficulty_bucket"],
                seed=_config["dataset"]["random_seed"] + 9000000 + index,
                attempt=attempt,
            )
            risk, score, _, _ = calculate_readability_metrics(image)
            bbox = image.convert("L").point(lambda p: 255 if p < 160 else 0).getbbox()
            clipped = not bbox or bbox[0] <= 1 or bbox[1] <= 1 or bbox[2] >= image.width - 1 or bbox[3] >= image.height - 1
            if not risk and not clipped:
                break
        else:
            raise RuntimeError(f"Sample {index} failed image quality validation")
        output_dir = _config.get("_repair_image_dir", str(Path(_config["paths"]["image_dir"]) / "font_holdout_v2"))
        path = Path(output_dir) / row["split"] / f"sample_{index:06d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        image.save(temporary, format="PNG")
        temporary.replace(path)
        updated = dict(row)
        updated.update({
            "image_path": path.as_posix(), "font_name": Path(font).stem,
            "actual_font_name": Path(font).stem, "requested_font_name": Path(font).stem,
            "font_fallback_used": "false", "glyph_coverage_passed": "true", "missing_glyphs": "",
            "renderer": renderer, "degradation_profile": profile,
            "degradation_family": "composed" if profile == "composed" else "single",
            "degradation_parameters_json": json.dumps(params),
            "difficulty_bucket": params.get("difficulty_bucket", row["difficulty_bucket"]),
            "unreadable_risk": "false", "readability_score_estimate": str(score),
            "width": str(image.width), "height": str(image.height),
            "font_size": str(size), "padding": str(padding),
            "repair_source_image_path": row["image_path"],
            "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "font_sha256": hashlib.sha256(Path(font).read_bytes()).hexdigest(),
            "label_text_sha256": hashlib.sha256(row["text"].encode("utf-8")).hexdigest(),
            "render_parameters_json": json.dumps({"font_size": size, "padding": padding, "engine": renderer}),
        })
        repaired.append({"index": index, "row": updated})
    return repaired


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage1_synth100k_config.yaml")
    parser.add_argument("--metadata", help="Defaults to the active metadata selected by --config")
    parser.add_argument("--work-dir", default="outputs/font_split_repair")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    started = time.time()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    fonts = configured_font_splits(config)
    metadata = Path(args.metadata or config["paths"].get(
        "final_metadata", "data/stage1_synthetic/metadata/synthetic_image_metadata_stage1_all_final.csv"
    ))
    original = metadata.read_bytes()
    with metadata.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = list(reader)
    items = [(i, r) for i, r in enumerate(rows) if not font_is_allowed(r, fonts)]
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    summary = {
        "total_rows": len(rows), "replacements_required": len(items),
        "fonts": {s: [Path(p).stem for p in paths] for s, paths in fonts.items()},
    }
    (work / "plan.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)
    if args.plan_only or not items:
        return
    fingerprint = hashlib.sha256(original + json.dumps(config, sort_keys=True).encode()).hexdigest()
    config["_repair_image_dir"] = str(_run_image_dir(config, work, fingerprint))
    state = work / "input.sha256"
    if state.exists() and state.read_text().strip() != fingerprint:
        raise ValueError("Repair inputs changed; use a new work-dir to preserve the earlier backup")
    state.write_text(fingerprint, encoding="ascii")
    backup = work / "original_metadata.csv"
    if not backup.exists():
        backup.write_bytes(original)
        for name in ("manifests", "reports"):
            source = Path("data/stage2_htr") / name
            if source.exists():
                shutil.copytree(source, work / ("original_" + name))
    journal = work / "replacements.jsonl"
    done = _read_journal(journal)
    pending = [item for item in items if item[0] not in done]
    chunks = [pending[i:i + 32] for i in range(0, len(pending), 32)]
    with journal.open("a", encoding="utf-8") as stream:
        with ProcessPoolExecutor(
            max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize, initargs=(config,),
        ) as executor:
            futures = [executor.submit(_repair_chunk, chunk) for chunk in chunks]
            for future in as_completed(futures):
                for record in future.result():
                    done[record["index"]] = record["row"]
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                print(f"Repaired {len(done)}/{len(items)}; elapsed {time.time() - started:.0f}s", flush=True)
    for index, updated in done.items():
        immutable = ("text", "line_id", "source_id", "source_file", "split", "variant_index")
        if (not 0 <= index < len(rows)
                or any(updated.get(key) != rows[index].get(key) for key in immutable)):
            raise ValueError("A replacement changed sample identity; active metadata was preserved")
        rows[index] = updated
    if any(not font_is_allowed(row, fonts) for row in rows):
        raise RuntimeError("Font validation failed; active metadata was not changed")
    fields = list(dict.fromkeys(fields + [key for row in rows for key in row]))
    temporary = metadata.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(metadata)
    summary.update(qa_status="PASS", replacements_completed=len(done), elapsed_seconds=round(time.time() - started, 2))
    (work / "repair_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Repair complete; active metadata published. Rebuild Stage 2 manifests next.", flush=True)


if __name__ == "__main__":
    main()
