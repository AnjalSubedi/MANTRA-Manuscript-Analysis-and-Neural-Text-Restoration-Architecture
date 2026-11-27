"""Recreate the manuscript pilot inputs from locally obtained scans and editions."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image
import yaml


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from manuscripts.segment_lines import crop_polygon


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def find_source(root, name):
    paths = list(root.rglob(name))
    if len(paths) != 1:
        raise ValueError(f"Expected one {name} in {root}, found {len(paths)}")
    return paths[0]


def read_body(path, count):
    text = path.read_text(encoding="utf-8-sig")
    entries = re.findall(r"^\[(\d+)\][ \t]+([^\n]+)", text, flags=re.MULTILINE)[-count:]
    if [int(number) for number, _ in entries] != list(range(1, count + 1)):
        raise ValueError(f"Expected {count} consecutive numbered body lines in {path.name}")
    return {int(number): reference for number, reference in entries}


def reference_text(reference, document):
    if document == "LD_0001_0001":
        return reference.split("¯", 1)[0].replace("\u00ad", "").strip()
    text = re.sub(r"\[Seal of [^\]]+\]", "", reference).replace("\u00ad", "")
    return re.sub(r"[ \t]*¯[¯ \t]*", "", text).strip()


def boundary(vertices, width):
    offsets, heights = zip(*vertices, strict=True)
    return np.rint(np.interp(np.arange(width), offsets, heights)).astype(np.int32)


def prepare(args):
    study = json.loads((EXPERIMENT_DIR / "study.json").read_text(encoding="utf-8"))
    with (EXPERIMENT_DIR / "crop_geometry.jsonl").open(encoding="utf-8") as stream:
        geometry = list(map(json.loads, stream))
    initial_charset = args.charset.read_bytes()
    if sha256(initial_charset) != study["initial_charset_sha256"]:
        raise ValueError("The supplied charset differs from the original synthetic checkpoint charset")

    sources = {}
    for identifier, document in study["documents"].items():
        image_file = find_source(args.source_dir, f"{identifier}_manuscript_01.png")
        edition = find_source(args.source_dir, f"{identifier}_transcription.txt")
        with Image.open(image_file) as image:
            source = image.convert("RGB")
        if list(source.size) != document["source_size"] or sha256(source.tobytes()) != document["source_rgb_sha256"]:
            raise ValueError(f"Scan pixels differ for {identifier}")
        sources[identifier] = source, read_body(edition, document["numbered_body_lines"])

    output = args.output_dir.resolve()
    splits = defaultdict(list)
    crops = []
    for row in geometry:
        identifier = row["document"]
        source, edition = sources[identifier]
        text = reference_text(edition[row["body_line"]], identifier)
        if sha256(text.encode("utf-8")) != row["reference_sha256"]:
            raise ValueError(f"Edition text differs for {row['line_id']}")
        left, right = row["left"], row["right"]
        top, bottom = (boundary(row[key], right - left) for key in ("top", "bottom"))
        crop, _, _ = crop_polygon(source, left, right, top, bottom)
        if sha256(crop.tobytes()) != row["crop_rgb_sha256"]:
            raise ValueError(f"Crop pixels differ for {row['line_id']}")
        path = output / "crops" / identifier / f"{row['line_id']}.png"
        split = study["documents"][identifier]["split"]
        splits[split].append({"line_id": row["line_id"], "image_path": path.as_posix(), "text": text,
                              "width": crop.width, "height": crop.height, "document": identifier, "split": split})
        crops.append((path, crop))

    counts = Counter(row["document"] for rows in splits.values() for row in rows)
    if counts != {key: value["scored_lines"] for key, value in study["documents"].items()}:
        raise ValueError("The prepared document counts differ from the published pilot")
    charset = json.loads(initial_charset)
    vocabulary = charset["char_to_id"]
    for char in sorted({char for row in splits["train"] for char in row["text"]} - vocabulary.keys()):
        index = len(vocabulary)
        vocabulary[char] = index
        charset["id_to_char"][str(index)] = char
    charset["num_classes"] = len(vocabulary)

    output.mkdir(parents=True)
    for path, crop in crops:
        path.parent.mkdir(parents=True, exist_ok=True)
        crop.save(path)
    fields = ("line_id", "image_path", "text", "width", "height", "document", "split")
    for split, rows in splits.items():
        with (output / f"{split}.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    (output / "baseline_charset.json").write_bytes(initial_charset)
    # Match the serialized vocabulary used by the recorded Windows checkpoint.
    (output / "charset.json").write_text(json.dumps(charset, ensure_ascii=False, indent=2) + "\n",
                                         encoding="utf-8", newline="\r\n")
    data_config = yaml.safe_load((PROJECT_ROOT / "configs/stage2_data_synth100k.yaml").read_text(encoding="utf-8"))
    data_config["paths"] = {f"manifest_{split}": (output / f"{split}.csv").as_posix() for split in splits}
    data_config["dataloader"].update(batch_size=8, num_workers=0)
    for name, vocabulary_file in (("baseline_data.yaml", "baseline_charset.json"), ("data.yaml", "charset.json")):
        data_config["paths"]["charset_json"] = (output / vocabulary_file).as_posix()
        (output / name).write_text(yaml.safe_dump(data_config, sort_keys=False), encoding="utf-8")
    report = json.loads((PROJECT_ROOT / "results/crnn_ctc/metrics.json").read_text(encoding="utf-8"))
    settings = report["manuscript_fine_tuning"]["training_settings"]
    (output / "training.yaml").write_text(yaml.safe_dump(settings, sort_keys=False), encoding="utf-8")
    print("Prepared " + ", ".join(f"{split}={len(rows)}" for split, rows in splits.items()) + f" in {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True, help="Local folder containing the named scans and editions")
    parser.add_argument("--charset", type=Path, required=True, help="Character vocabulary of the synthetic baseline")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for generated crops and manifests")
    args = parser.parse_args()
    try:
        prepare(args)
    except (KeyError, OSError, ValueError) as error:
        parser.exit(1, f"Preparing the manuscript pilot failed: {error}\n")


if __name__ == "__main__":
    main()
