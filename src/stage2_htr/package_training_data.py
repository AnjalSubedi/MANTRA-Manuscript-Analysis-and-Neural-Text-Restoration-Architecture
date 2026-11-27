"""Package the configured train/validation data for a remote training runtime."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stage2_htr.config import DEFAULT_CONFIG, PROJECT_ROOT, load_data_config


def package_training_data(output, *, data_config=DEFAULT_CONFIG, data_root=PROJECT_ROOT):
    """Keep the release paths intact so the same manifests work after extraction."""
    root = Path(data_root).resolve()
    paths = load_data_config(data_config, project_root=root)["paths"]
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Training archive already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(prefix=output.name + ".", suffix=".part", dir=output.parent, delete=False) as stream:
        partial = Path(stream.name)
    included = set()
    try:
        with (ZipFile(partial, "w", compression=ZIP_STORED) as archive,
              ThreadPoolExecutor(max_workers=16) as readers):
            for key in ("charset_json", "manifest_train", "manifest_val"):
                source = Path(paths[key]).resolve()
                archive.write(source, source.relative_to(root).as_posix(), compress_type=ZIP_DEFLATED)
                if not key.startswith("manifest_"):
                    continue
                image_names = []
                with source.open(encoding="utf-8", newline="") as stream:
                    for row in csv.DictReader(stream):
                        image = Path(row["image_path"])
                        if image.is_absolute() or image.anchor or ".." in image.parts:
                            raise ValueError(f"Expected a project-relative image path: {image}")
                        name = image.as_posix()
                        if name not in included:
                            image_names.append(name)
                            included.add(name)
                for start in range(0, len(image_names), 256):
                    names = image_names[start:start + 256]
                    image_bytes = readers.map(Path.read_bytes, (root / name for name in names))
                    for name, data in zip(names, image_bytes):
                        archive.writestr(name, data)
                    completed = start + len(names)
                    if completed // 10000 > start // 10000:
                        print(f"{key}: {completed:,} images packaged", flush=True)
                print(f"{key}: {len(image_names):,} images packaged", flush=True)
        partial.replace(output)
    finally:
        partial.unlink(missing_ok=True)
    print(f"Saved {len(included):,} image files to {output}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--data-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args(argv)
    package_training_data(args.output, data_config=args.data_config, data_root=args.data_root)


if __name__ == "__main__":
    main()
