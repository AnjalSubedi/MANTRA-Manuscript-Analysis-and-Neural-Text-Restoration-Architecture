"""Package the configured train/validation data for a remote training runtime."""

import argparse
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
        with ZipFile(partial, "w", compression=ZIP_STORED) as archive:
            for key in ("charset_json", "manifest_train", "manifest_val"):
                source = Path(paths[key]).resolve()
                archive.write(source, source.relative_to(root).as_posix(), compress_type=ZIP_DEFLATED)
                if not key.startswith("manifest_"):
                    continue
                count = 0
                with source.open(encoding="utf-8", newline="") as stream:
                    for row in csv.DictReader(stream):
                        image = Path(row["image_path"])
                        if image.is_absolute() or image.anchor or ".." in image.parts:
                            raise ValueError(f"Expected a project-relative image path: {image}")
                        name = image.as_posix()
                        if name not in included:
                            archive.write(root / image, name)
                            included.add(name)
                        count += 1
                        if count % 10000 == 0:
                            print(f"{key}: {count:,} rows packaged", flush=True)
                print(f"{key}: {count:,} rows packaged", flush=True)
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
