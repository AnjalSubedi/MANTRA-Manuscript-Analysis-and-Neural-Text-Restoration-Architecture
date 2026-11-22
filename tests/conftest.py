import csv
import json

from PIL import Image, ImageDraw
import pytest
import yaml


@pytest.fixture
def data_files(tmp_path):
    charset = tmp_path / "charset.json"
    charset.write_text(json.dumps({
        "char_to_id": {"<BLANK>": 0, "क": 1, " ": 2, "ख": 3},
        "id_to_char": {"0": "<BLANK>", "1": "क", "2": " ", "3": "ख"},
        "num_classes": 4,
    }), encoding="utf-8")

    def image(name="line.png", width=80, height=40):
        path = tmp_path / name
        canvas = Image.new("L", (width, height), 255)
        ImageDraw.Draw(canvas).rectangle((5, 5, min(width - 6, 30), height - 6), fill=0)
        canvas.save(path)
        return path

    def manifest(rows, name="manifest.csv"):
        path = tmp_path / name
        fields = list(dict.fromkeys(["image_path", "text", "split", "line_id", "variant_index"]
                                    + [key for row in rows for key in row]))
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for i, row in enumerate(rows):
                writer.writerow({"split": "train", "line_id": str(i), "variant_index": "1", **row})
        return path

    def config(manifest_path, **settings):
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump({
            "dataset": {"image_height": 64, "max_width": None, "invert_image": False,
                        "pixel_range": [0.0, 1.0], "augment": False, **settings},
            "paths": {"manifest_train": str(manifest_path), "manifest_all": str(manifest_path),
                      "charset_json": str(charset)},
            "dataloader": {"batch_size": 2, "num_workers": 0, "pin_memory": False},
            "font_split": {"use_font_aware_split": False},
        }), encoding="utf-8")
        return path
    return image, manifest, config, charset
