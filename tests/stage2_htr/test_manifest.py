import json

import pytest

from stage2_htr.build_htr_manifest import main


@pytest.mark.parametrize("bad_row, expected", [
    ({"image_path": "absent.png", "text": "क"}, "missing_image_paths"),
    ({"text": "ABC123"}, "latin_letter_rows"),
    ({"text": "  "}, "blank_text_rows"),
    ({"split": "validation"}, "invalid_split_rows"),
    ({"line_id": ""}, "missing_required_rows"),
    ({"text": "क&"}, "bad_ocr_symbol_rows"),
    ({"text": "्क"}, "unapproved_label_warning_rows"),
])
def test_failed_qa_never_overwrites_existing_manifests(data_files, tmp_path, bad_row, expected):
    image, manifest, config, _ = data_files
    path = manifest([{"image_path": str(image()), "text": "क", **bad_row}])
    out, reports = tmp_path / "manifests", tmp_path / "reports"
    out.mkdir()
    previous = out / "htr_train.csv"
    previous.write_bytes(b"previous valid dataset")
    code = main(["--metadata", str(path), "--config", str(config(path)),
                 "--output-dir", str(out), "--report-dir", str(reports), "--workers", "1"])
    assert code == 1 and previous.read_bytes() == b"previous valid dataset"
    assert not (out / "htr_all.csv").exists()
    report = json.loads((reports / "htr_manifest_report.json").read_text())
    assert report["qa_status"] == "FAIL" and report["validation"][expected] > 0


def test_valid_rows_publish_all_manifests(data_files, tmp_path):
    image, manifest, config, _ = data_files
    path = manifest([{"image_path": str(image()), "text": "क"}])
    out, reports = tmp_path / "out", tmp_path / "reports"
    assert main(["--metadata", str(path), "--config", str(config(path)),
                 "--output-dir", str(out), "--report-dir", str(reports), "--workers", "1"]) == 0
    assert all((out / f"htr_{split}.csv").exists() for split in ["all", "train", "val", "test"])


def test_evaluation_only_character_blocks_publication(data_files, tmp_path):
    image, manifest, config, _ = data_files
    path = manifest([
        {"image_path": str(image("train.png")), "text": "क", "split": "train"},
        {"image_path": str(image("val.png")), "text": "क°", "split": "val"},
    ])
    reports = tmp_path / "reports"
    code = main(["--metadata", str(path), "--config", str(config(path)),
                 "--output-dir", str(tmp_path / "out"), "--report-dir", str(reports), "--workers", "1"])
    assert code == 1
    report = json.loads((reports / "htr_manifest_report.json").read_text())
    assert report["validation"]["validation_only_characters"] == 1
    assert report["evaluation_character_gaps"]["val"][0]["codepoint"] == "U+00B0"


def test_corrupt_pixels_block_manifest_even_with_supplied_dimensions(data_files, tmp_path):
    image, manifest, config, _ = data_files
    damaged = image()
    damaged.write_bytes(damaged.read_bytes()[:41])
    path = manifest([{"image_path": str(damaged), "text": "क", "width": "80", "height": "40"}])
    out, reports = tmp_path / "out", tmp_path / "reports"
    assert main(["--metadata", str(path), "--config", str(config(path)),
                 "--output-dir", str(out), "--report-dir", str(reports), "--workers", "1"]) == 1
    assert json.loads((reports / "htr_manifest_report.json").read_text())["validation"]["unreadable_images"] == 1


def test_cross_split_labels_block_publication(data_files, tmp_path):
    image, manifest, config, _ = data_files
    path = manifest([
        {"image_path": str(image("a.png")), "text": "क", "split": "train"},
        {"image_path": str(image("b.png")), "text": "क", "split": "test"},
    ])
    reports = tmp_path / "reports"
    assert main(["--metadata", str(path), "--config", str(config(path)),
                 "--output-dir", str(tmp_path / "out"), "--report-dir", str(reports), "--workers", "1"]) == 1
    assert json.loads((reports / "htr_manifest_report.json").read_text())["validation"]["cross_split_texts"] == 1


@pytest.mark.parametrize("font_name,text", [("val", "क"), ("Poppins-Regular", "१२३")])
def test_font_holdout_or_digit_form_violation_blocks_publication(data_files, tmp_path, font_name, text):
    import yaml
    image, manifest, config, _ = data_files
    path = manifest([{"image_path": str(image()), "text": text, "font_name": font_name}])
    configuration = config(path)
    settings = yaml.safe_load(configuration.read_text(encoding="utf-8"))
    font_dir = tmp_path / "fonts"
    font_dir.mkdir()
    for name in ("train", "Poppins-Regular", "val", "test"):
        (font_dir / (name + ".ttf")).touch()
    settings["paths"]["font_dir"] = str(font_dir)
    settings["font_split"] = {
        "use_font_aware_split": True, "heldout_val_fonts": 1, "heldout_test_fonts": 1,
        "val_fonts": ["val"], "test_fonts": ["test"],
    }
    configuration.write_text(yaml.safe_dump(settings), encoding="utf-8")
    out, reports = tmp_path / "out", tmp_path / "reports"
    assert main(["--metadata", str(path), "--config", str(configuration),
                 "--output-dir", str(out), "--report-dir", str(reports), "--workers", "1"]) == 1
    assert not (out / "htr_all.csv").exists()
    assert json.loads((reports / "htr_manifest_report.json").read_text())["validation"]["font_split_violations"] == 1


def test_empty_or_wrong_schema_inputs_fail_with_reports(data_files, tmp_path):
    _, manifest, config, _ = data_files
    path = manifest([])
    path.write_text("unexpected_column\n", encoding="utf-8")
    reports = tmp_path / "reports"
    assert main(["--metadata", str(path), "--config", str(config(path)),
                 "--output-dir", str(tmp_path / "out"), "--report-dir", str(reports), "--workers", "1"]) == 1
    report = json.loads((reports / "htr_manifest_report.json").read_text())
    assert report["validation"]["empty_input"] == 1
    assert report["validation"]["missing_columns"] == 4
