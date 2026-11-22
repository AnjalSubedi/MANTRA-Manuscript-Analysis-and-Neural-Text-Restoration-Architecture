import json

from PIL import Image

from stage1_dataset.repair_font_splits import _read_journal, _run_image_dir


def test_resume_recovers_valid_record_before_incomplete_utf8(tmp_path):
    image = tmp_path / "image.png"
    Image.new("L", (20, 20), 255).save(image)
    row = {"image_path": str(image), "text": "क"}
    good = json.dumps({"index": 5, "row": row}, ensure_ascii=False).encode("utf-8") + b"\n"
    journal = tmp_path / "replacements.jsonl"
    journal.write_bytes(good + b'{"index": 6, "row": {"text": "\xe0\xa4')
    assert _read_journal(journal) == {5: row}
    assert journal.read_bytes() == good


def test_resume_does_not_reuse_a_damaged_png(tmp_path):
    image = tmp_path / "image.png"
    image.write_bytes(b"damaged")
    journal = tmp_path / "replacements.jsonl"
    journal.write_text(json.dumps({"index": 0, "row": {"image_path": str(image)}}), encoding="utf-8")
    assert _read_journal(journal) == {}
    assert journal.read_bytes().endswith(b"\n")


def test_new_repair_runs_cannot_overwrite_previous_images(tmp_path):
    config = {"paths": {"image_dir": str(tmp_path / "images")}}
    first = _run_image_dir(config, tmp_path / "run_one", "input_one")
    assert first == _run_image_dir(config, tmp_path / "run_one", "input_one")
    assert first != _run_image_dir(config, tmp_path / "run_two", "input_one")
    assert first != _run_image_dir(config, tmp_path / "run_one", "input_two")
