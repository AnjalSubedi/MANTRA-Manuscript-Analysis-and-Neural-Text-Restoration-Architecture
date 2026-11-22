from pathlib import Path

import pytest

from stage1_dataset.font_policy import font_is_allowed, font_split_errors, partition_fonts
from stage1_dataset.generate_final_dataset import select_font_and_check_coverage
from stage1_dataset.font_policy import font_preserves_devanagari_digits


@pytest.mark.parametrize("name", ["Hind-Regular", "Poppins-Regular", "Rajdhani-Regular", "RozhaOne-Regular", "Teko-Regular"])
def test_latin_form_devanagari_numerals_are_restricted_without_banning_letter_samples(name):
    assert not font_preserves_devanagari_digits(name, "श्री १२३")
    assert font_preserves_devanagari_digits(name, "श्री सुब्बा")
    groups = {"train": [name + ".ttf"]}
    assert not font_is_allowed({"font_name": name, "split": "train", "text": "१२३"}, groups)
    assert font_is_allowed({"font_name": name, "split": "train", "text": "श्री"}, groups)
    row = {"font_name": name, "split": "train", "text": "१२३"}
    assert font_split_errors([row], {"font_split": {"use_font_aware_split": False}}) == [row]


def test_generation_falls_back_from_latin_digit_shapes_even_with_complete_cmap():
    text = "श्री १२३"
    bad, good = "Poppins-Regular.ttf", "Karma-Regular.ttf"
    cmaps = {p: {ord(c): "glyph" for c in text} for p in (bad, good)}
    result = select_font_and_check_coverage(text, bad, [bad, good], cmaps)
    assert result[:3] == (good, "true", "true")


def policy():
    return {"font_split": {"use_font_aware_split": True, "heldout_val_fonts": 1,
                           "heldout_test_fonts": 1, "val_fonts": ["val"], "test_fonts": ["test"]}}


def test_font_partitions_are_disjoint_and_deterministic():
    paths = ["train.ttf", "val.ttf", "test.ttf"]
    groups = partition_fonts(paths, policy())
    assert groups == {"train": ["train.ttf"], "val": ["val.ttf"], "test": ["test.ttf"]}
    assert partition_fonts(list(reversed(paths)), policy()) == groups
    assert not font_is_allowed({"font_name": "test", "split": "train"}, groups)


def test_fallback_does_not_escape_the_split():
    cmaps = {"train.ttf": {ord("क"): 1}, "val.ttf": {ord("ख"): 1}}
    result = select_font_and_check_coverage("ख", "train.ttf", ["train.ttf"], cmaps)
    assert result[1] == "false"
    assert result[0] == "train.ttf"


@pytest.mark.parametrize("change", ["overlap", "missing", "count"])
def test_invalid_partition_fails_early(change):
    config = policy()
    if change == "overlap":
        config["font_split"]["test_fonts"] = ["val"]
    elif change == "missing":
        config["font_split"]["val_fonts"] = ["absent"]
    else:
        config["font_split"]["heldout_test_fonts"] = 2
    with pytest.raises(ValueError):
        partition_fonts(["train.ttf", "val.ttf", "test.ttf"], config)


def test_manifest_qa_rejects_a_font_from_another_split(tmp_path):
    for name in ["train", "val", "test"]:
        (tmp_path / (name + ".ttf")).touch()
    config = {**policy(), "paths": {"font_dir": str(tmp_path)}}
    row = {"font_name": "val", "split": "train"}
    assert font_split_errors([row], config) == [row]


def test_generation_restricts_real_worker_selection_and_output_paths(tmp_path, monkeypatch):
    import yaml
    from PIL import Image, ImageDraw
    from stage1_dataset import generate_final_dataset as generation
    from stage1_dataset import chromium_text_renderer

    fonts = tmp_path / "fonts"
    fonts.mkdir()
    for name in ("train", "val", "test"):
        (fonts / (name + ".ttf")).touch()
    config = {**policy(), "paths": {"font_dir": str(fonts)},
              "rendering": {"font_size_min": 32, "font_size_max": 32,
                            "padding_min": 20, "padding_max": 20},
              "degradation_distribution": {"D0_clean_print": 1}}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(generation, "IMAGE_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(generation, "FONTS_TO_DOWNLOAD", [
        {"file_name": name + ".ttf", "font_slug": name, "font_group": "test"}
        for name in ("train", "val", "test")
    ])

    class Renderer:
        def start(self):
            pass

    monkeypatch.setattr(chromium_text_renderer, "get_chromium_renderer", Renderer)
    monkeypatch.setattr(chromium_text_renderer, "shutdown_chromium_renderer", lambda: None)
    chosen = []

    def select(text, requested, candidates, cmaps):
        chosen.append([Path(p).stem for p in candidates])
        return requested, "true", "false", Path(requested).stem, Path(requested).stem, ""

    monkeypatch.setattr(generation, "select_font_and_check_coverage", select)
    canvas = Image.new("L", (100, 60), 255)
    ImageDraw.Draw(canvas).rectangle((10, 10, 80, 50), fill=0)
    monkeypatch.setattr(generation, "render_text_line", lambda **kwargs: (canvas.copy(), "chromium", "success", "ok"))
    monkeypatch.setattr(generation, "apply_degradation", lambda image, *args, **kwargs: (image, {"difficulty_bucket": "easy"}))
    rows = [{"line_id": str(i), "source_id": "fixture", "source_file": "fixture.txt",
             "text": "क", "domain": "coverage_matra", "split": split}
            for i, split in enumerate(("train", "val", "test"))]
    result = generation.render_worker_chunk((rows, "images/fixture", "v01", 42, 0, str(path), None, 0))
    assert chosen == [["train"], ["val"], ["test"]]
    assert [row["font_name"] for row in result] == ["train", "val", "test"]
    assert len({row["image_path"] for row in result}) == 3
    assert all(Path(row["image_path"]).is_file() for row in result)
