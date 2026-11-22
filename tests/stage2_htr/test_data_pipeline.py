import json

import pytest
import torch
from torch.utils.data import DataLoader

from stage2_htr.collate import htr_collate_fn
from stage2_htr.config import dataset_options, load_data_config, make_dataloader
from stage2_htr.dataset import HTRLineDataset, encode_text
from stage2_htr.inspect_dataset import _run_check_all, main


def test_width_filter_uses_actual_dimensions_and_keeps_valid_batches(data_files):
    image, manifest, _, charset = data_files
    narrow = image("narrow.png", width=40)
    wide = image("wide.png", width=200)
    path = manifest([
        {"image_path": str(wide), "text": "ख", "width": "1", "height": "999"},
        {"image_path": str(narrow), "text": "क", "width": "999", "height": "1"},
    ])
    ds = HTRLineDataset(path, charset, max_width=100)
    assert ds.source_row_count == 2 and len(ds.skipped_rows) == 1 and len(ds) == 1
    batch = next(iter(DataLoader(ds, batch_size=2, collate_fn=htr_collate_fn)))
    assert batch["texts"] == ["क"]
    assert batch["images"].shape == (1, 1, 64, 64)


def test_width_filter_and_resize_agree_at_a_rounding_boundary(data_files):
    image, manifest, _, charset = data_files
    path = manifest([{"image_path": str(image(width=13, height=26)), "text": "क"}])
    ds = HTRLineDataset(path, charset, image_height=63, max_width=31)
    assert len(ds) == 1
    assert ds[0]["image"].shape == (1, 63, 31)


def test_config_controls_inversion_height_and_background_padding(data_files):
    image, manifest, config, charset = data_files
    path = manifest([
        {"image_path": str(image("a.png", width=40)), "text": "क"},
        {"image_path": str(image("b.png", width=100)), "text": "क ख"},
    ])
    settings = load_data_config(config(path, invert_image=True, image_height=32))
    ds = HTRLineDataset(path, charset, **dataset_options(settings))
    normal = HTRLineDataset(path, charset, image_height=32)
    assert torch.equal(ds[0]["image"], 1 - normal[0]["image"])
    batch = next(iter(make_dataloader(ds, settings)))
    assert batch["images"].shape == (2, 1, 32, 80)
    assert (batch["images"][0, :, :, 32:] == 0).all()
    assert batch["labels"].tolist() == [1, 1, 2, 3]


def test_full_check_rejects_truncated_png_even_with_valid_header(data_files):
    image, manifest, _, charset = data_files
    damaged = image()
    damaged.write_bytes(damaged.read_bytes()[:41])
    path = manifest([{"image_path": str(damaged), "text": "क"}])
    report = _run_check_all(path, charset)
    assert report["image_open_failures"] == 1 and report["qa_status"] == "FAIL"


def test_width_limit_does_not_hide_pixel_corruption(data_files):
    image, manifest, _, charset = data_files
    damaged = image()
    damaged.write_bytes(damaged.read_bytes()[:41])
    path = manifest([{"image_path": str(damaged), "text": "क"}])
    ds = HTRLineDataset(path, charset, max_width=1)
    assert len(ds) == 1 and not ds.skipped_rows
    report = _run_check_all(path, charset, max_width=1)
    assert report["image_open_failures"] == 1 and report["qa_status"] == "FAIL"


@pytest.mark.parametrize("mode", [[], ["--check-all"]])
def test_all_failed_samples_save_reports(data_files, tmp_path, mode):
    _, manifest, config, _ = data_files
    path = manifest([{"image_path": str(tmp_path / "missing.png"), "text": "क"}])
    out = tmp_path / "reports"
    assert main(["--config", str(config(path)), "--report-dir", str(out), *mode]) == 1
    stem = "dataset_loader_check_all" if mode else "dataset_loader_report"
    report = json.loads((out / (stem + ".json")).read_text(encoding="utf-8"))
    assert report["qa_status"] == "FAIL" and report["load_failures"] == 1
    assert (out / (stem + ".md")).exists()
    assert not (out / "preprocessed_samples_grid.png").exists()


def test_mixed_failures_are_not_retried_in_dataloader(data_files, tmp_path):
    image, manifest, config, _ = data_files
    path = manifest([
        {"image_path": str(tmp_path / "missing.png"), "text": "क"},
        {"image_path": str(image()), "text": "ख"},
    ])
    out = tmp_path / "reports"
    assert main(["--config", str(config(path)), "--report-dir", str(out)]) == 1
    report = json.loads((out / "dataset_loader_report.json").read_text(encoding="utf-8"))
    assert report["dataloader_ok"] and report["samples_loaded"] == 1
    assert report["load_failures"] == 1


def test_every_width_filtered_is_reported_without_empty_batch(data_files, tmp_path):
    image, manifest, config, _ = data_files
    path = manifest([{"image_path": str(image()), "text": "क"}])
    out = tmp_path / "reports"
    assert main(["--config", str(config(path, max_width=1)), "--report-dir", str(out)]) == 1
    report = json.loads((out / "dataset_loader_report.json").read_text(encoding="utf-8"))
    assert report["skipped_width_rows"] == 1 and not report["errors"]


def test_empty_manifest_cannot_pass_full_qa(data_files):
    _, manifest, _, charset = data_files
    assert _run_check_all(manifest([]), charset)["qa_status"] == "FAIL"


def test_missing_setup_inputs_still_save_failure_report(data_files, tmp_path):
    _, manifest, config, _ = data_files
    path = manifest([])
    path.unlink()
    out = tmp_path / "reports"
    assert main(["--config", str(config(path)), "--report-dir", str(out)]) == 1
    assert json.loads((out / "dataset_loader_report.json").read_text())["errors"][0]["kind"] == "setup"


def test_unknown_character_and_blank_token_are_rejected():
    with pytest.raises(ValueError):
        encode_text("ख", {"क": 1})
    with pytest.raises(ValueError):
        encode_text("क", {"क": 0})
