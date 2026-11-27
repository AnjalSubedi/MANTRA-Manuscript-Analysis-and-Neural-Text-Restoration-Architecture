import json
from pathlib import Path
from zipfile import ZipFile

import pytest
import torch
import yaml

from stage2_htr.package_training_data import package_training_data
from stage2_htr.train import bucket_batches, main


@pytest.fixture
def release(data_files, tmp_path):
    image, manifest, config, charset = data_files
    train = manifest([
        {"image_path": image(f"train{index}.png", width=width, height=64).name, "text": text,
         "width": width, "height": 64}
        for index, (width, text) in enumerate([(39, "कक"), (74, "ख"), (58, "क ख"), (67, "खक")])
    ])
    val = manifest([
        {"image_path": image(f"val{index}.png", width=width, height=64).name,
         "text": text, "split": "val", "width": width, "height": 64}
        for index, (width, text) in enumerate([(47, "खख"), (68, "ख क")])
    ], "val.csv")
    test = manifest([
        {"image_path": image("test.png", width=70, height=64).name, "text": "ककख", "split": "test"}
    ], "test.csv")
    config_path = config(train)
    settings = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    settings["paths"].update(manifest_val=str(val), manifest_test=str(test))
    config_path.write_text(yaml.safe_dump(settings), encoding="utf-8")
    return tmp_path, config_path, train, val, charset


def test_buckets_keep_every_sample_and_have_reproducible_order():
    widths = [90, 10, 80, 20, 60, 30, 20, 40, 70, 50, 40]
    batches = bucket_batches(widths, 2, 6, seed=42)
    assert batches == bucket_batches(widths, 2, 6, seed=42)
    assert sorted(index for batch in batches for index in batch) == list(range(len(widths)))
    assert all(1 <= len(batch) <= 2 for batch in batches)
    assert all([widths[i] for i in batch] == sorted(widths[i] for i in batch) for batch in batches)


def test_phase5_resume_matches_uninterrupted_run_and_writes_best_checkpoint(release):
    root, data_config, *_ = release
    config = root / "training.yaml"
    config.write_text(yaml.safe_dump({
        "seed": 42, "epochs": 2, "hidden_size": 8, "learning_rate": 0.001, "weight_decay": 0.0,
        "gradient_clip": 5.0, "width_bucket_batches": 2, "checkpoint_every_steps": 2,
        "log_every_steps": 2,
    }), encoding="utf-8")
    common = ["--config", str(config), "--data-config", str(data_config), "--data-root", str(root),
              "--device", "cpu", "--threads", "1"]
    uninterrupted, interrupted = root / "continuous", root / "resumed"
    continuous_args = common + ["--output-dir", str(uninterrupted)]
    interrupted_args = common + ["--output-dir", str(interrupted)]
    old_threads = torch.get_num_threads()
    try:
        with torch.random.fork_rng(devices=[]):
            assert main(continuous_args) == 0
            assert main(interrupted_args + ["--max-steps", "1"]) == 0
            report_path = interrupted / "metrics.json"
            paused = json.loads(report_path.read_text(encoding="utf-8"))
            assert paused["status"] == "paused" and paused["progress"]["batches_done"] == 1
            checkpoint = interrupted / "last.pt"
            assert main(interrupted_args + ["--resume", str(checkpoint)]) == 0

            resumed = torch.load(checkpoint, weights_only=True)
            continuous = torch.load(uninterrupted / "last.pt", weights_only=True)
            assert resumed["progress"] == continuous["progress"]
            assert resumed["history"] == continuous["history"]
            for name, tensor in continuous["model_state_dict"].items():
                torch.testing.assert_close(resumed["model_state_dict"][name], tensor, rtol=0, atol=0)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            best = torch.load(interrupted / "best.pt", weights_only=True)
            assert report["status"] == "complete" and len(report["epochs"]) == 2
            assert report["epochs"][0]["train_lines"] == 4 and report["epochs"][0]["validation"]["lines"] == 2
            assert best["validation_cer"] == min(entry["validation"]["cer"] for entry in report["epochs"])

            with pytest.raises(SystemExit) as error:
                main(interrupted_args + ["--resume", str(checkpoint), "--batch-size", "3"])
            assert error.value.code == 1
            assert json.loads(report_path.read_text(encoding="utf-8")) == report
    finally:
        torch.set_num_threads(old_threads)


def test_training_archive_includes_only_configured_train_and_validation_assets(release):
    root, config, train, val, charset = release
    destination = root / "runtime" / "training.zip"
    package_training_data(destination, data_config=config, data_root=root)
    with ZipFile(destination) as archive:
        assert set(archive.namelist()) == {
            Path(train).name, Path(val).name, Path(charset).name,
            "train0.png", "train1.png", "train2.png", "train3.png", "val0.png", "val1.png",
        }
        assert archive.testzip() is None
