import json
from pathlib import Path
from zipfile import ZipFile

import pytest
import torch
import yaml

from stage2_htr.config import dataset_options, load_data_config
from stage2_htr.evaluate import main as evaluate_main
from stage2_htr.model import CRNN
from stage2_htr.package_training_data import package_training_data
from stage2_htr.train import CHECKPOINT_VERSION, bucket_batches, file_sha256, initialize_from_checkpoint, main


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


def test_fine_tuning_preserves_weights_when_characters_are_appended(tmp_path):
    source_options = {"num_classes": 3, "image_height": 64, "hidden_size": 8}
    source_vocab = {0: "<BLANK>", 1: "क", 2: "ख"}
    source = CRNN(**source_options)
    checkpoint = tmp_path / "best.pt"
    torch.save({"format_version": CHECKPOINT_VERSION, "model_options": source_options,
                "id_to_char": source_vocab, "model_state_dict": source.state_dict()}, checkpoint)

    expanded = {**source_options, "num_classes": 4}
    model = CRNN(**expanded)
    extra_weight = model.classifier.weight[3].detach().clone()
    extra_bias = model.classifier.bias[3].detach().clone()
    digest = initialize_from_checkpoint(model, checkpoint, expanded, {**source_vocab, 3: "∙"})
    assert digest == file_sha256(checkpoint)
    for name, expected in source.state_dict().items():
        actual = model.state_dict()[name]
        if name.startswith("classifier."):
            actual = actual[:source_options["num_classes"]]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(model.classifier.weight[3], extra_weight, rtol=0, atol=0)
    torch.testing.assert_close(model.classifier.bias[3], extra_bias, rtol=0, atol=0)
    with pytest.raises(ValueError, match="preserve all initial character IDs"):
        initialize_from_checkpoint(model, checkpoint, expanded, {0: "<BLANK>", 1: "ख", 2: "क", 3: "∙"})


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


def test_evaluation_scores_unseen_characters_and_keeps_unlabeled_predictions(release, data_files):
    root, data_config, *_ = release
    config = load_data_config(data_config, project_root=root)
    image, manifest, *_ = data_files
    external = manifest([
        {"image_path": image("correct.png").name, "text": "क"},
        {"image_path": image("outside_vocab.png").name, "text": "कॐॐ"},
        {"image_path": image("unlabeled.png").name, "text": ""},
    ], "external.csv")
    external.write_text(external.read_text(encoding="utf-8"), encoding="utf-8-sig")

    model_options = {"num_classes": 4, "image_height": 64, "hidden_size": 8}
    model = CRNN(**model_options)
    with torch.no_grad():
        model.classifier.weight.zero_()
        model.classifier.bias.copy_(torch.tensor([0, 1, 0, 0]))
    checkpoint = root / "best.pt"
    torch.save({
        "format_version": CHECKPOINT_VERSION,
        "model_options": model_options, "model_state_dict": model.state_dict(),
        "data_options": dataset_options(config), "id_to_char": {0: "<BLANK>", 1: "क", 2: " ", 3: "ख"},
        "data_sha256": {name: file_sha256(config["paths"][name])
                        for name in ("charset_json", "manifest_train", "manifest_val")},
        "epoch": 1, "validation_cer": 0.7,
    }, checkpoint)

    common = ["--checkpoint", str(checkpoint), "--data-config", str(data_config), "--data-root", str(root),
              "--device", "cpu", "--threads", "1"]
    test_output, external_output = root / "synthetic_test", root / "external_test"
    old_threads = torch.get_num_threads()
    try:
        evaluate_main(common + ["--output-dir", str(test_output)])
        evaluate_main(common + ["--output-dir", str(external_output), "--manifest", str(external)])
    finally:
        torch.set_num_threads(old_threads)
    test_report = json.loads((test_output / "metrics.json").read_text(encoding="utf-8"))
    assert test_report["evaluation"] == "synthetic_test" and test_report["scored_lines"] == 1
    assert test_report["cer"] == pytest.approx(2 / 3)
    external_report = json.loads((external_output / "metrics.json").read_text(encoding="utf-8"))
    assert external_report["lines"] == 3 and external_report["scored_lines"] == 2
    assert external_report["cer"] == 0.5
    assert external_report["out_of_vocabulary_reference_characters"] == {"ॐ": 2}
    records = (external_output / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    predictions = {row["image_path"]: row for row in map(json.loads, records)}
    assert predictions["unlabeled.png"]["reference"] is None
    assert predictions["unlabeled.png"]["character_edits"] is None
    assert predictions["unlabeled.png"]["prediction"] == "क"
