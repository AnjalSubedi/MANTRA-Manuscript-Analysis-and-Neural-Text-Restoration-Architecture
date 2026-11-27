import json

import pytest
import torch

from stage2_htr.overfit_batch import main


def test_tiny_batch_checkpoint_replays_predictions_and_detects_changed_inputs(data_files, tmp_path):
    image, manifest, config, _ = data_files
    rows = [
        {"image_path": str(image("short.png", width=37, height=64)), "text": "कक"},
        {"image_path": str(image("long.png", width=65, height=64)), "text": "क ख"},
    ]
    settings = config(manifest(rows))
    output = tmp_path / "tiny_batch"
    args = ["--config", str(settings), "--threads", "1"]
    verify = [*args, "verify", str(output / "checkpoint.pt")]
    original_threads = torch.get_num_threads()
    try:
        with torch.random.fork_rng(devices=[]):
            result = main([*args, "train", "--indices", "0", "1", "--max-steps", "1",
                           "--output-dir", str(output)])
        report = json.loads((output / "report.json").read_text(encoding="utf-8"))
        assert result == int(report["status"] != "PASS")
        assert [entry["step"] for entry in report["history"]] == [0, 1]
        assert main(verify) == 0

        rows[0]["text"] = "खक"
        manifest(rows)
        with pytest.raises(SystemExit) as error:
            main(verify)
        assert error.value.code == 1
    finally:
        torch.set_num_threads(original_threads)
