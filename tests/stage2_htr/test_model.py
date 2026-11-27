import pytest
import torch
from torch.nn import functional as F

from stage2_htr.config import dataset_options, load_data_config, make_dataloader
from stage2_htr.dataset import HTRLineDataset
from stage2_htr.model import CRNN


def test_crnn_accepts_stage2_batch_and_returns_exact_ctc_lengths(data_files):
    image, manifest, config, charset = data_files
    path = manifest([
        {"image_path": str(image("short.png", width=37, height=64)), "text": "क"},
        {"image_path": str(image("long.png", width=65, height=64)), "text": "क ख"},
    ])
    settings = load_data_config(config(path))
    dataset = HTRLineDataset(path, charset, **dataset_options(settings))
    batch = next(iter(make_dataloader(dataset, settings)))
    model = CRNN(dataset.num_classes, image_height=dataset.image_height, hidden_size=8).eval()

    with torch.inference_mode():
        log_probs, lengths = model(batch["images"], batch["processed_widths"])

    assert lengths.tolist() == [9, 16]
    assert lengths.dtype == torch.int64 and lengths.device.type == "cpu"
    assert log_probs.shape == (16, 2, dataset.num_classes)
    assert torch.isfinite(log_probs).all()
    torch.testing.assert_close(log_probs.logsumexp(dim=-1), torch.zeros(16, 2))


def test_crnn_valid_steps_are_independent_of_batch_padding():
    model = CRNN(4, image_height=32, hidden_size=8).eval()
    short = torch.rand(1, 1, 32, 39)
    long = torch.rand(1, 1, 32, 74)
    images = torch.cat([F.pad(short, (0, 35), value=1.0), long])

    with torch.inference_mode():
        alone, alone_length = model(short, torch.tensor([39]))
        batched, batch_lengths = model(images, torch.tensor([39, 74]))

    assert batch_lengths.tolist() == [9, 18]
    assert alone_length.tolist() == [9]
    torch.testing.assert_close(alone[:, 0], batched[:9, 0])


@pytest.mark.parametrize("width", [3, 33])
def test_crnn_rejects_widths_outside_the_cnn_input(width):
    model = CRNN(4, hidden_size=8)
    with pytest.raises(ValueError, match="processed widths"):
        model(torch.zeros(1, 1, 64, 32), torch.tensor([width]))
