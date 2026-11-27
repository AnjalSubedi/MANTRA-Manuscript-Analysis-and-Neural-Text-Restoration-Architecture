import pytest
import torch
from torch.nn import functional as F

from stage2_htr.config import dataset_options, load_data_config, make_dataloader
from stage2_htr.ctc import ctc_loss, greedy_decode
from stage2_htr.dataset import HTRLineDataset
from stage2_htr.metrics import character_error_counts, character_error_rate
from stage2_htr.model import CRNN


def test_crnn_and_ctc_backpropagate_on_a_mixed_width_loader_batch(data_files):
    image, manifest, config, charset = data_files
    path = manifest([
        {"image_path": str(image("short.png", width=37, height=64)), "text": "कक"},
        {"image_path": str(image("long.png", width=65, height=64)), "text": "क ख"},
    ])
    settings = load_data_config(config(path))
    dataset = HTRLineDataset(path, charset, **dataset_options(settings))
    batch = next(iter(make_dataloader(dataset, settings)))
    model = CRNN(dataset.num_classes, image_height=dataset.image_height, hidden_size=8)

    probabilities, lengths = model(batch["images"], batch["processed_widths"])
    loss = ctc_loss(probabilities, batch["labels"], lengths, batch["label_lengths"])
    loss.backward()

    assert torch.isfinite(loss)
    gradients = [parameter.grad for parameter in model.parameters()]
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
    assert model.feature_blocks[0][0].weight.grad.abs().sum() > 0
    assert model.classifier.weight.grad.abs().sum() > 0


def test_ctc_requires_an_extra_step_for_adjacent_repeated_target_ids():
    log_probs = torch.zeros(3, 2, 3).log_softmax(dim=-1)
    targets = torch.tensor([1, 1, 1, 2])
    sizes = torch.tensor([2, 2])

    with pytest.raises(ValueError, match="sample 0: needs 3, has 2"):
        ctc_loss(log_probs, targets, torch.tensor([2, 2]), sizes)
    assert torch.isfinite(ctc_loss(log_probs, targets, torch.tensor([3, 2]), sizes))


def test_ctc_padded_steps_do_not_contribute_to_loss_or_gradients():
    logits = torch.zeros(5, 2, 3, requires_grad=True)
    changed = logits.detach().clone()
    changed[3:, 0, 2] = 100
    targets = torch.tensor([1, 1, 2])
    lengths = torch.tensor([3, 5])
    sizes = torch.tensor([2, 1])

    loss = ctc_loss(logits.log_softmax(dim=-1), targets, lengths, sizes)
    torch.testing.assert_close(loss, ctc_loss(changed.log_softmax(dim=-1), targets, lengths, sizes))
    loss.backward()
    assert torch.count_nonzero(logits.grad[3:, 0]) == 0
    assert torch.count_nonzero(logits.grad[:3, 0]) > 0


def test_ctc_rejects_blanks_in_targets():
    log_probs = torch.zeros(4, 1, 3).log_softmax(dim=-1)
    with pytest.raises(ValueError, match="valid nonblank"):
        ctc_loss(log_probs, torch.tensor([0]), torch.tensor([4]), torch.tensor([1]))


def test_greedy_decoding_preserves_repeats_separated_by_blank_and_ignores_padding():
    paths = torch.tensor([
        [1, 1, 0, 1, 1, 2, 2],
        [0, 2, 2, 0, 1, 1, 0],
        [0, 0, 2, 2, 2, 2, 2],
    ]).transpose(0, 1)
    log_probs = F.one_hot(paths, num_classes=3).float().log_softmax(dim=-1)
    predictions = greedy_decode(log_probs, torch.tensor([5, 7, 2]), {0: "<BLANK>", 1: "क", 2: "ख"})

    assert predictions == ["कक", "खक", ""]


def test_cer_weights_by_reference_codepoints():
    predictions = ["शरी", ""]
    references = ["श्री", "क"]
    assert character_error_counts(predictions, references) == (2, 5)
    assert character_error_rate(predictions, references) == pytest.approx(0.4)
    assert character_error_rate(["क ख"], ["कख"]) == pytest.approx(0.5)


def test_cer_rejects_mismatched_lines_and_empty_corpus():
    with pytest.raises(ValueError, match="same number of lines"):
        character_error_rate(["क"], [])
    with pytest.raises(ValueError, match="at least one reference character"):
        character_error_rate(["क"], [""])
