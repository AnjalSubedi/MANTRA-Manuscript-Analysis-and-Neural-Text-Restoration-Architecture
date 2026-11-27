"""CTC loss and greedy decoding for the Stage 2 recognizer."""

from itertools import groupby

import torch
from torch.nn import functional as F

from stage2_htr.dataset import decode_ids


def _checked_input_lengths(
    log_probs: torch.Tensor, input_lengths: torch.Tensor, blank_id: int,
) -> torch.Tensor:
    if log_probs.ndim != 3 or log_probs.shape[1] == 0:
        raise ValueError("log_probs must have shape [time, batch, classes] for a nonempty batch")
    if not 0 <= blank_id < log_probs.shape[2]:
        raise ValueError("blank_id must be a valid class index")
    if input_lengths.shape != (log_probs.shape[1],) or input_lengths.dtype not in (torch.int32, torch.int64):
        raise ValueError("input_lengths must contain one integer length per image")
    lengths = input_lengths.to(device="cpu", dtype=torch.long)
    if ((lengths < 1) | (lengths > log_probs.shape[0])).any():
        raise ValueError("input_lengths must be between 1 and the number of model time steps")
    return lengths


def ctc_loss(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    *,
    blank_id: int = 0,
) -> torch.Tensor:
    """Compute mean CTC loss for flat targets and reject impossible alignments."""
    lengths = _checked_input_lengths(log_probs, input_lengths, blank_id)
    if targets.ndim != 1 or targets.dtype not in (torch.int32, torch.int64):
        raise ValueError("targets must be a flat integer tensor")
    if target_lengths.shape != lengths.shape or target_lengths.dtype not in (torch.int32, torch.int64):
        raise ValueError("target_lengths must contain one integer length per image")
    sizes = target_lengths.to(device="cpu", dtype=torch.long)
    if (sizes < 0).any() or sizes.sum().item() != targets.numel():
        raise ValueError("target_lengths must be nonnegative and sum to the number of target IDs")
    target_ids = targets.to(device="cpu", dtype=torch.long)
    if ((target_ids < 0) | (target_ids >= log_probs.shape[2]) | (target_ids == blank_id)).any():
        raise ValueError("target IDs must be valid nonblank class indices")

    # Equal neighbors need an intervening blank step within each target.
    repeated = torch.stack([
        (sequence[:-1] == sequence[1:]).sum()
        for sequence in target_ids.split(sizes.tolist())
    ])
    required = sizes + repeated
    impossible = (required > lengths).nonzero(as_tuple=True)[0].tolist()
    if impossible:
        details = ", ".join(
            f"sample {index}: needs {required[index].item()}, has {lengths[index].item()}"
            for index in impossible
        )
        raise ValueError(f"CTC alignment requires more input steps ({details})")

    if log_probs.dtype in (torch.float16, torch.bfloat16):
        log_probs = log_probs.float()
    return F.ctc_loss(
        log_probs, targets.to(device=log_probs.device, dtype=torch.long),
        lengths, sizes, blank=blank_id, reduction="mean",
    )


def greedy_decode(
    log_probs: torch.Tensor,
    input_lengths: torch.Tensor,
    id_to_char: dict[int, str],
    *,
    blank_id: int = 0,
) -> list[str]:
    """Decode the most likely class per valid step, merging repeats before stripping blanks."""
    lengths = _checked_input_lengths(log_probs, input_lengths, blank_id).tolist()
    paths = log_probs.argmax(dim=-1).transpose(0, 1).tolist()
    return [
        decode_ids([token for token, _ in groupby(path[:length]) if token != blank_id],
                   id_to_char, remove_blank=False)
        for path, length in zip(paths, lengths)
    ]
