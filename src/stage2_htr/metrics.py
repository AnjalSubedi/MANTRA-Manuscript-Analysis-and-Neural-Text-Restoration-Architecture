"""Edit-distance metrics for HTR transcriptions."""

from collections.abc import Sequence


def edit_distance(first: str, second: str) -> int:
    """Count the minimum insertions, deletions, and substitutions."""
    previous = list(range(len(second) + 1))
    for i, left in enumerate(first, 1):
        current = [i]
        for j, right in enumerate(second, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def character_error_counts(predictions: Sequence[str], references: Sequence[str]) -> tuple[int, int]:
    """Return total edits and total reference code points for corpus-level CER."""
    if len(predictions) != len(references):
        raise ValueError("predictions and references must contain the same number of lines")
    edits = sum(edit_distance(prediction, reference) for prediction, reference in zip(predictions, references))
    return edits, sum(map(len, references))


def character_error_rate(predictions: Sequence[str], references: Sequence[str]) -> float:
    """Return corpus-level CER as a ratio over Unicode code points, including spaces."""
    edits, reference_length = character_error_counts(predictions, references)
    if reference_length == 0:
        raise ValueError("CER requires at least one reference character")
    return edits / reference_length
