"""Conservative label-review and evaluation-character coverage gates.

A warning requests source verification; it does not declare a historical form
incorrect. Exact-text approvals require an independent source reference.
"""
import hashlib
import json
from pathlib import Path
import unicodedata


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def label_warnings(text):
    warnings = []
    for index, char in enumerate(text):
        if unicodedata.category(char).startswith("M") and (index == 0 or text[index - 1].isspace()):
            warnings.append(f"word_initial_combining_mark@{index}:U+{ord(char):04X}")
        if unicodedata.category(char) == "Cc":
            warnings.append(f"control_character@{index}:U+{ord(char):04X}")
    return warnings


def load_approvals(path=None):
    if path is None:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    approvals = {}
    for entry in data.get("approved_labels", []):
        if (entry.get("decision") != "verified_unchanged" or not entry.get("source_reference")
                or entry.get("evidence_type") not in ("source_scan", "authoritative_transcription")):
            raise ValueError("Label approval requires a verified decision and independent source reference")
        text = entry["text"]
        approvals[text_hash(text)] = entry
    return approvals


def needs_review(text, approvals=None):
    return bool(label_warnings(text)) and text_hash(text) not in (approvals or {})


def require_reviewed_label(text, approvals=None):
    if needs_review(text, approvals):
        raise ValueError(f"Label requires source review before rendering: {label_warnings(text)}")


def evaluation_character_gaps(rows):
    training = set()
    evaluation = {"val": set(), "test": set()}
    for row in rows:
        if row["split"] == "train":
            training.update(row["text"])
        elif row["split"] in evaluation:
            evaluation[row["split"]].update(row["text"])
    return {split: sorted(characters - training) for split, characters in evaluation.items()}
