import json

import pytest

from stage1_dataset.label_quality import (
    label_warnings, load_approvals, needs_review, require_reviewed_label,
    evaluation_character_gaps, text_hash,
)
from stage1_dataset.resolve_label_review import authored_specs, trace_source


def test_valid_attached_accents_are_preserved_and_not_flagged():
    text = "अग्नि॑म् इ॒न्द्रः श्री सुब्बा"
    assert label_warnings(text) == []
    assert not needs_review(text)


def test_warning_can_be_verified_instead_of_automatically_rejected(tmp_path):
    text = "॒अ"
    path = tmp_path / "approvals.json"
    path.write_text(json.dumps({"approved_labels": [{
        "text": text, "decision": "verified_unchanged", "evidence_type": "source_scan",
        "source_reference": "local-scan.pdf#page=12",
    }]}), encoding="utf-8")
    approvals = load_approvals(path)
    assert not needs_review(text, approvals)
    assert needs_review("॒आ", approvals)
    require_reviewed_label(text, approvals)


def test_circular_ocr_evidence_does_not_approve_a_warning(tmp_path):
    path = tmp_path / "approvals.json"
    path.write_text(json.dumps({"approved_labels": [{
        "text": "॒अ", "decision": "verified_unchanged", "evidence_type": "ocr_text",
        "source_reference": "same_ocr.txt",
    }]}), encoding="utf-8")
    with pytest.raises(ValueError, match="independent source"):
        load_approvals(path)
    with pytest.raises(ValueError, match="source review"):
        require_reviewed_label("्अ")


def test_trace_records_ocr_lineage_without_calling_it_verified(tmp_path):
    cleaned = tmp_path / "cleaned"
    cleaned.mkdir()
    (cleaned / "book.clean.txt").write_text("पूर्वपाठ देषमाला घंशावली ु अन्त्यपाठ", encoding="utf-8")
    evidence = trace_source({"source_file": "book.clean.txt", "text": "देषमाला घंशावली ु"}, tmp_path, {})
    assert evidence[0]["full_label_match"]
    assert evidence[0]["evidence_type"] == "ocr_lineage_only"
    assert evidence[0]["sha256"]


def test_replacement_bank_preserves_counts_and_splits_without_label_leakage():
    fonts = {"train": [f"train{i}.ttf" for i in range(8)], "val": ["val.ttf"], "test": ["test.ttf"]}
    counts = {"train": 308, "val": 44, "test": 30}
    specs = authored_specs(counts, fonts, "release")
    assert len(specs) == 382
    labels = {s: {r["text"] for r in specs if r["split"] == s} for s in counts}
    assert all(sum(r["split"] == s for r in specs) == n for s, n in counts.items())
    assert not labels["train"] & labels["val"]
    assert not labels["train"] & labels["test"]
    assert not labels["val"] & labels["test"]
    assert all(not label_warnings(r["text"]) for r in specs)
    degree = [r for r in specs if "°" in r["text"]]
    assert len(degree) == 32 and {r["split"] for r in degree} == {"train"}
    assert {r["font_path"] for r in degree} == set(fonts["train"])
    assert len({(r["line_id"], r["variant_index"]) for r in specs}) == 382


def test_eval_character_gate_requires_training_coverage():
    rows = [{"split": "train", "text": "क"}, {"split": "val", "text": "क°"}]
    assert evaluation_character_gaps(rows) == {"val": ["°"], "test": []}
    rows.append({"split": "train", "text": "३०°"})
    assert evaluation_character_gaps(rows) == {"val": [], "test": []}
