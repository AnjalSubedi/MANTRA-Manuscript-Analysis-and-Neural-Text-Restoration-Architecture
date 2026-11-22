from stage1_dataset.audit_rendering import label_warnings, structural_mark_warning, create_screened_manifests
from stage1_dataset.audit_ocr_review import edit_distance


def test_valid_conjuncts_and_attached_marks_are_not_flagged():
    assert label_warnings("श्री सुब्बा क्ष त्र ज्ञ कि की क॑ क॒") == []


def test_detached_vowels_and_halants_are_reported_without_rewriting():
    text = "देषमाला घंशावली ु"
    assert "U+0941" in label_warnings(text)[0]
    assert "U+094D" in label_warnings("्सच्चेपले")[0]


def test_vedic_marks_are_review_candidates_rather_than_silently_removed():
    assert label_warnings("अ॒") == []
    assert label_warnings("॒अ")


def test_ocr_distance_counts_codepoints_without_changing_ground_truth():
    assert edit_distance("श्री", "श्री") == 0
    assert edit_distance("श्री", "शरी") == 1
    assert edit_distance("सुब्बा", "सुब्ब") == 1


def test_structural_filter_keeps_accent_review_distinct():
    assert structural_mark_warning("word_initial_combining_mark@0:U+094D")
    assert structural_mark_warning("word_initial_combining_mark@3:U+0941")
    assert not structural_mark_warning("word_initial_combining_mark@0:U+0952")


def test_screening_preserves_active_files_and_kept_transcriptions(tmp_path):
    import csv
    source = tmp_path / "active"
    source.mkdir()
    original = "image_path,text\ngood.png,श्री सुब्बा\nbad.png,्word\n"
    for split in ("all", "train", "val", "test"):
        (source / f"htr_{split}.csv").write_text(original, encoding="utf-8")
    out = tmp_path / "screened"
    assert create_screened_manifests(source, out, {"bad.png"}) == dict.fromkeys(("all", "train", "val", "test"), 1)
    assert (source / "htr_all.csv").read_text(encoding="utf-8") == original
    with (out / "htr_all.csv").open(encoding="utf-8") as stream:
        assert list(csv.DictReader(stream)) == [{"image_path": "good.png", "text": "श्री सुब्बा"}]
