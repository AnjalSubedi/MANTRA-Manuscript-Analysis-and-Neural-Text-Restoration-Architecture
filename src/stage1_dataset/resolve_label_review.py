"""Stage a label-reviewed release with reversible quarantine and authored fill.

Unavailable scans are never replaced by guessed transcriptions. Every warning
gets an evidence/disposition record. Unverified originals are quarantined;
replacement samples have new IDs and explicitly authored synthetic provenance.
This command stages artifacts only. Promote configuration after downstream QA.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
import unicodedata

from PIL import Image
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage1_dataset.chromium_text_renderer import get_chromium_renderer, shutdown_chromium_renderer
from stage1_dataset.font_policy import configured_font_splits, font_preserves_devanagari_digits
from stage1_dataset.generate_synthetic_images import calculate_readability_metrics
from stage1_dataset.label_quality import label_warnings, text_hash, evaluation_character_gaps
from stage1_dataset.render_text_line import check_font_supports_text, render_text_line

LEGACY_METADATA = "data/stage1_synthetic/metadata/synthetic_image_metadata_stage1_all_final.csv"
DEGREE_TEXTS = (
    "यस चित्रको कोण ३०° छ ।",
    "दुई रेखाबीचको कोण ४५° छ ।",
    "समकोणको मापन ९०° हुन्छ ।",
    "पूरा वृत्तको कोण ३६०° हुन्छ ।",
)
TEMPLATES = {
    "train": (
        "अभिलेख सङ्ख्या {n} मा गाउँको विवरण सुरक्षित राखिएको छ ।",
        "दर्ता सङ्ख्या {n} को पत्र सम्बन्धित कार्यालयमा पठाइयो ।",
        "सूची सङ्ख्या {n} अनुसार पुराना कागजको जाँच गरियो ।",
        "विवरण सङ्ख्या {n} मा किसानको नाम लेखिएको छ ।",
        "रसिद सङ्ख्या {n} को रकम खातामा चढाइयो ।",
        "आदेश सङ्ख्या {n} अनुसार नापीको काम सुरु भयो ।",
        "पत्र सङ्ख्या {n} मा जग्गाको सिमाना उल्लेख गरिएको छ ।",
        "खाता सङ्ख्या {n} मा आय र व्ययको हिसाब राखियो ।",
        "निर्णय सङ्ख्या {n} को प्रतिलिपि गाउँमा पठाइयो ।",
        "लेखोट सङ्ख्या {n} मा मिति र सही स्पष्ट देखिन्छ ।",
    ),
    "val": (
        "पाण्डुलिपि सङ्ख्या {n} को बाहिरी आवरण मर्मत गरियो ।",
        "पुस्तक सङ्ख्या {n} का पानाहरू सावधानीपूर्वक छुट्याइयो ।",
        "सङ्ग्रह सङ्ख्या {n} को परिचय नयाँ सूचीमा राखियो ।",
        "सामग्री सङ्ख्या {n} लाई सुक्खा कोठामा सुरक्षित राखियो ।",
        "वस्तु सङ्ख्या {n} को आकार र रङ विवरणमा टिपियो ।",
        "प्रतिवेदन सङ्ख्या {n} मा संरक्षण कार्यको अवस्था लेखियो ।",
    ),
    "test": (
        "पाठशाला सङ्ख्या {n} मा नयाँ पाठको अभ्यास गरियो ।",
        "अभ्यास सङ्ख्या {n} का अक्षरहरू ध्यान दिएर पढियो ।",
        "प्रश्न सङ्ख्या {n} को उत्तर विद्यार्थीले स्पष्ट लेखे ।",
        "पाठ सङ्ख्या {n} मा स्थानीय इतिहासको चर्चा गरिएको छ ।",
        "कार्य सङ्ख्या {n} पूरा भएपछि शिक्षकले हस्ताक्षर गरे ।",
        "टिपोट सङ्ख्या {n} मा अध्ययनको समय उल्लेख गरियो ।",
    ),
}


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def normalized_ocr(text):
    digits = str.maketrans("0123456789", "०१२३४५६७८९")
    return " ".join(unicodedata.normalize("NFC", text).translate(digits).split())


def trace_source(row, text_root, cache):
    """OCR matches establish lineage only, never independent verification."""
    candidates = [Path(text_root) / folder / row["source_file"] for folder in ("cleaned", "raw")]
    raw_name = row["source_file"].replace(".clean.txt", ".txt").replace("_clean.txt", ".txt")
    candidates.append(Path(text_root) / "raw" / raw_name)
    evidence = []
    wanted = normalized_ocr(row["text"])
    for path in dict.fromkeys(candidates):
        if not path.is_file():
            continue
        key = str(path)
        if key not in cache:
            content = path.read_text(encoding="utf-8")
            cache[key] = (normalized_ocr(content), hashlib.sha256(path.read_bytes()).hexdigest())
        content, digest = cache[key]
        location = content.find(wanted)
        evidence.append({"path": path.as_posix(), "sha256": digest,
                         "full_label_match": location >= 0,
                         "normalized_offset": location,
                         "context": content[max(0, location - 50):location + len(wanted) + 50] if location >= 0 else "",
                         "evidence_type": "ocr_lineage_only"})
    return evidence


def authored_specs(counts, font_groups, release):
    """Keep split counts; use disjoint authored template banks per split."""
    specs = []
    for split in ("train", "val", "test"):
        remaining = counts[split]
        if split == "train":
            degree_fonts = [p for p in font_groups["train"] if font_preserves_devanagari_digits(p, DEGREE_TEXTS[0])]
            needed = len(DEGREE_TEXTS) * len(degree_fonts)
            if not degree_fonts:
                raise ValueError("No training font preserves Devanagari numeral forms")
            if remaining < needed:
                raise ValueError("Not enough replacement slots for degree coverage")
            for i, text in enumerate(DEGREE_TEXTS):
                for variant, font in enumerate(degree_fonts, 1):
                    specs.append({"split": split, "text": text,
                                  "line_id": f"{release}_degree_{i + 1:04d}",
                                  "variant_index": str(variant), "font_path": font,
                                  "domain": "coverage_degree"})
            remaining -= needed
        for i in range(remaining):
            number = str(101 + 13 * i).translate(str.maketrans("0123456789", "०१२३४५६७८९"))
            text = TEMPLATES[split][i % len(TEMPLATES[split])].format(n=number)
            eligible_fonts = [p for p in font_groups[split] if font_preserves_devanagari_digits(p, text)]
            if not eligible_fonts:
                raise ValueError(f"No {split} font preserves the authored numeral forms")
            specs.append({"split": split, "text": text,
                          "line_id": f"{release}_{split}_{i + 1:04d}",
                          "variant_index": "1", "font_path": eligible_fonts[i % len(eligible_fonts)],
                          "domain": "coverage_verified_admin"})
    return specs


def render_authored(spec, image_dir, source_file, release):
    text = spec["text"]
    if label_warnings(text) or text != unicodedata.normalize("NFC", text):
        raise ValueError("Authored text has an unresolved label warning or normalization change")
    font = spec["font_path"]
    if not check_font_supports_text(font, text):
        raise ValueError(f"Authored label is unsupported by {font}")
    image, engine, status, reason = render_text_line(
        text, font, 44, 16, 16, renderer_preference="chromium_canvas"
    )
    if status != "success":
        raise RuntimeError(f"Authored sample failed rendering: {reason}")
    risk, score, contrast, std = calculate_readability_metrics(image)
    if risk:
        raise ValueError("Authored clean image failed readability check")
    path = Path(image_dir) / spec["split"] / f"{spec['line_id']}_v{int(spec['variant_index']):02d}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    image.save(temporary, format="PNG")
    temporary.replace(path)
    with Image.open(path) as saved:
        saved.load()
    params = {"profile_name": "D0_clean_print", "degradation_family": "single",
              "difficulty_bucket": "easy", "operation": "identity", "reason": "clean authored ground truth"}
    return {
        "image_path": path.as_posix(), "line_id": spec["line_id"], "text": text,
        "source_id": f"{release}_authored_{spec['split']}", "source_file": str(source_file),
        "domain": spec["domain"], "split": spec["split"],
        "source_text_original": text, "source_text_normalized": text, "source_text_rendered": text,
        "digit_normalization_applied": "false", "renderer": engine,
        "degradation_profile": "D0_clean_print", "degradation_family": "single",
        "degradation_parameters_json": json.dumps(params), "difficulty_bucket": "easy",
        "unreadable_risk": "false", "glyph_coverage_passed": "true", "font_fallback_used": "false",
        "requested_font_name": Path(font).stem, "actual_font_name": Path(font).stem,
        "font_name": Path(font).stem, "missing_glyphs": "", "variant_index": spec["variant_index"],
        "generation_round": "authored_review_replacement", "readability_score_estimate": str(score),
        "width": str(image.width), "height": str(image.height), "font_size": "44", "padding": "16",
        "label_review_status": "authored_exact_input", "label_text_sha256": text_hash(text),
        "font_sha256": hashlib.sha256(Path(font).read_bytes()).hexdigest(),
        "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "render_parameters_json": json.dumps({"font_size": 44, "padding": 16, "engine": engine,
                                               "browser_version": get_chromium_renderer().browser.version}),
        "replacement_kind": "new_authored_sample_not_transcription_correction",
    }


def label_rows(metadata):
    labels = {}
    for row in metadata:
        labels.setdefault(row["line_id"], {
            "line_id": row["line_id"], "source_id": row["source_id"], "source_file": row["source_file"],
            "text": row["text"], "line_length": str(len(row["text"])), "split": row["split"], "domain": row["domain"],
            "has_digit": str(any(c.isdigit() for c in row["text"])).lower(),
            "has_danda": str("।" in row["text"]).lower(), "has_double_danda": str("॥" in row["text"]).lower(),
            "has_conjunct": str("्" in row["text"]).lower(),
        })
    return list(labels.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", default=LEGACY_METADATA)
    parser.add_argument("--config", default="configs/stage1_synth100k_config.yaml")
    parser.add_argument("--review", default="outputs/rendering_audit_2026-10-27/transcription_review.csv")
    parser.add_argument("--release", default="label_review_v1")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9_]+", args.release):
        parser.error("release must use lowercase letters, digits, and underscores")
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    base = Path("data/stage1_synthetic/releases") / args.release
    base.mkdir(parents=True, exist_ok=True)
    original_bytes = Path(args.metadata).read_bytes()
    fingerprint = hashlib.sha256(original_bytes + Path(args.review).read_bytes()).hexdigest()
    state = base / "input.sha256"
    if state.exists() and state.read_text().strip() != fingerprint:
        raise ValueError("Release input changed; choose a new release name")
    if (base / "release_report.json").exists():
        print("Release is already staged: " + str(base), flush=True)
        return
    state.write_text(fingerprint, encoding="ascii")
    with open(args.metadata, encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    with open(args.review, encoding="utf-8-sig", newline="") as stream:
        review = {row["original_text"]: row for row in csv.DictReader(stream)}
    actual_warnings = {row["text"] for row in rows if label_warnings(row["text"])}
    if actual_warnings != set(review):
        raise ValueError("Review queue does not match all current warning texts")
    groups = defaultdict(list)
    for row in rows:
        if row["text"] in review:
            groups[row["text"]].append(row)
    quarantined = [row for row in rows if row["text"] in review]
    counts = Counter(row["split"] for row in quarantined)
    cache, decisions = {}, []
    for text, group in groups.items():
        item = review[text]
        evidence = trace_source(group[0], "data/stage1_synthetic/text", cache)
        decisions.append({
            "text_sha256": text_hash(text), "original_text": text, "category": item["category"],
            "warnings": item["issue"], "image_count": str(len(group)),
            "line_ids": ";".join(sorted({row["line_id"] for row in group})),
            "source_id": group[0]["source_id"], "source_file": group[0]["source_file"],
            "decision": "quarantined_unverified", "label_declared_wrong": "false",
            "verification_status": "not_independently_verified",
            "reason": "Original scan or authoritative matching-edition transcription is unavailable; OCR lineage alone cannot verify it.",
            "source_files_found": str(len(evidence)),
            "ocr_lineage_matches": str(sum(entry["full_label_match"] for entry in evidence)),
            "source_evidence_json": json.dumps(evidence, ensure_ascii=False),
            "independent_source_reference": "", "verified_text": "",
        })
    write_csv(base / "review/decisions.csv", decisions)
    (base / "review/input_review.csv").write_bytes(Path(args.review).read_bytes())
    (base / "review/generation_config.yaml").write_bytes(Path(args.config).read_bytes())
    write_csv(base / "quarantine/metadata.csv", quarantined)
    write_csv(base / "quarantine/labels.csv", label_rows(quarantined))
    approvals = {"approved_labels": [], "note": "Warnings are not wrong-label verdicts. No unavailable-source transcript has been approved."}
    (base / "review/approved_labels.json").write_text(json.dumps(approvals, indent=2), encoding="utf-8")
    fonts = configured_font_splits(config)
    specs = authored_specs(counts, fonts, args.release)
    source_file = base / "authored_source.csv"
    authored = []
    seen_labels = set()
    for spec in specs:
        if spec["line_id"] not in seen_labels:
            authored.append({"line_id": spec["line_id"], "split": spec["split"], "text": spec["text"],
                             "source_type": "explicitly_authored_synthetic_text", "historical_transcription": "false"})
            seen_labels.add(spec["line_id"])
    write_csv(source_file, authored)
    (base / "input_metadata.csv").write_bytes(original_bytes)
    replacements = defaultdict(list)
    try:
        for index, spec in enumerate(specs, 1):
            rendered = render_authored(spec, Path("data/stage1_synthetic/images/releases") / args.release,
                                       source_file.as_posix(), args.release)
            replacements[spec["split"]].append(rendered)
            if index % 50 == 0:
                print(f"Authored replacements: {index}/{len(specs)}", flush=True)
    finally:
        shutdown_chromium_renderer()
    released, positions = [], Counter()
    for row in rows:
        if row["text"] in review:
            replacement = dict(replacements[row["split"]][positions[row["split"]]])
            positions[row["split"]] += 1
            replacement["replaces_image_path"] = row["image_path"]
            replacement["replaces_line_id"] = row["line_id"]
            released.append(replacement)
        else:
            released.append({**row, "label_review_status": "no_warning_under_current_review_rules"})
    if len(released) != len(rows) or any(label_warnings(row["text"]) for row in released):
        raise ValueError("Staged release has unresolved warnings or changed sample count")
    gaps = evaluation_character_gaps(released)
    if any(gaps.values()):
        raise ValueError(f"Evaluation-only characters remain: {gaps}")
    write_csv(base / "metadata.csv", released)
    write_csv(base / "labels.csv", label_rows(released))
    write_csv(base / "replacement_mapping.csv", [row for row in released if row.get("replaces_image_path")])
    summary = {
        "release": args.release, "status": "STAGED_REQUIRES_DOWNSTREAM_QA", "rows": len(released),
        "split_counts": dict(Counter(row["split"] for row in released)),
        "quarantined_images": len(quarantined), "quarantined_unique_labels": len(decisions),
        "structural_images": sum(len(groups[text]) for text, item in review.items() if item["category"] == "detached_vowel_or_halant"),
        "other_mark_images": sum(len(groups[text]) for text, item in review.items() if item["category"] != "detached_vowel_or_halant"),
        "original_transcriptions_independently_verified": 0,
        "original_transcriptions_declared_wrong": 0,
        "warning_cases_with_local_ocr_lineage": sum(bool(json.loads(item["source_evidence_json"])) for item in decisions),
        "replacement_images": len(specs), "authored_unique_labels": len(authored),
        "release_warning_rows": 0,
        "degree_images_by_split": {split: sum("°" in row["text"] for row in released if row["split"] == split) for split in ("train", "val", "test")},
        "evaluation_character_gaps": gaps,
        "preserved_accent_occurrences": {f"U+{ord(ch):04X}": sum(row["text"].count(ch) for row in released) for ch in ("॑", "॒")},
        "original_metadata_sha256": hashlib.sha256(original_bytes).hexdigest(),
        "limitations": "Original flagged transcriptions remain unverified in quarantine. Release screening is not a full human transcription certificate.",
    }
    (base / "release_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
