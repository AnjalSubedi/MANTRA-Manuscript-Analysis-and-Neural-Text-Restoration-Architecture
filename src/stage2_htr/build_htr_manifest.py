import os
import sys
import csv
import json
import argparse
import re
import random
import multiprocessing
from pathlib import Path
from collections import Counter
from PIL import Image
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage1_dataset.font_policy import font_split_errors
from stage1_dataset.label_quality import load_approvals, needs_review, evaluation_character_gaps

def process_row_worker(item):
    """Processes a single row image validation. Executed in worker processes."""
    r, helper_info = item
    image_path = (r.get("image_path") or "").strip()
    text = r.get("text") or ""
    split = (r.get("split") or "").strip().lower()
    line_id = (r.get("line_id") or "").strip()
    missing_required = not image_path or not split or not line_id
        
    exists = os.path.isfile(image_path)
    is_blank = not text.strip()
    
    # Regex inspections
    has_ascii_digit = bool(re.search(r'[0-9]', text))
    has_latin_letter = bool(re.search(r'[a-zA-Z]', text))
    
    bad_ocr_pattern = re.compile(r'[_|&%€$£₹«»\[\]]')
    has_bad_ocr = bool(bad_ocr_pattern.search(text))
    
    # Decode pixels even when dimensions already exist in metadata: PNG
    # headers alone cannot establish that an image will load during training.
    width, height = "", ""
    unreadable = False
    if exists:
        try:
            with Image.open(image_path) as img:
                img.load()
                width, height = img.size
        except (OSError, ValueError):
            unreadable = True
            
    # Resolve flags
    has_digit = (r.get("has_digit") or "").strip()
    has_danda = (r.get("has_danda") or "").strip()
    has_double_danda = (r.get("has_double_danda") or "").strip()
    has_conjunct = (r.get("has_conjunct") or "").strip()
    
    if not has_digit:
        has_digit = helper_info.get("has_digit", "")
        if not has_digit:
            has_digit = "true" if any(c in "०१२३४५६७८९" or c.isdigit() for c in text) else "false"
            
    if not has_danda:
        has_danda = helper_info.get("has_danda", "")
        if not has_danda:
            has_danda = "true" if "।" in text else "false"
            
    if not has_double_danda:
        has_double_danda = helper_info.get("has_double_danda", "")
        if not has_double_danda:
            has_double_danda = "true" if "॥" in text else "false"
            
    if not has_conjunct:
        has_conjunct = helper_info.get("has_conjunct", "")
        if not has_conjunct:
            has_conjunct = "true" if "\u094d" in text else "false"
            
    return {
        "status": "ok",
        "missing_required": missing_required,
        "unreadable": unreadable,
        "invalid_split": split not in ("train", "val", "test"),
        "image_path": image_path,
        "text": text,
        "split": split,
        "line_id": line_id,
        "source_id": r.get("source_id", ""),
        "source_file": r.get("source_file", ""),
        "domain": r.get("domain", ""),
        "font_name": r.get("actual_font_name") or r.get("font_name", ""),
        "degradation_profile": r.get("degradation_profile", ""),
        "degradation_family": r.get("degradation_family", ""),
        "difficulty_bucket": r.get("difficulty_bucket", ""),
        "variant_index": r.get("variant_index", ""),
        "generation_round": r.get("generation_round", ""),
        "renderer": r.get("renderer", ""),
        "width": str(width) if width else "",
        "height": str(height) if height else "",
        "has_digit": has_digit,
        "has_danda": has_danda,
        "has_double_danda": has_double_danda,
        "has_conjunct": has_conjunct,
        "exists": exists,
        "is_blank": is_blank,
        "has_ascii_digit": has_ascii_digit,
        "has_latin_letter": has_latin_letter,
        "has_bad_ocr": has_bad_ocr
    }

def main(argv=None):
    parser = argparse.ArgumentParser(description="MANTRA Stage 2.1 HTR Manifest Builder")
    parser.add_argument(
        "--metadata",
        type=str,
        default=None,
        help="Path to final Stage 1 metadata CSV"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save generated CSV manifests"
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default=None,
        help="Directory to save QA reports"
    )
    parser.add_argument("--config", default="configs/stage1_synth100k_config.yaml",
                        help="Stage 1 configuration defining the font holdouts")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--data-config", default="configs/stage2_data_synth100k.yaml")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("workers must be positive")
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    args.metadata = args.metadata or config.get("paths", {}).get(
        "final_metadata", "data/stage1_synthetic/metadata/synthetic_image_metadata_stage1_all_final.csv"
    )
    if args.output_dir is None or args.report_dir is None:
        data_config = yaml.safe_load(Path(args.data_config).read_text(encoding="utf-8"))
        manifest_dir = Path(data_config["paths"]["manifest_all"]).parent
        args.output_dir = args.output_dir or str(manifest_dir)
        args.report_dir = args.report_dir or str(manifest_dir.parent / "reports")
    approvals = load_approvals(config.get("label_review", {}).get("approved_labels_json"))

    print("=" * 70)
    print("MANTRA STAGE 2.1 HTR MANIFEST GENERATION PIPELINE (PARALLEL)")
    print("=" * 70)
    
    # 1. Paths verification
    if not os.path.exists(args.metadata):
        print(f"ERROR: Final Stage 1 metadata CSV not found at: {args.metadata}")
        sys.exit(1)
        
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.report_dir, exist_ok=True)
    
    # 2. Loading helper labels CSV if available for fallback flags
    helper_label_path = "data/stage1_synthetic/labels/stage1_all_lines.csv"
    label_helper_map = {}
    if os.path.exists(helper_label_path):
        print(f"Loading secondary helper labels from: {helper_label_path}")
        with open(helper_label_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                label_helper_map[r["line_id"]] = {
                    "has_digit": r.get("has_digit", ""),
                    "has_danda": r.get("has_danda", ""),
                    "has_double_danda": r.get("has_double_danda", ""),
                    "has_conjunct": r.get("has_conjunct", "")
                }
        print(f"Loaded {len(label_helper_map)} helper labels successfully.")
    else:
        print("Warning: Helper labels CSV not found. Will calculate flags dynamically.")

    # 3. Read metadata
    print(f"Reading metadata from: {args.metadata}")
    meta_rows = []
    with open(args.metadata, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing_columns = sorted({"image_path", "text", "split", "line_id"} - set(reader.fieldnames or []))
        for r in reader:
            meta_rows.append(r)
    total_input_rows = len(meta_rows)
    print(f"Loaded {total_input_rows} rows from metadata.")

    # 4. Processing manifests rows in parallel
    print("Processing manifests rows in parallel & resolving dimensions...")
    
    # Prepare items for multiprocessing pool
    work_items = []
    for r in meta_rows:
        line_id = r.get("line_id", "")
        helper_info = label_helper_map.get(line_id, {})
        work_items.append((r, helper_info))
        
    # Spawn context pool
    num_workers = args.workers
    print(f"Using {num_workers} processes for manifest builds...")
    
    ctx = multiprocessing.get_context("spawn")
    results = []
    if num_workers == 1:
        results = list(map(process_row_worker, work_items))
    else:
        with ctx.Pool(processes=num_workers) as pool:
            results = pool.map(process_row_worker, work_items)
        
    # Aggregate results
    output_rows = []
    missing_image_paths = 0
    blank_text_rows = 0
    duplicate_image_paths = 0
    duplicate_line_id_variants = 0
    ascii_digit_rows = 0
    latin_letter_rows = 0
    bad_ocr_symbol_rows = 0
    missing_required_rows = 0
    unreadable_images = 0
    invalid_split_rows = 0
    
    seen_image_paths = set()
    seen_image_paths_by_split = {"train": set(), "val": set(), "test": set()}
    seen_line_id_variants = set()
    
    # Categorical distributions
    domain_counts = Counter()
    split_counts = Counter()
    domain_split_counts = Counter()
    gen_round_counts = Counter()
    variant_counts = Counter()
    renderer_counts = Counter()
    difficulty_counts = Counter()
    degradation_counts = Counter()
    
    for idx, res in enumerate(results):
        missing_required_rows += int(res["missing_required"])
        unreadable_images += int(res["unreadable"])
        invalid_split_rows += int(res["invalid_split"])
            
        image_path = res["image_path"]
        text = res["text"]
        split = res["split"]
        line_id = res["line_id"]
        variant_idx = res["variant_index"]
        
        # QA Stats update
        if not res["exists"]:
            missing_image_paths += 1
        if res["is_blank"]:
            blank_text_rows += 1
            
        canonical_path = os.path.normcase(os.path.abspath(image_path))
        if canonical_path in seen_image_paths:
            duplicate_image_paths += 1
        seen_image_paths.add(canonical_path)
        
        sp_lower = split.lower()
        if sp_lower in seen_image_paths_by_split:
            seen_image_paths_by_split[sp_lower].add(image_path)
            
        line_var_key = (line_id, variant_idx)
        if line_var_key in seen_line_id_variants:
            duplicate_line_id_variants += 1
        seen_line_id_variants.add(line_var_key)
        
        if res["has_ascii_digit"]:
            ascii_digit_rows += 1
        if res["has_latin_letter"]:
            latin_letter_rows += 1
        if res["has_bad_ocr"]:
            bad_ocr_symbol_rows += 1
            
        # Distributions update
        domain = res["domain"]
        domain_counts[domain] += 1
        split_counts[split] += 1
        domain_split_counts[(domain, split)] += 1
        gen_round_counts[res["generation_round"]] += 1
        variant_counts[variant_idx] += 1
        renderer_counts[res["renderer"]] += 1
        difficulty_counts[res["difficulty_bucket"]] += 1
        degradation_counts[res["degradation_profile"]] += 1
        
        output_rows.append({
            "image_path": image_path,
            "text": text,
            "split": split,
            "line_id": line_id,
            "source_id": res["source_id"],
            "source_file": res["source_file"],
            "domain": domain,
            "font_name": res["font_name"],
            "degradation_profile": res["degradation_profile"],
            "degradation_family": res["degradation_family"],
            "difficulty_bucket": res["difficulty_bucket"],
            "variant_index": variant_idx,
            "generation_round": res["generation_round"],
            "renderer": res["renderer"],
            "width": res["width"],
            "height": res["height"],
            "has_digit": res["has_digit"],
            "has_danda": res["has_danda"],
            "has_double_danda": res["has_double_danda"],
            "has_conjunct": res["has_conjunct"]
        })

    # 5. Write manifest CSV files
    fieldnames = [
        "image_path", "text", "split", "line_id", "source_id", "source_file", "domain",
        "font_name", "degradation_profile", "degradation_family", "difficulty_bucket",
        "variant_index", "generation_round", "renderer", "width", "height",
        "has_digit", "has_danda", "has_double_danda", "has_conjunct"
    ]
    
    # Group by split
    splits_data = {"train": [], "val": [], "test": []}
    for row in output_rows:
        sp_lower = row["split"].lower()
        if sp_lower in splits_data:
            splits_data[sp_lower].append(row)
            
    # Publication is deferred until every hard QA gate passes. Existing
    # manifests stay intact on failure, while diagnostics are always saved.

    # 6. Compute duplicate images in split stats
    dup_within_split = 0
    for split_name, paths in seen_image_paths_by_split.items():
        split_rows = [row for row in output_rows if row["split"].lower() == split_name]
        paths_count = Counter(row["image_path"] for row in split_rows)
        for p, count in paths_count.items():
            if count > 1:
                dup_within_split += (count - 1)

    # Validate label/line grouping as well as unique physical paths.
    cross_split = {}
    for key in ("text", "line_id"):
        groups = {}
        for row in output_rows:
            groups.setdefault(row[key], set()).add(row["split"])
        cross_split[key] = sum(len(splits) > 1 for splits in groups.values())
    font_policy_error = ""
    try:
        font_violations = len(font_split_errors(output_rows, config))
    except ValueError as error:
        font_violations = len(output_rows)
        font_policy_error = str(error)
    failures = {
        "empty_input": int(total_input_rows == 0),
        "missing_columns": len(missing_columns),
        "missing_required_rows": missing_required_rows,
        "missing_image_paths": missing_image_paths,
        "unreadable_images": unreadable_images,
        "invalid_split_rows": invalid_split_rows,
        "blank_text_rows": blank_text_rows,
        "duplicate_image_paths": duplicate_image_paths,
        "duplicate_line_id_variants": duplicate_line_id_variants,
        "ascii_digit_rows": ascii_digit_rows,
        "latin_letter_rows": latin_letter_rows,
        "bad_ocr_symbol_rows": bad_ocr_symbol_rows,
        "cross_split_texts": cross_split["text"],
        "cross_split_line_ids": cross_split["line_id"],
        "font_split_violations": font_violations,
        "font_policy_error": int(bool(font_policy_error)),
    }
    gaps = evaluation_character_gaps(output_rows)
    failures["unapproved_label_warning_rows"] = sum(needs_review(row["text"], approvals) for row in output_rows)
    failures["validation_only_characters"] = len(gaps["val"])
    failures["test_only_characters"] = len(gaps["test"])
    qa_passed = not any(failures.values())
    qa_status = "PASS" if qa_passed else "FAIL"
    # 7. Write QA report files (JSON and Markdown)
    qa_report_json = {
        "qa_status": qa_status,
        "validation": failures,
        "missing_columns": missing_columns,
        "font_policy_error": font_policy_error,
        "evaluation_character_gaps": {split: [{"character": c, "codepoint": f"U+{ord(c):04X}"} for c in chars]
                                      for split, chars in gaps.items()},
        "summary": {
            "total_rows": total_input_rows,
            "train_rows": len(splits_data["train"]),
            "val_rows": len(splits_data["val"]),
            "test_rows": len(splits_data["test"]),
            "missing_image_paths": missing_image_paths,
            "blank_text_rows": blank_text_rows,
            "duplicate_image_paths": duplicate_image_paths,
            "duplicate_image_paths_within_splits": dup_within_split,
            "duplicate_line_id_variants": duplicate_line_id_variants,
            "ascii_digit_rows": ascii_digit_rows,
            "latin_letter_rows": latin_letter_rows,
            "bad_ocr_symbol_rows": bad_ocr_symbol_rows
        },
        "distributions": {
            "domain": dict(domain_counts),
            "split": dict(split_counts),
            "generation_round": dict(gen_round_counts),
            "variant_index": dict(variant_counts),
            "renderer": dict(renderer_counts),
            "difficulty_bucket": dict(difficulty_counts),
            "degradation_profile": dict(degradation_counts)
        }
    }
    
    json_path = os.path.join(args.report_dir, "htr_manifest_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(qa_report_json, f, indent=4)
    print(f"Saved QA report JSON to: {json_path}")
    
    # Samples for Markdown report
    random.seed(42)
    train_samples = random.sample(splits_data["train"], min(5, len(splits_data["train"]))) if splits_data["train"] else []
    val_samples = random.sample(splits_data["val"], min(5, len(splits_data["val"]))) if splits_data["val"] else []
    test_samples = random.sample(splits_data["test"], min(5, len(splits_data["test"]))) if splits_data["test"] else []
    
    # Compile Markdown content
    qa_pass_fail = lambda val: "PASS" if val == 0 else "FAIL"
    
    md_content = f"""# MANTRA Stage 2.1 HTR Manifest Quality Report

QA status: **{qa_status}**

## Executive Summary
This report summarizes the QA verification of the generated HTR training manifests derived from the finalized Stage 1 Synth100k metadata.

## Manifest Paths
- **Train Split:** `{os.path.join(args.output_dir, 'htr_train.csv')}`
- **Validation Split:** `{os.path.join(args.output_dir, 'htr_val.csv')}`
- **Test Split:** `{os.path.join(args.output_dir, 'htr_test.csv')}`
- **Unified Manifest:** `{os.path.join(args.output_dir, 'htr_all.csv')}`

## QA Pass/Fail Summary
| Metric / Check | Value | Target | Status |
| --- | --- | --- | --- |
| Missing image paths | {missing_image_paths} | 0 | {qa_pass_fail(missing_image_paths)} |
| Blank text rows | {blank_text_rows} | 0 | {qa_pass_fail(blank_text_rows)} |
| Duplicate image paths (global) | {duplicate_image_paths} | 0 | {qa_pass_fail(duplicate_image_paths)} |
| Duplicate image paths (within splits) | {dup_within_split} | 0 | {qa_pass_fail(dup_within_split)} |
| ASCII digit rows in text | {ascii_digit_rows} | 0 | {qa_pass_fail(ascii_digit_rows)} |
| Latin letter rows in text | {latin_letter_rows} | 0 | {qa_pass_fail(latin_letter_rows)} |
| Bad OCR symbol rows | {bad_ocr_symbol_rows} | - | Info |
| Duplicate line_id + variant_index pairs | {duplicate_line_id_variants} | - | Info |

## Row and Split Counts
- **Total Rows:** {total_input_rows:,}
- **Train Split Rows:** {len(splits_data["train"]):,} (Expected: 79,243)
- **Validation Split Rows:** {len(splits_data["val"]):,} (Expected: 12,649)
- **Test Split Rows:** {len(splits_data["test"]):,} (Expected: 8,108)

## Domain Distributions
| Domain | Count | Percentage |
| --- | --- | --- |
"""
    for dom, count in domain_counts.most_common():
        md_content += f"| {dom} | {count:,} | {count/total_input_rows*100:.2f}% |\n"
        
    md_content += """
## Domain × Split Distribution Table
| Domain | Train | Val | Test | Total |
| --- | --- | --- | --- | --- |
"""
    all_domains = sorted(list(domain_counts.keys()))
    for dom in all_domains:
        tr = domain_split_counts.get((dom, "train"), 0)
        vl = domain_split_counts.get((dom, "val"), 0)
        ts = domain_split_counts.get((dom, "test"), 0)
        tot = tr + vl + ts
        md_content += f"| {dom} | {tr:,} | {vl:,} | {ts:,} | {tot:,} |\n"

    md_content += """
## Variant Index Distribution
| Variant Index | Count | Percentage |
| --- | --- | --- |
"""
    for var, count in sorted(variant_counts.items()):
        md_content += f"| Variant {var} | {count:,} | {count/total_input_rows*100:.2f}% |\n"

    md_content += """
## Generation Round Distribution
| Generation Round | Count | Percentage |
| --- | --- | --- |
"""
    for gr, count in sorted(gen_round_counts.items()):
        md_content += f"| {gr} | {count:,} | {count/total_input_rows*100:.2f}% |\n"

    md_content += """
## Renderer Distribution
| Renderer | Count | Percentage |
| --- | --- | --- |
"""
    for rdr, count in sorted(renderer_counts.items()):
        md_content += f"| {rdr} | {count:,} | {count/total_input_rows*100:.2f}% |\n"

    md_content += """
## Degradation Distribution Summary (Top 10)
| Degradation Profile | Count | Percentage |
| --- | --- | --- |
"""
    for deg, count in degradation_counts.most_common(10):
        md_content += f"| {deg} | {count:,} | {count/total_input_rows*100:.2f}% |\n"
    if len(degradation_counts) > 10:
        md_content += f"| ...and {len(degradation_counts) - 10} more profiles... | | |\n"

    md_content += "\n## Sample Rows for Visual QA\n"
    
    def add_samples_section(title, samples):
        out = f"### {title}\n"
        out += "| Line ID | Font Name | Difficulty | Text Ground Truth | Image Path |\n"
        out += "| --- | --- | --- | --- | --- |\n"
        for s in samples:
            out += f"| {s['line_id']} | {s['font_name']} | {s['difficulty_bucket']} | `{s['text']}` | `{s['image_path']}` |\n"
        return out + "\n"

    md_content += add_samples_section("Train Samples", train_samples)
    md_content += add_samples_section("Validation Samples", val_samples)
    md_content += add_samples_section("Test Samples", test_samples)
    
    md_path = os.path.join(args.report_dir, "htr_manifest_report.md")
    md_content += "\n## Publication Gates\n\n| Gate | Failures |\n| --- | --- |\n"
    md_content += "".join(f"| {name} | {count} |\n" for name, count in failures.items())
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Saved QA report Markdown to: {md_path}")
    if not qa_passed:
        print("ERROR: QA failed. Manifests were not published; see the report.")
        return 1
    for name, rows in {"all": output_rows, **splits_data}.items():
        destination = Path(args.output_dir) / f"htr_{name}.csv"
        temporary = destination.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(destination)
        print(f"Saved {len(rows)} rows to: {destination}")
    print("=" * 70)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
