import os
import csv
import json
import random
import logging
import math
import multiprocessing
import numpy as np
from PIL import Image, ImageOps, ImageFilter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def check_single_row(row):
    """Processes a single row image validation. Executed in worker processes."""
    img_path = row["image_path"]
    line_id = row["line_id"]
    split = row["split"]
    source_id = row["source_id"]
    font_name = row["font_name"]
    degradation = row["degradation_profile"]
    renderer = row["renderer"]
    difficulty = row.get("difficulty_bucket", "easy")
    family = row.get("degradation_family", "single")
    readability_score = float(row.get("readability_score_estimate", 100.0))
    
    # Check ASCII digits
    rendered_text = row.get("source_text_rendered", row.get("text", ""))
    import re
    has_ascii_digit = bool(re.search(r'[0-9]', rendered_text))
    
    # 1. Existence
    if not os.path.exists(img_path):
        return {
            "line_id": line_id,
            "status": "failed",
            "reason": "file_not_found",
            "has_ascii_digit": has_ascii_digit
        }
        
    # 2. File size
    if os.path.getsize(img_path) == 0:
        return {
            "line_id": line_id,
            "status": "failed",
            "reason": "empty_file",
            "has_ascii_digit": has_ascii_digit
        }
        
    try:
        with Image.open(img_path) as img:
            w, h = img.size
            if w <= 0 or h <= 0:
                return {
                    "line_id": line_id,
                    "status": "failed",
                    "reason": "invalid_dimensions",
                    "has_ascii_digit": has_ascii_digit
                }
                
            # Pixel analysis
            gray = img.convert("L")
            arr = np.array(gray)
            avg_pixel = float(np.mean(arr))
            std_dev = float(np.std(arr))
            
            # Bounding box
            thresh_img = gray.point(lambda p: 255 if p < 160 else 0)
            bbox = thresh_img.getbbox()
            
            # Foreground ratio
            fg_mask = arr < 160
            fg_count = int(np.sum(fg_mask))
            foreground_ratio = fg_count / arr.size
            
            # Blank check
            is_blank = (not bbox) or (std_dev < 1.5) or (avg_pixel > 252.0) or (foreground_ratio < 0.0002)
            if is_blank:
                return {
                    "line_id": line_id,
                    "status": "failed",
                    "reason": "blank_image",
                    "has_ascii_digit": has_ascii_digit
                }
                
            # Contrast Check
            bg_mask = arr >= 160
            mean_bg = float(np.mean(arr[bg_mask])) if np.any(bg_mask) else 255.0
            mean_fg = float(np.mean(arr[fg_mask])) if np.any(fg_mask) else 0.0
            contrast = mean_bg - mean_fg
            
            is_low_contrast = (contrast < 45.0)
            is_unreadable_risk = (contrast < 35.0) or (std_dev < 10.0) or (h < 20)
            
            # Clipping
            filtered_thresh = thresh_img.filter(ImageFilter.MedianFilter(size=3))
            f_bbox = filtered_thresh.getbbox()
            clip_bbox = f_bbox if f_bbox else bbox
            is_clipped = (clip_bbox[0] <= 1 or clip_bbox[1] <= 1 or clip_bbox[2] >= w - 1 or clip_bbox[3] >= h - 1)
            
            if is_clipped:
                return {
                    "line_id": line_id,
                    "status": "failed",
                    "reason": "extreme_clipping",
                    "has_ascii_digit": has_ascii_digit
                }
                
            # Bounds checks
            if h < 15 or h > 250:
                return {
                    "line_id": line_id,
                    "status": "failed",
                    "reason": f"unusual_height_{h}",
                    "has_ascii_digit": has_ascii_digit
                }
            if w < 30 or w > 4000:
                return {
                    "line_id": line_id,
                    "status": "failed",
                    "reason": f"unusual_width_{w}",
                    "has_ascii_digit": has_ascii_digit
                }
                
            return {
                "line_id": line_id,
                "status": "passed",
                "width": w,
                "height": h,
                "avg_pixel": avg_pixel,
                "std_dev": std_dev,
                "contrast": contrast,
                "is_low_contrast": is_low_contrast,
                "is_unreadable_risk": is_unreadable_risk,
                "has_ascii_digit": has_ascii_digit,
                "readability_score": readability_score
            }
    except Exception as e:
        return {
            "line_id": line_id,
            "status": "failed",
            "reason": f"load_exception_{type(e).__name__}",
            "has_ascii_digit": has_ascii_digit
        }

def run_quality_checks(metadata_csv_path, output_json, output_md):
    if not os.path.exists(metadata_csv_path):
        logger.error(f"Metadata file does not exist: {metadata_csv_path}")
        return False
        
    records = []
    with open(metadata_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
            
    total_records = len(records)
    logger.info(f"Checking quality of {total_records} images in parallel...")
    
    passed_count = 0
    failed_count = 0
    failures = []
    
    # Global warning counters
    global_blank_count = 0
    global_clipping_warn_count = 0
    global_low_contrast_warn_count = 0
    global_unreadable_risk_count = 0
    ascii_digit_rendered_text_count = 0
    
    # Distributions
    splits = {}
    sources = {}
    fonts = {}
    degradations = {}
    renderers = {}
    difficulty_buckets = {}
    degradation_families = {}
    
    widths = []
    heights = []
    readability_scores = []
    
    # Stats per degradation profile
    profile_stats = {}
    # Stats per difficulty bucket
    difficulty_stats = {
        "easy": {"count": 0, "widths": [], "heights": [], "blank_count": 0, "clipping_warn_count": 0, "low_contrast_warn_count": 0, "unreadable_risk_count": 0, "readability_scores": []},
        "medium": {"count": 0, "widths": [], "heights": [], "blank_count": 0, "clipping_warn_count": 0, "low_contrast_warn_count": 0, "unreadable_risk_count": 0, "readability_scores": []},
        "hard": {"count": 0, "widths": [], "heights": [], "blank_count": 0, "clipping_warn_count": 0, "low_contrast_warn_count": 0, "unreadable_risk_count": 0, "readability_scores": []}
    }
    
    # Use ProcessPool to check in parallel
    num_workers = min(8, os.cpu_count() or 4)
    logger.info(f"Using {num_workers} processes for visual checks...")
    
    # Spawn context pool
    ctx = multiprocessing.get_context("spawn")
    results = []
    with ctx.Pool(processes=num_workers) as pool:
        results = pool.map(check_single_row, records)
        
    # Aggregate results
    for idx, row in enumerate(records):
        res = results[idx]
        split = row["split"]
        source_id = row["source_id"]
        font_name = row["font_name"]
        degradation = row["degradation_profile"]
        renderer = row["renderer"]
        difficulty = row.get("difficulty_bucket", "easy")
        family = row.get("degradation_family", "single")
        
        # Track distributions
        splits[split] = splits.get(split, 0) + 1
        sources[source_id] = sources.get(source_id, 0) + 1
        fonts[font_name] = fonts.get(font_name, 0) + 1
        degradations[degradation] = degradations.get(degradation, 0) + 1
        renderers[renderer] = renderers.get(renderer, 0) + 1
        difficulty_buckets[difficulty] = difficulty_buckets.get(difficulty, 0) + 1
        degradation_families[family] = degradation_families.get(family, 0) + 1
        
        # Initialize profile stats
        if degradation not in profile_stats:
            profile_stats[degradation] = {
                "count": 0, "widths": [], "heights": [],
                "blank_count": 0, "clipping_warn_count": 0,
                "low_contrast_warn_count": 0, "unreadable_risk_count": 0,
                "readability_scores": []
            }
            
        profile_stats[degradation]["count"] += 1
        difficulty_stats[difficulty]["count"] += 1
        
        if res["has_ascii_digit"]:
            ascii_digit_rendered_text_count += 1
            
        if res["status"] == "failed":
            failed_count += 1
            failures.append({
                "line_id": row["line_id"],
                "image_path": row["image_path"],
                "reason": res["reason"]
            })
            if res["reason"] == "blank_image":
                global_blank_count += 1
                profile_stats[degradation]["blank_count"] += 1
                difficulty_stats[difficulty]["blank_count"] += 1
            elif res["reason"] == "extreme_clipping":
                global_clipping_warn_count += 1
                profile_stats[degradation]["clipping_warn_count"] += 1
                difficulty_stats[difficulty]["clipping_warn_count"] += 1
        else:
            passed_count += 1
            widths.append(res["width"])
            heights.append(res["height"])
            readability_scores.append(res["readability_score"])
            
            profile_stats[degradation]["widths"].append(res["width"])
            profile_stats[degradation]["heights"].append(res["height"])
            profile_stats[degradation]["readability_scores"].append(res["readability_score"])
            
            difficulty_stats[difficulty]["widths"].append(res["width"])
            difficulty_stats[difficulty]["heights"].append(res["height"])
            difficulty_stats[difficulty]["readability_scores"].append(res["readability_score"])
            
            if res["is_low_contrast"]:
                global_low_contrast_warn_count += 1
                profile_stats[degradation]["low_contrast_warn_count"] += 1
                difficulty_stats[difficulty]["low_contrast_warn_count"] += 1
                
            if res["is_unreadable_risk"]:
                global_unreadable_risk_count += 1
                profile_stats[degradation]["unreadable_risk_count"] += 1
                difficulty_stats[difficulty]["unreadable_risk_count"] += 1
                
    # Global averages
    avg_w = sum(widths) / len(widths) if widths else 0
    avg_h = sum(heights) / len(heights) if heights else 0
    avg_readability = sum(readability_scores) / len(readability_scores) if readability_scores else 100.0
    min_w = min(widths) if widths else 0
    max_w = max(widths) if widths else 0
    min_h = min(heights) if heights else 0
    max_h = max(heights) if heights else 0
    
    profile_summary = {}
    for deg, stats in profile_stats.items():
        w_list = stats["widths"]
        h_list = stats["heights"]
        r_list = stats["readability_scores"]
        profile_summary[deg] = {
            "count": stats["count"],
            "avg_width": sum(w_list) / len(w_list) if w_list else 0.0,
            "avg_height": sum(h_list) / len(h_list) if h_list else 0.0,
            "avg_readability": sum(r_list) / len(r_list) if r_list else 100.0,
            "blank_count": stats["blank_count"],
            "clipping_warn_count": stats["clipping_warn_count"],
            "low_contrast_warn_count": stats["low_contrast_warn_count"],
            "unreadable_risk_count": stats["unreadable_risk_count"]
        }
        
    difficulty_summary = {}
    for diff, stats in difficulty_stats.items():
        w_list = stats["widths"]
        h_list = stats["heights"]
        r_list = stats["readability_scores"]
        difficulty_summary[diff] = {
            "count": stats["count"],
            "avg_width": sum(w_list) / len(w_list) if w_list else 0.0,
            "avg_height": sum(h_list) / len(h_list) if h_list else 0.0,
            "avg_readability": sum(r_list) / len(r_list) if r_list else 100.0,
            "blank_count": stats["blank_count"],
            "clipping_warn_count": stats["clipping_warn_count"],
            "low_contrast_warn_count": stats["low_contrast_warn_count"],
            "unreadable_risk_count": stats["unreadable_risk_count"]
        }
        
    random.seed(42)
    split_samples = {"train": [], "val": [], "test": []}
    for row in records:
        split = row["split"]
        if split in split_samples and len(split_samples[split]) < 5:
            if not any(f["line_id"] == row["line_id"] for f in failures):
                split_samples[split].append({
                    "line_id": row["line_id"],
                    "image_path": row["image_path"],
                    "text": row["text"],
                    "font": row["font_name"],
                    "degradation": row["degradation_profile"]
                })
                
    quality_report = {
        "summary": {
            "total_records": total_records,
            "passed": passed_count,
            "failed": failed_count,
            "success_rate": passed_count / total_records if total_records > 0 else 0,
            "global_blank_count": global_blank_count,
            "global_clipping_warn_count": global_clipping_warn_count,
            "global_low_contrast_warn_count": global_low_contrast_warn_count,
            "global_unreadable_risk_count": global_unreadable_risk_count,
            "ascii_digit_rendered_text_count": ascii_digit_rendered_text_count,
            "average_readability_score": avg_readability
        },
        "dimensions": {
            "average_width": avg_w, "average_height": avg_h,
            "min_width": min_w, "max_width": max_w,
            "min_height": min_h, "max_height": max_h
        },
        "distributions": {
            "split": splits, "source": sources, "font": fonts,
            "degradation": degradations, "renderer": renderers,
            "difficulty_bucket": difficulty_buckets,
            "degradation_family": degradation_families
        },
        "profile_summary": profile_summary,
        "difficulty_summary": difficulty_summary,
        "failures": failures,
        "qa_samples": split_samples
    }
    
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(quality_report, f, indent=4)
    logger.info(f"Quality report JSON written to {output_json}")
    
    md_content = f"""# MANTRA Synthetic Image Generation Quality Report

## Executive Summary

- **Total Images Attempted**: {total_records}
- **Successfully Generated & Passed QA**: {passed_count} ({passed_count/total_records*100:.2f}%)
- **Failed Quality Checks**: {failed_count} ({failed_count/total_records*100:.2f}%)

### Global Quality Metrics
- **Blank Images**: {global_blank_count}
- **Clipped Images (Failures)**: {global_clipping_warn_count}
- **Low Contrast Warnings**: {global_low_contrast_warn_count}
- **Unreadable Risk Warnings**: {global_unreadable_risk_count}
- **ASCII Digits in Rendered Text**: {ascii_digit_rendered_text_count}
- **Average Readability Score**: {avg_readability:.2f} / 100.0

"""
    if failed_count > 0:
        md_content += "### Warning: Quality Check Failures Detected\n\n"
        md_content += "| Line ID | Image Path | Failure Reason |\n"
        md_content += "| --- | --- | --- |\n"
        for f in failures[:20]:
            md_content += f"| {f['line_id']} | `{f['image_path']}` | {f['reason']} |\n"
        if len(failures) > 20:
            md_content += f"\n*...and {len(failures) - 20} more failures. See JSON report for details.*\n"
    else:
        md_content += "> [!IMPORTANT]\n"
        md_content += "> **All generated images successfully passed quality control checks!** No clipping, blank pages, or unreadable dimensions detected.\n\n"
        
    md_content += f"""## Dimensions Summary

- **Average Bounding Box**: {avg_w:.1f} x {avg_h:.1f}
- **Width Range**: {min_w}px to {max_w}px
- **Height Range**: {min_h}px to {max_h}px

## Difficulty Buckets Quality Summary

| Difficulty | Count | Ratio | Avg Width | Avg Height | Avg Readability | Blanks | Clipped | Low Contrast | Unreadable Risk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
"""
    for diff in ["easy", "medium", "hard"]:
        s = difficulty_summary.get(diff, {"count": 0, "avg_width": 0.0, "avg_height": 0.0, "avg_readability": 0.0, "blank_count": 0, "clipping_warn_count": 0, "low_contrast_warn_count": 0, "unreadable_risk_count": 0})
        ratio = (s['count'] / total_records * 100) if total_records > 0 else 0.0
        md_content += f"| {diff} | {s['count']} | {ratio:.1f}% | {s['avg_width']:.1f}px | {s['avg_height']:.1f}px | {s['avg_readability']:.2f} | {s['blank_count']} | {s['clipping_warn_count']} | {s['low_contrast_warn_count']} | {s['unreadable_risk_count']} |\n"

    md_content += """
## Degradation Profile Quality Summary

| Profile | Count | Avg Width | Avg Height | Avg Readability | Blanks | Clipped | Low Contrast | Unreadable Risk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
"""
    for deg in sorted(profile_summary.keys()):
        s = profile_summary[deg]
        md_content += f"| {deg} | {s['count']} | {s['avg_width']:.1f}px | {s['avg_height']:.1f}px | {s['avg_readability']:.2f} | {s['blank_count']} | {s['clipping_warn_count']} | {s['low_contrast_warn_count']} | {s['unreadable_risk_count']} |\n"

    md_content += "\n## Dataset Distributions\n\n"
    md_content += "### Difficulty Bucket Distribution\n"
    for k, v in difficulty_buckets.items():
        md_content += f"- **{k}**: {v} images ({v/total_records*100:.1f}%)\n"
    md_content += "\n### Degradation Family Distribution\n"
    for k, v in degradation_families.items():
        md_content += f"- **{k}**: {v} images ({v/total_records*100:.1f}%)\n"
    md_content += "\n### Split Distribution\n"
    for k, v in splits.items():
        md_content += f"- **{k}**: {v} images ({v/total_records*100:.1f}%)\n"
    md_content += "\n### Font Distribution\n"
    for k, v in fonts.items():
        md_content += f"- **{k}**: {v} images ({v/total_records*100:.1f}%)\n"
    md_content += "\n### Degradation Distribution\n"
    for k, v in degradations.items():
        md_content += f"- **{k}**: {v} images ({v/total_records*100:.1f}%)\n"
    md_content += "\n### Renderer Distribution\n"
    for k, v in renderers.items():
        md_content += f"- **{k}**: {v} images ({v/total_records*100:.1f}%)\n"

    md_content += """
## Recommended QA Samples for Manual Inspection

The following random samples have been chosen from each split for visual review:

### Train Samples
"""
    for s in split_samples["train"]:
        md_content += f"- **Line ID**: {s['line_id']} | **Font**: {s['font']} | **Degradation**: {s['degradation']}\n"
        md_content += f"  - Path: [`{s['image_path']}`](file:///{os.path.abspath(s['image_path']).replace('\\', '/')})\n"
        md_content += f"  - Text: `{s['text']}`\n"
        
    md_content += "\n### Validation Samples\n"
    for s in split_samples["val"]:
        md_content += f"- **Line ID**: {s['line_id']} | **Font**: {s['font']} | **Degradation**: {s['degradation']}\n"
        md_content += f"  - Path: [`{s['image_path']}`](file:///{os.path.abspath(s['image_path']).replace('\\', '/')})\n"
        md_content += f"  - Text: `{s['text']}`\n"
        
    md_content += "\n### Test Samples\n"
    for s in split_samples["test"]:
        md_content += f"- **Line ID**: {s['line_id']} | **Font**: {s['font']} | **Degradation**: {s['degradation']}\n"
        md_content += f"  - Path: [`{s['image_path']}`](file:///{os.path.abspath(s['image_path']).replace('\\', '/')})\n"
        md_content += f"  - Text: `{s['text']}`\n"
        
    with open(output_md, "w", encoding="utf-8") as f:
        f.write(md_content)
    logger.info(f"Quality report summary written to {output_md}")
    return True

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MANTRA Quality Check for Synthetic Images")
    parser.add_argument("--metadata", type=str, default="data/stage1_synthetic/metadata/synthetic_image_metadata.csv", help="Path to metadata CSV")
    parser.add_argument("--output-json", type=str, default="data/stage1_synthetic/reports/render_quality_report.json", help="Path to output JSON report")
    parser.add_argument("--output-md", type=str, default="data/stage1_synthetic/reports/render_quality_summary.md", help="Path to output markdown report")
    args = parser.parse_args()
    
    run_quality_checks(
        metadata_csv_path=args.metadata,
        output_json=args.output_json,
        output_md=args.output_md
    )
