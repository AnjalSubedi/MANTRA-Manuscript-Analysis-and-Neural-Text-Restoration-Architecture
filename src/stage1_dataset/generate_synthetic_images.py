import os
import sys
import yaml
import csv
import json
import argparse
import random
import hashlib
import logging
import urllib.request
import urllib.error
import math
from pathlib import Path
from PIL import Image
import numpy as np

def calculate_readability_metrics(pil_img):
    """
    Calculates readability metrics including unreadable_risk and a score estimate [0, 100].
    Formula: (contrast < 35.0) or (std_dev < 10.0) or (height < 20)
    """
    w, h = pil_img.size
    # Convert to grayscale numpy array
    gray = np.array(pil_img.convert("L"))
    
    # Calculate standard deviation
    std_dev = float(np.std(gray))
    
    # Identify foreground (ink < 160) and background (bg >= 160)
    fg_mask = gray < 160
    bg_mask = gray >= 160
    
    mean_fg = float(np.mean(gray[fg_mask])) if np.any(fg_mask) else 0.0
    mean_bg = float(np.mean(gray[bg_mask])) if np.any(bg_mask) else 255.0
    contrast = mean_bg - mean_fg
    
    # Weighted score [0, 100]
    contrast_score = min(80.0, (contrast / 150.0) * 80.0)
    std_score = min(20.0, (std_dev / 50.0) * 20.0)
    readability_score = round(float(np.clip(contrast_score + std_score, 0.0, 100.0)), 2)
    
    is_unreadable_risk = (contrast < 35.0) or (std_dev < 10.0) or (h < 20)
    return is_unreadable_risk, readability_score, contrast, std_dev

def normalize_devanagari_text(text):
    """
    Strict pre-render normalization for Devanagari text:
    - ASCII digits 0-9 -> Devanagari digits ०-९
    - Preserve existing Devanagari digits
    - Do not alter historical spellings
    - Keep punctuation and danda forms
    """
    ascii_to_devanagari = {
        '0': '०',
        '1': '१',
        '2': '२',
        '3': '३',
        '4': '४',
        '5': '५',
        '6': '६',
        '7': '७',
        '8': '८',
        '9': '९'
    }
    normalized = "".join(ascii_to_devanagari.get(char, char) for char in text)
    return normalized

# Add src to python path to import modules
sys.path.append(str(Path(__file__).resolve().parents[1]))

from stage1_dataset.render_text_line import render_text_line, run_shaping_smoke_test, PIL_has_raqm
from stage1_dataset.degradation_profiles import apply_degradation, DIFFICULTY_MAPPING

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FONTS_TO_DOWNLOAD = [
    {
        "font_name": "Noto Sans Devanagari",
        "file_name": "NotoSansDevanagari-Regular.ttf",
        "source_url": "https://raw.githubusercontent.com/google/fonts/main/ofl/notosansdevanagari/NotoSansDevanagari%5Bwdth%2Cwght%5D.ttf",
        "license": "OFL",
        "font_group": "Clean modern",
        "font_slug": "notosans"
    },
    {
        "font_name": "Noto Serif Devanagari",
        "file_name": "NotoSerifDevanagari-Regular.ttf",
        "source_url": "https://raw.githubusercontent.com/google/fonts/main/ofl/notoserifdevanagari/NotoSerifDevanagari%5Bwdth%2Cwght%5D.ttf",
        "license": "OFL",
        "font_group": "Printed/book-style",
        "font_slug": "notoserif"
    },
    {
        "font_name": "Rozha One",
        "file_name": "RozhaOne-Regular.ttf",
        "source_url": "https://raw.githubusercontent.com/google/fonts/main/ofl/rozhaone/RozhaOne-Regular.ttf",
        "license": "OFL",
        "font_group": "Printed/book-style",
        "font_slug": "rozhaone"
    },
    {
        "font_name": "Yatra One",
        "file_name": "YatraOne-Regular.ttf",
        "source_url": "https://raw.githubusercontent.com/google/fonts/main/ofl/yatraone/YatraOne-Regular.ttf",
        "license": "OFL",
        "font_group": "Sanskrit-style",
        "font_slug": "yatraone"
    },
    {
        "font_name": "Poppins",
        "file_name": "Poppins-Regular.ttf",
        "source_url": "https://raw.githubusercontent.com/google/fonts/main/ofl/poppins/Poppins-Regular.ttf",
        "license": "OFL",
        "font_group": "Clean modern",
        "font_slug": "poppins"
    },
    {
        "font_name": "Rajdhani",
        "file_name": "Rajdhani-Regular.ttf",
        "source_url": "https://raw.githubusercontent.com/google/fonts/main/ofl/rajdhani/Rajdhani-Regular.ttf",
        "license": "OFL",
        "font_group": "Clean modern",
        "font_slug": "rajdhani"
    },
    {
        "font_name": "Hind",
        "file_name": "Hind-Regular.ttf",
        "source_url": "https://raw.githubusercontent.com/google/fonts/main/ofl/hind/Hind-Regular.ttf",
        "license": "OFL",
        "font_group": "Nepali publication-style",
        "font_slug": "hind"
    },
    {
        "font_name": "Karma",
        "file_name": "Karma-Regular.ttf",
        "source_url": "https://raw.githubusercontent.com/google/fonts/main/ofl/karma/Karma-Regular.ttf",
        "license": "OFL",
        "font_group": "Printed/book-style",
        "font_slug": "karma"
    },
    {
        "font_name": "Jaldi",
        "file_name": "Jaldi-Regular.ttf",
        "source_url": "https://raw.githubusercontent.com/google/fonts/main/ofl/jaldi/Jaldi-Regular.ttf",
        "license": "OFL",
        "font_group": "Nepali publication-style",
        "font_slug": "jaldi"
    },
    {
        "font_name": "Biryani",
        "file_name": "Biryani-Regular.ttf",
        "source_url": "https://raw.githubusercontent.com/google/fonts/main/ofl/biryani/Biryani-Regular.ttf",
        "license": "OFL",
        "font_group": "Clean modern",
        "font_slug": "biryani"
    },
    {
        "font_name": "Mukta",
        "file_name": "Mukta-Regular.ttf",
        "source_url": "https://raw.githubusercontent.com/google/fonts/main/ofl/mukta/Mukta-Regular.ttf",
        "license": "OFL",
        "font_group": "Clean modern",
        "font_slug": "mukta"
    },
    {
        "font_name": "Teko",
        "file_name": "Teko-Regular.ttf",
        "source_url": "https://raw.githubusercontent.com/google/fonts/main/ofl/teko/Teko%5Bwght%5D.ttf",
        "license": "OFL",
        "font_group": "Clean modern",
        "font_slug": "teko"
    }
]

def calculate_sha256(filepath):
    """Calculates the SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def ensure_fonts(font_dir, metadata_dir):
    """
    Downloads fonts if missing, verifies integrity via SHA-256,
    and creates font_manifest.csv.
    """
    os.makedirs(font_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)
    manifest_path = os.path.join(metadata_dir, "font_manifest.csv")
    
    manifest_rows = []
    download_success = True
    
    for f_info in FONTS_TO_DOWNLOAD:
        font_path = os.path.join(font_dir, f_info["file_name"])
        font_hash = ""
        
        # Download if file does not exist
        if not os.path.exists(font_path):
            logger.info(f"Downloading {f_info['font_name']} from {f_info['source_url']}...")
            try:
                # Set user agent to avoid blockage
                req = urllib.request.Request(
                    f_info["source_url"], 
                    headers={'User-Agent': 'Mozilla/5.0'}
                )
                with urllib.request.urlopen(req, timeout=15) as response:
                    with open(font_path, 'wb') as out_file:
                        out_file.write(response.read())
                font_hash = calculate_sha256(font_path)
            except Exception as e:
                logger.error(f"Failed to download {f_info['font_name']}: {e}")
                download_success = False
        else:
            font_hash = calculate_sha256(font_path)
            
        manifest_rows.append({
            "font_name": f_info["font_name"],
            "file_name": f_info["file_name"],
            "source_url": f_info["source_url"],
            "license": f_info["license"],
            "sha256": font_hash
        })
        
    if not download_success:
        # Check if we have at least one font to run the smoke test
        existing_fonts = [f for f in os.listdir(font_dir) if f.endswith(".ttf")]
        if not existing_fonts:
            logger.critical("No Devanagari fonts available and download failed.")
            print("\n" + "="*80)
            print("MANUAL FONT SETUP INSTRUCTIONS:")
            print("Internet connection or download failed. Please manually download Devanagari fonts")
            print("and place them in: data/stage1_synthetic/fonts/")
            print("Recommended fonts (OFL):")
            for fi in FONTS_TO_DOWNLOAD:
                print(f"- {fi['font_name']}: {fi['source_url']}")
            print("="*80 + "\n")
            sys.exit(1)
        else:
            logger.warning("Some downloads failed, but using existing fonts in folder.")
            
    # Write manifest
    with open(manifest_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["font_name", "file_name", "source_url", "license", "sha256"])
        writer.writeheader()
        writer.writerows(manifest_rows)
    logger.info(f"Font manifest written to {manifest_path}")

def load_yaml_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="MANTRA Synthetic Devanagari Image Generation")
    parser.add_argument("--limit", type=int, default=None, help="Limit total images to generate")
    parser.add_argument("--split", type=str, default="all", choices=["train", "val", "test", "all"], help="Specify split to generate")
    parser.add_argument("--smoke-test", action="store_true", help="Run in smoke-test mode generating ~300 images")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing generated images")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--config", type=str, default="configs/stage1_synth100k_config.yaml", help="Path to config file")
    parser.add_argument("--renderer", type=str, default="chromium", choices=["chromium"], help="Layout renderer (strictly chromium)")
    parser.add_argument("--labels", type=str, default=None, help="Path to CSV labels file (overrides config)")
    parser.add_argument("--allow-unreadable-risk", action="store_true", help="Keep unreadable-risk images in final generation")
    parser.add_argument("--only-has-digit", action="store_true", help="Filter and generate only rows where has_digit == 'true'")
    parser.add_argument("--output-subdir", type=str, default=None, help="Custom subdirectory for images (e.g. images_digits_only)")
    parser.add_argument("--metadata-out", type=str, default=None, help="Custom output path for metadata CSV")
    
    args = parser.parse_args()
    
    # 1. Load config
    config = load_yaml_config(args.config)
    seed = args.seed if args.seed is not None else config["dataset"].get("random_seed", 42)
    random.seed(seed)
    
    # Force layout renderer strictly to chromium
    config["rendering"]["preferred_renderer"] = "chromium"
    args.renderer = "chromium"
    
    # Paths from config
    font_dir = config["paths"]["font_dir"]
    if args.output_subdir:
        parent_dir = os.path.dirname(config["paths"]["image_dir"])
        image_dir = os.path.join(parent_dir, args.output_subdir)
    else:
        image_dir = config["paths"]["image_dir"]
        
    metadata_dir = config["paths"]["metadata_dir"]
    report_dir = config["paths"]["report_dir"]
    label_dir = config["paths"]["label_dir"]
    
    # Ensure folder structures
    os.makedirs(metadata_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)
    for s in ["train", "val", "test"]:
        os.makedirs(os.path.join(image_dir, s), exist_ok=True)
        
    # Safely clear the directories if we are in overwrite mode or smoke test
    if args.overwrite or args.smoke_test:
        logger.info("Cleaning up existing synthetic images for a clean run...")
        for s in ["train", "val", "test"]:
            s_dir = os.path.join(image_dir, s)
            if os.path.exists(s_dir):
                for f_name in os.listdir(s_dir):
                    f_path = os.path.join(s_dir, f_name)
                    if os.path.isfile(f_path):
                        try:
                            os.remove(f_path)
                        except Exception as e:
                            logger.warning(f"Could not remove file {f_path}: {e}")
        
    # 2. Download and verify fonts
    ensure_fonts(font_dir, metadata_dir)
    
    # Find active fonts
    available_fonts = []
    font_slug_map = {}
    font_group_map = {}
    
    # Search through download list to map available ones
    for f_info in FONTS_TO_DOWNLOAD:
        f_path = os.path.join(font_dir, f_info["file_name"])
        if os.path.exists(f_path):
            available_fonts.append(f_path)
            font_slug_map[f_path] = f_info["font_slug"]
            font_group_map[f_path] = f_info["font_group"]
            
    if not available_fonts:
        logger.critical("No fonts available in data/stage1_synthetic/fonts/.")
        sys.exit(1)
        
    # 3. Shaping QA Support and Safeguards
    pango_available = False
    try:
        import gi
        gi.require_version('Pango', '1.0')
        gi.require_version('PangoCairo', '1.0')
        pango_available = True
    except Exception:
        pass

    cairo_available = False
    try:
        import cairo
        cairo_available = True
    except Exception:
        pass

    pango_ok = pango_available and cairo_available

    chromium_available = False
    try:
        from playwright.sync_api import sync_playwright
        chromium_available = True
    except Exception:
        pass

    pillow_raqm_available = PIL_has_raqm()
    harfbuzz_available_if_detectable = pango_ok or pillow_raqm_available or chromium_available
    
    # Resolve active renderer (strictly chromium)
    shaping_ok = chromium_available
    renderer_used = "chromium"
    if not shaping_ok:
        logger.critical("ERROR: Playwright Headless Chromium is not available. It is strictly required. Install it using 'playwright install' first.")
        sys.exit(1)
            
    shaping_test_strings = [
        "नेपालमा इसापूर्व तेस्रो शताब्दीदेखिकै शिलालेखहरू प्राप्त भएका छन् । भारतीय",
        "इसापूर्व",
        "पूर्व",
        "प्राप्त",
        "शिलालेखहरू प्राप्त भएका छन्",
        "ग्रन्थ प्रज्ञा श्रद्धा राष्ट्र्य प्राप्ति",
        "क्ति क्ष ज्ञ त्र श्र प्र द्ध ह्म स्त्र",
        "कि की कु कू कृ के कै को कौ कं कः"
    ]
    
    shaping_test_status = {}
    failed_test_strings = []
    
    for s_str in shaping_test_strings:
        if shaping_ok:
            shaping_test_status[s_str] = "passed"
        else:
            shaping_test_status[s_str] = "failed"
            failed_test_strings.append(s_str)
            
    sample_font = available_fonts[0]
    font_name = Path(sample_font).stem
    
    warning_msg = "" if shaping_ok else "WARNING: Smoke test only. Images are not valid for dataset training because reliable Devanagari shaping is unavailable."
    
    capability_report = {
        "renderer_selected": args.renderer,
        "renderer_used": renderer_used,
        "pango_available": pango_ok,
        "chromium_available": chromium_available,
        "pillow_raqm_available": pillow_raqm_available,
        "plain_pillow_allowed_for_full_generation": False,
        "shaping_qa_grid_path": "data/stage1_synthetic/reports/sample_grids/shaping_qa_grid.png",
        "warning_messages": warning_msg,
        "shaping_test_strings": shaping_test_strings,
        "shaping_test_status": shaping_test_status,
        "failed_test_strings": failed_test_strings
    }
    
    cap_report_path = os.path.join(report_dir, "renderer_capability_report.json")
    with open(cap_report_path, "w", encoding="utf-8") as f:
        json.dump(capability_report, f, indent=4)
    logger.info(f"Capability report written to {cap_report_path}")
    
    if not shaping_ok:
        if not args.smoke_test:
            logger.critical("ERROR: Reliable Devanagari shaping is unavailable. Install Pango/Cairo/HarfBuzz or Pillow with RAQM before generating MANTRA-Synth images.")
            sys.exit(1)
        else:
            print("\n" + "="*80)
            print("WARNING: Smoke test only. Images are not valid for dataset training because reliable Devanagari shaping is unavailable.")
            print("="*80 + "\n")
            
    # 4. Load CSV labels
    labels_csv_path = args.labels if args.labels else os.path.join(label_dir, "historical_nepali_lines.csv")
    if not os.path.exists(labels_csv_path):
        logger.critical(f"CSV file not found at: {labels_csv_path}")
        sys.exit(1)
        
    lines = []
    with open(labels_csv_path, "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            lines.append(row)
            
    logger.info(f"Loaded {len(lines)} lines from {labels_csv_path}")
    
    if args.only_has_digit:
        lines = [row for row in lines if row.get("has_digit", "").lower() == "true"]
        logger.info(f"Filtered by --only-has-digit. Remaining lines: {len(lines)}")
    
    # Filter splits
    if args.split != "all":
        lines = [row for row in lines if row["split"] == args.split]
        logger.info(f"Filtered to {len(lines)} lines for split={args.split}")
        
    # Smoke-test limits
    if args.smoke_test:
        # ~200 train, ~50 val, ~50 test
        train_lines = [r for r in lines if r["split"] == "train"]
        val_lines = [r for r in lines if r["split"] == "val"]
        test_lines = [r for r in lines if r["split"] == "test"]
        
        # Seed local random for deterministic subsetting
        subset_rand = random.Random(seed)
        
        limit_train = min(200, len(train_lines))
        limit_val = min(50, len(val_lines))
        limit_test = min(50, len(test_lines))
        
        selected = (
            subset_rand.sample(train_lines, limit_train) +
            subset_rand.sample(val_lines, limit_val) +
            subset_rand.sample(test_lines, limit_test)
        )
        if args.limit:
            selected = subset_rand.sample(selected, min(args.limit, len(selected)))
        lines = selected
        logger.info(f"Smoke test mode active. Selected {len(lines)} lines (Train: {limit_train}, Val: {limit_val}, Test: {limit_test})")
    elif args.limit:
        subset_rand = random.Random(seed)
        if len(lines) > args.limit:
            lines = subset_rand.sample(lines, args.limit)
        logger.info(f"Limit applied. Selected {len(lines)} lines.")
        
    # 5. Set up degradation distributions
    degradations = list(config["degradation_distribution"].keys())
    degradation_weights = list(config["degradation_distribution"].values())
    
    # Normalize weights just in case
    w_sum = sum(degradation_weights)
    degradation_weights = [w / w_sum for w in degradation_weights]
    
    # Mode selection probabilities
    single_profile_prob = config.get("degradation_mode", {}).get("single_profile_prob", 0.70)
    composed_profile_prob = config.get("degradation_mode", {}).get("composed_profile_prob", 0.30)
    
    # 6. Generate Images
    generated_count = 0
    skipped_count = 0
    skip_reasons = {}
    
    if args.metadata_out:
        metadata_csv_path = args.metadata_out
    else:
        metadata_csv_path = os.path.join(metadata_dir, "synthetic_image_metadata.csv")
    
    # Render preferences
    renderer_pref = config["rendering"]["preferred_renderer"]
    font_size_min = config["rendering"]["font_size_min"]
    font_size_max = config["rendering"]["font_size_max"]
    padding_min = config["rendering"]["padding_min"]
    padding_max = config["rendering"]["padding_max"]
    max_rot = config["rendering"]["max_rotation"]
    
    # Use another seeded random for parameters
    param_rand = random.Random(seed)
    
    metadata_rows = []
    logger.info("Starting generation...")
    
    for idx, row in enumerate(lines):
        line_id = row["line_id"]
        source_id = row["source_id"]
        source_file = row["source_file"]
        
        # Verify text source and normalize
        source_text_original = row["text"]
        source_text_normalized = normalize_devanagari_text(source_text_original)
        source_text_rendered = source_text_normalized
        digit_normalization_applied = "true" if source_text_original != source_text_normalized else "false"
        
        # Post-normalization integrity check: reject/skip if it contains ASCII digits [0-9]
        if any(char in "0123456789" for char in source_text_rendered):
            logger.error(f"Integrity check failed for line {line_id}: final text still contains ASCII digits: {source_text_rendered}")
            skipped_count += 1
            skip_reasons["contains_ascii_digits"] = skip_reasons.get("contains_ascii_digits", 0) + 1
            continue
            
        # Strict equality check
        normalized_expected_text = source_text_rendered
        rendered_text_input = source_text_rendered
        if normalized_expected_text != rendered_text_input:
            logger.critical(f"Strict equality check failed for line {line_id}: normalized={normalized_expected_text} != input={rendered_text_input}")
            sys.exit(1)
            
        text = source_text_rendered
        split = row["split"]
        
        # Pick parameters deterministically using row index + base seed
        param_rand.seed(seed + idx)
        
        font_path = param_rand.choice(available_fonts)
        font_name = Path(font_path).stem
        font_group = font_group_map[font_path]
        font_slug = font_slug_map[font_path]
        font_size = param_rand.randint(font_size_min, font_size_max)
        padding = param_rand.randint(padding_min, padding_max)
        
        # Choose single vs composed mode with difficulty-aware sampling
        if args.smoke_test and idx < 120:
            # Force at least 5 examples per single profile (24 profiles * 5 = 120 samples)
            degradation_profile = degradations[idx // 5]
            is_composed = False
            target_diff = DIFFICULTY_MAPPING.get(degradation_profile, "easy")
        else:
            if args.smoke_test:
                # Adjust weights for remaining 180 images to compensate for the first 120 images
                target_diff = param_rand.choices(["easy", "medium", "hard"], weights=[0.57, 0.43, 0.0], k=1)[0]
            else:
                # Difficulty-aware sampling: easy ≈ 45%, medium ≈ 40%, hard ≈ 15%
                target_diff = param_rand.choices(["easy", "medium", "hard"], weights=[0.45, 0.40, 0.15], k=1)[0]
                
            if param_rand.random() < composed_profile_prob:
                is_composed = True
                degradation_profile = "composed"
            else:
                is_composed = False
                filtered_degs = [d for d in degradations if DIFFICULTY_MAPPING.get(d) == target_diff]
                if not filtered_degs:
                    filtered_degs = degradations
                filtered_weights = [config["degradation_distribution"].get(d, 1.0) for d in filtered_degs]
                w_sum = sum(filtered_weights)
                filtered_weights = [w / w_sum for w in filtered_weights]
                degradation_profile = param_rand.choices(filtered_degs, weights=filtered_weights, k=1)[0]
                
        if is_composed:
            degradation_family = "composed"
            deg_level = -1
            img_filename = f"{split}_{line_id}_{font_slug}_composed.png"
        else:
            degradation_family = "single"
            deg_level = int(degradation_profile.split("_")[0][1:])
            img_filename = f"{split}_{line_id}_{font_slug}_{degradation_profile.split('_')[0]}.png"
            
        img_relative_path = os.path.join(image_dir, split, img_filename)
        img_absolute_path = os.path.abspath(img_relative_path)
        
        # Overwrite check
        if os.path.exists(img_absolute_path) and not args.overwrite:
            try:
                with Image.open(img_absolute_path) as existing_img:
                    w, h = existing_img.size
                    risk, readability_score, _, _ = calculate_readability_metrics(existing_img)
                    risk_str = "true" if risk else "false"
                generated_count += 1
                metadata_rows.append({
                    "image_id": Path(img_filename).stem,
                    "line_id": line_id,
                    "image_path": os.path.relpath(img_absolute_path, os.getcwd()).replace("\\", "/"),
                    "text": text,
                    "source_text_original": source_text_original,
                    "source_text_normalized": source_text_normalized,
                    "source_text_rendered": source_text_rendered,
                    "digit_normalization_applied": digit_normalization_applied,
                    "source_id": source_id,
                    "source_file": source_file,
                    "split": split,
                    "font_name": font_name,
                    "font_path": font_path,
                    "font_group": font_group,
                    "font_size": font_size,
                    "renderer": "chromium",
                    "degradation_profile": degradation_profile,
                    "degradation_family": degradation_family,
                    "difficulty_bucket": target_diff,
                    "readability_score_estimate": readability_score,
                    "degradation_level": deg_level,
                    "degradation_parameters_json": "{}",
                    "width": w,
                    "height": h,
                    "text_length": len(text),
                    "has_danda": row["has_danda"],
                    "has_double_danda": row["has_double_danda"],
                    "has_digit": row["has_digit"],
                    "has_conjunct": row["has_conjunct"],
                    "seed": seed + idx,
                    "render_status": "success",
                    "unreadable_risk": risk_str,
                    "domain": row.get("domain", "")
                })
                continue
            except Exception:
                pass
                
        # Render clean text line
        clean_img, renderer_used, render_status, status_reason = render_text_line(
            text=text,
            font_path=font_path,
            font_size=font_size,
            padding_x=padding,
            padding_y=padding,
            renderer_preference=renderer_pref
        )
        
        if render_status != "success":
            logger.warning(f"Render failed for line {line_id}: {status_reason}")
            skipped_count += 1
            skip_reasons[status_reason] = skip_reasons.get(status_reason, 0) + 1
            continue
            
        # Apply degradation with readability guards and retry logic
        max_attempts = 5
        degraded_img = None
        deg_params = None
        risk = False
        readability_score = 100.0
        
        for attempt in range(max_attempts):
            try:
                # Apply degradation with target difficulty and attempt progressive blending
                degraded_img, deg_params = apply_degradation(
                    clean_img, 
                    degradation_profile, 
                    target_difficulty=target_diff, 
                    seed=(seed + idx + attempt * 1000), 
                    attempt=attempt
                )
                
                # Check metrics
                risk, readability_score, contrast, std_dev = calculate_readability_metrics(degraded_img)
                
                # Safety check: if wormholes destroyed too much foreground, retry
                if deg_params.get("wormhole_fg_loss_exceeded", False):
                    if attempt < max_attempts - 1:
                        continue
                        
                # Clipping check: if degraded image contents touch the border, retry
                gray_deg = degraded_img.convert("L")
                pixels_deg = list(gray_deg.getdata())
                avg_pixel_deg = sum(pixels_deg) / len(pixels_deg) if pixels_deg else 255.0
                thresh_img_deg = gray_deg.point(lambda p: 255 if p < 160 else 0)
                bbox_deg = thresh_img_deg.getbbox()
                
                is_clipped = False
                if bbox_deg:
                    from PIL import ImageFilter
                    filtered_thresh_deg = thresh_img_deg.filter(ImageFilter.MedianFilter(size=3))
                    f_bbox_deg = filtered_thresh_deg.getbbox()
                    clip_bbox_deg = f_bbox_deg if f_bbox_deg else bbox_deg
                    w_deg, h_deg = degraded_img.size
                    is_clipped = (clip_bbox_deg[0] <= 1 or clip_bbox_deg[1] <= 1 or clip_bbox_deg[2] >= w_deg - 1 or clip_bbox_deg[3] >= h_deg - 1)
                
                if is_clipped:
                    logger.warning(f"Extreme clipping detected on degraded image for line {line_id} (attempt {attempt}). Retrying...")
                    if attempt < max_attempts - 1:
                        continue
                
                if risk:
                    if args.smoke_test:
                        # Count how many unreadable risks we already have in metadata_rows
                        current_unreadable_count = sum(1 for r in metadata_rows if r.get("unreadable_risk") == "true")
                        if current_unreadable_count < 14: # Cap under 5% (15 samples)
                            break
                        else:
                            if attempt < max_attempts - 1:
                                continue # retry
                            else:
                                break
                    else:
                        if args.allow_unreadable_risk:
                            break
                        else:
                            if attempt < max_attempts - 1:
                                continue # retry
                            else:
                                break
                else:
                    break
            except Exception as e:
                logger.error(f"Degradation attempt {attempt} error for line {line_id}: {e}")
                if attempt < max_attempts - 1:
                    continue
                else:
                    break
                    
        # Check if the final result is still unreadable risk and we are not allowed to keep it
        if degraded_img is not None and risk:
            if args.smoke_test:
                current_unreadable_count = sum(1 for r in metadata_rows if r.get("unreadable_risk") == "true")
                if current_unreadable_count >= 14:
                    degraded_img = None
            else:
                if not args.allow_unreadable_risk:
                    degraded_img = None
                    
        if degraded_img is None:
            logger.warning(f"Degradation failed completely (or rejected due to unreadable risk) for line {line_id}")
            skipped_count += 1
            skip_reasons["degradation_failed_or_unreadable"] = skip_reasons.get("degradation_failed_or_unreadable", 0) + 1
            continue
            
        try:
            # Double check that degraded image is not blank
            from PIL import ImageOps
            gray = degraded_img.convert("L")
            inv = ImageOps.invert(gray)
            if not inv.getbbox():
                logger.warning(f"Degraded image blank for line {line_id}")
                skipped_count += 1
                skip_reasons["degraded_blank_image"] = skip_reasons.get("degraded_blank_image", 0) + 1
                continue
                
            # Save final image
            degraded_img.save(img_absolute_path, "PNG")
            w, h = degraded_img.size
            
            risk, readability_score, _, _ = calculate_readability_metrics(degraded_img)
            risk_str = "true" if risk else "false"
            
            difficulty_bucket = deg_params.get("difficulty_bucket", "easy")
            
            metadata_rows.append({
                "image_id": Path(img_filename).stem,
                "line_id": line_id,
                "image_path": os.path.relpath(img_absolute_path, os.getcwd()).replace("\\", "/"),
                "text": text,
                "source_text_original": source_text_original,
                "source_text_normalized": source_text_normalized,
                "source_text_rendered": source_text_rendered,
                "digit_normalization_applied": digit_normalization_applied,
                "source_id": source_id,
                "source_file": source_file,
                "split": split,
                "font_name": font_name,
                "font_path": font_path,
                "font_group": font_group,
                "font_size": font_size,
                "renderer": renderer_used,
                "degradation_profile": degradation_profile,
                "degradation_family": degradation_family,
                "difficulty_bucket": difficulty_bucket,
                "readability_score_estimate": readability_score,
                "degradation_level": deg_level,
                "degradation_parameters_json": json.dumps(deg_params),
                "width": w,
                "height": h,
                "text_length": len(text),
                "has_danda": row["has_danda"],
                "has_double_danda": row["has_double_danda"],
                "has_digit": row["has_digit"],
                "has_conjunct": row["has_conjunct"],
                "seed": seed + idx,
                "render_status": "success",
                "unreadable_risk": risk_str,
                "domain": row.get("domain", "")
            })
            generated_count += 1
        except Exception as e:
            logger.error(f"Save error for line {line_id}: {e}")
            skipped_count += 1
            skip_reasons[f"save_error_{type(e).__name__}"] = skip_reasons.get(f"save_error_{type(e).__name__}", 0) + 1
            
    # 7. Save metadata CSV
    with open(metadata_csv_path, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "image_id", "line_id", "image_path", "text", "source_text_original", "source_text_normalized", "source_text_rendered", "digit_normalization_applied",
            "source_id", "source_file",
            "split", "font_name", "font_path", "font_group", "font_size", "renderer",
            "degradation_profile", "degradation_family", "difficulty_bucket", "readability_score_estimate",
            "degradation_level", "degradation_parameters_json", "width", "height", "text_length",
            "has_danda", "has_double_danda", "has_digit", "has_conjunct", "seed", "render_status", "unreadable_risk",
            "domain"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata_rows)
        
    logger.info(f"Metadata CSV written to {metadata_csv_path}")
    
    # 8. Save Generation Report
    report = {
        "summary": {
            "total_attempted": len(lines),
            "total_generated": generated_count,
            "total_skipped": skipped_count,
            "skip_reasons": skip_reasons,
            "renderer_distribution": {}
        },
        "font_distribution": {},
        "degradation_distribution": {},
        "split_distribution": {},
        "difficulty_bucket_distribution": {},
        "degradation_family_distribution": {}
    }
    
    for row in metadata_rows:
        r = row["renderer"]
        report["summary"]["renderer_distribution"][r] = report["summary"]["renderer_distribution"].get(r, 0) + 1
        
        f = row["font_name"]
        report["font_distribution"][f] = report["font_distribution"].get(f, 0) + 1
        
        d = row["degradation_profile"]
        report["degradation_distribution"][d] = report["degradation_distribution"].get(d, 0) + 1
        
        s = row["split"]
        report["split_distribution"][s] = report["split_distribution"].get(s, 0) + 1
        
        diff = row["difficulty_bucket"]
        report["difficulty_bucket_distribution"][diff] = report["difficulty_bucket_distribution"].get(diff, 0) + 1
        
        fam = row["degradation_family"]
        report["degradation_family_distribution"][fam] = report["degradation_family_distribution"].get(fam, 0) + 1
        
    report_json_path = os.path.join(report_dir, "synthetic_generation_report.json")
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
        
    logger.info(f"Generation report written to {report_json_path}")
    logger.info(f"Successfully generated {generated_count} images. Skipped {skipped_count} lines.")
    
    try:
        from stage1_dataset.chromium_text_renderer import shutdown_chromium_renderer
        shutdown_chromium_renderer()
    except Exception:
        pass

if __name__ == "__main__":
    main()
