import os
import sys
import yaml
import csv
import json
import random
import logging
import collections
import multiprocessing
from pathlib import Path
from PIL import Image

# Add src to python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from stage1_dataset.render_text_line import render_text_line
from stage1_dataset.degradation_profiles import apply_degradation, DIFFICULTY_MAPPING
from stage1_dataset.generate_synthetic_images import calculate_readability_metrics, normalize_devanagari_text, FONTS_TO_DOWNLOAD, ensure_fonts

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

METADATA_DIR = "data/stage1_synthetic/metadata"
IMAGE_DIR = "data/stage1_synthetic"
LABEL_DIR = "data/stage1_synthetic/labels"

def select_font_and_check_coverage(text, requested_font_path, available_fonts, font_cmaps):
    requested_font_name = Path(requested_font_path).stem
    
    def cmap_supports(font_path):
        if font_path not in font_cmaps:
            from fontTools.ttLib import TTFont
            try:
                font_cmaps[font_path] = TTFont(font_path).getBestCmap()
            except Exception:
                return False
        cmap = font_cmaps[font_path]
        if not cmap:
            return False
        for char in text:
            if char in (' ', '\n', '\r', '\t', '\u200c', '\u200d'):
                continue
            if ord(char) not in cmap:
                return False
        return True

    if cmap_supports(requested_font_path):
        return requested_font_path, "true", "false", requested_font_name, requested_font_name, ""
        
    for font_path in available_fonts:
        if font_path == requested_font_path:
            continue
        if cmap_supports(font_path):
            actual_font_name = Path(font_path).stem
            return font_path, "true", "true", requested_font_name, actual_font_name, ""
            
    cmap = font_cmaps.get(requested_font_path, {})
    missing = []
    for char in text:
        if char not in (' ', '\n', '\r', '\t', '\u200c', '\u200d') and ord(char) not in cmap:
            missing.append(char)
    missing_str = "".join(missing)
    return requested_font_path, "false", "false", requested_font_name, requested_font_name, missing_str

def render_worker_chunk(chunk_data):
    """Worker function executed in a separate process."""
    rows, output_subdir, suffix, seed, start_idx, config_path, target_diff_override, composed_prob_override = chunk_data
    
    # Load config locally
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    font_dir = config["paths"]["font_dir"]
    
    # Get available fonts list
    available_fonts = []
    font_slug_map = {}
    font_group_map = {}
    for f_info in FONTS_TO_DOWNLOAD:
        f_path = os.path.join(font_dir, f_info["file_name"])
        if os.path.exists(f_path):
            available_fonts.append(f_path)
            font_slug_map[f_path] = f_info["font_slug"]
            font_group_map[f_path] = f_info["font_group"]

    font_cmaps = {}
    metadata_records = []
    
    # Pre-launch playwright context
    try:
        from stage1_dataset.chromium_text_renderer import get_chromium_renderer
        get_chromium_renderer().start()
    except Exception as e:
        logger.error(f"Failed to pre-start playwright in worker: {e}")

    for local_idx, row in enumerate(rows):
        global_idx = start_idx + local_idx
        line_id = row["line_id"]
        source_id = row["source_id"]
        source_file = row["source_file"]
        split = row["split"]
        
        # Normalize text
        source_text_original = row["text"]
        source_text_normalized = normalize_devanagari_text(source_text_original)
        source_text_rendered = source_text_normalized
        digit_normalization_applied = "true" if source_text_original != source_text_normalized else "false"
        
        rng = random.Random(seed + global_idx)
        
        # Pick requested font and check coverage
        requested_font_path = rng.choice(available_fonts)
        actual_font_path, glyph_cov, fallback_used, req_font_name, act_font_name, missing_gl = select_font_and_check_coverage(
            source_text_rendered, requested_font_path, available_fonts, font_cmaps
        )
        
        font_group = font_group_map[actual_font_path]
        font_slug = font_slug_map[actual_font_path]
        
        font_size_min = config["rendering"]["font_size_min"]
        font_size_max = config["rendering"]["font_size_max"]
        font_size = rng.randint(font_size_min, font_size_max)
        
        padding_min = config["rendering"]["padding_min"]
        padding_max = config["rendering"]["padding_max"]
        padding = rng.randint(padding_min, padding_max)
        
        # Degradations
        degradations = list(config["degradation_distribution"].keys())
        degradation_weights = list(config["degradation_distribution"].values())
        w_sum = sum(degradation_weights)
        degradation_weights = [w / w_sum for w in degradation_weights]
        
        composed_prob = composed_prob_override if composed_prob_override is not None else config.get("degradation_mode", {}).get("composed_profile_prob", 0.30)
        
        if target_diff_override:
            target_diff = target_diff_override
        else:
            target_diff = rng.choices(["easy", "medium", "hard"], weights=[0.45, 0.40, 0.15], k=1)[0]
            
        if rng.random() < composed_prob:
            is_composed = True
            degradation_profile = "composed"
            degradation_family = "composed"
            deg_level = -1
        else:
            is_composed = False
            degradation_family = "single"
            filtered_degs = [d for d in degradations if DIFFICULTY_MAPPING.get(d) == target_diff]
            if not filtered_degs:
                filtered_degs = degradations
            filtered_weights = [config["degradation_distribution"].get(d, 1.0) for d in filtered_degs]
            f_sum = sum(filtered_weights)
            filtered_weights = [w / f_sum for w in filtered_weights]
            degradation_profile = rng.choices(filtered_degs, weights=filtered_weights, k=1)[0]
            deg_level = int(degradation_profile.split("_")[0][1:])
            
        # Render line
        clean_img, renderer_used, render_status, status_reason = render_text_line(
            text=source_text_rendered,
            font_path=actual_font_path,
            font_size=font_size,
            padding_x=padding,
            padding_y=padding,
            renderer_preference="chromium"
        )
        
        if render_status != "success":
            logger.error(f"Render failed in worker for line {line_id}: {status_reason}")
            continue
            
        # Apply degradation
        max_attempts = 5
        degraded_img = None
        deg_params = None
        risk = False
        readability_score = 100.0
        
        for attempt in range(max_attempts):
            try:
                degraded_img, deg_params = apply_degradation(
                    clean_img,
                    degradation_profile,
                    target_difficulty=target_diff,
                    seed=(seed + global_idx + attempt * 1000),
                    attempt=attempt
                )
                risk, readability_score, contrast, std_dev = calculate_readability_metrics(degraded_img)
                
                # Check clipping
                gray_deg = degraded_img.convert("L")
                bbox_deg = gray_deg.point(lambda p: 255 if p < 160 else 0).getbbox()
                is_clipped = False
                if bbox_deg:
                    w_deg, h_deg = degraded_img.size
                    is_clipped = (bbox_deg[0] <= 1 or bbox_deg[1] <= 1 or bbox_deg[2] >= w_deg - 1 or bbox_deg[3] >= h_deg - 1)
                    
                if is_clipped or risk:
                    if attempt < max_attempts - 1:
                        continue
                break
            except Exception as e:
                if attempt < max_attempts - 1:
                    continue
                break
                
        if degraded_img is None or risk:
            # Fallback clean print
            degraded_img, deg_params = apply_degradation(
                clean_img,
                "D0_clean_print",
                target_difficulty="easy",
                seed=(seed + global_idx),
                attempt=0
            )
            risk, readability_score, _, _ = calculate_readability_metrics(degraded_img)
            
        # Save image file
        save_dir = os.path.join(IMAGE_DIR, output_subdir, split)
        os.makedirs(save_dir, exist_ok=True)
        
        img_filename = f"{line_id}_{suffix}.png"
        img_relative_path = os.path.join(IMAGE_DIR, output_subdir, split, img_filename)
        img_absolute_path = os.path.abspath(img_relative_path)
        
        degraded_img.save(img_absolute_path, "PNG")
        
        metadata_records.append({
            "image_path": f"{output_subdir}/{split}/{img_filename}",
            "line_id": line_id,
            "source_id": source_id,
            "source_file": source_file,
            "domain": row["domain"],
            "split": split,
            "text": source_text_rendered,
            "source_text_original": source_text_original,
            "source_text_normalized": source_text_normalized,
            "source_text_rendered": source_text_rendered,
            "digit_normalization_applied": digit_normalization_applied,
            "renderer": "chromium",
            "degradation_profile": degradation_profile,
            "degradation_family": degradation_family,
            "degradation_parameters_json": json.dumps(deg_params),
            "difficulty_bucket": deg_params.get("difficulty_bucket", target_diff),
            "unreadable_risk": "true" if risk else "false",
            "glyph_coverage_passed": glyph_cov,
            "font_fallback_used": fallback_used,
            "requested_font_name": req_font_name,
            "actual_font_name": act_font_name,
            "missing_glyphs": missing_gl
        })
        
    # Shutdown local playwright
    try:
        from stage1_dataset.chromium_text_renderer import shutdown_chromium_renderer
        shutdown_chromium_renderer()
    except Exception:
        pass
        
    return metadata_records

def run_parallel_generation(rows, output_subdir, suffix, seed, config_path, target_diff_override=None, composed_prob_override=None):
    """Splits work into chunks and executes in parallel using multiprocessing."""
    num_workers = min(8, os.cpu_count() or 4)
    logger.info(f"Starting parallel generation for {len(rows)} items in {output_subdir} (suffix={suffix}) using {num_workers} processes...")
    
    # Clean output directories first to prevent leftovers
    clear_subdirs = [os.path.join(IMAGE_DIR, output_subdir, s) for s in ["train", "val", "test"]]
    for sd in clear_subdirs:
        if os.path.exists(sd):
            for file_name in os.listdir(sd):
                f_path = os.path.join(sd, file_name)
                if os.path.isfile(f_path):
                    try:
                        os.remove(f_path)
                    except Exception:
                        pass
                        
    # Chunk rows
    chunk_size = (len(rows) + num_workers - 1) // num_workers
    chunks_data = []
    
    for i in range(num_workers):
        start_idx = i * chunk_size
        end_idx = min(len(rows), (i + 1) * chunk_size)
        chunk = rows[start_idx:end_idx]
        if not chunk:
            continue
        chunks_data.append((
            chunk,
            output_subdir,
            suffix,
            seed,
            start_idx,
            config_path,
            target_diff_override,
            composed_prob_override
        ))
        
    # Run in Multiprocessing Pool
    ctx = multiprocessing.get_context("spawn") # Playwright works best with spawn context
    results = []
    with ctx.Pool(processes=num_workers) as pool:
        for res in pool.imap_unordered(render_worker_chunk, chunks_data):
            results.extend(res)
            
    logger.info(f"Completed parallel generation. Generated {len(results)} images in {output_subdir}.")
    return results

def main():
    print("=" * 70)
    print("MANTRA Stage 1 Final Parallel Synthetic Image Generation Pipeline")
    print("=" * 70)
    
    config_path = "configs/stage1_synth100k_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    base_seed = config["dataset"].get("random_seed", 42)
    font_dir = config["paths"]["font_dir"]
    
    # Download/verify fonts
    ensure_fonts(font_dir, METADATA_DIR)
    
    available_fonts = []
    font_slug_map = {}
    font_group_map = {}
    for f_info in FONTS_TO_DOWNLOAD:
        f_path = os.path.join(font_dir, f_info["file_name"])
        if os.path.exists(f_path):
            available_fonts.append(f_path)
            font_slug_map[f_path] = f_info["font_slug"]
            font_group_map[f_path] = f_info["font_group"]
            
    # 2. Load merged labels
    all_lines_csv = os.path.join(LABEL_DIR, "stage1_all_lines.csv")
    if not os.path.exists(all_lines_csv):
        print(f"ERROR: Merged labels file not found: {all_lines_csv}")
        return
        
    all_rows = []
    with open(all_lines_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            all_rows.append(r)
    N = len(all_rows)
    print(f"Loaded N = {N} final merged labels from {all_lines_csv}")
    
    # 3. Pass A: Load existing successful core images metadata
    core_meta_csv = os.path.join(METADATA_DIR, "synthetic_image_metadata_core_merged.csv")
    if not os.path.exists(core_meta_csv):
        print(f"ERROR: Existing core metadata file not found: {core_meta_csv}")
        return
        
    print("\n[Pass A] Loading existing successful core images metadata...")
    core_metadata_rows = []
    with open(core_meta_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            mapped = {
                "image_path": r["image_path"],
                "line_id": r["line_id"],
                "source_id": r["source_id"],
                "source_file": r["source_file"],
                "domain": r["domain"],
                "split": r["split"],
                "text": r["text"],
                "source_text_original": r["source_text_original"],
                "source_text_normalized": r["source_text_normalized"],
                "source_text_rendered": r["source_text_rendered"],
                "digit_normalization_applied": r["digit_normalization_applied"],
                "renderer": r["renderer"],
                "degradation_profile": r["degradation_profile"],
                "degradation_family": r["degradation_family"],
                "degradation_parameters_json": r["degradation_parameters_json"],
                "difficulty_bucket": r["difficulty_bucket"],
                "unreadable_risk": r["unreadable_risk"],
                "glyph_coverage_passed": "true",
                "font_fallback_used": "false",
                "requested_font_name": r.get("font_name", "NotoSansDevanagari-Regular"),
                "actual_font_name": r.get("font_name", "NotoSansDevanagari-Regular"),
                "missing_glyphs": "",
                "variant_index": 1,
                "generation_round": "core_reuse"
            }
            core_metadata_rows.append(mapped)
            
    print(f"Loaded {len(core_metadata_rows)} existing core metadata rows.")
    
    # 4. Pass B Part 1: Identify and generate the 823 missing core variant-1 images
    core_lines_csv = os.path.join(LABEL_DIR, "stage1_core_merged_lines.csv")
    core_rows = []
    with open(core_lines_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            core_rows.append(r)
            
    existing_core_ids = set(r["line_id"] for r in core_metadata_rows)
    missing_core_rows = [r for r in core_rows if r["line_id"] not in existing_core_ids]
    print(f"\n[Pass B Part 1] Identified {len(missing_core_rows)} missing core variant-1 rows.")
    
    for row in missing_core_rows:
        if "domain" not in row or not row["domain"]:
            if "historical_nepali" in row["line_id"]:
                row["domain"] = "historical_nepali"
            elif "nepali_literary" in row["line_id"]:
                row["domain"] = "nepali_literary"
            else:
                row["domain"] = "sanskrit"
                
    missing_core_metadata = []
    if missing_core_rows:
        missing_core_metadata = run_parallel_generation(
            missing_core_rows,
            output_subdir="images_stage1_all_variant1_missing_core",
            suffix="v01",
            seed=base_seed + 1000000,
            config_path=config_path
        )
        for rec in missing_core_metadata:
            rec["variant_index"] = 1
            rec["generation_round"] = "coverage_variant1"
            
        missing_core_meta_csv = os.path.join(METADATA_DIR, "synthetic_image_metadata_stage1_all_variant1_missing_core.csv")
        with open(missing_core_meta_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=missing_core_metadata[0].keys() if missing_core_metadata else ["image_path"])
            writer.writeheader()
            writer.writerows(missing_core_metadata)
        print(f"Saved missing core metadata to {missing_core_meta_csv}")
        
    # 5. Pass B Part 2: Generate coverage variant-1 images (6,776 rows)
    coverage_rows = [r for r in all_rows if r["domain"].startswith("coverage_")]
    print(f"\n[Pass B Part 2] Found {len(coverage_rows)} coverage rows in stage1_all_lines.csv.")
    
    coverage_metadata = []
    if coverage_rows:
        coverage_metadata = run_parallel_generation(
            coverage_rows,
            output_subdir="images_stage1_all_variant1_coverage_only",
            suffix="v01",
            seed=base_seed + 2000000,
            config_path=config_path
        )
        for rec in coverage_metadata:
            rec["variant_index"] = 1
            rec["generation_round"] = "coverage_variant1"
            
        coverage_meta_csv = os.path.join(METADATA_DIR, "synthetic_image_metadata_stage1_all_variant1_coverage_only.csv")
        with open(coverage_meta_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=coverage_metadata[0].keys() if coverage_metadata else ["image_path"])
            writer.writeheader()
            writer.writerows(coverage_metadata)
        print(f"Saved coverage metadata to {coverage_meta_csv}")

    # 6. Pass C: Generate variant-2 for all final labels (N rows)
    print(f"\n[Pass C] Generating variant-2 images for all {N} rows in stage1_all_lines.csv...")
    variant2_metadata = run_parallel_generation(
        all_rows,
        output_subdir="images_stage1_all_variant2",
        suffix="v02",
        seed=base_seed + 3000000,
        config_path=config_path,
        target_diff_override=None,
        composed_prob_override=0.40 # Broader degradation diversity
    )
    for rec in variant2_metadata:
        rec["variant_index"] = 2
        rec["generation_round"] = "all_variant2"
        
    variant2_meta_csv = os.path.join(METADATA_DIR, "synthetic_image_metadata_stage1_all_variant2.csv")
    with open(variant2_meta_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=variant2_metadata[0].keys() if variant2_metadata else ["image_path"])
        writer.writeheader()
        writer.writerows(variant2_metadata)
    print(f"Saved variant-2 metadata to {variant2_meta_csv}")

    # 7. Pass D: Exact top-up to 100,000 images
    topup_count = 100000 - (2 * N)
    print(f"\n[Pass D] Computing top-up: 100000 - (2 * {N}) = {topup_count}")
    
    topup_metadata = []
    if topup_count > 0:
        print(f"Sampling exactly {topup_count} hard examples with priority mix...")
        # Targeted sampling
        numerals = [r for r in all_rows if r["domain"] == "coverage_numeral"]
        conjuncts = [r for r in all_rows if r["domain"] == "coverage_conjunct"]
        admins = [r for r in all_rows if r["domain"] == "coverage_admin"]
        matras = [r for r in all_rows if r["domain"] == "coverage_matra"]
        diff_core = [
            r for r in all_rows
            if r["domain"] in ("historical_nepali", "sanskrit")
            and (r["has_conjunct"].lower() == "true" or r["has_digit"].lower() == "true")
        ]
        
        n_num = int(round(0.30 * topup_count))
        n_conj = int(round(0.30 * topup_count))
        n_adm = int(round(0.20 * topup_count))
        n_mat = int(round(0.10 * topup_count))
        n_core = topup_count - (n_num + n_conj + n_adm + n_mat)
        
        topup_rng = random.Random(base_seed + 9999)
        def safe_sample(pool, target):
            if not pool:
                return []
            if len(pool) >= target:
                return topup_rng.sample(pool, target)
            res = list(pool)
            while len(res) < target:
                res.append(topup_rng.choice(pool))
            return res
            
        sampled = []
        sampled.extend(safe_sample(numerals, n_num))
        sampled.extend(safe_sample(conjuncts, n_conj))
        sampled.extend(safe_sample(admins, n_adm))
        sampled.extend(safe_sample(matras, n_mat))
        sampled.extend(safe_sample(diff_core, n_core))
        
        if len(sampled) != topup_count:
            if len(sampled) < topup_count:
                sampled.extend(safe_sample(all_rows, topup_count - len(sampled)))
            else:
                sampled = sampled[:topup_count]
                
        print(f"Achieved top-up sample distribution count: {len(sampled)}")
        
        topup_metadata = run_parallel_generation(
            sampled,
            output_subdir="images_stage1_all_topup",
            suffix="v03",
            seed=base_seed + 4000000,
            config_path=config_path,
            target_diff_override="hard"
        )
        for rec in topup_metadata:
            rec["variant_index"] = 3
            rec["generation_round"] = "targeted_topup"
            
        topup_meta_csv = os.path.join(METADATA_DIR, "synthetic_image_metadata_stage1_all_topup.csv")
        with open(topup_meta_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=topup_metadata[0].keys() if topup_metadata else ["image_path"])
            writer.writeheader()
            writer.writerows(topup_metadata)
        print(f"Saved top-up metadata to {topup_meta_csv}")
        
    # 8. Combine metadata and write final unified metadata file
    print("\nCombining all metadata rows...")
    final_combined_rows = []
    final_combined_rows.extend(core_metadata_rows)
    final_combined_rows.extend(missing_core_metadata)
    final_combined_rows.extend(coverage_metadata)
    final_combined_rows.extend(variant2_metadata)
    final_combined_rows.extend(topup_metadata)
    
    total_images = len(final_combined_rows)
    print(f"Total final image metadata rows combined: {total_images}")
    
    # 9. Verify constraints and physical existence
    print("\nVerifying constraints and physical existence...")
    not_exists = []
    for idx, r in enumerate(final_combined_rows):
        path = os.path.join(IMAGE_DIR, r["image_path"])
        if not os.path.exists(path):
            not_exists.append(path)
            
    if not_exists:
        print(f"ERROR: {len(not_exists)} physical images do not exist!")
        for p in not_exists[:5]:
            print(f"  Missing: {p}")
    else:
        print("PASS: All image paths in metadata physically exist.")
        
    # Check counts
    if total_images == 100000:
        print("PASS: Total image count is EXACTLY 100,000.")
    else:
        print(f"FAIL: Total image count is {total_images} (expected 100,000)!")
        
    # Save unified metadata
    final_meta_csv = os.path.join(METADATA_DIR, "synthetic_image_metadata_stage1_all_final.csv")
    fieldnames = [
        "image_path", "line_id", "source_id", "source_file", "domain", "split", "text",
        "source_text_original", "source_text_normalized", "source_text_rendered", "digit_normalization_applied",
        "renderer", "degradation_profile", "degradation_family", "degradation_parameters_json",
        "difficulty_bucket", "unreadable_risk", "glyph_coverage_passed", "font_fallback_used",
        "requested_font_name", "actual_font_name", "missing_glyphs", "variant_index", "generation_round"
    ]
    with open(final_meta_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_combined_rows)
    print(f"Saved final unified metadata to {final_meta_csv}")
    print("=" * 70)
    print("ALL GENERATION TASKS COMPLETED")
    print("=" * 70)

if __name__ == "__main__":
    main()
