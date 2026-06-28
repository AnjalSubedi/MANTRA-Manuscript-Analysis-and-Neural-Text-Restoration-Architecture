import os
import sys
import csv
import random
import logging
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Add src to python path to import modules
sys.path.append(str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def make_grid(cells, output_path, cols=3, cell_width=600, cell_height=140, label_font_path=None):
    """
    Composes a grid image from a list of cells.
    Each cell is a dict: {"image_path": str, "label": str} or {"image": PIL.Image, "label": str}
    """
    if not cells:
        logger.warning(f"No cells to draw for grid: {output_path}")
        return False
        
    rows = (len(cells) + cols - 1) // cols
    grid_w = cols * cell_width
    grid_h = rows * cell_height
    
    grid_img = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
    draw = ImageDraw.Draw(grid_img)
    
    if label_font_path and os.path.exists(label_font_path):
        try:
            font = ImageFont.truetype(label_font_path, 11)
        except Exception:
            font = ImageFont.load_default()
    else:
        font = ImageFont.load_default()
        
    for idx, cell in enumerate(cells):
        r = idx // cols
        c = idx % cols
        
        cell_img = None
        if "image" in cell:
            cell_img = cell["image"]
        elif "image_path" in cell and os.path.exists(cell['image_path']):
            try:
                cell_img = Image.open(cell["image_path"])
            except Exception as e:
                logger.error(f"Failed to open image {cell['image_path']}: {e}")
                
        if cell_img is not None:
            margin = 10
            max_w = cell_width - 2 * margin
            max_h = cell_height - 35 - 2 * margin
            
            w, h = cell_img.size
            ratio = min(max_w / w, max_h / h)
            new_w = int(w * ratio)
            new_h = int(h * ratio)
            
            resized_img = cell_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            x_offset = c * cell_width + (cell_width - new_w) // 2
            y_offset = r * cell_height + margin + (max_h - new_h) // 2
            grid_img.paste(resized_img, (x_offset, y_offset))
            
        cx1 = c * cell_width
        cy1 = r * cell_height
        cx2 = cx1 + cell_width
        cy2 = cy1 + cell_height
        draw.rectangle([cx1, cy1, cx2, cy2], outline=(220, 220, 220), width=1)
        
        label = cell.get("label", "")
        try:
            t_bbox = draw.textbbox((0, 0), label, font=font)
            tw = t_bbox[2] - t_bbox[0]
            tx = cx1 + (cell_width - tw) // 2
            ty = cy2 - 25
            draw.text((tx, ty), label, font=font, fill=(50, 50, 50))
        except Exception:
            draw.text((cx1 + 10, cy2 - 25), label, font=font, fill=(50, 50, 50))
            
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    grid_img.save(output_path, "PNG")
    logger.info(f"Grid saved to {output_path}")
    return True

def generate_shaping_qa_grid(font_dir, output_path, label_font_path=None):
    """
    Generates the Devanagari shaping QA grid showing test strings
    rendered by each font clean (D0) without degradation.
    """
    from stage1_dataset.render_text_line import render_text_line
    
    test_strings = [
        "नेपालमा इसापूर्व तेस्रो शताब्दीदेखिकै शिलालेखहरू प्राप्त भएका छन् । भारतीय",
        "इसापूर्व",
        "पूर्व",
        "प्राप्त",
        "शिलालेखहरू प्राप्त भएका छन्",
        "ग्रन्थ प्रज्ञा श्रद्धा राष्ट्र्य प्राप्ति",
        "क्ति क्ष ज्ञ त्र श्र प्र द्ध ह्म स्त्र",
        "कि की कु कू कृ के कै को कौ कं कः"
    ]
    
    font_files = [os.path.join(font_dir, f) for f in os.listdir(font_dir) if f.endswith(".ttf")]
    if not font_files:
        logger.error(f"No fonts found in {font_dir} for shaping QA grid.")
        return False
        
    cells = []
    for font_path in font_files:
        font_name = os.path.splitext(os.path.basename(font_path))[0]
        for text in test_strings:
            img, renderer_used, status, reason = render_text_line(
                text=text,
                font_path=font_path,
                font_size=32,
                padding_x=15,
                padding_y=15,
                bg_color=(255, 255, 255),
                text_color=(0, 0, 0),
                renderer_preference="chromium"
            )
            label = f"{font_name} | {renderer_used}"
            cells.append({"image": img, "label": label})
            
    return make_grid(cells, output_path, cols=len(test_strings), cell_width=1200, cell_height=180, label_font_path=label_font_path)

def generate_numeral_symbol_qa_grid(font_dir, output_path, label_font_path=None):
    """
    Generates the Devanagari numerals and symbols QA grid showing test strings
    rendered by each font clean (D0) without degradation.
    All strings MUST contain only Devanagari digits ०-९ (U+0966–U+096F), never ASCII 0-9.

    IMPORTANT: All digit Unicode escapes verified against the Devanagari digit block:
        ० U+0966, १ U+0967, २ U+0968, ३ U+0969, ४ U+096A,
        ५ U+096B, ६ U+096C, ७ U+096D, ८ U+096E, ९ U+096F
    """
    from stage1_dataset.render_text_line import render_text_line, normalize_devanagari_text

    # All digit sequences built digit-by-digit using the verified codepoint map:
    #   D0=\u0966  D1=\u0967  D2=\u0968  D3=\u0969  D4=\u096A
    #   D5=\u096B  D6=\u096C  D7=\u096D  D8=\u096E  D9=\u096F
    #
    # String 1: "त्यतिबेला रू. १५ सम्म पर्दथ्यो ।"
    #   १=\u0967  ५=\u096B
    _s1 = ("\u0924\u094d\u092f\u0924\u093f\u092c\u0947\u0932\u093e "
           "\u0930\u0942. \u0967\u096b "
           "\u0938\u092e\u094d\u092e \u092a\u0930\u094d\u0926\u0925\u094d\u092f\u094b \u0964")

    # String 2: "वि. सं १६६० मा निर्माण भएको"
    #   १=\u0967  ६=\u096C  ६=\u096C  ०=\u0966
    _s2 = ("\u0935\u093f. \u0938\u0902 "
           "\u0967\u096c\u096c\u0966 "
           "\u092e\u093e \u0928\u093f\u0930\u094d\u092e\u093e\u0923 \u092d\u090f\u0915\u094b")

    # String 3: "यो वर्ष २०२७ सालको हो ।"
    #   २=\u0968  ०=\u0966  २=\u0968  ७=\u096D
    _s3 = ("\u092f\u094b \u0935\u0930\u094d\u0937 "
           "\u0968\u0966\u0968\u096d "
           "\u0938\u093e\u0932\u0915\u094b \u0939\u094b \u0964")

    # String 4: "उमेर ५८ वर्ष पुगेका"
    #   ५=\u096B  ८=\u096E
    _s4 = ("\u0909\u092e\u0947\u0930 "
           "\u096b\u096e "
           "\u0935\u0930\u094d\u0937 \u092a\u0941\u0917\u0947\u0915\u093e")

    # String 5: "अङ्कहरू: १२३४५६७८९०"
    #   sequence: \u0967\u0968\u0969\u096A\u096B\u096C\u096D\u096E\u096F\u0966
    _s5 = ("\u0905\u0919\u094d\u0915\u0939\u0930\u0942: "
           "\u0967\u0968\u0969\u096a\u096b\u096c\u096d\u096e\u096f\u0966")

    # String 6: "विविध चिह्नहरू: ? ! , ( ) [ ] । ॥"  (no digits)
    _s6 = ("\u0935\u093f\u0935\u093f\u0927 \u091a\u093f\u0939\u094d\u0928\u0939\u0930\u0942: "
           "? ! , ( ) [ ] \u0964 \u0965")

    test_strings = [_s1, _s2, _s3, _s4, _s5, _s6]

    # Belt-and-suspenders: normalize each string (converts any ASCII 0-9 → ०-९)
    test_strings = [normalize_devanagari_text(ts) for ts in test_strings]

    # QA assertion 1: no ASCII digit codepoints
    ascii_digit_count = sum(1 for ts in test_strings for c in ts if c in "0123456789")
    # QA assertion 2: log actual digit codepoints found in strings for manual verification
    digit_codepoints = [(hex(ord(c)), c) for ts in test_strings for c in ts
                        if 0x0966 <= ord(c) <= 0x096F]
    if ascii_digit_count > 0:
        logger.error(
            f"[QA FAIL] numeral_symbol_qa_grid: {ascii_digit_count} ASCII digit(s) remain "
            f"after normalization! Strings: {test_strings}"
        )
    else:
        logger.info(
            f"[QA PASS] numeral_symbol_qa_grid: ascii_digit_count=0  "
            f"devanagari_digit_count={len(digit_codepoints)}  "
            f"digits_found={[c for _, c in digit_codepoints]}"
        )

    font_files = [os.path.join(font_dir, f) for f in os.listdir(font_dir) if f.endswith(".ttf")]
    if not font_files:
        logger.error(f"No fonts found in {font_dir} for numeral/symbol QA grid.")
        return False

    cells = []
    for font_path in font_files:
        font_name = os.path.splitext(os.path.basename(font_path))[0]
        for text in test_strings:
            img, renderer_used, status, reason = render_text_line(
                text=text,
                font_path=font_path,
                font_size=32,
                padding_x=15,
                padding_y=15,
                bg_color=(255, 255, 255),
                text_color=(0, 0, 0),
                renderer_preference="chromium"
            )
            label = f"{font_name} | {renderer_used}"
            cells.append({"image": img, "label": label})

    return make_grid(cells, output_path, cols=len(test_strings),
                     cell_width=1200, cell_height=180, label_font_path=label_font_path)

def generate_ascii_negative_control_grid(font_dir, output_path, label_font_path=None):
    """
    Generates a separate ASCII negative-control grid that intentionally shows
    ASCII digits for comparison/debugging only.
    This must NEVER be included in the main numeral_symbol_qa_grid.
    """
    from stage1_dataset.render_text_line import render_text_line
    
    # Explicitly labelled ASCII-negative-control strings (NOT normalized)
    test_strings_ascii = [
        "\u0930\u0942. 15 \u0938\u092e\u094d\u092e [\u0041\u0053\u0043\u0049\u0049-\u006e\u0065\u0067\u0061\u0074\u0069\u0076\u0065-\u0063\u006f\u006e\u0074\u0072\u006f\u006c]",
        "1234567890 [\u0041\u0053\u0043\u0049\u0049-\u006e\u0065\u0067\u0061\u0074\u0069\u0076\u0065-\u0063\u006f\u006e\u0074\u0072\u006f\u006c]",
    ]
    
    font_files = [os.path.join(font_dir, f) for f in os.listdir(font_dir) if f.endswith(".ttf")]
    if not font_files:
        logger.error(f"No fonts found in {font_dir} for ASCII negative control grid.")
        return False
        
    cells = []
    for font_path in font_files[:3]:  # Only first 3 fonts for brevity
        font_name = os.path.splitext(os.path.basename(font_path))[0]
        for text in test_strings_ascii:
            img, renderer_used, status, reason = render_text_line(
                text=text,
                font_path=font_path,
                font_size=32,
                padding_x=15,
                padding_y=15,
                bg_color=(255, 240, 240),  # Pink tint to signal "control" grid
                text_color=(180, 0, 0),
                renderer_preference="chromium"
            )
            label = f"[ASCII-CTRL] {font_name}"
            cells.append({"image": img, "label": label})
            
    return make_grid(cells, output_path, cols=len(test_strings_ascii), cell_width=1200, cell_height=180, label_font_path=label_font_path)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="MANTRA Sample Grid Generation")
    parser.add_argument("--metadata", type=str, default="data/stage1_synthetic/metadata/synthetic_image_metadata.csv", help="Path to metadata CSV")
    parser.add_argument("--output-dir", type=str, default="data/stage1_synthetic/reports/sample_grids", help="Path to output grids directory")
    args = parser.parse_args()
    
    metadata_csv = args.metadata
    grids_dir = args.output_dir
    font_dir = "data/stage1_synthetic/fonts"
    
    is_digits_only = "digits_only" in grids_dir or "digits_only" in metadata_csv
    
    label_font_path = os.path.join(font_dir, "Poppins-Regular.ttf")
    if not os.path.exists(label_font_path):
        label_font_path = None
        
    if not is_digits_only:
        shaping_grid_path = os.path.join(grids_dir, "shaping_qa_grid.png")
        generate_shaping_qa_grid(font_dir, shaping_grid_path, label_font_path)
        
        numeral_grid_path = os.path.join(grids_dir, "numeral_symbol_qa_grid.png")
        generate_numeral_symbol_qa_grid(font_dir, numeral_grid_path, label_font_path)
        
        ascii_control_grid_path = os.path.join(grids_dir, "ascii_negative_control_grid.png")
        generate_ascii_negative_control_grid(font_dir, ascii_control_grid_path, label_font_path)

    
    if not os.path.exists(metadata_csv):
        logger.error(f"Metadata file {metadata_csv} not found. Cannot generate grids.")
        return
        
    records = []
    with open(metadata_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
            
    if not records:
        logger.warning("No records found in metadata CSV.")
        return
        
    rand = random.Random(42)
    
    is_core_merged = len(records) > 0 and "domain" in records[0]
    
    if is_core_merged:
        # 1. mixed_core_merged_grid.png (15 random samples)
        sample_size = min(15, len(records))
        sampled = rand.sample(records, sample_size)
        cells = [
            {
                "image_path": r["image_path"],
                "label": f"{r['line_id']}, {r['font_name']}, {r['degradation_profile'].split('_')[0]}"
            }
            for r in sampled
        ]
        out_path = os.path.join(grids_dir, "mixed_core_merged_grid.png")
        make_grid(cells, out_path, cols=3, label_font_path=label_font_path)
        
        # 2. historical_nepali_grid.png (12 random samples)
        hist_rows = [r for r in records if r.get("domain") == "historical_nepali"]
        if hist_rows:
            sample_size = min(12, len(hist_rows))
            sampled = rand.sample(hist_rows, sample_size)
            cells = [
                {
                    "image_path": r["image_path"],
                    "label": f"{r['line_id']}, {r['font_name']}, {r['degradation_profile'].split('_')[0]}"
                }
                for r in sampled
            ]
            out_path = os.path.join(grids_dir, "historical_nepali_grid.png")
            make_grid(cells, out_path, cols=3, label_font_path=label_font_path)
            
        # 3. nepali_literary_grid.png (12 random samples)
        lit_rows = [r for r in records if r.get("domain") == "nepali_literary"]
        if lit_rows:
            sample_size = min(12, len(lit_rows))
            sampled = rand.sample(lit_rows, sample_size)
            cells = [
                {
                    "image_path": r["image_path"],
                    "label": f"{r['line_id']}, {r['font_name']}, {r['degradation_profile'].split('_')[0]}"
                }
                for r in sampled
            ]
            out_path = os.path.join(grids_dir, "nepali_literary_grid.png")
            make_grid(cells, out_path, cols=3, label_font_path=label_font_path)
            
        # 4. sanskrit_grid.png (12 random samples)
        sk_rows = [r for r in records if r.get("domain") == "sanskrit"]
        if sk_rows:
            sample_size = min(12, len(sk_rows))
            sampled = rand.sample(sk_rows, sample_size)
            cells = [
                {
                    "image_path": r["image_path"],
                    "label": f"{r['line_id']}, {r['font_name']}, {r['degradation_profile'].split('_')[0]}"
                }
                for r in sampled
            ]
            out_path = os.path.join(grids_dir, "sanskrit_grid.png")
            make_grid(cells, out_path, cols=3, label_font_path=label_font_path)
            
        # 5. difficulty_easy_grid.png (12 samples)
        easy_rows = [r for r in records if r.get("difficulty_bucket", "easy") == "easy"]
        if easy_rows:
            sample_size = min(12, len(easy_rows))
            sampled = rand.sample(easy_rows, sample_size)
            cells = [
                {
                    "image_path": r["image_path"],
                    "label": f"{r['line_id']}, {r['font_name']}, {r['degradation_profile'].split('_')[0]}"
                }
                for r in sampled
            ]
            out_path = os.path.join(grids_dir, "difficulty_easy_grid.png")
            make_grid(cells, out_path, cols=3, label_font_path=label_font_path)
            
        # 6. difficulty_medium_grid.png (12 samples)
        med_rows = [r for r in records if r.get("difficulty_bucket", "easy") == "medium"]
        if med_rows:
            sample_size = min(12, len(med_rows))
            sampled = rand.sample(med_rows, sample_size)
            cells = [
                {
                    "image_path": r["image_path"],
                    "label": f"{r['line_id']}, {r['font_name']}, {r['degradation_profile'].split('_')[0]}"
                }
                for r in sampled
            ]
            out_path = os.path.join(grids_dir, "difficulty_medium_grid.png")
            make_grid(cells, out_path, cols=3, label_font_path=label_font_path)
            
        # 7. difficulty_hard_grid.png (12 samples)
        hard_rows = [r for r in records if r.get("difficulty_bucket", "easy") == "hard"]
        if hard_rows:
            sample_size = min(12, len(hard_rows))
            sampled = rand.sample(hard_rows, sample_size)
            cells = [
                {
                    "image_path": r["image_path"],
                    "label": f"{r['line_id']}, {r['font_name']}, {r['degradation_profile'].split('_')[0]}"
                }
                for r in sampled
            ]
            out_path = os.path.join(grids_dir, "difficulty_hard_grid.png")
            make_grid(cells, out_path, cols=3, label_font_path=label_font_path)
            
        # 8. degradation_profile_grid.png (12 mixed samples)
        by_deg = {}
        for r in records:
            by_deg.setdefault(r["degradation_profile"], []).append(r)
            
        deg_sampled = []
        for deg_profile, rows in sorted(by_deg.items()):
            deg_sampled.append(rand.choice(rows))
            
        if len(deg_sampled) > 12:
            deg_sampled = rand.sample(deg_sampled, 12)
        elif len(deg_sampled) < 12 and records:
            remaining = [r for r in records if r not in deg_sampled]
            needed = 12 - len(deg_sampled)
            deg_sampled.extend(rand.sample(remaining, min(needed, len(remaining))))
            
        cells = [
            {
                "image_path": r["image_path"],
                "label": f"{r['line_id']}, {r['degradation_profile'].split('_')[0]}"
            }
            for r in deg_sampled
        ]
        out_path = os.path.join(grids_dir, "degradation_profile_grid.png")
        make_grid(cells, out_path, cols=3, label_font_path=label_font_path)
        
        return

    if is_digits_only:
        # Generate the three digits-only grids:
        # 1. mixed_digits_only_grid.png (random 15 samples)
        sample_size = min(15, len(records))
        sampled = rand.sample(records, sample_size)
        cells = [
            {
                "image_path": r["image_path"],
                "label": f"{r['line_id']}, {r['font_name']}, {r['degradation_profile'].split('_')[0]}"
            }
            for r in sampled
        ]
        out_path = os.path.join(grids_dir, "mixed_digits_only_grid.png")
        make_grid(cells, out_path, cols=3, label_font_path=label_font_path)
        
        # 2. digits_symbol_qa_grid.png (12 prioritized samples)
        def compute_priority_score(text):
            score = 0
            # Devanagari numerals: ०-९
            if any(c in "०१२३४५६७८९" for c in text):
                score += 10
            # Currency markers: "रू.", "रु."
            if "रू." in text or "रु." in text:
                score += 8
            # Dates: "वि. सं", "वि.सं.", "साल", "संवत्", "संबत्", "इसापूर्व", "शताब्दी"
            date_keywords = ["वि. सं", "वि.सं.", "साल", "संवत्", "संबत्", "इसापूर्व", "शताब्दी"]
            if any(kw in text for kw in date_keywords):
                score += 8
            # Punctuation/brackets: ? ! , ( ) [ ] { } " ; : - । ॥
            punct_chars = "?!,()[]{}\";:-।॥"
            if any(c in punct_chars for c in text):
                score += 4
            return score
            
        sorted_records = sorted(records, key=lambda r: (-compute_priority_score(r["text"]), r["line_id"]))
        cells = [
            {
                "image_path": r["image_path"],
                "label": f"{r['line_id']}, score={compute_priority_score(r['text'])}"
            }
            for r in sorted_records[:12]
        ]
        out_path = os.path.join(grids_dir, "digits_symbol_qa_grid.png")
        make_grid(cells, out_path, cols=3, label_font_path=label_font_path)
        
        # 3. digits_only_hard_examples_grid.png (12 hard-difficulty samples)
        hard_rows = [r for r in records if r.get("difficulty_bucket", "easy") == "hard"]
        if not hard_rows:
            hard_rows = records
        sample_size = min(12, len(hard_rows))
        sampled_hard = rand.sample(hard_rows, sample_size)
        cells = [
            {
                "image_path": r["image_path"],
                "label": f"{r['line_id']}, {r['degradation_profile'].split('_')[0]}"
            }
            for r in sampled_hard
        ]
        out_path = os.path.join(grids_dir, "digits_only_hard_examples_grid.png")
        make_grid(cells, out_path, cols=3, label_font_path=label_font_path)
        
        return
        
    # Group images
    by_split = {}
    by_degradation = {}
    by_font = {}
    by_difficulty = {}
    
    for row in records:
        split = row["split"]
        deg = row["degradation_profile"]
        font = row["font_name"]
        diff = row.get("difficulty_bucket", "easy")
        
        by_split.setdefault(split, []).append(row)
        by_degradation.setdefault(deg, []).append(row)
        by_font.setdefault(font, []).append(row)
        by_difficulty.setdefault(diff, []).append(row)
        
    # 1. Grid per Split
    for split, rows in by_split.items():
        sample_size = min(12, len(rows))
        sampled = rand.sample(rows, sample_size)
        cells = [
            {
                "image_path": r["image_path"],
                "label": f"{r['line_id']}, {r['font_name']}, {r['degradation_profile'].split('_')[0]}"
            }
            for r in sampled
        ]
        out_path = os.path.join(grids_dir, f"split_{split}_grid.png")
        make_grid(cells, out_path, cols=3, label_font_path=label_font_path)
        
    # 2. Grid per Degradation Profile
    for deg, rows in by_degradation.items():
        sample_size = min(9, len(rows))
        sampled = rand.sample(rows, sample_size)
        cells = [
            {
                "image_path": r["image_path"],
                "label": f"{r['line_id']}, {r['font_name']}"
            }
            for r in sampled
        ]
        out_path = os.path.join(grids_dir, f"degradation_{deg}_grid.png")
        make_grid(cells, out_path, cols=3, label_font_path=label_font_path)
        
    # 3. Difficulty Grids
    for diff in ["easy", "medium", "hard"]:
        rows = by_difficulty.get(diff, [])
        if rows:
            sample_size = min(12, len(rows))
            sampled = rand.sample(rows, sample_size)
            cells = [
                {
                    "image_path": r["image_path"],
                    "label": f"{r['line_id']}, {r['degradation_profile'].split('_')[0]}"
                }
                for r in sampled
            ]
            out_path = os.path.join(grids_dir, f"difficulty_{diff}_grid.png")
            make_grid(cells, out_path, cols=3, label_font_path=label_font_path)
            
    # 4. Composed Profile Grid
    composed_rows = [r for r in records if r.get("degradation_family") == "composed"]
    if composed_rows:
        sample_size = min(12, len(composed_rows))
        sampled = rand.sample(composed_rows, sample_size)
        cells = [
            {
                "image_path": r["image_path"],
                "label": f"{r['line_id']}, {r['font_name']}"
            }
            for r in sampled
        ]
        out_path = os.path.join(grids_dir, "composed_profile_grid.png")
        make_grid(cells, out_path, cols=3, label_font_path=label_font_path)
        
    # 5. Manuscript Hard Examples Grid
    hard_rows = by_difficulty.get("hard", [])
    if hard_rows:
        sample_size = min(12, len(hard_rows))
        sampled = rand.sample(hard_rows, sample_size)
        cells = [
            {
                "image_path": r["image_path"],
                "label": f"{r['line_id']}, {r['degradation_profile'].split('_')[0]}"
            }
            for r in sampled
        ]
        out_path = os.path.join(grids_dir, "manuscript_hard_examples_grid.png")
        make_grid(cells, out_path, cols=3, label_font_path=label_font_path)
        
    # 6. Readability Borderline Grid
    borderline_rows = []
    for r in records:
        score = float(r.get("readability_score_estimate", 100.0))
        is_unreadable = r.get("unreadable_risk", "false").lower() == "true"
        if (35.0 <= score <= 68.0) or is_unreadable:
            borderline_rows.append(r)
            
    if borderline_rows:
        sample_size = min(12, len(borderline_rows))
        sampled = rand.sample(borderline_rows, sample_size)
        cells = [
            {
                "image_path": r["image_path"],
                "label": f"{r['line_id']}, score={r.get('readability_score_estimate')}"
            }
            for r in sampled
        ]
        out_path = os.path.join(grids_dir, "readability_borderline_grid.png")
        make_grid(cells, out_path, cols=3, label_font_path=label_font_path)
        
    # 7. Mixed Random QA Grid (15 samples)
    mixed_sample_size = min(15, len(records))
    sampled_mixed = rand.sample(records, mixed_sample_size)
    cells_mixed = [
        {
            "image_path": r["image_path"],
            "label": f"{r['line_id']}, {r['font_name']}, {r['degradation_profile'].split('_')[0]}"
        }
        for r in sampled_mixed
    ]
    mixed_out_path = os.path.join(grids_dir, "mixed_qa_grid.png")
    make_grid(cells_mixed, mixed_out_path, cols=3, label_font_path=label_font_path)
    
    # 8. Grid per Font
    for font, rows in by_font.items():
        if len(rows) >= 3:
            sample_size = min(9, len(rows))
            sampled = rand.sample(rows, sample_size)
            cells = [
                {
                    "image_path": r["image_path"],
                    "label": f"{r['line_id']}, {r['degradation_profile'].split('_')[0]}"
                }
                for r in sampled
            ]
            slug = font.replace(" ", "").lower()
            out_path = os.path.join(grids_dir, f"font_{slug}_grid.png")
            make_grid(cells, out_path, cols=3, label_font_path=label_font_path)

if __name__ == "__main__":
    main()
