import os
import sys
import csv
import json
from collections import Counter

SOURCE_MAPPINGS = {
    "nepal_ko_itihas_peshal_dahal.txt": "historical_nepali_001",
    "devmala_vamshavali.txt": "historical_nepali_002",
    "bhasha_vamshavali_1.txt": "historical_nepali_003",
    "gorkha_vamshavali.txt": "historical_nepali_004",
    "nepali_sanskriti_vishayak_agralekhharu.txt": "historical_nepali_005"
}

def main():
    csv_path = "data/stage1_synthetic/labels/historical_nepali_lines.csv"
    cleaning_stats_path = "data/stage1_synthetic/reports/cleaning_stats.json"
    gen_stats_path = "data/stage1_synthetic/reports/label_generation_stats.json"
    skipped_qa_path = "data/stage1_synthetic/reports/skipped_lines_qa.json"
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return
        
    # Load stats
    cleaning_stats = {}
    if os.path.exists(cleaning_stats_path):
        with open(cleaning_stats_path, "r", encoding="utf-8") as f:
            cleaning_stats = json.load(f)
            
    gen_stats = {}
    if os.path.exists(gen_stats_path):
        with open(gen_stats_path, "r", encoding="utf-8") as f:
            gen_stats = json.load(f)
            
    skipped_examples = []
    if os.path.exists(skipped_qa_path):
        with open(skipped_qa_path, "r", encoding="utf-8") as f:
            skipped_examples = json.load(f)
            
    # Compile label counts
    total_lines = 0
    line_count_by_source = {}
    line_count_by_split = {}
    total_len = 0
    min_len = float('inf')
    max_len = 0
    danda_count = 0
    double_danda_count = 0
    digit_count = 0
    conjunct_count = 0
    
    char_counter = Counter()
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_lines += 1
            
            source_id = row["source_id"]
            line_count_by_source[source_id] = line_count_by_source.get(source_id, 0) + 1
            
            split = row["split"]
            line_count_by_split[split] = line_count_by_split.get(split, 0) + 1
            
            length = int(row["line_length"])
            total_len += length
            if length < min_len:
                min_len = length
            if length > max_len:
                max_len = length
                
            if row["has_danda"] == "true":
                danda_count += 1
            if row["has_double_danda"] == "true":
                double_danda_count += 1
            if row["has_digit"] == "true":
                digit_count += 1
            if row["has_conjunct"] == "true":
                conjunct_count += 1
                
            for char in row["text"]:
                if '\u0900' <= char <= '\u097F':
                    char_counter[char] += 1
                    
    avg_len = total_len / total_lines if total_lines > 0 else 0
    if min_len == float('inf'):
        min_len = 0
        
    top_50_chars = [{"char": char, "freq": freq} for char, freq in char_counter.most_common(50)]
    
    # Select 20 example skipped lines for QA display
    qa_list = []
    seen_examples = set()
    for ex in skipped_examples:
        txt = ex["text"].strip()
        if txt and txt not in seen_examples and len(qa_list) < 20:
            qa_list.append(ex)
            seen_examples.add(txt)
            
    # Build complete JSON report
    report = {
        "summary": {
            "total_raw_lines": sum(stats.get("raw_line_count", 0) for stats in cleaning_stats.values()),
            "total_cleaned_txt_lines": sum(stats.get("cleaned_line_count", 0) for stats in cleaning_stats.values()),
            "total_final_label_csv_lines": total_lines,
            "average_line_length": avg_len,
            "min_line_length": min_len,
            "max_line_length": max_len,
            "danda_count": danda_count,
            "double_danda_count": double_danda_count,
            "digit_count": digit_count,
            "conjunct_count": conjunct_count
        },
        "skipped_categories": {
            "skipped_short_line_count": gen_stats.get("skipped_short_line_count", 0),
            "skipped_long_line_count": gen_stats.get("skipped_long_line_count", 0),
            "skipped_ocr_junk_count": gen_stats.get("skipped_ocr_junk_count", 0),
            "skipped_low_devanagari_ratio_count": gen_stats.get("skipped_low_devanagari_ratio_count", 0),
            "skipped_symbol_heavy_count": gen_stats.get("skipped_symbol_heavy_count", 0),
            "skipped_frontmatter_or_reference_count": gen_stats.get("skipped_frontmatter_or_reference_count", 0),
            "skipped_ascii_digit_count": gen_stats.get("skipped_ascii_digit_count", 0),
            "skipped_noisy_header_count": gen_stats.get("skipped_noisy_header_count", 0),
            "skipped_fragment_like_count": gen_stats.get("skipped_fragment_like_count", 0),
            "skipped_bad_symbol_count": gen_stats.get("skipped_bad_symbol_count", 0),
            "skipped_bracket_quote_count": gen_stats.get("skipped_bracket_quote_count", 0),
            "global_duplicate_removed_count": gen_stats.get("global_duplicate_removed_count", 0)
        },
        "by_source": {
            file_name: {
                "source_id": SOURCE_MAPPINGS.get(file_name),
                "raw_lines": stats.get("raw_line_count"),
                "cleaned_txt_lines": stats.get("cleaned_line_count"),
                "final_label_csv_lines": line_count_by_source.get(SOURCE_MAPPINGS.get(file_name), 0),
                "skipped_short": gen_stats.get("skipped_short_by_source", {}).get(SOURCE_MAPPINGS.get(file_name), 0),
                "skipped_long": gen_stats.get("skipped_long_by_source", {}).get(SOURCE_MAPPINGS.get(file_name), 0),
                "skipped_ocr": gen_stats.get("skipped_ocr_by_source", {}).get(SOURCE_MAPPINGS.get(file_name), 0),
                "skipped_low_ratio": gen_stats.get("skipped_low_ratio_by_source", {}).get(SOURCE_MAPPINGS.get(file_name), 0),
                "skipped_symbol_heavy": gen_stats.get("skipped_symbol_heavy_by_source", {}).get(SOURCE_MAPPINGS.get(file_name), 0),
                "skipped_frontmatter": gen_stats.get("skipped_frontmatter_by_source", {}).get(SOURCE_MAPPINGS.get(file_name), 0),
                "skipped_ascii_digit": gen_stats.get("skipped_ascii_digit_by_source", {}).get(SOURCE_MAPPINGS.get(file_name), 0),
                "skipped_noisy_header": gen_stats.get("skipped_noisy_header_by_source", {}).get(SOURCE_MAPPINGS.get(file_name), 0),
                "skipped_fragment_like": gen_stats.get("skipped_fragment_like_by_source", {}).get(SOURCE_MAPPINGS.get(file_name), 0),
                "skipped_bad_symbol": gen_stats.get("skipped_bad_symbol_by_source", {}).get(SOURCE_MAPPINGS.get(file_name), 0),
                "skipped_bracket_quote": gen_stats.get("skipped_bracket_quote_by_source", {}).get(SOURCE_MAPPINGS.get(file_name), 0)
            }
            for file_name, stats in cleaning_stats.items()
        },
        "by_split": line_count_by_split,
        "top_50_devanagari_characters": top_50_chars,
        "qa_skipped_examples_top20": qa_list
    }
    
    # Save phase3_cleaning_report.json
    report_json_path = "data/stage1_synthetic/reports/phase3_cleaning_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
    print(f"Saved JSON report to {report_json_path}")
    
    # Generate phase3_cleaning_summary.md
    summary_md_path = "data/stage1_synthetic/reports/phase3_cleaning_summary.md"
    
    md_content = f"""# MANTRA Phase 3 Text Cleaning & Label Generation Summary

This report provides the detailed metrics for the targeted text cleaning and final CSV label generation.

## Overall Dataset Volume
- **Total Raw Lines (Input):** {report['summary']['total_raw_lines']:,}
- **Total Cleaned lines (in .clean.txt):** {report['summary']['total_cleaned_txt_lines']:,}
- **Final Labeled Lines (in CSV):** {total_lines:,}

## Skip / Rejection Categories
- **Skipped Short Lines (< 5 chars):** {report['skipped_categories']['skipped_short_line_count']:,}
- **Skipped Long Lines (> 160 chars):** {report['skipped_categories']['skipped_long_line_count']:,}
- **Skipped OCR Junk Lines (ASCII/Scanner artifacts/Code fragments):** {report['skipped_categories']['skipped_ocr_junk_count']:,}
- **Skipped Low Devanagari Ratio (< 0.50):** {report['skipped_categories']['skipped_low_devanagari_ratio_count']:,}
- **Skipped Symbol-Heavy Lines (> 0.15):** {report['skipped_categories']['skipped_symbol_heavy_count']:,}
- **Skipped Frontmatter/Reference Lines:** {report['skipped_categories']['skipped_frontmatter_or_reference_count']:,}
- **Skipped Lines with ASCII Digits:** {report['skipped_categories']['skipped_ascii_digit_count']:,}
- **Skipped Noisy Header Lines:** {report['skipped_categories']['skipped_noisy_header_count']:,}
- **Skipped Fragment-Like Lines:** {report['skipped_categories']['skipped_fragment_like_count']:,}
- **Skipped Lines with OCR-Symbol Artifacts:** {report['skipped_categories']['skipped_bad_symbol_count']:,}
- **Skipped Lines with Brackets/Guillemets:** {report['skipped_categories']['skipped_bracket_quote_count']:,}
- **Removed Duplicate Lines (Global Deduplication):** {report['skipped_categories']['global_duplicate_removed_count']:,}

## Source breakdown
| Source ID | File Name | Raw Lines | Cleaned .TXT Lines | Final Label CSV Lines | Skipped Short | Skipped Long | Skipped OCR | Skipped Low Ratio | Skipped Symbol Heavy | Skipped Ref/Front | Skipped ASCII Digit | Skipped Noisy Header | Skipped Frag-Like | Skipped Bad Symbol | Skipped Bracket/Quote |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
"""
    for file_name, s_data in report["by_source"].items():
        md_content += f"| {s_data['source_id']} | {file_name} | {s_data['raw_lines']:,} | {s_data['cleaned_txt_lines']:,} | {s_data['final_label_csv_lines']:,} | {s_data['skipped_short']:,} | {s_data['skipped_long']:,} | {s_data['skipped_ocr']:,} | {s_data['skipped_low_ratio']:,} | {s_data['skipped_symbol_heavy']:,} | {s_data['skipped_frontmatter']:,} | {s_data['skipped_ascii_digit']:,} | {s_data['skipped_noisy_header']:,} | {s_data['skipped_fragment_like']:,} | {s_data['skipped_bad_symbol']:,} | {s_data['skipped_bracket_quote']:,} |\n"
        
    md_content += f"""
## Split Distribution
| Split | Line Count | Percentage |
|---|---|---|
"""
    for split, count in line_count_by_split.items():
        pct = (count / total_lines) * 100 if total_lines > 0 else 0
        md_content += f"| {split} | {count:,} | {pct:.2f}% |\n"
        
    md_content += f"""
## Feature Constraints & Metadata Counts
- **Lines with Single Danda (`।`):** {danda_count:,} ({danda_count/total_lines*100:.2f}% of total)
- **Lines with Double Danda (`॥`):** {double_danda_count:,} ({double_danda_count/total_lines*100:.2f}% of total)
- **Lines with Digits (Devanagari/Latin):** {digit_count:,} ({digit_count/total_lines*100:.2f}% of total)
- **Lines with Conjuncts (Halant or complex):** {conjunct_count:,} ({conjunct_count/total_lines*100:.2f}% of total)

## 20 Example Skipped Lines for Manual QA
The following lines from the cleaned text files were filtered out during final CSV label generation:

| File Name | Skipped Text Sample | Skip Reason |
|---|---|---|
"""
    for ex in qa_list:
        clean_text = ex["text"].replace("|", "\\|")
        md_content += f"| {ex['file']} | `{clean_text}` | `{ex['reason']}` |\n"
        
    md_content += f"""
## Top 50 Devanagari Characters by Frequency
| Rank | Character | Hex Code | Occurrence Count | Description |
|---|---|---|---|---|
"""
    char_names = {
        '\u0964': "Danda (।)",
        '\u0965': "Double Danda (॥)",
        '\u0905': "Vowel A (अ)",
        '\u0906': "Vowel AA (आ)",
        '\u0907': "Vowel I (इ)",
        '\u0908': "Vowel II (ई)",
        '\u0909': "Vowel U (उ)",
        '\u090a': "Vowel UU (ऊ)",
        '\u090b': "Vowel Vocalic R (ऋ)",
        '\u090f': "Vowel E (ए)",
        '\u0910': "Vowel AI (ऐ)",
        '\u0913': "Vowel O (ओ)",
        '\u0914': "Vowel AU (औ)",
        '\u0915': "Consonant KA (क)",
        '\u0916': "Consonant KHA (ख)",
        '\u0917': "Consonant GA (ग)",
        '\u0918': "Consonant GHA (घ)",
        '\u0919': "Consonant NGA (ङ)",
        '\u091a': "Consonant CA (च)",
        '\u091b': "Consonant CHA (छ)",
        '\u091c': "Consonant JA (ज)",
        '\u091d': "Consonant JHA (झ)",
        '\u091e': "Consonant NYA (ञ)",
        '\u091f': "Consonant TTA (ट)",
        '\u0920': "Consonant TTHA (ठ)",
        '\u0921': "Consonant DDA (ड)",
        '\u0922': "Consonant DDHA (ढ)",
        '\u0923': "Consonant NNA (ण)",
        '\u0924': "Consonant TA (त)",
        '\u0925': "Consonant THA (थ)",
        '\u0926': "Consonant DA (द)",
        '\u0927': "Consonant DHA (ध)",
        '\u0928': "Consonant NA (न)",
        '\u092a': "Consonant PA (प)",
        '\u092b': "Consonant PHA (फ)",
        '\u092c': "Consonant BA (ब)",
        '\u092d': "Consonant BHA (भ)",
        '\u092e': "Consonant MA (म)",
        '\u092f': "Consonant YA (य)",
        '\u0930': "Consonant RA (र)",
        '\u0931': "Consonant RRA",
        '\u0932': "Consonant LA (ल)",
        '\u0933': "Consonant LLA",
        '\u0935': "Consonant VA (व)",
        '\u0936': "Consonant SHA (श)",
        '\u0937': "Consonant SSA (ष)",
        '\u0938': "Consonant SA (स)",
        '\u0939': "Consonant HA (ह)",
        '\u093e': "Vowel Sign AA (ा)",
        '\u093f': "Vowel Sign I (ि)",
        '\u0940': "Vowel Sign II (ी)",
        '\u0941': "Vowel Sign U (ु)",
        '\u0942': "Vowel Sign UU (ू)",
        '\u0943': "Vowel Sign Vocalic R (ृ)",
        '\u0947': "Vowel Sign E (े)",
        '\u0948': "Vowel Sign AI (ै)",
        '\u094b': "Vowel Sign O (ो)",
        '\u094c': "Vowel Sign AU (ौ)",
        '\u094d': "Sign Virama / Halant (्)",
        '\u0902': "Sign Anusvara (ं)",
        '\u0901': "Sign Candrabindu (ँ)",
        '\u0903': "Sign Visarga (ः)",
        '\u093d': "Sign Avagraha (ऽ)",
    }
    
    for rank, (char, freq) in enumerate(char_counter.most_common(50), 1):
        hex_code = f"U+{ord(char):04X}"
        desc = char_names.get(char, "Devanagari character")
        md_content += f"| {rank} | `{char}` | `{hex_code}` | {freq:,} | {desc} |\n"
        
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Saved markdown summary to {summary_md_path}")

if __name__ == "__main__":
    main()
