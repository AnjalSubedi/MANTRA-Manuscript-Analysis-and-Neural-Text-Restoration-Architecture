import os
import sys
import csv
import json
import re

# Ensure output directories exist
os.makedirs("data/stage1_synthetic/labels", exist_ok=True)
os.makedirs("data/stage1_synthetic/reports", exist_ok=True)

SOURCE_MAPPINGS = {
    "nepal_ko_itihas_peshal_dahal.clean.txt": "historical_nepali_001",
    "devmala_vamshavali.clean.txt": "historical_nepali_002",
    "bhasha_vamshavali_1.clean.txt": "historical_nepali_003",
    "gorkha_vamshavali.clean.txt": "historical_nepali_004",
    "nepali_sanskriti_vishayak_agralekhharu.clean.txt": "historical_nepali_005"
}

FALLBACK_SPLITS = {
    "historical_nepali_001": "train",
    "historical_nepali_002": "train",
    "historical_nepali_003": "val",
    "historical_nepali_004": "train",
    "historical_nepali_005": "test"
}

CONJUNCTS = ["क्ष", "त्र", "ज्ञ", "श्र", "द्ध", "द्व", "द्य", "ह्म", "स्त्र", "क्त्य", "प्र", "ब्र", "भ्र", "ग्र", "क्र"]
JUNK_PATTERNS = ["ककाामारााक", "उयिकजिम", "यिकडमिन", "सिनपितििक", "कछ ०", "०200", "2000 %", "कडककहकाट"]

def detect_conjunct(text):
    if '\u094d' in text:
        return True
    for conj in CONJUNCTS:
        if conj in text:
            return True
    return False

def is_devanagari_char(c):
    return '\u0900' <= c <= '\u097F'

def is_devanagari_letter(c):
    return '\u0900' <= c <= '\u097F' and not ('\u0964' <= c <= '\u096F')

def is_ocr_junk_label(line):
    # Pattern match
    for p in JUNK_PATTERNS:
        if p in line:
            return True, f"pattern_{p}"
            
    # Long ASCII digit sequences
    if re.search(r'[0-1]{4,}', line):
        return True, "consecutive_01"
    if re.search(r'[0-9]{5,}', line):
        return True, "consecutive_digits"
    
    # Repeating digit patterns
    if re.search(r'0000|010101|1111|००००', line):
        return True, "repeating_digits"
        
    total_len = len(line)
    if total_len == 0:
        return True, "empty"
        
    # Too many ASCII digits or Latin/OCR code fragments
    ascii_digit_count = sum(1 for c in line if '0' <= c <= '9')
    if ascii_digit_count > 1:
        return True, "too_many_ascii_digits"
        
    if re.search(r'[0-9][\u0900-\u097F]|[\u0900-\u097F][0-9]', line):
        return True, "ascii_digit_adjacent_to_devanagari"
        
    latin_count = sum(1 for c in line if 'a' <= c.lower() <= 'z')
    if latin_count > 0:
        return True, "latin_characters"
        
    # Long words without spaces
    if any(len(w) >= 40 for w in line.split()):
        return True, "long_word_garbage"
        
    return False, None

def is_symbol_heavy(line):
    total_len = len(line)
    if total_len == 0:
        return True
    symbols_count = sum(1 for c in line if c in ":.-=_*#@%/\\|()[]{}<>+`'\"$;?,!^&~€£¥“”‘’")
    # Using 0.15 threshold for symbol heavy
    return (symbols_count / total_len) > 0.15

def is_fragment_like(line):
    # Count Devanagari letters (excluding digits and dandas)
    devanagari_letters = sum(1 for c in line if is_devanagari_letter(c))
    if devanagari_letters < 4:
        return True, "too_few_letters"
        
    # If the line is short (length < 12) and consists mostly of digits/symbols/spaces
    total_len = len(line)
    if total_len > 0:
        non_letter_chars = sum(1 for c in line if not is_devanagari_letter(c))
        if total_len < 12 and (non_letter_chars / total_len) > 0.60:
            return True, "mostly_non_letters"
            
    return False, None

def is_ref_or_frontmatter_label(line):
    # Frontmatter or TOC keywords
    keywords = ["अशुद्ध", "शुद्ध अशुद्ध", "बिषय-सूची", "विषय-सूची", "सूची-पत्र", "सहायक ग्रन्थ"]
    for kw in keywords:
        if kw in line:
            return True
            
    # Page citation formats
    if re.search(r'(ऐजन|एैजन|एजन)\s*(पृ|पू)\.', line):
        return True
    if re.search(r'पृष्ठ\s*[०-९0-9]+', line):
        return True
        
    # Footnote indicators
    if re.match(r'^\s*[०-९0-9]+\s*\.\s*[a-zA-Z\u0900-\u097F]', line) and ("पृ" in line or "संस्थान" in line or "प्रकाश" in line or "ऐजन" in line):
        return True
        
    return False

def load_split_mapping():
    split_mapping = {}
    manifest_path = "data/stage1_synthetic/corpus_manifest.csv"
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    sid = row.get("source_id")
                    split = row.get("split_group")
                    if sid and split:
                        split_mapping[sid] = split.strip()
            print(f"Loaded split mapping from manifest: {split_mapping}")
        except Exception as e:
            print(f"Warning: Failed to load manifest splits: {e}")
    return split_mapping

def main():
    split_mapping = load_split_mapping()
    output_csv = "data/stage1_synthetic/labels/historical_nepali_lines.csv"
    
    # Global counters
    skipped_short_line_count = 0
    skipped_long_line_count = 0
    skipped_ocr_junk_count = 0
    skipped_low_devanagari_ratio_count = 0
    skipped_symbol_heavy_count = 0
    skipped_frontmatter_or_reference_count = 0
    skipped_ascii_digit_count = 0
    skipped_noisy_header_count = 0
    skipped_fragment_like_count = 0
    skipped_bad_symbol_count = 0
    skipped_bracket_quote_count = 0
    global_duplicate_removed_count = 0
    total_written = 0
    
    # Per-source metrics
    skipped_short_by_source = {sid: 0 for sid in SOURCE_MAPPINGS.values()}
    skipped_long_by_source = {sid: 0 for sid in SOURCE_MAPPINGS.values()}
    skipped_ocr_by_source = {sid: 0 for sid in SOURCE_MAPPINGS.values()}
    skipped_low_ratio_by_source = {sid: 0 for sid in SOURCE_MAPPINGS.values()}
    skipped_symbol_heavy_by_source = {sid: 0 for sid in SOURCE_MAPPINGS.values()}
    skipped_frontmatter_by_source = {sid: 0 for sid in SOURCE_MAPPINGS.values()}
    skipped_ascii_digit_by_source = {sid: 0 for sid in SOURCE_MAPPINGS.values()}
    skipped_noisy_header_by_source = {sid: 0 for sid in SOURCE_MAPPINGS.values()}
    skipped_fragment_like_by_source = {sid: 0 for sid in SOURCE_MAPPINGS.values()}
    skipped_bad_symbol_by_source = {sid: 0 for sid in SOURCE_MAPPINGS.values()}
    skipped_bracket_quote_by_source = {sid: 0 for sid in SOURCE_MAPPINGS.values()}
    written_by_source = {sid: 0 for sid in SOURCE_MAPPINGS.values()}
    
    skipped_examples_bucket = []
    seen_texts = set()
    
    fieldnames = [
        "line_id", "source_id", "source_file", "text", "line_length",
        "has_danda", "has_double_danda", "has_digit", "has_conjunct", "split"
    ]
    
    with open(output_csv, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for file_name, source_id in SOURCE_MAPPINGS.items():
            cleaned_path = os.path.join("data/stage1_synthetic/text/cleaned", file_name)
            if not os.path.exists(cleaned_path):
                print(f"Warning: Cleaned file not found {cleaned_path}")
                continue
                
            with open(cleaned_path, "r", encoding="utf-8") as f:
                lines = f.read().splitlines()
                
            line_idx = 1
            for line in lines:
                line_stripped = line.strip()
                line_len = len(line_stripped)
                
                # Check for length constraint
                if line_len < 5:
                    skipped_short_line_count += 1
                    skipped_short_by_source[source_id] += 1
                    skipped_examples_bucket.append({
                        "file": file_name,
                        "text": line_stripped,
                        "reason": "too_short"
                    })
                    continue
                    
                if line_len > 160:
                    skipped_long_line_count += 1
                    skipped_long_by_source[source_id] += 1
                    skipped_examples_bucket.append({
                        "file": file_name,
                        "text": line_stripped,
                        "reason": "too_long"
                    })
                    continue
                
                # Check for ASCII digits
                if re.search(r'[0-9]', line_stripped):
                    skipped_ascii_digit_count += 1
                    skipped_ascii_digit_by_source[source_id] += 1
                    skipped_examples_bucket.append({
                        "file": file_name,
                        "text": line_stripped,
                        "reason": "contains_ascii_digit"
                    })
                    continue

                # Check for bad OCR symbols
                if re.search(r'[_|&%€$£₹]', line_stripped):
                    skipped_bad_symbol_count += 1
                    skipped_bad_symbol_by_source[source_id] += 1
                    skipped_examples_bucket.append({
                        "file": file_name,
                        "text": line_stripped,
                        "reason": "bad_ocr_symbols"
                    })
                    continue

                # Check for brackets and guillemets (« » [ ])
                if re.search(r'[«»\[\]]', line_stripped):
                    skipped_bracket_quote_count += 1
                    skipped_bracket_quote_by_source[source_id] += 1
                    skipped_examples_bucket.append({
                        "file": file_name,
                        "text": line_stripped,
                        "reason": "brackets_or_guillemets"
                    })
                    continue

                # Check for noisy header string
                if "भाषाबंशावती" in line_stripped:
                    skipped_noisy_header_count += 1
                    skipped_noisy_header_by_source[source_id] += 1
                    skipped_examples_bucket.append({
                        "file": file_name,
                        "text": line_stripped,
                        "reason": "noisy_header"
                    })
                    continue

                # Check for fragment-like line
                is_frag, frag_reason = is_fragment_like(line_stripped)
                if is_frag:
                    skipped_fragment_like_count += 1
                    skipped_fragment_like_by_source[source_id] += 1
                    skipped_examples_bucket.append({
                        "file": file_name,
                        "text": line_stripped,
                        "reason": f"fragment_like_{frag_reason}"
                    })
                    continue
                
                # Devanagari ratio check
                devanagari_chars_count = sum(1 for c in line_stripped if is_devanagari_char(c))
                if line_len > 0 and (devanagari_chars_count / line_len) < 0.50:
                    skipped_low_devanagari_ratio_count += 1
                    skipped_low_ratio_by_source[source_id] += 1
                    skipped_examples_bucket.append({
                        "file": file_name,
                        "text": line_stripped,
                        "reason": "low_devanagari_ratio"
                    })
                    continue
                
                # Check for OCR junk
                is_junk, junk_reason = is_ocr_junk_label(line_stripped)
                if is_junk:
                    skipped_ocr_junk_count += 1
                    skipped_ocr_by_source[source_id] += 1
                    skipped_examples_bucket.append({
                        "file": file_name,
                        "text": line_stripped,
                        "reason": f"ocr_junk_{junk_reason}"
                    })
                    continue
                    
                # Symbol heavy check
                if is_symbol_heavy(line_stripped):
                    skipped_symbol_heavy_count += 1
                    skipped_symbol_heavy_by_source[source_id] += 1
                    skipped_examples_bucket.append({
                        "file": file_name,
                        "text": line_stripped,
                        "reason": "symbol_heavy"
                    })
                    continue
                    
                # Check for bibliography/references/frontmatter fragments
                if is_ref_or_frontmatter_label(line_stripped):
                    skipped_frontmatter_or_reference_count += 1
                    skipped_frontmatter_by_source[source_id] += 1
                    skipped_examples_bucket.append({
                        "file": file_name,
                        "text": line_stripped,
                        "reason": "reference_or_frontmatter_fragment"
                    })
                    continue
                
                # Global deduplication
                if line_stripped in seen_texts:
                    global_duplicate_removed_count += 1
                    continue
                seen_texts.add(line_stripped)
                    
                line_id = f"{source_id}_{line_idx:06d}"
                line_idx += 1
                
                has_danda = "true" if '\u0964' in line_stripped else "false"
                has_double_danda = "true" if '\u0965' in line_stripped else "false"
                has_digit = "true" if any(c.isdigit() or ('\u0966' <= c <= '\u096f') for c in line_stripped) else "false"
                has_conjunct = "true" if detect_conjunct(line_stripped) else "false"
                
                split = split_mapping.get(source_id, FALLBACK_SPLITS.get(source_id, "train"))
                
                writer.writerow({
                    "line_id": line_id,
                    "source_id": source_id,
                    "source_file": file_name,
                    "text": line_stripped,
                    "line_length": line_len,
                    "has_danda": has_danda,
                    "has_double_danda": has_double_danda,
                    "has_digit": has_digit,
                    "has_conjunct": has_conjunct,
                    "split": split
                })
                total_written += 1
                written_by_source[source_id] += 1
                
            print(f"Processed {file_name}: {written_by_source[source_id]} written")
            
    # Save statistics
    label_stats = {
        "skipped_short_line_count": skipped_short_line_count,
        "skipped_long_line_count": skipped_long_line_count,
        "skipped_ocr_junk_count": skipped_ocr_junk_count,
        "skipped_low_devanagari_ratio_count": skipped_low_devanagari_ratio_count,
        "skipped_symbol_heavy_count": skipped_symbol_heavy_count,
        "skipped_frontmatter_or_reference_count": skipped_frontmatter_or_reference_count,
        "skipped_ascii_digit_count": skipped_ascii_digit_count,
        "skipped_noisy_header_count": skipped_noisy_header_count,
        "skipped_fragment_like_count": skipped_fragment_like_count,
        "skipped_bad_symbol_count": skipped_bad_symbol_count,
        "skipped_bracket_quote_count": skipped_bracket_quote_count,
        "global_duplicate_removed_count": global_duplicate_removed_count,
        "total_lines_written_to_csv": total_written,
        "written_by_source": written_by_source,
        "skipped_short_by_source": skipped_short_by_source,
        "skipped_long_by_source": skipped_long_by_source,
        "skipped_ocr_by_source": skipped_ocr_by_source,
        "skipped_low_ratio_by_source": skipped_low_ratio_by_source,
        "skipped_symbol_heavy_by_source": skipped_symbol_heavy_by_source,
        "skipped_frontmatter_by_source": skipped_frontmatter_by_source,
        "skipped_ascii_digit_by_source": skipped_ascii_digit_by_source,
        "skipped_noisy_header_by_source": skipped_noisy_header_by_source,
        "skipped_fragment_like_by_source": skipped_fragment_like_by_source,
        "skipped_bad_symbol_by_source": skipped_bad_symbol_by_source,
        "skipped_bracket_quote_by_source": skipped_bracket_quote_by_source
    }
    
    stats_path = os.path.join("data/stage1_synthetic/reports", "label_generation_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(label_stats, f, indent=4)
        
    # Save skipped lines QA log
    skipped_qa_path = os.path.join("data/stage1_synthetic/reports", "skipped_lines_qa.json")
    with open(skipped_qa_path, "w", encoding="utf-8") as f:
        json.dump(skipped_examples_bucket, f, indent=4, ensure_ascii=False)
        
    print(f"Saved label generation stats to {stats_path}")
    print(f"Saved skipped QA samples to {skipped_qa_path}")

if __name__ == "__main__":
    main()
