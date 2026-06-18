import os
import sys
import re
import unicodedata
import json

# Ensure directories exist
os.makedirs("data/stage1_synthetic/text/cleaned", exist_ok=True)
os.makedirs("data/stage1_synthetic/labels", exist_ok=True)
os.makedirs("data/stage1_synthetic/reports", exist_ok=True)

TARGET_FILES = [
    "nepal_ko_itihas_peshal_dahal.txt",
    "devmala_vamshavali.txt",
    "bhasha_vamshavali_1.txt",
    "gorkha_vamshavali.txt",
    "nepali_sanskriti_vishayak_agralekhharu.txt"
]

# Hardcoded first pass line counts for reference/comparison
FIRST_PASS_COUNTS = {
    "nepal_ko_itihas_peshal_dahal.txt": 11338,
    "devmala_vamshavali.txt": 3560,
    "bhasha_vamshavali_1.txt": 2677,
    "gorkha_vamshavali.txt": 3887,
    "nepali_sanskriti_vishayak_agralekhharu.txt": 2162
}

def is_devanagari_char(c):
    return '\u0900' <= c <= '\u097F'

def is_devanagari_letter(c):
    return '\u0900' <= c <= '\u097F' and not ('\u0964' <= c <= '\u096F')

def is_ocr_junk(line_stripped):
    # 1. Long digit sequences (ASCII 0 and 1 sequences, e.g. 0000, 010100001000)
    if re.search(r'[0-1]{4,}', line_stripped):
        return True
        
    # 2. Long sequences of any ASCII digits (5 or more consecutive)
    if re.search(r'[0-9]{5,}', line_stripped):
        return True
        
    # 3. Mixed scanner-like numeric/symbol fragments
    total_len = len(line_stripped)
    if total_len == 0:
        return True
        
    ascii_digit_count = sum(1 for c in line_stripped if '0' <= c <= '9')
    devanagari_letter_count = sum(1 for c in line_stripped if is_devanagari_letter(c))
    devanagari_char_count = sum(1 for c in line_stripped if is_devanagari_char(c))
    
    # Short lines with high ASCII digit density and very few Devanagari letters (e.g. scanner codes, page numbers)
    if total_len < 25 and ascii_digit_count > 2 and devanagari_letter_count < 6:
        if (ascii_digit_count / total_len) > 0.15:
            return True
            
    # Mixed colons and numbers like :::::222::::
    if re.search(r'[:]{3,}[0-9]+|[:]{2,}[0-9]+[:]{2,}', line_stripped):
        return True
        
    # Lines with too many non-word OCR-like symbols
    symbols_count = sum(1 for c in line_stripped if c in ":.-=_*#@%/\\|")
    if symbols_count / total_len > 0.20:
        return True
        
    # Long word check (consecutive non-space chars without spaces)
    if any(len(w) >= 40 for w in line_stripped.split()):
        return True
        
    return False

def should_keep_line(line):
    line_stripped = line.strip()
    total_len = len(line_stripped)
    
    # 1. Empty lines or very short lines (shorter than 5 characters)
    if total_len < 5:
        return False, "too_short"
        
    # Count character types
    devanagari_char_count = sum(1 for c in line_stripped if is_devanagari_char(c))
    devanagari_letter_count = sum(1 for c in line_stripped if is_devanagari_letter(c))
    latin_count = sum(1 for c in line_stripped if ('a' <= c.lower() <= 'z'))
    
    # 2. No Devanagari characters
    if devanagari_char_count == 0:
        return False, "no_devanagari"
        
    # 3. Not enough actual Devanagari letters (vowels/consonants/matras)
    if devanagari_letter_count < 2:
        return False, "insufficient_devanagari_letters"
        
    # 4. Too much Latin text (more than 15% Latin chars)
    if latin_count / total_len > 0.15:
        return False, "too_much_latin"
        
    # 5. Mostly symbols/punctuation/digits (Devanagari characters should be at least 40% of the line)
    if devanagari_char_count / total_len < 0.40:
        return False, "insufficient_devanagari_ratio"
        
    # 6. OCR / repeated junk patterns
    if re.search(r'[^a-zA-Z0-9\u0900-\u097F\s\u0964\u0965]{3,}', line_stripped):
        return False, "consecutive_junk_symbols"
        
    # 7. Second-pass specific OCR junk check
    if is_ocr_junk(line_stripped):
        return False, "ocr_junk_filter"
        
    return True, None

def get_body_range(file_name, raw_lines):
    start_idx = 0
    end_idx = len(raw_lines)
    
    if file_name == "bhasha_vamshavali_1.txt":
        # Start from श्रीणणेशाय नमः or भाषावंशावली
        for idx, line in enumerate(raw_lines):
            if "श्रीणणेशाय नमः" in line or "श्रीगणेशाय नमः" in line:
                start_idx = idx
                # Keep the title line immediately before if it is "भाषावंशावली"
                for back_idx in range(max(0, idx - 5), idx):
                    if "भाषावंशावली" in raw_lines[back_idx]:
                        start_idx = back_idx
                        break
                break
                
    elif file_name == "nepali_sanskriti_vishayak_agralekhharu.txt":
        # Start from "डा. जगदीश चन्द्र रेग्मी"
        for idx, line in enumerate(raw_lines):
            if "डा. जगदीश चन्द्र रेग्मी" in line:
                start_idx = max(0, idx - 3)
                break
        # End at "धन्यवाद ज्ञापन गर्दछु"
        for idx in range(len(raw_lines) - 1, start_idx, -1):
            if "धन्यवाद ज्ञापन गर्दछु" in raw_lines[idx]:
                end_idx = idx + 1
                break
                
    elif file_name == "devmala_vamshavali.txt":
        # Start from "नमो गोरङ्गकालीम्याम्‌"
        for idx, line in enumerate(raw_lines):
            if "नमो गोरङ्गकालीम्याम्‌" in line:
                start_idx = idx
                break
        # End at "समाप्त भयो" near the end
        for idx in range(len(raw_lines) - 1, start_idx, -1):
            if "समाप्त भयो" in raw_lines[idx]:
                end_idx = idx + 1
                break
                
    elif file_name == "nepal_ko_itihas_peshal_dahal.txt":
        # End of TOC is marked by "सहायक ग्रन्थस्‌ची"
        toc_end_idx = -1
        for idx, line in enumerate(raw_lines):
            if "सहायक ग्रन्थस्‌ची" in line or "सहायक ग्रन्थसूची" in line:
                toc_end_idx = idx
                break
        if toc_end_idx != -1:
            for idx in range(toc_end_idx + 1, len(raw_lines)):
                if "नेपालको इतिहासका स्रोतहरू" in raw_lines[idx] or "नेपालको इतिहास" in raw_lines[idx]:
                    start_idx = idx
                    break
        else:
            for idx, line in enumerate(raw_lines):
                if "नेपालको इतिहासका स्रोतहरू" in line and idx > 100:
                    start_idx = idx
                    break
                    
        # Trim bibliography at the end of the file
        for idx in range(start_idx, len(raw_lines)):
            if "सहायक ग्रन्थ" in raw_lines[idx] or "सहायक ग्रन्थहरूको सूची" in raw_lines[idx] or "सहायक ग्रन्थहरूको स्‌ची" in raw_lines[idx]:
                if idx > start_idx + 1000:
                    end_idx = idx
                    break
                    
    return start_idx, end_idx

def clean_file(file_name, removed_examples_bucket):
    raw_path = os.path.join("data/stage1_synthetic/text/raw", file_name)
    cleaned_name = file_name.replace(".txt", ".clean.txt")
    cleaned_path = os.path.join("data/stage1_synthetic/text/cleaned", cleaned_name)
    
    if not os.path.exists(raw_path):
        print(f"Error: File not found {raw_path}")
        return None
        
    with open(raw_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
        
    normalized_content = unicodedata.normalize("NFC", content)
    raw_char_count = len(normalized_content)
    
    raw_lines = normalized_content.splitlines()
    raw_line_count = len(raw_lines)
    
    start_idx, end_idx = get_body_range(file_name, raw_lines)
    
    front_matter_skipped = start_idx
    back_matter_skipped = len(raw_lines) - end_idx
    total_skipped_sections = front_matter_skipped + back_matter_skipped
    
    body_lines = raw_lines[start_idx:end_idx]
    
    cleaned_lines = []
    seen_lines = set()
    duplicate_removed_count = 0
    ocr_junk_removed_count = 0
    removed_line_count = 0
    
    for line in body_lines:
        line_normalized = " ".join(line.split())
        
        if not line_normalized:
            removed_line_count += 1
            ocr_junk_removed_count += 1
            continue
            
        keep, reason = should_keep_line(line_normalized)
        if not keep:
            removed_line_count += 1
            ocr_junk_removed_count += 1
            removed_examples_bucket.append({
                "file": file_name,
                "text": line_normalized,
                "reason": reason
            })
            continue
            
        if line_normalized in seen_lines:
            duplicate_removed_count += 1
            removed_line_count += 1
            continue
            
        seen_lines.add(line_normalized)
        cleaned_lines.append(line_normalized)
        
    cleaned_line_count = len(cleaned_lines)
    cleaned_content = "\n".join(cleaned_lines)
    
    with open(cleaned_path, "w", encoding="utf-8") as f:
        f.write(cleaned_content)
        
    cleaned_character_count = len(cleaned_content)
    devanagari_character_count = sum(1 for c in cleaned_content if is_devanagari_char(c))
    
    stats = {
        "file_name": file_name,
        "raw_line_count": raw_line_count,
        "first_pass_cleaned_count": FIRST_PASS_COUNTS[file_name],
        "cleaned_line_count": cleaned_line_count,
        "removed_line_count": removed_line_count,
        "raw_character_count": raw_char_count,
        "cleaned_character_count": cleaned_character_count,
        "devanagari_character_count": devanagari_character_count,
        "duplicate_removed_count": duplicate_removed_count,
        "front_matter_skipped_count": front_matter_skipped,
        "back_matter_skipped_count": back_matter_skipped,
        "total_skipped_sections_count": total_skipped_sections,
        "ocr_junk_removed_count": ocr_junk_removed_count
    }
    
    print(f"Cleaned {file_name}:")
    print(f"  First Pass Cleaned: {FIRST_PASS_COUNTS[file_name]} -> Second Pass Cleaned: {cleaned_line_count}")
    print(f"  Skipped Section Lines (TOC/Front/Bibliography): {total_skipped_sections}")
    print(f"  OCR Junk/Filtered Lines: {ocr_junk_removed_count} | Duplicates Removed: {duplicate_removed_count}")
    
    return stats

def main():
    all_stats = {}
    removed_examples_bucket = []
    for file_name in TARGET_FILES:
        stats = clean_file(file_name, removed_examples_bucket)
        if stats:
            all_stats[file_name] = stats
            
    stats_path = os.path.join("data/stage1_synthetic/reports", "cleaning_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=4)
        
    qa_path = os.path.join("data/stage1_synthetic/reports", "removed_lines_qa.json")
    with open(qa_path, "w", encoding="utf-8") as f:
        json.dump(removed_examples_bucket, f, indent=4, ensure_ascii=False)
        
    print(f"Saved cleaning stats to {stats_path}")
    print(f"Saved QA removed lines samples to {qa_path}")

if __name__ == "__main__":
    main()
