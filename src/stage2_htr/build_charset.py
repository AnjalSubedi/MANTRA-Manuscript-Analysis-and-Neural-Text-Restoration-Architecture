# -*- coding: utf-8 -*-
"""
MANTRA Stage 2.2: Charset & Vocabulary Builder for HTR.
Constructs character vocabulary from the single source of truth manifest file,
performs QA checks, and generates txt/json vocab files and reports.
"""
import os
import sys
import csv
import json
import argparse
import re
import unicodedata
from collections import Counter

def get_char_sort_key(char):
    if char == " ":
        return (0, 0)
    
    # Devanagari digits (०-९)
    if '\u0966' <= char <= '\u096f':
        return (1, ord(char))
        
    # Devanagari punctuation (। and ॥)
    if char in ('\u0964', '\u0965'):
        return (2, ord(char))
        
    # Common punctuation (standard ASCII punctuation or category starts with 'P')
    # Exclude Devanagari punctuation from this bucket
    is_punc = (unicodedata.category(char).startswith('P') or char in '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
    if is_punc:
        return (3, ord(char))
        
    # Devanagari block (U+0900 to U+097F)
    if '\u0900' <= char <= '\u097f':
        return (4, ord(char))
        
    # Remaining characters
    return (5, ord(char))

def encode_text(text, char_to_id):
    """
    Encodes only real label characters into a list of IDs.
    Never encodes <BLANK> token.
    Raises ValueError if any character is missing in the mapping.
    """
    ids = []
    for char in text:
        if char == "<BLANK>":
            raise ValueError("encode_text should never encode the <BLANK> token.")
        if char not in char_to_id:
            raise ValueError(f"Character {repr(char)} (Unicode: U+{ord(char):04X}) not in charset.")
        ids.append(char_to_id[char])
    return ids

def decode_ids(ids, id_to_char, remove_blank=True):
    """
    Decodes a list of IDs back to text.
    Removes only <BLANK> token (id 0) during CTC decoding if remove_blank is True.
    """
    chars = []
    for idx in ids:
        char = id_to_char.get(idx) or id_to_char.get(str(idx))
        if char is None:
            raise ValueError(f"ID {idx} not found in id_to_char.")
        
        if char == "<BLANK>":
            if not remove_blank:
                chars.append(char)
        else:
            chars.append(char)
            
    return "".join(chars)

def main():
    parser = argparse.ArgumentParser(description="MANTRA Stage 2.2 Charset & Vocab Builder")
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/stage2_htr/manifests/htr_all.csv",
        help="Path to the unified HTR manifest file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/stage2_htr/vocab",
        help="Directory to save output vocabulary files"
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default="data/stage2_htr/reports",
        help="Directory to save QA report files"
    )
    
    args = parser.parse_args()
    
    # Enforce UTF-8 output on standard streams (especially on Windows)
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

    print("=" * 80)
    print("MANTRA STAGE 2.2: CHARSET & VOCABULARY GENERATION")
    print("=" * 80)
    
    if not os.path.exists(args.manifest):
        print(f"ERROR: Manifest file does not exist at '{args.manifest}'")
        sys.exit(1)
        
    print(f"Reading manifest: {args.manifest}")
    
    rows = []
    char_counts = Counter()
    blank_rows_count = 0
    ascii_digit_rows_count = 0
    latin_letter_rows_count = 0
    
    with open(args.manifest, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r_idx, row in enumerate(reader):
            text = row.get("text", "")
            rows.append(row)
            
            if not text:
                blank_rows_count += 1
                continue
                
            char_counts.update(text)
            
            # Check for ASCII digits [0-9]
            if any('0' <= c <= '9' for c in text):
                ascii_digit_rows_count += 1
                
            # Check for Latin letters [a-zA-Z]
            if any(('a' <= c <= 'z') or ('A' <= c <= 'Z') for c in text):
                latin_letter_rows_count += 1

    total_manifest_rows = len(rows)
    print(f"Total manifest rows read: {total_manifest_rows}")
    print(f"Blank text rows: {blank_rows_count}")
    print(f"ASCII digit rows: {ascii_digit_rows_count}")
    print(f"Latin letter rows: {latin_letter_rows_count}")
    
    # Real unique characters
    unique_real_chars = list(char_counts.keys())
    print(f"Total unique real characters found: {len(unique_real_chars)}")
    
    # Stable deterministic sorting
    sorted_real_chars = sorted(unique_real_chars, key=get_char_sort_key)
    
    # Special token mapping
    # <BLANK> id = 0, real chars start from id = 1
    char_to_id = {"<BLANK>": 0}
    id_to_char = {0: "<BLANK>"}
    
    for idx, char in enumerate(sorted_real_chars, start=1):
        char_to_id[char] = idx
        id_to_char[idx] = char
        
    num_characters = len(sorted_real_chars)
    num_classes = num_characters + 1
    
    print(f"Total model classes (including CTC blank): {num_classes}")
    
    # QA verification loop: encode and decode all rows
    mismatch_rows_count = 0
    unencodable_chars_count = 0
    
    for r_idx, row in enumerate(rows):
        text = row.get("text", "")
        if not text:
            continue
            
        try:
            encoded = encode_text(text, char_to_id)
            decoded = decode_ids(encoded, id_to_char)
            if decoded != text:
                mismatch_rows_count += 1
        except ValueError as e:
            unencodable_chars_count += 1
            print(f"Encoding Error on row {r_idx + 2} (line_id: {row.get('line_id')}): {e}")

    print(f"Encode-decode verified. Mismatches: {mismatch_rows_count}, Unencodable: {unencodable_chars_count}")
    
    # Character presence checking
    def check_presence(char_list):
        return all(c in char_to_id for c in char_list)
        
    devanagari_digits = [chr(c) for c in range(0x0966, 0x0970)] # ०-९
    danda = "\u0964"
    double_danda = "\u0965"
    halant = "\u094d"
    anusvara = "\u0902"
    visarga = "\u0903"
    chandrabindu = "\u0901"
    avagraha = "\u093d"
    vedic_udatta = "\u0951"
    vedic_anudatta = "\u0952"
    
    presence = {
        "devanagari_digits": check_presence(devanagari_digits),
        "danda": danda in char_to_id,
        "double_danda": double_danda in char_to_id,
        "halant": halant in char_to_id,
        "anusvara": anusvara in char_to_id,
        "visarga": visarga in char_to_id,
        "chandrabindu": chandrabindu in char_to_id,
        "avagraha": avagraha in char_to_id,
        "vedic_accent_udatta": vedic_udatta in char_to_id,
        "vedic_accent_anudatta": vedic_anudatta in char_to_id
    }
    
    # Frequency splits
    sorted_freqs = char_counts.most_common()
    top_50 = sorted_freqs[:50]
    bottom_50 = sorted_freqs[-50:] if len(sorted_freqs) >= 50 else sorted_freqs
    
    # Digit frequency table
    digit_freqs = {c: char_counts[c] for c in devanagari_digits}
    
    # Punctuation character categories
    punctuation_chars = []
    for char in sorted_real_chars:
        # Check if punctuation (Devanagari or common)
        is_punc = (char in (danda, double_danda) or 
                   unicodedata.category(char).startswith('P') or 
                   char in '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
        if is_punc:
            punctuation_chars.append(char)
            
    punc_freqs = {c: char_counts[c] for c in punctuation_chars}
    
    # Compile outputs
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.report_dir, exist_ok=True)
    
    vocab_txt_path = os.path.join(args.output_dir, "charset_synth100k.txt")
    vocab_json_path = os.path.join(args.output_dir, "charset_synth100k.json")
    report_md_path = os.path.join(args.report_dir, "charset_synth100k_report.md")
    report_json_path = os.path.join(args.report_dir, "charset_synth100k_report.json")
    
    # 1. Save charset_synth100k.txt (real characters one per line)
    # Open with encoding utf-8 or utf-8-sig
    with open(vocab_txt_path, "w", encoding="utf-8") as f:
        for char in sorted_real_chars:
            f.write(f"{char}\n")
    print(f"Saved txt vocabulary to: {vocab_txt_path}")
            
    # 2. Save charset_synth100k.json
    vocab_json_data = {
        "blank_token": "<BLANK>",
        "blank_id": 0,
        "characters": sorted_real_chars,
        "char_to_id": char_to_id,
        "id_to_char": {str(k): v for k, v in id_to_char.items()},
        "num_characters": num_characters,
        "num_classes": num_classes
    }
    with open(vocab_json_path, "w", encoding="utf-8") as f:
        json.dump(vocab_json_data, f, ensure_ascii=False, indent=2)
    print(f"Saved json vocabulary to: {vocab_json_path}")
        
    # Check overall QA status
    qa_passed = (
        blank_rows_count == 0 and
        unencodable_chars_count == 0 and
        mismatch_rows_count == 0 and
        ascii_digit_rows_count == 0 and
        latin_letter_rows_count == 0 and
        presence["devanagari_digits"] and
        presence["danda"] and
        presence["double_danda"] and
        presence["halant"]
    )
    
    qa_status_str = "PASS" if qa_passed else "FAIL"
    print(f"QA Overall Status: {qa_status_str}")
    
    # 3. Save charset_synth100k_report.json
    report_json_data = {
        "qa_status": qa_status_str,
        "total_manifest_rows": total_manifest_rows,
        "blank_text_rows": blank_rows_count,
        "total_unique_real_characters": num_characters,
        "total_model_classes": num_classes,
        "unencodable_characters": unencodable_chars_count,
        "encode_decode_mismatch_rows": mismatch_rows_count,
        "ascii_digit_rows": ascii_digit_rows_count,
        "latin_letter_rows": latin_letter_rows_count,
        "required_presence": presence,
        "top_50_frequent": [{"char": c, "unicode": f"U+{ord(c):04X}", "count": count} for c, count in top_50],
        "bottom_50_rare": [{"char": c, "unicode": f"U+{ord(c):04X}", "count": count} for c, count in bottom_50],
        "digit_frequencies": {c: count for c, count in digit_freqs.items()},
        "punctuation_frequencies": {c: count for c, count in punc_freqs.items()}
    }
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report_json_data, f, ensure_ascii=False, indent=2)
    print(f"Saved json report to: {report_json_path}")
        
    # 4. Save charset_synth100k_report.md
    with open(report_md_path, "w", encoding="utf-8-sig") as f:
        f.write(f"# MANTRA HTR Dataset Vocabulary QA Report\n\n")
        
        f.write(f"## Executive Summary\n")
        f.write(f"- **QA Status:** {qa_status_str}\n")
        f.write(f"- **Total Manifest Rows:** {total_manifest_rows}\n")
        f.write(f"- **Unique Characters (Real):** {num_characters}\n")
        f.write(f"- **Model Output Classes:** {num_classes} (including `<BLANK>`)\n\n")
        
        f.write(f"## Directory Paths\n")
        f.write(f"- **Input Manifest:** `{args.manifest}`\n")
        f.write(f"- **Output Charset TXT:** `{vocab_txt_path}`\n")
        f.write(f"- **Output Charset JSON:** `{vocab_json_path}`\n\n")
        
        f.write(f"## QA Validation Status\n")
        f.write(f"| QA Check | Target Limit | Observed Value | Status |\n")
        f.write(f"| --- | --- | --- | --- |\n")
        f.write(f"| Blank Text Rows | 0 | {blank_rows_count} | {'PASS' if blank_rows_count == 0 else 'FAIL'} |\n")
        f.write(f"| Unencodable Characters | 0 | {unencodable_chars_count} | {'PASS' if unencodable_chars_count == 0 else 'FAIL'} |\n")
        f.write(f"| Encode-Decode Mismatches | 0 | {mismatch_rows_count} | {'PASS' if mismatch_rows_count == 0 else 'FAIL'} |\n")
        f.write(f"| ASCII Digit Rows | 0 | {ascii_digit_rows_count} | {'PASS' if ascii_digit_rows_count == 0 else 'FAIL'} |\n")
        f.write(f"| Latin Letter Rows | 0 | {latin_letter_rows_count} | {'PASS' if latin_letter_rows_count == 0 else 'FAIL'} |\n\n")
        
        f.write(f"## Required Character Presence\n")
        f.write(f"| Character Group | Expected | Found / Checked | Status |\n")
        f.write(f"| --- | --- | --- | --- |\n")
        f.write(f"| Devanagari Digits (०-९) | Present | {''.join(devanagari_digits)} | {'PASS' if presence['devanagari_digits'] else 'FAIL'} |\n")
        f.write(f"| Danda (।) | Present | । | {'PASS' if presence['danda'] else 'FAIL'} |\n")
        f.write(f"| Double Danda (॥) | Present | ॥ | {'PASS' if presence['double_danda'] else 'FAIL'} |\n")
        f.write(f"| Halant (्) | Present | ् | {'PASS' if presence['halant'] else 'FAIL'} |\n")
        f.write(f"| Anusvara (ं) | Present | ं | {'PASS' if presence['anusvara'] else 'FAIL'} |\n")
        f.write(f"| Visarga (ः) | Present | ः | {'PASS' if presence['visarga'] else 'FAIL'} |\n")
        f.write(f"| Chandrabindu (ँ) | Present | ँ | {'PASS' if presence['chandrabindu'] else 'FAIL'} |\n")
        f.write(f"| Avagraha (ऽ) | Present | ऽ | {'PASS' if presence['avagraha'] else 'FAIL'} |\n")
        f.write(f"| Vedic Accent Udatta (॑) | Preserved | ॑ | {'PASS' if presence['vedic_accent_udatta'] else 'FAIL'} |\n")
        f.write(f"| Vedic Accent Anudatta (॒) | Preserved | ॒ | {'PASS' if presence['vedic_accent_anudatta'] else 'FAIL'} |\n\n")
        
        f.write(f"## Digit Frequency Table\n")
        f.write(f"| Character | Unicode Code Point | Count |\n")
        f.write(f"| --- | --- | --- |\n")
        for digit in devanagari_digits:
            f.write(f"| {digit} | U+{ord(digit):04X} | {digit_freqs[digit]} |\n")
        f.write(f"\n")
        
        f.write(f"## Punctuation Frequency Table\n")
        f.write(f"| Character | Unicode Code Point | Name | Count |\n")
        f.write(f"| --- | --- | --- | --- |\n")
        for p_char in punctuation_chars:
            p_name = unicodedata.name(p_char, "UNKNOWN")
            f.write(f"| {p_char} | U+{ord(p_char):04X} | {p_name} | {punc_freqs[p_char]} |\n")
        f.write(f"\n")
        
        f.write(f"## Top 50 Most Frequent Characters\n")
        f.write(f"| Rank | Character | Unicode Code Point | Name | Count |\n")
        f.write(f"| --- | --- | --- | --- | --- |\n")
        for rank, (c, count) in enumerate(top_50, start=1):
            c_name = unicodedata.name(c, "UNKNOWN")
            f.write(f"| {rank} | {c} | U+{ord(c):04X} | {c_name} | {count} |\n")
        f.write(f"\n")
        
        f.write(f"## Bottom 50 Rarest Characters\n")
        f.write(f"| Rank | Character | Unicode Code Point | Name | Count |\n")
        f.write(f"| --- | --- | --- | --- | --- |\n")
        start_rank = max(1, len(sorted_freqs) - len(bottom_50) + 1)
        for idx, (c, count) in enumerate(bottom_50):
            c_name = unicodedata.name(c, "UNKNOWN")
            f.write(f"| {start_rank + idx} | {c} | U+{ord(c):04X} | {c_name} | {count} |\n")
        f.write(f"\n")
        
        f.write(f"## Sample Encode-Decode Verification\n")
        f.write(f"Example encoding mappings:\n")
        f.write(f"```python\n")
        # Write some sample mappings
        for c in sorted_real_chars[:10]:
            f.write(f"{repr(c)} -> ID {char_to_id[c]}\n")
        f.write(f"```\n\n")
        
        f.write(f"Verification check for first 3 rows in manifest:\n")
        for i in range(min(3, len(rows))):
            orig_txt = rows[i].get("text", "")
            enc = encode_text(orig_txt, char_to_id)
            dec = decode_ids(enc, id_to_char)
            f.write(f"- **Row {i+1}:**\n")
            f.write(f"  - Original: `{orig_txt}`\n")
            f.write(f"  - Encoded IDs: `{enc}`\n")
            f.write(f"  - Decoded Text: `{dec}`\n")
        f.write(f"\n")
        
        f.write(f"## Full Character Index Table\n")
        f.write(f"| Class ID | Character | Unicode Code Point | Name | Frequency |\n")
        f.write(f"| --- | --- | --- | --- | --- |\n")
        f.write(f"| 0 | `<BLANK>` | - | CTC BLANK TOKEN | -\n")
        for char in sorted_real_chars:
            c_id = char_to_id[char]
            c_name = unicodedata.name(char, "UNKNOWN")
            c_freq = char_counts[char]
            f.write(f"| {c_id} | {char} | U+{ord(char):04X} | {c_name} | {c_freq} |\n")
            
    print(f"Saved markdown report to: {report_md_path}")
    print("=" * 80)
    print("CHARSET AND VOCABULARY GENERATION TASK FINISHED")
    print("=" * 80)
    
    if not qa_passed:
        print("ERROR: QA checks failed! Check the output logs above.")
        sys.exit(1)
    else:
        print("SUCCESS: All QA checks passed successfully.")

if __name__ == "__main__":
    main()
