import os
import csv
import re
import json
import random
import shutil
import sys
import collections

# Path setup
LABEL_DIR = "data/stage1_synthetic/labels"
REPORT_DIR = "data/stage1_synthetic/reports"

os.makedirs(LABEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# Suffix clean-up helper
def standardize_filenames():
    literary_suffixed = os.path.join(LABEL_DIR, "nepali_literary_lines (1).csv")
    literary_clean = os.path.join(LABEL_DIR, "nepali_literary_lines.csv")
    if os.path.exists(literary_suffixed):
        print(f"Standardizing: Copying {literary_suffixed} to {literary_clean}")
        shutil.copy2(literary_suffixed, literary_clean)
        
    sanskrit_suffixed = os.path.join(LABEL_DIR, "sanskrit_lines (2).csv")
    sanskrit_clean = os.path.join(LABEL_DIR, "sanskrit_lines.csv")
    if os.path.exists(sanskrit_suffixed):
        print(f"Standardizing: Copying {sanskrit_suffixed} to {sanskrit_clean}")
        shutil.copy2(sanskrit_suffixed, sanskrit_clean)

# Load CSV
def load_csv(filepath):
    rows = []
    with open(filepath, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

# Proportional sampling using Largest Remainder Method
def sample_sanskrit_proportionally(sanskrit_rows, target_total, seed=42):
    # Group rows by split and then by source_id
    split_groups = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in sanskrit_rows:
        split_groups[r['split']][r['source_id']].append(r)
        
    split_targets = {
        'train': 6000,
        'val': 750,
        'test': 750
    }
    
    sampled_sanskrit = []
    sampling_summary = {}
    
    for split, target in split_targets.items():
        groups = split_groups.get(split, {})
        if not groups:
            continue
            
        total_available = sum(len(g) for g in groups.values())
        print(f"Sanskrit split '{split}': available={total_available}, target={target}")
        
        # Calculate fractional targets
        fractional_targets = {}
        for s_id, rows in groups.items():
            fractional_targets[s_id] = (len(rows) / total_available) * target
            
        # Integer parts
        integer_targets = {s_id: int(f) for s_id, f in fractional_targets.items()}
        remainders = {s_id: f - integer_targets[s_id] for s_id, f in fractional_targets.items()}
        
        allocated = sum(integer_targets.values())
        unallocated = target - allocated
        
        # Distribute unallocated samples using largest remainders, tie-broken by source_id name
        sorted_rem = sorted(remainders.items(), key=lambda x: (-x[1], x[0]))
        for i in range(unallocated):
            s_id = sorted_rem[i][0]
            integer_targets[s_id] += 1
            
        # Deterministically sample from each group
        rand = random.Random(seed)
        for s_id, count in sorted(integer_targets.items()):
            # Sort rows by line_id for absolute deterministic ordering before sampling
            sorted_rows = sorted(groups[s_id], key=lambda x: x['line_id'])
            sampled = rand.sample(sorted_rows, count)
            sampled_sanskrit.extend(sampled)
            sampling_summary[f"{split}_{s_id}"] = {
                "available": len(groups[s_id]),
                "target_allocated": count
            }
            
    return sampled_sanskrit, sampling_summary

# Metadata recomputation logic
CONJUNCTS = ["क्ष", "त्र", "ज्ञ", "श्र", "द्ध", "द्व", "द्य", "ह्म", "स्त्र", "क्त्य", "प्र", "ब्र", "भ्र", "ग्र", "क्र"]
def recompute_metadata(text):
    line_len = len(text)
    has_danda = "true" if '।' in text else "false"
    has_double_danda = "true" if '॥' in text else "false"
    has_digit = "true" if any(c.isdigit() or ('\u0966' <= c <= '\u096f') for c in text) else "false"
    
    # Conjunct checking matches generate_line_labels
    has_conjunct = "false"
    if '\u094d' in text:
        has_conjunct = "true"
    else:
        for conj in CONJUNCTS:
            if conj in text:
                has_conjunct = "true"
                break
                
    return line_len, has_danda, has_double_danda, has_digit, has_conjunct

# Refined question mark classification rules
B_START = r"(?:^|\s|[।॥,\.;:\?\!\'\"\“\”\‘\’\(\)\[\]])"
B_END = r"(?:\s|$|[।॥,\.;:\?\!\'\"\“\”\‘\’\(\)\[\]])"

QUESTION_PATTERNS = [
    r"के(?!ही)(?:को|ले|मा|बाट|विना|का|भो|के|-के|-केमा)?",
    r"(?:कहाँ|काहाँ)(?:बाट|सम्म|को|नेर)?",
    r"कसर[ीि]",
    r"किन",
    r"कहिले(?:काहीँ|देखि)?",
    r"कति(?:-कति)?",
    r"कस्त[ोाी](?:लाई|लाइ|को|ले|सँग|सङ्ग)?",
    r"कुन्?(?!ै)(?:-कुन|कुन|बेला|चीज|कुरा)?",
    r"कता(?:बाट|तिर)?",
    r"को(?:-को)?\s+(?:हौ|हुन्|हन्|छ|थिइन्|थियो|हुन्‌|हन्‌|हुनुहुन्छ|हो|छन्|छौ)",
    r"कस(?:को|लाई|ले|बाट|सँग|सङ्ग|का|की)?",
    r"कस्(?:को|लाई|ले|का|की|लाइ|ले)?",
    r"कसो(?:री)?",
    # Sanskrit question words
    r"किम्",
    r"किं",
    r"कथम्",
    r"कथं",
    r"कुत्र",
    r"कदा",
    r"कुतः",
    r"कियत्",
    r"कः"
]

COMPILED_PATTERNS = [re.compile(B_START + pat + B_END) for pat in QUESTION_PATTERNS]

ENDING_VERBS = r"(?:छ|हो|छैन|होइन|हैन|हुन्|छन्|छौ|हुन्छ|हुनुहुन्छ|थियो|थिइन्|भयो|सकिन्छ|सक्छ|गर्छ|भनुँ|भनूँ|पाउन्न थ्यो|सक्दछ र)"
ENDING_VERB_PAT = re.compile(B_START + ENDING_VERBS + r"\s*\?(?:\s*|[।॥,\.;:\?\!\'\"\“\”\‘\’\(\)\[\]])*$")
ENDING_PARTICLE_PAT = re.compile(B_START + r"(?:र|त|पो)\s*\?(?:\s*|[।॥,\.;:\?\!\'\"\“\”\‘\’\(\)\[\]])*$")
ENDING_KI_ATTACHED = r"(?:हो|छ|छैन|होइन|हैन|हुन्न|हुँदैन|भयो|सकिएला|होला|थियो|थी|सकिन्छ|सक्छ)की?"
ENDING_KI_PAT = re.compile(r"(?:" + B_START + r"की?|" + ENDING_KI_ATTACHED + r")\s*\?(?:\s*|[।॥,\.;:\?\!\'\"\“\”\‘\’\(\)\[\]])*$")

def check_keep_question_mark_row(text):
    # Rule B checks:
    # 1. Multiple question marks
    if text.count('?') > 1:
        return False, "multiple_question_marks"
        
    # 2. '?' appears beside quotes, digits, or punctuation
    stripped_text = re.sub(r'\s+', '', text)
    if re.search(r'[0-9०-९\'"“”‘’`«»\[\]\(\)\{\};:,\.\-_|&%€$£₹!\/\\]\?', stripped_text) or \
       re.search(r'\?[0-9०-९\'"“”‘’`«»\[\]\(\)\{\};:,\.\-_|&%€$£₹!\/\\]', stripped_text):
        return False, "adjacent_junk_or_digit_or_quote"
        
    # 3. Ends with unclear OCR fragments
    match_prev = re.search(r'(\S+)\s*\?', text)
    if match_prev:
        prev_word = match_prev.group(1)
        if len(prev_word) <= 1 and prev_word not in ["के", "को", "छ", "त", "नै", "र", "वा", "हो"]:
            return False, f"unclear_preceding_word_{prev_word}"
            
    # Rule A: Check question patterns
    for pat in COMPILED_PATTERNS:
        if pat.search(text):
            return True, "matched_question_pattern"
            
    # Check ending verb/particle patterns
    if ENDING_VERB_PAT.search(text) or ENDING_PARTICLE_PAT.search(text) or ENDING_KI_PAT.search(text):
        return True, "matched_ending_pattern"
        
    return False, "missing_question_indicators"

def main():
    print("Step 1: Standardizing label filenames...")
    standardize_filenames()
    
    hist_path = os.path.join(LABEL_DIR, "historical_nepali_lines.csv")
    lit_path = os.path.join(LABEL_DIR, "nepali_literary_lines.csv")
    sk_path = os.path.join(LABEL_DIR, "sanskrit_lines.csv")
    
    if not (os.path.exists(hist_path) and os.path.exists(lit_path) and os.path.exists(sk_path)):
        print(f"Error: Missing one of the required CSVs under {LABEL_DIR}.")
        sys.exit(1)
        
    print("Loading CSV files...")
    hist_rows = load_csv(hist_path)
    lit_rows = load_csv(lit_path)
    sanskrit_all = load_csv(sk_path)
    
    print(f"Loaded rows: Historical={len(hist_rows)}, Literary={len(lit_rows)}, Sanskrit={len(sanskrit_all)}")
    
    # Assign domains
    for r in hist_rows: r['domain'] = 'historical_nepali'
    for r in lit_rows: r['domain'] = 'nepali_literary'
    for r in sanskrit_all: r['domain'] = 'sanskrit'
    
    # Deterministic Sanskrit sampling
    print("Sampling Sanskrit rows proportionally...")
    sanskrit_sampled, sanskrit_sampling_summary = sample_sanskrit_proportionally(sanskrit_all, 7500, seed=42)
    print(f"Sampled Sanskrit rows: {len(sanskrit_sampled)}")
    
    # Combined set of candidate rows
    candidates = []
    candidates.extend(hist_rows)
    candidates.extend(lit_rows)
    candidates.extend(sanskrit_sampled)
    
    print(f"Combined candidates total: {len(candidates)}")
    
    # Deduplicate text labels based on priority:
    # historical_nepali (3) > nepali_literary (2) > sanskrit (1)
    priority_map = {
        'historical_nepali': 3,
        'nepali_literary': 2,
        'sanskrit': 1
    }
    
    text_to_row = {}
    duplicates_removed = []
    
    # Sort candidates stably by domain priority descending, then by line_id ascending
    # to guarantee deterministic conflict resolution (higher priority domain kept)
    candidates_sorted = sorted(
        candidates, 
        key=lambda x: (-priority_map[x['domain']], x['line_id'])
    )
    
    for r in candidates_sorted:
        text = r['text']
        if text not in text_to_row:
            text_to_row[text] = r
        else:
            existing = text_to_row[text]
            duplicates_removed.append({
                "text": text,
                "discarded_line_id": r['line_id'],
                "discarded_domain": r['domain'],
                "kept_line_id": existing['line_id'],
                "kept_domain": existing['domain']
            })
            
    merged_rows = list(text_to_row.values())
    print(f"Deduplication complete. Unique texts kept: {len(merged_rows)} (Removed {len(duplicates_removed)} duplicates)")
    
    # Filter question mark rows
    cleaned_rows = []
    removed_q_rows = []
    kept_q_rows_count = 0
    removed_q_rows_count = 0
    total_q_in_kept_count = 0
    
    for r in merged_rows:
        text = r['text']
        if '?' in text:
            keep, reason = check_keep_question_mark_row(text)
            if keep:
                cleaned_rows.append(r)
                kept_q_rows_count += 1
                total_q_in_kept_count += text.count('?')
            else:
                removed_q_rows.append({
                    "line_id": r['line_id'],
                    "domain": r['domain'],
                    "source_id": r['source_id'],
                    "source_file": r['source_file'],
                    "text": text,
                    "reason": reason
                })
                removed_q_rows_count += 1
        else:
            cleaned_rows.append(r)
            
    print(f"Question mark cleanup summary:")
    print(f"  Total rows with '?': {kept_q_rows_count + removed_q_rows_count}")
    print(f"  kept_valid_question_rows: {kept_q_rows_count}")
    print(f"  removed_question_mark_artifact_rows: {removed_q_rows_count}")
    print(f"  question_mark_rows (kept): {kept_q_rows_count}")
    print(f"  question_mark_count (total occurrences in kept): {total_q_in_kept_count}")
    
    # Save the removed question mark rows to a CSV for inspection
    removed_q_csv_path = os.path.join(REPORT_DIR, "removed_question_mark_rows.csv")
    with open(removed_q_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["line_id", "domain", "source_id", "source_file", "text", "reason"])
        writer.writeheader()
        writer.writerows(removed_q_rows)
    print(f"Saved removed question mark rows to {removed_q_csv_path}")
    
    merged_rows = cleaned_rows
    
    # Recompute and validate metadata
    print("Recomputing metadata values and conducting validation checks...")
    final_rows = []
    line_ids = set()
    
    validation_failures = []
    
    for r in merged_rows:
        text = r['text']
        line_id = r['line_id']
        domain = r['domain']
        split = r['split']
        
        # Check duplicate line_ids
        if line_id in line_ids:
            validation_failures.append(f"Duplicate line_id collision detected: {line_id}")
        line_ids.add(line_id)
        
        # Recompute metadata
        line_len, has_danda, has_double_danda, has_digit, has_conjunct = recompute_metadata(text)
        
        # Check constraints
        if not text.strip():
            validation_failures.append(f"Blank text row at line_id={line_id}")
        if line_len < 5:
            validation_failures.append(f"Line too short (<5 chars) at line_id={line_id}: length={line_len}, text='{text}'")
        if line_len > 160:
            validation_failures.append(f"Line too long (>160 chars) at line_id={line_id}: length={line_len}, text='{text}'")
        if re.search(r'[0-9]', text):
            validation_failures.append(f"ASCII digits detected at line_id={line_id}: text='{text}'")
        if re.search(r'[a-zA-Z]', text):
            validation_failures.append(f"Latin/English letters detected at line_id={line_id}: text='{text}'")
        if re.search(r'[_|&%€$£₹«»\[\]]', text):
            validation_failures.append(f"Bad OCR symbols detected at line_id={line_id}: text='{text}'")
            
        final_rows.append({
            "line_id": line_id,
            "source_id": r['source_id'],
            "source_file": r['source_file'],
            "text": text,
            "line_length": str(line_len),
            "has_danda": has_danda,
            "has_double_danda": has_double_danda,
            "has_digit": has_digit,
            "has_conjunct": has_conjunct,
            "split": split,
            "domain": domain
        })
        
    # Check splits list
    distinct_splits = set(r['split'] for r in final_rows)
    if not {"train", "val", "test"}.issubset(distinct_splits):
        validation_failures.append(f"Missing required splits. Found splits: {distinct_splits}")
        
    # Check domain counts
    domain_counts = collections.Counter(r['domain'] for r in final_rows)
    print("Domain counts:")
    for d, c in sorted(domain_counts.items()):
        print(f"  {d}: {c}")
        
    expected_hist = len(hist_rows) # 21,855
    expected_lit = len(lit_rows)   # 13,869
    expected_sanskrit = 7500
    
    # We allow slight difference due to duplicates removed
    print(f"Expectations check:")
    print(f"  Historical Nepali: {domain_counts['historical_nepali']} vs expected {expected_hist}")
    print(f"  Nepali Literary: {domain_counts['nepali_literary']} vs expected {expected_lit}")
    print(f"  Sanskrit: {domain_counts['sanskrit']} vs expected {expected_sanskrit}")
    
    # Compile Report data
    split_counts = collections.Counter(r['split'] for r in final_rows)
    
    domain_split_matrix = collections.defaultdict(lambda: collections.defaultdict(int))
    for r in final_rows:
        domain_split_matrix[r['domain']][r['split']] += 1
        
    source_counts = collections.Counter(r['source_id'] for r in final_rows)
    
    qa_status = "PASS"
    if validation_failures:
        qa_status = "FAIL"
        print("\n=== VALIDATION FAILURES ===")
        for f in validation_failures[:20]:
            print(f"  * {f}")
        if len(validation_failures) > 20:
            print(f"  ... and {len(validation_failures)-20} more failures.")
        print("===========================\n")
        # Fail hard
        sys.exit(1)
    else:
        print("[QA PASS] All validation checks passed successfully.")
        
    # Sort final rows deterministically by line_id before saving
    final_rows_sorted = sorted(final_rows, key=lambda x: x['line_id'])
    
    # Save CSV file
    out_csv_path = os.path.join(LABEL_DIR, "stage1_core_merged_lines.csv")
    fieldnames = [
        "line_id", "source_id", "source_file", "text", "line_length",
        "has_danda", "has_double_danda", "has_digit", "has_conjunct", "split", "domain"
    ]
    with open(out_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_rows_sorted)
    print(f"Saved merged labels CSV to {out_csv_path}")
    
    # Build JSON report
    report_json = {
        "qa_status": qa_status,
        "validation_errors": validation_failures,
        "total_rows": len(final_rows_sorted),
        "domain_counts": dict(domain_counts),
        "split_counts": dict(split_counts),
        "domain_split_table": {d: dict(split_counts_dict) for d, split_counts_dict in domain_split_matrix.items()},
        "source_id_counts": dict(source_counts),
        "duplicates_removed_count": len(duplicates_removed),
        "duplicates_removed_samples": duplicates_removed[:50],  # list all or subset
        "sanskrit_sampling_summary": sanskrit_sampling_summary,
        "question_mark_cleanup": {
            "question_mark_rows": kept_q_rows_count,
            "question_mark_count": total_q_in_kept_count,
            "removed_question_mark_artifact_rows": removed_q_rows_count,
            "kept_valid_question_rows": kept_q_rows_count
        }
    }
    
    report_json_path = os.path.join(REPORT_DIR, "stage1_core_merge_report.json")
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report_json, f, indent=4)
    print(f"Saved JSON report to {report_json_path}")
    
    # Build Markdown report
    md_report = f"""# MANTRA Stage 1 Core Merge Report
    
## Executive Summary
- **QA Status**: **{qa_status}**
- **Total Merged Rows**: {len(final_rows_sorted)}
- **Total Duplicate Rows Removed**: {len(duplicates_removed)}
- **Question Mark Rows (Kept)**: {kept_q_rows_count} (valid questions)
- **Question Mark Count (Occurrences in Kept)**: {total_q_in_kept_count}
- **Removed Question Mark Artifact Rows**: {removed_q_rows_count} (junk rows)

## Domain-Wise Counts
| Domain | Count | Expected | Difference |
| --- | --- | --- | --- |
| Historical Nepali | {domain_counts['historical_nepali']} | {expected_hist} | {domain_counts['historical_nepali'] - expected_hist} |
| Nepali Literary | {domain_counts['nepali_literary']} | {expected_lit} | {domain_counts['nepali_literary'] - expected_lit} |
| Sanskrit | {domain_counts['sanskrit']} | {expected_sanskrit} | {domain_counts['sanskrit'] - expected_sanskrit} |

## Split-Wise Counts
- **train**: {split_counts['train']} ({split_counts['train']/len(final_rows_sorted)*100:.2f}%)
- **val**: {split_counts['val']} ({split_counts['val']/len(final_rows_sorted)*100:.2f}%)
- **test**: {split_counts['test']} ({split_counts['test']/len(final_rows_sorted)*100:.2f}%)

## Domain × Split Table
| Domain | Train | Val | Test | Total |
| --- | --- | --- | --- | --- |
| Historical Nepali | {domain_split_matrix['historical_nepali']['train']} | {domain_split_matrix['historical_nepali']['val']} | {domain_split_matrix['historical_nepali']['test']} | {domain_counts['historical_nepali']} |
| Nepali Literary | {domain_split_matrix['nepali_literary']['train']} | {domain_split_matrix['nepali_literary']['val']} | {domain_split_matrix['nepali_literary']['test']} | {domain_counts['nepali_literary']} |
| Sanskrit | {domain_split_matrix['sanskrit']['train']} | {domain_split_matrix['sanskrit']['val']} | {domain_split_matrix['sanskrit']['test']} | {domain_counts['sanskrit']} |

## Source ID Breakdown
"""
    for src, c in sorted(source_counts.items()):
        md_report += f"- **{src}**: {c} rows\n"
        
    md_report += f"""
## Sanskrit Sampling Details (Seed = 42)
Target target splits: Train 6000, Val 750, Test 750.
Proportional allocation counts achieved:
"""
    for k, v in sorted(sanskrit_sampling_summary.items()):
        md_report += f"- **{k}**: allocated target = {v['target_allocated']} (out of {v['available']} available)\n"
        
    if duplicates_removed:
        md_report += f"""
## Duplicate Text Removals ({len(duplicates_removed)} occurrences)
We prioritized text labels using `historical_nepali > nepali_literary > sanskrit`. Below are the duplicate rows removed:
| Discarded ID | Kept ID | Kept Domain | Discarded Domain | Text |
| --- | --- | --- | --- | --- |
"""
        for dup in duplicates_removed[:20]:
            md_report += f"| {dup['discarded_line_id']} | {dup['kept_line_id']} | {dup['kept_domain']} | {dup['discarded_domain']} | `{dup['text']}` |\n"
        if len(duplicates_removed) > 20:
            md_report += f"| ... | ... | ... | ... | ... ({len(duplicates_removed)-20} more) |\n"
            
    if removed_q_rows:
        md_report += f"""
## Question Mark Artifact Removals ({len(removed_q_rows)} occurrences)
We removed rows where '?' appeared as OCR uncertainty / junk, and kept valid questions. Below are the first 20 removed rows:
| Discarded ID | Domain | Source | Text | Reason |
| --- | --- | --- | --- | --- |
"""
        for r_row in removed_q_rows[:20]:
            clean_text = r_row['text'].replace('|', '\\|')
            md_report += f"| {r_row['line_id']} | {r_row['domain']} | {r_row['source_id']} | `{clean_text}` | `{r_row['reason']}` |\n"
        if len(removed_q_rows) > 20:
            md_report += f"| ... | ... | ... | ... | ... ({len(removed_q_rows)-20} more) |\n"
            
    md_report += """
## QA Validation Summary
- **Blank texts check**: Passed (0 found)
- **Duplicate line_id check**: Passed (0 found)
- **Duplicate text labels check**: Passed (0 found after deduplication)
- **Line length constraint check (5-160 chars)**: Passed
- **No ASCII digits [0-9] check**: Passed
- **No English letters check**: Passed
- **No bad symbols excluding valid '?' check**: Passed (0 found)
"""

    report_md_path = os.path.join(REPORT_DIR, "stage1_core_merge_report.md")
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(md_report)
    print(f"Saved Markdown report to {report_md_path}")

if __name__ == "__main__":
    main()
