import os
import csv
import re
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="MANTRA QA Audit")
    parser.add_argument("--metadata", type=str, default="data/stage1_synthetic/metadata/synthetic_image_metadata.csv", help="Path to metadata CSV")
    args = parser.parse_args()
    
    metadata_csv_path = args.metadata
    if not os.path.exists(metadata_csv_path):
        logger.error(f"Metadata file not found: {metadata_csv_path}")
        sys.exit(1)
        
    records = []
    with open(metadata_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
            
    total_records = len(records)
    logger.info(f"Loaded {total_records} metadata records for strict QA audit.")
    
    # 1. Audit target sample 'historical_nepali_001_008239'
    target_id = "historical_nepali_001_008239"
    target_row = next((r for r in records if r["line_id"] == target_id), None)
    
    if not target_row:
        logger.info(f"Target line '{target_id}' was not found in metadata CSV. Skipping specific target inspection.")
    else:
        logger.info("=" * 80)
        logger.info(f"SPECIFIC INSPECTION FOR LINE: {target_id}")
        logger.info(f"  Metadata image_path: {target_row['image_path']}")
        logger.info(f"  source_text_original: '{target_row.get('source_text_original')}'")
        logger.info(f"  source_text_rendered: '{target_row.get('source_text_rendered')}'")
        logger.info(f"  text (rendered label): '{target_row.get('text')}'")
        logger.info("=" * 80)
        
        # Strict validation of target line
        orig_text = target_row.get('source_text_original')
        rend_text = target_row.get('source_text_rendered')
        text_label = target_row.get('text')
        
        if re.search(r'[0-9]', rend_text):
            logger.error(f"Target line {target_id} contains ASCII digits in rendered text: {rend_text}")
            sys.exit(1)
        if rend_text != text_label:
            logger.error(f"Target line {target_id} has mismatched rendered fields: '{rend_text}' != '{text_label}'")
            sys.exit(1)
        if text_label != "त्यतिबेला रू. १५ सम्म पर्दथ्यो ।":
            logger.error(f"Target line {target_id} rendered text mismatch: '{text_label}' != 'त्यतिबेला रू. १५ सम्म पर्दथ्यो ।'")
            sys.exit(1)
            
        logger.info("Target line check passed successfully!")
    
    # 2. Audit all records
    errors_found = 0
    ascii_digit_failures = 0
    symbol_failures = 0
    mismatch_failures = 0
    
    for idx, row in enumerate(records):
        line_id = row["line_id"]
        orig = row.get("source_text_original", "")
        rend = row.get("source_text_rendered", "")
        text = row.get("text", "")
        
        # Check text fields exist
        if not orig or not rend or not text:
            logger.error(f"Missing text fields in row {idx} (line_id={line_id})")
            errors_found += 1
            continue
            
        # Check 2a: No ASCII digits [0-9] in rendered text
        if re.search(r'[0-9]', rend):
            logger.error(f"ASCII digits detected in rendered text for line {line_id}: '{rend}'")
            ascii_digit_failures += 1
            errors_found += 1
            
        # Check 2b: rendered text equals label text
        if rend != text:
            logger.error(f"Rendered mismatch for line {line_id}: rendered='{rend}' != text='{text}'")
            mismatch_failures += 1
            errors_found += 1
            
        # Check 2c: Preserving crucial symbols
        # Check 'रू.'
        if "रू." in orig and "रू." not in rend:
            logger.error(f"Symbol 'रू.' lost in rendered text for line {line_id}: orig='{orig}' -> rend='{rend}'")
            symbol_failures += 1
            errors_found += 1
        # Check 'वि. सं'
        if "वि. सं" in orig and "वि. सं" not in rend:
            logger.error(f"Symbol 'वि. सं' lost in rendered text for line {line_id}: orig='{orig}' -> rend='{rend}'")
            symbol_failures += 1
            errors_found += 1
            
        # Check dandas
        if "।" in orig and "।" not in rend:
            logger.error(f"Danda '।' lost in rendered text for line {line_id}: orig='{orig}' -> rend='{rend}'")
            symbol_failures += 1
            errors_found += 1
        if "॥" in orig and "॥" not in rend:
            logger.error(f"Double Danda '॥' lost in rendered text for line {line_id}: orig='{orig}' -> rend='{rend}'")
            symbol_failures += 1
            errors_found += 1
            
    logger.info("QA AUDIT COMPLETED.")
    logger.info(f"  Total audited: {total_records}")
    logger.info(f"  ASCII digit failures: {ascii_digit_failures}")
    logger.info(f"  Symbol/Punctuation failures: {symbol_failures}")
    logger.info(f"  Rendered/Label mismatch failures: {mismatch_failures}")
    logger.info(f"  Total failures: {errors_found}")
    
    if errors_found > 0:
        logger.error("[FAILURE] Strict QA audit failed. Stale/bad text values remain in pipeline.")
        sys.exit(1)
    else:
        logger.info("[SUCCESS] Strict QA audit passed successfully. No ASCII digits or symbol losses detected.")
        sys.exit(0)

if __name__ == "__main__":
    main()
