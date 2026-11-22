import os
import sys
import json
import csv
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

def fetch_json(url, retries=3):
    req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode('utf-8'))
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(1 + attempt)

def download_image(url, output_path, retries=3):
    if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
        return output_path, os.path.getsize(output_path), "already_exists"
    
    req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
                if len(data) < 1000:
                    raise ValueError(f"Downloaded file too small: {len(data)} bytes")
                with open(output_path, "wb") as f:
                    f.write(data)
                return output_path, len(data), "downloaded"
        except Exception as e:
            if attempt == retries - 1:
                return output_path, 0, f"failed: {str(e)}"
            time.sleep(1 + attempt * 2)

def main():
    record_id = "EAP272-1-2-11"
    clean_id = record_id.replace("-", "_")
    base_dir = os.path.join("data", "raw_manuscripts", record_id)
    images_dir = os.path.join(base_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    manifest_url = f"https://eap.bl.uk/archive-file/{record_id}/manifest"
    print(f"Fetching IIIF manifest from {manifest_url}...")
    manifest_data = fetch_json(manifest_url)

    # Save manifest
    manifest_file = os.path.join(base_dir, "manifest.json")
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2, ensure_ascii=False)
    print(f"Saved manifest to {manifest_file}")

    # Extract metadata
    metadata_raw = manifest_data.get("metadata", [])
    parsed_metadata = {}
    for item in metadata_raw:
        label_dict = item.get("label", {})
        val_dict = item.get("value", {})
        label = list(label_dict.values())[0][0] if label_dict else ""
        val = list(val_dict.values())[0][0] if val_dict else ""
        if label:
            parsed_metadata[label] = val

    metadata_info = {
        "record_id": record_id,
        "title": manifest_data.get("label", {}).get("en", [""])[0],
        "source_url": f"https://eap.bl.uk/archive-file/{record_id}",
        "manifest_url": manifest_url,
        "metadata_fields": parsed_metadata
    }

    metadata_file = os.path.join(base_dir, "metadata.json")
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata_info, f, indent=2, ensure_ascii=False)
    print(f"Saved metadata to {metadata_file}")

    # Parse items/canvases
    items = manifest_data.get("items", [])
    print(f"Found {len(items)} image canvases to download.")

    download_tasks = []
    records = []

    for idx, item in enumerate(items):
        page_num = idx + 1
        label = item.get("label", {}).get("en", [str(page_num)])[0]
        w = item.get("width")
        h = item.get("height")
        
        sub_items = item.get("items", [{}])[0].get("items", [{}])[0]
        body = sub_items.get("body", {})
        img_url = body.get("id")

        filename = f"{clean_id}_page_{page_num:03d}.jpg"
        filepath = os.path.join(images_dir, filename)

        records.append({
            "page_num": page_num,
            "label": label,
            "filename": filename,
            "filepath": filepath.replace("\\", "/"),
            "width": w,
            "height": h,
            "url": img_url
        })
        download_tasks.append((img_url, filepath, page_num))

    # Download in parallel
    print(f"Starting parallel download with 8 threads...")
    results = {}
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_task = {
            executor.submit(download_image, url, path): (url, path, page_num)
            for url, path, page_num in download_tasks
        }
        completed = 0
        total = len(download_tasks)
        for future in as_completed(future_to_task):
            url, path, page_num = future_to_task[future]
            try:
                out_path, size, status = future.result()
                results[page_num] = (size, status)
            except Exception as e:
                results[page_num] = (0, f"error: {str(e)}")
            
            completed += 1
            if completed % 15 == 0 or completed == total:
                elapsed = time.time() - start_time
                print(f"Progress: {completed}/{total} pages downloaded ({completed/total*100:.1f}%) in {elapsed:.1f}s")

    # Write CSV manifest
    csv_file = os.path.join(base_dir, "pages_manifest.csv")
    with open(csv_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["page_num", "label", "filename", "filepath", "width", "height", "file_size_bytes", "status", "iiif_image_url"])
        for r in records:
            p_num = r["page_num"]
            size, status = results.get(p_num, (0, "unknown"))
            writer.writerow([
                p_num,
                r["label"],
                r["filename"],
                r["filepath"],
                r["width"],
                r["height"],
                size,
                status,
                r["url"]
            ])

    print(f"Saved CSV page manifest to {csv_file}")
    
    # Summary
    success_count = sum(1 for s in results.values() if "downloaded" in s[1] or "already_exists" in s[1])
    total_bytes = sum(s[0] for s in results.values())
    print(f"\n--- Download Summary ---")
    print(f"Total pages: {len(items)}")
    print(f"Successfully downloaded: {success_count}/{len(items)}")
    print(f"Total size: {total_bytes / (1024 * 1024):.2f} MB")
    print(f"Directory: {base_dir}")

if __name__ == "__main__":
    main()
