"""Propose manuscript line crops and retain source geometry for human review.

Run with --help for arguments. Outputs are draft annotations. The detector is
intended for a single main text block with roughly horizontal Devanagari lines.
It preserves source pixels inside polygons and writes white outside them.
"""

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps


VERSION = "3"
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".jp2", ".j2k"}


@dataclass(frozen=True)
class ScanPage:
    source: Path
    frame_index: int
    frame_count: int
    stem: str
    error: str = ""


def discover_pages(source):
    """Expand scan files in natural filename order, preserving bad inputs as errors."""
    paths = sorted(
        (path for path in source.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS),
        key=lambda path: [int(part) if part.isdigit() else part.casefold()
                          for part in re.split(r"(\d+)", path.name)],
    )
    pages = []
    for path in paths:
        try:
            with Image.open(path) as image:
                count = image.n_frames if hasattr(image, "n_frames") else 1
            for frame in range(count):
                stem = path.stem if count == 1 else f"{path.stem}_page{frame + 1:04d}"
                pages.append(ScanPage(path, frame, count, stem))
        except (OSError, ValueError, EOFError) as error:
            pages.append(ScanPage(path, 0, 0, path.stem, str(error)))
    counts = Counter(page.stem.casefold() for page in pages)
    identified = []
    for page in pages:
        stem = page.stem
        if counts[stem.casefold()] > 1:
            suffix = hashlib.sha256(page.source.name.encode("utf-8")).hexdigest()[:12]
            stem = f"{stem}__{suffix}"
        identified.append(ScanPage(page.source, page.frame_index, page.frame_count, stem, page.error))
    if len({page.stem.casefold() for page in identified}) != len(identified):
        raise ValueError("Input filenames produce duplicate page IDs; rename the conflicting files")
    return identified


def normalize_scan(scan, destination):
    """Convert one source frame to RGB and save a PNG when normalization is needed."""
    with Image.open(scan.source) as opened:
        opened.seek(scan.frame_index)
        source_format = opened.format
        source_mode = opened.mode
        source_size = list(opened.size)
        orientation = int(opened.getexif().get(274, 1))
        image = ImageOps.exif_transpose(opened)
        conversion = []
        if orientation != 1:
            conversion.append(f"EXIF orientation {orientation} applied")
        if image.mode.startswith("I;16"):
            values = np.asarray(image).astype(np.uint32)
            image = Image.fromarray(((values + 128) // 257).astype(np.uint8))
            conversion.append("16-bit grayscale scaled to 8-bit using the full 0..65535 range")
        if image.mode in {"I", "F"}:
            raise ValueError(f"Unsupported {image.mode} pixel mode; supply an explicitly mapped 8-bit or 16-bit scan")
        if "A" in image.getbands() or "transparency" in image.info:
            background = Image.new("RGBA", image.size, "white")
            image = Image.alpha_composite(background, image.convert("RGBA"))
            conversion.append("Transparency composited on white")
        image = image.convert("RGB")
        if source_mode != "RGB":
            conversion.append(f"Source mode {source_mode} converted to RGB")
        normalized_png = str(scan.source.resolve())
        if source_format != "PNG" or scan.frame_count != 1 or orientation != 1 or source_mode not in {"RGB", "L", "1"}:
            normalized_png = "normalized.png"
            image.save(destination / normalized_png, format="PNG")
    return image, {
        "source_format": source_format, "source_mode": source_mode,
        "source_size": source_size, "source_frame_index": scan.frame_index,
        "source_frame_count": scan.frame_count, "normalized_png": normalized_png,
        "normalized_size": list(image.size), "normalization": conversion,
    }


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def detection_gray(image):
    """Ignore long red ruling during detection while retaining red text marks."""
    pixels = np.asarray(image)
    gray = np.asarray(image.convert("L")).copy()
    red = pixels[:, :, 0].astype(np.int16)
    colored = ((red - pixels[:, :, 1] > 15) & (red - pixels[:, :, 2] > 15)).astype(np.uint8)
    height, width = gray.shape
    joined = cv2.morphologyEx(colored, cv2.MORPH_CLOSE, np.ones((max(1, height // 100), 1), dtype=np.uint8))
    _, labels, components, _ = cv2.connectedComponentsWithStats(joined, 8)
    rules = np.zeros(gray.shape, dtype=np.uint8)
    for index, component in enumerate(components[1:], start=1):
        if component[3] > height * 0.25 and component[2] < width * 0.04:
            rules[labels == index] = 1
    rules = cv2.dilate(rules, np.ones((1, 3), dtype=np.uint8))
    gray[rules.astype(bool)] = 255
    return gray


def largest_band(active, close_size, horizontal=False):
    values = active.astype(np.uint8).reshape(-1, 1)
    kernel = np.ones((close_size, 1), dtype=np.uint8)
    if horizontal:
        values = values.T
        kernel = kernel.T
    joined = cv2.morphologyEx(values, cv2.MORPH_CLOSE, kernel)
    _, _, components, _ = cv2.connectedComponentsWithStats(joined, 8)
    if len(components) < 2:
        raise ValueError("No usable text band detected")
    axis = 0 if horizontal else 1
    length = 2 if horizontal else 3
    component = max(components[1:], key=lambda item: int(item[length]))
    start = int(component[axis])
    return start, start + int(component[length])


def line_headlines(profile, page_width):
    """Estimate line pitch, then suppress secondary peaks within each line."""
    smooth = np.convolve(profile, [0.2, 0.6, 0.2], mode="same")
    if len(profile) < 3 or not smooth.max():
        raise ValueError("Text projection is empty")
    if len(profile) < page_width * 0.045:
        return [int(smooth.argmax())], None
    centered = smooth - smooth.mean()
    correlation = np.correlate(centered, centered, mode="full")[len(profile) - 1:]
    correlation /= max(float(correlation[0]), 1e-8)
    lower = max(12, int(page_width * 0.012))
    upper = min(int(page_width * 0.065), len(profile) // 2)
    peaks = [
        index for index in range(lower, upper)
        if correlation[index] >= correlation[index - 1]
        and correlation[index] > correlation[index + 1]
    ]
    if not peaks:
        raise ValueError("Could not estimate line spacing; set explicit headlines")
    strongest = max(correlation[index] for index in peaks)
    pitch = next(index for index in peaks if correlation[index] >= 0.62 * strongest)
    candidates = [
        index for index in range(1, len(smooth) - 1)
        if smooth[index] >= smooth[index - 1]
        and smooth[index] > smooth[index + 1]
        and smooth[index] > 0.30 * smooth.max()
    ]
    selected = []
    for candidate in sorted(candidates, key=lambda index: smooth[index], reverse=True):
        if all(abs(candidate - other) > max(8, pitch * 0.62) for other in selected):
            selected.append(candidate)
    if not selected:
        raise ValueError("No line peaks detected")
    return sorted(selected), pitch


def detect_layout(image, override):
    width, height = image.size
    scale = min(1.0, 1600 / width)
    analysis_width = round(width * scale)
    analysis_height = round(height * scale)
    small = image.resize((analysis_width, analysis_height), Image.Resampling.LANCZOS)
    gray = detection_gray(small)
    threshold, _ = cv2.threshold(
        gray[int(analysis_height * 0.18):int(analysis_height * 0.8),
             int(analysis_width * 0.2):int(analysis_width * 0.85)],
        0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    mask = (gray < threshold).astype(np.uint8)
    if "roi_fraction" in override:
        fraction = override["roi_fraction"]
        if len(fraction) != 4 or not all(0 <= value <= 1 for value in fraction):
            raise ValueError("ROI fractions must be four values between zero and one")
        left, top, right, bottom = (
            round(fraction[0] * analysis_width), round(fraction[1] * analysis_height),
            round(fraction[2] * analysis_width), round(fraction[3] * analysis_height),
        )
    else:
        left, right = int(analysis_width * 0.09), int(analysis_width * 0.93)
        top, bottom = int(analysis_height * 0.10), int(analysis_height * 0.89)
        profile = mask[top:bottom, left:right].mean(axis=1)
        active = profile > max(0.025, float(np.percentile(profile, 90)) * 0.16)
        band_top, band_bottom = largest_band(active, 15)
        top, bottom = top + band_top, top + band_bottom
        columns = mask[top:bottom, left:right].mean(axis=0)
        active = columns > max(0.025, float(np.percentile(columns, 85)) * 0.16)
        band_left, band_right = largest_band(active, 7, horizontal=True)
        left, right = left + band_left, left + band_right
    if not (0 <= left < right <= analysis_width and 0 <= top < bottom <= analysis_height):
        raise ValueError("Text region must have positive area within the image")
    if "headlines_fraction" in override:
        headlines = [round(value * analysis_height) for value in override["headlines_fraction"]]
        if not headlines or any(not top <= value < bottom for value in headlines):
            raise ValueError("Explicit headlines must fall within the text region")
        if headlines != sorted(set(headlines)):
            raise ValueError("Explicit headlines must be increasing and distinct")
        pitch = float(np.median(np.diff(headlines))) if len(headlines) > 1 else bottom - top
    else:
        projection_top = max(0, top - 8)
        projection_bottom = min(analysis_height, bottom + 8)
        profile = mask[projection_top:projection_bottom, left:right].mean(axis=1)
        positions, pitch = line_headlines(profile, analysis_width)
        headlines = [projection_top + position for position in positions]
        pitch = pitch or bottom - top
    padding = max(5, round(pitch * 0.22))
    left = max(0, left - padding)
    right = min(analysis_width, right + padding)
    top = max(0, min(top, headlines[0]) - round(pitch * 0.65))
    bottom = min(analysis_height, max(bottom, headlines[-1]) + round(pitch * 0.45))
    return {
        "roi": [math.floor(left * width / analysis_width),
                math.floor(top * height / analysis_height),
                math.ceil(right * width / analysis_width),
                math.ceil(bottom * height / analysis_height)],
        "headlines": [round(value * height / analysis_height) for value in headlines],
        "pitch": pitch * height / analysis_height,
        "ink_threshold": float(threshold),
    }


def gap_path(gray, left, right, upper_headline, lower_headline, ink_threshold):
    """Find a low-ink path between neighbouring lines in native coordinates."""
    pitch = lower_headline - upper_headline
    if pitch < 4:
        raise ValueError("Line spacing is too small")
    low = upper_headline + max(2, round(pitch * 0.42))
    high = lower_headline - 1
    center = upper_headline + pitch * 0.83
    patch = gray[low:high, left:right].astype(np.float32)
    if not patch.size:
        raise ValueError("Empty corridor between lines")
    ink = (patch < ink_threshold).astype(np.float32)
    darkness = np.maximum(0, ink_threshold + 40 - patch) / 80
    distance = (np.arange(low, high)[:, None] - center) / max(pitch, 1)
    cost = 180 * ink + darkness + 18 * distance ** 2
    row_count, column_count = cost.shape
    history = np.empty((row_count, column_count), dtype=np.int32)
    scores = cost[:, 0].copy()
    rows = np.arange(row_count)
    for column in range(1, column_count):
        predecessors = np.stack([
            np.r_[np.inf, scores[:-1] + 0.25], scores,
            np.r_[scores[1:] + 0.25, np.inf],
        ])
        choice = predecessors.argmin(axis=0)
        scores = cost[:, column] + predecessors[choice, rows]
        history[:, column] = rows + choice - 1
    position = int(scores.argmin())
    seam = np.empty(column_count, dtype=np.int32)
    for column in range(column_count - 1, -1, -1):
        seam[column] = position + low
        if column:
            position = int(history[position, column])
    hits = gray[seam, np.arange(left, right)] < ink_threshold
    return seam, float(hits.mean())


def crop_polygon(image, left, right, top, bottom):
    """Extract unchanged source pixels inside an exclusive-bottom polygon mask."""
    width, height = image.size
    if not (0 <= left < right <= width):
        raise ValueError("Invalid horizontal crop bounds")
    if len(top) != right - left or len(bottom) != right - left:
        raise ValueError("Boundary lengths do not match crop width")
    if np.any(top < 0) or np.any(bottom > height) or np.any(top >= bottom):
        raise ValueError("Crossing or out-of-image boundaries")
    min_y, max_y = int(np.min(top)), int(np.max(bottom))
    source = np.asarray(image)[min_y:max_y, left:right]
    rows = np.arange(min_y, max_y)[:, None]
    mask = (rows >= top[None, :]) & (rows < bottom[None, :])
    result = np.full(source.shape, 255, dtype=np.uint8)
    result[mask] = source[mask]
    polygon = [[left + index, int(value)] for index, value in enumerate(top)]
    polygon += [[left + index, int(bottom[index])] for index in range(right - left - 1, -1, -1)]
    return Image.fromarray(result), [left, min_y, right, max_y], polygon


def export_page(path, destination, override, binding, scan=None, source_checksum=None):
    scan = scan or ScanPage(path, 0, 1, path.stem)
    image, normalization = normalize_scan(scan, destination)
    record = {
        "source_file": str(path.resolve()), "source_sha256": source_checksum or digest(path),
        **normalization, "page_id": scan.stem, "binding": binding,
        "review_status": "pending", "flags": [], "lines": [],
        "scope": "Main text block; marginal notes and folio numbers need separate review",
        "coordinate_system": "Normalized PNG pixels, top-left origin, exclusive right/bottom bbox",
    }
    overview = image.copy()
    draw = ImageDraw.Draw(overview)
    if override.get("action") == "review_only":
        record["flags"] = [override.get("reason", "Manual layout decision required")]
        record["status"] = "review_only"
    else:
        layout = detect_layout(image, override)
        record["layout"] = layout
        left, top, right, bottom = layout["roi"]
        headlines = layout["headlines"]
        gray = detection_gray(image)
        boundaries = [np.full(right - left, top, dtype=np.int32)]
        crossings = []
        for upper, lower in zip(headlines, headlines[1:]):
            seam, fraction = gap_path(gray, left, right, upper, lower, layout["ink_threshold"])
            boundaries.append(seam)
            crossings.append(fraction)
        boundaries.append(np.full(right - left, bottom, dtype=np.int32))
        if len(headlines) > 2:
            spacing = np.diff(headlines)
            median = float(np.median(spacing))
            if spacing.min() < median * 0.65 or spacing.max() > median * 1.4:
                record["flags"].append("Irregular spacing: inspect for a missed or extra line")
        lines_dir = destination / "lines"
        lines_dir.mkdir()
        thumbnails = []
        for number, (upper, lower) in enumerate(zip(boundaries, boundaries[1:]), start=1):
            crop, box, polygon = crop_polygon(image, left, right, upper, lower)
            filename = f"{scan.stem}_line{number:02d}.png"
            crop.save(lines_dir / filename)
            line_flags = []
            edge_hits = [0.0 if number == 1 else crossings[number - 2],
                         0.0 if number == len(headlines) else crossings[number - 1]]
            if max(edge_hits) > 0:
                line_flags.append("Boundary crosses detected ink; inspect touching characters")
            if max(edge_hits) > 0.003:
                record["flags"].append(f"Line {number}: boundary crosses detected ink")
            record["lines"].append({
                "line_id": filename[:-4], "file": "lines/" + filename,
                "sha256": digest(lines_dir / filename), "bbox_xyxy": box,
                "polygon_xy": polygon, "review_status": "pending", "flags": line_flags,
                "boundary_ink_fraction_top_bottom": edge_hits,
                "pixels": "Normalized RGB inside polygon; white outside; no spatial resampling",
            })
            draw.line([tuple(point) for point in polygon + [polygon[0]]], fill=(220, 35, 35), width=max(1, image.width // 800))
            draw.text((min(right + 4, image.width - 35), int(np.mean(upper))),
                      f"L{number:02d}", fill=(220, 35, 35), stroke_width=max(0, image.width // 2500))
            crop.thumbnail((1400, 170), Image.Resampling.LANCZOS)
            thumbnails.append((number, crop.copy()))
        sheet = Image.new("RGB", (1460, sum(crop.height + 26 for _, crop in thumbnails)), "white")
        sheet_draw = ImageDraw.Draw(sheet)
        y = 0
        for number, thumbnail in thumbnails:
            sheet_draw.text((5, y + 4), f"L{number:02d}", fill="black")
            sheet.paste(thumbnail, (50, y + 18))
            y += thumbnail.height + 26
        sheet.save(destination / "line_contact_sheet.jpg", quality=92)
        record["status"] = "draft_segmented"
    overview.thumbnail((1800, 1800), Image.Resampling.LANCZOS)
    overview.save(destination / "page_review.jpg", quality=93)
    record["artifacts"] = {file.relative_to(destination).as_posix(): digest(file)
                           for file in destination.rglob("*") if file.is_file()}
    atomic_json(destination / "page.json", record)
    return record


def extract_page_num(path):
    parts = path.stem.rsplit("_", 1)
    if len(parts) > 1 and parts[1].isdigit():
        return int(parts[1])
    match = re.search(r'(\d+)$', path.stem)
    if match:
        return int(match.group(1))
    raise ValueError(f"Cannot parse page number from scan filename: {path.name}")


def get_override_for_path(path, overrides):
    if not overrides:
        return {}
    if path.stem in overrides:
        return overrides[path.stem]
    stem_hyphen = path.stem.replace("_", "-")
    for key, val in overrides.items():
        if key == stem_hyphen or key.replace("_", "-") == stem_hyphen:
            return val
    try:
        page_num = extract_page_num(path)
        for key, val in overrides.items():
            key_match = re.search(r'(\d+)$', key)
            if key_match and int(key_match.group(1)) == page_num:
                return val
    except ValueError:
        pass
    return {}


def select_pages(paths, selection):
    if not selection:
        return paths
    numbers = set()
    for group in selection.split(","):
        ends = group.split("-")
        if len(ends) == 1:
            numbers.add(int(ends[0]))
        elif len(ends) == 2:
            start, stop = map(int, ends)
            if start > stop:
                raise ValueError("Page ranges must be increasing")
            numbers.update(range(start, stop + 1))
        else:
            raise ValueError("Invalid page selection")
    try:
        labels = [extract_page_num(Path(path.stem + ".png")) for path in paths]
    except ValueError:
        labels = []
    if len(labels) == len(paths) and len(set(labels)) == len(paths):
        available = dict(zip(labels, paths))
    else:
        available = dict(enumerate(paths, start=1))
    missing = numbers - available.keys()
    if missing:
        raise ValueError(f"Requested scan numbers absent: {sorted(missing)}")
    return [available[number] for number in sorted(numbers)]


def run(args):
    source = args.input_dir.resolve()
    output = args.output_dir.resolve()
    if source == output or source in output.parents:
        raise ValueError("Output must be outside the source image directory")
    scans = select_pages(discover_pages(source), args.pages)
    if not scans:
        raise ValueError(f"No supported scan images found. Extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
    overrides = json.loads(args.overrides.read_text(encoding="utf-8")) if args.overrides else {}
    binding = {"version": VERSION, "implementation_sha256": digest(Path(__file__)), "overrides": overrides}
    output.mkdir(parents=True, exist_ok=True)
    lock = output / ".segmentation.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(descriptor)
    records, errors = [], []
    source_hashes = {}
    try:
        identity_path = output / "run_identity.json"
        identity = {"source": str(source), "scans": [
            {"file": scan.source.name, "frame_index": scan.frame_index, "page_id": scan.stem}
            for scan in scans
        ], "settings": binding}
        if identity_path.exists():
            if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
                raise ValueError("This output directory belongs to another selection/settings; choose a new output directory")
        else:
            atomic_json(identity_path, identity)
        pages_dir = output / "pages"
        pages_dir.mkdir(exist_ok=True)
        for position, scan in enumerate(scans, start=1):
            path = scan.source
            destination = pages_dir / scan.stem
            temporary = None
            try:
                if scan.error:
                    raise ValueError(f"Could not read scan: {scan.error}")
                state = (path.stat().st_size, path.stat().st_mtime_ns)
                if path not in source_hashes:
                    source_hashes[path] = (digest(path), state)
                source_checksum, original_state = source_hashes[path]
                if state != original_state:
                    raise ValueError("Source file changed during processing")
                if destination.exists():
                    record = json.loads((destination / "page.json").read_text(encoding="utf-8"))
                    if record["binding"] != binding or record["source_sha256"] != source_checksum:
                        raise ValueError("Existing output has different source/settings; choose a new output directory")
                    if any(digest(destination / name) != checksum for name, checksum in record["artifacts"].items()):
                        raise ValueError("Existing crop or preview changed; choose a new output directory")
                else:
                    temporary = Path(tempfile.mkdtemp(prefix=f".{scan.stem}_", dir=pages_dir))
                    override = get_override_for_path(Path(scan.stem + ".png"), overrides)
                    record = export_page(path, temporary, override, binding,
                                         scan=scan, source_checksum=source_checksum)
                    if (path.stat().st_size, path.stat().st_mtime_ns) != original_state:
                        raise ValueError("Source file changed during processing")
                    for attempt in range(5):
                        try:
                            temporary.rename(destination)
                            break
                        except (PermissionError, OSError):
                            if attempt == 4:
                                if (temporary.resolve().parent != pages_dir.resolve()
                                        or destination.resolve().parent != pages_dir.resolve()):
                                    raise ValueError("Page move target is outside the output directory")
                                shutil.move(str(temporary), str(destination))
                            time.sleep(0.1)
                    temporary = None
                records.append(record)
                print(f"[{position}/{len(scans)}] {scan.stem}: {len(record['lines'])} draft lines, {len(record['flags'])} flags", flush=True)
            except Exception as error:
                errors.append({"source": str(path), "frame_index": scan.frame_index, "error": str(error)})
                print(f"[{position}/{len(scans)}] {scan.stem}: ERROR {error}", flush=True)
            finally:
                if temporary is not None:
                    if temporary.resolve().parent != pages_dir.resolve() or not temporary.name.startswith(f".{scan.stem}_"):
                        raise ValueError("Temporary cleanup target is outside this page's output directory")
                    shutil.rmtree(temporary)
        summary = {
            "version": VERSION, "source_directory": str(source), "selected_scans": len(scans),
            "processed_scans": len(records), "draft_lines": sum(len(page["lines"]) for page in records),
            "approved_lines": 0, "review_status": "All proposed boundaries require visual review",
            "review_only_scans": [Path(page["source_file"]).name for page in records if page["status"] == "review_only"],
            "flagged_scans": [Path(page["source_file"]).name for page in records if page["flags"]],
            "errors": errors,
        }
        atomic_json(output / "summary.json", summary)
        temporary_index = output / "line_index.csv.part"
        with temporary_index.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["line_id", "source_scan", "source_frame", "normalized_png", "image_path", "review_status", "flags"])
            writer.writeheader()
            for page in records:
                stem = page["page_id"]
                normalized = page["normalized_png"]
                if not Path(normalized).is_absolute():
                    normalized = f"pages/{stem}/{normalized}"
                for line in page["lines"]:
                    writer.writerow({"line_id": line["line_id"], "source_scan": page["source_file"],
                                     "source_frame": page["source_frame_index"] + 1, "normalized_png": normalized,
                                     "image_path": f"pages/{stem}/{line['file']}", "review_status": "pending",
                                     "flags": " | ".join(line["flags"])})
        temporary_index.replace(output / "line_index.csv")
        print(json.dumps(summary, ensure_ascii=True), flush=True)
        return 1 if errors else 0
    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Folder of PNG, JPEG, TIFF, BMP, WebP or JPEG 2000 scans; multi-page files are expanded")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overrides", type=Path, help="JSON mapping scan stems to layout decisions")
    parser.add_argument("--pages", help="Scan suffix numbers when unique, otherwise 1-based expanded page positions; e.g. 2-5,20")
    args = parser.parse_args()
    try:
        return run(args)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Segmentation failed: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
