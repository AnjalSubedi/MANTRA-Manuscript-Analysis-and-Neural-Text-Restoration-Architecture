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


def normalize_scan(scan, destination, rotation=0):
    """Convert one source frame to RGB and save a PNG when normalization is needed."""
    with Image.open(scan.source) as opened:
        opened.seek(scan.frame_index)
        source_format = opened.format
        source_mode = opened.mode
        source_size = list(opened.size)
        orientation = int(opened.getexif().get(274, 1))
        image = ImageOps.exif_transpose(opened)
        oriented_size = image.size
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
        if rotation:
            transpose = {90: Image.Transpose.ROTATE_270, 180: Image.Transpose.ROTATE_180,
                         270: Image.Transpose.ROTATE_90}
            if rotation not in transpose:
                raise ValueError("rotation_degrees must be 0, 90, 180 or 270 clockwise")
            image = image.transpose(transpose[rotation])
            conversion.append(f"Rotated {rotation} degrees clockwise")
        if source_mode != "RGB":
            conversion.append(f"Source mode {source_mode} converted to RGB")
        normalized_png = str(scan.source.resolve())
        if source_format != "PNG" or scan.frame_count != 1 or orientation != 1 or rotation or source_mode not in {"RGB", "L", "1"}:
            normalized_png = "normalized.png"
            image.save(destination / normalized_png, format="PNG")
    return image, {
        "source_format": source_format, "source_mode": source_mode,
        "source_size": source_size, "source_frame_index": scan.frame_index,
        "source_frame_count": scan.frame_count, "normalized_png": normalized_png,
        "normalized_size": list(image.size), "normalization": conversion,
        **({"rotation_degrees": rotation, "exif_orientation": orientation,
            "exif_oriented_size": list(oriented_size)} if rotation else {}),
    }


def source_point(x, y, normalization):
    """Map normalized image-edge coordinates back to the encoded source raster."""
    width, height = normalization["exif_oriented_size"]
    rotation = normalization["rotation_degrees"]
    if rotation == 90:
        x, y = y, height - x
    elif rotation == 180:
        x, y = width - x, height - y
    elif rotation == 270:
        x, y = width - y, x
    width, height = normalization["source_size"]
    orientation = normalization["exif_orientation"]
    return {
        2: (width - x, y), 3: (width - x, height - y), 4: (x, height - y),
        5: (y, x), 6: (y, height - x), 7: (width - y, height - x), 8: (width - y, x),
    }.get(orientation, (x, y))


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


def paper_text_region(gray, inset=(0.09, 0.09, 0.09, 0.10)):
    """Locate the largest bright folio on a dark scan bed and inset its margins."""
    _, paper = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    paper = cv2.morphologyEx(paper, cv2.MORPH_CLOSE, np.ones((17, 17), np.uint8))
    _, _, components, _ = cv2.connectedComponentsWithStats(paper, 8)
    if len(components) < 2:
        raise ValueError("No folio detected; set roi_fraction")
    x, y, width, height, area = map(int, max(components[1:], key=lambda row: row[4]))
    if area < gray.size * 0.15:
        raise ValueError("Folio detection is uncertain; set roi_fraction")
    left, top, right, bottom = inset
    if not all(0 <= value < 0.5 for value in inset):
        raise ValueError("folio_inset_fraction values must be between zero and 0.5")
    return (x + round(width * left), y + round(height * top),
            x + width - round(width * right), y + height - round(height * bottom))


def folio_detection_gray(gray, region):
    """Enhance local ink contrast and mask long straight rules for detection."""
    left, top, right, bottom = region
    patch = gray[top:bottom, left:right]
    background = cv2.GaussianBlur(patch, (0, 0), sigmaX=max(3, patch.shape[1] * 0.025))
    contrast = 255 - cv2.subtract(background, patch)
    ink = (contrast < 243).astype(np.uint8)
    height, width = ink.shape
    horizontal = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((1, 5), np.uint8))
    horizontal = cv2.morphologyEx(horizontal, cv2.MORPH_OPEN, np.ones((1, max(1, round(width * 0.60))), np.uint8))
    vertical = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((5, 1), np.uint8))
    vertical = cv2.morphologyEx(vertical, cv2.MORPH_OPEN, np.ones((max(1, round(height * 0.55)), 1), np.uint8))
    rules = cv2.dilate(horizontal | vertical, np.ones((3, 3), np.uint8))
    contrast[rules != 0] = 255
    result = np.full_like(gray, 255)
    result[top:bottom, left:right] = contrast
    return result


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
    pos_peaks = [index for index in peaks if correlation[index] > 0]
    if pos_peaks:
        strongest = max(correlation[index] for index in pos_peaks)
        pitch = next((index for index in pos_peaks if correlation[index] >= 0.62 * strongest), pos_peaks[0])
    else:
        pitch = max(peaks, key=lambda index: correlation[index])
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
    
    if "dark_scan_bed" in override:
        folio_mode = override["dark_scan_bed"]
    else:
        h_border = max(1, int(analysis_height * 0.04))
        w_border = max(1, int(analysis_width * 0.04))
        border_pixels = np.concatenate([
            gray[:h_border, :].ravel(),
            gray[-h_border:, :].ravel(),
            gray[:, :w_border].ravel(),
            gray[:, -w_border:].ravel(),
        ])
        folio_mode = bool(np.median(border_pixels) < 55)

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
        leaf_regions = [(left, top, right, bottom)]
        use_folio_detection = folio_mode
    elif folio_mode:
        _, paper = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)
        paper = cv2.morphologyEx(paper, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        num, labels, stats, _ = cv2.connectedComponentsWithStats(paper, 8)
        raw_leaves = []
        for i in range(1, num):
            x, y, cw, ch, area = stats[i]
            if (area > analysis_height * analysis_width * 0.07
                    and cw > analysis_width * 0.25
                    and ch > analysis_height * 0.12
                    and x < analysis_width * 0.8):
                raw_leaves.append((x, y, cw, ch, area))
        if not raw_leaves:
            region = paper_text_region(gray, override.get("folio_inset_fraction", (0.09, 0.09, 0.09, 0.10)))
            leaf_regions = [region]
        else:
            raw_leaves.sort(key=lambda item: item[1])
            inset = override.get("folio_inset_fraction", (0.08, 0.08, 0.08, 0.08))
            ileft, itop, iright, ibottom = inset
            leaf_regions = [
                (x + round(cw * ileft), y + round(ch * itop),
                 x + cw - round(cw * iright), y + ch - round(ch * ibottom))
                for x, y, cw, ch, _ in raw_leaves
            ]
        use_folio_detection = True
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
        leaf_regions = [(left, top, right, bottom)]
        use_folio_detection = False

    blocks = []
    for b_left, b_top, b_right, b_bottom in leaf_regions:
        if not (0 <= b_left < b_right <= analysis_width and 0 <= b_top < b_bottom <= analysis_height):
            continue
        detection_region = (b_left, b_top, b_right, b_bottom)
        if use_folio_detection:
            b_contrast = folio_detection_gray(gray, detection_region)
            b_threshold = 243.0
            b_mask = (b_contrast < b_threshold).astype(np.uint8)
        else:
            b_threshold = float(threshold)
            b_mask = mask
        
        if "headlines_fraction" in override and len(leaf_regions) == 1:
            headlines = [round(value * analysis_height) for value in override["headlines_fraction"]]
            if not headlines or any(not b_top <= value < b_bottom for value in headlines):
                raise ValueError("Explicit headlines must fall within the text region")
            if headlines != sorted(set(headlines)):
                raise ValueError("Explicit headlines must be increasing and distinct")
            pitch = float(np.median(np.diff(headlines))) if len(headlines) > 1 else b_bottom - b_top
        else:
            proj_top = max(0, b_top - 6)
            proj_bottom = min(analysis_height, b_bottom + 6)
            profile = b_mask[proj_top:proj_bottom, b_left:b_right].mean(axis=1)
            try:
                positions, pitch = line_headlines(profile, analysis_width)
                headlines = [proj_top + position for position in positions]
                pitch = pitch or b_bottom - b_top
            except ValueError:
                continue
        
        if not headlines:
            continue
        
        padding = max(5, round(pitch * 0.22))
        pad_left = max(0, b_left - padding)
        pad_right = min(analysis_width, b_right + padding)
        pad_top = max(0, min(b_top, headlines[0]) - round(pitch * 0.65))
        pad_bottom = min(analysis_height, max(b_bottom, headlines[-1]) + round(pitch * 0.45))
        
        native_roi = [
            math.floor(pad_left * width / analysis_width),
            math.floor(pad_top * height / analysis_height),
            math.ceil(pad_right * width / analysis_width),
            math.ceil(pad_bottom * height / analysis_height),
        ]
        native_headlines = [round(value * height / analysis_height) for value in headlines]
        native_pitch = pitch * height / analysis_height
        
        blocks.append({
            "roi": native_roi,
            "headlines": native_headlines,
            "pitch": native_pitch,
            "ink_threshold": float(b_threshold),
            **({"detection_region": [
                round(value * (width / analysis_width if i % 2 == 0 else height / analysis_height))
                for i, value in enumerate(detection_region)
            ]} if use_folio_detection else {}),
        })

    if not blocks:
        raise ValueError("No usable text blocks detected")
    
    primary = blocks[0]
    if len(blocks) == 1:
        return {**primary, "blocks": blocks}
    
    overall_roi = [
        min(b["roi"][0] for b in blocks),
        min(b["roi"][1] for b in blocks),
        max(b["roi"][2] for b in blocks),
        max(b["roi"][3] for b in blocks),
    ]
    all_headlines = [h for b in blocks for h in b["headlines"]]
    median_pitch = float(np.median([b["pitch"] for b in blocks]))
    return {
        "roi": overall_roi,
        "headlines": all_headlines,
        "pitch": median_pitch,
        "ink_threshold": primary["ink_threshold"],
        "blocks": blocks,
        **({"detection_region": primary["detection_region"]} if "detection_region" in primary else {}),
    }


def gap_path(gray, left, right, upper_headline, lower_headline, ink_threshold, *, symmetric=False):
    """Find a low-ink path between neighbouring lines in native coordinates."""
    pitch = lower_headline - upper_headline
    if pitch < 4:
        raise ValueError("Line spacing is too small")
    low = upper_headline + max(2, round(pitch * (0.12 if symmetric else 0.42)))
    high = lower_headline - 1
    center = upper_headline + pitch * (0.50 if symmetric else 0.83)
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
    image, normalization = normalize_scan(scan, destination, override.get("rotation_degrees", 0))
    record = {
        "source_file": str(path.resolve()), "source_sha256": source_checksum or digest(path),
        **normalization, "page_id": scan.stem, "binding": binding,
        "review_status": "pending", "flags": [], "lines": [],
        "scope": "Main text block; marginal notes and folio numbers need separate review",
        "coordinate_system": "Normalized PNG pixels, top-left origin, exclusive right/bottom bbox",
    }
    if "rotation_degrees" in normalization:
        record["source_coordinate_system"] = "Encoded source raster, top-left origin, image-edge coordinates"
    overview = image.copy()
    draw = ImageDraw.Draw(overview)
    reason = None
    if override.get("action") == "review_only":
        reason = override.get("reason", "Manual layout decision required")
    else:
        try:
            layout = detect_layout(image, override)
            record["layout"] = layout
            if "expected_line_count" in override:
                minimum, maximum = override["expected_line_count"]
                if not 1 <= minimum <= maximum:
                    raise ValueError("expected_line_count must be a positive minimum and maximum")
                count = len(layout["headlines"])
                if not minimum <= count <= maximum:
                    reason = f"Detected {count} lines outside expected range {minimum}–{maximum}"
        except ValueError as error:
            reason = f"Layout detection uncertain: {error}"
    if reason is not None:
        record["flags"] = [reason]
        record["status"] = "review_only"
    else:
        gray = detection_gray(image)
        blocks = layout.get("blocks", [layout])
        lines_dir = destination / "lines"
        lines_dir.mkdir(exist_ok=True)
        thumbnails = []
        line_number = 1
        for block in blocks:
            b_left, b_top, b_right, b_bottom = block["roi"]
            b_headlines = block["headlines"]
            b_ink_threshold = block["ink_threshold"]
            b_detection_region = block.get("detection_region")
            proc_gray = gray
            if b_detection_region:
                proc_gray = folio_detection_gray(gray, b_detection_region)
            
            if override.get("check_outer_edges"):
                region = b_detection_region or block["roi"]
                rleft, rtop, rright, rbottom = region
                contrast = gray if b_detection_region else folio_detection_gray(gray, region)
                ink = contrast[rtop:rbottom, rleft:rright] < 243
                edges = {"left": ink[:, :3], "top": ink[:3, :],
                         "right": ink[:, -3:], "bottom": ink[-3:, :]}
                for name, edge in edges.items():
                    if edge.mean() > 0.025:
                        record["flags"].append(f"Detected ink near the {name} edge of the text region")
            
            boundaries = [np.full(b_right - b_left, b_top, dtype=np.int32)]
            crossings = []
            for upper, lower in zip(b_headlines, b_headlines[1:]):
                seam, fraction = gap_path(
                    proc_gray, b_left, b_right, upper, lower, b_ink_threshold,
                    symmetric="detection_region" in block
                )
                boundaries.append(seam)
                crossings.append(fraction)
            boundaries.append(np.full(b_right - b_left, b_bottom, dtype=np.int32))
            
            if len(b_headlines) > 2:
                spacing = np.diff(b_headlines)
                median = float(np.median(spacing))
                if spacing.min() < median * 0.65 or spacing.max() > median * 1.4:
                    record["flags"].append("Irregular spacing: inspect for a missed or extra line")
            
            for index, (upper, lower) in enumerate(zip(boundaries, boundaries[1:]), start=1):
                crop, box, polygon = crop_polygon(image, b_left, b_right, upper, lower)
                filename = f"{scan.stem}_line{line_number:02d}.png"
                crop.save(lines_dir / filename)
                line_flags = []
                edge_hits = [
                    0.0 if index == 1 else crossings[index - 2],
                    0.0 if index == len(b_headlines) else crossings[index - 1],
                ]
                if max(edge_hits) > 0:
                    line_flags.append("Boundary crosses detected ink; inspect touching characters")
                if max(edge_hits) > 0.003:
                    record["flags"].append(f"Line {line_number}: boundary crosses detected ink")
                record["lines"].append({
                    "line_id": filename[:-4], "file": "lines/" + filename,
                    "sha256": digest(lines_dir / filename), "bbox_xyxy": box,
                    "polygon_xy": polygon, "review_status": "pending", "flags": line_flags,
                    "boundary_ink_fraction_top_bottom": edge_hits,
                    "pixels": "Normalized RGB inside polygon; white outside; no spatial resampling",
                })
                if "rotation_degrees" in normalization:
                    split = b_right - b_left
                    edge_polygon = (polygon[:split] + [(b_right, int(upper[-1])), (b_right, int(lower[-1]))]
                                    + polygon[split:])
                    source_polygon = [source_point(x, y, normalization) for x, y in edge_polygon]
                    x0, y0, x1, y1 = box
                    corners = [source_point(x, y, normalization) for x in (x0, x1) for y in (y0, y1)]
                    xs, ys = zip(*corners)
                    record["lines"][-1].update(
                        source_polygon_xy=source_polygon,
                        source_bbox_xyxy=[min(xs), min(ys), max(xs), max(ys)],
                    )
                draw.line([tuple(point) for point in polygon + [polygon[0]]],
                          fill=(220, 35, 35), width=max(1, image.width // 800))
                draw.text((min(b_right + 4, image.width - 35), int(np.mean(upper))),
                          f"L{line_number:02d}", fill=(220, 35, 35), stroke_width=max(0, image.width // 2500))
                crop.thumbnail((1400, 170), Image.Resampling.LANCZOS)
                thumbnails.append((line_number, crop.copy()))
                line_number += 1
        
        if thumbnails:
            sheet = Image.new("RGB", (1460, sum(crop.height + 26 for _, crop in thumbnails)), "white")
            sheet_draw = ImageDraw.Draw(sheet)
            y = 0
            for number, thumbnail in thumbnails:
                sheet_draw.text((5, y + 4), f"L{number:02d}", fill="black")
                sheet.paste(thumbnail, (50, y + 18))
                y += thumbnail.height + 26
            sheet.save(destination / "line_contact_sheet.jpg", quality=92)
            record["status"] = "draft_segmented"
        else:
            record["status"] = "review_only"
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
    defaults = overrides.get("defaults", {})
    if path.stem in overrides:
        return {**defaults, **overrides[path.stem]}
    stem_hyphen = path.stem.replace("_", "-")
    for key, val in overrides.items():
        if key == stem_hyphen or key.replace("_", "-") == stem_hyphen:
            return {**defaults, **val}
    try:
        page_num = extract_page_num(path)
        for key, val in overrides.items():
            key_match = re.search(r'(\d+)$', key)
            if key_match and int(key_match.group(1)) == page_num:
                return {**defaults, **val}
    except ValueError:
        pass
    return defaults


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
