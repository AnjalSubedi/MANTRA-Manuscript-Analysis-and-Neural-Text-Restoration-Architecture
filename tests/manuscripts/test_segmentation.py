"""Regression checks for source geometry and preservation of close line marks."""

from argparse import Namespace
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from manuscripts.segment_lines import (
    crop_polygon, detection_gray, discover_pages, export_page, folio_detection_gray,
    gap_path, get_override_for_path, largest_band, line_headlines, line_ink_support,
    normalize_scan, paper_text_region, protect_boundary_ink, run, select_pages, source_point,
)


def test_ink_protection_extends_only_where_a_stroke_straddles_the_seam():
    ink = np.zeros((12, 4), dtype=bool)
    ink[3:8, 1] = True
    ink[5:6, 2] = True
    upper, lower = protect_boundary_ink(ink, np.full(4, 26), 3, image_top=20)
    assert upper.tolist() == [26, 23, 26, 26]
    assert lower.tolist() == [26, 28, 26, 26]


def test_outer_band_support_distinguishes_a_rule_from_handwriting():
    ink = np.zeros((40, 30), dtype=bool)
    ink[2:4] = True
    ink[6:8] = True
    ink[10:12] = True
    ink[22:31, 4:26:2] = True
    top, middle, bottom = (np.full(30, row) for row in (0, 15, 40))
    assert line_ink_support(ink, top, middle, 30, 0) == 0
    assert line_ink_support(ink, middle, bottom, 30, 0) > 0.3


def test_rotation_maps_crop_back_to_original_raster(tmp_path):
    path = tmp_path / "scan.png"
    Image.new("RGB", (100, 160), "white").save(path)
    output = tmp_path / "page"
    output.mkdir()
    record = export_page(path, output, {
        "rotation_degrees": 90, "roi_fraction": [0.1, 0.1, 0.9, 0.9],
        "headlines_fraction": [0.3],
    }, {})
    assert record["normalized_size"] == [160, 100]
    assert source_point(0, 0, record) == (0, 160)
    line = record["lines"][0]
    x0, y0, x1, y1 = line["bbox_xyxy"]
    assert line["source_bbox_xyxy"] == [y0, 160 - x1, y1, 160 - x0]


def test_unexpected_line_count_goes_to_review(tmp_path):
    path = tmp_path / "scan.png"
    Image.new("RGB", (160, 100), "white").save(path)
    output = tmp_path / "page"
    output.mkdir()
    record = export_page(path, output, {
        "roi_fraction": [0.1, 0.1, 0.9, 0.9], "headlines_fraction": [0.3],
        "expected_line_count": [7, 15],
    }, {})
    assert record["status"] == "review_only"
    assert not record["lines"]
    assert "Detected 1 lines" in record["flags"][0]


def test_dark_scan_bed_is_excluded_and_rules_are_masked_for_detection():
    gray = np.full((200, 180), 35, dtype=np.uint8)
    gray[20:180, 30:150] = 200
    gray[50, 46:134] = 60
    gray[61:65, 65:77] = 60
    region = paper_text_region(gray)
    assert region == (41, 34, 139, 164)
    detection = folio_detection_gray(gray, region)
    assert detection[0, 0] == 255
    assert detection[50, 70] == 255
    assert detection[62, 70] < 243
    assert gray[50, 70] == 60


def test_page_overrides_inherit_collection_defaults():
    overrides = {"defaults": {"dark_scan_bed": True}, "scan_002": {"action": "review_only"}}
    assert get_override_for_path(Path("scan_001.jpg"), overrides) == {"dark_scan_bed": True}
    assert get_override_for_path(Path("scan_002.jpg"), overrides) == {
        "dark_scan_bed": True, "action": "review_only",
    }


def test_gap_follows_space_between_descender_and_next_line_matra():
    pixels = np.full((80, 100), 230, dtype=np.uint8)
    pixels[20:30, :] = 60
    pixels[30:43, 10:15] = 60
    pixels[50:60, :] = 60
    pixels[36:50, 55:60] = 60
    boundary, crossing = gap_path(pixels, 0, 100, 20, 50, 145)
    assert crossing == 0
    assert np.all(boundary[10:15] >= 43)
    assert np.all(boundary[55:60] < 36)
    rgb = Image.fromarray(np.repeat(pixels[:, :, None], 3, axis=2))
    first, _, _ = crop_polygon(rgb, 0, 100, np.full(100, 10), boundary)
    second, _, _ = crop_polygon(rgb, 0, 100, boundary, np.full(100, 70))
    assert sum(np.count_nonzero(np.asarray(crop)[:, :, 0] == 60) for crop in [first, second]) == np.count_nonzero(pixels == 60)


def test_polygon_coordinates_roundtrip_source_without_resampling():
    pixels = np.arange(30 * 40 * 3, dtype=np.uint16).reshape(30, 40, 3).astype(np.uint8)
    top = np.array([5, 6, 7, 8, 8, 7, 6, 5])
    bottom = top + 9
    crop, box, polygon = crop_polygon(Image.fromarray(pixels), 10, 18, top, bottom)
    assert box == [10, 5, 18, 17]
    assert polygon[0] == [10, 5]
    actual = np.asarray(crop)
    for column in range(8):
        np.testing.assert_array_equal(actual[top[column]-5:bottom[column]-5, column], pixels[top[column]:bottom[column], column+10])
        assert np.all(actual[:top[column]-5, column] == 255)


def test_crossing_boundaries_are_rejected():
    image = Image.new("RGB", (20, 20))
    with pytest.raises(ValueError, match="Crossing"):
        crop_polygon(image, 0, 20, np.full(20, 12), np.full(20, 10))


def test_one_short_text_band_is_one_line():
    profile = np.zeros(30)
    profile[8:11] = 0.7
    profile[19:21] = 0.5
    positions, pitch = line_headlines(profile, 1250)
    assert len(positions) == 1
    assert pitch is None


def test_absent_requested_pages_are_not_silently_skipped():
    paths = [Path("scan_0001.png"), Path("scan_0002.png")]
    with pytest.raises(ValueError, match="absent"):
        select_pages(paths, "1-3")


def test_red_rules_are_ignored_but_red_letters_remain_detectable():
    pixels = np.full((100, 200, 3), 220, dtype=np.uint8)
    pixels[:, 5:8] = (190, 50, 40)
    pixels[30:45, 60:75] = (190, 50, 40)
    pixels[40:55, 100:115] = (60, 60, 60)
    gray = detection_gray(Image.fromarray(pixels))
    assert np.all(gray[:, 5:8] == 255)
    assert np.all(gray[30:45, 60:75] < 150)
    assert np.all(gray[40:55, 100:115] < 150)


def test_folio_number_is_not_joined_to_main_text_across_a_margin_gap():
    columns = np.zeros(100, dtype=bool)
    columns[2:8] = True
    columns[23:90] = True
    assert largest_band(columns, 7, horizontal=True) == (23, 90)


@pytest.mark.parametrize("extension,format", [
    (".JPG", "JPEG"), (".jpeg", "JPEG"), (".TIFF", "TIFF"),
    (".bmp", "BMP"), (".webp", "WEBP"), (".jp2", "JPEG2000"),
])
def test_scan_formats_convert_to_png_at_original_dimensions(tmp_path, extension, format):
    source = tmp_path / "inputs"
    source.mkdir()
    path = source / ("scan" + extension)
    Image.new("RGB", (73, 41), (190, 180, 170)).save(path, format=format)
    destination = tmp_path / "page"
    destination.mkdir()
    scan = discover_pages(source)[0]
    image, metadata = normalize_scan(scan, destination)
    normalized = Image.open(destination / metadata["normalized_png"])
    assert normalized.format == "PNG"
    assert image.size == normalized.size == (73, 41)
    assert path.exists()
    assert metadata["source_frame_index"] == 0
    assert metadata["source_frame_count"] == 1


def test_png_inputs_keep_ids_and_need_no_duplicate_conversion(tmp_path):
    source = tmp_path / "inputs"
    source.mkdir()
    Image.new("RGB", (30, 20)).save(source / "scan_0001.PNG")
    destination = tmp_path / "page"
    destination.mkdir()
    scan = discover_pages(source)[0]
    assert scan.stem == "scan_0001"
    _, metadata = normalize_scan(scan, destination)
    assert Path(metadata["normalized_png"]) == scan.source.resolve()
    assert not (destination / "normalized.png").exists()


def test_multipage_tiff_preserves_frame_order_and_pixels(tmp_path):
    source = tmp_path / "inputs"
    source.mkdir()
    first = Image.new("RGB", (80, 40), (20, 30, 40))
    second = Image.new("RGB", (61, 39), (90, 100, 110))
    first.save(source / "book.tif", save_all=True, append_images=[second])
    scans = discover_pages(source)
    assert [scan.stem for scan in scans] == ["book_page0001", "book_page0002"]
    assert [scan.frame_index for scan in scans] == [0, 1]
    for scan, expected in zip(scans, [first, second]):
        destination = tmp_path / scan.stem
        destination.mkdir()
        image, metadata = normalize_scan(scan, destination)
        np.testing.assert_array_equal(np.asarray(image), np.asarray(expected))
        assert metadata["source_frame_count"] == 2


def test_mixed_formats_and_colliding_stems_get_distinct_stable_ids(tmp_path):
    for name in ["scan_1.png", "scan_1.jpg", "scan_2.bmp", "scan_10.tif"]:
        Image.new("RGB", (30, 20)).save(tmp_path / name)
    scans = discover_pages(tmp_path)
    assert len({scan.stem.casefold() for scan in scans}) == 4
    assert [scan.source.name for scan in scans][-2:] == ["scan_2.bmp", "scan_10.tif"]
    assert [scan.stem for scan in scans] == [scan.stem for scan in discover_pages(tmp_path)]
    assert select_pages(scans, "2-3") == scans[1:3]


def test_exif_orientation_is_applied_before_segmentation(tmp_path):
    source = tmp_path / "inputs"
    source.mkdir()
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (80, 40)).save(source / "rotated.jpg", exif=exif)
    destination = tmp_path / "page"
    destination.mkdir()
    image, metadata = normalize_scan(discover_pages(source)[0], destination)
    assert metadata["source_size"] == [80, 40]
    assert image.size == (40, 80)
    assert metadata["normalized_size"] == [40, 80]


def test_transparent_and_16bit_scans_have_explicit_rgb_normalization(tmp_path):
    source = tmp_path / "inputs"
    source.mkdir()
    Image.new("RGBA", (20, 15), (0, 0, 0, 0)).save(source / "alpha.png")
    values = np.array([[0, 32768, 65535]], dtype=np.uint16)
    Image.fromarray(values).save(source / "gray.tif")
    scans = discover_pages(source)
    for scan in scans:
        destination = tmp_path / scan.stem
        destination.mkdir()
        image, metadata = normalize_scan(scan, destination)
        assert metadata["normalization"]
        if scan.source.name == "alpha.png":
            assert np.all(np.asarray(image) == 255)
        else:
            np.testing.assert_array_equal(np.asarray(image)[0, :, 0], [0, 128, 255])


def test_cli_run_converts_tiff_and_jpeg_and_can_resume(tmp_path):
    source = tmp_path / "inputs"
    source.mkdir()
    first = Image.new("RGB", (140, 80), "white")
    second = Image.new("RGB", (150, 90), "ivory")
    first.save(source / "book.tiff", save_all=True, append_images=[second])
    first.save(source / "scan.jpg")
    overrides = tmp_path / "overrides.json"
    overrides.write_text(json.dumps({
        name: {"roi_fraction": [0.1, 0.1, 0.9, 0.9], "headlines_fraction": [0.4]}
        for name in ["book_page0001", "book_page0002", "scan"]
    }), encoding="utf-8")
    output = tmp_path / "output"
    args = Namespace(input_dir=source, output_dir=output, pages=None, overrides=overrides)
    assert run(args) == 0
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["selected_scans"] == summary["processed_scans"] == summary["draft_lines"] == 3
    with (output / "line_index.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["source_frame"] for row in rows] == ["1", "2", "1"]
    for row in rows:
        assert (output / row["normalized_png"]).exists()
        assert (output / row["image_path"]).exists()
    stamps = {path: path.stat().st_mtime_ns for path in output.rglob("*.png")}
    assert run(args) == 0
    assert {path: path.stat().st_mtime_ns for path in stamps} == stamps
    normalized = output / rows[0]["normalized_png"]
    Image.new("RGB", (140, 80), "black").save(normalized)
    assert run(args) == 1
    assert "changed" in json.loads((output / "summary.json").read_text(encoding="utf-8"))["errors"][0]["error"]


def test_bad_scan_is_reported_while_valid_inputs_finish(tmp_path):
    source = tmp_path / "inputs"
    source.mkdir()
    (source / "bad.tif").write_bytes(b"broken scan")
    Image.new("RGB", (120, 60), "white").save(source / "good.bmp")
    overrides = tmp_path / "overrides.json"
    overrides.write_text(json.dumps({"good": {"action": "review_only"}}), encoding="utf-8")
    output = tmp_path / "output"
    assert run(Namespace(input_dir=source, output_dir=output, pages=None, overrides=overrides)) == 1
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["selected_scans"] == 2
    assert summary["processed_scans"] == 1
    assert len(summary["errors"]) == 1
    assert (output / "pages/good/normalized.png").exists()


def test_bare_numeric_names_keep_suffix_page_selection():
    paths = [Path("0003.jpg"), Path("0007.tif")]
    assert select_pages(paths, "7") == [paths[1]]


def test_flexible_collection_overrides_apply_after_conversion(tmp_path):
    source = tmp_path / "inputs"
    source.mkdir()
    Image.new("RGB", (120, 60), "white").save(source / "EAP272_1_2_11_0001.JPG")
    overrides = tmp_path / "overrides.json"
    overrides.write_text(json.dumps({"EAP272-1-2-11_0001": {"action": "review_only"}}), encoding="utf-8")
    output = tmp_path / "output"
    assert run(Namespace(input_dir=source, output_dir=output, pages=None, overrides=overrides)) == 0
    page = json.loads((output / "pages/EAP272_1_2_11_0001/page.json").read_text(encoding="utf-8"))
    assert page["status"] == "review_only"
    assert page["normalized_png"] == "normalized.png"


def test_converted_page_publication_retries_a_transient_windows_lock(tmp_path, monkeypatch):
    source = tmp_path / "inputs"
    source.mkdir()
    Image.new("RGB", (120, 60), "white").save(source / "scan.jpg")
    overrides = tmp_path / "overrides.json"
    overrides.write_text(json.dumps({"scan": {"action": "review_only"}}), encoding="utf-8")
    original_rename = Path.rename
    attempts = []

    def retry_once(path, target):
        if path.name.startswith(".scan_"):
            attempts.append(path)
            if len(attempts) == 1:
                raise PermissionError("temporary Windows file lock")
        return original_rename(path, target)

    monkeypatch.setattr(Path, "rename", retry_once)
    output = tmp_path / "output"
    assert run(Namespace(input_dir=source, output_dir=output, pages=None, overrides=overrides)) == 0
    assert len(attempts) == 2
    assert (output / "pages/scan/normalized.png").exists()


def test_line_headlines_handles_negative_correlation_peaks_without_stop_iteration():
    # Profile with low periodic contrast where correlation peaks are <= 0
    profile = np.array([10, 10, 11, 10, 10, 11, 10, 10, 11, 10, 10, 10] * 5, dtype=float)
    positions, pitch = line_headlines(profile, 1000)
    assert positions
    assert pitch is not None


def test_uncertain_layout_page_falls_back_to_review_only(tmp_path):
    path = tmp_path / "blank_cover.png"
    # Uniform dark wooden cover
    Image.new("RGB", (120, 80), (30, 30, 30)).save(path)
    output = tmp_path / "page"
    output.mkdir()
    record = export_page(path, output, {}, {})
    assert record["status"] == "review_only"
    assert "Layout detection uncertain" in record["flags"][0]


@pytest.mark.parametrize("protect_ink", [False, True])
def test_two_leaf_manuscript_segments_both_leaves(tmp_path, protect_ink):
    path = tmp_path / "two_leaves.png"
    pixels = np.full((400, 600, 3), 25, dtype=np.uint8)
    # Top leaf (paper at 200, ink lines at 60)
    pixels[30:170, 80:520] = 200
    for y in [55, 77, 99, 121, 143]:
        for x in range(100, 500, 25):
            pixels[y:y+4, x:x+18] = 60
    # Bottom leaf
    pixels[230:370, 80:520] = 200
    for y in [255, 277, 299, 321, 343]:
        for x in range(100, 500, 25):
            pixels[y:y+4, x:x+18] = 60
    Image.fromarray(pixels).save(path)
    output = tmp_path / "page"
    output.mkdir()
    record = export_page(path, output, {"protect_line_ink": protect_ink}, {})
    assert record["status"] == "draft_segmented"
    assert len(record["lines"]) == 10
    assert [line["line_id"] for line in record["lines"]] == [
        f"two_leaves_line{i:02d}" for i in range(1, 11)
    ]

