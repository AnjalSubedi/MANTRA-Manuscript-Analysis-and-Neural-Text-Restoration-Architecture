"""Regression checks for source geometry and preservation of close line marks."""

from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from manuscripts.segment_lines import crop_polygon, detection_gray, gap_path, largest_band, line_headlines, select_pages


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
