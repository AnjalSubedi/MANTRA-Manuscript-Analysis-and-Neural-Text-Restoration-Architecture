"""One font partition shared by generation, repairs, and manifest QA."""
from pathlib import Path

SPLITS = ("train", "val", "test")

# These bundled faces draw U+0966..U+096F with Latin numeral forms. They
# remain usable for text without digits; digit-bearing OCR labels need a
# face with distinguishable Devanagari numerals. See the release digit audit.
LATIN_FORM_DEVANAGARI_DIGIT_FONTS = frozenset({
    "Hind-Regular", "Poppins-Regular", "Rajdhani-Regular",
    "RozhaOne-Regular", "Teko-Regular",
})


def font_preserves_devanagari_digits(font_path_or_name, text):
    return not (
        Path(font_path_or_name).stem in LATIN_FORM_DEVANAGARI_DIGIT_FONTS
        and any("\u0966" <= char <= "\u096f" for char in text)
    )


def partition_fonts(font_paths, config):
    by_name = {Path(path).stem: str(path) for path in font_paths}
    if not by_name:
        raise ValueError("No fonts are available")
    policy = config.get("font_split", {})
    if not policy.get("use_font_aware_split", False):
        return {split: list(by_name.values()) for split in SPLITS}
    heldout = {}
    for split in ("val", "test"):
        names = policy.get(f"{split}_fonts", [])
        expected = policy.get(f"heldout_{split}_fonts", 2)
        if len(names) != expected or len(names) != len(set(names)):
            raise ValueError(f"font_split.{split}_fonts must name {expected} distinct fonts")
        missing = set(names) - by_name.keys()
        if missing:
            raise ValueError(f"Missing {split} fonts: {sorted(missing)}")
        heldout[split] = names
    if set(heldout["val"]) & set(heldout["test"]):
        raise ValueError("Validation and test fonts must be disjoint")
    train = sorted(set(by_name) - set(heldout["val"]) - set(heldout["test"]))
    if not train:
        raise ValueError("At least one training font is required")
    return {
        split: [by_name[name] for name in names]
        for split, names in {"train": train, **heldout}.items()
    }


def configured_font_splits(config):
    return partition_fonts(sorted(Path(config["paths"]["font_dir"]).glob("*.ttf")), config)


def font_is_allowed(row, fonts_by_split):
    font = row.get("actual_font_name") or row.get("font_name", "")
    return (font in {Path(path).stem for path in fonts_by_split.get(row.get("split"), [])}
            and font_preserves_devanagari_digits(font, row.get("text", "")))


def font_split_errors(rows, config):
    if not config.get("font_split", {}).get("use_font_aware_split", False):
        return [row for row in rows if not font_preserves_devanagari_digits(
            row.get("actual_font_name") or row.get("font_name", ""), row.get("text", "")
        )]
    fonts = configured_font_splits(config)
    return [row for row in rows if not font_is_allowed(row, fonts)]
