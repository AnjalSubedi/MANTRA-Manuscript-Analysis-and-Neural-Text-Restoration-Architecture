# -*- coding: utf-8 -*-
"""
MANTRA Stage 2.3 — PyTorch Dataset for HTR Line Images.

Wraps an HTR manifest CSV and charset JSON into a map-style
``torch.utils.data.Dataset`` that yields preprocessed image tensors,
encoded CTC labels, and rich per-sample metadata.
"""
import csv
import json
import logging
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset

from stage2_htr.transforms import preprocess_image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text encoding / decoding helpers (shared with build_charset.py interface)
# ---------------------------------------------------------------------------

def encode_text(text: str, char_to_id: dict[str, int]) -> list[int]:
    """Encode a label string into a list of character class IDs.

    Only real label characters are encoded.  ``<BLANK>`` (CTC blank) must
    never appear in a target label.

    Raises:
        ValueError: if *any* character in ``text`` is not present in
            ``char_to_id``.  This is intentional — the charset was built
            from the full 100 k corpus, so every character must be
            encodable.
    """
    ids: list[int] = []
    for ch in text:
        if ch not in char_to_id:
            raise ValueError(
                f"Character {ch!r} (U+{ord(ch):04X}) not in charset."
            )
        cid = char_to_id[ch]
        if cid == 0:
            raise ValueError(
                "encode_text must never encode the <BLANK> token."
            )
        ids.append(cid)
    return ids


def decode_ids(
    ids: list[int],
    id_to_char: dict[int, str],
    remove_blank: bool = True,
) -> str:
    """Decode a list of class IDs back to a string.

    During CTC decoding, ``remove_blank=True`` strips any occurrences of
    the blank token (id 0).
    """
    chars: list[str] = []
    for idx in ids:
        ch = id_to_char.get(idx) or id_to_char.get(str(idx))
        if ch is None:
            raise ValueError(f"ID {idx} not found in id_to_char.")
        if ch == "<BLANK>" and remove_blank:
            continue
        chars.append(ch)
    return "".join(chars)


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class HTRLineDataset(Dataset):
    """Map-style dataset that pairs line images with CTC label encodings.

    Parameters
    ----------
    manifest_csv : str
        Path to an HTR manifest CSV (``htr_train.csv``, etc.).
    charset_json : str
        Path to ``charset_synth100k.json``.
    image_height : int
        Fixed target height for every image (default 64).
    max_width : int | None
        Optional width limit.  Images wider than this after height-based
        resize are **skipped** (never cropped or squeezed).
    augment : bool
        Reserved for future augmentation (Phase 2.3 keeps this False).
    return_metadata : bool
        If True (default), each sample dict includes rich metadata fields
        (domain, font_name, degradation_profile, …).
    invert_image : bool
        Invert pixels so text is bright and the background is dark.
    project_root : str | None
        Base for project-relative image paths; defaults to the repository root.
    encode_labels : bool
        Encode CTC targets by default. Inference can keep references as text,
        including characters outside the model vocabulary.

    Width-filtered rows are available in ``skipped_rows``. ``source_row_count``
    includes them, while ``len(dataset)`` counts only loadable-width rows.
    """

    def __init__(
        self,
        manifest_csv: str,
        charset_json: str,
        image_height: int = 64,
        max_width: int | None = None,
        augment: bool = False,
        return_metadata: bool = True,
        invert_image: bool = False,
        project_root: str | None = None,
        encode_labels: bool = True,
    ):
        super().__init__()
        self.image_height = image_height
        self.max_width = max_width
        self.augment = augment
        self.return_metadata = return_metadata
        self.invert_image = invert_image
        self.encode_labels = encode_labels
        self.project_root = Path(project_root) if project_root else Path(__file__).resolve().parents[2]
        if image_height <= 0 or (max_width is not None and max_width <= 0):
            raise ValueError("image_height and max_width must be positive")
        if augment:
            raise ValueError("Augmentation is not implemented in Stage 2.3")

        # Load charset -------------------------------------------------
        with open(charset_json, "r", encoding="utf-8") as f:
            charset = json.load(f)
        self.char_to_id: dict[str, int] = charset["char_to_id"]
        # id_to_char keys may be strings in JSON — normalise to int keys
        self.id_to_char: dict[int, str] = {
            int(k): v for k, v in charset["id_to_char"].items()
        }
        self.num_classes: int = charset["num_classes"]

        # Load manifest rows -------------------------------------------
        self.rows: list[dict[str, str]] = []
        self.skipped_rows: list[dict] = []
        self.source_row_count = 0
        with open(manifest_csv, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.source_row_count += 1
                if max_width is not None:
                    try:
                        # Check actual image dimensions rather than trusting
                        # possibly stale metadata. No pixel resize is needed.
                        with Image.open(self.resolve_image_path(row["image_path"])) as img:
                            width = max(1, round(img.width * (image_height / img.height)))
                            if width > max_width:
                                # A readable header must not hide a damaged
                                # over-wide file from the inspection report.
                                img.load()
                        if width > max_width:
                            self.skipped_rows.append({
                                "image_path": row["image_path"],
                                "processed_width": width,
                            })
                            continue
                    except (OSError, ValueError):
                        # Keep unreadable files so inspection can report their
                        # load failure rather than hide them as width skips.
                        pass
                self.rows.append(row)

        logger.info(
            "HTRLineDataset: loaded %d rows from %s  (height=%d, max_width=%s)",
            len(self.rows), manifest_csv, image_height, max_width,
        )

    # ---- public helpers exposed for external use ----

    def resolve_image_path(self, path: str) -> Path:
        return self.project_root / path

    def encode(self, text: str) -> list[int]:
        """Encode text using this dataset's charset."""
        return encode_text(text, self.char_to_id)

    def decode(self, ids: list[int], remove_blank: bool = True) -> str:
        """Decode IDs using this dataset's charset."""
        return decode_ids(ids, self.id_to_char, remove_blank=remove_blank)

    # ---- Dataset interface ----

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        image_path = row["image_path"]
        text = row["text"]

        # Image preprocessing -------------------------------------------
        tensor, original_size, processed_size = preprocess_image(
            path=self.resolve_image_path(image_path),
            target_height=self.image_height,
            max_width=self.max_width,
            invert=self.invert_image,
        )

        if tensor is None:
            raise RuntimeError(
                f"Image dimensions changed after width filtering: {image_path}"
            )

        # Label encoding ------------------------------------------------
        label_ids = encode_text(text, self.char_to_id) if self.encode_labels else []
        label_tensor = torch.tensor(label_ids, dtype=torch.long)

        sample = {
            "image": tensor,                        # [1, H, W]
            "label": label_tensor,                  # [L]
            "label_length": len(label_ids),
            "text": text,
            "image_path": image_path,
            "line_id": row.get("line_id", ""),
            "split": row.get("split", ""),
            "width": int(original_size[0]),
            "height": int(original_size[1]),
            "processed_width": int(processed_size[0]),
            "processed_height": int(processed_size[1]),
        }

        if self.return_metadata:
            sample["domain"] = row.get("domain", "")
            sample["difficulty_bucket"] = row.get("difficulty_bucket", "")
            sample["degradation_profile"] = row.get("degradation_profile", "")
            sample["font_name"] = row.get("font_name", "")

        return sample
