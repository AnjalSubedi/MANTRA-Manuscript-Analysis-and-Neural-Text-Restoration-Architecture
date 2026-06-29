# -*- coding: utf-8 -*-
"""
MANTRA Stage 2.3 — Collate Function for HTR Batches.

Pads variable-width line-image tensors to the maximum width in each
batch and concatenates CTC labels into a flat tensor, ready for
``torch.nn.CTCLoss``.
"""
import torch


def htr_collate_fn(batch: list[dict], pad_value: float = 1.0) -> dict:
    """Collate a list of HTRLineDataset samples into a single batch dict.

    Padding strategy
    ~~~~~~~~~~~~~~~~
    * Images are right-padded to the **maximum width in the batch** using
      ``pad_value``.  Default is 1.0 (white background when
      ``invert_image=False``).  Set to 0.0 when ``invert_image=True``.
    * Aspect ratio is never changed — padding is additive, not a resize.
    * Labels are **flat-concatenated** (no padding) as required by
      ``torch.nn.CTCLoss(reduction='mean')``.

    Parameters
    ----------
    batch : list[dict]
        Individual sample dicts from ``HTRLineDataset.__getitem__``.
    pad_value : float
        Value used for right-padding images.  Should match the
        background intensity after preprocessing:
        * ``1.0`` when ``invert_image=False`` (white bg, dark text).
        * ``0.0`` when ``invert_image=True``  (black bg, white text).

    Returns
    -------
    dict with keys:
        images            – ``[B, 1, H, max_W]`` float32
        labels            – ``[sum(label_lengths)]`` int64 (flat)
        label_lengths     – ``[B]`` int64
        processed_widths  – ``[B]`` int64  (actual image widths before pad)
        texts             – list[str]
        image_paths       – list[str]
        line_ids          – list[str]
        domains           – list[str]
        degradation_profiles – list[str]
        difficulty_buckets   – list[str]
    """
    # Determine max width in this batch
    max_w = max(s["image"].shape[2] for s in batch)
    height = batch[0]["image"].shape[1]  # all are target_height (64)

    images = []
    all_labels: list[torch.Tensor] = []
    label_lengths: list[int] = []
    processed_widths: list[int] = []

    texts: list[str] = []
    image_paths: list[str] = []
    line_ids: list[str] = []
    domains: list[str] = []
    degradation_profiles: list[str] = []
    difficulty_buckets: list[str] = []

    for s in batch:
        img = s["image"]  # [1, H, w_i]
        w_i = img.shape[2]

        # Right-pad to max_w
        if w_i < max_w:
            pad_tensor = torch.full(
                (1, height, max_w - w_i), pad_value, dtype=img.dtype
            )
            img = torch.cat([img, pad_tensor], dim=2)

        images.append(img)

        all_labels.append(s["label"])
        label_lengths.append(s["label_length"])
        processed_widths.append(s["processed_width"])

        texts.append(s["text"])
        image_paths.append(s["image_path"])
        line_ids.append(s.get("line_id", ""))
        domains.append(s.get("domain", ""))
        degradation_profiles.append(s.get("degradation_profile", ""))
        difficulty_buckets.append(s.get("difficulty_bucket", ""))

    return {
        "images": torch.stack(images, dim=0),               # [B,1,H,max_W]
        "labels": torch.cat(all_labels, dim=0),              # [sum(L_i)]
        "label_lengths": torch.tensor(label_lengths, dtype=torch.long),
        "processed_widths": torch.tensor(processed_widths, dtype=torch.long),
        "texts": texts,
        "image_paths": image_paths,
        "line_ids": line_ids,
        "domains": domains,
        "degradation_profiles": degradation_profiles,
        "difficulty_buckets": difficulty_buckets,
    }
