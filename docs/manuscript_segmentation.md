# Manuscript line segmentation

The segmentation command proposes line boundaries for Devanagari manuscript pages. It exports cropped PNGs, source coordinates, numbered page previews, contact sheets, and a line index. Pixels inside each proposed polygon retain their resolution, and the crop uses white outside the polygon.

## Inputs and coordinates

The command accepts PNG, JPEG, TIFF, BMP, WebP, and JPEG 2000, including uppercase extensions. It expands multipage scans in frame order. It stores converted inputs as `normalized.png` in each page's output directory and reuses ordinary PNGs directly. The page metadata and line index record source paths, checksums, frames, and normalization details. Same-name scans with different extensions get distinct page IDs.

Normalization applies EXIF orientation, composites transparency on white, and converts to 8-bit RGB. For 16-bit grayscale scans it scales the full 0–65535 range. Exported coordinates refer to the normalized image.

## Segment a collection

For locally acquired EAP272/1/2/11 scans named `EAP272-1-2-11_0001.png` through `EAP272-1-2-11_0129.png`, use the collection's layout overrides.

```bash
python src/manuscripts/segment_lines.py \
  --input-dir outputs/manuscript_downloads/EAP272-1-2-11/images \
  --output-dir outputs/manuscript_segmentation/EAP272-1-2-11/segmentation_v3 \
  --overrides configs/manuscripts/eap272_1_2_11.json
```

For a smaller selection, add `--pages 2-5,20,129` and use its own output directory. Unique numeric suffixes supply page numbers. Other filename patterns use positions in the expanded list, sorted by natural filename order and frame number. Repeating a run verifies source and output hashes before reusing completed pages. Use a fresh output directory for changed inputs, settings, or code.

Review proposed crops before pairing them with transcriptions. Flags mark possible ink intersections or irregular spacing. This collection's configuration leaves its handwritten cover for layout review and supplies the final page's single-line region. Review marginal notes and folio numbers separately. [Source and access notes](research/manuscript_dataset_sources.md)
