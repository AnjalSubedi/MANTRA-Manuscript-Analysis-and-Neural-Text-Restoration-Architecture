# MANTRA Synthetic Image Generation Quality Report

## Executive Summary

- **Total Images Attempted**: 100000
- **Successfully Generated & Passed QA**: 100000 (100.00%)
- **Failed Quality Checks**: 0 (0.00%)

### Global Quality Metrics
- **Blank Images**: 0
- **Clipped Images (Failures)**: 0
- **Low Contrast Warnings**: 0
- **Unreadable Risk Warnings**: 0
- **ASCII Digits in Rendered Text**: 0
- **Average Readability Score**: 95.29 / 100.0

> [!IMPORTANT]
> **All generated images successfully passed quality control checks!** No clipping, blank pages, or unreadable dimensions detected.

## Dimensions Summary

- **Average Bounding Box**: 888.2 x 102.5
- **Width Range**: 78px to 3383px
- **Height Range**: 67px to 138px

## Difficulty Buckets Quality Summary

| Difficulty | Count | Ratio | Avg Width | Avg Height | Avg Readability | Blanks | Clipped | Low Contrast | Unreadable Risk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| easy | 41873 | 41.9% | 891.0px | 102.4px | 95.93 | 0 | 0 | 0 | 0 |
| medium | 40661 | 40.7% | 887.3px | 102.5px | 94.66 | 0 | 0 | 0 | 0 |
| hard | 17466 | 17.5% | 883.3px | 102.5px | 95.22 | 0 | 0 | 0 | 0 |

## Degradation Profile Quality Summary

| Profile | Count | Avg Width | Avg Height | Avg Readability | Blanks | Clipped | Low Contrast | Unreadable Risk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D0_clean_print | 5624 | 893.8px | 102.6px | 97.19 | 0 | 0 | 0 | 0 |
| D10_noisy_scan_crop | 2824 | 889.2px | 102.0px | 96.68 | 0 | 0 | 0 | 0 |
| D11_mild_warped_line | 2147 | 894.0px | 102.6px | 96.89 | 0 | 0 | 0 | 0 |
| D12_dark_thick_scan | 1995 | 884.1px | 102.8px | 97.30 | 0 | 0 | 0 | 0 |
| D13_washed_out_faint_manuscript | 1233 | 888.7px | 102.2px | 90.47 | 0 | 0 | 0 | 0 |
| D14_low_dpi_compressed_scan | 2501 | 880.5px | 102.7px | 92.55 | 0 | 0 | 0 | 0 |
| D15_bleedthrough_archival | 1276 | 887.9px | 103.0px | 97.19 | 0 | 0 | 0 | 0 |
| D16_ink_bleed_spread | 1239 | 877.4px | 102.5px | 95.03 | 0 | 0 | 0 | 0 |
| D17_local_stain_shadow | 1211 | 891.0px | 102.8px | 97.20 | 0 | 0 | 0 | 0 |
| D18_fold_crease_damage | 1016 | 894.5px | 102.8px | 96.85 | 0 | 0 | 0 | 0 |
| D19_weak_shirorekha_dropout | 743 | 882.6px | 101.7px | 95.06 | 0 | 0 | 0 | 0 |
| D1_light_scan | 6657 | 884.9px | 102.2px | 97.17 | 0 | 0 | 0 | 0 |
| D20_partial_erosion_thin_strokes | 765 | 892.0px | 103.2px | 95.95 | 0 | 0 | 0 | 0 |
| D21_mixed_manuscript_hard | 1039 | 876.5px | 102.5px | 87.78 | 0 | 0 | 0 | 0 |
| D22_reverse_showthrough_soft | 781 | 902.2px | 102.6px | 96.94 | 0 | 0 | 0 | 0 |
| D23_fragmented_scan_crop | 722 | 872.9px | 101.2px | 96.89 | 0 | 0 | 0 | 0 |
| D2_low_contrast_faded_ink | 5406 | 887.9px | 102.6px | 93.78 | 0 | 0 | 0 | 0 |
| D3_uneven_background | 4705 | 891.6px | 102.6px | 96.90 | 0 | 0 | 0 | 0 |
| D4_broken_strokes | 3495 | 867.6px | 102.0px | 96.98 | 0 | 0 | 0 | 0 |
| D5_noisy_archive_scan | 3504 | 887.8px | 102.6px | 88.75 | 0 | 0 | 0 | 0 |
| D6_blurred_scan | 3542 | 881.4px | 102.5px | 94.70 | 0 | 0 | 0 | 0 |
| D7_old_paper_texture | 3614 | 902.3px | 102.7px | 95.06 | 0 | 0 | 0 | 0 |
| D8_old_scan_soft | 4632 | 892.3px | 102.5px | 91.45 | 0 | 0 | 0 | 0 |
| D9_faded_gray_ink | 4142 | 897.3px | 102.9px | 92.22 | 0 | 0 | 0 | 0 |
| composed | 35187 | 887.9px | 102.4px | 96.08 | 0 | 0 | 0 | 0 |

## Dataset Distributions

### Difficulty Bucket Distribution
- **medium**: 40661 images (40.7%)
- **hard**: 17466 images (17.5%)
- **easy**: 41873 images (41.9%)

### Degradation Family Distribution
- **composed**: 35187 images (35.2%)
- **single**: 64813 images (64.8%)

### Split Distribution
- **train**: 79243 images (79.2%)
- **val**: 12649 images (12.6%)
- **test**: 8108 images (8.1%)

### Font Distribution
- **Mukta-Regular**: 8463 images (8.5%)
- **NotoSansDevanagari-Regular**: 9645 images (9.6%)
- **Hind-Regular**: 7924 images (7.9%)
- **Poppins-Regular**: 7993 images (8.0%)
- **NotoSerifDevanagari-Regular**: 8434 images (8.4%)
- **Rajdhani-Regular**: 8430 images (8.4%)
- **Jaldi-Regular**: 8367 images (8.4%)
- **Karma-Regular**: 8332 images (8.3%)
- **YatraOne-Regular**: 7859 images (7.9%)
- **Biryani-Regular**: 8423 images (8.4%)
- **RozhaOne-Regular**: 8241 images (8.2%)
- **Teko-Regular**: 7889 images (7.9%)

### Degradation Distribution
- **composed**: 35187 images (35.2%)
- **D20_partial_erosion_thin_strokes**: 765 images (0.8%)
- **D0_clean_print**: 5624 images (5.6%)
- **D5_noisy_archive_scan**: 3504 images (3.5%)
- **D10_noisy_scan_crop**: 2824 images (2.8%)
- **D18_fold_crease_damage**: 1016 images (1.0%)
- **D1_light_scan**: 6657 images (6.7%)
- **D6_blurred_scan**: 3542 images (3.5%)
- **D2_low_contrast_faded_ink**: 5406 images (5.4%)
- **D13_washed_out_faint_manuscript**: 1233 images (1.2%)
- **D7_old_paper_texture**: 3614 images (3.6%)
- **D9_faded_gray_ink**: 4142 images (4.1%)
- **D16_ink_bleed_spread**: 1239 images (1.2%)
- **D23_fragmented_scan_crop**: 722 images (0.7%)
- **D4_broken_strokes**: 3495 images (3.5%)
- **D17_local_stain_shadow**: 1211 images (1.2%)
- **D11_mild_warped_line**: 2147 images (2.1%)
- **D14_low_dpi_compressed_scan**: 2501 images (2.5%)
- **D19_weak_shirorekha_dropout**: 743 images (0.7%)
- **D3_uneven_background**: 4705 images (4.7%)
- **D8_old_scan_soft**: 4632 images (4.6%)
- **D15_bleedthrough_archival**: 1276 images (1.3%)
- **D12_dark_thick_scan**: 1995 images (2.0%)
- **D21_mixed_manuscript_hard**: 1039 images (1.0%)
- **D22_reverse_showthrough_soft**: 781 images (0.8%)

### Renderer Distribution
- **chromium**: 100000 images (100.0%)

## Recommended QA Samples for Manual Inspection

The following random samples have been chosen from each split for visual review:

### Train Samples
- **Line ID**: historical_nepali_001_000001 | **Font**: Mukta-Regular | **Degradation**: composed
  - Path: [`data/stage1_synthetic/images_core_merged/train/train_historical_nepali_001_000001_mukta_composed.png`](file:///D:/MANTRA/data/stage1_synthetic/images_core_merged/train/train_historical_nepali_001_000001_mukta_composed.png)
  - Text: `नेपालको इतिहासका स्रोतहरू`
- **Line ID**: historical_nepali_001_000002 | **Font**: NotoSansDevanagari-Regular | **Degradation**: D20_partial_erosion_thin_strokes
  - Path: [`data/stage1_synthetic/images_core_merged/train/train_historical_nepali_001_000002_notosans_D20.png`](file:///D:/MANTRA/data/stage1_synthetic/images_core_merged/train/train_historical_nepali_001_000002_notosans_D20.png)
  - Text: `विगत समयमा भएका घटनाहरूको तथ्य-सङ्गलन तथा तिनीहरूको विवेचना नै`
- **Line ID**: historical_nepali_001_000003 | **Font**: Hind-Regular | **Degradation**: composed
  - Path: [`data/stage1_synthetic/images_core_merged/train/train_historical_nepali_001_000003_hind_composed.png`](file:///D:/MANTRA/data/stage1_synthetic/images_core_merged/train/train_historical_nepali_001_000003_hind_composed.png)
  - Text: `इतिहास हो । निश्चित प्रमाण वा आधारविना इतिहास तयार पार्न सकिदैन । यस्ता`
- **Line ID**: historical_nepali_001_000004 | **Font**: Poppins-Regular | **Degradation**: D0_clean_print
  - Path: [`data/stage1_synthetic/images_core_merged/train/train_historical_nepali_001_000004_poppins_D0.png`](file:///D:/MANTRA/data/stage1_synthetic/images_core_merged/train/train_historical_nepali_001_000004_poppins_D0.png)
  - Text: `आधार वा स्रोतहरू विभिन्न प्रकारका हुन्छन्‌ । प्राप्त यिनै स्रोतहरूको राम्रोसँग अध्ययन`
- **Line ID**: historical_nepali_001_000005 | **Font**: NotoSerifDevanagari-Regular | **Degradation**: D5_noisy_archive_scan
  - Path: [`data/stage1_synthetic/images_core_merged/train/train_historical_nepali_001_000005_notoserif_D5.png`](file:///D:/MANTRA/data/stage1_synthetic/images_core_merged/train/train_historical_nepali_001_000005_notoserif_D5.png)
  - Text: `मनन गरेर मात्र इतिहासकारले विगत समयको सही चित्र प्रस्तुत गर्ने गर्दछ । त्यसैले :`

### Validation Samples
- **Line ID**: historical_nepali_003_000001 | **Font**: RozhaOne-Regular | **Degradation**: D2_low_contrast_faded_ink
  - Path: [`data/stage1_synthetic/images_core_merged/val/val_historical_nepali_003_000001_rozhaone_D2.png`](file:///D:/MANTRA/data/stage1_synthetic/images_core_merged/val/val_historical_nepali_003_000001_rozhaone_D2.png)
  - Text: `भाषावंशावली`
- **Line ID**: historical_nepali_003_000002 | **Font**: YatraOne-Regular | **Degradation**: composed
  - Path: [`data/stage1_synthetic/images_core_merged/val/val_historical_nepali_003_000002_yatraone_composed.png`](file:///D:/MANTRA/data/stage1_synthetic/images_core_merged/val/val_historical_nepali_003_000002_yatraone_composed.png)
  - Text: `श्रीणणेशाय नमः`
- **Line ID**: historical_nepali_003_000003 | **Font**: Jaldi-Regular | **Degradation**: D4_broken_strokes
  - Path: [`data/stage1_synthetic/images_core_merged/val/val_historical_nepali_003_000003_jaldi_D4.png`](file:///D:/MANTRA/data/stage1_synthetic/images_core_merged/val/val_historical_nepali_003_000003_jaldi_D4.png)
  - Text: `गुरु गणपति हुगी बटुक शिवमच्युतम्‌`
- **Line ID**: historical_nepali_003_000004 | **Font**: Hind-Regular | **Degradation**: composed
  - Path: [`data/stage1_synthetic/images_core_merged/val/val_historical_nepali_003_000004_hind_composed.png`](file:///D:/MANTRA/data/stage1_synthetic/images_core_merged/val/val_historical_nepali_003_000004_hind_composed.png)
  - Text: `ब्रह्माणं गिरिजा लक्ष्मौं वाणी बन्दै बिभूतये ॥`
- **Line ID**: historical_nepali_003_000005 | **Font**: Mukta-Regular | **Degradation**: composed
  - Path: [`data/stage1_synthetic/images_core_merged/val/val_historical_nepali_003_000005_mukta_composed.png`](file:///D:/MANTRA/data/stage1_synthetic/images_core_merged/val/val_historical_nepali_003_000005_mukta_composed.png)
  - Text: `अनाद्यायाखिलाद्याय मायिने गतमायिने`

### Test Samples
- **Line ID**: historical_nepali_005_000001 | **Font**: Teko-Regular | **Degradation**: D8_old_scan_soft
  - Path: [`data/stage1_synthetic/images_core_merged/test/test_historical_nepali_005_000001_teko_D8.png`](file:///D:/MANTRA/data/stage1_synthetic/images_core_merged/test/test_historical_nepali_005_000001_teko_D8.png)
  - Text: `नेपालमा पाशुपत सम्प्रदायः एक ग्रध्ययन`
- **Line ID**: historical_nepali_005_000002 | **Font**: Hind-Regular | **Degradation**: composed
  - Path: [`data/stage1_synthetic/images_core_merged/test/test_historical_nepali_005_000002_hind_composed.png`](file:///D:/MANTRA/data/stage1_synthetic/images_core_merged/test/test_historical_nepali_005_000002_hind_composed.png)
  - Text: `डा. जगदीश चन्द्र रेग्मी`
- **Line ID**: historical_nepali_005_000003 | **Font**: Mukta-Regular | **Degradation**: D7_old_paper_texture
  - Path: [`data/stage1_synthetic/images_core_merged/test/test_historical_nepali_005_000003_mukta_D7.png`](file:///D:/MANTRA/data/stage1_synthetic/images_core_merged/test/test_historical_nepali_005_000003_mukta_D7.png)
  - Text: `(प्रस्तुत लेख ति.वि. केन्द्रीय संस्कृति शिक्षण विभागद्वारा श्रायोजित`
- **Line ID**: historical_nepali_005_000004 | **Font**: RozhaOne-Regular | **Degradation**: D8_old_scan_soft
  - Path: [`data/stage1_synthetic/images_core_merged/test/test_historical_nepali_005_000004_rozhaone_D8.png`](file:///D:/MANTRA/data/stage1_synthetic/images_core_merged/test/test_historical_nepali_005_000004_rozhaone_D8.png)
  - Text: `गेष्ठीको लागि लेखिएको हो। यसको स्वीक्कति त्विवि शिक्षाध्यक्षद्वारा`
- **Line ID**: historical_nepali_005_000005 | **Font**: Jaldi-Regular | **Degradation**: composed
  - Path: [`data/stage1_synthetic/images_core_merged/test/test_historical_nepali_005_000005_jaldi_composed.png`](file:///D:/MANTRA/data/stage1_synthetic/images_core_merged/test/test_historical_nepali_005_000005_jaldi_composed.png)
  - Text: `गरिएको हुनाले उहाँ शिक्षाध्यक्षज्युमा श्राभार प्रकट गर्नु श्रावश्यक ठान्दछु ।`
