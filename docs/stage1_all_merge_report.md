# MANTRA Stage 1 Final Label Merge Report

## Executive Summary
- **QA Status**: **PASS**
- **Total Merged Rows (N)**: 49825
- **Total Duplicate Rows Removed**: 0

## Domain-Wise Counts
| Domain | Count |
| --- | --- |
| historical_nepali | 21792 |
| nepali_literary | 13757 |
| sanskrit | 7500 |
| coverage_admin | 1776 |
| coverage_numeral | 1500 |
| coverage_conjunct | 2000 |
| coverage_matra | 1500 |

## Split-Wise Counts
- **train**: 39483 (79.24%)
- **val**: 6308 (12.66%)
- **test**: 4034 (8.10%)

## Domain × Split Table
| Domain | Train | Val | Test | Total |
| --- | --- | --- | --- | --- |
| historical_nepali | 17925 | 2095 | 1772 | 21792 |
| nepali_literary | 10138 | 2785 | 834 | 13757 |
| sanskrit | 6000 | 750 | 750 | 7500 |
| coverage_admin | 1420 | 178 | 178 | 1776 |
| coverage_numeral | 1200 | 150 | 150 | 1500 |
| coverage_conjunct | 1600 | 200 | 200 | 2000 |
| coverage_matra | 1200 | 150 | 150 | 1500 |

## QA Validation Summary
- **Blank texts check**: Passed (0 found)
- **Duplicate line_id check**: Passed (0 found)
- **Duplicate text labels check**: Passed (0 found after deduplication)
- **Line length constraint check (5-160 chars)**: Passed
- **No ASCII digits [0-9] check**: Passed
- **No English letters check**: Passed
- **No bad symbols check**: Passed
