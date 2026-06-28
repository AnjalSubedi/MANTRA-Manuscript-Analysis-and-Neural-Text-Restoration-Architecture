# MANTRA Phase 3 Text Cleaning & Label Generation Summary

This report provides the detailed metrics for the targeted text cleaning and final CSV label generation.

## Overall Dataset Volume
- **Total Raw Lines (Input):** 35,697
- **Total Cleaned lines (in .clean.txt):** 22,726
- **Final Labeled Lines (in CSV):** 21,855

## Skip / Rejection Categories
- **Skipped Short Lines (< 5 chars):** 0
- **Skipped Long Lines (> 160 chars):** 0
- **Skipped OCR Junk Lines (ASCII/Scanner artifacts/Code fragments):** 12
- **Skipped Low Devanagari Ratio (< 0.50):** 0
- **Skipped Symbol-Heavy Lines (> 0.15):** 59
- **Skipped Frontmatter/Reference Lines:** 22
- **Skipped Lines with ASCII Digits:** 290
- **Skipped Noisy Header Lines:** 1
- **Skipped Fragment-Like Lines:** 40
- **Skipped Lines with OCR-Symbol Artifacts:** 336
- **Skipped Lines with Brackets/Guillemets:** 108
- **Removed Duplicate Lines (Global Deduplication):** 3

## Source breakdown
| Source ID | File Name | Raw Lines | Cleaned .TXT Lines | Final Label CSV Lines | Skipped Short | Skipped Long | Skipped OCR | Skipped Low Ratio | Skipped Symbol Heavy | Skipped Ref/Front | Skipped ASCII Digit | Skipped Noisy Header | Skipped Frag-Like | Skipped Bad Symbol | Skipped Bracket/Quote |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| historical_nepali_001 | nepal_ko_itihas_peshal_dahal.txt | 17,019 | 11,119 | 11,023 | 0 | 0 | 0 | 0 | 1 | 8 | 45 | 0 | 8 | 27 | 7 |
| historical_nepali_002 | devmala_vamshavali.txt | 5,168 | 3,319 | 3,120 | 0 | 0 | 1 | 0 | 1 | 0 | 44 | 0 | 10 | 124 | 19 |
| historical_nepali_003 | bhasha_vamshavali_1.txt | 4,470 | 2,319 | 2,110 | 0 | 0 | 11 | 0 | 9 | 0 | 67 | 1 | 9 | 93 | 19 |
| historical_nepali_004 | gorkha_vamshavali.txt | 5,317 | 3,884 | 3,803 | 0 | 0 | 0 | 0 | 4 | 0 | 12 | 0 | 0 | 35 | 30 |
| historical_nepali_005 | nepali_sanskriti_vishayak_agralekhharu.txt | 3,723 | 2,085 | 1,799 | 0 | 0 | 0 | 0 | 44 | 14 | 122 | 0 | 13 | 57 | 33 |

## Split Distribution
| Split | Line Count | Percentage |
|---|---|---|
| train | 17,946 | 82.11% |
| val | 2,110 | 9.65% |
| test | 1,799 | 8.23% |

## Feature Constraints & Metadata Counts
- **Lines with Single Danda (`।`):** 9,885 (45.23% of total)
- **Lines with Double Danda (`॥`):** 691 (3.16% of total)
- **Lines with Digits (Devanagari/Latin):** 3,117 (14.26% of total)
- **Lines with Conjuncts (Halant or complex):** 21,289 (97.41% of total)

## 20 Example Skipped Lines for Manual QA
The following lines from the cleaned text files were filtered out during final CSV label generation:

| File Name | Skipped Text Sample | Skip Reason |
|---|---|---|
| nepal_ko_itihas_peshal_dahal.clean.txt | `छन्‌ ।` | `fragment_like_too_few_letters` |
| nepal_ko_itihas_peshal_dahal.clean.txt | `» हेमराज शाक्य र तुलसी राम वैद्य, मेडिइभल नेपाल, पू. १० ।` | `brackets_or_guillemets` |
| nepal_ko_itihas_peshal_dahal.clean.txt | `बनावटको कारणले यो देश \| इतिहासको आरम्भदेखि नै शारणार्थीहरूको सुरक्षास्थल बन्न` | `bad_ocr_symbols` |
| nepal_ko_itihas_peshal_dahal.clean.txt | `नेपालको भौगोलिक स्थितिको अध्ययन गर्दा मुलुकको कल क्षेत्रफलमध्ये १५%` | `bad_ocr_symbols` |
| nepal_ko_itihas_peshal_dahal.clean.txt | `हिमाली प्रदेश, ६८% पहाडी प्रदेश र १७% तराई प्रदेश रहेको छ । विश्वको सर्वोच्च` | `bad_ocr_symbols` |
| nepal_ko_itihas_peshal_dahal.clean.txt | `मानव सभ्यताको पारम्भिक चरण, जहाँ मान्छे आफ्नो वानर (206) गुण छाडेर` | `contains_ascii_digit` |
| nepal_ko_itihas_peshal_dahal.clean.txt | `पुच्छ र अपुच्छको विभाजन गर्दा मानिस अपुच्छ (406) गणमा पर्ने र अपुच्छमा पनि` | `contains_ascii_digit` |
| nepal_ko_itihas_peshal_dahal.clean.txt | `मानवपूर्व युग (फ2-1पफा191 6100) भन्ने गरिएको छ । यसपछि देखिएका प्राणीहरू` | `contains_ascii_digit` |
| nepal_ko_itihas_peshal_dahal.clean.txt | `अवशेष पाइयो । यसको खोजी गर्ने एल्‌.एस्‌. लिके (6916) र उनकी पत्नी मेरी थिए ।` | `contains_ascii_digit` |
| nepal_ko_itihas_peshal_dahal.clean.txt | `आएर ढुङ्गाको बाहिरी भाग फलक (7196) को हाते बन्चरो (910 49:९6), एक छेउमा` | `contains_ascii_digit` |
| nepal_ko_itihas_peshal_dahal.clean.txt | `धार भएका ($106-5019]0615), चुच्चो परेको (॥%/15), तथा दुवैतिर धार भएका विभिन्न` | `contains_ascii_digit` |
| nepal_ko_itihas_peshal_dahal.clean.txt | `हतियारहरू बिना पालिसका चोट दिएर वा ताछेर (1191:60 ०1 070८) बनाइएका` | `contains_ascii_digit` |
| nepal_ko_itihas_peshal_dahal.clean.txt | `ढुङ्गे युग' (5101 982) भन्ने बित्तिकै मानिसले आफ्नो विकासक्रममा बिताएको` | `contains_ascii_digit` |
| nepal_ko_itihas_peshal_dahal.clean.txt | `भयो ।` | `fragment_like_too_few_letters` |
| nepal_ko_itihas_peshal_dahal.clean.txt | `छ भने लेप्चा भाषामा पवित्र देश वा गुफा (ने 5 पवित्र, पाल 5 देश वा गुफा) भन्ने` | `contains_ascii_digit` |
| nepal_ko_itihas_peshal_dahal.clean.txt | `क ज्ञानमणि नेपाल, नेपाल-निरुक्त, पृष्ठ १३-१७ ।` | `reference_or_frontmatter_fragment` |
| nepal_ko_itihas_peshal_dahal.clean.txt | `भए ।”` | `fragment_like_too_few_letters` |
| nepal_ko_itihas_peshal_dahal.clean.txt | `गरेको छ। \|` | `bad_ocr_symbols` |
| nepal_ko_itihas_peshal_dahal.clean.txt | `भण्डारलाई दुई भागमा बाँडिएको छ - पूर्व वैदिककाल (2917 ५९०८ 86) र उत्तर` | `contains_ascii_digit` |
| nepal_ko_itihas_peshal_dahal.clean.txt | `वैदिककाल (9161 ५९०८ 4६९) । सामान्यतया क्रग्बेदमा वर्णित समाजलाई पूर्व वैदिक` | `contains_ascii_digit` |

## Top 50 Devanagari Characters by Frequency
| Rank | Character | Hex Code | Occurrence Count | Description |
|---|---|---|---|---|
| 1 | `ा` | `U+093E` | 125,799 | Vowel Sign AA (ा) |
| 2 | `्` | `U+094D` | 98,605 | Sign Virama / Halant (्) |
| 3 | `र` | `U+0930` | 84,734 | Consonant RA (र) |
| 4 | `न` | `U+0928` | 64,946 | Consonant NA (न) |
| 5 | `क` | `U+0915` | 57,045 | Consonant KA (क) |
| 6 | `ि` | `U+093F` | 56,645 | Vowel Sign I (ि) |
| 7 | `म` | `U+092E` | 43,291 | Consonant MA (म) |
| 8 | `त` | `U+0924` | 43,124 | Consonant TA (त) |
| 9 | `ल` | `U+0932` | 40,922 | Consonant LA (ल) |
| 10 | `े` | `U+0947` | 38,013 | Vowel Sign E (े) |
| 11 | `ो` | `U+094B` | 35,565 | Vowel Sign O (ो) |
| 12 | `स` | `U+0938` | 35,430 | Consonant SA (स) |
| 13 | `प` | `U+092A` | 34,960 | Consonant PA (प) |
| 14 | `य` | `U+092F` | 33,642 | Consonant YA (य) |
| 15 | `व` | `U+0935` | 25,929 | Consonant VA (व) |
| 16 | `ह` | `U+0939` | 25,910 | Consonant HA (ह) |
| 17 | `ु` | `U+0941` | 25,556 | Vowel Sign U (ु) |
| 18 | `द` | `U+0926` | 25,166 | Consonant DA (द) |
| 19 | `ी` | `U+0940` | 23,759 | Vowel Sign II (ी) |
| 20 | `ग` | `U+0917` | 23,754 | Consonant GA (ग) |
| 21 | `भ` | `U+092D` | 17,194 | Consonant BHA (भ) |
| 22 | `ज` | `U+091C` | 16,252 | Consonant JA (ज) |
| 23 | `ब` | `U+092C` | 16,005 | Consonant BA (ब) |
| 24 | `श` | `U+0936` | 15,045 | Consonant SHA (श) |
| 25 | `।` | `U+0964` | 11,216 | Danda (।) |
| 26 | `थ` | `U+0925` | 11,039 | Consonant THA (थ) |
| 27 | `ए` | `U+090F` | 9,771 | Vowel E (ए) |
| 28 | `ै` | `U+0948` | 9,709 | Vowel Sign AI (ै) |
| 29 | `ू` | `U+0942` | 9,412 | Vowel Sign UU (ू) |
| 30 | `छ` | `U+091B` | 9,335 | Consonant CHA (छ) |
| 31 | `च` | `U+091A` | 8,456 | Consonant CA (च) |
| 32 | `उ` | `U+0909` | 7,570 | Vowel U (उ) |
| 33 | `ध` | `U+0927` | 7,247 | Consonant DHA (ध) |
| 34 | `ष` | `U+0937` | 7,030 | Consonant SSA (ष) |
| 35 | `ट` | `U+091F` | 6,996 | Consonant TTA (ट) |
| 36 | `ण` | `U+0923` | 6,847 | Consonant NNA (ण) |
| 37 | `ख` | `U+0916` | 6,685 | Consonant KHA (ख) |
| 38 | `ं` | `U+0902` | 6,655 | Sign Anusvara (ं) |
| 39 | `ई` | `U+0908` | 5,956 | Vowel II (ई) |
| 40 | `अ` | `U+0905` | 5,418 | Vowel A (अ) |
| 41 | `ँ` | `U+0901` | 4,999 | Sign Candrabindu (ँ) |
| 42 | `आ` | `U+0906` | 4,980 | Vowel AA (आ) |
| 43 | `ड` | `U+0921` | 3,898 | Consonant DDA (ड) |
| 44 | `इ` | `U+0907` | 3,624 | Vowel I (इ) |
| 45 | `फ` | `U+092B` | 3,423 | Consonant PHA (फ) |
| 46 | `ठ` | `U+0920` | 2,871 | Consonant TTHA (ठ) |
| 47 | `ौ` | `U+094C` | 2,532 | Vowel Sign AU (ौ) |
| 48 | `ृ` | `U+0943` | 2,415 | Vowel Sign Vocalic R (ृ) |
| 49 | `१` | `U+0967` | 2,078 | Devanagari character |
| 50 | `घ` | `U+0918` | 1,950 | Consonant GHA (घ) |
