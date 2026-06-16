# MANTRA-Synth100k Dataset Schema

Each generated line image must have one metadata row.

## Required Columns

| Column | Description |
|---|---|
| id | Unique sample ID |
| image_path | Relative path to generated image |
| text | Ground-truth Devanagari label |
| source | Source ID or generated source name |
| language | nepali, sanskrit, hindi, mixed, generated |
| domain | prose, verse, religious, historical, administrative, coverage, names_dates |
| font | Font filename or font family |
| font_group | modern, book, sanskrit, nepali_publication, handwriting_like |
| font_size | Font size used for rendering |
| renderer | pango_cairo_harfbuzz or pil_raqm |
| degradation_profile | D0_clean_print, D1_light_scan, etc. |
| degradation_level | Integer from 0 to 4 |
| width | Image width in pixels |
| height | Image height in pixels |
| line_length | Number of Unicode characters in label |
| has_conjunct | true/false |
| has_digit | true/false |
| has_danda | true/false |
| split | train, val, or test |

## CSV Example

id,image_path,text,source,language,domain,font,font_group,font_size,renderer,degradation_profile,degradation_level,width,height,line_length,has_conjunct,has_digit,has_danda,split
synth_000001,images/train/synth_000001.png,श्री गणेशाय नमः,sanskrit_manual_001,sanskrit,religious,TiroDevanagariSanskrit.ttf,sanskrit,38,pango_cairo_harfbuzz,D1_light_scan,1,512,72,17,true,false,false,train

## Label Rules

1. Labels must exactly match the rendered text.
2. Do not remove matras, halant, anusvara, chandrabindu, visarga, danda, or digits.
3. Do not include Latin metadata in labels.
4. Avoid labels shorter than 5 characters except special title/date cases.
5. Avoid labels longer than 130 characters in the first version.
6. Preserve punctuation such as । and ॥.
