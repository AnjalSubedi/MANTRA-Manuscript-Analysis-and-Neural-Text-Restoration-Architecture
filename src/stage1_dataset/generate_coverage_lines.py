#!/usr/bin/env python3
"""
MANTRA Stage 1 Coverage-Line Generation Pipeline

Generates four controlled Devanagari coverage label CSVs.
Usage:  python src/stage1_dataset/generate_coverage_lines.py
"""

import os, csv, re, random, collections, sys, io

# Fix Windows console encoding for Devanagari output
if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

SEED = 42
LABEL_DIR = "data/stage1_synthetic/labels"
REPORT_DIR = "data/stage1_synthetic/reports"
CORE_MERGED_PATH = os.path.join(LABEL_DIR, "stage1_core_merged_lines.csv")
os.makedirs(LABEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

CSV_FIELDNAMES = [
    "line_id", "source_id", "source_file", "text", "line_length",
    "has_danda", "has_double_danda", "has_digit", "has_conjunct", "split",
]
BAD_SYMBOLS = set("_|&%\u20ac$\u00a3\u20b9\u00ab\u00bb[]")

TARGET_CONJUNCTS = [
    "\u0915\u094d\u0937", "\u0924\u094d\u0930", "\u091c\u094d\u091e",
    "\u0936\u094d\u0930", "\u0926\u094d\u0927", "\u0926\u094d\u0935",
    "\u0915\u094d\u0924", "\u0928\u094d\u0924\u094d\u0930",
    "\u0938\u094d\u0924\u094d\u0930", "\u0939\u094d\u0928",
    "\u0939\u094d\u092e", "\u0926\u094d\u092f", "\u0930\u094d\u092f",
    "\u0915\u094d\u0930", "\u0917\u094d\u0930", "\u092a\u094d\u0930",
    "\u092c\u094d\u0930", "\u092d\u094d\u0930", "\u092e\u094d\u0930",
    "\u0915\u094d\u0932", "\u0917\u094d\u0932", "\u0936\u094d\u0932",
]
TARGET_MATRAS = list("\u093f\u0940\u0941\u0942\u0947\u0948\u094b\u094c\u0902\u0903\u0901\u0943")
DEVANAGARI_DIGITS = list("\u0966\u0967\u0968\u0969\u096a\u096b\u096c\u096d\u096e\u096f")

CONJUNCTS_FOR_DETECTION = [
    "\u0915\u094d\u0937", "\u0924\u094d\u0930", "\u091c\u094d\u091e",
    "\u0936\u094d\u0930", "\u0926\u094d\u0927", "\u0926\u094d\u0935",
    "\u0926\u094d\u092f", "\u0939\u094d\u092e", "\u0938\u094d\u0924\u094d\u0930",
    "\u092a\u094d\u0930", "\u092c\u094d\u0930", "\u092d\u094d\u0930",
    "\u0917\u094d\u0930", "\u0915\u094d\u0930", "\u0928\u094d\u0924\u094d\u0930",
    "\u0939\u094d\u0928", "\u092e\u094d\u0930", "\u0915\u094d\u0932",
    "\u0917\u094d\u0932", "\u0936\u094d\u0932", "\u0930\u094d\u092f",
    "\u0915\u094d\u0924",
]

# ====================================================================
# Metadata & validation
# ====================================================================

def recompute_metadata(text):
    ll = len(text)
    hd = "true" if "\u0964" in text else "false"
    hdd = "true" if "\u0965" in text else "false"
    hdg = "true" if any("\u0966" <= c <= "\u096F" for c in text) else "false"
    hc = "false"
    if "\u094D" in text:
        hc = "true"
    else:
        for conj in CONJUNCTS_FOR_DETECTION:
            if conj in text:
                hc = "true"
                break
    return ll, hd, hdd, hdg, hc

def find_conjuncts_in_text(text):
    return {c for c in TARGET_CONJUNCTS if c in text}

def find_matras_in_text(text):
    return {m for m in TARGET_MATRAS if m in text}

def find_digits_in_text(text):
    return {c for c in text if "\u0966" <= c <= "\u096F"}

def validate_text(text):
    if not text or not text.strip(): return False, "blank"
    if len(text) < 5: return False, "too_short"
    if len(text) > 160: return False, "too_long"
    if re.search(r"[0-9]", text): return False, "ascii_digit"
    if re.search(r"[a-zA-Z]", text): return False, "latin_letter"
    for ch in text:
        if ch in BAD_SYMBOLS: return False, "bad_symbol"
    if not any("\u0900" <= c <= "\u097F" for c in text): return False, "no_devanagari"
    return True, "ok"

# ====================================================================
# Template expansion engine
# ====================================================================

def _fill(template, slots, rng):
    result = template
    for sn, opts in slots.items():
        ph = "{" + sn + "}"
        while ph in result:
            result = result.replace(ph, rng.choice(opts), 1)
    return result

def _postprocess(text):
    """Fix known linguistic issues in generated text."""
    # Fix reversed year ranges: YEAR1 must <= YEAR2
    m = re.search(r'(वि\. सं )(\S+)( देखि )(\S+)( सम्म)', text)
    if m:
        y1, y2 = m.group(2), m.group(4)
        if y1 > y2:
            text = text[:m.start(2)] + y2 + m.group(3) + y1 + text[m.end(4):]
    return text

def _generate_lines(templates, target_count, template_cap, seen_texts, rng,
                    coverage_targets=None, coverage_threshold=0, coverage_fn=None):
    lines = []
    t_usage = collections.Counter()
    cov = collections.Counter()

    # Pre-compute which templates can produce which targets
    t_for_target = collections.defaultdict(list)
    if coverage_targets and coverage_fn:
        for tidx, (tmpl, slots) in enumerate(templates):
            frags = [tmpl]
            for opts in slots.values():
                frags.extend(opts)
            combined = " ".join(frags)
            for t in coverage_targets:
                if t in combined:
                    t_for_target[t].append(tidx)

    max_att = target_count * 30
    for _ in range(max_att):
        if len(lines) >= target_count:
            break
        avail = [i for i in range(len(templates)) if t_usage[i] < template_cap]
        if not avail:
            break

        tidx = None
        if coverage_targets and coverage_fn:
            under = [(t, cov[t]) for t in coverage_targets if cov[t] < coverage_threshold]
            if under:
                under.sort(key=lambda x: x[1])
                tgt = under[0][0]
                cands = [i for i in t_for_target.get(tgt, []) if i in set(avail)]
                if cands:
                    tidx = rng.choice(cands)
        if tidx is None:
            tidx = rng.choice(avail)

        tmpl, slots = templates[tidx]
        text = _fill(tmpl, slots, rng) if slots else tmpl
        text = _postprocess(text)

        ok, _ = validate_text(text)
        if not ok or text in seen_texts:
            continue

        seen_texts.add(text)
        t_usage[tidx] += 1
        if coverage_fn:
            for t in coverage_fn(text):
                cov[t] += 1
        lines.append((text, tidx))

    return lines, t_usage, cov

# ====================================================================
# WORD BANKS  (large pools for combinatorial capacity)
# ====================================================================

# ----- Conjunct domain word banks -----
_CJ_AGENTS = [
    "क्षत्रिय राजा", "विद्वान् ब्राह्मण", "ज्ञानी पण्डित", "श्रेष्ठ आचार्य",
    "भक्त पुजारी", "प्रधान सेनापति", "वृद्ध गुरु", "सम्राट",
    "चक्रवर्ती नरेश", "ग्रामवासी", "विद्यार्थी", "प्रज्ञावान् मुनि",
    "दक्ष शिल्पकार", "नम्र सेवक", "शक्तिशाली योद्धा", "मन्त्री",
    "विद्वान् ऋषि", "ब्रह्मचारी", "क्रान्तिकारी", "प्रसिद्ध कवि",
    "अग्रणी नेता", "भ्रमणकर्ता", "विक्रमी सैनिक", "सूर्यवंशी राजा",
    "क्षमाशील गुरु", "प्रतापी क्षत्रिय", "शुद्ध हृदयी भक्त",
    "स्त्री", "मुक्तिकामी प्रजा", "कार्यकर्ता",
]
_CJ_OBJECTS = [
    "रक्षा", "शिक्षा", "विद्या", "शक्ति", "भक्ति", "मुक्ति", "बुद्धि",
    "सिद्धि", "ज्ञान", "प्रज्ञा", "श्रद्धा", "धैर्य", "पराक्रम",
    "प्रमाण", "ग्रन्थ", "शास्त्र", "मन्त्र", "तन्त्र", "क्षेत्र",
    "चरित्र", "सूत्र", "पत्र", "वस्त्र", "अस्त्र", "श्लोक",
    "क्लेश", "आश्रय", "कार्य", "विक्रम", "उद्योग", "सिद्धान्त",
    "प्रगति", "आक्रमण", "चिह्न", "ब्रह्मविद्या", "ग्लानि", "प्रभाव",
]
_CJ_VERBS = [
    "अध्ययन", "प्रचार", "व्याख्या", "प्रसार", "रक्षा",
    "प्रबन्ध", "व्यवस्था", "प्रयोग", "प्रशंसा",
    "उद्धार", "प्रवचन", "सम्मान", "परिवर्तन", "विस्तार", "संरक्षण",
]
_CJ_PLACES = [
    "क्षेत्र", "आश्रम", "विद्यालय", "ग्राम", "भक्तपुर", "प्रदेश",
    "द्वार", "मन्दिर", "राजदरबार", "गोरखा", "काठमाडौँ", "ललितपुर",
    "पाटन", "तीर्थस्थल", "राज्य", "देवालय", "उद्यान", "किर्तिपुर",
    "लुम्बिनी", "स्वयम्भू",
]
_CJ_DOCS = [
    "ग्रन्थ", "शिलालेख", "ताम्रपत्र", "वंशावली", "इतिहास",
    "धर्मशास्त्र", "पुराण", "अभिलेख", "श्लोक", "सूत्र", "वेद",
    "उपनिषद्", "स्मृति", "पत्र", "लिपि",
]
_CJ_TIMES = [
    "प्राचीन काल", "मध्यकाल", "पुरानो समय", "विक्रम संवत्",
    "लिच्छविकाल", "मल्लकाल", "त्यस युग", "शाहकाल",
    "किरातकाल", "ठकुरी वंशको समय",
]
_CJ_ENDINGS = ["गरे ।", "गर्नुभयो ।", "गरेको थियो ।", "भयो ।"]

# ----- Matra domain word banks -----
_MT_SUBJ = [
    "विद्यार्थीहरू", "शिक्षकहरू", "बालबालिकाहरू", "गुरुहरू", "साधकहरू",
    "कृषकहरू", "कलाकारहरू", "नागरिकहरू", "जनता", "विद्वान्हरू",
    "कवि", "महिला", "गायिका", "चित्रकार", "मूर्तिकार",
]
_MT_VERB = [
    "अध्ययन गरे", "लेखे", "पढे", "सुने", "बुझे", "सिके",
    "गाए", "भने", "रचना गरे", "प्रदर्शन गरे",
]
_MT_OBJ_BOOKS = [
    "पुस्तक", "ग्रन्थ", "पाठ", "श्लोक", "कविता", "गीत",
    "कथा", "उपन्यास", "निबन्ध", "नाटक",
]
_MT_VISARGA_WORDS = [
    "दुःख", "सुखः", "नमः", "प्रातः", "अन्ततः", "क्रमशः",
    "विशेषतः", "मूलतः", "स्वतः", "प्रायः", "पुनः", "अतः",
]
_MT_VISARGA_ADV = [
    "अन्ततः", "क्रमशः", "विशेषतः", "मूलतः",
    "स्वतः", "प्रायः", "पुनः", "अतः",
]
_MT_RI_WORDS = [
    "कृषि", "कृपा", "पृथ्वी", "मृत्यु", "हृदय", "नृत्य",
    "वृक्ष", "गृह", "सृष्टि", "संस्कृत", "अमृत", "संस्कृति",
    "तृप्त", "कृत", "कृष्ण", "कृति", "भृत्य", "श्रृङ्गार",
]
_MT_CHAND_WORDS = [
    "गाउँ", "काठमाडौँ", "दशैँ", "आँखा", "साँझ",
    "चाँदनी", "छाँया", "ढुँगा", "बाँसबारी", "गाउँघर",
    "हाँडीगाउँ", "चाँगुनारायण",
]
_MT_PLACES = [
    "पोखरा", "विराटनगर", "जनकपुर", "बुटवल", "धनगढी",
    "भरतपुर", "काठमाडौँ", "ललितपुर", "भक्तपुर", "बिरगञ्ज",
]
_MT_FESTIVALS = [
    "तिहार", "दशैँ", "छठ", "होली", "तीज", "लोसार",
    "इन्द्रजात्रा", "गाईजात्रा", "माघे संक्रान्ति", "फागुपूर्णिमा",
]
_MT_QUALITY = [
    "कौशल", "गौरव", "सौन्दर्य", "योग्यता", "निपुणता",
    "कुशलता", "दक्षता", "सामर्थ्य", "प्रतिभा", "क्षमता",
]
_MT_ADV = ["अत्यन्त", "अति", "निकै", "वास्तवमा", "सदैव", "सधैँ"]

# Consonant bases for syllable contrast lines
_MT_CONSONANTS = list("कखगघचछजटठडढणतथदधनपफबभमयरलवशषसह")

# ----- Numeral domain word banks -----
_NM_YEARS = [
    "१४३२", "१५६७", "१६६०", "१७२५", "१७८३", "१८०५", "१८२५", "१८५६",
    "१९०१", "१९१२", "१९३४", "१९४५", "१९५६", "१९७८", "२००७", "२०१५",
    "२०२७", "२०३६", "२०४६", "२०७२",
]
_NM_MONTHS = [
    "माघ", "फागुन", "चैत्र", "वैशाख", "जेठ", "असार",
    "श्रावण", "भदौ", "असोज", "कार्तिक", "मङ्सिर", "पुस",
]
_NM_DAYS = [
    "१", "२", "३", "४", "५", "६", "७", "८", "९", "१०",
    "११", "१२", "१३", "१४", "१५", "१६", "१७", "१८", "१९", "२०",
    "२१", "२२", "२३", "२४", "२५", "२६", "२७", "२८", "२९", "३०",
]
_NM_AMOUNTS = [
    "५", "१०", "१५", "२०", "२५", "३०", "३५", "४०", "४५", "५०",
    "६५", "७५", "८०", "९०", "१००", "१२५", "१५०", "२००", "२५०",
    "३००", "३५०", "४००", "४५०", "५००", "६५०", "७५०", "१०००",
]
_NM_EVENTS = [
    "भूकम्प आयो ।", "राज्याभिषेक भयो ।", "सन्धि भयो ।", "युद्ध भयो ।",
    "नयाँ नियम लागू भयो ।", "ताम्रपत्र जारी भयो ।", "शिलालेख खोदियो ।",
    "नयाँ शासन स्थापित भयो ।", "विजय प्राप्त भयो ।", "दरबार निर्माण भयो ।",
]
_NM_PURPOSES = [
    "कर", "भाडा", "शुल्क", "दस्तुर", "खर्च", "तिरो",
    "भत्ता", "ज्याला", "दान", "चन्दा",
]
_NM_ROYALS = ["त्रिभुवन", "महेन्द्र", "वीरेन्द्र", "पृथ्वीनारायण", "गिर्वाणयुद्ध"]
_NM_PLACES = ["दिल्ली", "काठमाडौँ", "गोरखा", "पाटन", "भक्तपुर", "बनारस"]
_NM_ITEMS = ["धान", "गहुँ", "मकै", "तेल", "घिउ", "नून", "चिनी", "दाल", "चामल"]
_NM_UNITS = ["मुरी", "पाथी", "मानो", "धार्नी", "सेर"]
_NM_PEOPLE = ["सिपाही", "सैनिक", "कामदार", "मजदुर", "सेवक", "प्रहरी", "दूत", "कर्मचारी"]
_NM_STRUCTURES = ["मन्दिर", "पाटी", "धर्मशाला", "विद्यालय", "पुल", "सडक", "किल्ला"]
_NM_COUNTS = ["३", "४", "५", "७", "८", "१०", "१२", "१५", "२०", "२५", "३०", "५०"]

# ----- Admin domain word banks -----
_AD_ORDERS = ["हुकुम", "आज्ञा", "फर्मान", "आदेश", "निर्देश", "राजाज्ञा", "ताकेता"]
_AD_OFFICIALS = [
    "सुब्बा", "अमालदार", "बडाहाकिम", "फौजदार", "कारिन्दा",
    "मुखिया", "जिम्मावाल", "तहसिलदार", "अमीन", "नाइब",
]
_AD_DOCS = [
    "आज्ञापत्र", "सनद", "हुकुमनामा", "ताकेता", "पत्र",
    "लालमोहर", "ताम्रपत्र", "शिलालेख", "अभिलेख", "वंशावली",
]
_AD_PLACES = [
    "गोरखा", "नुवाकोट", "तनहुँ", "पाल्पा", "लमजुङ", "काठमाडौँ",
    "ललितपुर", "भक्तपुर", "कपिलवस्तु", "मकवानपुर",
]
_AD_DEST = [
    "गाउँ", "नगर", "जिल्ला", "प्रदेश", "किल्ला", "चौकी",
    "अड्डा", "कचहरी", "पञ्चायत",
]
_AD_TASKS = [
    "कर असुली", "भूमि सर्वेक्षण", "जनगणना", "न्याय प्रशासन",
    "सीमा निर्धारण", "शासन", "रक्षा", "व्यवस्था", "कर संकलन",
]
_AD_KINGS = [
    "पृथ्वीनारायण शाह", "नरभूपाल शाह", "रणबहादुर शाह", "गिर्वाणयुद्ध",
    "राजेन्द्र", "सुरेन्द्र", "भूपतीन्द्र मल्ल", "जयप्रकाश मल्ल",
    "यक्षमल्ल", "रामशाह",
]
_AD_RELATIONS = ["छोरा", "नाति", "उत्तराधिकारी", "पुत्र", "वंशज"]
_AD_PMS = [
    "भीमसेन थापा", "जङ्गबहादुर", "चन्द्रशमशेर", "जुद्धशमशेर",
    "पद्मशमशेर", "मोहनशमशेर", "देवशमशेर",
]
_AD_REFORMS = [
    "भूमिसुधार", "न्यायसुधार", "शिक्षासुधार", "प्रशासनिक सुधार",
    "सामाजिक सुधार", "आर्थिक सुधार", "सैनिक सुधार",
]
_AD_TOPICS = [
    "जग्गा", "कर", "सम्पत्ति", "खेत", "भूमि", "वन",
    "सीमाना", "अधिकार", "दान", "बाँडफाँड",
]
_AD_DIRS = ["पूर्व", "पश्चिम", "उत्तर", "दक्षिण"]
_AD_FEATURES = ["नदी", "खोला", "पहाड", "वन", "बाटो", "खेत", "बारी", "नाला"]
_AD_PARTICIPANTS = [
    "सामन्तहरू", "भारदारहरू", "सरदारहरू", "प्रमुखहरू", "मन्त्रीहरू",
    "अमात्यहरू", "काजीहरू", "सभासदहरू",
]
_AD_CEREMONIES = ["दरबार", "सभा", "समारोह", "उत्सव", "राज्याभिषेक"]

# ====================================================================
# TEMPLATES  (reference large word banks -> massive combinatorial space)
# ====================================================================

CONJUNCT_TEMPLATES = [
    ("{a}ले {o}को {v} {e}", {"a": _CJ_AGENTS, "o": _CJ_OBJECTS, "v": _CJ_VERBS, "e": _CJ_ENDINGS}),
    ("{p}मा {o}को {v} भयो ।", {"p": _CJ_PLACES, "o": _CJ_OBJECTS, "v": _CJ_VERBS}),
    ("{a}ले {p}मा {v} गर्नुभयो ।", {"a": _CJ_AGENTS, "p": _CJ_PLACES, "v": _CJ_VERBS}),
    ("{d}मा {o}को विवरण लेखिएको छ ।", {"d": _CJ_DOCS, "o": _CJ_OBJECTS}),
    ("{a}ले {o} प्राप्त गरेको {d}मा लेखिएको छ ।", {"a": _CJ_AGENTS, "o": _CJ_OBJECTS, "d": _CJ_DOCS}),
    ("{t}मा {a}ले {v} शुरु गरे ।", {"t": _CJ_TIMES, "a": _CJ_AGENTS, "v": _CJ_VERBS}),
    ("{o} र {o2}को सम्बन्ध {d}मा वर्णित छ ।", {"o": _CJ_OBJECTS, "o2": _CJ_OBJECTS, "d": _CJ_DOCS}),
    ("{p}का {a}ले {o}को {v} गरे ।", {"p": _CJ_PLACES, "a": _CJ_AGENTS, "o": _CJ_OBJECTS, "v": _CJ_VERBS}),
    ("{a}को {o} सबैमा प्रसिद्ध थियो ।", {"a": _CJ_AGENTS, "o": _CJ_OBJECTS}),
    ("यो {o} {p}बाट प्राप्त भएको हो ।", {"o": _CJ_OBJECTS, "p": _CJ_PLACES}),
    ("{d}अनुसार {a}ले {o}को {v} गरेको थियो ।", {"d": _CJ_DOCS, "a": _CJ_AGENTS, "o": _CJ_OBJECTS, "v": _CJ_VERBS}),
    ("{a}ले {d}को {v} गर्दै {p}मा बस्नुभयो ।", {"a": _CJ_AGENTS, "d": _CJ_DOCS, "v": _CJ_VERBS, "p": _CJ_PLACES}),
    ("{t}मा {p}को {o} प्रसिद्ध थियो ।", {"t": _CJ_TIMES, "p": _CJ_PLACES, "o": _CJ_OBJECTS}),
    ("{a}ले {p}मा {o}को {v} गरेको {d}मा लेखिएको छ ।", {"a": _CJ_AGENTS, "p": _CJ_PLACES, "o": _CJ_OBJECTS, "v": _CJ_VERBS, "d": _CJ_DOCS}),
    ("चिह्नहरू {p}को {d}मा अङ्कित भएका थिए ।", {"p": _CJ_PLACES, "d": _CJ_DOCS}),
    ("ग्लानिको समयमा {o}को पुनर्स्थापना हुन्छ ।", {"o": _CJ_OBJECTS}),
    ("प्राचीन श्लोकमा {o}को वर्णन गरिएको छ ।", {"o": _CJ_OBJECTS}),
    ("क्लेशमा परेका {a}को उद्धार गर्न प्रयत्न भयो ।", {"a": _CJ_AGENTS}),
    ("नम्र {a}ले विनम्रतापूर्वक {v} गरे ।", {"a": _CJ_AGENTS, "v": _CJ_VERBS}),
    ("ब्रह्माण्डको {o} {d}मा वर्णित छ ।", {"o": _CJ_OBJECTS, "d": _CJ_DOCS}),
    ("{a}ले {o}सम्बन्धी {d} तयार गरे ।", {"a": _CJ_AGENTS, "o": _CJ_OBJECTS, "d": _CJ_DOCS}),
    # Sanskrit seed lines (fixed text) -- avagraha, visarga, conjunct-heavy
    ("कर्मण्येवाधिकारस्ते मा फलेषु कदाचन ।", {}),
    ("धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः ।", {}),
    ("यज्ञार्थात् कर्मणोऽन्यत्र लोकोऽयं कर्मबन्धनः ।", {}),
    ("श्रद्धावान् लभते ज्ञानम् ।", {}),
    ("विद्वान् ब्राह्मणः शास्त्रं पठति ।", {}),
    ("असतो मा सद्गमय तमसो मा ज्योतिर्गमय ।", {}),
    ("सर्वे भवन्तु सुखिनः सर्वे सन्तु निरामयाः ।", {}),
    ("धर्मो रक्षति रक्षितः ।", {}),
    ("विद्या ददाति विनयं विनयाद्याति पात्रताम् ।", {}),
    ("योगक्षेमं वहाम्यहम् ।", {}),
]

# --- Expanded matra word banks for capacity ---
_MT_NOUNS = [
    "विद्या", "शिक्षा", "रक्षा", "भक्ति", "शक्ति", "मुक्ति", "बुद्धि",
    "नीति", "रीति", "प्रीति", "गीता", "सीता", "कीर्ति",
    "गुरु", "सुख", "पुत्र", "मधु", "तनु", "धनु",
    "भूमि", "सूर्य", "मूल", "दूत", "पूर्ण", "रूप",
    "केश", "देव", "सेवा", "प्रेम", "वेद", "मेला",
    "दैव", "वैराग्य", "सैनिक", "कैवल्य", "नैवेद्य",
    "लोक", "मोक्ष", "शोक", "बोध", "योग", "कोट",
    "कौशल", "गौरव", "सौन्दर्य", "मौलिक", "लौकिक",
    "शान्ति", "सम्मान", "संगीत", "संस्कृति", "सम्पत्ति",
    "दुःख", "नमः",
    "कृषि", "कृपा", "पृथ्वी", "हृदय", "नृत्य", "सृष्टि", "अमृत",
    "गाउँ", "आँखा", "दशैँ", "काठमाडौँ",
]
_MT_VERBS_EXP = [
    "अध्ययन गरे", "लेखे", "पढे", "सुने", "बुझे", "सिके",
    "गाए", "भने", "रचना गरे", "प्रदर्शन गरे", "वर्णन गरे",
    "व्याख्या गरे", "शिक्षण दिए", "प्रचार गरे", "सम्मान गरे",
]
# Compound verbs only -- safe for "Xको {vc}" patterns (no bare intransitive verbs)
_MT_VERBS_COMPOUND = [
    "अध्ययन गरे", "रचना गरे", "प्रदर्शन गरे", "वर्णन गरे",
    "व्याख्या गरे", "शिक्षण दिए", "प्रचार गरे", "सम्मान गरे",
]
_MT_ADJECTIVES = [
    "सुन्दर", "प्राचीन", "विशेष", "नयाँ", "पुरानो", "महान्",
    "प्रसिद्ध", "ऐतिहासिक", "धार्मिक", "सांस्कृतिक", "पारम्परिक",
    "अमूल्य", "दुर्लभ", "उत्कृष्ट", "अनुपम",
]
_MT_VERB_NOUNS = [
    "अध्ययन", "रचना", "प्रदर्शन", "वर्णन", "व्याख्या",
    "प्रचार", "सम्मान", "प्रशंसा", "संरक्षण", "अभ्यास",
]

MATRA_TEMPLATES = [
    # 0: Syllable contrast lines - each consonant position filled independently
    ("{c}ि {c}ी {c}ु {c}ू {c}े {c}ै {c}ो {c}ौ ।", {"c": _MT_CONSONANTS}),
    # 1: Confusing-pair sentences (large capacity: 16*16=256)
    ("{w1} र {w2} दुबै शब्द लेखिएका छन् ।", {
        "w1": ["निति", "गिरि", "कवि", "मणि", "रवि", "हरि", "बलि", "गुरु", "सुख", "कुल", "नेत्र", "देव", "केश", "कोट", "लोक", "बोध"],
        "w2": ["नीति", "गीता", "कवी", "मणी", "रवी", "हरी", "बली", "गूढ", "सूर्य", "कूल", "नैवेद्य", "दैव", "कैवल्य", "कौशल", "लौकिक", "बौद्ध"],
    }),
    # 2: Subject + object + verb (SOV) (15*10*15=2250)
    ("{s}ले {o} {v} ।", {"s": _MT_SUBJ, "o": _MT_OBJ_BOOKS, "v": _MT_VERB}),
    # 3: Noun + place + adj (63*10*15=9450)
    ("{adj} {n}को विवरण {p}मा पाइन्छ ।", {"adj": _MT_ADJECTIVES, "n": _MT_NOUNS, "p": _MT_PLACES}),
    # 4: Subject + noun + compound verb SOV (15*55*8=6600)
    ("{s}ले {n}को {vc} ।", {"s": _MT_SUBJ, "n": _MT_NOUNS, "vc": _MT_VERBS_COMPOUND}),
    # 5: Adverb visarga + ri word (8*18=144)
    ("{va} {rw}को वर्णन गरिएको छ ।", {"va": _MT_VISARGA_ADV, "rw": _MT_RI_WORDS}),
    # 6: Noun + noun description (55*55=3025)
    ("{n1} र {n2}को वर्णन पुस्तकमा छ ।", {"n1": _MT_NOUNS, "n2": _MT_NOUNS}),
    # 7: Place + festival (10*10=100)
    ("{p}मा {f} मनाइन्छ ।", {"p": _MT_PLACES, "f": _MT_FESTIVALS}),
    # 8: Chandrabindu place + festivals (12*10=120)
    ("{cw}मा {f} मनाइन्छ ।", {"cw": _MT_CHAND_WORDS, "f": _MT_FESTIVALS}),
    # 9: Subject + quality + adverb (15*10*6=900)
    ("{s}को {q} {ad} प्रशंसनीय थियो ।", {"s": _MT_SUBJ, "q": _MT_QUALITY, "ad": _MT_ADV}),
    # 10: Ri words + object books (18*10=180)
    ("{rw}सम्बन्धी पुरानो {o} पाइयो ।", {"rw": _MT_RI_WORDS, "o": _MT_OBJ_BOOKS}),
    # 11: Place + subject + object (10*15*10=1500)
    ("{p}की {s}ले सुन्दर {o} रचना गरिन् ।", {"p": _MT_PLACES, "s": _MT_SUBJ, "o": _MT_OBJ_BOOKS}),
    # 12: Place + adj + noun + verb-noun (10*15*55*10=82500) - no double verb
    ("{p}मा {adj} {n}को {vn} भयो ।", {"p": _MT_PLACES, "adj": _MT_ADJECTIVES, "n": _MT_NOUNS, "vn": _MT_VERB_NOUNS}),
    # 13: Adverb visarga in natural sentence (8*55=440)
    ("{va} जीवनमा {n}को महत्त्व छ ।", {"va": _MT_VISARGA_ADV, "n": _MT_NOUNS}),
    # 14: Ri word + adj + compound verb (18*15*8=2160)
    ("{adj} {rw}को {vc} ।", {"adj": _MT_ADJECTIVES, "rw": _MT_RI_WORDS, "vc": _MT_VERBS_COMPOUND}),
    # 15: Chandrabindu place + noun + adj (12*15*55=9900)
    ("{cw}मा {adj} {n} देखियो ।", {"cw": _MT_CHAND_WORDS, "adj": _MT_ADJECTIVES, "n": _MT_NOUNS}),
    # 16: Noun visarga in sentence (55 combos)
    ("दुःख र सुखमा {n}को अनुभव हुन्छ ।", {"n": _MT_NOUNS}),
]

NUMERAL_TEMPLATES = [
    ("वि. सं {y} {m} {d} गते {e}", {"y": _NM_YEARS, "m": _NM_MONTHS, "d": _NM_DAYS, "e": _NM_EVENTS}),
    ("रू. {a} सम्म {pu} पर्दथ्यो ।", {"a": _NM_AMOUNTS, "pu": _NM_PURPOSES}),
    ("श्री ५ {n} {pl} जानुभयो ।", {"n": _NM_ROYALS, "pl": _NM_PLACES}),
    ("{y} सालमा नयाँ व्यवस्था भयो ।", {"y": _NM_YEARS}),
    ("जग्गा {r} रोपनी {an} आना क्षेत्रफल थियो ।", {"r": _NM_COUNTS, "an": _NM_COUNTS}),
    ("रू. {a} खर्च भएको विवरण लेखियो ।", {"a": _NM_AMOUNTS}),
    ("संवत् {y} मा ताम्रपत्र जारी भयो ।", {"y": _NM_YEARS}),
    ("{c1} जना {p1} र {c2} जना {p2} आए ।", {"c1": _NM_COUNTS, "p1": _NM_PEOPLE, "c2": _NM_COUNTS, "p2": _NM_PEOPLE}),
    ("वि. सं {y1} देखि {y2} सम्म शासनकाल चल्यो ।", {"y1": _NM_YEARS, "y2": _NM_YEARS}),
    ("{it} {c} {u} थियो ।", {"it": _NM_ITEMS, "c": _NM_COUNTS, "u": _NM_UNITS}),
    ("रू. {a} प्रति {u}को दरमा {it} बिक्री भयो ।", {"a": _NM_AMOUNTS, "u": _NM_UNITS, "it": _NM_ITEMS}),
    ("{y} मा {pl}मा {c} वटा {st} बनाइयो ।", {"y": _NM_YEARS, "pl": _NM_PLACES, "c": _NM_COUNTS, "st": _NM_STRUCTURES}),
    ("कुल {c1} मध्ये {c2} जना {p1} थिए ।", {"c1": _NM_COUNTS, "c2": _NM_COUNTS, "p1": _NM_PEOPLE}),
    ("{y} सालको {m}मा {c} दिनसम्म उत्सव चल्यो ।", {"y": _NM_YEARS, "m": _NM_MONTHS, "c": _NM_COUNTS}),
    ("श्री ५ महाराजाधिराज {n}का पालामा {e}", {"n": _NM_ROYALS, "e": _NM_EVENTS}),
]

ADMIN_TEMPLATES = [
    ("श्री ५ महाराजाधिराजबाट {o} भयो ।", {"o": _AD_ORDERS}),
    ("ताम्रपत्रमा लेखिएको व्यहोरा स्पष्ट छ ।", {}),
    ("गाउँको सिमाना {d1}मा {f1} र {d2}मा {f2} थियो ।", {"d1": _AD_DIRS, "f1": _AD_FEATURES, "d2": _AD_DIRS, "f2": _AD_FEATURES}),
    ("वंशावलीअनुसार राजा {k1}का {rel} {k2} भए ।", {"k1": _AD_KINGS, "rel": _AD_RELATIONS, "k2": _AD_KINGS}),
    ("दरबारबाट जारी गरिएको {doc} {dest}मा पठाइयो ।", {"doc": _AD_DOCS, "dest": _AD_DEST}),
    ("मोहर लागेको कागजमा {t}को विवरण थियो ।", {"t": _AD_TOPICS}),
    ("यसरी लेखिएको व्यहोराबमोजिम {task} गरियो ।", {"task": _AD_TASKS}),
    ("राजाको आज्ञाले {off}लाई {pl}मा {task} तोकियो ।", {"off": _AD_OFFICIALS, "pl": _AD_PLACES, "task": _AD_TASKS}),
    ("{doc}मा {k}को शासनकालको विवरण पाइन्छ ।", {"doc": _AD_DOCS, "k": _AD_KINGS}),
    ("{off}ले {dest}को {task} गर्नुपर्ने आदेश भयो ।", {"off": _AD_OFFICIALS, "dest": _AD_DEST, "task": _AD_TASKS}),
    ("लालमोहर अनुसार {dest}लाई {t} दिइयो ।", {"dest": _AD_DEST, "t": _AD_TOPICS}),
    ("प्रधानमन्त्री {pm}को पालामा {ref} भयो ।", {"pm": _AD_PMS, "ref": _AD_REFORMS}),
    ("{k}ले {pl}मा किल्ला निर्माण गराउनुभयो ।", {"k": _AD_KINGS, "pl": _AD_PLACES}),
    ("{doc}मा {t}सम्बन्धी {o} जारी भयो ।", {"doc": _AD_DOCS, "t": _AD_TOPICS, "o": _AD_ORDERS}),
    ("राजकीय {cer}मा {par} सहभागी भए ।", {"cer": _AD_CEREMONIES, "par": _AD_PARTICIPANTS}),
    ("पुरानो {doc}मा {t}को विस्तृत वर्णन छ ।", {"doc": _AD_DOCS, "t": _AD_TOPICS}),
    ("{off}ले {t}को नाप नक्सा सम्पन्न गरे ।", {"off": _AD_OFFICIALS, "t": _AD_TOPICS}),
    ("{k}ले {pl}मा {ref} गराउनुभयो ।", {"k": _AD_KINGS, "pl": _AD_PLACES, "ref": _AD_REFORMS}),
    ("{doc} {off}को सहिछापसहित {dest}मा पठाइयो ।", {"doc": _AD_DOCS, "off": _AD_OFFICIALS, "dest": _AD_DEST}),
    ("{pm}ले {task} सम्बन्धी {o} जारी गरे ।", {"pm": _AD_PMS, "task": _AD_TASKS, "o": _AD_ORDERS}),
]

# ====================================================================
# File configurations
# ====================================================================

FILE_CONFIGS = [
    {
        "name": "coverage_conjunct",
        "csv_filename": "coverage_conjunct_lines.csv",
        "report_filename": "coverage_conjunct_report.md",
        "source_id": "coverage_conjunct",
        "source_file": "generated_coverage_conjunct.txt",
        "line_id_prefix": "coverage_conjunct",
        "total": 2000,
        "splits": {"train": 1600, "val": 200, "test": 200},
        "templates": CONJUNCT_TEMPLATES,
        "coverage_targets": TARGET_CONJUNCTS,
        "coverage_threshold": 40,
        "coverage_fn": find_conjuncts_in_text,
    },
    {
        "name": "coverage_matra",
        "csv_filename": "coverage_matra_lines.csv",
        "report_filename": "coverage_matra_report.md",
        "source_id": "coverage_matra",
        "source_file": "generated_coverage_matra.txt",
        "line_id_prefix": "coverage_matra",
        "total": 1500,
        "splits": {"train": 1200, "val": 150, "test": 150},
        "templates": MATRA_TEMPLATES,
        "coverage_targets": TARGET_MATRAS,
        "coverage_threshold": 80,
        "coverage_fn": find_matras_in_text,
    },
    {
        "name": "names_dates_numerals",
        "csv_filename": "names_dates_numerals_lines.csv",
        "report_filename": "names_dates_numerals_report.md",
        "source_id": "coverage_numeral",
        "source_file": "generated_names_dates_numerals.txt",
        "line_id_prefix": "coverage_numeral",
        "total": 1500,
        "splits": {"train": 1200, "val": 150, "test": 150},
        "templates": NUMERAL_TEMPLATES,
        "coverage_targets": DEVANAGARI_DIGITS,
        "coverage_threshold": 100,
        "coverage_fn": find_digits_in_text,
    },
    {
        "name": "historical_admin_style",
        "csv_filename": "historical_admin_style_lines.csv",
        "report_filename": "historical_admin_style_report.md",
        "source_id": "coverage_admin",
        "source_file": "generated_historical_admin_style.txt",
        "line_id_prefix": "coverage_admin",
        "total": 1776,
        "splits": {"train": 1420, "val": 178, "test": 178},
        "templates": ADMIN_TEMPLATES,
        "coverage_targets": None,
        "coverage_threshold": 0,
        "coverage_fn": None,
    },
]

# ====================================================================
# CSV & row construction
# ====================================================================

def build_rows(texts_with_tidx, config, rng):
    items = list(texts_with_tidx)
    rng.shuffle(items)
    splits = config["splits"]
    split_list = []
    for sn, cnt in splits.items():
        split_list.extend([sn] * cnt)
    rows = []
    for idx, ((text, tidx), split) in enumerate(zip(items, split_list)):
        lid = f"{config['line_id_prefix']}_{idx + 1:06d}"
        ll, hd, hdd, hdg, hc = recompute_metadata(text)
        rows.append({
            "line_id": lid, "source_id": config["source_id"],
            "source_file": config["source_file"], "text": text,
            "line_length": str(ll), "has_danda": hd, "has_double_danda": hdd,
            "has_digit": hdg, "has_conjunct": hc, "split": split,
            "_template_idx": tidx,
        })
    return rows

def write_csv(rows, filepath):
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in CSV_FIELDNAMES})

# ====================================================================
# QA validation
# ====================================================================

def run_qa_checks(rows, config, prior_cov_texts, core_texts):
    res = {
        "total_rows": len(rows), "split_counts": collections.Counter(),
        "blank_count": 0, "duplicate_line_ids": 0, "duplicate_texts": 0,
        "line_length_mismatch": 0, "line_length_out_of_range": 0,
        "ascii_digit_count": 0, "latin_letter_count": 0, "bad_symbol_count": 0,
        "no_devanagari_count": 0, "has_digit_mismatch": 0,
        "has_danda_mismatch": 0, "has_double_danda_mismatch": 0,
        "has_conjunct_mismatch": 0, "cross_file_duplicates": 0,
        "core_merged_duplicates": 0, "template_usage": collections.Counter(),
        "template_cap_violations": 0, "coverage_counts": collections.Counter(),
        "coverage_below_threshold": [], "digit_ratio_ok": True,
        "passed": True, "failures": [],
    }
    seen_ids, seen_txt = set(), set()
    tcap = int(config["total"] * 0.20)
    cfn = config.get("coverage_fn")

    for row in rows:
        res["split_counts"][row["split"]] += 1
        res["template_usage"][row.get("_template_idx", -1)] += 1
        text = row["text"]
        if not text or not text.strip(): res["blank_count"] += 1
        if row["line_id"] in seen_ids: res["duplicate_line_ids"] += 1
        seen_ids.add(row["line_id"])
        if text in seen_txt: res["duplicate_texts"] += 1
        seen_txt.add(text)
        if text in prior_cov_texts: res["cross_file_duplicates"] += 1
        if core_texts and text in core_texts: res["core_merged_duplicates"] += 1
        ll, hd, hdd, hdg, hc = recompute_metadata(text)
        if str(ll) != str(row["line_length"]): res["line_length_mismatch"] += 1
        if ll < 5 or ll > 160: res["line_length_out_of_range"] += 1
        if re.search(r"[0-9]", text): res["ascii_digit_count"] += 1
        if re.search(r"[a-zA-Z]", text): res["latin_letter_count"] += 1
        if any(ch in BAD_SYMBOLS for ch in text): res["bad_symbol_count"] += 1
        if not any("\u0900" <= c <= "\u097F" for c in text): res["no_devanagari_count"] += 1
        if hd != row["has_danda"]: res["has_danda_mismatch"] += 1
        if hdd != row["has_double_danda"]: res["has_double_danda_mismatch"] += 1
        if hdg != row["has_digit"]: res["has_digit_mismatch"] += 1
        if hc != row["has_conjunct"]: res["has_conjunct_mismatch"] += 1
        if cfn:
            for t in cfn(text): res["coverage_counts"][t] += 1

    for tidx, cnt in res["template_usage"].items():
        if cnt > tcap: res["template_cap_violations"] += 1
    ct = config.get("coverage_targets")
    cthr = config.get("coverage_threshold", 0)
    if ct and cthr > 0:
        for t in ct:
            if res["coverage_counts"][t] < cthr:
                res["coverage_below_threshold"].append((t, res["coverage_counts"][t], cthr))
    if config["name"] == "names_dates_numerals":
        dr = sum(1 for r in rows if r["has_digit"] == "true")
        res["digit_ratio"] = dr / len(rows) if rows else 0
        res["digit_ratio_ok"] = res["digit_ratio"] >= 0.80

    fails = []
    if res["blank_count"]: fails.append("blank_rows")
    if res["duplicate_line_ids"]: fails.append("dup_line_ids")
    if res["duplicate_texts"]: fails.append("dup_texts")
    if res["line_length_mismatch"]: fails.append("length_mismatch")
    if res["line_length_out_of_range"]: fails.append("length_out_of_range")
    if res["ascii_digit_count"]: fails.append("ascii_digits")
    if res["latin_letter_count"]: fails.append("latin_letters")
    if res["bad_symbol_count"]: fails.append("bad_symbols")
    if res["no_devanagari_count"]: fails.append("no_devanagari")
    if res["has_digit_mismatch"]: fails.append("digit_mismatch")
    if res["has_danda_mismatch"]: fails.append("danda_mismatch")
    if res["has_double_danda_mismatch"]: fails.append("double_danda_mismatch")
    if res["has_conjunct_mismatch"]: fails.append("conjunct_mismatch")
    if res["template_cap_violations"]: fails.append("template_cap_exceeded")
    if res["coverage_below_threshold"]: fails.append("coverage_below_threshold")
    if config["name"] == "names_dates_numerals" and not res["digit_ratio_ok"]:
        fails.append("digit_ratio_below_80pct")
    res["passed"] = len(fails) == 0
    res["failures"] = fails
    return res

# ====================================================================
# Report generation
# ====================================================================

def write_report(rows, qa, config, filepath):
    verdict = "PASSED" if qa["passed"] else "FAILED: " + ", ".join(qa["failures"])
    md = [f"# Coverage Report: {config['name']}", "", f"## Verdict: {verdict}", "", "## Summary"]
    md.append(f"- **Total rows**: {qa['total_rows']}")
    for s, c in sorted(qa["split_counts"].items()): md.append(f"- **{s}**: {c}")
    md += [
        f"- **Blank rows**: {qa['blank_count']}",
        f"- **Duplicate line_ids**: {qa['duplicate_line_ids']}",
        f"- **Duplicate texts (intra)**: {qa['duplicate_texts']}",
        f"- **Cross-file duplicates**: {qa['cross_file_duplicates']}",
        f"- **Core-merged duplicates**: {qa['core_merged_duplicates']}",
        f"- **ASCII digits**: {qa['ascii_digit_count']}",
        f"- **Latin letters**: {qa['latin_letter_count']}",
        f"- **Bad symbols**: {qa['bad_symbol_count']}",
        f"- **Length mismatches**: {qa['line_length_mismatch']}",
        f"- **Length out of range**: {qa['line_length_out_of_range']}",
        "",
    ]
    lengths = [int(r["line_length"]) for r in rows]
    if lengths:
        md += ["## Line Length Statistics",
               f"- **Min**: {min(lengths)}", f"- **Max**: {max(lengths)}",
               f"- **Average**: {sum(lengths)/len(lengths):.1f}", ""]
    # Coverage tables
    if config["name"] == "coverage_conjunct":
        md += ["## Target Conjunct Coverage", "", "| Conjunct | Count | Threshold | Status |", "|---|---|---|---|"]
        for c in TARGET_CONJUNCTS:
            cnt = qa["coverage_counts"].get(c, 0)
            md.append(f"| {c} | {cnt} | >={config['coverage_threshold']} | {'PASS' if cnt >= config['coverage_threshold'] else 'FAIL'} |")
        md.append("")
    if config["name"] == "coverage_matra":
        md += ["## Target Matra Coverage", "", "| Matra | Count | Threshold | Status |", "|---|---|---|---|"]
        for m in TARGET_MATRAS:
            cnt = qa["coverage_counts"].get(m, 0)
            md.append(f"| {m} | {cnt} | >={config['coverage_threshold']} | {'PASS' if cnt >= config['coverage_threshold'] else 'FAIL'} |")
        md.append("")
    if config["name"] == "names_dates_numerals":
        md += ["## Devanagari Digit Frequency", "", "| Digit | Count | Threshold | Status |", "|---|---|---|---|"]
        for d in DEVANAGARI_DIGITS:
            cnt = qa["coverage_counts"].get(d, 0)
            md.append(f"| {d} | {cnt} | >={config['coverage_threshold']} | {'PASS' if cnt >= config['coverage_threshold'] else 'FAIL'} |")
        dr = sum(1 for r in rows if r["has_digit"] == "true")
        ratio = dr / len(rows) * 100 if rows else 0
        md += ["", f"- **has_digit=true rows**: {dr} ({ratio:.1f}%) {'PASS' if qa.get('digit_ratio_ok', True) else 'FAIL'} (>=80%)", ""]
    # Template usage
    tcap = int(config["total"] * 0.20)
    md += ["## Template Usage Distribution", f"Template cap (20%): {tcap}", "",
           "| Template # | Count | % | Status |", "|---|---|---|---|"]
    for tidx in sorted(qa["template_usage"].keys()):
        cnt = qa["template_usage"][tidx]
        pct = cnt / qa["total_rows"] * 100 if qa["total_rows"] else 0
        md.append(f"| {tidx} | {cnt} | {pct:.1f}% | {'PASS' if cnt <= tcap else 'FAIL'} |")
    md.append("")
    # Samples
    md += ["## Sample Rows", "", "| # | line_id | text | split |", "|---|---|---|---|"]
    srng = random.Random(42)
    for i, r in enumerate(srng.sample(rows, min(10, len(rows)))):
        md.append(f"| {i+1} | {r['line_id']} | {r['text'].replace('|', '/')} | {r['split']} |")
    md.append("")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

def write_summary_report(all_results, filepath):
    ap = all(r["qa"]["passed"] for r in all_results)
    md = ["# Coverage Generation Summary Report", "",
          f"## Overall Verdict: {'ALL PASSED' if ap else 'SOME FAILED'}", "",
          "## Per-File Summary", "", "| File | Rows | Passed | Failures |", "|---|---|---|---|"]
    for r in all_results:
        n = r["config"]["name"]
        t = r["qa"]["total_rows"]
        p = "PASS" if r["qa"]["passed"] else "FAIL"
        f = ", ".join(r["qa"]["failures"]) or "-"
        md.append(f"| {n} | {t} | {p} | {f} |")
    tc = sum(r["qa"]["cross_file_duplicates"] for r in all_results)
    tcore = sum(r["qa"]["core_merged_duplicates"] for r in all_results)
    cc = os.path.exists(CORE_MERGED_PATH)
    md += ["", "## Cross-File Duplicate Check", f"- **Total**: {tc}", "",
           "## Core-Merged Duplicate Check",
           f"- **Checked**: {'Yes' if cc else 'No'}", f"- **Total**: {tcore}", "",
           "## Grand Totals",
           f"- **Total coverage rows**: {sum(r['qa']['total_rows'] for r in all_results)}", ""]
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

# ====================================================================
# Main
# ====================================================================

def main():
    print("=" * 70)
    print("MANTRA Stage 1 Coverage-Line Generation Pipeline")
    print("=" * 70)
    rng = random.Random(SEED)
    core_texts = set()
    if os.path.exists(CORE_MERGED_PATH):
        print(f"Loading core-merged texts from {CORE_MERGED_PATH}...")
        with open(CORE_MERGED_PATH, "r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                core_texts.add(row["text"])
        print(f"  Loaded {len(core_texts)} core-merged texts.")
    else:
        print("  Core-merged file not found, skipping dedup.")
    global_seen = set(core_texts)
    all_results = []
    prior_cov = set()

    for config in FILE_CONFIGS:
        print(f"\n--- Generating: {config['name']} ({config['total']} rows) ---")
        total = config["total"]
        tcap = int(total * 0.20)
        lines, t_usage, cov = _generate_lines(
            templates=config["templates"], target_count=total, template_cap=tcap,
            seen_texts=global_seen, rng=rng,
            coverage_targets=config.get("coverage_targets"),
            coverage_threshold=config.get("coverage_threshold", 0),
            coverage_fn=config.get("coverage_fn"),
        )
        print(f"  Generated {len(lines)} / {total} lines")
        if len(lines) < total:
            print(f"  [WARNING] Only {len(lines)}/{total} lines generated!")
        rows = build_rows(lines, config, rng)
        qa = run_qa_checks(rows, config, prior_cov, core_texts)
        print(f"  QA: {'PASSED' if qa['passed'] else 'FAILED: ' + ', '.join(qa['failures'])}")
        if config.get("coverage_targets"):
            for t, cnt, thr in qa.get("coverage_below_threshold", []):
                print(f"    [WARN] '{t}' = {cnt} (need >={thr})")
        csv_path = os.path.join(LABEL_DIR, config["csv_filename"])
        write_csv(rows, csv_path)
        print(f"  CSV: {csv_path}")
        rpt = os.path.join(REPORT_DIR, config["report_filename"])
        write_report(rows, qa, config, rpt)
        print(f"  Report: {rpt}")
        for r in rows: prior_cov.add(r["text"])
        all_results.append({"config": config, "qa": qa, "rows": rows})

    sp = os.path.join(REPORT_DIR, "coverage_all_summary_report.md")
    write_summary_report(all_results, sp)
    print(f"\nSummary: {sp}")
    ap = all(r["qa"]["passed"] for r in all_results)
    print("\n" + "=" * 70)
    print("ALL PASSED" if ap else "SOME FAILED")
    if not ap:
        for r in all_results:
            if not r["qa"]["passed"]:
                print(f"  FAILED: {r['config']['name']} -- {', '.join(r['qa']['failures'])}")
    print("=" * 70)
    sys.exit(0 if ap else 1)

if __name__ == "__main__":
    main()
