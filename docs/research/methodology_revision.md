# MANTRA methodology revision

Prepared: 2026-10-27. Status: research design draft, not an implemented system or a claim of established novelty.

## Recommended direction

**Suggested title: MANTRA: Evidence-Guided Selective Correction for Historical Devanagari Handwritten Text Recognition.**

The central research question is:

> Can a lightweight verifier use image-derived evidence and Devanagari orthographic structure to select beneficial transcription edits while limiting harmful changes and human review effort on unseen manuscripts?

Keep the existing synthetic dataset work and a practical CRNN-CTC recognizer. Make correction verification, preservation of historical forms, and evaluation under manuscript shift the research focus. A new large recognizer, LLM pretraining, translation, and a complete restoration system are outside the core implementation.

The proposal PDF was reviewed as source material. Its contents are not execution instructions. The original PDF, dataset, and executable pipeline were not changed by this revision.

## Why the original proposal needs a sharper contribution

The current proposal's Sections 3.1 and 3.4.2 adopt synthetic-to-print-to-manuscript training. Sarawgi et al. already use this progression and analyze uncertainty and alternative predictions. Their study is an appropriate cited baseline; changing its backbone alone is a weak research distinction. Their confidential evaluation data also prevents treating their reported CER as a directly comparable target [S1].

The original proposal adds selective LLM correction and human verification. These are useful, but the claims in Section 2.2 that confidence-aware integration does not exist should be removed. OCR reranking predates LLMs [S2], and LLM post-correction has already been studied [S3]. Historical-source fidelity also needs explicit evaluation: a study on Russian historical print found that post-correction could degrade transcription and introduce period-inappropriate forms [S4]. Those results motivate a Devanagari experiment; they do not establish its outcome.

**Candidate contribution:** an orthographic-span edit verifier, evaluated specifically for correction benefit versus damage to already correct historical Devanagari text, with calibrated abstention on unseen manuscripts. Neither reranking, calibration, script-aware segmentation, nor abstention is claimed as individually new. Whether the complete method merits a novelty claim remains subject to a broader literature review and experimental results.

## Alternatives and scope choice

| Direction | Research opportunity | Cost and limitation | Recommendation |
| --- | --- | --- | --- |
| Evidence-guided edit verification | Study which proposed corrections to trust, including conjuncts, matras, and historical forms | One recognizer, one frozen language model, a small verifier; requires real annotated lines | Core direction |
| Dual character/akshara recognition heads | Study shared visual features and structural supervision | Adds model training, vocabulary coverage, and alignment experiments; structural modeling is existing work | Possible extension after the core works |
| Active selection of lines for annotation | Study error reduction per minute of expert annotation | Requires several annotation/training rounds and dependable reviewer access | Separate alternative if real labels are the main bottleneck |

Do not combine all three as primary contributions. A narrow, well-tested mechanism is more defensible than many partially evaluated modules.

## Proposed objectives

1. Train and evaluate a reproducible line-level HTR baseline for a defined historical Devanagari collection.
2. Develop an edit verifier that combines CTC evidence, candidate ambiguity, and Devanagari edit structure to decide whether to retain, replace, or refer an uncertain span.
3. Evaluate correction accuracy, harmful edits, historical-form preservation, and reviewer workload under document-disjoint evaluation.

Primary language scope should be one collection of Old Nepali if suitable ground truth is available. Sanskrit or another collection can be an external transfer test, reported separately. Devanagari is the script, not a single language.

## Proposed methodology

### 1. Define transcription and evidence policies

Use **diplomatic transcription** as the primary target: retain the source's spelling, punctuation, numerals, and meaningful marks according to a written annotation guide. Store raw transcription and any scoring normalization separately. Document canonical Unicode handling, joiners, whitespace, damaged regions, abbreviation conventions, and genuinely undecipherable spans.

Any modernized reading or interpretation is a separate optional output. Do not train or score modernization as if it were faithful transcription. Do not silently delete accents or unusual combinations solely because a modern dictionary rejects them.

Retain an image crop, raw HTR output, candidate list, scores, proposed change, and decision provenance for every reviewed span. CTC-derived coordinates are approximate; the reviewer should also see the complete line and surrounding context.

### 2. Establish a real manuscript benchmark

Synthetic images provide pretraining and controlled probes; they do not establish historical handwriting performance. Secure a labeled historical collection early. AnciDev reports 3,000 transcribed lines and is worth investigating for external evaluation, subject to access, licensing, language, and annotation compatibility [S5]. Its existence does not guarantee suitability for Old Nepali.

An illustrative annotation target is 1,000-2,000 carefully checked real lines, adjusted after a pilot and reviewer/time assessment. This is a planning target, not a proven minimum for statistical power. Independently transcribe a subset and adjudicate disagreements with a qualified reader. Mark unreadable source spans explicitly; do not use contextual guesses as unquestionable ground truth.

Partition by manuscript/document and, where feasible, writer. Keep all lines from a page and all augmented versions of a sample together. Reserve distinct data for recognizer training, verifier fitting, calibration/threshold selection, and final testing. If the collection is too small for four reliable partitions, use grouped cross-fitting within development data and disclose uncertainty. Test manuscripts must not supply prompt examples, retrieval passages, training text, or threshold tuning.

Keep current font-disjoint synthetic splits as a synthetic generalization experiment. They are not a substitute for manuscript-disjoint testing. Audit source-book overlap and near-duplicate passages as well as exact line duplicates.

### 3. Train a fixed recognition baseline

Implement one CRNN-CTC model with aspect-ratio-preserving inputs. Confirm that each sample's output time steps support its target length, including extra CTC steps needed for adjacent repeated labels. Train on the audited synthetic corpus, then fine-tune on real handwritten lines.

Treat an intermediate printed-document stage as an **optional training ablation**, rather than the identity of MANTRA. Compare synthetic-to-manuscript training with synthetic-to-print-to-manuscript training using matched fine-tuning budgets. Record any additional data/compute. Printed scans address scan variation but do not replace handwriting supervision.

Use identical frozen recognizer checkpoints across correction comparisons. A TrOCR comparison is optional if compute allows; it should not delay the primary study. Report the raw recognizer's performance before language-model correction.

### 4. Generate alternatives and identify orthographic spans

Decode a small N-best list with CTC prefix beam search; start with 5 candidates and examine 10 as a development ablation. Include the original output. Align candidate strings to expose disputed spans and expand edit boundaries to whole Devanagari orthographic units. Include insertion/deletion gaps, since deleted characters cannot be flagged by confidence on emitted characters alone.

Preserve the small character vocabulary for recognition. Orthographic units organize candidate differences and edit decisions; they do not require a closed, enormous akshara output vocabulary. Explicitly test conjuncts such as `श्री`, `क्ष`, `त्र`, `ज्ञ`, and forms involving reph, dependent vowel signs, nukta, and permitted joiner behavior. The grouping rules and version must be documented and tested. Script structure motivates grouping, but is not itself a new contribution [S6].

Use disagreement and confidence to prioritize spans. Do not assume high-confidence output is always correct: measure missed errors in the retained spans. Freeze text outside selected spans when constructing a corrected line, and verify this restriction in code.

### 5. Use a language model as a bounded candidate ranker

Provide the language model with surrounding transcription context and candidate IDs. Ask it to rank candidates or return `ABSTAIN`; map IDs back to exact candidate strings. Reject invalid responses. The core system does not permit arbitrary rewriting of the line. Keep prompts, model/version, context window, decoding settings, and invocation counts fixed and logged.

Use one frozen model selected through a small Nepali/Sanskrit development pilot. Do not require LLM fine-tuning for the core experiment. A small local model or a budgeted API can be used after actual hardware and access are checked. Include a character n-gram ranker as a cheaper baseline: an LLM must earn its place experimentally.

Compute candidate-set oracle error on labeled development/test data **for analysis only**. If the correct reading is absent, selection cannot recover it. Report that ceiling. Free-form language-model suggestions may be displayed separately for human consideration, but they do not enter the automatic accepted transcription in the core method.

### 6. Verify proposed edits with image-derived evidence

For line image x and candidate string y, compute the CTC sequence probability by summing valid alignment paths [S7]:

```text
P_CTC(y | x) = sum over pi with B(pi)=y of product_t p_t(pi_t | x)
```

Use the forward algorithm in log space. The proposal's autoregressive expression `P(y_i | x, y_<i)` should not be used as the confidence definition for this CRNN-CTC model. Averaging framewise maximum probabilities would also mix blank frames with emitted characters.

For a proposed replacement, insert it into the otherwise unchanged line and compare its CTC evidence with the original. Useful verifier features are:

- Difference in full-line CTC log likelihood, with explicit length features or a development-validated length correction.
- N-best margin, candidate disagreement, and uncertainty aggregated over the approximate aligned span.
- Edit size and structural type: matra, virama/conjunct, base letter, punctuation, spacing, or other mark.
- Optionally, stability of the **same candidate comparison** across two mild, stroke-preserving image views.

Fit a small logistic model or comparably simple classifier. Start with the likelihood/margin features and add structural features as the main ablation. Mild-view stability is a stretch feature, not a prerequisite. Never treat synthetic damage that removes real strokes as label-preserving training augmentation without review.

Generate verifier training examples from images not used to fit the recognizer, or from grouped out-of-fold predictions. Use adjudicated transcriptions to label a candidate edit as supported or harmful/incorrect. Match the annotation granularity to the evaluated edit: a span containing several changes is supported only if all changes agree with the reference policy. Include difficult wrong-but-fluent candidates and no-change examples. Synthetic confusions can supplement these examples, but report verifier results on real recognition errors separately.

The verifier estimates `P(edit is supported | features)`. Calibrate and choose the acceptance threshold on separate development data. The HTR score is correlated evidence from the same recognizer, not an independent proof that an edit is visually correct. Test this limitation on held-out manuscripts and historically unusual forms.

### 7. Make three explicit decisions

```text
Manuscript line
    -> CRNN-CTC recognition and N-best alternatives
    -> disputed orthographic spans
    -> bounded language-model ranking
    -> edit verification
         -> retain original when evidence supports it
         -> accept supported replacement
         -> refer unresolved span, with alternatives, to a human
```

For physically missing or genuinely illegible text, preserve an uncertainty/lacuna annotation. A plausible completion belongs in an explicitly labeled conjecture field. Recognition error correction and reconstruction of absent writing have different evidential targets.

Select thresholds to maximize useful automatic processing subject to a predeclared harmful-edit budget. For example, 1% harmful accepted edits can be a development target, not a promised result. Report observed rates, sample counts, and uncertainty intervals. Calibration does not guarantee that the same risk holds on a new archive; conformal guarantees are not claimed by this design.

## Evaluation plan

### Required comparisons

| System | Purpose |
| --- | --- |
| A. Fixed HTR with greedy/beam decoding | Recognition baseline |
| B. HTR plus character n-gram reranking | Tests a cheap language prior |
| C. HTR plus unrestricted LLM post-correction | Measures benefits and damage from free rewriting |
| D. HTR plus confidence-threshold LLM correction | Represents the original MANTRA proposal |
| E. Bounded LLM candidate ranking | Isolates candidate restriction |
| F. E plus CTC evidence verifier and abstention | Tests visual-evidence acceptance |
| G. F plus orthographic edit features/grouping | Tests the proposed script-specific component |

Compare on the same held-out documents and report a sweep over automatic coverage/review budgets. Include random or confidence-only review selection at matched budgets. Never count referred spans as automatically correct. Report machine-only results and human-assisted results separately; if every output is human verified, report human time and pre-review machine errors explicitly.

### Metrics and definitions

- **CER/WER:** report corpus-level edit counts and a frozen normalization policy, before and after correction. Also report an orthographic-unit error rate under the published grouping rules.
- **Accepted-edit precision:** supported accepted edits / all accepted edits, using adjudicated span references.
- **Harmful-edit rate:** incorrect accepted edits / all accepted edits; give numerator and denominator. Separately count initially correct spans changed to incorrect text.
- **Correction recall:** original erroneous eligible spans corrected / original erroneous eligible spans. Also report the fraction of all original errors missed by span selection.
- **Preservation:** accuracy on a manually tagged set of historical forms, names, numerals, matras, and conjuncts. Record unwanted normalization explicitly. This adapts an existing preservation-evaluation idea rather than claiming to invent it [S4].
- **Selective risk/coverage:** residual error among automatically handled units as coverage changes, plus the human referral fraction. Abstention should not be allowed to hide poor full-corpus accuracy.
- **Calibration:** Brier score and reliability plots for the defined binary edit-support event. If reporting ECE, disclose bins and sample counts.
- **Cost:** recognizer/verifier runtime, language-model calls and tokens, and wall-clock review time on the same annotation task.
- **Candidate ceiling:** oracle CER and reference-in-N-best rate, used only for analysis.

Use paired document-level bootstrap intervals or another justified grouped analysis. Tiny percentages need enough independent examples: zero observed harmful edits in a small sample is not proof of a zero rate. Human-effort claims require timed, counterbalanced tasks with comparable lines and reviewer expertise; referral count alone is a workload proxy.

### Stress tests that make the study meaningful

1. Unseen manuscripts/writers, with no test-time tuning.
2. Visually confusing matras, reph, conjuncts, and thin marks.
3. Rare but valid historical forms that a modern language model may try to normalize.
4. Controlled clean/degraded image pairs, evaluated separately from real manuscripts.
5. Missing-stroke/occlusion cases where contextual completion may be plausible but not visually recoverable.

Retain every version of an underlying sample within its original partition. Synthetic clean/degraded pairs are a diagnostic resource, not evidence that all real historical damage is modeled.

## Connection to current project work

The existing 100,000 synthetic line images, font isolation, manifests, charset, and data loader remain useful. No regeneration is needed merely because the research question changes. However, synthetic labels copied from OCR text are not automatically reliable ground truth.

The earlier project audit reported 176 images with structural mark warnings and 206 additional mark/accent review candidates. These require source verification; they are not all proven erroneous. A separate readiness check also found U+00B0 in 62 validation samples and none in training. Resolve or explicitly account for these issues before freezing the experiment. Font holdouts and pixel checks establish only parts of data quality.

Current runnable components are data preparation and loading. Recognition training, decoding, verification, and the proposed experiments are future work. Do not present this draft as an implemented architecture.

Suggested future organization, after the research direction is agreed:

```text
docs/research/methodology_revision.md      # This research design
docs/research/annotation_policy.md         # Diplomatic transcription conventions
configs/experiments/                      # Baselines and ablation settings
src/stage2_htr/                           # Recognizer, training, decoding
src/correction/                           # Span selection, ranking, verification
src/evaluation/                           # CER, edit support, coverage, preservation
data/manuscripts/                        # Licensed local images and annotations
outputs/experiments/<run_id>/             # Immutable predictions and reports
```

Only this methodology document is created now. Avoid empty directories and premature package refactors.

## Feasible implementation sequence

An indicative ten-week plan, subject to actual GPU access and annotation availability:

| Period | Deliverable |
| --- | --- |
| Weeks 1-2 | Secure real manuscript lines, annotation policy, resolve synthetic review queue, freeze grouped partitions |
| Weeks 3-4 | One trained recognizer, CTC decoding, reproducible baseline and candidate ceiling |
| Weeks 5-6 | N-gram/LLM comparison, disputed spans, logged bounded candidate ranking |
| Weeks 7-8 | Small verifier, calibration, orthographic-feature ablations |
| Weeks 9-10 | Locked-test experiments, preservation cases, timed reviewer pilot, write-up |

If reliable real ground truth cannot be secured, revise the claim to a controlled synthetic/printed Devanagari study. Unlabeled manuscript case studies cannot support historical CER, calibrated correction accuracy, or quantitative reconstruction claims. Prioritize annotation over adding another large model.

Stretch work should add **one** component: mild-view stability, an external manuscript collection, or a second recognizer. Neither LLM training nor a new vision-language architecture is necessary for the core contribution.

## Proposal-ready abstract draft

Historical Devanagari manuscript recognition is affected by visual ambiguity, limited labeled data, and spelling conventions that differ from modern text. Language-model post-correction may improve readability while introducing changes unsupported by the manuscript. This project proposes MANTRA, an evidence-guided framework for selective correction of line-level handwritten transcription. A CRNN-CTC recognizer will produce an initial transcription and alternative candidates. Disputed spans will be grouped using Devanagari orthographic structure, and a frozen language model will rank a bounded set of candidates. A lightweight verifier will use CTC evidence, candidate ambiguity, and edit structure to select supported replacements or refer uncertain spans for human review. Diplomatic transcription will be maintained separately from any normalized interpretation. The framework will be evaluated on document-disjoint historical manuscript data using character and word error rates, accepted-edit precision, harmful edits, preservation of historical forms, and risk-coverage curves. Comparisons with unrestricted and confidence-threshold language-model correction will test whether evidence-guided verification improves the trade-off between transcription accuracy and reviewer effort.

## Changes to the original proposal

- Replace the broad title about recognition, restoration, and interpretation with the narrower proposed title.
- Rewrite Sections 1.2, 1.3, and 2.2 around edit verification and historical fidelity. Remove unsupported absence/first-system claims.
- Update Section 3.1 with actual labeled manuscript access, grouped partitions, and annotation roles.
- Keep Sections 3.2-3.4 as baseline implementation, and treat the printed stage as an ablation.
- Replace Section 3.5's autoregressive confidence equation with CTC-specific candidate evidence and a calibrated edit-support event.
- Rewrite Section 3.6 to specify bounded candidates, retain/replace/refer decisions, and operational metric definitions.
- Replace Figure 3.1 and Table 3.2 to reflect the revised flow and the distinction between machine results and human-reviewed results.
- Align expected outcomes with demonstrable transcription fidelity; leave translation and semantic interpretation as optional future work.

## Sources checked

These sources support the background and design rationale. They do not establish that the proposed combination is globally novel. Literature checked on 2026-10-27.

- **S1.** Sarawgi, Arias, and Zotter. *Digitizing Nepal's Written Heritage: A Comprehensive HTR Pipeline for Old Nepali Manuscripts*, arXiv:2512.17111v2, revised April 2026. Relevant: Section 3.3, Section 5.5, and Appendix P. https://arxiv.org/html/2512.17111v2
- **S2.** Tomeh et al. *Reranking with Linguistic and Semantic Features for Arabic Optical Character Recognition*, ACL 2013. https://aclanthology.org/P13-2098/
- **S3.** Thomas, Gaizauskas, and Lu. *Leveraging LLMs for Post-OCR Correction of Historical Newspapers*, LT4HALA 2024. https://aclanthology.org/2024.lt4hala-1.14/
- **S4.** Levchenko. *Evaluating LLMs for Historical Document OCR: A Methodological Framework for Digital Humanities*, LM4DH 2025. https://aclanthology.org/2025.lm4dh-1.7/
- **S5.** Sharma, Verma, and Saluja. *AnciDev: A Dataset for High-Accuracy Handwritten Text Recognition of Ancient Devanagari Manuscripts*, BHASHA 2025. https://aclanthology.org/2025.bhasha-1.8/
- **S6.** Microsoft. *Developing OpenType Fonts for Devanagari Script*. Technical background on orthographic units, reordering, and conjunct forms. https://learn.microsoft.com/en-us/typography/script-development/devanagari
- **S7.** Graves et al. *Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks*, ICML 2006. https://www.cs.toronto.edu/~graves/icml_2006.pdf
- **S8.** Garrido-Munoz and Calvo-Zaragoza. *On the Generalization of Handwritten Text Recognition Models*, CVPR 2025. Background for document/domain-shift evaluation. https://openaccess.thecvf.com/content/CVPR2025/html/Garrido-Munoz_On_the_Generalization_of_Handwritten_Text_Recognition_Models_CVPR_2025_paper.html
