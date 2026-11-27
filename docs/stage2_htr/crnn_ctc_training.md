# MANTRA CRNN–CTC training

The baseline uses the active Synth100k training and validation manifests and
the matching character vocabulary. Each batch groups lines of similar resized
width to reduce padding. Validation CER is computed over all validation
characters. The default configuration runs ten epochs with AdamW and selects
the lowest validation CER, breaking ties on validation CTC loss.

## Train and resume

From the repository root, run the trainer with the configured data paths.
Use `--device cuda` on GPU or `--device cpu` for a small local check.

```bash
python src/stage2_htr/train.py --device cuda --output-dir outputs/crnn_ctc/baseline
python src/stage2_htr/train.py --device cuda --output-dir outputs/crnn_ctc/baseline \
  --resume outputs/crnn_ctc/baseline/last.pt
```

The run directory contains the following files.

- `last.pt` stores the model, optimizer, random state, epoch and batch position,
  metrics, and the best weights known at that point. The trainer writes it
  atomically at startup, every 500 updates by default, and after each epoch.
- `best.pt` stores the validation-selected model, its constructor options,
  preprocessing settings, and the character map.
- `metrics.json` records training and validation CTC loss, validation CER,
  sample counts, the effective settings, and SHA-256 fingerprints of both
  manifests and the charset.

Keep `last.pt` for resumption. The trainer accepts the same data and batch
settings from a different storage path. `--max-steps` pauses at a specified
total optimizer step. `--time-limit-minutes` saves progress at a batch boundary
when the current invocation reaches its budget. Leave time for a validation
pass and for the notebook platform to save outputs. `--train-limit` and
`--val-limit` make small smoke runs.

## Fine-tune on manuscript lines

Start fine-tuning from a selected checkpoint in a new output directory.

```bash
python src/stage2_htr/train.py --init-from path/to/best.pt \
  --data-config path/to/manuscript_data.yaml --config path/to/finetune.yaml \
  --output-dir outputs/crnn_ctc/manuscript_finetune
```

The run starts with the checkpoint weights and a fresh optimizer. A manuscript
vocabulary may append characters while preserving existing class IDs. Choose
the checkpoint on validation manuscripts and report test CER on separate
manuscripts. Use `last.pt` with `--resume` to continue the new run.

## Run on Kaggle

Package the active training and validation images, manifests, and charset into
a ZIP that preserves their repository-relative paths.

```bash
python src/stage2_htr/package_training_data.py \
  outputs/crnn_ctc/kaggle/mantra-synth100k-train-val.zip
```

Attach the ZIP as a private Kaggle dataset. Extract it to a writable runtime
directory and pass that directory as `--data-root`. Run the trainer from a
checkout of the code in a single-GPU CUDA runtime. Write the output to
`/kaggle/working` so it can be saved with the notebook version.

```bash
python src/stage2_htr/train.py --device cuda --data-root /kaggle/temp/mantra_data \
  --output-dir /kaggle/working/phase5 --time-limit-minutes 600
```

For a new session, attach the previous notebook output and pass its
`last.pt` to `--resume`. The trainer can recreate `best.pt` from that file.
The held-out test manifest and images belong to the later evaluation phase.

## Evaluate the selected checkpoint

Run the evaluator from the repository root after training finishes.

```bash
python src/stage2_htr/evaluate.py --checkpoint path/to/best.pt \
  --output-dir outputs/crnn_ctc/synthetic_test
```

The default input is the configured `manifest_test`. The evaluator verifies
preprocessing, the vocabulary, and the train and validation manifest hashes
against the checkpoint. It writes `metrics.json` with corpus-level CER and
`predictions.jsonl` with linked images, references, and greedy transcriptions.

For manuscript crops, pass `--manifest` with a UTF-8 CSV containing `image_path`
and `text`. Use `--data-root` to resolve relative paths. Optional `line_id`,
`width`, and `height` columns identify the line and help group similar widths.
Leave `text` empty for a crop awaiting transcription. The evaluator writes its
prediction for review and computes CER over the rows with references. Characters
outside the training vocabulary still contribute to the reported errors.
