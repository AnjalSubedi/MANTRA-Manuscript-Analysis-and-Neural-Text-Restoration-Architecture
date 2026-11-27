# Reproduce the manuscript fine-tuning pilot

This recipe fixes the manuscript split and crop geometry for the comparison reported in the [CRNN–CTC results](../../docs/stage2_htr/crnn_ctc_results.md). [Study metadata](study.json) lists each document and its published diplomatic edition. The source scans, transcriptions, synthetic vocabulary, and checkpoint are local run inputs.

Use the [repository setup](../../README.md#setup), then obtain local copies of the source scans and numbered diplomatic transcriptions. For each study identifier, provide `<identifier>_manuscript_01.png` and `<identifier>_transcription.txt` anywhere inside the source folder. The recipe verifies the original scan pixels, each crop, and each scoring reference by SHA-256. It creates manifests for 78 training, 16 validation, and 23 test lines.

```bash
python experiments/manuscript_pilot/prepare.py \
  --source-dir path/to/local/manuscripts \
  --charset path/to/charset_synth100k.json \
  --output-dir outputs/manuscript_pilot/inputs
```

Use the epoch 10 `best.pt` from the [synthetic training run](../../docs/stage2_htr/crnn_ctc_training.md). Its fingerprint is recorded in `study.json`. The manuscript fine-tuning command writes its own selected checkpoint.

```bash
python src/stage2_htr/train.py \
  --init-from path/to/base/best.pt \
  --data-config outputs/manuscript_pilot/inputs/data.yaml \
  --config outputs/manuscript_pilot/inputs/training.yaml \
  --output-dir outputs/manuscript_pilot/run --device cpu --threads 2
```

Evaluate both checkpoints on the same held-out manifest. Each evaluation writes `metrics.json` and `predictions.jsonl` in a fresh directory.

```bash
python src/stage2_htr/evaluate.py \
  --checkpoint path/to/base/best.pt \
  --data-config outputs/manuscript_pilot/inputs/baseline_data.yaml \
  --manifest outputs/manuscript_pilot/inputs/test.csv \
  --output-dir outputs/manuscript_pilot/baseline_test --device cpu --threads 2
python src/stage2_htr/evaluate.py \
  --checkpoint outputs/manuscript_pilot/run/best.pt \
  --data-config outputs/manuscript_pilot/inputs/data.yaml \
  --manifest outputs/manuscript_pilot/inputs/test.csv \
  --output-dir outputs/manuscript_pilot/fine_tuned_test --device cpu --threads 2
```

The images and references follow the source-specific access terms of the [Documenta Nepalica editions](https://nepalica.hadw-bw.de/nepal/). Keep the generated labeled crops and predictions with the local run. [Public aggregate results](../../results/crnn_ctc/metrics.json)
