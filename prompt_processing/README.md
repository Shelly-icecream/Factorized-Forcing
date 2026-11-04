# P/A/B prompt preparation

This directory contains offline caption parsing, the semantic vocabulary
(`rules.json`), audits, and encoding/token-length helpers. Parsed data and
reports go in `prompts/processed/`. Training uses the repository's `train.py`
and `trainer/distillation.py`; it does not load spaCy.

## Parse and check captions

Use an independent CPU environment for spaCy to keep its NumPy dependencies
separate from the training environment:

```bash
python -m venv .venv-spacy
.venv-spacy/bin/python -m pip install -r prompt_processing/requirements-spacy.txt
.venv-spacy/bin/python -m prompt_processing.parse_captions \
  prompts/vidprom_filtered_extended.txt \
  prompts/processed/vidprom_filtered_extended.parsed.spacy.jsonl \
  --input-format text --workers 4
python -m prompt_processing.check_data \
  prompts/processed/vidprom_filtered_extended.parsed.spacy.jsonl \
  --parser-version spacy_en_v1_multi_action
```

`--rules path/to/rules.json` supplies a different vocabulary. The default path
is resolved relative to the parser module. The spaCy backend uses the local
`en_core_web_sm` pipeline; no LLM or API is used during parsing. Each worker
loads its own CPU pipeline. Missing dependencies fail explicitly.

The rules-only baseline remains available with `--backend rules`, using parser
version `rules_en_v3_multi_action`. `CaptionParser` is this baseline;
`SpacyCaptionParser` supplies syntax assistance. Parser versions identify
behavior, independently of the vocabulary filename.

Both backends support multiple action spans, joined in original caption order
for A/PA/AB. All action spans are removed from empty/P/B/PB. PAB remains exactly
the input caption. One subject and one background are required; negation and
overlapping spans are rejected. The spaCy backend checks verb POS/lemmas,
uses direct objects for action phrases, excludes identifiable camera actions,
checks subject/action relations, and recovers standalone known location noun
phrases. These checks do not resolve all word senses or coreference, and do not
guarantee that remaining shared text contains no factor information.

Rows with `parse_status != ok` remain in the parsed JSONL for diagnosis and are
excluded by the training dataset. Missing conditions, mismatched parser versions,
and a PAB different from the input caption fail validation explicitly.

## Audit

```bash
.venv-spacy/bin/python -m prompt_processing.sample_audit \
  prompts/vidprom_filtered_extended.txt prompts/processed/spacy_sample_audit.json \
  --sample-size 3000 --seed 42
.venv-spacy/bin/python -m prompt_processing.audit \
  prompts/processed/vidprom_filtered_extended.parsed.spacy.jsonl
```

The paired reservoir sample report includes both parsers' counts, gained/lost
examples, seed, and actual spaCy/model versions. Existing sample reports have
been moved to `prompts/processed/` without changing their contents.

`encoding.py` contains versioned `ConditionEncodingCache` keys and token-length
checks for preprocessing. Actual Wan text embeddings are computed in the
training objective with the loaded text encoder; no new offline embedding file
format or cache is required by training.

## Training

```bash
torchrun --standalone --nproc_per_node=6 train.py \
  --config_path configs/factorized_dmd_framewise.yaml \
  --max-steps 3 --logdir logs/factorized_smoke --disable-wandb --no_visualize
```

Use the same command without `--max-steps 3` for the configured run, choosing a
fresh log directory. Set `generator_ckpt` to the actual causal forcing checkpoint.
The config uses `trainer: score_distillation` and `factorized_enabled: true`.
The existing training loop controls generator/critic updates, clipping, EMA,
and saving. The P/A/B objective lives in `model/factorized_objective.py`.
The original configs keep the ordinary DMD objective by default.

Generator rollout uses PAB. Teacher and fake evaluate eight conditions on the
same noisy sample. Component weights operate on the inclusion-exclusion terms;
all-one weights reconstruct the full PAB difference. Fake updates sample one
subset on rank zero and broadcast it, then use exactly that subset for both
rollout and prediction. The configured ratio is one generator update per five
fake updates. Per-rank batch size is currently one, and only T2V is supported.

Full factorized-training checkpoints contain both models/optimizers, data
position, RNG state, EMA and normalization state. Resume on the same world size
with identical data and training settings:

```bash
torchrun --standalone --nproc_per_node=6 train.py \
  --config_path configs/factorized_dmd_framewise.yaml \
  --resume logs/factorized_dmd/checkpoint_0000100 \
  --max-steps 1000 --logdir logs/factorized_resumed --disable-wandb --no_visualize
```

The checkpoint exporter is `python -m utils.export_distillation --help`; launch
it with torchrun and the checkpoint's world size to gather shards on CPU.
Multi-GPU Wan training still requires validation on the training server.

## Tests

```bash
python -m unittest discover -s tests -p 'test_*.py'
.venv-spacy/bin/python -m unittest discover -s tests -p 'test_prompt_spacy.py'
```

CPU training checks exercise the same `Trainer.train` loop, both objectives,
gradient isolation, EMA, periodic/final saves, and exact checkpoint resume.
spaCy-specific tests skip in environments without the preprocessing dependencies.
