# Stage 3: current integration status

## Corrected score helpers

- Tensor order remains `empty/P/A/B/PA/PB/AB/PAB`. Factor masks are
  `0/1/2/4/3/5/6/7`, independently of tensor positions.
- `evaluate_conditions` evaluates **one caption and one shared noisy sample**
  at a time. Its input latent has batch size one; each condition microbatch
  repeats that sample, its timestep and optional mask. The output is `[8, ...]`.
  Analysis runs without gradients. Put the models in evaluation mode beforehand.
- `make_wan_predictor(model, prediction_type="score", scheduler=...)` adapts
  Wan's `(flow, x0)` output and converts flow to score using the sigma associated
  with each timestep in the supplied scheduler. Pass the actual model scheduler;
  timestep IDs are not assumed to equal noise fractions. `flow` and `x0` modes
  explicitly return those quantities instead. Masks must broadcast to the output.
- `make_fake_score_batch` now requires `rollout_subset` and `rollout_condition`
  as keyword arguments. These describe the rollout's actual generation condition;
  the subset label (e.g. `P`) is never compared to prompt text (e.g. `A man`).
  The rollout and noise targets are detached. Noise coefficients support scalars,
  per-example `[B]` and per-frame `[B, F]` values.
- Use `flow_target_loss` for Wan linear-flow predictions. It checks
  `alpha + sigma = 1` and uses `noise - rollout` as the target.
  Use `score_target_loss` only for score predictions with target `-noise/sigma`.
  The old `epsilon_target_loss` name is a compatibility alias for the latter;
  it does not train an epsilon prediction.

Example adapter wiring (model loading and rollout are caller responsibilities):

```python
teacher_predict = make_wan_predictor(teacher, "score", teacher_scheduler)
fake_predict = make_wan_predictor(fake, "score", fake_scheduler)
report = analyze_teacher_and_fake(
    noisy, timestep, conditions,
    lambda texts: text_encoder(text_prompts=texts),
    teacher_predict, fake_predict, microbatch=1,
)
```

## Training entry

The dedicated Stage 3 path is now connected through `stage3.train` and
`configs/stage3_framewise.yaml`. `train.py` also dispatches `trainer: stage3` to
this trainer. Ordinary DMD configurations continue to use their existing path.
The standalone entry avoids importing unrelated trainers.

Check the data before allocating model memory:

```bash
python -m stage3.train --check-data --data-path stage3_outputs/vidprom_filtered_extended.parsed.v13.jsonl
```

Rows whose `parse_status` is not `ok` are excluded before distributed sampling.
Malformed valid rows, missing conditions, parser-version mismatches, and a PAB
condition different from the caption fail explicitly. The dataset stores offsets
instead of retaining all caption strings. Each rank checks the same dataset hash.
Distributed sampling pads to equal lengths if necessary (a tiny dataset can repeat
samples across ranks). Data position and epoch are included in checkpoints.

On the Linux training server, with the repository's existing dependencies and Wan
weights installed, first run three real steps:

```bash
torchrun --standalone --nproc_per_node=6 -m stage3.train \
  --config_path configs/stage3_framewise.yaml \
  --max-steps 3 --logdir stage3_outputs/stage3_smoke
```

For the configured training run:

```bash
torchrun --standalone --nproc_per_node=6 -m stage3.train \
  --config_path configs/stage3_framewise.yaml \
  --logdir stage3_outputs/training
```

Set `generator_ckpt` to the actual causal generator checkpoint. The default is
`checkpoints/framewise/causal_forcing.pt`. Loading an ordinary base checkpoint
allows missing LoRA adapter keys but rejects other missing or unexpected keys.
The supplied config uses Wan 1.3B, six-way full FSDP, per-rank batch size 1,
12 latent frames, generator LoRA rank 8, and condition microbatch 1. The real
teacher and text encoder use CPU offload; the trainable fake score does not.
The number of ranks comes from torchrun, not a hard-coded total batch size.
These settings have not been measured on six 4090s.

## Objective and condition alignment

- The generator rolls out using PAB and receives the weighted decomposition of
  `score_fake - score_real` on a **single shared noisy sample**. Each model sees
  all eight texts. Both score adapters use the rollout scheduler's noise process.
- Noise levels are sampled from the rollout scheduler's discrete sigmas in
  `(0.02, 0.98)`, with identical time for all frames of a video. Score differences
  are multiplied by `sigma^2 / (1-sigma)` to obtain the x0-space DMD update.
  The optional existing per-sample / timestep-EMA normalization is then applied.
  The rollout's gradient mask selects the entries contributing to the loss.
- `stage3_component_weights` controls the eight decomposition terms. All ones
  reconstruct the full PAB conditional difference; changing weights changes the
  objective. CFG is not additionally applied in this path. Parsed `empty` retains
  remaining caption text and is not replaced by the ordinary negative prompt.
- On each fake update, a subset is sampled on rank 0 and broadcast. Rollout and
  fake prediction use exactly that subset's same prompt. The detached generated
  video is trained against `noise - rollout` with a flow MSE. All eight subsets
  must have positive sampling probability.
- The default ratio is one generator update for every five fake updates. All
  ranks execute matching rollout, condition and optimizer-call counts. Metrics
  include losses, subset, component norms and step time in `metrics.jsonl`.

## Checkpoints and resume

The trainer writes `checkpoint_0000100/metadata.json` plus one `rank_XXXXX.pt`
per rank. Files are first written to temporary names; metadata is published only
when all rank files are complete. Each rank stores sharded generator and fake
weights, both optimizer states, local trainable-parameter EMA, normalization
buffers, step, data position and Python/NumPy/PyTorch/CUDA RNG states.

```bash
torchrun --standalone --nproc_per_node=6 -m stage3.train \
  --config_path configs/stage3_framewise.yaml \
  --resume stage3_outputs/training/checkpoint_0000100 \
  --max-steps 1000 --logdir stage3_outputs/training
```

Resume requires the same world size, dataset contents and training settings.
Run length, output directory and checkpoint interval may change. These directories
are training checkpoints, not the ordinary single-file inference export. EMA is
stored per rank over trainable parameters and restored with the same sharding.
The teacher remains frozen and is reloaded from the pretrained Wan files.

## Verification

```bash
python -m stage3.train --smoke-test
python -m unittest discover -s tests -p 'test_stage3*.py'
```

The smoke test uses small simulated modules with the actual objectives, data
reader and checkpoint functions. It verifies both optimizers update, the teacher
stays frozen, fake loss cannot update the generator, and the next losses and
parameters match exactly after saving/restoring. Tests also exercise the real
trainer loop, EMA, final/periodic saves, factor semantics and data validation.

An optional single-GPU FSDP smoke test is available on Linux with NCCL:

```bash
python -m stage3.train --fsdp-smoke-test
```

The local Windows environment lacks NCCL and the full Wan runtime. CPU/simulated
training and YAML configuration have been verified; full Wan forward/backward,
sharded multi-GPU recovery, GPU memory use and throughput require the server run.

## Multi-action parsing

The rules-only parser version is `rules_en_v3_multi_action` (using the existing
`rules_en_v2.json` vocabulary). A caption may contain multiple action spans;
all are joined in caption order for A/PA/AB and removed from empty/P/B/PB.
PAB stays identical to the original caption. Exactly one subject and one
background are still required, and negation and overlapping spans still fail.
This change does not distinguish camera verbs or resolve action word senses.
Regenerate parsed data before training with the updated version:

```bash
python -m stage3.parse_captions prompts/vidprom_filtered_extended.txt \
  stage3_outputs/vidprom_filtered_extended.parsed.multi_action.jsonl \
  --backend rules --input-format text --workers 8
```

Set `data_path` to the regenerated JSONL and `stage3_parser_version` to
`rules_en_v3_multi_action` when training on rules-only data.

## Local spaCy preprocessing (default CLI backend)

`stage3.parse_captions` and `stage3.audit` default to the local CPU
`en_core_web_sm` pipeline, with parser version `spacy_en_v1_multi_action`.
No LLM, GPU or API call is used during parsing. The existing `CaptionParser`
Python class remains the rules-only baseline; use `SpacyCaptionParser` for syntax.
Missing spaCy/model dependencies fail explicitly rather than silently changing backends.

Use a separate preprocessing environment to keep spaCy's NumPy dependencies
independent of the training environment:

```bash
python -m venv .venv-spacy
.venv-spacy/bin/python -m pip install -r stage3/requirements-spacy.txt
.venv-spacy/bin/python -m stage3.parse_captions \
  prompts/vidprom_filtered_extended.txt \
  stage3_outputs/vidprom_filtered_extended.parsed.spacy.jsonl \
  --input-format text --workers 4
```

The training configuration expects this new JSONL and parser version. Training
does not load spaCy. Each preprocessing worker loads one CPU pipeline; use fewer
workers if RAM is limited. `--backend rules` selects the previous rules parser.
The legacy `python -m stage3.parser --input ... --output ...` stays rules-only.

Syntax assists verb POS checks, lemmatization, direct objects, camera-subject
exclusion, standalone known location noun phrases, and unknown noun/proper-name
subjects of accepted actions. The JSON remains the semantic vocabulary for
verbs and locations. One subject and one background are still required; all
recognized action spans form A. This does not perform coreference resolution,
resolve all location/argument ambiguities, or guarantee a leakage-free remainder.
Camera descriptions remain shared context in all subsets.

Reproduce the paired reservoir sample comparison:

```bash
.venv-spacy/bin/python -m stage3.sample_audit \
  prompts/vidprom_filtered_extended.txt stage3_outputs/spacy_sample_audit.json \
  --sample-size 3000 --seed 42
.venv-spacy/bin/python -m unittest discover -s tests -p 'test_stage3_spacy.py'
```

The report records both parsers' counts, sample transitions, examples, and actual
spaCy/model versions. The spaCy-specific tests skip when dependencies are absent.
