# Low-commitment language as self-conditioning

This repository implements the pre-registered study in [`plan.md`](plan.md).
It tests contrastive negation and enumerative preambles across the matched
OLMo 3 32B base, SFT, DPO, and RLVR checkpoints.

The claim under test is contextual: emitted scaffold text changes the prefix
used to predict later tokens. It is not a claim that output tokens provide
wall-clock thinking time.

## Experiments and results

The first experiment detects contrastive negation and enumerative preambles,
aligns each event to exact generated tokens, measures full-vocabulary entropy,
and deletes the scaffold while scoring the original continuation. Contrastive
negation produced a meaningful conditioning gain of `0.424` nats per lexical
token (95% CI `[0.267, 0.593]`), but its matched entropy effect was negligible:
`0.009` nats (95% CI `[-0.032, 0.054]`). Enumerative preambles occurred only
three usable times, so they do not support inference.

The local follow-up mines recurrent lexical and abstract discourse templates
without using stage labels, then tests stage changes on prompt-matched,
length-matched responses. Of 1,119 frequent closed templates, none of the 40
stage-blind candidates survived FDR correction. However, within-prompt 4-gram
overlap increased from `0.0454` after SFT to `0.0641` after RLVR, a difference
of `0.0187` (95% CI `[0.0116, 0.0259]`). Abstract-template Jaccard overlap also
increased by `0.0336` (95% CI `[0.0038, 0.0655]`). Both changes occurred mainly
from SFT to DPO.

Removing detected contrastive, concessive, and enumerative scaffolds left the
surface-overlap increase essentially unchanged. The current evidence therefore
supports broader surface convergence after preference training, but does not
show that a smaller repertoire of identified low-commitment scaffolds causes
that convergence. Base-model comparisons remain exploratory because the base
prompt format and generation-length cap differ.

## Commands

```bash
python -m llmisms build-corpus --output outputs/full
python -m llmisms generate --output outputs/full --backend transformers --stage sft
python -m llmisms tag --output outputs/full
python -m llmisms score --output outputs/full --backend transformers --stage sft
python -m llmisms analyze --output outputs/full
python -m llmisms patterns --output outputs/full
python -m llmisms smoke --output outputs/smoke
```

`smoke` uses the cached BF16 MLX SmolLM3-3B checkpoint by default and runs 12
prompts with two stochastic seeds. Large records are Parquet; run parameters
and summaries are JSON. Generated outputs and model caches are not tracked.

The remote runner accepts only `smoke` or `full`:

```bash
scripts/run_upnquick.sh smoke
scripts/run_upnquick.sh full
```

The full remote run processes one checkpoint at a time, reports existing GPU
use, and requires at least 320 GB free.
