# Focused local follow-up: recurrent discourse templates

## Goal

Test whether post-training concentrates OLMo generations onto a smaller,
more recurrent repertoire of discourse templates, without generating or
scoring any new model responses.

## Data and scope

- Reuse the completed 240-prompt, three-seed base/SFT/DPO/RLVR corpus.
- Keep base comparisons exploratory because their prompt format and length cap
  differ. Use SFT/DPO/RLVR for primary stage trends.
- Exclude greedy responses.
- Select candidates by pooled frequency while hiding stage labels. Because
  candidate selection is invariant to stage-label permutations, use all
  matched prompts for the stage tests without circular enrichment selection.

## Experiments

1. **Broader discourse detection**
   - Extend contrastive detection beyond `not ... but` to correction,
     reframing, and concessive forms with exact character boundaries.
   - Keep rules deterministic and auditable.

2. **Recurrent-template discovery**
   - Convert clauses into lexical n-grams and discourse skeletons that preserve
     function/cue words while collapsing content-word runs.
   - Mine frequent closed templates pooled across stages, counting support once
     per response and requiring support across unique prompts.
   - Freeze the most frequent nonredundant patterns without inspecting their
     stage trajectories, then test them with paired prompt bootstrap intervals
     and permutation p-values.
   - Correct the tested pattern family with Benjamini-Hochberg.

3. **Matched diversity**
   - Compare the three responses per stage using length-matched lexical n-gram
     overlap and shared-template probability.
   - Bootstrap whole prompts.
   - Repeat lexical overlap after removing detected discourse scaffolds to
     estimate how much recurrent rhetoric contributes to surface convergence.

## Outputs and decision rules

- Write a JSON summary, a Parquet pattern table, and compact figures.
- Treat a template as confirmed only when its SFT-to-RLVR increase has a 95%
  prompt-bootstrap interval above zero and FDR-adjusted one-sided permutation
  `q < 0.05`.
- Evidence for rhetorical concentration requires both increased sharing of
  mined templates and increased length-matched surface overlap.
- Do not call reduced surface diversity semantic mode collapse.
