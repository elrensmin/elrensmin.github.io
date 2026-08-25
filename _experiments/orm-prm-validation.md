---
title: ORM & PRM Validation-Split Runs
date: 2026-08-13
description: Experiment log for validating the ORM and PRM training scripts 
tags: [experiment, rlhf]
---


Experiment log for validating the ORM and PRM training scripts after the config-driven validation splits and 
metric namespacing landed in PR #523. Both runs log a held-out validation split (10%) with metrics every 25 steps.
The code used here (`feat/orm-val`) is identical to `main` at `3478ff4` [PR #523](https://github.com/natolambert/rlhf-book/pull/523).

## Goal

- Sanity-check that the new ORM/PRM validation path reports a truthful held-out signal (not just training loss).
- Get reference numbers for the default configs so later tuning has a baseline.

## Commands

Run from `code/` with `WANDB_PROJECT` set (or `WANDB_MODE=disabled`):

```bash
uv run python -m reward_models.train_orm --config reward_models/configs/orm.yaml
uv run python -m reward_models.train_prm --config reward_models/configs/prm.yaml
```

## Configs (defaults, unchanged from `main`)

### ORM [`reward_models/configs/orm.yaml`](https://github.com/elrensmin/rlhf-book/blob/feat/orm-val/code/reward_models/configs/orm.yaml)

- Model: `Qwen/Qwen3-0.6B-Base`, full FT, 596.05M trainable
- Dataset: `openai/gsm8k`, train split, `samples: 2000` → 3600 train / 400 validation
- Training: `lr 5.0e-5`, `warmup_ratio 0.1`, `warmup_only`, `epochs 3`,`batch_size 2`, `grad_accum_steps 16`, `seed 7`, 
 `val_ratio 0.1`, `eval_interval 25`

### PRM [`reward_models/configs/prm.yaml`](https://github.com/elrensmin/rlhf-book/blob/feat/orm-val/code/reward_models/configs/prm.yaml)

- Model: `Qwen/Qwen3-0.6B-Base`, full FT, 596.05M trainable
- Dataset: `tasksource/PRM800K`, train split, `samples: 2000` (raw problems, capped before step-chunking) → 3667 train / 368 validation
- Training: `lr 5.0e-5`, `warmup_ratio 0.1`, `warmup_only`, `epochs 1`, `batch_size 1`, `grad_accum_steps 16`, `seed 13`, `val_ratio 0.1`,
  `eval_interval 25`

## Results

### ORM (GSM8K)

- Training accuracy hits **1.000 by step ~22** and stays there for the run.
- **Validation Accuracy = 1.000 at every eval** (steps 25, 50, 75, 100, ...).
- Validation Loss is **flat ≈ 0.65** the whole run:
  - step 25: 0.670 → step 100: 0.659 → step 200: 0.677 → step 300: 0.656 →
    epoch 2 end: 0.680.

The accuracy metric is **degenerate** here: GSM8K completion labels are skewed toward "correct", so a model that 
predicts everything-correct reports 1.0 accuracy while the cross-entropy stays near its baseline. Loss is the
only truthful signal in this setup, and it shows the ORM barely learns to discriminate. It flatlines around 0.65 
and rises slightly at the end (mild overfit on the noise). I treat this ORM's accuracy as uninformative; judge it by 
held-out loss (or better, by downstream reward margins / best-of-N calibration).

### PRM (PRM800K)

- Validation step-accuracy: **0.539 → 0.63** (eval 25 → eval 225).
- Validation loss: **0.93 → 0.80** over the run.

PRM shows a genuine, non-degenerate learning signal. A clean contrast with the ORM. This is the healthiest reward-model 
run of the two and a good baseline for further tuning.

## Failure Mode / Follow-ups

- **ORM:** fix the label imbalance or switch eval to a loss/margin-based metric before trusting it as a reward source. 
  The 1.0 accuracy should not be quoted as evidence the ORM "works".
- **PRM:** healthy; next step is evaluating step-level margins on held-out problems and checking best-of-N / search behavior.


I'm not able to make my wandb logs public.. will fix this and upload from next time.
