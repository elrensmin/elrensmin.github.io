---
title: PPO - phase 1
date: 2026-09-10
description: Experiment log for trying different things out with PPO and CartPole
tags: [experiment, rlhf]
---

## exp1: VPG / no-clip baseline on CartPole

- **Why run it:** This is the vanilla policy gradient that PPO is trying to improve. Getting our baseline.
- **What to vary:** Nothing. No clipping, no KL penalty, one epoch per batch, no data reuse.
- logs: [wandb logs](https://wandb.ai/elrensmin/experiments/runs/spb1irdm)
- **What it teaches:** Large, unconstrained policy-gradient steps can collapse the policy or destabilize training.

## exp2: Add the clipped surrogate objective

- **Why run it:** Isolate the single most important PPO ingredient.
- **What to vary:** `epsilon` ∈ {0.1, 0.2, 0.3}. Also include a "no clip" branch (same as experiment 1). Keep everything else fixed: value baseline, GAE, entropy coefficient, learning rate.
- [wandb logs](https://wandb.ai/elrensmin/experiments/runs/q6yt1pnz)

Clipping alone fixed the exp1 collapse: entropy stays ~0.68 and the score doesn't fall back to random. But the initial clip_fraction reading of 1.0 was a **metric bug** - we were testing `|ratio| > eps` instead of `|ratio - 1| > eps`. Since `ratio` hovers near 1.0, that test is true almost always. Fixed in code; the `cf` column is only trustworthy for runs with the `_lam` suffix (run after the fix).

[lr 1e-4 eps 0.1](https://wandb.ai/elrensmin/experiments/runs/9spdtsh9)
![9spdtsh9](/images/ppo-exp/9spdtsh9.png)

[lr 1e-4 eps 0.2](https://wandb.ai/elrensmin/experiments/runs/lj5ljv56)
![lj5ljv56](/images/ppo-exp/lj5ljv56.png)

[lr 3e-4 eps 0.3](https://wandb.ai/elrensmin/experiments/runs/5dlouiqk)
![5dlouiqk](/images/ppo-exp/5dlouiqk.png)

[lr 5e-4 eps 0.1](https://wandb.ai/elrensmin/experiments/runs/edupei9d)
![edupei9d](/images/ppo-exp/edupei9d.png)

- **What it teaches:** Small epsilon → conservative steps, slower but stable. Large epsilon → closer to VPG, can still collapse. Also KL(old \|\| new) should stay bounded by roughly epsilon on healthy runs.

The recurring signal across this whole phase: **KL bounded by eps learns, KL past eps collapses.** Every plot below is the same shape — train your eye here.

What we actually see:

| run    | eps | lr   | score_max | entropy_last | kl_max    |
| ------ | --- | ---- | --------- | ------------ | --------- |
| eps0.1 | 0.1 | 1e-4 | 39.1      | 0.677        | 0.029     |
| eps0.2 | 0.2 | 1e-4 | 56.1      | 0.662        | 0.054     |
| eps0.3 | 0.3 | 3e-4 | **127.7** | 0.657        | 0.060     |
| eps0.1 | 0.1 | 5e-4 | 38.4      | 0.657        | **0.211** |


The lesson is *not* "lower lr." It's **match the update size to the trust region**:
- **Wide trust region + moderate lr** (eps 0.3, lr 3e-4) is the clear winner at ~127 - the biggest, most effective single update step so far.
- **Tight eps + high lr** (0.1, 5e-4) blows out the trust region: kl_max 0.211 is >2x its own eps, the policy is stepping too far each update and getting noisier (score 27 and falling).
- Entropy stays healthy everywhere (0.66-0.68), so clipping is doing its job - no collapse, just different step sizes.

So, **KL bounded by eps is the clean signal**, and the winner is the config whose step size uses the whole trust region without spilling past it.

## exp3: add a value baseline and vary GAE

[lr 1e-4 eps 0.1 lambda 0.95 gamma 0.98](https://wandb.ai/elrensmin/experiments/runs/xbh2rbz6)

[lr 1e-4 eps 0.1 lambda 0.5 gamma 0.95](https://wandb.ai/elrensmin/experiments/runs/omig9uaw)

**Why run it:** The advantage estimator is where a lot of PPO's sample efficiency comes from. Understand the bias-variance tradeoff.

What we see:

| run                    | eps | lr   | lambda | gamma | score_max | entropy_last | grad_norm_max |
| ---------------------- | --- | ---- | ------ | ----- | --------- | ------------ | ------------- |
| eps0.1 lam0.95 gam0.98 | 0.1 | 1e-4 | 0.95   | 0.98  | 43.8      | 0.686        | 0.41          |
| eps0.1 lam0.5 gam0.95  | 0.1 | 1e-4 | 0.5    | 0.95  | 44.6      | 0.686        | 0.22          |
| eps0.3 lam0.95 gam0.98 | 0.3 | 3e-4 | 0.95   | 0.98  | **127.7** | 0.657        | 1.03          |

- **What it teaches:** `lambda` and `gamma` control the bias-variance tradeoff of the advantage estimate. Lower `lambda`/`gamma` = higher-bias, lower-variance, but here it doesn't move the needle much (44 vs 44.6). The eps/lr combo dominates - the best run is still the wide-eps + moderate-lr one from exp2.
- grad_norm spikes to 1.0 in the strong run: the larger policy steps carry bigger gradients. That's the cost of a wide trust region, worth watching but not fatal here (score still climbs to 127).
- Note `lambda`=0.5 keeps grad_norm calmer (0.22) but trades away the extra sample efficiency - score plateaus lower. Classic bias-variance: tame gradients, weaker learning.

## exp4: Entropy coefficient ablation

Base config eps0.3 / lr3e-4 / lam0.95 / gam0.98, vary `entropy_coef` ∈ {0.0, 0.001, 0.01, 0.05}:

[coef 0.0](https://wandb.ai/elrensmin/experiments/runs/r808tvkn)

[coef 0.001](https://wandb.ai/elrensmin/experiments/runs/bpx9paoi)

[coef 0.01](https://wandb.ai/elrensmin/experiments/runs/ck7185qs)

[coef 0.05](https://wandb.ai/elrensmin/experiments/runs/ah55c3qm)

What we see (reproduce with `uv run python metrics.py r808tvkn bpx9paoi ck7185qs ah55c3qm`):

| entropy_coef | score_max | score_last | entropy_mean | kl_max | grad_norm_max |
| ------------ | --------- | ---------- | ------------ | ------ | ------------- |
| 0.0          | **130.4** | **130.4**  | 0.637        | 0.088  | 0.384         |
| 0.001        | 121.3     | 104.3      | 0.643        | 0.089  | 0.342         |
| 0.01         | 76.9      | 57.7       | 0.666        | 0.025  | 0.504         |
| 0.05         | 35.9      | 33.9       | 0.686        | 0.278  | 0.997         |

![kl vs entropy](/images/ppo-exp/kl-vs-entropy.png)

- **The coefficient does what it's told** - entropy rises monotonically (0.637 → 0.686) with the coef. It just *hurts* here: score_max tanks 130 → 121 → 77 → 36.
- CartPole's optimal policy is near-deterministic - forcing exploration with a high coef keeps the policy flailing, capping score at ~35.
- **0.001 is the only defensible nonzero value** (121 vs 130, nearly a wash). 0.01+ is pure penalty.
- The 0.05 run shows the instability signature: grad_norm 0.997 and kl_max 0.278 (≫ eps) - the policy keeps re-randomizing, so each update fights the entropy term and produces big noisy steps.
- Caveat: a 2-action space is too small for entropy regularization to matter. Re-test on a richer discrete space or the continuous Pendulum / LLM phase.

## exp5: batch size and update epochs

Base config eps0.3 / lr3e-4 / lam0.95 / gam0.98, vary `epochs` and `batch_size`:

[bs20 e4](https://wandb.ai/elrensmin/experiments/runs/ywumn3tb)

[bs20 e10](https://wandb.ai/elrensmin/experiments/runs/poq3fsdn)

[bs32 e3](https://wandb.ai/elrensmin/experiments/runs/uyagk4m5)

[bs128 e3](https://wandb.ai/elrensmin/experiments/runs/iioopt9k)

What we see:

| run                      | epochs | batch_size | score_max | score_last | entropy_last | kl_max   | grad_norm_max |
| ------------------------ | ------ | ---------- | --------- | ---------- | ------------ | -------- | ------------- |
| ep3_bs20 (exp4 baseline) | 3      | 20         | 130.4     | 130.4      | 0.586        | 0.088    | 0.384         |
| ep4_bs20                 | 4      | 20         | **159.8** | **159.8**  | 0.606        | 0.171    | 0.706         |
| ep10_bs20                | 10     | 20         | 92.0      | 89.7       | 0.654        | **0.92** | 0.376         |
| ep3_bs32                 | 3      | 32         | **177.1** | 114.4      | 0.619        | 0.195    | 0.687         |
| ep3_bs128                | 3      | 128        | 89.2      | 89.2       | 0.624        | 0.125    | 0.58          |

- **More epochs is a double-edged sword.** ep3→ep4 improves (130→160) - reusing experience helps. But ep3→ep10 collapses (130→92) and **kl_max explodes to 0.92, ~3x the clip epsilon (0.3)**. Ten epochs drifts the policy so far from the collected data that the trust region is blown out. This is the textbook "overdo the reuse" failure.
- **Larger batch hurts here, but not for the usual reason.** With a fixed 20-step rollout, a batch of 128 re-samples the *same* 20 transitions ~6x per epoch (via `randperm`). So bs128 isn't "more data" - it's the same tiny rollout seen many more times, overfitting the policy to a handful of transitions. That's why it underperforms bs20/bs32.
- **The combination matters more than either knob** (the experiment's key claim): ep4_bs20 (159) and ep3_bs32 (177) both beat the baseline for different reasons - one reuses more, one sees a slightly bigger batch. The worst runs are the extremes: ep10 (too much reuse) and bs128 (too much re-sampling of a tiny rollout).
- **KL bounded by eps is the health check again.** Every run that stays bounded (kl_max ≲ 0.2) learns; the one that blows past eps (ep10, kl 0.92) collapses. Same lesson as exp2.

## exp6: network size ablation

Base config eps0.3 / lr3e-4 / lam0.95 / gam0.98, vary `hidden_units` {32, 128, 512} × `hidden_layers` {1, 2}:

[hu 32 lay 1](https://wandb.ai/elrensmin/experiments/runs/uwq3xkru)

[hu 128 lay 1](https://wandb.ai/elrensmin/experiments/runs/43z896f2)

[hu 512 lay 1](https://wandb.ai/elrensmin/experiments/runs/o7b9o09h)

[hu 32 lay 2](https://wandb.ai/elrensmin/experiments/runs/7sbswivb)

[hu 128 lay 2](https://wandb.ai/elrensmin/experiments/runs/sviyis3a)

[hu 512 lay 2](https://wandb.ai/elrensmin/experiments/runs/coi6l6i7)

What we see (reproduce with `uv run python metrics.py uwq3xkru 43z896f2 o7b9o09h 7sbswivb sviyis3a coi6l6i7`):

| run       | hu  | hl  | score_max | score_last | entropy_last | kl_max | grad_norm_max |
| --------- | --- | --- | --------- | ---------- | ------------ | ------ | ------------- |
| hu32_hl1  | 32  | 1   | 116.7     | 116.7      | 0.670        | 0.028  | 0.494         |
| hu128_hl1 | 128 | 1   | 146.0     | 86.5       | 0.601        | 0.085  | 0.307         |
| hu512_hl1 | 512 | 1   | 124.8     | 124.8      | 0.596        | 0.147  | 0.439         |
| hu32_hl2  | 32  | 2   | 137.4     | 137.4      | 0.642        | 0.063  | 0.517         |
| hu128_hl2 | 128 | 2   | 79.1      | 79.1       | 0.665        | 0.073  | 0.739         |
| hu512_hl2 | 512 | 2   | **187.6** | **187.6**  | 0.638        | 0.176  | 0.610         |

![kl-vs-score](/images/ppo-exp/exp6-kl-vs-score.png)

- **Network size is essentially a non-factor on CartPole.** All six runs land in the 79-188 band, straddling the 130 baseline (hu256_hl1 from exp4). No monotonic trend - the best (hu512_hl2, 188) and worst (hu128_hl2, 79) are both 2-layer, and the spread looks like seed noise, not signal.
- **The smallest network is nearly as good as the best.** hu32_hl1 (1315 params) hits 116.7, within ~10% of the baseline with a fraction of the parameters. This confirms the experiment's hypothesis: CartPole saturates with a tiny network.
- **kl_max and grad_norm grow with size** (0.028→0.176, 0.3→0.7) but don't buy score - just more churn. Bigger nets move the policy further per update without learning better.
- **Lesson:** on a task this simple, capacity is wasted. Use the smallest net that solves the task; save the big networks for Pendulum / the LLM phase where capacity will actually matter.
