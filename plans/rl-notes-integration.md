# Plan: integrate transcribed RL notes into `_writing/RL-notes.md`

## Context

`_writing/RL-notes.md` already covers the policy-gradient spine (MDP → objective → log-derivative
trick → generalized PG → baselines → RLOO → discounting → advantage/TD/GAE → PPO → KL → ladder).
New handwritten notes were transcribed covering: (1) Monte-Carlo + importance sampling primer,
(2) REINFORCE/policy-gradient derivation, (3) GAE, (4) the on-policy problem and importance
sampling for trajectories, (5) PPO, (6) GRPO.

Most of 2, 3, and 5 already exist in the document. So this is **integration, not append**: add the
genuinely-new material, wire in transitions so the pieces connect, and renumber equations. The
priceless part per the user is the *intuition tying the pieces together* — in particular why the
trajectory expectation `τ ~ p_θ` becomes a **per-token** expectation `a_t ~ π_old` in PPO, which
was asserted in the notes with no derivation.

Constraints from the user:
- keep the author's existing voice intact (lowercase, first-person, casual);
- add interim intuition prose tying sections together;
- fix equation numbering (currently inherited from source notes with gaps: 1, 4, 7, 8, 9, 10, 11, 12);
- where the math has no precedent, give the *real* math intuition + cite concept/papers, else a
  plain intuition.

## What already exists vs. what is new

| New-note topic | Status in doc | Action |
| --- | --- | --- |
| MC approximation + CLT + importance sampling | absent | **add new section** |
| REINFORCE objective / `L^PG` loss | largely in "Gradient Trick" + "Generalized PG" | **fold in only the non-duplicated bits** (loss form, note on non-differentiable sampled tokens) |
| GAE | exists as subsection | **augment** with "reward is per-sequence, not per-token" framing + advantage intuition |
| On-policy problem + trajectory IS | ratio mentioned in PPO, derivation absent | **add new section** (the key intuition request) |
| PPO clipped objective | exists | **augment** with the "drift → unstable" motivation from notes; dedup the objective |
| GRPO | absent | **add new section** |

## Key technical content for the new "ratio" intuition

This is the load-bearing new prose. The notes jump from `E_{a_t ~ π_old}[ ... ]` to a per-token
ratio with no justification. The honest resolution:

1. The **exact** trajectory-level importance-sampling identity is a *product* of per-token ratios
   times the sum of score terms:
   `∇J = E_{τ~p_old}[ (∏_t ρ_t) (Σ_t ∇log π_θ(a_t|s_t)) R(τ) ]`, `ρ_t = π_θ(a_t|s_t)/π_old(a_t|s_t)`.
   It does **not** factor into one ratio per token.
2. The per-token form is **not an exact off-policy estimator** — it is a surrogate objective.
   Two standard results justify it:
   - **Performance-difference / CPI surrogate** (Kakade & Langford 2002; Schulman et al. 2015 TRPO):
     condition on the state and treat old state-visitation as fixed, so only the *action* sum is
     reweighted by IS. That is exactly why the ratio is per-action / per-token.
   - **Per-decision importance sampling** (Precup, Sutton & Singh 2000): future ratios integrate
     to 1; only the prefix matters, so the weight truncates. At `θ = θ_old` every `ρ=1`, the
     surrogate matches the true gradient to first order, and clipping keeps `ρ ≈ 1` so the ignored
     prefix mismatch stays second-order.
3. The "per-token vs sequence" granularity is itself a design axis: GSPO (sequence-level ratio,
   Zheng et al. 2025) and CISPO (clip the weight, not the objective) exist precisely because the
   per-token ratio is an approximation with bias/variance trade-offs. Reference: rlhfbook ch. 6.

References to cite inline: Williams 1992 (REINFORCE), Kakade & Langford 2002, Schulman et al. 2015
(TRPO), Schulman et al. 2016 (GAE), Schulman et al. 2017 (PPO), Precup/Sutton/Singh 2000,
Shao et al. 2024 (DeepSeekMath, GRPO), Liu et al. 2025 (Dr. GRPO), Zheng et al. 2025 (GSPO).

## Proposed structure (insertions marked +)

1. `## Markov Decision Process and Trajectories`
2. `## The Core RL Objective`
3. `## The Gradient Trick`
4. `## Generalized Policy Gradient`  (folds in `L^PG` loss form + non-differentiable-token note)
5. `## Why Baselines Don't Bias`
6. `## RLOO ...`
7. `## Discounting γ and Return`
8. `## Advantage and the Bellman Link` → `### GAE` (augment with per-sequence-reward framing)
9. + `## Monte-Carlo Estimation & Importance Sampling` (notes §1; placement TBD — see questions)
10. + `## Off-Policy Reuse: the on-policy problem and where the ratio comes from` (notes §4 + the
    product-vs-per-token intuition above)
11. `## PPO` (augment with drift motivation; reference §10 for the ratio)
12. `## KL Penalty`
13. + `## GRPO` (notes §6, plus Dr. GRPO ↔ RLOO tie-in already foreshadowed in RLOO section)
14. `## The Progression Ladder` (add GRPO row, maybe GSPO/CISPO pointer)

## Equation renumbering

- Renumber **all** numbered equations sequentially `(1)…(N)` across the whole document (existing
  markers at lines 27/35/43/46/54/69/82/94 become contiguous; new equations continue the sequence).
- Number the currently-unnumbered `Δθ ∝ …` update equation so the PG section is consistent.
- Delete the intro sentence "so I didn't bother to change the eq number from my notes.." since it
  will no longer be true.

## Files to modify

- `_writing/RL-notes.md` — the only content file.
- `plans/rl-notes-integration.md` — this plan (untracked).

## Reuse / conventions

- Math style already in file: `$$ … $$`, `\text{- (n)}` tags, `\text{clip}`, `\mathbb{E}`,
  `\operatorname`/`\mathrm`. MathJax is configured in `_includes/head.html` (`$…$`, `$$…$$`).
- Voice: match `_writing/kl-divergence-notes.md` and the existing RL notes (lowercase, first
  person, "here's what i finally got", load-bearing/casual asides).
- Existing cross-links to reuse: RLOO already promises "later dr. GRPO"; `reward-models.md` and
  `kl-divergence-notes.md` hold RM/KL material, so no need to re-derive KL here.

## Steps

- [ ] Fix the two existing typos in RL-notes while editing (`ny` → `my`, `improvememtsn` → `improvements`).
- [ ] Insert MC + IS primer section (notes §1), in the chosen location.
- [ ] Insert "Off-Policy Reuse / per-token ratio" section (notes §4) with the product-vs-per-token
      derivation and inline references.
- [ ] Fold the non-duplicated REINFORCE bits (loss form, non-differentiable token sampling) into
      the gradient/PG sections.
- [ ] Augment the Advantage/GAE section with the per-sequence-reward framing and advantage intuition.
- [ ] Augment PPO with the "policy drifts too far → unstable" motivation; point at the ratio section
      instead of re-deriving.
- [ ] Add GRPO section (group-relative advantage, no critic, KL-in-loss, Dr. GRPO ↔ RLOO tie-in).
- [ ] Extend the Progression Ladder table with GRPO (+ brief GSPO/CISPO pointer).
- [ ] Renumber all equations sequentially; drop the stale intro sentence about numbering.
- [ ] Add any new references to the bottom `resources:` list.

## Verification

- `bundle exec jekyll build` (or `bundle exec jekyll serve`) → confirm no Liquid/kramdown errors.
- Render check: MathJax parses every `$$` block; scan the built page for stray `\text`/unclosed `$$`.
- Read-through for continuity: each new section has a transition sentence in and out; no section is
  duplicated; equation tags are strictly increasing with no gaps.
- Confirm the per-token intuition section explicitly answers "why did τ~π become per-token a_t~π_old"
  with both the exact identity and the two approximations + citations.

## Open questions (see message)

1. Where should the MC/IS primer live?
2. GRPO scope: canonical only, or also Dr. GRPO ↔ RLOO + GSPO/CISPO pointer?
3. Existing prose edits: OK to fix typos + renumber, keeping all prose otherwise verbatim?
