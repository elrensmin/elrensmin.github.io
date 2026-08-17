---
title: Reward Models
date: 2026-08-16
description: notes in RLHF
tags: [rl, notes]
---

These are my notes for ch. 5 of the [RLHF book](https://rlhfbook.com/c/05-reward-models). This is the part I actually need to keep in my head: the intuition, the datasets, and where it lives in the `open-instruct` repo and TRL.

## logsigmoid

```python
loss = -F.logsigmoid(chosen_reward - rejected_reward).mean()   # reward_modeling.py:357
```

that's the negative log-likelihood of the Bradley-Terry preference model, article eq. 6, `−log σ(r_c − r_r)`. the `.mean()` is the expectation over the batch. 
some artihmetic magic and eq. 7 with the softplus is mathematically identical.

- the loss only ever sees the *difference* `r_c − r_r`, never the absolute values (eq. 2). 
- add a constant to every score and nothing changes. so the RM's absolute score is meaningless. only the ordering matters. 
- the model's entire job is to make the chosen score bigger than the rejected one. that's it. that's the whole model.

## where chosen/rejected rewards come from

```python
# reward_modeling.py:351-357
query_responses = torch.cat((data[CHOSEN_INPUT_IDS_KEY], data[REJECTED_INPUT_IDS_KEY]), dim=0)
_, predicted_reward, _ = get_reward(model, query_responses, tokenizer.pad_token_id, 0)
chosen_reward = predicted_reward[: data[CHOSEN_INPUT_IDS_KEY].shape[0]]
rejected_reward = predicted_reward[data[CHOSEN_INPUT_IDS_KEY].shape[0]:]
accuracy = (chosen_reward > rejected_reward).float().mean()
loss = -F.logsigmoid(chosen_reward - rejected_reward).mean()
```

chosen and rejected get concatenated into one batch, chosen on top, rejected below. `get_reward` scores the whole thing, one scalar per sequence. top half is chosen, bottom half is rejected. `accuracy` is just the discrete version of the same preference. "did the model give chosen a higher score than rejected?"

## `get_reward`: turning a sequence of tokens into one scalar

Full function: `model_utils.py:372-436`.

### the head

```python
# reward_modeling.py:293-295
model = AutoModelForSequenceClassification.from_pretrained(..., num_labels=1)
# olmo_adapter/__init__.py:14
self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)  # num_labels=1
# olmo_adapter/__init__.py:57-58
hidden_states = transformer_outputs[0]       # (batch, seq, hidden)
logits = self.score(hidden_states)           # (batch, seq, 1)
```

`score` is a single linear layer, `hidden_size - 1`, producing a logit at **every** token position. this matches the article's "Default Reward Model Architecture." one linear layer.

### the function

```python
# model_utils.py:372-436
def get_reward(model, query_responses, pad_token_id, context_length):
    attention_mask = query_responses != pad_token_id          # (batch, seq)
    position_ids = attention_mask.cumsum(1) - attention_mask.long()   # exclusive cumsum
    lm_backbone = getattr(model, model.base_model_prefix)      # e.g. model.model
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
    output = lm_backbone(input_ids, attention_mask, position_ids,
                         return_dict=True, output_hidden_states=True, use_cache=False)
    reward_logits = model.score(output.hidden_states[-1])     # (batch, seq, 1)
    sequence_lengths = first_true_indices(query_responses[:, context_length:] == pad_token_id) - 1 + context_length
    final_scores = reward_logits[torch.arange(reward_logits.size(0)), sequence_lengths].squeeze(-1)  # (batch,)
    return reward_logits, final_scores, sequence_lengths
```

the caller only uses the second return value: `_, predicted_reward, _ = get_reward(...)`. so the whole function exists to make one thing true: one scalar per sequence, shape `(batch,)`.

- we want one scalar per sequence so we pick one representative position per sequence.
- we pick EOS because its hidden state has attended to (and summarized) all prior context - the natural "read the whole thing and grade it" token (article Figure 2). `first_true_indices(... == pad_token_id) - 1` lands on the last real token before padding.
- to pick EOS we need per-token logits so we score every position.
- to score positions we need a linear head `model.score` maps hidden-1.
- to feed the head we need final layer hidden states -> run the backbone with `output_hidden_states=True`.
- to run the backbone cleanly we need a mask + correct positions so build them from the padding, and zero the pad ids.

backward flow: **desired scalar ← EOS position ← per-token logits ← head ← final hidden states ← clean forward pass.**

why `lm_backbone` not `model`? the wrapper would run its own head + pooling + loss; we want raw hidden states so *we* decide pooling (the EOS trick). why `use_cache=False`? inference-only full-sequence forward, no generation, avoids a Mistral bug. i had to look that one up twice, which is sloppy hacky coding but whatever.

## the data IS the algorithm

ok this is the part i actually want to remember, the thing that made everything click. the best way to understand BT vs ORM vs PRM is the *shape of the labels*. the loss and the code are just the mechanical consequence of the label structure. you don't choose a model type, you choose a label shape, and the model type follows.

**BT - a PAIR with one "chosen" label** (repo: `allenai/llama-3.1-tulu-3-8b-preference-mixture`, `scripts/train/tulu3/reward_modeling_8b.sh:23`):

```
prompt:   "Explain the difference between a compiler and an interpreter."
chosen:   "A compiler translates the entire source program into machine code in one go, then runs it. An interpreter executes the source code line by line, translating and running each statement as it goes."
rejected: "compiler is thing that makes code fast. interpreter is other thing. they both do code stuff."
```

two completions, one relative label ("chosen is better", it better be. the day an LLM answers with the rejected answer is the day AIs become my friend not a useful tool). the only sensible loss is CONTRASTIVE: make chosen score higher than rejected - `-log σ(r_c − r_r)`. no absolute truth, only a comparison - the model learns *relative* quality.

**ORM - ONE completion with ONE binary label, broadcast everywhere** (label from `GSM8KVerifier`, `ground_truth_utils.py:199`, which extracts the last number and compares to ground truth):

```
prompt:   "If a train travels 60 miles in 1.5 hours, what is its average speed in mph?"
completion: "Average speed = distance / time = 60 / 1.5 = 40 mph."
label:    1   (correct - verifier extracted 40 == ground truth 40)
```

one completion, one label (correct/incorrect), and that SAME label is copied onto every completion token. the only sensible loss is per-token binary cross-entropy against the broadcast label. no step information - the model can only learn "does this look like it ends correct," never "which step was wrong."

**the one-line takeaway:** the label granularity in the data dictates the loss. pair -> contrastive. whole-response -> per-token BCE. per-step -> per-step CE. the code is just the faithful transcription of the data's structure.

### intuition

- **BT RM** = "How good is this whole answer?" - a *relative* quality score from human preference. used to **rank** completions (best-of-N) or provide a terminal reward. captures subjective quality (helpfulness, style) but needs expensive paired human labels.
- **ORM** = "Does this answer end up correct?" - an *objective* correctness probability from cheap verifiable labels (math, code, exact-match). used to **score** a finished completion or guide search.

the subtle bit, and this is the part that genuinely surprised me: even though the ORM only ever sees the *final* correctness label, by predicting it at **every token** it implicitly learns "early tokens that look like they're heading toward a correct answer get high probability." so per-token prediction acts as a **proxy for intermediate quality** - it rewards a promising reasoning trajectory, not just the final answer. but it cannot catch a wrong step that happens to lead to a right answer, because reasoning errors are never labeled. Amazing. 

### training

the ORM head is the **same** `nn.Linear(hidden_size, 1)` as the BT RM (`olmo_adapter/__init__.py:14`), and `get_reward` already produces a logit at **every** token (`model_utils.py:417`). the only changes to turn BT into ORM:

1. **use all completion-token logits**, not just the EOS one.
2. **apply binary cross-entropy** against the broadcast label, instead of the contrastive BT loss.
3. **mask prompt tokens with `-100`** so they don't contribute.

```python
# per-token logits from the same head
logits_per_token = model.score(hidden_states[-1]).squeeze(-1)   # (batch, seq)
# labels: prompt tokens = -100, every completion token = r (0 or 1)
mask = labels != -100
loss = F.binary_cross_entropy_with_logits(logits_per_token[mask], labels[mask].float())
```

two loss forms (article eq. 8 vs 9):
- **per-token BCE (eq. 8):** each completion token trained toward `r` individually.
- **sequence-level CE (eq. 9):** sigmoid of the *average* of per-token logits -> one probability for the whole completion. (sigmoid applied *after* averaging, not average of sigmoids)

### inference 

once trained, the ORM outputs a correctness probability `p_t` at every completion token. to score a **finished** completion you aggregate:

- **mean** of `p_t` - average confidence
- **min** of `p_t` - tail risk (penalize if any token looks bad)
- **product** `Π_t p_t` (equivalently sum of log-probs) - joint confidence
- **average over the last m tokens** - focus on the conclusion
- **threshold** - flag the completion if any `p_t < τ`

the repo's *runtime* analog of the label source is `apply_verifiable_reward` (`model_utils.py:439-514`): a verifier returns `score ∈ {0,1}`, scaled by `reward_mult * score * reward_weight` (line 507). a trained ORM is the learned, differentiable proxy for this verifier. It predicts the correctness the verifier would assign, at every token, without needing the ground truth at inference time.

## PRM (Process Reward Models)

### the data IS the algorithm

**PRM - ONE completion split into STEPS, each with its OWN label:**

```
prompt:  "If a train travels 60 miles in 1.5 hours, what is its average speed?"
step 1:  "Average speed = distance / time."          - +1 (correct)
step 2:  "distance = 60 miles, time = 1.5 hours."  - +1 (correct)
step 3:  "speed = 60 * 1.5 = 90 mph."               - −1 (incorrect)
```

one completion, but now a label PER STEP. the only sensible loss is cross-entropy at each step boundary against that step's own label. the data HAS step information - so the model can learn "which step went wrong," which neither BT nor ORM can.

### code

the head is the same pattern as BT/ORM, but with 3 outputs (correct/neutral/incorrect):

```python
self.head = nn.Linear(self.lm.config.hidden_size, num_classes)   # num_classes=3

logits = model.head(hidden)          # (batch, seq, 3)
mask = labels != -100                # only step-boundary tokens
loss = F.cross_entropy(logits[mask], labels[mask])
```

the `-100` masking is the same trick as ORM but now the **unmasked positions are ONLY the step boundaries, not every completion token**.

how the step labels get attached:

```python
separator_ids = tokenizer.encode(step_separator, add_special_tokens=False)
completions_ids = [completion + separator_ids for completion in completions_ids]
labels = [[-100] * (len(completion) - 1) + [label] for completion, label in zip(completions_ids, labels)]
```

---> append a separator token to each step, put the step's label on the LAST token of that step (the separator), and `-100` on everything else. the model learns "the score at the step-boundary token = that step's validity."

### trl's closest analog: per-step rewards

this repo doesn't train a standalone PRM, but the CONCEPT of a per-step reward is all over the RL/environment code.

`StepResult.reward` - a reward per environment step (`environments/base.py:21-25`):

```python
class StepResult(Observation):
    result: str = ""
    reward: float = 0.0
```

`CounterEnv` returns a reward at EVERY step (`environments/examples.py:69-86`):

```python
if call.name == "increment":
    self._current += 1
    return StepResult(result=f"Counter is now {self._current}.", reward=-0.1)   # per-step reward
elif call.name == "submit":
    if self._current == self._target:
        return StepResult(..., reward=1.0, done=True)
    return StepResult(..., reward=-0.5, done=True)
```

this is the RUNTIME version of what a PRM learns to predict: a scalar reward at each step of a trajectory. a PRM is the learned, differentiable version. it predicts step quality from text alone, without needing the environment.

dense per-token rewards in GRPO (`rl_utils.py:384-417`, `calculate_advantages`):

```python
delta = rewards[:, t] + gamma * nextvalues - values[:, t]
```

the dense-reward machinery a PRM's per-step scores would feed into.

### intuition

- **BT RM:** "How good is this whole answer?" - one scalar, human preference, for ranking.
- **ORM:** "Does this answer end up correct?" - per-token probability, verifiable labels, for scoring.
- **PRM:** "Are the reasoning STEPS sound?" - per-step score, step-level labels, for scoring AND guiding search.

#### key advantage of PRM over ORM
it can CATCH A WRONG STEP THAT LEADS TO A RIGHT ANSWER (or a right step in a wrong chain). because it judges each step, it gives credit/blame where it belongs, not just at the end. this makes it useful for search/decoding. We can prune a branch the moment a step scores low.

### aggregation at inference

score a finished chain-of-thought by aggregating OVER STEPS (not tokens):
- **mean** step score
- **minimum** (fail-fast: penalize if any step is bad)
- **weighted sum favoring later steps**

and for search: prune low-scoring branches as you go.

### ORM vs Value Function - same head, different question

ORMs and value functions look almost identical on the surface: both use the same per-token head, both output a number at every token. but they answer different questions and get their targets from different places.

- **ORM:** at every token, will this completion end up CORRECT? A probability of the final outcome. targets come from **OFFLINE labels** (a verifier or dataset), broadcast to every token, and never change.
- **value function:** at every token, how much REWARD is still coming from here? the expected remaining return `V(s_t) = E[ Σ_{k≥t} γ^{k−t} r_k | s_t ]`. targets come from **ON-POLICY rollouts** and CHANGE as the policy improves.

**the clean way to see it** (from the article): define a dense token reward `r_t = 1[token is correct]` and set `γ = 1`. then the ORM is learning `r_t` (or `p(r_t = 1)`) - "is THIS token correct?" - while the value head is learning the remaining sum `Σ_{k≥t} r_k` - "how many correct tokens are still ahead of me?"

**intuition in one line:** an ORM is a *static judge* ("does this end right?"), a value function is a *live forecast* ("how much reward is left from here?") that updates as the policy improves. value functions are trained on-policy and used to compute advantages `A_t = R̂_t − V_t` for policy gradients.

## inference across reward model types

### code hooks

- **BT RM** = exactly `get_reward` (`model_utils.py:372-436`): the head `model.score` produces a logit at every token, but we **discard all but the EOS position** (lines 421, 432-434). one scalar per completion, no aggregation. the repo's BT training loop even tracks the discrete version (`reward_modeling.py:356`): `accuracy = (chosen_reward > rejected_reward)`.
- **ORM** = the same `get_reward`, but we **keep the per-token logits** (line 417, `(batch, seq, 1)`). the BT path throws them away; the ORM path aggregates them. runtime analog: `apply_verifiable_reward` (`model_utils.py:507`).
- **PRM** = the per-step reward concept. outputs a score at each step boundary (3-class head). at inference you aggregate **over steps**, not tokens. the "prune a branch mid-chain" use is the runtime version of the per-step `StepResult.reward` (`environments/examples.py:69-86`).
- **value function** = the dense-reward machinery. `calculate_advantages` (`rl_utils.py:384-417`): `delta = rewards[:, t] + gamma * nextvalues - values[:, t]`. `values[:, t]` is the value head's prediction; `rewards[:, t]` is observed reward. the value head is trained to predict the *remaining* return, and is subtracted from the observed return to get the advantage `A_t = R̂_t − V_t`.

### the unifying thread

every one of these is the **same backbone + head**, differing only in:
1. **which positions you read** (EOS only / every token / step boundaries / every token)
2. **what the number means** (quality / correctness / step validity / remaining return)
3. **how you aggregate** (none / over tokens / over steps / take last)

and the code hook is always the same place: `get_reward`'s `reward_logits` at every token (`model_utils.py:417`).

### intuition

- **BT RM is a grader.** hand it a whole answer, it gives one score: "how good is this?" use it to pick the best of several answers (best-of-N) or as the final reward in RL. no aggregation. it's the *relative* judge. 
- **ORM is a fortune-teller.** reads the answer token by token, asking at each "is this heading toward a correct final answer?" gives a probability at every token. to score the whole answer you *collapse* those per-token guesses: **mean** = "on average, how confident?"; **min** = "worst moment, tail risk"; **product** = "did it stay confident the whole way?"
- **PRM is a step-by-step inspector.** doesn't wait for the end. scores each reasoning step as it goes. this is what lets us *prune*: abandon a branch the moment a step scores low. aggregate over steps (mean, min, weighted-later), because the unit of judgment is the step, not the token.
- **value function is a live forecast.** not judging quality or correctness at all. It's predicting "how much reward is still coming from here?" trained on-policy, updates as the policy improves. its job is to be a *baseline*: subtract its forecast from the observed return to get the advantage `A_t = R̂_t − V_t`. it's the only one of the four that isn't a reward model. It's the RL critic.

## generative reward modeling (LLM-as-a-judge)

**what:** not a trained RM at all: **prompt an LLM** to judge two completions (MT-Bench style). spawned AlpacaEval, Arena-Hard, WildBench. common trick: **temperature 0** to reduce variance.

**code hook - open-instruct repo actually implements it.** `judge_utils.py:30-49` is the judge prompt, nearly verbatim from MT-Bench:

```python
general_quality_template = """
### Task Description
Please act as an impartial judge and evaluate the quality of the response provided by an
AI assistant to the user query displayed below.
...
- Be as objective as possible. After providing your short explanation, please output a score on a scale of 1 to 10.
...
[Your judgement]
Respond in JSON format. {"REASONING": "[...]", "SCORE": "<your-score>"}"""
```

and `LMJudgeVerifier` (`ground_truth_utils.py:697`) is the repo's actual LLM-as-a-judge verifier. It calls an LLM API, parses the JSON score, and even tracks cost via `PRICE_PER_MILLION_TOKENS` (`judge_utils.py:8-27`). the temperature is configurable (`data_loader.py:463`, `llm_judge_temperature`).

**where better:** no training needed, cheap, flexible, good for *evaluation*.

**where worse / why not used as a reward:** on RM benchmarks they **tend to be behind trained RMs**. LLM-as-a-judge has position/length biases and is expensive at RL scale (you'd call it per-step). it's a *proxy of a proxy* and shows it.

anyway. i still don't fully trust myself on the ORM vs value distinction, and i've rewritten that section twice. but the label-shape thing - pair vs broadcast vs per-step - that one i'm confident about, and it's the thing i'll actually reach for next time i'm staring at a dataset and wondering what model to train.
