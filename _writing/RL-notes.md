---
title: Reinforcement Learning Notes
date: 2026-08-20
description: rl basics notes for later recall
tags: [RL, notes]
---

I've mostly just pasted my notes from reading [policy-gradients](https://rlhfbook.com/c/06-policy-gradients) here, plus some stitched-in intuition to tie the pieces together.

## Markov Decision Process and Trajectories

- **State transitions:** $s_1 \rightarrow s_2 \rightarrow s_3$ with actions $a_1, a_2$
- $P(s_3 \mid s_2, s_1) = P(s_3 \mid s_2)$ - **Markov property:** the future depends on the world only through the present state.
- Rewards live on a probability distribution over the trajectory.
  $$P(\tau) = P(s_1) \prod_{t=1}^T \pi(a_t | s_t)\, p(s_{t+1} | s_t, a_t)$$
  where $\pi$ is the policy (our decision rule, the only thing we control) and $p$ is the
  environment's randomness (dynamics), which we do **not** control.

For an LLM this is almost embarrassingly simple. State = the token sequence so far. Action = next token. Transition = deterministic append, $s_{t+1} = [s_t, a_t]$, so all the randomness is in the policy, none in the environment. One prompt → one completion → one episode, dead at the EOS token. I mean there's complexity of batched stuff and the mult-turn convos but let's stay focused on the basics for now.

## The Core RL Objective

So this is the thing everything builds on. just one expectation.

$$J(\theta) = \mathbb{E}_{\tau \sim p_\theta}[R(\tau)] \quad \text{- (1)}$$

- $\tau = (s_0, a_0, s_1, a_1, \dots)$ trajectory
- $R(\tau) = \sum_{t=0}^\infty r_t$ total reward (for now: all rewards weighted equally; discounting arrives later)
- $p_\theta$ means the policy $\pi_\theta$ is baked into the trajectory distribution.

We want the max over $\theta$. This average is never what you literally compute. You estimate it with a Monte-Carlo mean over a batch of B completions:

$$\hat{J}(\theta) = \frac{1}{B} \sum_{i=1}^B R(x_i, y_i) \quad \text{- (2)}$$

So $y_i$ is a completion, $x_i$ is a prompt, $R$ is the reward. In RLHF there's no per-step reward to add up. the reward model scores the whole completion, one terminal scalar. That's why the whole temporal chain collapses and why discount $\gamma$ ends up at 1.0.

You can also write it as an integral, all possible futures:

from (1)

$$J(\theta) = \int_\tau p_\theta(\tau) R(\tau)\, d\tau \quad \text{- (3)}$$


$$p_\theta(\tau) = d_0(s_0) \prod_{t=0}^\infty \pi_\theta(a_t | s_t)\, p(s_{t+1} | s_t, a_t) \quad \text{- (4)}$$

The integral form matters more than you'd think. because to differentiate it you hit the gradient trick.

## The Gradient Trick

The whole reason RL is tractable. We want $\nabla_\theta J(\theta)$. Product rule on (3):

$$\nabla_\theta J(\theta) = \int_\tau \nabla_\theta p_\theta(\tau)\, R(\tau)\, d\tau \quad \text{- (5)}$$

Now the log-derivative identity, the load-bearing line:

$$\nabla_\theta \log f = \frac{1}{f}\nabla_\theta f \quad\Rightarrow\quad \nabla_\theta f = f\, \nabla_\theta \log f$$

Multiply and divide by $p_\theta(\tau)$:

$$\nabla_\theta J(\theta)
= \int_\tau p_\theta(\tau)\, \underbrace{\frac{\nabla_\theta p_\theta(\tau)}{p_\theta(\tau)}}_{\nabla_\theta \log p_\theta(\tau)}\, R(\tau)\, d\tau
= \int_\tau p_\theta(\tau)\, R(\tau)\, \nabla_\theta \log p_\theta(\tau)\, d\tau$$

It's a sleight of hand. The integrand is now $R(\tau)\nabla_\theta \log p_\theta(\tau)$ weighted by $p_\theta(\tau)$ - that's an **expectation**:

$$\mathbb{E}_{\tau \sim p_\theta}[f(\tau)] = \int_\tau f(\tau) p_\theta(\tau)\, d\tau$$
$$\Rightarrow \nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim p_\theta}\big[R(\tau)\, \nabla_\theta \log p_\theta(\tau)\big] \quad \text{- (6)}$$

An expectation means you can sample it. roll out trajectories, prompt the model, average. no integration over futures needed.

Now expand $\log p_\theta(\tau)$ from (4) and differentiate:

$$\nabla_\theta \log p_\theta(\tau)
= \nabla_\theta \log d_0(s_0)
+ \sum_{t=0}^\infty \nabla_\theta \log \pi_\theta(a_t | s_t)
+ \sum_{t=0}^\infty \nabla_\theta \log p(s_{t+1} | s_t, a_t)$$

Look at the first and last terms. no $\theta$ in them. the environment doesn't care about our policy parameters. so they vanish, and the happy accident lands:

$$\nabla_\theta \log p_\theta(\tau) = \sum_{t=0}^\infty \nabla_\theta \log \pi_\theta(a_t | s_t) \quad \text{- (7)}$$

For an LLM the dynamics are deterministic anyway, so $p(s_{t+1}\mid s_t,a_t)$ is constant and the cancellation is exact.

## Generalized Policy Gradient

Substitute (7) into (6):

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim p_\theta}\!\left[\sum_{t=0}^\infty R(\tau)\, \nabla_\theta \log \pi_\theta(a_t | s_t)\right]$$

Then the generalization. swap the total return for a per-action scalar $\Psi_t$:

$$\nabla_\theta J(\theta) = g = \mathbb{E}_{\tau \sim p_\theta}\!\left[\sum_{t=0}^\infty \Psi_t\, \nabla_\theta \log \pi_\theta(a_t | s_t)\right] \quad \text{- (8)}$$

And the update is just gradient ascent:

$$\Delta\theta \propto \Psi_t\, \nabla_\theta \log \pi_\theta(a_t | s_t), \qquad \theta \leftarrow \theta + \alpha \nabla_\theta J(\theta) \quad \text{- (9)}$$

$\nabla_\theta \log \pi$ points toward making $a_t$ more likely in parameter space. scale by $\Psi_t$, how good it was, and good actions rise while bad ones sink.

In code we never assemble that gradient by hand. we write the *loss* whose gradient it is and let autodiff do the rest:

$$L^{\text{PG}} = -\mathbb{E}_{\tau \sim p_\theta}\!\left[\sum_{t=0}^\infty \Psi_t\, \log \pi_\theta(a_t | s_t)\right]$$

the minus sign is just because we're maximizing the objective. and one thing worth pointing out, since it's the whole reason the log-derivative trick exists: sampling a token is **not differentiable**. $a_t \sim \pi_\theta$ is a categorical draw - there's no gradient path from $\theta$ through the sampled index. the trick sidesteps that by moving the gradient onto the *log-probability* of the action that was actually sampled, which is a smooth function of $\theta$. we never differentiate through the choice, only through its probability. that's the non-differentiable-samples problem in one line.

But here's the annoying part. $\Psi_t$ can be a whole zoo of things, and they're all valid, unbiased gradients. they only differ in variance:

| $\Psi_t$                | Meaning                       | Property                                                     |
| ----------------------- | ----------------------------- | ------------------------------------------------------------ |
| $\sum r_t$              | total trajectory return       | high variance: reward noise hits every action                |
| $G_t$ (return from $t$) | discounted return at step $t$ | better - rewards *before* $t$ shouldn't credit action at $t$ |
| $G_t - b(s_t)$          | return minus **baseline**     | same expectation, lower variance                             |
| $Q^\pi(s_t, a_t)$       | state-action value            | = expected return from $(s_t,a_t)$                           |
| $A^\pi(s_t, a_t)$       | **advantage**                 | best variance; the practical choice                          |

The table is the whole reason there are so many methods. they're all the same formula with different credit assignment.

## Why Baselines Don't Bias (and why they reduce variance)

a baseline $b(s_t)$ that doesn't depend on $a_t$ vanishes in expectation:

$$\begin{aligned}
\mathbb{E}_{a_t \sim \pi_\theta}\!\left[b(s_t)\nabla_\theta \log\pi_\theta(a_t|s_t)\right]
&= b(s_t)\int \pi_\theta(a|s_t)\nabla_\theta \log\pi_\theta(a|s_t)\, da \\
&= b(s_t)\nabla_\theta \!\int \pi_\theta(a|s_t)\, da \\
&= b(s_t)\nabla_\theta[1] = 0
\end{aligned}$$

So $\mathbb{E}[G_t] = \mathbb{E}[G_t - b_t]$: the gradient estimate is **unbiased** for any baseline. That's the formal bit.

LLM rewards are almost always positive, so every action has $G_t > 0$, and all of them look good. the gradient pushes everything up regardless of quality. subtract a baseline (for ex, the average reward) and now the scale centers on 0, only actions better than average get pushed. $\mathrm{Var}[G_t - b_t] < \mathrm{Var}[G_t]$ when $b_t$ is a decent estimate of $\mathbb{E}[G_t]$.

For RLOO the baseline is dead simple, per prompt, no learned critic - derived in its own section below.

## RLOO (Reinforce Leave-One-Out): the per-prompt baseline

RLOO is the rung between REINFORCE and PPO on the ladder. no learned critic, no value network - just sample $K$ completions per prompt and build the baseline from the other $K-1$. this lays the groundwork for later Dr. GRPO improvements.

### The intuition: why "leave one out"?

In policy gradient we update on the advantage - how much better this action was than what we normally expect:

$$\text{Advantage} = \text{Reward} - \text{Baseline}$$

Ideally the baseline is the expected value of the state, $V(s)$. with $K$ sampled trajectories the obvious baseline is just the average reward of all $K$ samples.

But there's a constraint in policy gradients: the baseline must **not** depend on the specific action $a_k$ you're evaluating. if the baseline includes $R(s, a_k)$, it introduces bias - the action is being compared partially against itself, and the gradient trick's unbiased-baseline proof (above) no longer holds because $b$ is no longer independent of $a$.

The fix: leave $a_k$ out of the baseline. estimate the expected reward using only the *other* trajectories.

### The reason for the $K-1$ denominator

$K$ total samples, leave one out (the current $a_k$), you have exactly $K-1$ left. to get the true average of those remaining, sum their rewards and divide by the number of items in the sum:

$$b(s,a_k) = \frac{1}{K-1} \sum_{i \neq k} R(s,a_i)$$

### Deriving the final formula

Let $\bar{R}$ be the average of all $K$ rewards:

$$\bar{R} = \frac{1}{K} \sum_{i=1}^{K} R(s,a_i)$$

so the total sum is $K\bar{R}$. the leave-one-out sum is the total minus the one we dropped:

$$\sum_{i \neq k} R(s,a_i) = K \bar{R} - R(s,a_k)$$

Plug into the advantage:

$$A(s,a_k) = R(s,a_k) - \frac{1}{K-1} \left( K \bar{R} - R(s,a_k) \right)$$

Distribute:

$$A(s,a_k) = R(s,a_k) - \frac{K}{K-1}\bar{R} + \frac{1}{K-1}R(s,a_k)$$

Factor $R(s,a_k)$ from the first and third terms:

$$A(s,a_k) = R(s,a_k) \left( 1 + \frac{1}{K-1} \right) - \frac{K}{K-1}\bar{R}$$

And $1 + \frac{1}{K-1} = \frac{K}{K-1}$, so:

$$A(s,a_k) = \frac{K}{K-1} R(s,a_k) - \frac{K}{K-1} \bar{R}$$

Factor out $\frac{K}{K-1}$:

$$A(s,a_k) = \frac{K}{K-1} \left( R(s,a_k) - \frac{1}{K}\sum_{i=1}^{K} R(s,a_i) \right)$$

That's the final form. the inner term is just $R(s,a_k) - \bar{R}$, the standard mean-baseline advantage, and $\frac{K}{K-1}$ is a constant scaler. so RLOO = REINFORCE-with-mean-baseline, rescaled. no critic, no extra network, one prompt's worth of samples is enough. the only cost is you need $K \ge 2$ completions per prompt, and variance drops as $K$ grows.

## Discounting $\gamma$ and Return

$$G_t = r_t + \gamma G_{t+1} = \sum_{k=0}^\infty \gamma^k r_{t+k}$$

$G_t$ is the return, what we maximize. $\gamma \in [0,1]$ - convergent, and near-term reward weighs more. For LLMs, $\gamma = 1.0$, no discount. an episode is one finite completion, so there's no infinite horizon to worry about, and discounting would just miscredit the later tokens. $G_t \equiv R(\tau)$.

The value function is the expected return from $s$:

$$V^\pi(s) = \mathbb{E}[G_t \mid S_t = s]$$

## Advantage and the Bellman Link (why TD is a shortcut)

The awkward fact under all of this: ideally every token would get its own reward. that "th" was a good token, that wrong digit was a bad one. we don't get that. the reward model scores the whole completion and hands back one scalar at the end, so every token in the sequence inherits the *same* number. the credit assignment is completely flat. push on the raw reward and you're telling all $T$ tokens they were equally responsible for an outcome most of them had nothing to do with.

So don't push on the raw reward. push on how much better than expected this trajectory turned out - reward measured against a baseline. that's the advantage: what actually happened minus what we expected to get here. in shorthand $A = R(\tau) - V(s_t)$, but the proper object is the state-action gap:

$$A^\pi(s_t, a_t) = Q^\pi(s_t, a_t) - V^\pi(s_t)$$

How much better action $a_t$ is than the policy's average. and "average" means the average over the kinds of tasks this model sees, not just this one prompt - $V(s_t)$ is a general prior about how well the model does from here, and the advantage is this particular action beating that prior. but fitting a full $Q$ is expensive, so instead we lean on the Bellman equation - $Q$ at $(s_t,a_t)$ is the immediate reward plus the discounted value of the next state:

$$Q^\pi(s_t, a_t) = \mathbb{E}\big[r_t + \gamma V^\pi(s_{t+1})\big]$$

and the advantage collapses to the Temporal-Difference residual, which only needs a value estimate, one network:

$$A(s_t, a_t) = r_t + \gamma V(s_{t+1}) - V(s_t)$$

The reading: $r_t + \gamma V(s_{t+1})$ is a sample of how good it turned out *having taken $a$*. $V(s_t)$ is the value before choosing. the gap is exactly how much this action deviated from average. train $V$ to minimize TD error and you have your critic.

### GAE (Generalized Advantage Estimation) - the one PPO actually uses

Single-step TD is noisy and biased. GAE exponentially weights TD residuals to trade the two:

$$A_t^{\text{GAE}(\gamma,\lambda)} = \sum_{k=0}^{T-t} (\gamma\lambda)^k\, \delta_{t+k}, \qquad \delta_{t+k} = r_{t+k} + \gamma V(s_{t+k+1}) - V(s_{t+k})$$

- **$\lambda \to 0$:** just the 1-step TD residual - low variance, high bias.
- **$\lambda \to 1$:** the full Monte-Carlo return - unbiased but high variance.

So $\lambda$ trades them. For a completion, only the terminal reward is nonzero, so $\delta_t = \gamma V(s_{t+1}) - V(s_t)$ for $t<T$ and $\delta_T = R - V(s_T)$. I still don't have a great gut feel for where the balance should land. I'm still unsure about this.. needs a bit more work.

## Monte-Carlo Estimation and Importance Sampling (the two tools we keep leaning on)

Before the off-policy jump, the two tools it's built from. both are things i'd half-known for ages and never actually written down, which is probably why the PPO ratio never sat right with me.

### Monte-Carlo: an integral is just an average

The objective is an integral over all possible futures, and nobody can enumerate those - the state is a whole token sequence, astronomically many of them. so we approximate the integral with a sample average. for any $p$ and $f$:

$$\int p(x) f(x)\, dx = \mathbb{E}_p[f(x)] \approx \frac{1}{N} \sum_{i=1}^N f(x_i), \qquad x_i \sim p(x)$$

$x$ is usually multidimensional, so what we're approximating is a high-dimensional expectation - and the sample average still converges as $N$ grows. the central limit theorem tells you *how*: the estimate is approximately normal around the true mean $\mu = \mathbb{E}_p[f(x)]$ with variance $\frac{1}{N}\operatorname{Var}_p[f(x)]$. that $1/N$ is the whole reason "just sample more" works, and the variance term is why the *choice* of estimator matters so much. two estimators can be unbiased for the same thing and still have wildly different variance, and at finite $N$ that's the entire game.

### Importance sampling: sample from the wrong distribution on purpose

Sometimes $p$ is hard to sample from but easy to *evaluate*, and we happen to have some other $q$ that's easy to sample. multiply and divide by $q$ and the expectation is unchanged:

$$\mathbb{E}_p[f(x)] = \int p(x) f(x)\, dx = \int q(x)\, \frac{p(x)}{q(x)} f(x)\, dx = \mathbb{E}_q\!\left[\frac{p(x)}{q(x)} f(x)\right] \quad \text{- (10)}$$

so we can draw $x_i \sim q$ and average $\frac{p(x_i)}{q(x_i)} f(x_i)$ as if the samples came from $p$. the ratio $p/q$ is the **importance weight**: it corrects each sample for being over- or under-represented under the distribution we actually care about.

when does it help? when $p$ is hard to sample, $p$ is easy to evaluate, $q$ is easy to both evaluate and sample, and - the one people skip - you *choose* $q$ to put mass where $\lvert p(x)f(x)\rvert$ is large. that last condition is what keeps the weights from exploding. if $q$ is tiny exactly where $pf$ is big, you get a few enormous weights and the estimator's variance blows up. the whole art is picking $q$ close enough to $p$ that the weights stay tame. when it works: $\operatorname{Var}_q[\frac{p}{q}f] < \operatorname{Var}_p[f]$.

Hold that. "sample from the old policy, reweight to pretend it came from the new one" is exactly PPO's move, and the ratio $\pi_\theta/\pi_{\theta_{\text{old}}}$ is just $p/q$ with policy names filled in.

## Off-Policy Reuse: where the ratio comes from, and why it goes per-token

This is the step the notes asserted with no derivation, and the one that bugged me the most. here it is split into two questions: *how* do we reuse old data, and *why* does PPO end up using one ratio per token.

the short version, before the details: the exact off-policy correction is a **product** of per-token ratios, which is numerically useless. PPO doesn't use it. instead PPO builds a *local approximation* of the objective, and inside that approximation the only thing that needs reweighting is the single action at each state - and for an LLM, one state-action is one token. the clip then keeps the approximation honest. now the details.

### the on-policy problem

Our advantage-form gradient is

$$\nabla_\theta J(\theta) = \mathbb{E}_{a_t \sim \pi_\theta(\cdot | s_t)}\!\left[\nabla_\theta \log \pi_\theta(a_t | s_t)\, A(s_t, a_t)\right]$$

the expectation is under $\pi_\theta$ - the policy we're updating. so every gradient step needs fresh rollouts: the moment $\theta$ moves, the data is stale. rollout, one step, throw it away, rollout again. that's the annoyance we want to fix.

### the naive fix: importance-sample the whole trajectory

Resample under the old policy. the thing being resampled is a whole trajectory, so the importance weight is over trajectories. and since the initial-state and dynamics terms are the same top and bottom, they cancel, leaving a **product of one ratio per token**:

$$\frac{p_\theta(\tau)}{p_{\theta_{\text{old}}}(\tau)} = \prod_{t=0}^{T} \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)} \quad \text{- (11)}$$

this is the mathematically correct answer, and it's also a dead end. multiplying one ratio per token over a long completion either collapses toward zero or explodes, and the variance is unusable. this is exactly the failure mode GSPO and CISPO were later built to fix, so nobody estimates the true off-policy gradient this way.

### what PPO actually does: approximate the objective instead

PPO avoids the product entirely. it doesn't correct an off-policy gradient - it builds a *local approximation* of the objective that's only valid near the old policy, then takes steps that stay near it.

the approximation is a standard result (Kakade & Langford 2002; Schulman et al. 2015, TRPO): the performance gap between two policies can be written as an expectation over **states** of the advantage. if we assume the state distribution doesn't move, only the action choice needs reweighting, and we get the surrogate

$$L(\theta) = \mathbb{E}_{s \sim \rho_{\text{old}},\, a \sim \pi_{\text{old}}}\!\left[\frac{\pi_\theta(a|s)}{\pi_{\text{old}}(a|s)}\, A(s,a)\right]$$

read the expectation closely, because it's the whole answer. the states are taken from the old rollout and left alone; only the action at each state gets reweighted. and because a state in an LLM is just the token prefix, "one action at one state" **is** one token. that's where the per-token ratio comes from: it's per-state, and one state is one token position.

(the formal backing for dropping the rest is per-decision importance sampling, Precup, Sutton & Singh 2000: ratios for tokens *after* $t$ average to 1, so the weight truncates at the prefix. the CPI result above handles the state-distribution half.)

### and *that's* why the clip exists

$L(\theta)$ is a local approximation, so picture a tangent line. at $\theta = \theta_{\text{old}}$ every ratio is exactly 1, the state distribution really is the old one, and $L$'s gradient equals the true policy gradient. step far away and the approximation drifts, because the state distribution is quietly moving and we told ourselves it doesn't.

so the clip isn't a separate hack bolted next to the per-token ratio - it's what keeps us inside the region where the per-token approximation is honest. that's the closure that finally made it click.

the actual loss, then:

$$L^{\text{IS}}(\theta) = \mathbb{E}\!\left[\frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}\, A_t\right] = \mathbb{E}[\rho_t\, A_t] \quad \text{- (12)}$$

with the trajectory structure now hidden inside how $A_t$ was estimated - that's the GAE job from the previous section.

One aside worth keeping: the ratio's *granularity* is a design choice, not a law. GSPO (Zheng et al. 2025) uses one length-normalized ratio per *sequence*; CISPO clips the weight itself instead of the objective, so no token's gradient is ever zeroed out. per-token is just what PPO settled on.

## PPO (Proximal Policy Optimization)

So the setup is: we want to reuse a batch, the exact correction is an intractable product of ratios, and the per-token surrogate is only trustworthy near the old policy. PPO takes that surrogate and *enforces* the "near the old policy" part with a clip. (skipping TRPO here - it solves an explicit KL-constrained trust region; PPO just replaces the constraint with something cheaper that does the same job.)

The ratio again, in PPO's own notation:

$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t|s_t)} \quad \text{- (13)}$$

- $r_t > 1$: new policy more likely than old.
- $r_t < 1$: less likely.
- The ratio is a cheap divergence estimate.

The clipped surrogate objective:

$$J^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}\!\left[\min\!\Big( r_t(\theta)\hat{A}_t,\; \text{clip}(r_t(\theta),\, 1-\epsilon,\, 1+\epsilon)\,\hat{A}_t \Big)\right] \quad \text{- (14)}$$

The clip is the whole point. the $\min$ picks the smaller update. good action ($\hat{A}_t>0$): ratio capped at $1+\epsilon$, rewards the move but won't over-commit. bad action ($\hat{A}_t<0$): floored at $1-\epsilon$, pushes away but no further. Outside that window the gradient is zero, so there's no incentive to drift past it. net effect: the policy can't wander far from the old one - a trust region without solving a KL constraint.

**PPO flow:**
1. $s_t \to$ **Policy (Actor)** $\to a_t$
2. $a_t \to$ **Reward Model** (frozen) & **Reference Model** (frozen) $\to \hat{r} = r - \beta\cdot\text{KL}$
3. Simultaneously $s_t \to$ **Value Model (Critic)** $\to V$
4. $\hat r$ and $V$ $\to$ **GAE** $\to A_t$
5. $A_t$ (via the clipped objective) updates the **Actor**.

## KL Penalty: staying near the reference

We freeze the pretrained model as $\pi_{\text{ref}}$ and penalize drift from it:

$$\hat{r} = r - \beta\cdot \mathrm{KL}\big(\pi_\theta(\cdot|s)\,\big\|\, \pi_{\mathrm{ref}}(\cdot|s)\big), \qquad \mathrm{KL}(P\|Q) = \sum_a P(a)\log\frac{P(a)}{Q(a)}$$

Why: reward hacking is real. a policy free to deviate can overfit the reward model - fluent but wrong, or degenerate tokens that game the RM. The KL anchors it to the sensible pretrained distribution. And it bounds each step, complementing the clip.

## GRPO (Group Relative Policy Optimization)

PPO works but it's a little unstable, and a lot of that traces back to the critic. you're training a second full model to predict $V(s_t)$ from an LM backbone, there's no settled best practice for how to do that, and every policy update invalidates the value targets it was just fit to. GRPO's pitch is blunt: *delete the critic, and get the baseline from the group instead.*

The idea is almost embarrassingly simple. for one prompt, sample a group of $G$ completions and score them with the reward model. now you have $G$ reward values for the same state. use their mean as the baseline and normalize by their spread:

$$\text{Advantage}_i = \frac{R(z_i) - \mu\big(\{R(z_1),\dots,R(z_G)\}\big)}{\sigma\big(\{R(z_1),\dots,R(z_G)\}\big)} \quad \text{- (15)}$$

that's it - no value network, no per-token critic, no Bellman bootstrap. the same advantage is broadcast to every token in completion $i$. it's a pure Monte-Carlo baseline, and the cleanest possible form of the "compare each answer to its peers" intuition: the model becomes more like the completions that scored above the group average and less like the ones below.

Two things worth noticing, because they connect back:

- **It's RLOO's cousin.** the RLOO section already worked out that a leave-one-out mean baseline equals $\frac{K}{K-1}(R - \bar R)$ - a rescale of the plain mean baseline. Dr. GRPO (Liu et al. 2025) drops the std normalization and lands, up to that constant scale, on the RLOO advantage. so the rung from RLOO to GRPO is mostly "drop the std, add the clip", and the constant doesn't matter because implementations renormalize advantages anyway.
- **Why the std normalization is debated.** dividing by the group std *rewards* prompts with low reward variance - if everyone in the group is right or everyone is wrong, the std is small and the advantage gets amplified. Dr. GRPO removes it to kill that bias, at the cost of down-weighting the hard prompts where only one sample in many gets it right. those may be the most valuable samples of all. known trade-off, not a bug.

The surrogate loss is the same clipped objective as PPO, with the group advantage substituted in and the KL penalty moved into the loss (canonical GRPO) rather than folded into the reward:

$$J(\theta) = \frac{1}{G}\sum_{i=1}^G \left( \frac{1}{|a_i|}\sum_{t=1}^{|a_i|} \min\!\Big( \rho_{i,t}(\theta) A_i,\; \text{clip}(\rho_{i,t}(\theta),\, 1-\epsilon,\, 1+\epsilon) A_i \Big) - \beta\, \mathcal{D}_{\text{KL}}\!\big(\pi_\theta \,\|\, \pi_{\text{ref}}\big) \right) \quad \text{- (16)}$$

with $\rho_{i,t}(\theta)$ the per-token ratio broadcast across the completion - so all the "why per-token" machinery from the previous section applies unchanged. and the reward being verifiable (right/wrong) makes the whole thing especially clean: you don't need a calibrated critic to know that one correct answer out of eight was much better than the rest. that's why it took over for reasoning and RLVR.

## The Progression Ladder (VPG → REINFORCE → RLOO → PPO → GRPO)

| Method               | $\Psi_t$                          | Fix vs previous                                  |
| -------------------- | --------------------------------- | ------------------------------------------------ |
| **Vanilla PG (VPG)** | $G_t$ (full return)               | baseline objective; high variance                |
| **REINFORCE**        | $G_t - b(s_t)$                    | adds a baseline → less variance, still per-step  |
| **RLOO**             | per-prompt leave-one-out baseline | per-prompt, no learned critic                    |
| **PPO**              | clipped-surrogate + GAE           | trust region + critic → stable, but the critic is a second model to train |
| **GRPO**             | group-relative advantage          | drops the critic; the baseline falls out of the sampled group |

it's one chain, each rung shaving variance or adding stability. and the "five approaches" everyone lists at the start - imitation, policy-gradient, actor-critic, value-based, model-based - they all reduce to the same objective plus the gradient trick. they only differ in how $\Psi$ gets estimated and how hard you update. RLHF proper ships the actor-critic one (PPO); the reasoning/RLVR wave mostly ships the critic-free one (GRPO).

---

I'll keep adding the other RL algos here as I go along...

---

resources: 
 - http://joschu.net/blog/kl-approx.html
 - https://github.com/verl-project/verl/pull/2953#issuecomment-3162113848
 - https://fengyao.notion.site/off-policy-rl
 - https://rlhfbook.com/c/06-policy-gradients
 - Williams (1992), *Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning* - the REINFORCE paper
 - Kakade & Langford (2002), *Approximately Optimal Approximate Reinforcement Learning* - performance-difference lemma
 - Schulman et al. (2015), *Trust Region Policy Optimization* (TRPO) - the CPI surrogate and trust region; https://arxiv.org/abs/1502.05477
 - Schulman et al. (2016), *High-Dimensional Continuous Control Using Generalized Advantage Estimation* (GAE); https://arxiv.org/abs/1506.02438
 - Schulman et al. (2017), *Proximal Policy Optimization Algorithms* (PPO); https://arxiv.org/abs/1707.06347
 - Precup, Sutton & Singh (2000), *Eligibility Traces for Off-Policy Policy Evaluation* - per-decision importance sampling
 - Shao et al. (2024), *DeepSeekMath* - GRPO; https://arxiv.org/abs/2402.03300
 - Liu et al. (2025), *Understanding R1-Zero-Like Training: A Critical Perspective* - Dr. GRPO; https://arxiv.org/abs/2503.20783
 - Zheng et al. (2025), *Group Sequence Policy Optimization* (GSPO); https://arxiv.org/abs/2507.18071
