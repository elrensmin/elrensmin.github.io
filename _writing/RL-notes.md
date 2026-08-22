---
title: Reinforcement Learning Notes
date: 2026-08-20
description: rl basics notes for later recall
tags: [RL, notes]
---

I've mostly just pasted ny notes here so I didn't bother to change the eq number from my notes.. 

## Markov Decision Process and Trajectories

- **State transitions:** $s_1 \rightarrow s_2 \rightarrow s_3$ with actions $a_1, a_2$
- $P(s_3 \mid s_2, s_1) = P(s_3 \mid s_2)$ - **Markov property:** the future depends on the world only through the present state.
- Rewards live on a probability distribution over the trajectory.
  $$P(\tau) = P(s_1) \prod_{t=1}^T \pi(a_t | s_t)\, p(s_{t+1} | s_t, a_t)$$
  where $\pi$ is the policy (our decision rule, the only thing we control) and $p$ is the
  environment's randomness (dynamics), which we do **not** control.

For an LLM this is almost embarrassingly simple. State = the token sequence so far. Action = next token. Transition = deterministic append, $s_{t+1} = [s_t, a_t]$, so all the randomness is in the policy, none in the environment. One prompt → one completion → one episode, dead at the EOS token. Nothing else to it.

I mean there's complexity of batched stuff and the mult-turn convos but let's stay focused on the basics for now.

## The Core RL Objective

So this is the thing everything builds on, and it took me a while to realize it's just one expectation.

$$J(\theta) = \mathbb{E}_{\tau \sim p_\theta}[R(\tau)] \quad \text{- (1)}$$

- $\tau = (s_0, a_0, s_1, a_1, \dots)$ trajectory
- $R(\tau) = \sum_{t=0}^\infty r_t$ total reward (for now: all rewards weighted equally; discounting arrives later)
- $p_\theta$ means the policy $\pi_\theta$ is baked into the trajectory distribution.

We want the max over $\theta$. And here's the thing - this average is never what you literally compute. You estimate it with a Monte-Carlo mean over a batch of B completions:

$$\hat{J}(\theta) = \frac{1}{B} \sum_{i=1}^B R(x_i, y_i) \quad \text{- (4)}$$

So $y_i$ is a completion, $x_i$ is a prompt, $R$ is the reward. In RLHF there's no per-step reward to add up. the reward model scores the whole completion, one terminal scalar. That's why the whole temporal chain collapses and why discount $\gamma$ ends up at 1.0.

You can also write it as an integral, all possible futures:

from (1)

$$J(\theta) = \int_\tau p_\theta(\tau) R(\tau)\, d\tau \quad \text{- (7)}$$


$$p_\theta(\tau) = d_0(s_0) \prod_{t=0}^\infty \pi_\theta(a_t | s_t)\, p(s_{t+1} | s_t, a_t) \quad \text{- (8)}$$

The integral form matters more than you'd think. because to differentiate it you hit the gradient trick.

## The Gradient Trick (the load-bearing move)

The whole reason RL is tractable. We want $\nabla_\theta J(\theta)$. Product rule on (7):

$$\nabla_\theta J(\theta) = \int_\tau \nabla_\theta p_\theta(\tau)\, R(\tau)\, d\tau \quad \text{- (9)}$$

Now the log-derivative identity, the load-bearing line:

$$\nabla_\theta \log f = \frac{1}{f}\nabla_\theta f \quad\Rightarrow\quad \nabla_\theta f = f\, \nabla_\theta \log f$$

Multiply and divide by $p_\theta(\tau)$:

$$\nabla_\theta J(\theta)
= \int_\tau p_\theta(\tau)\, \underbrace{\frac{\nabla_\theta p_\theta(\tau)}{p_\theta(\tau)}}_{\nabla_\theta \log p_\theta(\tau)}\, R(\tau)\, d\tau
= \int_\tau p_\theta(\tau)\, R(\tau)\, \nabla_\theta \log p_\theta(\tau)\, d\tau$$

It's a sleight of hand. i first thought we needed to know the dynamics to evaluate that. no. watch. The integrand is now $R(\tau)\nabla_\theta \log p_\theta(\tau)$ weighted by $p_\theta(\tau)$ - that's an **expectation**:

$$\mathbb{E}_{\tau \sim p_\theta}[f(\tau)] = \int_\tau f(\tau) p_\theta(\tau)\, d\tau$$
$$\Rightarrow \nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim p_\theta}\big[R(\tau)\, \nabla_\theta \log p_\theta(\tau)\big] \quad \text{- (10)}$$

An expectation means you can sample it. roll out trajectories, prompt the model, average. no integration over futures needed.

Now expand $\log p_\theta(\tau)$ from (8) and differentiate:

$$\nabla_\theta \log p_\theta(\tau)
= \nabla_\theta \log d_0(s_0)
+ \sum_{t=0}^\infty \nabla_\theta \log \pi_\theta(a_t | s_t)
+ \sum_{t=0}^\infty \nabla_\theta \log p(s_{t+1} | s_t, a_t)$$

Look at the first and last terms. no $\theta$ in them. the environment doesn't care about our policy parameters. so they vanish, and the happy accident lands:

$$\nabla_\theta \log p_\theta(\tau) = \sum_{t=0}^\infty \nabla_\theta \log \pi_\theta(a_t | s_t) \quad \text{- (11)}$$

For an LLM the dynamics are deterministic anyway, so $p(s_{t+1}\mid s_t,a_t)$ is constant and the cancellation is exact.

## Generalized Policy Gradient

Substitute (11) into (10):

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim p_\theta}\!\left[\sum_{t=0}^\infty R(\tau)\, \nabla_\theta \log \pi_\theta(a_t | s_t)\right]$$

Then the generalization. swap the total return for a per-action scalar $\Psi_t$:

$$\nabla_\theta J(\theta) = g = \mathbb{E}_{\tau \sim p_\theta}\!\left[\sum_{t=0}^\infty \Psi_t\, \nabla_\theta \log \pi_\theta(a_t | s_t)\right] \quad \text{- (12)}$$

And the update is just gradient ascent:

$$\Delta\theta \propto \Psi_t\, \nabla_\theta \log \pi_\theta(a_t | s_t), \qquad \theta \leftarrow \theta + \alpha \nabla_\theta J(\theta) \quad$$

$\nabla_\theta \log \pi$ points toward making $a_t$ more likely in parameter space. scale by $\Psi_t$, how good it was, and good actions rise while bad ones sink. clean.

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

I had to look this up twice before I believed the baseline really drops out. a baseline $b(s_t)$ that doesn't depend on $a_t$ vanishes in expectation:

$$\begin{aligned}
\mathbb{E}_{a_t \sim \pi_\theta}\!\left[b(s_t)\nabla_\theta \log\pi_\theta(a_t|s_t)\right]
&= b(s_t)\int \pi_\theta(a|s_t)\nabla_\theta \log\pi_\theta(a|s_t)\, da \\
&= b(s_t)\nabla_\theta \!\int \pi_\theta(a|s_t)\, da \\
&= b(s_t)\nabla_\theta[1] = 0
\end{aligned}$$

So $\mathbb{E}[G_t] = \mathbb{E}[G_t - b_t]$: the gradient estimate is **unbiased** for any baseline. That's the formal bit.

LLM rewards are almost always positive, so every action has $G_t > 0$, and all of them look good. the gradient pushes everything up regardless of quality. noise. subtract a baseline (for ex, the average reward) and now the scale centers on 0, only actions better than average get pushed. $\mathrm{Var}[G_t - b_t] < \mathrm{Var}[G_t]$ when $b_t$ is a decent estimate of $\mathbb{E}[G_t]$.

For RLOO the baseline is dead simple, per prompt, no learned critic. generate $K$ completions, baseline = mean of the other $K-1$:

$$b(s, a_k) = \frac{1}{K-1}\sum_{i\ne k} R(s_i, a_i), \qquad A(s,a_k) = R(s,a_k) - b(s,a_k)$$

## Discounting $\gamma$ and Return

$$G_t = r_t + \gamma G_{t+1} = \sum_{k=0}^\infty \gamma^k r_{t+k}$$

$G_t$ is the return, what we maximize. $\gamma \in [0,1]$ - convergent, and near-term reward weighs more. For LLMs, $\gamma = 1.0$, no discount. an episode is one finite completion, so there's no infinite horizon to worry about, and discounting would just miscredit the later tokens. $G_t \equiv R(\tau)$.

The value function is the expected return from $s$:

$$V^\pi(s) = \mathbb{E}[G_t \mid S_t = s]$$

## Advantage and the Bellman Link (why TD is a shortcut)

This is the jump that tripped me. advantage:

$$A^\pi(s_t, a_t) = Q^\pi(s_t, a_t) - V^\pi(s_t)$$

How much better action $a_t$ is than the policy's average. but fitting a full $Q$ is expensive, so instead we lean on the Bellman equation - $Q$ at $(s_t,a_t)$ is the immediate reward plus the discounted value of the next state:

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

## PPO (Proximal Policy Optimization)

Vanilla steps update from fresh trajectories each time and the variance gets unstable as the policy shifts. PPO reuses a batch via an importance-sampling ratio:

$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t|s_t)}$$

- $r_t > 1$: new policy more likely than old.
- $r_t < 1$: less likely.
- The ratio is a cheap divergence estimate.

The clipped surrogate objective:

$$J^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}\!\left[\min\!\Big( r_t(\theta)\hat{A}_t,\; \text{clip}(r_t(\theta),\, 1-\epsilon,\, 1+\epsilon)\,\hat{A}_t \Big)\right]$$

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

## The Progression Ladder (VPG → REINFORCE → RLOO → PPO)

| Method               | $\Psi_t$                          | Fix vs previous                                  |
| -------------------- | --------------------------------- | ------------------------------------------------ |
| **Vanilla PG (VPG)** | $G_t$ (full return)               | baseline objective; high variance                |
| **REINFORCE**        | $G_t - b(s_t)$                    | adds a baseline → less variance, still per-step  |
| **RLOO**             | per-prompt leave-one-out baseline | per-prompt, no learned critic                    |
| **PPO**              | clipped-surrogate + GAE           | trust region + critic → stable for real training |

it's one chain, each rung shaving variance or adding stability. and the "five approaches" everyone lists at the start - imitation, policy-gradient, actor-critic, value-based, model-based - they all reduce to the same objective plus the gradient trick. they only differ in how $\Psi$ gets estimated and how hard you update. RLHF ships the actor-critic one.

---

I'll keep adding the other RL algos here as I go along...

---

resources: 
 - http://joschu.net/blog/kl-approx.html
 - https://github.com/verl-project/verl/pull/2953#issuecomment-3162113848
 - https://fengyao.notion.site/off-policy-rl
 - https://rlhfbook.com/c/06-policy-gradients
