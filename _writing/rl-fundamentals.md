---
title: RL fundamentals
date: 2026-09-30
description: mdp, returns, value functions, bellman equations, dp/mc/td, q-learning, data regimes and reward design
tags: [RL, notes]
---

i split the fundamentals half out of my [RL notes](/writing/RL-notes/) and folded my mdp writeups into it too. everything here is the modeling language - the MDP tuple, $G_t$, $V$, $Q$, the Bellman equations - plus how you actually learn those quantities when the environment is a black box (TD error, DP/MC/TD, Q-learning), where the training data comes from, and what makes a reward function good or catastrophic. the policy-gradient machinery (gradient trick, RLOO, PPO, GRPO) stays in the other note.

## the whole picture, before the math

1. **describe the task as an MDP.** states, actions, transitions, reward, discount. this is the contract.
2. **say what "good" means.** the return $G_t$ (the discounted sum of rewards), and the objective $J$ (its average over trajectories).
3. **say what you control.** the policy $\pi$ - the only thing in the whole setup you get to change.
4. **define the scores.** $V^\pi(s)$: on average, how good is it to be here? $Q^\pi(s,a)$: how good is it to do this here?
5. **make the scores computable.** the Bellman equations turn "sum over all possible futures" into a one-step recursion over the table. then DP, MC and TD are three different ways to actually fill the numbers in - they differ only in where the target comes from.
6. **turn numbers into a better policy.** act greedily on $Q$, or push $\pi$ in the direction of positive advantage.

the discount factor $\gamma$ decides how far ahead any of this looks, and the reward function decides what "good" even means.

## the MDP tuple

$$M = (\mathcal{S}, \mathcal{A}, P, R, \gamma)$$

five questions, and you can ask them of any RL problem in this order:

- $\mathcal{S}$ - where am I? the state space, every situation the agent can be in.
- $\mathcal{A}$ - what can I do? the action space.
- $P$ - what happens next? the transition dynamics, how the world evolves after an action.
- $R$ - how am I scored? the reward function.
- $\gamma$ - how much does the future count? the discount factor, how short-sighted or long-horizon the agent is.

so the loop is: you're in a situation ($\mathcal{S}$), you make a choice ($\mathcal{A}$), the world changes ($P$), you get feedback ($R$), and you land in the next situation and repeat.

what the tuple condenses is that it is the *contract* that lets us write $V$, $Q$ and $\pi$ without ever mentioning physics or tokens again. once a problem fits the tuple, the same theorems and the same algorithms apply - that is the entire reason the notation earns its keep. and when two problems "look nothing alike" but both fit, you already know what transfers and what doesn't.

the action space already shapes the algorithm choice:

| action space | property | typical algorithms |
| --- | --- | --- |
| discrete (e.g. {left, right}) | can enumerate actions and pick the best | Q-learning / DQN |
| continuous (e.g. joint torques) | cannot enumerate | policy gradient (PPO, etc.) |

## trajectories and the markov property

- state transitions $s_1 \rightarrow s_2 \rightarrow s_3$ with actions $a_1, a_2$.
- $P(s_3 \mid s_2, s_1) = P(s_3 \mid s_2)$  
- the **markov property**: the future depends on the world only through the present state.

rewards live on a probability distribution over the trajectory:

$$P(\tau) = P(s_1) \prod_{t=1}^T \pi(a_t \mid s_t)\, p(s_{t+1} \mid s_t, a_t)$$

$\pi$ is the policy, our decision rule, the only thing we control. $p$ is the environment's randomness (the dynamics), which we do **not**.

that product condenses the probability of one whole story happening. pick a starting state, then at every step the policy picks an action and the environment picks where that lands you, and one specific trajectory is the product of all of it. $\log P(\tau)$ is the thing policy gradient methods differentiate, and this product is what makes that derivative collapse into a sum over per-token log-probs.

**why the markov property is doing real work here** a chess position is a state: the moves that produced it don't change what happens next, so the position is a sufficient summary. Without that, "the value of this state" wouldn't even be well defined. You would need the value of the entire history, and every table lookup would need the whole past as its key. the markov property is precisely what licenses writing $V(s)$ instead of $V(s_1, \dots, s_t)$.

## the reward hypothesis

Sutton's reward hypothesis: any goal can be described as "maximizing expected cumulative reward" (Sutton & Barto, 2018). win a game, drive a car, chat politely - if you can encode it as a reward signal, RL can in principle learn it.

**the reward is the only channel from your intent to the algorithm.** the optimizer never sees the goal. it sees a list of numbers and pushes them up. that one fact explains both halves of RL's reputation: why the framework is general enough to cover chess and chat, and why it fails so spectacularly when the numbers don't mean what you thought. (the failure modes get their own section at the end.)

one distinction to fix early, because the two words get used interchangeably:

- the **reward** $R$ is immediate, one step, and part of the *task specification*. it's what you write down.
- the **value** $V^\pi$ is a long-run estimate the algorithm computes *from* those rewards and its own experience. it's what gets learned.

so a value function is never "true" on its own, it's only true relative to a reward function. 

## return $G_t$ and the discount factor

To evaluate how an episode went we aggregate rewards across steps, and distant rewards count less. That gives the discounted return:

$$G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \gamma^3 r_{t+3} + \cdots = \sum_{k=0}^{\infty} \gamma^k r_{t+k}$$

it has a recursive form:

$$G_t = r_t + \gamma G_{t+1}$$

why: expand $G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots$ and factor $\gamma$ out of the tail, $G_t = r_t + \gamma(r_{t+1} + \gamma r_{t+2} + \cdots) = r_t + \gamma G_{t+1}$. meaning the return from $t$ equals the immediate reward plus the discounted return from $t+1$. The whole future is summarized by $G_{t+1}$.

$\gamma \in [0,1]$ controls how much weight the future keeps. smaller values decay fast, values near 1 preserve the distant reward. take a three-step trajectory where all rewards are 1: at $\gamma = 0.9$, $G_0 = 1 + 0.9 \times 1 + 0.9^2 \times 1 = 2.71$; at $\gamma = 0.5$, the same trajectory gives $1 + 0.5 + 0.25 = 1.75$. and $\gamma < 1$ is what makes an infinite-horizon return converge at all.

a number worth keeping in your head: $\gamma$ has a half-life, because the weight on a reward $k$ steps away is $\gamma^k$ and that halves every $\ln 0.5 / \ln \gamma$ steps. $\gamma = 0.9$ means a reward 7 steps out counts half as much; $\gamma = 0.99$ means about 69 steps. that is the concrete meaning of "how far-sighted is this agent", and it's the first thing to check when an agent behaves myopically or ignores a goal that's too far away.

## the policy

the policy is how actions get chosen, and it comes in two flavors:

- **deterministic**, $a = \pi(s)$ always the same action for a given state. Q-learning and DQN end up here: learn $Q(s,a)$, then pick $\arg\max_a Q(s,a)$. simple, but no exploration built in: if the early estimates are wrong the agent gets stuck exploiting a suboptimal action, which is why in practice DQN pairs it with $\epsilon$-greedy.
- **stochastic**, $\pi(a \mid s) = P(a \mid s)$ a probability distribution over actions. exploration comes for free, there's always some probability of trying a non-greedy action with no separate mechanism, and it's differentiable, which is what policy gradient methods need.

what it condenses: $\pi$ is the *only* free variable in the whole setup. $P$, $R$ and $\gamma$ are given by the task; $\pi$ is the part you get to change. so "solving an RL problem" means exactly "find the $\pi$ that makes $J$ large", and every algorithm in this note is either a way of scoring $\pi$ or a way of moving it.

## the objective

everything builds on one expectation:

$$J(\theta) = \mathbb{E}_{\tau \sim p_\theta}[R(\tau)]$$

with $\tau = (s_0, a_0, s_1, a_1, \dots)$ a trajectory, $R(\tau) = \sum_{t=0}^\infty r_t$ the total reward, and $p_\theta$ meaning the policy $\pi_\theta$ is baked into the trajectory distribution. we want the max over $\theta$.

this average is never what you literally compute. you estimate it with a monte-carlo mean over a batch of $B$ completions:

$$\hat{J}(\theta) = \frac{1}{B} \sum_{i=1}^B R(x_i, y_i)$$

$y_i$ is a completion, $x_i$ is a prompt. in RLHF there's no per-step reward. The reward model scores the whole completion and hands back one terminal scalar. that's why the whole temporal chain collapses and why $\gamma$ ends up at 1.0.

**a tiny batch, so the estimate isn't abstract.** say $B = 8$ completions come back with rewards $[1, 0, 1, 0, 0, 1, 1, 0]$. then $\hat{J} = 4/8 = 0.5$. that's the entire training signal: four of the eight were good, so nudge the policy toward the good four and away from the other four. it's an unbiased estimate precisely because the trajectories were drawn from the policy you're scoring.

the optimal policy is the one with the largest expected long-term return:

$$\pi^* = \arg\max_\pi \mathbb{E}_\pi\left[\sum_{t=0}^\infty \gamma^t R(s_t, a_t)\right]$$

what $J$ condenses: "how good is my policy?" into a single scalar. that's what lets RL be handed to a gradient method.

## state value and action value

$$V^\pi(s) = \mathbb{E}_\pi[G_t \mid s_t = s] = \mathbb{E}_\pi\left[\sum_{k=0}^\infty \gamma^k r_{t+k} \mid s_t = s\right]$$

$V$ is for value, the $\pi$ superscript reminds us the value depends on the policy followed afterward, and the bar means "given". so $V^\pi(s)$ is the average of $G_t$ given that we're in $s$ now and follow $\pi$ from here on.

- $\sum_k \gamma^k r_{t+k}$ adds up the future rewards
- $\gamma^k$ discounts the far ones harder ($\gamma = 0$ cares only about the present, $\gamma$ near 1 more about the long term)
- $\mathbb{E}_\pi$ is there because environment and policy can both be random, so it's the average over many attempts.

action value fixes the first action too:

$$Q^\pi(s,a) = \mathbb{E}_\pi[G_t \mid s_t = s, a_t = a]$$

if repeated trials from the same state average 70 when the first action is push left and 110 when it's push right, then $Q^\pi(s, \text{push left}) = 70$ and $Q^\pi(s, \text{push right}) = 110$. Both assuming the agent follows $\pi$ after that first move. it sharpens "how good is this state?" into "how good is this particular first action in this state?"

and the reason $Q$ has to exist at all: $V$ averages over actions, so **it can tell you a state is good but not which move is responsible**. any decision needs the comparison, and comparisons need the action pulled out of the average.

| | $V^\pi(s)$ | $Q^\pi(s,a)$ |
| --- | --- | --- |
| question asked | standing in $s$, following $\pi$, how good on average? | standing in $s$, first take $a$, then follow $\pi$, how good on average? |
| how actions are handled | no action specified; chosen by $\pi$ | the first action $a$ is specified |
| effect of actions | mixed into the average of $\pi$ | pulled out and scored separately |
| good for answering | is this situation good overall? | which action is better in this situation? |

and $V$ is just the policy-weighted average of $Q$:

$$V^\pi(s) = \sum_a \pi(a \mid s) Q^\pi(s,a)$$

## advantage

how much better action $a$ is than the average action in that state:

$$A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$$

**intuition** --> 80% marks is good news only if the class average was 60%. the raw reward is the 80%, $V^\pi(s)$ is the class average, and the advantage is how far above the curve this action landed. that's why algorithms push on the advantage rather than the raw reward. The raw number says something went well, the advantage says whether *this action* deserves the credit.

## GAE (generalized advantage estimation)

single-step TD is noisy and biased. GAE exponentially weights the residuals to trade the two off:

$$A_t^{\text{GAE}(\gamma,\lambda)} = \sum_{k=0}^{T-t} (\gamma\lambda)^k\, \delta_{t+k}, \qquad \delta_{t+k} = r_{t+k} + \gamma V(s_{t+k+1}) - V(s_{t+k})$$

- **$\lambda \to 0$:** just the 1-step TD residual - low variance, high bias.
- **$\lambda \to 1$:** the full monte-carlo return - unbiased, high variance.

so $\lambda$ picks the point between them. for a completion, only the terminal reward is nonzero, so $\delta_t = \gamma V(s_{t+1}) - V(s_t)$ for $t < T$ and $\delta_T = R - V(s_T)$.

GAE kind of condenses the whole DP-to-MC spectrum into one dial. $\lambda$ isn't a new idea, it's the same "how much reality do you want to wait for" question from the DP/MC/TD section below, expressed as a decay on the residuals instead of a choice of algorithm.

## the Bellman equations

$V^\pi(s)$ is the expected discounted sum of all future rewards. the intuitive way to get it is to follow the policy forward and add rewards along one long trajectory, and that hits two walls:

- the future is too long: some tasks have no clear endpoint (a robot balancing), so how far do you add?
- there are too many possibilities: environment and policy can both be random, so states branch exponentially and an accurate expectation would need countless trajectories.

Bellman's principle of optimality (1950s, out of dynamic programming theory) is the way out: we don't need to see the entire future at once, because today's value must contain tomorrow's value. it turns policy evaluation from an infinite summation problem into a recursion that only depends on neighboring states.

$V^\pi(s) = r^\pi(s) + \gamma\sum_{s'} P_\pi(s'\mid s)V^\pi(s')$ says "the value of standing here is the reward you get now plus the discounted value of wherever you land". The infinite tail is hidden inside $V^\pi(s')$, which is the *same kind of object* as the thing being computed.

expand the definition:

$$\begin{aligned}
V^\pi(s) &= \mathbb{E}_\pi[G_t \mid s_t = s] \\
&= \mathbb{E}_\pi[r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots \mid s_t = s] \\
&= \mathbb{E}_\pi[r_t \mid s_t = s] + \gamma\, \mathbb{E}_\pi[r_{t+1} + \gamma r_{t+2} + \cdots \mid s_t = s] \\
&= r^\pi(s) + \gamma\, \mathbb{E}_\pi[G_{t+1} \mid s_t = s]
\end{aligned}$$

where $r^\pi(s)$ is the expected immediate reward in state $s$ after fixing $\pi$, already averaged over action uncertainty:

$$r^\pi(s) = \sum_a \pi(a \mid s) R(s,a)$$

### the subtle step

the tempting move is to write $\mathbb{E}\_\pi[G\_{t+1} \mid s\_t = s] = V^\pi(s\_{t+1})$, and that's wrong. an analogy for the difference:

- $\mathbb{E}\_\pi[G\_{t+1} \mid s\_t = s]$ is you today, standing at $s_t$, predicting tomorrow's earnings/rewards $G_{t+1}$. The number has already averaged out where tomorrow lands.
- $V^\pi(s_{t+1})$ is you tomorrow, already standing in a concrete $s_{t+1}$, computing the future from there. but from today's perspective $s_{t+1}$ hasn't happened, so $V^\pi(s_{t+1})$ is still a random variable.

the left side is a number already averaged under $s_t = s$; the right side still depends on the random next state. the correct statement keeps it inside the same conditional expectation:

$$\mathbb{E}_\pi[G_{t+1} \mid s_t = s] = \mathbb{E}_\pi[V^\pi(s_{t+1}) \mid s_t = s]$$

proof:

**step 1 - expand the right-hand side.** by definition $V^\pi(s\_{t+1}) = \mathbb{E}\_\pi[G\_{t+1} \mid s\_{t+1}] = \sum\_{G\_{t+1}} G\_{t+1} P\_\pi(G\_{t+1} \mid s\_{t+1})$, so

$$\mathbb{E}_\pi[V^\pi(s_{t+1}) \mid s_t] = \mathbb{E}_\pi\Big[\sum_{G_{t+1}} G_{t+1}\, P_\pi(G_{t+1} \mid s_{t+1}) \Big| s_t\Big]$$

**step 2 - expand the outer expectation over the next state.** what tomorrow is, is random, so multiply every possible tomorrow by its probability and add:

$$\mathbb{E}_\pi[V^\pi(s_{t+1}) \mid s_t] = \sum_{s_{t+1}} \Big(\sum_{G_{t+1}} G_{t+1} P_\pi(G_{t+1} \mid s_{t+1})\Big) P_\pi(s_{t+1} \mid s_t) = \sum_{s_{t+1}} \sum_{G_{t+1}} G_{t+1}\, P_\pi(G_{t+1} \mid s_{t+1})\, P_\pi(s_{t+1} \mid s_t)$$

**step 3 - inject the markov property.** once $\pi$ is fixed, $G_{t+1}$ depends only on $s_{t+1}$ at that moment, not on the earlier $s_t$. like a dice roll: the second roll depends on how the second roll is made, not the first. so we can force $s_t$ into the condition without changing the value, $P_\pi(G_{t+1} \mid s_{t+1}) = P_\pi(G_{t+1} \mid s_{t+1}, s_t)$, and substitute:

$$\mathbb{E}_\pi[V^\pi(s_{t+1}) \mid s_t] = \sum_{s_{t+1}} \sum_{G_{t+1}} G_{t+1}\, P_\pi(G_{t+1} \mid s_{t+1}, s_t)\, P_\pi(s_{t+1} \mid s_t)$$

**step 4 - multiplication rule and marginalization.** the basic conditional probability identity $P(A \mid B) = P(A,B)/P(B)$ rearranges to the multiplication rule $P(A,B) = P(A \mid B)P(B)$, and it still holds with an extra premise: $P(A,B \mid C) = P(A \mid B,C)P(B \mid C)$. take $G_{t+1}$ as $A$, $s_{t+1}$ as $B$, $s_t$ as $C$, and the two probabilities merge:

$$\begin{aligned}
\mathbb{E}_\pi[V^\pi(s_{t+1}) \mid s_t] &= \sum_{s_{t+1}} \sum_{G_{t+1}} G_{t+1}\, P_\pi(G_{t+1}, s_{t+1} \mid s_t) \\
&= \sum_{G_{t+1}} G_{t+1} \sum_{s_{t+1}} P_\pi(G_{t+1}, s_{t+1} \mid s_t) \qquad \text{(swap the order of summation)} \\
&= \sum_{G_{t+1}} G_{t+1}\, P_\pi(G_{t+1} \mid s_t) \qquad \text{(sum over all } s_{t+1} \text{ marginalizes it out)} \\
&= \mathbb{E}_\pi[G_{t+1} \mid s_t] \qquad \text{(definition of conditional expectation)}
\end{aligned}$$

proof complete.

### the standard form

substituting back:

$$\begin{aligned}
V^\pi(s) &= r^\pi(s) + \gamma\, \mathbb{E}_\pi[V^\pi(s_{t+1}) \mid s_t = s] \\
&= r^\pi(s) + \gamma \sum_{s'} P_\pi(s' \mid s) V^\pi(s')
\end{aligned}$$

the second line is just the definition of a discrete conditional expectation, $\mathbb{E}[X \mid Y] = \sum_x x \cdot P(x \mid Y)$, with $X = V^\pi(s_{t+1})$. starting from $s$ and acting under $\pi$, there's probability $P_\pi(s' \mid s)$ of jumping to $s'$, and $s'$ is worth $V^\pi(s')$; weighting all next states by their probability is "how much the next step is worth on average".

the same move for $Q$: first the immediate reward of $a$, then the value of wherever that action lands you:

$$\begin{aligned}
Q^\pi(s_t, a_t) &= \mathbb{E}_\pi[G_t \mid s_t, a_t] = \mathbb{E}_\pi[r_t + \gamma G_{t+1} \mid s_t, a_t] \\
&= R(s_t, a_t) + \gamma\, \mathbb{E}_\pi[V^\pi(s_{t+1}) \mid s_t, a_t] \\
&= R(s_t, a_t) + \gamma \sum_{s' \in \mathcal{S}} P(s' \mid s_t, a_t) V^\pi(s')
\end{aligned}$$

### the optimality equation

the expectation version averages over the policy; the optimality version stops averaging and takes the best action. for $V^*$ the policy is greedy:

$$V^*(s) = \max_a \left[ R(s,a) + \gamma \sum_{s' \in \mathcal{S}} P(s' \mid s, a) V^*(s') \right]$$

the only difference from the equation above is $\max_a$ replacing $\sum_a \pi(a\mid s)$.

### matrix form and the analytic solution

with a fixed policy, the action uncertainty folds into two quantities:

$$r^\pi(s) = \sum_a \pi(a \mid s) R(s,a), \qquad P^\pi(s' \mid s) = \sum_a \pi(a \mid s) P(s' \mid s, a)$$

$P^\pi$ is not the raw environment transition $P(s' \mid s, a)$ - it's the state-to-state probability after "first pick an action according to the policy, then let the environment transition". with $N$ states, the $N$ simultaneous equations stack up as

$$v^\pi = r^\pi + \gamma P^\pi v^\pi$$

where $v^\pi$ is the $N \times 1$ vector of state values under $\pi$, $r^\pi$ the $N \times 1$ vector of expected immediate rewards, and $P^\pi$ the $N \times N$ policy-induced transition matrix. rearrange:

$$v^\pi - \gamma P^\pi v^\pi = r^\pi \;\Longrightarrow\; (I - \gamma P^\pi) v^\pi = r^\pi \;\Longrightarrow\; v^\pi = (I - \gamma P^\pi)^{-1} r^\pi$$

with $I$ the identity matrix. so as long as the environment rules and $\pi$ are fully transparent, this is the exact value of every state under that policy.

then why learn any algorithm at all? the analytic solution needs inverting $(I - \gamma P^\pi)$, and matrix inversion is $O(N^3)$. let the state space get even slightly larger and it's over.

## learning without the model: TD error

the Bellman equations quietly assume we can look up $P$ and $R$. real tasks don't hand you a transition table. the agent takes a step, sees a reward, sees where it landed, and that's the whole data stream. so the exact target,

$$\sum_a \pi(a \mid s) \Big[ R(s,a) + \gamma \sum_{s'} P(s' \mid s, a) V(s') \Big]$$

is not computable: the $R(s,a)$ in it is an *average* reward, $P(s' \mid s,a)$ is a distribution over next states, and we have neither. what we do have is one transition that actually happened. TD error is the reconciliation. Compare the target implied by that one experience against the old estimate sitting in the table:

$$\text{target} = r + \gamma V(s'), \qquad \delta = \underbrace{r + \gamma V(s')}_{\text{one-step target}} - \underbrace{V(s)}_{\text{current estimate}}$$

why is one sample allowed to stand in for the average? because the Bellman target *is* an expectation, so it's a probability-weighted average over outcomes. each real step draws one outcome from that distribution: common outcomes get drawn, and therefore learned from, more often; rare ones less. let the updates accumulate and the table drifts in the probability-weighted direction.

why "temporal difference": it is the gap between two predictions made at neighboring times. standing at $s$ we held the old prediction $V(s)$; after moving to $s'$ the term $r + \gamma V(s')$ hands us a new one.

**intuition** --> you're driving somewhere and your ETA says 30 minutes. ten minutes in, you've covered less ground than you hoped and the remaining estimate says 28 more, so you revise the total to 38. you did not wait until you arrived to find out you were wrong (that would be monte carlo). You noticed the *difference between two predictions made at different times* and corrected mid-trip. TD is exactly that, and it's why the signal can be computed after one step instead of one episode.

| $\delta$ | meaning | how to adjust |
| --- | --- | --- |
| $\delta > 0$ | target above the estimate. we underestimated | raise $V(s)$ |
| $\delta < 0$ | target below the estimate. we overestimated | lower $V(s)$ |
| $\delta = 0$ | on this sample the two agree | barely move |

$$V(s) \leftarrow V(s) + \alpha\, \delta$$

$\alpha$ is the learning rate: how much of one sample's opinion to believe. at $\alpha = 1$ a single noisy step would overwrite the whole estimate, which is why we creep toward it instead.

## DP, MC and TD: three answers to "where does the target come from"

the Bellman equations say what the values must satisfy. they don't say how to find them. that gap is where the classic trio lives, and the payoff is realizing they are the *same algorithm* with three different target constructors:

$$V(s) \leftarrow V(s) + \alpha\big[\text{target} - V(s)\big]$$

| | needs the model? | when can it update? | target |
| --- | --- | --- | --- |
| **DP** | yes | any time | $\sum_a \pi(a\mid s)\big[R(s,a) + \gamma\sum_{s'} P(s'\mid s,a)V_k(s')\big]$ |
| **MC** | no | only when an episode ends | $G_t$, the realized full return |
| **TD** | no | after every single step | $r + \gamma V(s')$ |

- **DP** knows $P$ and $R$, so it can expand *every* branch and average exactly. No sampling at all. it's a sweep over the table: plug in the old values, get better ones, repeat until nothing moves. (this is the "iterate instead of invert" escape from the $O(N^3)$ problem above.)
- **MC** has no model, so it can't average over branches. it waits, observes one complete episode, and uses what actually happened. unbiased, because $G_t$ is a genuine sample of the return. but it needs the episode to *end*: from an intermediate state $G_t$ isn't knowable yet, and the variance is the variance of everything from here to termination, which gets brutal on long trajectories.
- **TD** has no model either, but refuses to wait. one real reward plus the table's current guess about the next state. Biased, because $V(s')$ is wrong for a while, but low variance and it starts learning immediately.

the substitution in TD is called **bootstrapping**: using an estimate to update an estimate. it's the same self-reference that made the Bellman equations solvable, now applied numerically, one step at a time.

I'm not going to memorize these algos. They are three positions on a spectrum of *how much reality are you willing to wait for*. DP has the model tell it the answer; MC waits for the whole truth; TD takes the truth one step at a time and guesses the rest. the same dial reappears in [GAE](#gae-generalized-advantage-estimation) as $\lambda$, sliding between "one-step guess" and "wait for the full return".

## control: the Q table and Q-learning

$V$ answers "is this situation good or bad". Control needs a sharper question (**in state $s$, with candidates $a_1$ and $a_2$, which one is better?**) and that is a comparison of $Q^\pi(s,a_1)$ against $Q^\pi(s,a_2)$, not a score for the state.

With a known model you get $Q$ from $V$ for free, because "take $a$ first, then follow $\pi$" is just the immediate reward plus the discounted value of wherever you land:

$$Q^\pi(s,a) = R(s,a) + \gamma \sum_{s'} P(s' \mid s, a) V^\pi(s')$$

same shape as the $V$ recursion, except the first action is pinned instead of averaged, but with $R$ and $P$ unavailable that formula is dead on arrival, so we do what we did for $V$: learn the values directly. grow the table from one cell per state to one cell per state-action pair (the $Q$ table) and correct one entry at a time from one transition.

three lines carry the whole idea:

$$\begin{aligned}
Q^\pi(s,a) &= \mathbb{E}_\pi[G_t \mid S_t = s, A_t = a] && \text{what } Q \text{ means} \\
Q^*(s,a) &= \mathbb{E}\big[\, r + \gamma \max_{a'} Q^*(s', a') \mid s, a \,\big] && \text{the target, for the optimal table} \\
Q(s,a) &\leftarrow Q(s,a) + \alpha \big[\, r + \gamma \max_{a'} Q(s', a') - Q(s,a) \,\big] && \text{the update}
\end{aligned}$$

- line 1 says what $Q$ means: take $a$ first, then follow $\pi$; $Q$ is the expected return of that.
- line 2 is the Bellman optimality equation for $Q$. The self-consistency a correct optimal table has to satisfy. An entry $Q^*(s,a)$ should equal "take one step from this entry, then consult the best entry in the next state". we may not know whether our table is right, but we do know what it has to satisfy, and the size of the violation is the update direction.
- line 3 is the algorithm: build a TD target from one sampled transition and step toward it. the $\max$ is the entire difference from SARSA. The target assumes optimal play from $s'$ regardless of what the agent actually went on to do.

that $\max$ is worth dwelling on, because it's the difference between two ways of thinking. SARSA asks "if I keep behaving like I'm behaving, what happens?" and learns the value of the policy you actually run. Q-learning asks "if I play perfectly from here on, what happens?" and learns the value of a policy you never quite execute. the second question is the one you want answered, and it's why Q-learning keeps converging to greedy play even while the agent wanders around exploring.

once the table is good, the policy falls straight out of it:

$$\pi(s) = \arg\max_a Q(s,a)$$

which is why Q-learning is **off-policy**: it learns about the greedy target policy while the behaviour policy is still $\epsilon$-greedy exploring. the thing that generated the data is not the thing being scored.

### a 4x4 gridworld, one update at a time

4x4 grid, every step costs $r = -1$, $\gamma = 0.9$, the $Q$ table starts at all zeros, and the agent starts top-left. it picks "right" and lands on $(0,1)$ with $r = -1$.

only one number moves: $Q((0,0), \text{right})$. the other three actions at $(0,0)$ and every entry in every other state are untouched. the target looks at what we just got, then at what the next cell is worth, and the next cell is still all zeros, so

$$\max_{a'} Q((0,1), a') = 0 \quad\Longrightarrow\quad \text{target} = -1 + 0.9 \times 0 = -1$$

read it plainly: this step already costs a point, and the cell we're sitting on looks neither good nor bad yet, so "go right" is worth $-1$ for now. but Q-learning doesn't overwrite the old $0$ with $-1$, it takes a step toward it:

$$Q((0,0), \text{right}) = 0 + 0.1 \times (-1 - 0) = -0.1$$

that's a mild lesson after one trial (this discounted small step crawl is ex actly what ends up stabilizing PPO as well, conceptually. just a simple heuristic to keep in mind. if making changes, make them against a base so the changes are small and tracable, let the GPUs handle the rest). keep sampling and the values along negative-reward paths sink while the values along the good path rise. Information propagates backward from the goal to the start until the table fills in. The core loop is genuinely this small:

```python
Q = np.zeros((16, 4))   # state = row*4 + col, actions = up/right/down/left
for episode in range(2000):
    state, eps = 0, max(0.02, 0.3 * 0.995 ** episode)
    for _ in range(100):
        action = rng.integers(4) if rng.random() < eps else Q[state].argmax()
        next_state, reward, done = step(state, action)   # clamp to grid, -1 a step, done at (3,3)
        bootstrap = 0 if done else Q[next_state].max()   # terminal state has no future
        Q[state, action] += 0.1 * (reward + 0.9 * bootstrap - Q[state, action])
        state = next_state
        if done:
            break
```

nothing in this loop ever computes a policy, and nothing ever evaluates a trajectory end to end. the policy is implicit in the arithmetic, and the values propagate backward from the goal one cell per few episodes. that's why tabular RL is usually the first thing to implement: the whole idea fits on one screen.

## where the data comes from: on/off-policy, online/offline

two independent questions, and conflating them is the usual source of confusion:

- **who produced this data?** the **behaviour policy** $\mu$ is what actually acted - an $\epsilon$-greedy explorer, yesterday's checkpoint, an expert, or a human. the **target policy** $\pi_\theta$ is what you're trying to learn.
- **is the data still being collected?** online RL keeps interacting and appending; offline RL gets a fixed dataset and a ban on interaction.

| data regime | meaning | typical examples | main risk |
| --- | --- | --- | --- |
| online + on-policy | keep interacting, learn only from data the current policy just produced | REINFORCE, SARSA, PPO, GRPO | sample inefficiency |
| online + off-policy | keep interacting, but store and reuse older data | Q-learning, DQN, SAC, TD3 | stored data drifts away from what the target policy would do |
| offline + off-policy | one fixed historical dataset, no interaction | CQL, IQL, DPO-style preference training | extrapolating to actions never seen in the data |
| offline + on-policy | fixed data that's required to match the current policy | fixed-policy evaluation | the data goes stale the moment the policy moves |

$\mu = \pi_\theta$ means on-policy; $\mu \neq \pi_\theta$ means off-policy. and the axes really are independent: **DQN is off-policy but still online**, it reuses replay data *and* keeps playing the game. Offline RL is usually off-policy as a side effect, because the dataset was produced before the current policy existed.

**why anyone tolerates on-policy's cost?** online interaction is a self-correcting loop. If the value estimate for an action is wrong, try it and watch. Offline takes that away, which is the entire difficulty and the setting exists because sometimes trial and error isn't available. You cannot let a half-trained policy discover what happens when a real car crashes, so you learn from logged human driving instead.

## when "on-policy" is only approximately true

LLM RL has a wrinkle the textbook version doesn't. PPO's ratio is

$$r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\text{old}}(a_t \mid s_t)}$$

and clipping assumes the denominator is really the policy that *sampled* the action. In practice the log-probs recorded during rollout come from a different inference stack (different kernels, different precision, different batching) than the log-probs recomputed during training. So $\pi_{\text{rollout}} \neq \pi_{\text{old}}$ even when the weights are identical, and the ratio is biased before any optimization happens. clipping limits how far you drift from the old policy; it cannot detect that the map you're drifting from was already off.

what cab be done for this is to match precision between rollout and training (BF16 vs FP8 is the usual culprit); add importance-sampling corrections with truncated ratios (TIS) or per-token minimums within a prefix (MinPRO); drop the extreme long-tail tokens where the mismatch concentrates; replay the MoE routing distribution during training (R3); or **recompute the rollout log-probs with the training engine before the update** expensive, but the most reliable.

on-policy essentialyl means close enough to on-policy. 

## reward design

Everything the algorithm knows about your goal arrives as a scalar, so the shape of that scalar decides the behaviour you get.

### sparse, dense, delayed

- **sparse** - score only the outcome. a maze pays $+1$ at the exit, $0$ everywhere else:

$$R(s,a,s') = \begin{cases} +1 & s' \text{ is the goal} \\ 0 & \text{otherwise} \end{cases}$$

clean, honest, and nearly unlearnable in a large maze: the agent wanders for thousands of steps collecting zeros. It knows it failed, but not which step was the mistake. That's the credit-assignment problem in its purest form.

in rlhf this like ORM.

- **dense** - pay at every step. reward the reduction in distance to the goal:

$$R_{\text{dense}}(s,a,s') = d(s, \text{goal}) - d(s', \text{goal})$$

moving closer pays, moving away costs, and now there's signal on every step. the catch: you just told the agent *your* idea of progress, and your idea is now part of the objective.

in rlhf this is like PRM.

- **delayed** - rewards exist at every step, but the informative one is far from its cause. CartPole pays $+1$ for every step the pole stays up, but the push that doomed the episode happened twenty steps before the pole fell. preference scores over a whole answer are the same shape: the sentence that lost the points is the first one. every step technically has a reward, and credit still isn't local.

sparse is honest but hard to learn from; dense is easy to learn from but quietly rewrites the objective. that tension is the whole subject.

### shaping, and the one form that's provably safe

If dense rewards help learning but change the objective, can you get one without the other? Reward shaping adds an extra term on top of the original reward:

$$R'(s,a,s') = R(s,a,s') + F(s,s')$$

the useful result is that **if $F$ takes the potential form $F(s,s') = \gamma\Phi(s') - \Phi(s)$, the optimal policy is unchanged**. 

intuition --> you aren't adding preferences, you're re-timing the *same* preferences. $\Phi$ is a score per state, and $F$ pays you for moving to a better-scored state and charges you for moving to a worse one, so the sum telescopes. Every $\Phi(s)$ appears once as a payment and once as a charge, and everything cancels except the endpoints. You get a dense signal on every step without changing which policy is optimal.

### where the reward comes from

hand-writing rewards is hard, so the usual move is to learn one from preferences:

- **ORM** (outcome reward model): score the final answer. what standard RLHF does.
- **PRM** (process reward model): score every step of the reasoning. "correct answer, broken derivation" is a real failure mode, and step-level scores are what let you say *where* it broke - a sparse terminal reward turned into a dense per-step one.
- **RLAIF**: replace the human labeller with a model, to make labelling affordable.
- **GRPO**: drop the value network and use the spread of rewards within a sampled group as the baseline. (grpo is one of those ideas so simple in its nature, I wonder if I was doing this earlier would it have occured to me? I think that about gravitation laws too so doesn't mean much)

each attacks the same gap from a different end, and each buys its fix with a new risk: PRMs cost labelling labour, RLAIF bets that AI preferences track human ones, ORMs get gamed.

### the proxy is not the goal

the formal version is Goodhart's law: the proxy reward $R$ is almost never the true intent $R^*$, and optimization is precisely the process that drives them apart. Optimize hard enough and you get the gap in high definition.

a checklist for a reward you're about to train against:

- when the agent scores high, would a human actually say the task was done well?
- can it farm reward by repeating one local behaviour?
- are the intermediate rewards helping learning, or have they quietly become the objective?
- is the signal dense enough to discover successful behaviour at all?
- if a model learned the reward from preferences, how does it fail under optimization?

## in one breath

1. the tuple $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$ is the rules of the game; the five questions are how you ask them of any task, and two problems that fit the tuple get the same algorithms.
2. $G_t$ is the objective, and $G_t = r_t + \gamma G_{t+1}$ condenses an infinite sum into two terms. $\gamma$ has a half-life ($\approx 7$ steps at $0.9$, $\approx 69$ at $0.99$).
3. $\pi$ is the only free variable in the setup, and training means finding the $\pi^*$ that maximizes $J$ which is just "how good is my policy", reduced to one scalar.
4. $V^\pi(s)$ and $Q^\pi(s,a)$ score a policy, with $V = \sum_a \pi(a\mid s)Q$; the advantage $A = Q - V$ is the "better than average" signal that actually drives updates.
5. Bellman equations hide the infinite future inside $V(s')$, turning an infinite expectation into a simultaneous system; the fixed-policy matrix form solves it exactly at $O(N^3)$ cost, which is why we iterate instead.
6. with no model to look up, the learning signal is the TD error $\delta = r + \gamma V(s') - V(s)$, where $V(s')$ being an estimate is the bootstrap.
7. DP, MC and TD are one update rule with three targets: the model's average, the realized episode return, or one step plus a guess. They trade bias against how long you wait.
8. Control grows the table to $Q(s,a)$; Q-learning's $\max$ makes it learn the greedy policy from exploratory data, one cell at a time. That's what off-policy buys.
9. on/off-policy is about who generated the data; online/offline is about whether it's still growing. They're independent (DQN is off-policy *and* online), and in LLM RL "on-policy" is often an approximation you have to engineer.
10. Reward design decides whether any of it works: sparse is honest but unsignalled, dense is learnable but biased, and the proxy-vs-intent gap (Goodhart) is where RL most often fails.

---

resources:
 - [walkinglabs, *hands-on modern RL* - the mdp writeup came from here](https://walkinglabs.github.io/hands-on-modern-rl/en)
 - Sutton & Barto (2018), *Reinforcement Learning: An Introduction* - the reward hypothesis, the Bellman material, DP/MC/TD and the weather-prediction framing of TD
 - [Watkins & Dayan (1992), *Q-Learning* - the off-policy TD control algorithm](https://link.springer.com/article/10.1007/BF00992698)
 - [Gymnasium, *FrozenLake-v1*n](https://gymnasium.farama.org/environments/toy_text/frozen_lake/)
 - Ng, Harada & Russell (1999), *Policy Invariance Under Reward Transformations* - potential-based reward shaping, i.e. why $F = \gamma\Phi(s') - \Phi(s)$ leaves the optimal policy alone
 - [*Balance Between Efficient and Effective Learning: Dense2Sparse Reward Shaping for Robot Manipulation with Environment Uncertainty* (2020](https://arxiv.org/abs/2003.02740)
 - [OpenAI, *Faulty Reward Functions in the Wild*](https://openai.com/index/faulty-reward-functions/)
 - [Gao, Schulman & Hilton (2022), *Scaling Laws for Reward Model Overoptimization* - the proxy-vs-intent gap, measured](https://arxiv.org/abs/2210.10760)
 - [Schulman et al. (2016), *High-Dimensional Continuous Control Using Generalized Advantage Estimation* (GAE)](https://arxiv.org/abs/1506.02438)
 - the policy-gradient half of these notes: [RL notes](/writing/RL-notes/)
