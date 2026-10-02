---
title: RL fundamentals
date: 2026-09-30
description: mdp, returns, value functions, bellman equations, td error and q-learning
tags: [RL, notes]
---

i split the fundamentals half out of my [RL notes](/writing/RL-notes/) and folded my mdp writeups into it too. everything here is the modeling language - the MDP tuple, $G_t$, $V$, $Q$, the Bellman equations - plus how you actually learn those quantities when the environment is a black box (TD error, Q-learning). the policy-gradient machinery (gradient trick, RLOO, PPO, GRPO) stays in the other note.

the point of this half: turn a real task into a mathematical object RL can operate on, then say what "solving" even means.

## the MDP tuple

$$M = (\mathcal{S}, \mathcal{A}, P, R, \gamma)$$

five questions, and you can ask them of any RL problem in this order:

- $\mathcal{S}$ - where am I? the state space, every situation the agent can be in.
- $\mathcal{A}$ - what can I do? the action space.
- $P$ - what happens next? the transition dynamics, how the world evolves after an action.
- $R$ - how am I scored? the reward function.
- $\gamma$ - how much does the future count? the discount factor, how short-sighted or long-horizon the agent is.

so the loop is: you're in a situation ($\mathcal{S}$), you make a choice ($\mathcal{A}$), the world changes ($P$), you get feedback ($R$), and you land in the next situation and repeat. bandits, CartPole, and LLM alignment look nothing alike, but it's the same language for all of them.

the action space already shapes the algorithm choice:

| action space | property | typical algorithms |
| --- | --- | --- |
| discrete (e.g. {left, right}) | can enumerate actions and pick the best | Q-learning / DQN |
| continuous (e.g. joint torques) | cannot enumerate | policy gradient (PPO, etc.) |

## trajectories and the markov property

- state transitions $s_1 \rightarrow s_2 \rightarrow s_3$ with actions $a_1, a_2$.
- $P(s_3 \mid s_2, s_1) = P(s_3 \mid s_2)$ - the **markov property**: the future depends on the world only through the present state.

rewards live on a probability distribution over the trajectory:

$$P(\tau) = P(s_1) \prod_{t=1}^T \pi(a_t \mid s_t)\, p(s_{t+1} \mid s_t, a_t)$$

$\pi$ is the policy, our decision rule, the only thing we control. $p$ is the environment's randomness (the dynamics), which we do **not**.

for an LLM this is almost embarrassingly simple. state = the token sequence so far, action = next token, transition = deterministic append, $s_{t+1} = [s_t, a_t]$. so all the randomness is in the policy and none is in the environment. one prompt → one completion → one episode, dead at the EOS token. (batched rollouts and multi-turn convos add real complexity, but let's stay on the basics.)

## the reward hypothesis

Sutton's reward hypothesis: any goal can be described as "maximizing expected cumulative reward" (Sutton & Barto, 2018). win a game, drive a car, chat politely - if you can encode it as a reward signal, RL can in principle learn it. designing a good reward function is one of the hardest parts of RL engineering.

## return $G_t$ and the discount factor

to evaluate how an episode went we aggregate rewards across steps, and distant rewards count less. that gives the discounted return:

$$G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \gamma^3 r_{t+3} + \cdots = \sum_{k=0}^{\infty} \gamma^k r_{t+k}$$

it has a recursive form:

$$G_t = r_t + \gamma G_{t+1}$$

why: expand $G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots$ and factor $\gamma$ out of the tail, $G_t = r_t + \gamma(r_{t+1} + \gamma r_{t+2} + \cdots) = r_t + \gamma G_{t+1}$. meaning: the return from $t$ equals the immediate reward plus the discounted return from $t+1$ - the whole future is summarized by $G_{t+1}$. this recursion is the foundation of the Bellman equations.

$\gamma \in [0,1]$ controls how much weight the future keeps. smaller values decay fast, values near 1 preserve the distant reward. take a three-step trajectory where all rewards are 1: at $\gamma = 0.9$, $G_0 = 1 + 0.9 \times 1 + 0.9^2 \times 1 = 2.71$; at $\gamma = 0.5$, the same trajectory gives $1 + 0.5 + 0.25 = 1.75$. and $\gamma < 1$ is what makes an infinite-horizon return converge at all.

## the policy

the policy is how actions get chosen, and it comes in two flavors:

- **deterministic**, $a = \pi(s)$ - always the same action for a given state. Q-learning and DQN end up here: learn $Q(s,a)$, then pick $\arg\max_a Q(s,a)$ (the Q table section below does this end to end). simple, but no exploration built in: if the early estimates are wrong the agent gets stuck exploiting a suboptimal action, which is why in practice DQN pairs it with $\epsilon$-greedy.
- **stochastic**, $\pi(a \mid s) = P(a \mid s)$ - a probability distribution over actions. exploration comes for free, there's always some probability of trying a non-greedy action with no separate mechanism, and it's differentiable, which is what policy gradient methods need.

## the objective

everything builds on one expectation:

$$J(\theta) = \mathbb{E}_{\tau \sim p_\theta}[R(\tau)]$$

with $\tau = (s_0, a_0, s_1, a_1, \dots)$ a trajectory, $R(\tau) = \sum_{t=0}^\infty r_t$ the total reward (all rewards weighted equally here; discounting is the section above), and $p_\theta$ meaning the policy $\pi_\theta$ is baked into the trajectory distribution. we want the max over $\theta$.

this average is never what you literally compute. you estimate it with a monte-carlo mean over a batch of $B$ completions:

$$\hat{J}(\theta) = \frac{1}{B} \sum_{i=1}^B R(x_i, y_i)$$

$y_i$ is a completion, $x_i$ is a prompt. in RLHF there's no per-step reward - the reward model scores the whole completion and hands back one terminal scalar. that's why the whole temporal chain collapses and why $\gamma$ ends up at 1.0.

the optimal policy is the one with the largest expected long-term return:

$$\pi^* = \arg\max_\pi \mathbb{E}_\pi\left[\sum_{t=0}^\infty \gamma^t R(s_t, a_t)\right]$$

## state value and action value

$$V^\pi(s) = \mathbb{E}_\pi[G_t \mid s_t = s] = \mathbb{E}_\pi\left[\sum_{k=0}^\infty \gamma^k r_{t+k} \mid s_t = s\right]$$

$V$ is for value, the $\pi$ superscript reminds us the value depends on the policy followed afterward, and the bar means "given". so $V^\pi(s)$ is the average of $G_t$ given that we're in $s$ now and follow $\pi$ from here on.

three layers to read into that: $\sum_k \gamma^k r_{t+k}$ adds up the future rewards; $\gamma^k$ discounts the far ones harder ($\gamma = 0$ cares only about the present, $\gamma$ near 1 more about the long term); and the $\mathbb{E}_\pi$ is there because environment and policy can both be random, so it's the average over many attempts.

action value fixes the first action too:

$$Q^\pi(s,a) = \mathbb{E}_\pi[G_t \mid s_t = s, a_t = a]$$

if repeated trials from the same state average 70 when the first action is push left and 110 when it's push right, then $Q^\pi(s, \text{push left}) = 70$ and $Q^\pi(s, \text{push right}) = 110$ - both assuming the agent follows $\pi$ after that first move. it sharpens "how good is this state?" into "how good is this particular first action in this state?"

| | $V^\pi(s)$ | $Q^\pi(s,a)$ |
| --- | --- | --- |
| question asked | standing in $s$, following $\pi$, how good on average? | standing in $s$, first take $a$, then follow $\pi$, how good on average? |
| how actions are handled | no action specified; chosen by $\pi$ | the first action $a$ is specified |
| effect of actions | mixed into the average of $\pi$ | pulled out and scored separately |
| good for answering | is this situation good overall? | which action is better in this situation? |

and $V$ is just the policy-weighted average of $Q$:

$$V^\pi(s) = \sum_a \pi(a \mid s) Q^\pi(s,a)$$

$V^\pi(s)$ isn't a separate object redefined from scratch - it averages all the action values in that state by their probabilities.

### $G_t$ vs $V^\pi(s)$

$r_t$ is the immediate reward of this step, known right after you take it. $G_t$ is the discounted total actually obtained along one concrete trajectory, so different trajectories give different $G_t$. $V^\pi(s)$ is not the score of one trajectory - it's the average of $G_t$ over many trajectories that start from $s$ and then follow $\pi$. $G_t$ is the score report from this particular run; $V^\pi(s)$ is the prediction before the run starts, standing in the same state. same chess position, a master vs a beginner continuing it: different win rates, and that difference is $V$. a good policy has high $V^\pi(s)$, a poor one has low $V^\pi(s)$, and the goal of RL is the policy that makes value highest.

## advantage

how much better action $a$ is than the average action in that state:

$$A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$$

## the Bellman equations

$V^\pi(s)$ is the expected discounted sum of all future rewards. the intuitive way to get it is to follow the policy forward and add rewards along one long trajectory, and that hits two walls:

- the future is too long: some tasks have no clear endpoint (a robot balancing), so how far do you add?
- there are too many possibilities: environment and policy can both be random, so states branch exponentially and an accurate expectation would need countless trajectories.

Bellman's principle of optimality (1950s, out of dynamic programming theory) is the way out: we don't need to see the entire future at once, because today's value must contain tomorrow's value. it turns policy evaluation from an infinite summation problem into a recursion that only depends on neighboring states.

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

- $\mathbb{E}\_\pi[G\_{t+1} \mid s\_t = s]$ is you today, standing at $s_t$, predicting tomorrow's earnings $G_{t+1}$ - the number has already averaged out where tomorrow lands.
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

the second line is just the definition of a discrete conditional expectation, $\mathbb{E}[X \mid Y] = \sum_x x \cdot P(x \mid Y)$, with $X = V^\pi(s_{t+1})$. intuition per term: starting from $s$ and acting under $\pi$, there's probability $P_\pi(s' \mid s)$ of jumping to $s'$, and $s'$ is worth $V^\pi(s')$; weighting all next states by their probability is "how much the next step is worth on average".

the same move for $Q$: first the immediate reward of $a$, then the value of wherever that action lands you:

$$\begin{aligned}
Q^\pi(s_t, a_t) &= \mathbb{E}_\pi[G_t \mid s_t, a_t] = \mathbb{E}_\pi[r_t + \gamma G_{t+1} \mid s_t, a_t] \\
&= R(s_t, a_t) + \gamma\, \mathbb{E}_\pi[V^\pi(s_{t+1}) \mid s_t, a_t] \\
&= R(s_t, a_t) + \gamma \sum_{s' \in \mathcal{S}} P(s' \mid s_t, a_t) V^\pi(s')
\end{aligned}$$

### the optimality equation

the expectation version averages over the policy; the optimality version stops averaging and takes the best action. for $V^*$ the policy is greedy:

$$V^*(s) = \max_a \left[ R(s,a) + \gamma \sum_{s' \in \mathcal{S}} P(s' \mid s, a) V^*(s') \right]$$

### matrix form and the analytic solution

with a fixed policy, the action uncertainty folds into two quantities:

$$r^\pi(s) = \sum_a \pi(a \mid s) R(s,a), \qquad P^\pi(s' \mid s) = \sum_a \pi(a \mid s) P(s' \mid s, a)$$

$P^\pi$ is not the raw environment transition $P(s' \mid s, a)$ - it's the state-to-state probability after "first pick an action according to the policy, then let the environment transition". with $N$ states, the $N$ simultaneous equations stack up as

$$v^\pi = r^\pi + \gamma P^\pi v^\pi$$

where $v^\pi$ is the $N \times 1$ vector of state values under $\pi$, $r^\pi$ the $N \times 1$ vector of expected immediate rewards, and $P^\pi$ the $N \times N$ policy-induced transition matrix. rearrange:

$$v^\pi - \gamma P^\pi v^\pi = r^\pi \;\Longrightarrow\; (I - \gamma P^\pi) v^\pi = r^\pi \;\Longrightarrow\; v^\pi = (I - \gamma P^\pi)^{-1} r^\pi$$

with $I$ the identity matrix. so as long as the environment rules and $\pi$ are fully transparent, this is the exact value of every state under that policy - an analytic solution.

### then why learn any algorithm at all

because reality is harsh. the analytic solution needs inverting $(I - \gamma P^\pi)$, and matrix inversion is $O(N^3)$. let the state space get even slightly large and it's over - Go has around $10^{170}$ board positions, you don't finish that in a lifetime. this god's-eye-view solution exists in theory and in extremely simple toy environments; for everything else we approximate by iteration, with dynamic programming, monte carlo, or TD.

## learning without the model: TD error

the Bellman equations quietly assume we can look up $P$ and $R$. real tasks don't hand you a transition table. the agent takes a step, sees a reward, sees where it landed, and that's the whole data stream. so the exact target,

$$\sum_a \pi(a \mid s) \Big[ R(s,a) + \gamma \sum_{s'} P(s' \mid s, a) V(s') \Big]$$

is not computable: the $R(s,a)$ in it is an *average* reward, $P(s' \mid s,a)$ is a distribution over next states, and we have neither. what we do have is one transition that actually happened. TD error is the reconciliation - compare the target implied by that one experience against the old estimate sitting in the table:

$$\text{target} = r + \gamma V(s'), \qquad \delta = \underbrace{r + \gamma V(s')}_{\text{one-step target}} - \underbrace{V(s)}_{\text{current estimate}}$$

why is one sample allowed to stand in for the average? because the Bellman target *is* an expectation, so it's a probability-weighted average over outcomes. each real step draws one outcome from that distribution: common outcomes get drawn, and therefore learned from, more often; rare ones less. let the updates accumulate and the table drifts in the probability-weighted direction - plain monte-carlo estimation wearing a different hat.

why "temporal difference": it is the gap between two predictions made at neighboring times. standing at $s$ we held the old prediction $V(s)$; after moving to $s'$ the term $r + \gamma V(s')$ hands us a new one. comparing predictions from two moments is the whole name.

and notice what the target is built out of: $V(s')$, which is itself only an estimate. that bootstrapping is why TD can learn from incomplete episodes - it never needs the episode to finish and hand over the full $G_t$ the way monte carlo does.

| $\delta$ | meaning | how to adjust |
| --- | --- | --- |
| $\delta > 0$ | target above the estimate - we underestimated | raise $V(s)$ |
| $\delta < 0$ | target below the estimate - we overestimated | lower $V(s)$ |
| $\delta = 0$ | on this sample the two agree | barely move |

$$V(s) \leftarrow V(s) + \alpha\, \delta$$

$\alpha$ is the learning rate: how much of one sample's opinion to believe. at $\alpha = 1$ a single noisy step would overwrite the whole estimate, which is why we creep toward it instead.

## control: the Q table and Q-learning

$V$ answers "is this situation good or bad". control needs a sharper question - in state $s$, with candidates $a_1$ and $a_2$, which one is better? - and that is a comparison of $Q^\pi(s,a_1)$ against $Q^\pi(s,a_2)$, not a score for the state.

with a known model you get $Q$ from $V$ for free, because "take $a$ first, then follow $\pi$" is just the immediate reward plus the discounted value of wherever you land:

$$Q^\pi(s,a) = R(s,a) + \gamma \sum_{s'} P(s' \mid s, a) V^\pi(s')$$

same shape as the $V$ recursion, except the first action is pinned instead of averaged. but with $R$ and $P$ unavailable that formula is dead on arrival, so we do what we did for $V$: learn the values directly. grow the table from one cell per state to one cell per state-action pair - the $Q$ table - and correct one entry at a time from one transition.

three lines carry the whole idea:

$$\begin{aligned}
Q^\pi(s,a) &= \mathbb{E}_\pi[G_t \mid S_t = s, A_t = a] && \text{what } Q \text{ means} \\
Q^*(s,a) &= \mathbb{E}\big[\, r + \gamma \max_{a'} Q^*(s', a') \mid s, a \,\big] && \text{the target, for the optimal table} \\
Q(s,a) &\leftarrow Q(s,a) + \alpha \big[\, r + \gamma \max_{a'} Q(s', a') - Q(s,a) \,\big] && \text{the update}
\end{aligned}$$

- line 1 says what $Q$ means: take $a$ first, then follow $\pi$; $Q$ is the expected return of that.
- line 2 is the Bellman optimality equation for $Q$ - the self-consistency a correct optimal table has to satisfy. an entry $Q^*(s,a)$ should equal "take one step from this entry, then consult the best entry in the next state". we may not know whether our table is right, but we do know what it has to satisfy, and the size of the violation is the update direction.
- line 3 is the algorithm: build a TD target from one sampled transition and step toward it. the $\max$ is the entire difference from SARSA - the target assumes optimal play from $s'$ regardless of what the agent actually went on to do.

once the table is good, the policy falls straight out of it:

$$\pi(s) = \arg\max_a Q(s,a)$$

which is why Q-learning is **off-policy**: it learns about the greedy target policy while the behaviour policy is still $\epsilon$-greedy exploring. the thing that generated the data is not the thing being scored.

### a 4x4 gridworld, one update at a time

4x4 grid, every step costs $r = -1$, $\gamma = 0.9$, the $Q$ table starts at all zeros, and the agent starts top-left. it picks "right" and lands on $(0,1)$ with $r = -1$.

only one number moves: $Q((0,0), \text{right})$. the other three actions at $(0,0)$ and every entry in every other state are untouched. the target looks at what we just got, then at what the next cell is worth - and the next cell is still all zeros, so

$$\max_{a'} Q((0,1), a') = 0 \quad\Longrightarrow\quad \text{target} = -1 + 0.9 \times 0 = -1$$

read it plainly: this step already costs a point, and the cell we're sitting on looks neither good nor bad yet, so "go right" is worth $-1$ for now. but Q-learning doesn't overwrite the old $0$ with $-1$, it takes a step toward it:

$$Q((0,0), \text{right}) = 0 + 0.1 \times (-1 - 0) = -0.1$$

that's a mild lesson after one trial. keep sampling and the values along negative-reward paths sink while the values along the good path rise - information propagates backward from the goal to the start until the table fills in. the core loop is genuinely this small:

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

the greedy policy read off the finished table walks right along the top row, drops down the right edge to the goal, and the start state ends up around $-5$ - roughly the six $-1$ steps that path costs. the goal cell gets no bootstrap because it's terminal: nothing follows it.

## advantage and the TD residual

$Q$ and $V$ are the objects that make choices possible, and the gap between them is the advantage - how much better $a$ is than the policy's average:

$$A^\pi(s_t, a_t) = Q^\pi(s_t, a_t) - V^\pi(s_t)$$

here's the awkward fact under all of this, at least for LLMs: ideally every token would get its own reward. that "th" was a good token, that wrong digit was a bad one. we don't get that - the reward model scores the whole completion and hands back one scalar at the end, so every token in the sequence inherits the *same* number. credit assignment is completely flat, and pushing on the raw reward tells all $T$ tokens they were equally responsible for an outcome most of them had nothing to do with.

so don't push on the raw reward, push on how much better than expected this trajectory turned out - reward measured against a baseline. in shorthand $A = R(\tau) - V(s_t)$, but the proper object is the state-action gap above. and "average" means the average over the kinds of tasks the model sees, not just this one prompt: $V(s_t)$ is a general prior about how well the model does from here, and the advantage is this particular action beating that prior.

fitting a full $Q$ is expensive, so we lean on the Bellman relation - $Q$ at $(s_t, a_t)$ is the immediate reward plus the discounted value of the next state - and the advantage collapses into a temporal-difference residual that only needs a value estimate, one network:

$$Q^\pi(s_t, a_t) = \mathbb{E}\big[r_t + \gamma V^\pi(s_{t+1})\big], \qquad A(s_t, a_t) = r_t + \gamma V(s_{t+1}) - V(s_t)$$

the reading: $r_t + \gamma V(s_{t+1})$ is a sample of how good it turned out *having taken* $a$; $V(s_t)$ is the value before choosing; the gap is exactly how much this action deviated from average. so the advantage is just $\delta$ from the TD section with the action pinned - train $V$ to push $\delta$ toward zero and you have your critic. for a completion only the terminal reward is nonzero and $\gamma = 1.0$, so $G_t \equiv R(\tau)$ and there's no infinite horizon to worry about - discounting would just miscredit the later tokens.

### GAE (generalized advantage estimation)

single-step TD is noisy and biased. GAE exponentially weights the residuals to trade the two off:

$$A_t^{\text{GAE}(\gamma,\lambda)} = \sum_{k=0}^{T-t} (\gamma\lambda)^k\, \delta_{t+k}, \qquad \delta_{t+k} = r_{t+k} + \gamma V(s_{t+k+1}) - V(s_{t+k})$$

- **$\lambda \to 0$:** just the 1-step TD residual - low variance, high bias.
- **$\lambda \to 1$:** the full monte-carlo return - unbiased, high variance.

so $\lambda$ picks the point between them. for a completion, only the terminal reward is nonzero, so $\delta_t = \gamma V(s_{t+1}) - V(s_t)$ for $t < T$ and $\delta_T = R - V(s_T)$.

## in one breath

1. the tuple $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$ is the rules of the game; the five questions are how you ask them of any task.
2. $G_t$ is the objective, and $G_t = r_t + \gamma G_{t+1}$ is the recursion that makes Bellman possible. $\gamma < 1$ is what keeps infinite-horizon returns finite.
3. $\pi$ is how actions are chosen - deterministic (DQN) or stochastic (PPO) - and training looks for the $\pi^*$ that maximizes return.
4. $V^\pi(s)$ and $Q^\pi(s,a)$ score a policy, with $V = \sum_a \pi(a\mid s)Q$; the advantage $A = Q - V$ is the "better than average" signal that actually drives updates.
5. the Bellman equations turn the infinite summation into a recursion over neighboring states, and the fixed-policy matrix form gives the exact answer at $O(N^3)$ cost - which is why we iterate with DP, MC, or TD instead.
6. with no model to look up, the learning signal is the TD error $\delta = r + \gamma V(s') - V(s)$: one sampled transition standing in for the probability-weighted Bellman target, with $V(s')$ itself an estimate (the bootstrap).
7. control means growing the table to $Q(s,a)$. Q-learning's target takes the $\max$ over the next state, so it learns the greedy policy from $\epsilon$-greedy data - off-policy, one cell corrected per transition.

---

resources:
 - walkinglabs, *hands-on modern RL* - the mdp writeup came from here; https://walkinglabs.github.io/hands-on-modern-rl/en
 - Sutton & Barto (2018), *Reinforcement Learning: An Introduction* - the reward hypothesis, the Bellman material and TD learning
 - Watkins & Dayan (1992), *Q-Learning* - the off-policy TD control algorithm; https://link.springer.com/article/10.1007/BF00992698
 - Gymnasium, *FrozenLake-v1* - the 4x4 grid environment the example above is modelled on; https://gymnasium.farama.org/environments/toy_text/frozen_lake/
 - Schulman et al. (2016), *High-Dimensional Continuous Control Using Generalized Advantage Estimation* (GAE); https://arxiv.org/abs/1506.02438
 - the policy-gradient half of these notes: [RL notes](/writing/RL-notes/)
