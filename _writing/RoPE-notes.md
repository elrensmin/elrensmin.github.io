---
title: RoPE notes
date: 2026-08-28
description: rotary position embeddings notes for later recall
tags: [RoPE, notes]
---

I owe a debt of gratitude to this dude for making [RoPE click for me](https://www.youtube.com/watch?v=hCzJo4ui1P8)

so BERT and GPT-2 just add the position vector to the token vector $$\vec{v}_{\text{token}} + \vec{v}_{\text{pos}}$$. the whole thing is one move (rotate) done in a way that turns position into an angle, and angles add.

the whole idea answers four questions in order, and each answer hands you the next one.

## 1. How do you turn position into an angle?

scale it linearly. for position index $m$, the angle is $m \cdot \theta$ - position times some base angle $\theta$. walk forward in position, you spin around:

| pos $m$ |   angle   | rotation    |
| :-----: | :-------: | :---------- |
|    0    |    $0$    | no spin     |
|    1    | $\theta$  | one spin    |
|    2    | $2\theta$ | two spins   |
|    3    | $3\theta$ | three spins |
|   $m$   | $m\theta$ | $m$ spins   |

geometrically this traces a helix in vector space.

![helix](/images/notes/rope-1.png)

the rotation itself is the 2D matrix:

$$R(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$$

**worked example.** $v = \begin{bmatrix} 1 \\ 2 \end{bmatrix}$, rotated by $\theta = \frac{\pi}{2}$ ($90^\circ$). $\cos\frac{\pi}{2} = 0$, $\sin\frac{\pi}{2} = 1$, so:

$$R\!\left(\tfrac{\pi}{2}\right) = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}, \qquad
v' = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}\begin{bmatrix} 1 \\ 2 \end{bmatrix} = \begin{bmatrix} -2 \\ 1 \end{bmatrix}$$

$\lVert v \rVert = \sqrt{1+4} = \sqrt{5}$, $\lVert v' \rVert = \sqrt{4+1} = \sqrt{5}$. magnitude preserved, only the angle moved. that's the whole point. the semantic content is the magnitude, the position is the angle, and rotating keeps one while moving the other.

a real hidden state is $d$-dimensional, not 2D. how does this scale?

## 2. How do you rotate a big-dimensional vector?

split it into 2D pairs and rotate each one. a $d$-dim vector is just $d/2$ stacked 2D planes:

$$[(x_0, x_1),\; (x_2, x_3),\; \dots,\; (x_{d-2}, x_{d-1})]$$

rotate each pair with its own $R$. the 2D trick from §1 scales by copy-paste. $d/2$ independent little rotations, one per plane.

clean. but there's a problem. if every pair rotates by the same $\theta$, all pairs are identical. pair $i$ and pair $j$ spin in lockstep, so the model can't tell them apart. every pair carries the same positional signal. that's useless.

## 3. How do you make each pair unique?

give each pair $i$ a different **frequency** $\theta_i$. a frequency here is essentially just a number attached to the pair to make it unique. how fast that particular pair spins per step of position.

$$\theta_i = 10000^{-2i/d}$$

**how fast does the $i$-th pair rotate?** walk the formula:

- $i = 0$: $\theta_0 = 10000^{0} = 1$ → rotates fastest, one full unit per position step. short-range.
- $i \to d/2$: $\theta_i \to 10000^{-1} \approx 0.0001$ → barely moves. long-range.

so the spectrum goes $fast → slow$ across the pairs. one half of your hidden state is a fast-rotating clock, the other half barely moves. short-range dependencies read off the fast pairs, long-range off the slow ones. neat split.

**algorithm steps:**
1. calculate base frequencies $\theta_i = 10000^{-2i/d}$ for each pair $i$.
2. compute total angle $\alpha = m \cdot \theta_i$ (position × frequency - §1's move, now per-pair).
3. rotate each pair: $$(x'_{2i},\, x'_{2i+1}) = \text{Rotate}(x_{2i},\, x_{2i+1},\, \alpha)$$.

ok so we've turned position into per-pair angles. applying this to our core attention operation.

## 4. Does relative distance actually fall out?

query $q$ at position $m$ becomes $q' = R(m\theta)\,q$, key $k$ at position $n$ becomes $k' = R(n\theta)\,k$. the dot-product attention score between them:

$$\text{score} = q'^{\top} k' = q^{\top} R(m\theta)^{\top} R(n\theta)\, k$$

apply the rotation identities $R(\alpha)^{\top} = R(-\alpha)$ and $R(\alpha)R(\beta) = R(\alpha + \beta)$:

$$R(m\theta)^{\top} R(n\theta) = R(-m\theta)\, R(n\theta) = R\!\big((n - m)\theta\big)$$

so:

$$\text{score} = q^{\top} R\!\big((n - m)\theta\big)\, k$$

the inner product depends only on $(n - m)$, the relative distance. no explicit relative position bias term needed.

## 5. PyTorch: make it fast

coda, not part of the intuition chain. just how you avoid paying for what you can precompute.

**precompute the cache.** precalculate $\text{inv\_freq} = 10000^{-2i/d}$, outer-product with position indices $m$, and cache the $\sin$ and $\cos$ matrices so the whole thing can JIT-compile. positions are known up front, frequencies are known up front. no reason to recompute them in the forward pass.

**fast vectorized rotation.** instead of doing explicit 2D matrix multiplications, rewrite the standard rotation math. the definition:

$$x'_0 = x_0 \cos\alpha - x_1 \sin\alpha, \qquad x'_1 = x_0 \sin\alpha + x_1 \cos\alpha$$

rearrange so each output is a linear combination with the same pair:

$$x'_0 = x_0 \cos\alpha + (-x_1) \sin\alpha, \qquad x'_1 = x_1 \cos\alpha + x_0 \sin\alpha$$

now read it as elementwise: take $[x_0, x_1]$ and $[-x_1, x_0]$ (swap the pairs and negate the first element), then

$$[x'_0,\, x'_1] = [x_0,\, x_1] \odot \cos\alpha \;+\; [-x_1,\, x_0] \odot \sin\alpha$$

all elementwise muls against the precomputed $\cos$ and $\sin$ tensors. no matmul, no gather. the rotation is just a swap-negate plus two elementwise products. done.

