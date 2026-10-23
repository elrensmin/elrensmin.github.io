---
title: Mamba in vision, seen as linear attention (MLLA)
date: 2024-12-20
description: "one unified formulation: selective SSM = linear attention with six distinctions; forget gate + block design carry the win"
tags: [ssm, linear-attention, study-notes, paper]
---

[Demystify Mamba in Vision: A Linear Attention Perspective](https://arxiv.org/abs/2405.16605) (the MLLA paper) is the "wait, what *is* Mamba actually doing?" paper. it rewrites the selective SSM and linear attention in one unified formulation, declares Mamba a linear-attention transformer with **six distinctions**, then ablates them. verdict: the forget gate and the block design are the core contributors; the other four barely matter. this note is my walkthrough of the paper + [repo](https://github.com/LeapLabTHU/MLLA). the SSM side lives in my [Mamba notes](/o/mamba-notes/).

## linear attention in one move

softmax attention's quadratic cost is brutal on high-resolution images. linear attention attacks the bottleneck at the softmax: replace the non-linear softmax with linear normalization, apply a kernel $\phi$ to $Q$ and $K$:

$$\text{Attn}(Q, K, V) = \frac{\phi(Q)\left(\phi(K)^\top V\right)}{\phi(Q)\phi(K)^\top \mathbb{1}_N} \qquad \text{instead of } \text{softmax}(QK^\top)V$$

![global linear attention](/images/mlla-notes/pasted-image-20241221121534.png)

the trick is the computation order: $\left(\phi(Q)\phi(K)^\top\right)V$ is $O(N^2)$ materialization, but regrouping to $\phi(Q)\left(\phi(K)^\top V\right)$ is $O(N)$ in sequence length.

![causal linear attention](/images/mlla-notes/pasted-image-20241221121651.png)

in an autoregressive setting the receptive field just gets restricted to $j \le i$ - causal linear attention, same regrouping.

## the selective SSM, written the same way

![unified formulation](/images/mlla-notes/pasted-image-20241221121831.png)

for input $x \in \mathbb{R}^{N\times C}$, Mamba applies its recurrence per-channel:

![per channel recurrence](/images/mlla-notes/pasted-image-20241221122258.png)

![unified selective ssm](/images/mlla-notes/pasted-image-20241221121935.png)

## one formulation, six distinctions

![equivalence derivation](/images/mlla-notes/pasted-image-20241221122424.png)

the equivalence, in one line:

**selective SSM = linear attention + input gate + forget gate + shortcut, minus attention normalization, minus multi-head, plus a modified block design.**

1. **input gate** - the step size $\Delta_i$, computed as $\Delta_i = \text{softplus}(x_i W_1 W_2)$, gates how much of the *current* token $x_i$ gets into the hidden state. the model learns per-token "utility": big $\Delta_i$ for useful tokens, small for filler.
2. **forget gate** - $\tilde{A}$ (henceforth $A$), decays the previous hidden state as each new token arrives.
3. **shortcut** - the $Dx$ term, skipping state entirely.
4. **no attention normalization** - linear attention's denominator is dropped; there's nothing forcing the attention weights to sum to anything.
5. **single-head** - linear attention usually runs many heads/subspaces; the SSM recurrence is one head per channel.
6. **modified block design** - the Mamba-block shape itself ([SSM notes](/o/ssm-notes/) has it).

## the forget gate carries the baggage

the forget gate has two properties that do a lot of work:

- **local bias**: every element of $A$ is strictly in $(0,1)$, so the previous hidden state *always* decays when a token arrives. recency is structurally baked in.
- **positional information**: because decay compounds over steps, the model is sensitive to token *order* - that's positional encoding for free.

![forget gate properties](/images/mlla-notes/pasted-image-20241221123007.png)

the authors' tension point: to use $A$ on images you first flatten the image into a 1D sequence and run a recurrence per token - the receptive field shrinks to *preceding* tokens only (a typo in my vault note said "proceeding"), and you pay extra latency. and images are not causal data. their speculation, which is the paper's best idea: **a positional encoding that provides local bias + order information can substitute for the forget gate.** LePE and RoPE both qualify.

![forget gate substitution experiments](/images/mlla-notes/pasted-image-20241221123508.png)

**and it works: the forget gate can be replaced with RoPE.** (the experiments swap $A$ for RoPE-style encoding and keep the results - the "must be a recurrence" intuition for SSMs takes a hit.)

![rope replacement results](/images/mlla-notes/pasted-image-20241221123611.png)

reading this right after [S4 notes](/o/s4-notes/) reframes it nicely: S4's whole identity was LTI (that's what Mamba removed), and here the *selection mechanism* itself turns out to be optional plumbing over "linear attention + positional bias". the SSM is becoming an implementation strategy, not an essence.

## training + what MESA is (and isn't, in this note)

the setup: AdamW from scratch, 300 epochs, cosine decay with 20-epoch warmup, weight decay 0.05, total batch 4096, lr $4\times10^{-3}$. RandAugment / Mixup / CutMix / random erasing. UPerNet for segmentation with Swin's training settings.

![results](/images/mlla-notes/pasted-image-20241221130205.png)

![results continued](/images/mlla-notes/pasted-image-20241221130254.png)

plus one extra: **MESA** - an overfitting-prevention strategy the authors apply when training the MLLA models, because vision-Mamba training protocols aren't as settled as ViT ones. the ablation says MESA adds 0.1-0.3 accuracy and MLLA beats vision-Mamba SOTA with or without it. what MESA *is* mechanically, my notes never say - see todos.

## architecture + the takeaway

![mlla architecture](/images/mlla-notes/pasted-image-20241221130349.png)

![block comparison](/images/mlla-notes/pasted-image-20241221130417.png)

**Mamba can be seen as a variant of linear attention transformer with specialized linear attention and a modified block design.** once you hold that, Mamba stops being exotic: it's one point in the linear-attention family tree - which also contains the [FLatten transformer](/o/flatten-transformer-notes/) with its focused linear attention. same $O(N)$ goal, different fixes for linear attention's known weaknesses (smooth attention distributions, low-rank).

## papers & sources

- MLLA: <https://arxiv.org/abs/2405.16605> - code: <https://github.com/LeapLabTHU/MLLA>

## todos

- MESA: pull the mechanism out of the paper's reference [11] (overfitting prevention via ??? - my notes never say; i got it wrong twice while guessing arXiv IDs, so look it up properly).
- transcribe the paper's unified-equation derivation (the Eq. 9-10 step-by-step inside the screenshot above is the load-bearing math) into text under "one formulation, six distinctions".
- put actual ImageNet/UPerNet score tables next to the screenshots with my own takeaways - right now the numbers live only in images.