---
title: some heuristics
date: 2026-08-31
description: some experimental heuristics for LM training
tags: [logs]
---

## [Main-8b-retro](https://marin.readthedocs.io/en/latest/reports/marin-8b-retro/)

in the cooldown phase
  > To better utilize the hardware, we increased our batch size by 3x, to 12Mi tokens at around 2.77e12 tokens.... we increased the learning rate to 1.7e-3, which is approximately the old learning rate multiplied by $\sqrt{3}$ (the square root of the batch size increase).

microannealing
> The most important takeaway was that, in microannealing experiments, naively oversampling “High Quality” (HQ) data did not improve task performance. It did, however, consistently improve loss on HQ validation sets (e.g., Paloma’s various subsets)—which isn’t surprising, since those eval sets often came from the same or similar domains as the HQ training data. **We believe this is because typical "high quality" data sources (e.g. ArXiv, Wikipedia) don't have as much fewshot-learning-inducing data (e.g. multiple choice questions) as the broader web does. When you replace such a large fraction of the Pretraining (PT) mix with a HQ source, you lose out on this data and task performance suffers.**

> FLAN was designed to improve fewshot-learning performance, and so it was a perfect fit for our microannealing experiments. Instead of the 70% PT / 30% HQ recommended in the Llama 3 paper, we found that 70% PT/ 15% FLAN/15% HQ led to the best results for our experiment budget.

z-loss

>  when starting from scratch, z-loss increases the norm of the lm_head, while it decreases the scale on the final layer norm. (In retrospect, this makes sense, for two reasons. First, the final layer norm has a disproportionate impact on the scale of the logits compared to any one component in the lm_head. Second, layer norms are typically not weight decayed, so z-loss is the only regularization pressure on it.) So, z-loss, it's not just for avoiding explosions.

huh?

gemini says:

> Step A: Z-loss shrinks $\gamma$
> 
> Z-loss demands that the overall logits $z$ get smaller. The optimizer looks at $z = W(\gamma \cdot h)$ and realizes it's mathematically "cheaper" to shrink the single parameter $\gamma$ than to shrink the millions of parameters in $W$, especially since $\gamma$ has no weight decay pushing back against it. So, $\gamma$ decreases.
> 
> Step B: Cross-Entropy forces $W$ to grow
> 
> Now $\gamma$ is very small. But look at the Cross-Entropy requirement:
> $$(W_c - W_i)(\gamma \cdot h) \rightarrow \text{Still needs to be large}$$
> If $\gamma$ becomes tiny, the only mathematical way to keep the final output large enough to satisfy Cross-Entropy is if $(W_c - W_i)$ gets much bigger.Therefore, to compensate for the tiny $\gamma$ being fed into it, the norm of $W$ (the lm_head) must increase, fighting against its own weight decay.

Also check out [this](https://wandb.ai/marin-community/marin/reports/ZLoss-vs-Not-1-4B--VmlldzoxMjEzMzA1NA)

wsd-s phase

By periodically doing rapid cooldowns they got an accurate read on the model's converged performance and evaluation metrics without wasting computational resources (FLOPs) committing to a final, permanent cooldown.
