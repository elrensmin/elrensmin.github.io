---
title: Diffusion notes
date: 2026-10-13
description: the map across my diffusion papers
tags: [diffusion, study-notes]
---

I wanted a single place to conoslidate all the papers im reading about diffusion, so here it is.

read until now:
- [Denoising Diffusion Probabilistic Models (DDPM)](/o/2006-11239/)
- [Improved Denoising Diffusion Probabilistic Models (iDDPM)](/o/2102-09672/)
- [Common Diffusion Noise Schedules and Sample Steps are Flawed](/o/2305-08891/)

## idea

take an image and destroy it on purpose with a *fixed* process, add a little Gaussian noise, over and over, $T$ times. then train one network to undo a single step. at sampling time you run that same network $T$ times and an image falls out.

that's it. no adversarial game, no bottleneck. the VAE squeezed everything through one low-dimensional latent and paid for it in blur; the GAN skipped the bottleneck and paid for it in training instability. Diffusion keeps the full dimensionality the whole way and only has to learn how to denoise.

## forward and reverse

the forward chain is a fixed Gaussian step, and the useful part is that you can jump to any timestep in closed form instead of simulating the chain:

$$
q(x_t \mid x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}\,x_0, (1-\bar{\alpha}_t)I)
$$

call the signal coefficient $a\_t = \sqrt{\bar{\alpha}\_t}$ and the noise coefficient $\sigma\_t = \sqrt{1-\bar{\alpha}\_t}$, so $x\_t = a\_t x\_0 + \sigma\_t \epsilon$. at $t=1$ the image is barely speckled. by $t=T$ the signal is essentially gone and $x\_T$ is pure Gaussian noise, which is the whole point, because that's a distribution you can sample from for free.

The reverse process is also a Gaussian chain, except this one is learned. Our bet is that one denoising step is *easy enough to learn* even though noise → image in a single shot is hopeless. The forward process being fixed is also what makes the variational bound tractable: everything in it is a KL between Gaussians. the derivation is in [DDPM, the mathematical foundation](/o/2006-11239/).

## two lenses

**the denoising lens.** write down the variational bound and it collapses into something embarrassingly simple: predict the noise $\epsilon$ that was added. That's the whole training objective.

**the score / Langevin lens.** Langevin dynamics samples from a distribution using only its gradient, $\nabla\_x \log p(x)$ (the score). The problem is that for images we can't compute the score, because we don't have $p(x)$. but for a *Gaussian* we corrupted ourselves, the score has a closed form in terms of the noise we added:

$$
\nabla_{x_t} \log q(x_t \mid x_0) = -\frac{\epsilon}{\sqrt{1-\bar{\alpha}_t}}
$$

so the score is just the noise (up to a scalar). 

That means an $\epsilon$-predicting network is a score network, and the $\epsilon$-MSE loss is denoising score matching in disguise. Sampling then looks like Langevin dynamics run at a ladder of noise levels (annealed Langevin) with the temperature schedule baked into the forward process. [DDPM, the score-function lens](/o/2006-11239/) has the full walkthrough.

## the target family: $\epsilon$, $x\_0$, $v$

once you accept that you're learning *something* about the clean data, the question is what exactly to predict. the obvious options are the noise $\epsilon$ or the clean image $x\_0$. they're related by the forward equation, so they aren't fundamentally different tasks, but they are different optimization problems: $x\_0$-prediction has huge gradients at high noise, and $\epsilon$-prediction gets trivial at low noise.

the cleaner choice is $v = a\_t \epsilon - \sigma\_t x\_0$, which is a rotation of the $(\epsilon, x\_0)$ pair and keeps the target at a roughly constant scale across all $t$. it also stays finite at the last step, which $\epsilon$-prediction doesn't. the [noise schedules note](/o/2305-08891/) has the v-target deep dive, including why it's literally the velocity of the probability-flow ODE.

## the schedule is part of the model

the noise schedule you train with and the schedule you sample with have to agree at the endpoints. Stable Diffusion's schedule never quite reaches zero signal-to-noise at $t=T$, so the model learns to respect a little leaked low-frequency signal that simply isn't there at inference, where you start from pure Gaussian noise. the result is dead-mids / limited-brightness problem.

Fix is to rescale the schedule so the terminal SNR is actually zero, train with $v$-prediction so the last step is learnable, and start sampling from $t=T$. The SNR equations are in the [noise schedules note](/o/2305-08891/).

## what iDDPM actually adds (the operational bits)

DDPM proves the idea works; iDDPM is the paper that makes it *behave*, and most of the practical instinct i use when thinking about a diffusion setup comes from here. full detail is in the [iDDPM note](/o/2102-09672/), but the parts worth carrying around are these.

**the variance is a knob, so learn it.** DDPM pins the reverse variance to one of two known bounds ($\beta\_t$ or $\tilde{\beta}\_t$) and only learns the mean. but the earliest timesteps carry almost all of the likelihood, so pinning the variance there is exactly where it costs you. instead, let the network interpolate between the two bounds **in log space**:

$$
\Sigma_\theta(x_t,t) = \exp\!\big(v \log \beta_t + (1-v)\log\tilde{\beta}_t\big)
$$

predicting the variance directly is unstable; interpolating between two things you already trust is not, and it can't leave the valid range by construction. (this $v$ is iDDPM's interpolation knob, *not* the $v$-target from the schedule paper. same letter, different job.)

**the loss has to know the variance exists.** $L\_{\text{simple}}$ doesn't depend on $\Sigma\_\theta$ at all, so it can't train it. iDDPM adds a small $L\_{\text{vlb}}$ term ($\lambda = 0.001$) with a stop-gradient on the mean. 

Think of it like $L\_{\text{vlb}}$ shapes the variance and $L\_{\text{simple}}$ shapes the mean.

**uniform $t$ is a bad default.** the per-timestep VLB terms differ in magnitude by orders of magnitude, so sampling $t$ uniformly burns the gradient on terms that barely matter, which is why $L\_{\text{vlb}}$ is so noisy. the fix is importance sampling, with the weight estimated from a running history of the last 10 values per timestep:

$$
p_t \propto \sqrt{\mathbb{E}[L_t^2]}, \qquad \sum_t p_t = 1
$$

the general lesson: if the loss is a sum over $t$, how you sample $t$ is a choice, and uniform is usually the wrong one.

**the schedule decides where the noise budget goes.** the linear schedule destroys information too fast, so the tail of the reverse chain operates on almost-pure noise and contributes nothing. Cosine schedule spreads the change out. (separate from the terminal-SNR fix above: that one is about the *endpoint*, this one is about the *shape*.)

$\Sigma\_\theta$ interpolates between schedule-dependent bounds, it **rescales itself automatically** when you sample with a shorter subsequence of timesteps. train once at $T=4000$, sample at $K=100$, no fine-tuning. that's what makes fast ancestral sampling work; DDIM is better below ~50 steps, learned variance wins above.

## sampling and guidance

sampling is the reverse process discretized, which in the continuous-time view is a reverse SDE. the practical knobs:

- **which timesteps to visit.** plain linspace is fine at 25 steps; at very few steps, trailing (denser near the data end) is noticeably better.
- **classifier-free guidance.** you extrapolate away from the unconditional prediction to push the prompt. at low terminal SNR this blows up the scale and over-exposes the image, so you rescale the guided output back to the conditional's standard deviation and blend.
- **the sampler's algebra.** the $\epsilon$-form updates divide by $a\_t$, which is zero at the terminal step. the $x\_0$-form updates don't.

all three are in the [noise schedules note](/o/2305-08891/) with the equations.

## the map

where to find the actual math, so i stop re-deriving it:

| idea | one line | math lives in |
|---|---|---|
| forward process | closed form lets you skip to any $t$ | [DDPM, math foundation](/o/2006-11239/) |
| reverse process + bound | learned Gaussian chain; the bound is all Gaussian KLs | [DDPM, math foundation](/o/2006-11239/) |
| why $\epsilon$ | score $\propto \epsilon$, so $\epsilon$-MSE is score matching | [DDPM, score-function lens](/o/2006-11239/) |
| learned variance | interpolate $\log \Sigma$ between $\beta\_t$ and $\tilde{\beta}\_t$ | [iDDPM](/o/2102-09672/) |
| cosine schedule | spend the noise budget evenly across $t$ | [iDDPM](/o/2102-09672/) |
| importance sampling | sample $t$ where the VLB term is large | [iDDPM](/o/2102-09672/) |
| fast sampling | skip timesteps; learned variance rescales itself | [iDDPM](/o/2102-09672/) |
| targets ($\epsilon$, $x\_0$, $v$) | $v$ is the stable target at the terminal step | [noise schedules, v-target](/o/2305-08891/) |
| schedule / terminal SNR | non-zero terminal SNR is the train/inference mismatch | [noise schedules, SNR](/o/2305-08891/) |
| guidance | CFG plus the rescale that stops over-exposure | [noise schedules, CFG](/o/2305-08891/) |
| sampler corrections | $x\_0$-form updates avoid dividing by $a\_t$ | [noise schedules, samplers](/o/2305-08891/) |

## still fuzzy

things i haven't nailed down, so future me knows where to dig:

- why the model bothers to read the leaked channel mean at all, and whether that's a bug or a feature.
- how this interacts with distilled / consistency samplers.
- what the continuous-time analogue of "zero terminal SNR" actually means.



## breadcrumbs

- [ddpm eq-by-equation (2006.11239)](/o/2006-11239/)
- [improved ddpm (2102.09672)](/o/2102-09672/)
- [noise schedules, cfg & samplers (2305.08891)](/o/2305-08891/)
- [kl divergence notes](/o/kl-divergence-notes/) - the ELBO gap-is-KL step used by both bound write-ups

## todos

- the [still fuzzy](#still-fuzzy) list is the canonical backlog for this cluster; fold items into the paper notes as they resolve rather than growing it here.
- add distilled / consistency samplers as their own breadcrumb row once i read one properly.
