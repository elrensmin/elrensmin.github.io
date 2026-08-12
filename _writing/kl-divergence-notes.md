---
title: KL divergence notes
date: 2026-08-12
description: notes on entropy, cross-entropy, conditional entropy, KL divergence and mutual information
tags: [ML, info-theory, notes]
---

i kept running into KL divergence and cross-entropy in like four different places at once and i could never give a straight answer when someone asked what KL actually *was*. embarrassing honestly. so i saved a pile of pages, sat with it for an evening, and these are the notes i want to come back to.

a warning before you read on: this is me thinking out loud, not a textbook. Also the example spoken of here is essentialy from [a primer on information theory](https://michael-franke.github.io/logic-materials/01-handouts/07-handout-info-theory.pdf)

# surprisal, or: where the log comes from

everything builds off one idea. an event with probability p has information content `-log(p)`. that's the seed. Primer writes it `I_P(x) = -log2 P(x)`. it's saying the surprise is measured *against a specific agent's beliefs*. not floating free. Jones thinks sunny is likely, Smith doesn't, so the same sunny day surprises them differently. the P is whose eyes you're looking through.

why the negative log? three jobs, and once i saw them separately the whole thing clicked:

- certain event (p=1) carries zero info. `-log(1) = 0`. you already knew it was gonna happen.
- rare event carries more. p → 0 means `-log(p)` → ∞. something that almost never happens, happening, is genuinely surprising.
- independent events add. `I(p1·p2) = I(p1) + I(p2)`. logs turn products into sums, that's the whole reason we use them.

fair coin: p=0.5, `-log2(0.5) = 1` bit. one bit, you need one binary digit to pick heads or tails. biased coin at 0.9 heads though, heads is only worth 0.152 bits (boring, you expected it) and tails is 3.32 bits. the surprise lives in the rare outcomes.

# entropy

> measures the expected surprisal of an agent whose beliefs are P when true distribution is P.

entropy is the *expected* surprisal. average it out over everything that can happen.

```
H(P) = Σ P(x) I_P(x) = -Σ P(x) log P(x)
```

spread out = high entropy = hard to predict. concentrated = low = easy. fair die over six faces sits at 2.585 bits, that's the ceiling for six outcomes. load the die so 6 comes up half the time and it drops to 2.161. a die that *always* rolls 6? zero.

the uniform distribution always wins on entropy, `log(n)`. not a dice fact, it's general.

# joint entropy

this one's almost not worth its own name. if you have a joint distribution R over two variables (say weather and whether swallows fly high or low in the evening) the joint entropy is just... the entropy of R.

```
H(P, Q) = H(R) = -Σ R(x,y) log R(x,y)
```

Clark believes sunny-and-high is 0.24, cloudy-and-high is 0.08, and so on across the grid. crunch it and `H(R) ≈ 2.22`. that's it. it's just entropy applied to pairs instead of singles. slightly boring but you need it for the next two.

# conditional entropy

this is where it gets useful. conditional entropy `H(P|Q)` is the expected surprisal about X *after* you've already observed Y.

```
H(P|Q) = -Σ Q(y) Σ R(x|y) log R(x|y)
```

read it as: for each value of Y, compute the entropy of X given that value, then average over Y weighted by how likely each Y is. Clark watches the swallows every evening and *then* predicts weather. `H(P|Q) ≈ 1.242` for his beliefs. that's his remaining uncertainty about weather once he's seen the swallows.

if X and Y are independent, `H(P|Q) = H(P)`. seeing Y tells you nothing, so your uncertainty doesn't budge. conditional entropy is always ≤ the marginal entropy. knowing Y can only help or do nothing, never hurt.

# KL divergence

> measures how much more surprised an agent is (on avg) when they hold beliefs described by Q instead of true distribution P.

```
D_KL(P || Q) = Σ P(x) log(P(x)/Q(x))
```

the framing that stuck: how many *extra* bits do i burn if i built my encoding around Q but the data really follows P? that one sentence did more for me than the formula did in a week. Mr Franke puts it even better -> "expected *difference* in information content." KL is `Σ P(x) (I_Q(x) - I_P(x))`. the surprisal under Q minus the surprisal under P, averaged using the true distribution. excess surprisal. unnecessary perplexity. same idea, said cleaner.

properties:

- never negative. ever. Gibbs' inequality.
- zero exactly when P = Q.
- **not symmetric.** D_KL(P\|\|Q) ≠ D_KL(Q\|\|P). 

remember, this is not a distance as it doesn't work both ways. just a divergence. 

# cross-entropy (the one that matters for ML)

> measures expected surprisal of an agent whose beliefs are Q when true distribution is P.

formula first:

```
H(P, Q) = -Σ P(x) log Q(x)
```

Also:

```
H(P, Q) = H(P) + D_KL(P || Q)
```

cross-entropy equals the true entropy plus the KL divergence between truth and your model. and H(P) is fixed... it's the entropy of the actual data, you can't touch it. 

so the only thing you control is the KL part. minimizing cross-entropy *is* minimizing KL. same thing. 

that's why cross-entropy is the default classification loss and not some arbitrary choice, which is what i'd been treating it as for way too long.

quick numbers. P = (0.5, 0.3, 0.2), Q = (0.4, 0.4, 0.2). H(P) = 1.0297 nats, H(P,Q) = 1.0549. and 1.0297 + 0.0253 = 1.0550. the tiny gap is rounding. cross-entropy is always ≥ entropy because KL never goes negative. bottomed out when Q = P.

# the asymmetry thing, which i keep messing up

ok this is the part that actually got me. D_KL(P\|\|Q) and D_KL(Q\|\|P) are different numbers. the Jones/Smith example: one direction is ≈ 1.19, the reverse is ≈ 1.01. not even that close, and those distributions aren't wildly different. push them further apart and the two directions diverge hard, and not in a predictable way.

and this matters in practice. if you're using KL as a loss or a regularizer or in variational stuff, *which direction you write it changes what you're optimizing*. forward KL (P\|\|Q) is mode-seeking, reverse KL (Q\|\|P) is mode-covering, or maybe i have those backwards, i had to look it up twice while writing this and i'll probably have to look it up again next month. point is: the direction is a design decision. treat it like one.

# mutual information

> measure of the change in uncertainity about P between a sttae before learning about Q and after learning about Q

so if conditional entropy is "uncertainty about X after seeing Y," mutual information is the flip side: how much did seeing Y *reduce* that uncertainty?

```
I(P, Q) = H(P) - H(P|Q)
```

that's it. entropy before minus entropy after. the drop. if Y tells you nothing about X, `H(P|Q) = H(P)` and `I = 0`. if Y fully determines X, `H(P|Q) = 0` and `I = H(P)`.

Clark knows weather and swallows are correlated (his joint is R). Jackson has the same marginals but assumes they're independent (his joint is S, just P(x)·Q(y)). mutual information is the KL between R and S -> how wrong Jackson is, on average, for treating them as independent. works out to `I(X,Y) ≈ 0.24` bits.

there's an equivalent form i like better for intuition:

```
I(P, Q) = H(P) + H(Q) - H(P, Q)
```

uncertainty in X plus uncertainty in Y, minus the uncertainty in the joint. whatever's left over is the shared information. 

mutual information *is* symmetric. `I(P,Q) = I(Q,P)`. the one measure in this whole family that goes both ways. 


# the shape of the whole thing

here's what i finally got. every measure here is one of two templates:

- **expected information content**: `Σ P(x) I(x)`. entropy, cross-entropy, joint entropy, conditional entropy all live here. you're averaging surprisal under some distribution.
- **expected *difference* in information content**: `Σ P(x) (I_Q(x) - I_P(x))`. KL divergence and mutual information live here. you're averaging the *gap* between two surprisals.

that's the spine. surprisal is the atom. everything else is either an average of surprisals or an average of differences between surprisals. once you see it that way the menagerie stops being six separate formulas and becomes two patterns with variations.

i still get the asymmetry direction wrong. working on it.
