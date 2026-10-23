---
title: Viola-Jones notes
date: 2024-11-20
description: "haar features + integral image + adaboost + cascade: real-time face detection before deep learning made it easy"
tags: [cv, study-notes, paper]
---

[Viola & Jones, CVPR 2001](https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf) (+ the [IJCV 2004 journal version](https://www.sciencedirect.com/science/article/pii/S1077314203001759)): painfully slow to train, real-time to run. before deep neural nets, this was *the* face detector, and its three tricks - integral images, boosting, cascades - are still worth knowing cold.

confession: i didn't get AdaBoost until [Krish Naik's video](https://www.youtube.com/watch?v=NLRO1-jp5F8) walked it.

## why haar-like features

the paper contrasts them with steerable filters: steerable filters are great for boundary/texture analysis, but rectangle features - sensitive to edges, bars, and simple structure - are **coarse yet cheap**. the paper's own framing: the only orientations available are vertical, horizontal, and diagonal, and the rectangle feature set "provides a rich image representation which supports effective learning". in conjunction with the integral image, their efficiency compensates for the limited flexibility.

the intuition: to find which region is lighter/darker, sum pixel values in both regions and compare. an eye region being darker than the cheeks below, a nose bridge brighter than its sides - faces contain exactly these high-contrast patterns at known relative locations. a haar feature is a pattern of rectangles (say a black pair flanking a white pair); its value = (sum of pixel values in the dark rectangles) − (sum in the light rectangles). zero for flat regions, large when the rectangles sit over very different intensities. three families in the paper's set:

- **edge features** (two rectangles side by side) - fire on edges;
- **line features** (a stripe inside a surrounding pair) - fire on line/bar structures;
- **four-rectangle features** (two pairs of opposite rectangles) - compare *diagonally opposite* region pairs against each other, the closest thing to diagonal structure sensitivity in the original set. (the truly rotated-$45°$ features people associate with OpenCV come from the later [Lienhart extension](https://link.springer.com/chapter/10.1007/978-3-540-24670-1_15), not the 2001 paper - and the "four-sided features find diagonal features" shorthand you'll meet in summaries needs exactly this untangling.)
- a known-good example: the eye-region-vs-nose-bridge contrast feature gives a strong response on the nose bridge. lots of such features together = a face signature.

## the integral image (the enabling trick)

you want every rectangle sum in $O(1)$, so build the **integral image**: a same-size table where cell $(x, y)$ holds the sum of all pixels up to and including $(x, y)$. fill it with one pass - adding the pixel plus the cumulative sums from the left and above, minus the double-counted overlap:

```python
import numpy as np

def integral_image(image):
    ii = np.zeros_like(image, dtype=int)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            ii[y, x] = image[y, x]
            if y > 0:
                ii[y, x] += ii[y - 1, x]      # cumulative from above
            if x > 0:
                ii[y, x] += ii[y, x - 1]      # cumulative from the left
            if y > 0 and x > 0:
                ii[y, x] -= ii[y - 1, x - 1]  # fix the double-counted corner
    return ii
```

then any rectangle sum is four lookups with the corner identity - with $A$ the integral value above-left of the rectangle, $B$ the value left of it across its bottom row, $C$ the value above it across its right column, $D$ its bottom-right corner:

$$\text{rect sum} = D - B - C + A$$

a rectangle sum that cost $O(wh)$ now costs 4 memory reads - and every single haar feature is a signed combination of a handful of rectangle sums.

one more thing the integral image buys, per the paper: every candidate detection window is normalized by its own **variance** (also computable from integral images, via $\sum x^2$) - so haar responses are comparable across lighting conditions. without this, every "dark eye region" feature is meaningless in a dark photo.

## boosting - adaboost picks the features

the challenge inverted: there are tens of thousands of possible features per window, and "a very small number of these features can be combined to form an effective classifier" - the question is finding them. adaboost does feature selection *and* learning at once:

- each weak classifier is one haar feature + a **threshold** $\theta_j$ and **parity** $p_j$: $h_j(x) = 1$ if $p_j f_j(x) < p_j \theta_j$, else 0. the threshold/parity get chosen (searched) to minimize *weighted* error on the training set.
- each round, adaboost **greedily picks the single weak classifier with the lowest weighted error**, adds it to the set with importance weight $\alpha_t = \frac12 \log(1/\epsilon_t)$, then **reweights the examples**: misclassified ones get *more* weight, so the next round hunts what the current set can't do yet.
- the final strong classifier is the weighted vote: $\text{sign}\big(\sum_t \alpha_t h_t(x) - b\big)$.

the "sort models by error rate, drop the useless ones" mental model is wrong - there's no global sort-and-drop; it's greedy per-round selection where the *reweighting* is the actual engine. the "useless features dropped" intuition survives only as "features never selected".

## the cascade - speed comes from structure

faces are rare: over a full scan, essentially every window is a non-face, and running the full classifier everywhere would be wasteful. the cascade splits the classifier into **stages** of a few weak classifiers each, applied in order:

- the first stage uses the single best feature (adaboost's first pick: rejects many negatives, catches nearly all positives).
- pass → try the next, more complex stage; **fail anywhere → window rejected immediately**.
- only windows that pass *every* stage are detections.

tuned with explicit stage targets (e.g. "catch 99.9% of faces with ≤50% false alarms per stage") - which is also the practical answer to "how does haar know a face is a face and not just any edge?": a single haar feature *doesn't*. it takes the boosted ensemble's selectivity plus the cascade compounding: each stage's false-positive rate shrinks the false-positive set geometrically (50% per stage → 25% after two → 12.5% after three...), while the high per-stage detection rate keeps true faces from being culled. the trade: more stages = lower false alarms but more training data needed and a higher false-negative risk, so you tune the false-alarm target and stage count experimentally ([MathWorks' cascade detector docs](https://www.mathworks.com/help/vision/ug/train-a-cascade-object-detector.html) say it the operational way).

## how it all runs together (lifecycle)

- **scan**: slide a window over the image at multiple scales (or scale *the image* up per scale level).
- **evaluate**: integral image → cheap haar responses at every position/scale.
- **cascade**: early stages kill the non-face windows instantly; survivors pass through progressively pickier stages.
- **decision**: any window passing all stages is a detection; overlapping detections across nearby windows/scales get merged into final boxes (the plain greedy merging that the paper hand-waves and every real implementation has to own).

resource notes worth keeping: haar/LBP are good at fine-scale texture → faces; [HOG](https://lear.inrialpes.fr/people/triggs/pubs/Dalal-CVPR05.pdf) fits coarse-shaped objects → pedestrians, cars. and sentdex has a hands-on [cascade walkthrough](https://www.youtube.com/watch?v=88HdqNDQsEk).

## papers & sources

- rapid object detection (CVPR'01): <https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf>
- robust real-time face detection (IJCV'04): <https://www.sciencedirect.com/science/article/pii/S1077314203001759>
- blog walkthroughs (an aaron ward facial-detection post and a haar-feature primer shaped these notes - links long since lost; go via the papers above)

## todos

- count the candidate features in a $24 \times 24$ window properly (~160k in the paper's count) and verify against the paper's table 1.
- the detection-merging step (overlapping windows → final face box): none of my sources described the actual algorithm - dig the paper's post-processing.
- note where the Lienhart/OpenCV haar set diverges from the original paper (rotated features, stage-target tweaks) since that's what anyone today actually runs.