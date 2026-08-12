---
title: ML basics notes
date: 2026-08-07
description: ml basics notes for later recall
tags: [ML, notes]
---

[silvia's blog](https://silviasapora.github.io/blog/ml-interviews) suggests a bunch of basics that I don't know, or if I know them, I'm a little patchy, so I figured I'll make some notes here while I'm revising things so I have somehting to reference back later.

# Curse of Dimensionality

volume grows like L^n. every extra dim multiplies the space by another L. sounds fine until you see what it does to the data.

unit cube. 1D a line, 2D a square, 3D a cube. hit 10 dims and the volume is still 1 but the points are basically floating alone. sparse. high-dim spaces are mostly empty, and the emptier they get the less you can say about them.

remember: points get far apart in absolute terms but the relative distances collapse. largest distance in a hypercube grows ~O(√n). double the dims, only a little more spread. distances between them stop meaning much. nearest neighbor just kind of vanishes. 

real problem for anything that leans on distance, like kNN. if every point is roughly the same distance from every other point, "nearest" is basically a coin flip.

also tangled up with overfitting. volume explodes with dim, so you need a ton of data to fill the space. not just a lot, an exponential amount. more features than samples and the model memorizes noise instead of learning signal. classic overfitting setup.

occam's razor: prefer the simpler model. especially relevant here. simpler means less room to overfit. in high dim that often means shrinking the problem itself, cutting features with PCA or feature selection, so distances mean something again. dim reduction is a common warm-up before kNN. newer tricks like approximate nearest neighbors exist partly to dodge this whole mess.

not all bad though. high dim can help with linear separability, which is why kernel methods work. and deep learning is pretty good at finding structure in these big spaces, even when the geometry is against us.

source:
- [the math behind the curse of dimensionality](https://towardsdatascience.com/the-math-behind-the-curse-of-dimensionality-cf8780307d74/)
