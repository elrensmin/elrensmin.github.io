---
title: SegMamba notes
date: 2024-12-21
description: "3D medical image segmentation with Mamba: tri-orientated scan + gated spatial conv + uncertainty-filtered skips"
tags: [ssm, medical-imaging, study-notes, paper]
---

[SegMamba](https://arxiv.org/abs/2401.13560) is where my [SSM cluster](/o/ssm-notes/) stops being theory: a Mamba-based whole-volume 3D medical segmentation backbone. this is the architecture family i actually work in - [UNet-style](/o/unet-notes/) encoders (UNETR's transformer variant, SwinUNETR) getting the [Mamba](/o/mamba-notes/) treatment. the paper also ships a dataset: **CRC-500**, 500 3D colorectal-cancer CT scans with expert annotations.

## the shape of it

three components:

![segmamba architecture](/images/segmamba-notes/pasted-image-20241221202117.png)

1. a **3D feature encoder** with multi-scale **TSMamba** blocks doing global modeling at each scale,
2. a **conv-based 3D decoder** predicting the segmentation,
3. **skip connections** with **feature-level uncertainty estimation (FUE)** filtering the encoder features.

the encoder design question is the interesting one. transformer-based segmenters like UNETR downsample the 3D input straight to $D/16 \times H/16 \times W/16$ before the attention layers, because sequence length is what kills them - but that nukes the multi-scale features a decoder needs. TSMamba's pitch: mamba's linear cost lets you keep the multi-scale encoder *and* get global modeling at every scale, at speed.

## the TSMamba block

![tsmamba block](/images/segmamba-notes/pasted-image-20241226094000.png)

![tsmamba block details](/images/segmamba-notes/pasted-image-20241226095808.png)

### GSC - gated spatial convolution before the scan

the problem: a Mamba layer flattens 3D features into a 1D sequence (see [tri-orientation below](#tom---the-scan-in-three-directions)), which throws away spatial structure. GSC restores it *before* flattening:

- two parallel conv branches on the input - kernels $3\times3\times3$ (context) and $1\times1\times1$ (per-voxel), each a norm+conv+nonlinearity block;
- multiply the two pixel-wise - a gate deciding what information gets through;
- a third conv block fuses the result, with a residual reconnecting the input.

![gated spatial convolution](/images/segmamba-notes/pasted-image-20241226100641.png)

(the gate idea will feel familiar from the [mamba block](/o/mamba-notes/): branch one computes content, branch two is the valve.)

### toM - the scan in three directions

the stock Mamba block models global dependencies in *one* direction, which is nonsense for a 3D volume - scan order would arbitrarily privilege one axis. so: flatten the 3D feature map along three orientations and run the selective scan per direction, then fuse:

![tri-orientated mamba](/images/segmamba-notes/pasted-image-20241226100919.png)

three passes over the same volume from three sides, each with a different "front". the fusion is what makes the global context direction-agnostic again.

## FUE - uncertainty-filtered skips

![feature level uncertainty estimation](/images/segmamba-notes/pasted-image-20241226101741.png)

FUE sits on the skip connections: it estimates feature-level uncertainty and filters the multiscale encoder features before they reach the decoder, improving feature reuse. fair warning: this section is currently *only* the screenshot above - zero text explaining the mechanism. see todos.

## training

pytorch 2.0.1 + monai 1.2.0, random crops of $128^3$, batch 2 per GPU, cross-entropy loss, SGD with polynomial lr schedule ($10^{-2}$ initial, $10^{-5}$ decay), 1000 epochs, augmentations: additive brightness, gamma, rotation, scaling, mirror, elastic deformation. 4×A100. per dataset: 70/10/20 train/valid/test split of the volumes. (the elastic deformation should ring a bell from the original [U-Net](/o/unet-notes/) paper - same trick, twelve years later.)

## ablations, in numbers

![ablation tables](/images/segmamba-notes/pasted-image-20241226102421.png)

![ablation tables 2](/images/segmamba-notes/pasted-image-20241226102441.png)

module ablation (dice / hd95), reading off the table:

| config | dice | hd95 | delta |
| :--- | :---: | :---: | :--- |
| M1: mamba layer only | - | - | baseline |
| M2: + GSC | - | - | +2.88 dice, −13.95 hd95 |
| M3: M2 + toM | 47.22 | 33.32 | +1.22 / −9.97 over M2 |
| full: + FUE | **48.46** | **28.52** | GSC+ToM pair gains +1.69 / −7.29 |

the two structural modules each pay for themselves; GSC (spatial repair) buys more than ToM (tri-direction) does alone, and together they're the bulk of the win.

efficiency ablation: UX-Net (large-kernel convs) and SwinUNETR (windowed attention) both grow receptive fields locally but can't do *global* range; a plain self-attention backbone is computationally infeasible at this resolution. SegMamba's TSMamba gets globally-aware modeling with better training memory and inference time *even though the flattened sequence length reaches 260k* - that's the linear-complexity payoff of the ssm lineage showing up in wall-clock.

## papers

- SegMamba: <https://arxiv.org/abs/2401.13560>

## todos

- FUE mechanism: extract §2.2 of the paper properly (my notes only kept the figure). it's the module I most want to steal for my own stacks.
- confirm CRC-500 is actually public and where (the paper promises it; verify release).
- pull the main comparison table (vs nnUNet/UNETR/SwinUNETR across the three datasets) into the note instead of leaning on the ablations only.
- try the tri-orientation idea against my own 2D stacks - axial/coronal/sagittal flattening is a cheap trick to test.