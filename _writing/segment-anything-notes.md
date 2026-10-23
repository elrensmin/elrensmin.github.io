---
title: Segment Anything notes
date: 2024-12-03
description: "SAM: task + model + data engineered together - promptable segmentation, 3 masks for ambiguity, a billion-mask data engine"
tags: [segmentation, transformers, study-notes, paper]
---

[Kirillov et al 2023](https://arxiv.org/abs/2304.02643): the paper that reframed segmentation from "train a model per dataset" to "train once, prompt forever". the plan hangs on three components - **task**, **model**, **data** - and three questions:

1. what task enables zero-shot generalization to new segmentation problems?
2. what model supports that task?
3. what data can power it?

answer: **promptable segmentation** - give the model an image + a prompt (point, box, mask, text) and it returns a segmentation mask, even without knowing what the object is. the design that satisfies all three constraints is deliberately boring: powerful image encoder → image embedding; lightweight prompt encoder → prompt embedding; combine in a light mask decoder that predicts masks.

![sam three components](/images/segment-anything-notes/pasted-image-20241203160817.png)

## model design

![sam model design](/images/segment-anything-notes/pasted-image-20241203162858.png)

- **image encoder**: an [MAE-pretrained ViT](https://openaccess.thecvf.com/content/CVPR2022/papers/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.pdf), minimally adapted for high-resolution inputs. runs **once per image**, before any prompting - prompts come cheap *after* this.
- **prompt encoder**: two sets of prompts - **sparse** (points, boxes, text) and **dense** (masks). points/boxes become positional encodings summed with learned per-type embeddings; free-form text goes through CLIP's text encoder. dense mask prompts are embedded with convolutions and summed element-wise into the image embedding.
- **mask decoder**: maps image embedding + prompt embeddings + an output token to a mask. a modified transformer decoder block (the lineage is [DETR](https://arxiv.org/abs/2005.12872) and Mask2Former-style per-pixel-classification work) with **two-way cross-attention** (prompt↔image) prompt self-attention to update all embeddings. after **two blocks**: upsample the image embedding, and an MLP maps the output token to a dynamic linear classifier over mask channels.
- **runtime is the point**: with the image embedding precomputed, prompt encoder + mask decoder run in ~**50 ms in a web browser, on CPU**. that's what makes interactive, real-time prompting possible.

## resolving ambiguity - the clever bit

one output mask per prompt would *average* over several valid interpretations of an ambiguous prompt (is the point on a shirt the fiber, the shirt, or the person?). fix: predict **3 output masks** per prompt - whole, part, subpart (ambiguous nesting rarely runs deeper than 3). during training, backprop only through the **minimum-loss** mask; each mask also gets a confidence score (predicted IoU) so the model can rank them at inference.

## losses, training, and the data engine

supervision: linear combination of **focal loss + dice loss** (the focal+dice recipe DETR uses). training simulates interactivity: for each mask, sample geometric prompts in **11 rounds** so SAM learns the full prompt→mask→refine loop.

the data engine, three stages, each feeding a retrain:

1. **model-assisted manual annotation**: annotators click foreground/background points in a browser tool powered by SAM. annotating time drops from **34 s → 14 s** per mask as the model improves.
2. **semi-automatic, prominent objects first**: SAM proposes candidate masks; annotators add the missing prominent objects. **5.9 M masks** added here, bringing the running total to ~10.2 M before retraining; the stage's real function is hardening the model against confusion between classes and instances.
3. **fully automatic**: prompt SAM with a **32×32 grid** of points over every image, keep stable masks (mean confidence ≥0.5 ± δ), filter duplicates with NMS, plus zoomed-in crop augmentation for small objects.

the output: SA-1B - **11 M images and 1.1 B masks** - released openly, masks of higher quality than fully-manual datasets, and the foundation of the zero-shot results.

## zero-shot results (the part worth remembering)

- **23 segmentation datasets**, single prompt-able model, compared against a per-dataset [RITM](https://github.com/saic-vul/ritm_interactive_segmentation): with a human-in-the-loop oracle, 16/23 improvements; fully automatic mask quality numbers trail the interactive setting.
- **edges** (BSDS500): prompt with a 16×16 foreground point grid → 16×16×3 = 768 candidate masks → filter → Sobel. low-input-resolution baselines still beat it there, but so do human disagreements.
- **object proposals** (as a proposal-generator stage for detection) and **instance segmentation** by prompting SAM with [ViTDet](https://arxiv.org/abs/2203.16527) boxes - composing SAM as a *component* of bigger systems, which is the usage pattern that stuck.
- **text-to-mask**: a proof-of-concept with a CLIP text encoder, foreshadowing the promptable-everything world.

## takeaways

the lesson i keep: task/model/data designed *together* beat any single clever architecture. the architecture is deliberately unremarkable (ViT + small decoder); the task definition (promptability) and the data engine loop are the moat. this is also where [my segmentation stack](/o/segmamba-notes/) got its "foundation-model first" habit from.

## papers & sources

- SAM: <https://arxiv.org/abs/2304.02643> - project: <https://segment-anything.com>
- intro walkthrough: [V7 Labs blog](https://www.v7labs.com/blog/segment-anything-model-sam)

## todos

- SA-1B license + size-on-disk, and the SAM 2 (video, memory-attention) delta worth a second note.
- the text-to-mask experiment: the paper introduces it but my note never recorded the outcome - dig §D and write what actually happened.
- SAM's limitations section (weak semantic grouping, no per-region semantics) - map it against the MLLA/segmentation notes.
- try SAM as a mask-prior generator for my own segmentation training loops - cheap win probably sitting there.