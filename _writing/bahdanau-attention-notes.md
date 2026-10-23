---
title: Bahdanau attention notes
date: 2024-12-04
description: "neural machine translation by jointly learning to align and translate: where attention starts"
tags: [nlp, attention, study-notes, paper]
---

[Bahdanau, Cho, Bengio 2014](https://arxiv.org/abs/1409.0473): the paper that introduced the alignment mechanism everyone now just calls *attention*. four years before transformers, and the direct ancestor of everything in my [BERT notes](/o/bert-notes/).

## the bottleneck

the encoder-decoder family (Sutskever 2014; Cho 2014a) reads the whole source sentence into **one fixed-length vector**, then decodes the translation from it. the failure mode is built into the shape: a neural net must compress *everything* about the source into that one vector, which falls apart exactly on long sentences - and, nastier, on sentences longer than anything in the training corpus.

the paper's move: don't encode the sentence into a single vector. encode it into a **sequence of vectors** (annotations) and let the decoder pick a subset *adaptively, per output word* - a soft search over the relevant parts, no hard segmentation. the decoder is no longer fed a summary; it's fed a memory and a lookup method.

## the base encoder-decoder

encoder as an RNN: hidden state $h_t = f(x_t, h_{t-1})$, context from the final states $c = q(\{h_1, \dots, h_{T_x}\})$. the decoder predicts the next word given context and the previously emitted words, decomposing the joint probability into ordered conditionals:

$$p(y) = \prod_{t=1}^{T} p(y_t \mid y_1, \dots, y_{t-1},\, c) \qquad p(y_t \mid y_1, \dots, y_{t-1},\, c) = g(y_{t-1}, s_t, c)$$

(my earlier notes welded these two into a single `=` chain - wrong; the product is the joint, the $g(\cdot)$ form is each conditional.)

with: $h$ encoder hidden states, $c$ the context vector from them, $g$ a nonlinear (potentially multilayered) function producing the word probability, $s_t$ the decoder's RNN hidden state.

## learning to align and translate

![alignment attention diagram](/images/bahdanau-attention-notes/pasted-image-20241127085152.png)

the decoder's probability is now conditioned on a *distinct* context vector per target word:

$$p(y_i \mid y_1, \dots, y_{i-1}, x) = g(y_{i-1}, s_i, c_i) \qquad s_i = f(s_{i-1}, y_{i-1}, c_i)$$

the context is a weighted average over the encoder's annotations:

$$c_i = \sum_{j=1}^{T_x} \alpha_{ij}\, h_j \qquad \alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{T_x}\exp(e_{ik})} \qquad e_{ij} = a(s_{i-1}, h_j)$$

- $\alpha_{ij}$ is a [softmax](/o/softmax-1/) over alignment scores - a probability distribution over source positions for target position $i$. (the weights multiply annotation $h_j$; my earlier notes wrote $h_{ij}$ once, which doesn't exist.)
- $e_{ij} = a(s_{i-1}, h_j)$ is the **alignment model**: a *learned* score for how well the inputs around position $j$ match the output at position $i$, jointly trained with everything else. the paper's parametrization is the additive form

$$e_{ij} = v_a^\top \tanh\!\big( W_a s_{i-1} + U_a h_j \big)$$

- additivity is why "Bahdanau attention" and "additive attention" are synonyms; later work swapped the score for a plain dot product (multiplicative attention → the transformer's scaled dot-product), but the shape of the *mechanism* - score, normalize, weighted-sum - never changed.

the part to internalize: $\alpha_{ij}$ (or its energy $e_{ij}$) reflects the importance of annotation $h_j$ w.r.t. the previous decoder state $s_{i-1}$ when deciding $s_i$ and generating $y_i$. in other words - the decoder decides which parts of the source to attend to. the mechanism is *attention*, and its side effect is that the encoder is relieved of the fixed-length-vector burden.

## the bidirectional encoder, or why h_j is whole-sentence-aware

"each annotation contains information about the whole input sequence, with a strong focus on the parts surrounding the $j$-th word" - that's not an RNN property, it's a *construction* choice. the encoder is bidirectional: a forward RNN reading left-to-right and a backward RNN reading right-to-left, and

$$h_j = \big[\, \overrightarrow{f}(x_j, h_{j-1}) \,;\, \overleftarrow{f}(x_j, h_{j+1}) \,\big]$$

their concatenation. the forward half knows the prefix, the backward half knows the suffix - so an annotation is a full-sentence embedding centered around position $j$. the alignment weights can then match an output to *precisely the right neighborhood* without the encoder having compressed anything.

## why this mattered

- translations stop degrading with sentence length (the fixed-vector ceiling is gone).
- the alignment weights are *interpretable*: heat maps over source words per translated word - soft alignments learned with zero alignment supervision.
- and as a mechanism, it scaled beyond RNNs: same score-normalize-weighted-sum, different backbone. everything from [transformers](https://arxiv.org/abs/1706.03762) onward is this equation with the RNNs removed.

## papers & sources

- Bahdanau et al: <https://arxiv.org/abs/1409.0473>

## todos

- the paper's GRU implementation details ($f$, $g$, reset/update gates) never made it into these notes - add the GRU cell equations so the note stands alone.
- pull the BLEU results (RNNsearch-50 en-fr vs RNNencdec, the long-sentence plot) - the "length robustness" claim deserves its numbers.
- connect forward properly: additive → dot-product → scaled → multi-head, and where q/k/v naming appears ([Rush's annotated transformer](https://nlp.seas.harvard.edu/annotated-transformer/) is queued for that).