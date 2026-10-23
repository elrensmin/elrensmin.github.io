---
title: BERT notes
date: 2024-11-29
description: "masked LM + next sentence prediction on a bidirectional transformer encoder: why masking beats left-to-right"
tags: [nlp, study-notes, paper]
---

[Devlin et al](https://arxiv.org/abs/1810.04805): pretraining that fuses context from *both directions at the same time*. the argument it opens with (still worth internalizing): standard language models are unidirectional, which restricts what architectures you can even choose during pretraining. GPT's left-to-right attention is sub-optimal for sentence-level tasks and actively harmful for token-level tasks like question answering, where you crucially need context from both sides.

the fix that came to be called **Masked LM**: instead of predicting the *next* word, randomly mask words in the sentence and predict them from the full surrounding context - left and right *simultaneously*. the older "combine a left-to-right and a right-to-left model" approaches (ELMo's two LSTMs) were missing this same-time part - two half-blind models that don't talk to each other, doing sub-optimal work. (pedantic point that i still like: "non-directional" is actually more accurate than "bidirectional".)

one note on the word *novel*: MLM as a technique is the Cloze task ([Taylor 1953](https://www.tandfonline.com/doi/abs/10.1080/00437956.1953.11679589)); what BERT made novel is running it as *the scaled pretraining objective* on a huge corpus. credit where due.

## pre-training task 1: masked lm

why can't you just train a bidirectional conditional model? because bidirectional conditioning lets every word indirectly *see itself* - in a deep stack the target word can be trivially copied through the layers. masking breaks that identity leakage.

the procedure:

- mask **15%** of token positions at random; run the whole sequence through the deep bidirectional encoder; predict **only the masked tokens** (the original vocabulary ID, cross-entropy loss). e.g. input `the man went to the MASK1 . he bought a MASK2 of milk.` → labels `store` / `gallon`.

here's the subtlety that i think explains half of BERT's behavior: masked-predict creates a **pretraining/fine-tuning mismatch** - `[MASK]` never appears in fine-tuning. the mitigation is the 80/10/10 rule for each chosen token:

1. `[MASK]` **80%**
2. a random token **10%**
3. the unchanged token **10%**

keeping 10% unchanged forces the representations to stay faithful to *every* token, since only ~1.5% of tokens are visibly corrupted but 15% are targets. and the 10% random tokens tax the model for guessing "same word" when the input was swapped.

this also explains the `bank` example: unidirectionally, `bank` in `i made a bank deposit` sees only `i made a`. BERT represents it from `i made a ... deposit` from the *bottom* of the network - deeply bidirectional, not a shallow concat of two context models.

## pre-training task 2: nsp (with a modern asterisk)

second task from any monolingual corpus: are two sentences connected, or random?

- `the man went to the store . | he bought a gallon of milk .` → IsNextSentence (50% of pairs)
- `the man went to the store . | penguins are flightless .` → NotNextSentence (the other 50%)

modern asterisk: **NSP is deprecated practice** - [RoBERTa](https://arxiv.org/abs/1907.11692) dropped it and matched/beating BERT with sentence-pair packing (full contiguous documents) instead, blaming NSP for hurting generalization. keep it in the historical picture, not the recipe.

## the input decorations

only the **encoder** half of the transformer is used. inputs get three kinds of metadata before processing:

![input representations](/images/bert-notes/pasted-image-20241128180443.png)

1. **token embeddings** with `[CLS]` prepended to the first sentence and `[SEP]` after each sentence
2. **segment embeddings** marking sentence A or B on each token
3. **positional embeddings** marking position

the `[CLS]` token's final hidden state becomes a summary vector - that's the anchor for sentence-level tasks (the pooler runs a small MLP on it).

## fine-tuning

initialize from pretrained parameters, then fine-tune *all of them* on labeled downstream data - one fine-tuned model per task, all sharing the same pretrained init. the task head is the only new thing: classification = `[CLS]` vector + a classifier layer; token-level tasks (NER, QA spans) = per-token vectors + classifiers; feature-based usage = feed BERT's hidden states into a task-specific architecture instead of fine-tuning. training corpus: BooksCorpus + English Wikipedia.

## sizes

- **BERT-Base**: 12 layers, 768 hidden, 12 heads, 110M params
- **BERT-Large**: 24 layers, 1024 hidden, 16 heads, 340M params
- trained on 4 cloud TPUs (Base) / 16 TPUs (Large) for *4 days each*

## wordpiece tokenization

`I like surfboarding! → ['[CLS]', 'i', 'like', 'surf', '##board', '##ing', '!', '[SEP]'] → [1, 48, 250, 4033, 3588, 154, 5, 2]`

every BERT input starts with `[CLS]` and ends with `[SEP]`. wordpiece splits words into frequent-character-pair chunks; unlike BPE it scores each candidate pair by:

$$\text{score} = \frac{\text{freq of the pair}}{\text{freq(first)} \times \text{freq(second)}}$$

dividing by the product of the parts' own frequencies prioritizes merges whose parts are individually *rare* - so `("un", "##able")` merges reluctantly (both are common on their own) while `("hu", "##gging")` merges fast. ([HuggingFace NLP course ch.6](https://huggingface.co/learn/nlp-course/chapter6/6) explanation above.)

## breadcrumbs

- lineage: [word2vec](/o/word2vec-notes/) gave static per-word embeddings; ELMo made them shallowly contextual; BERT made them deeply contextual. from word2vec to BERT is "one vector per word" → "one vector per occurrence".
- the attention mechanism itself is the same as GPT's - just bidirectional rather than causal. where did attention start? my [Bahdanau attention notes](/o/bahdanau-attention-notes/) predate both by four years.
- i also have a hands-on BERT worklog: [pr1-luminal-bert](/o/pr1-luminal-bert/) (open-source example work).
- code: [build BERT from scratch walkthrough](https://medium.com/data-and-beyond/complete-guide-to-building-bert-model-from-sratch-3e6562228891), [pytorch_bert](https://github.com/coaxsoft/pytorch_bert) and [BERT-pytorch](https://github.com/codertimo/BERT-pytorch)

## papers & sources

- BERT: <https://arxiv.org/abs/1810.04805>
- RoBERTa (the NSP removal): <https://arxiv.org/abs/1907.11692>
- talks: [Yannic Kilcher](https://www.youtube.com/watch?v=-9evrZnBorM), [another walkthrough](https://www.youtube.com/watch?v=RvCyukM40tU)
- my secondary sources: [Samia Khalid's guide](https://medium.com/@samia.khalid/bert-explained-a-complete-guide-with-theory-and-tutorial-3ac9ebc8fa7c), [HuggingFace course](https://huggingface.co/learn/nlp-course/chapter6/6), [sentence-transformers](https://github.com/UKPLab/sentence-transformers)

## todos

- the pretraining loss itself (masked cross-entropy) and Q/K/V self-attention mechanics: the attention mechanics now live in [transformers notes](/o/transformers-notes/); the BERT-specific §3.1 fill is what's still pending here.
- sentence-transformer pooling strategies (the UKPLab repo link was sitting in Sources unused) - what exactly does mean-pooling do to `[CLS]`-trained representations?