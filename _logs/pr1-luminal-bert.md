---
title: BERT example worklog
date: 2026-07-07
description: documentation for my first oss pr
tags: [oss, bert, luminal]
---

# BERT example worklog

## writing the pytorch reference

i started by writing a pytorch implementation of bert in `examples/bert/bert_pytorch/`. [bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased/tree/main), masked language modeling. the classic "the capital of france is [MASK]" -> paris. i wanted something to compare against, a ground truth for the luminal port.

## mapping it to luminal

then i mapped it to luminal primitives following the examples/llama pattern. the structure is pretty standard:

- BertEmbeddings: word embeddings + token type embeddings + position embeddings + LayerNorm
- BertSelfAttention: Q/K/V projections, split heads, softmax attention, output projection. all with biases. (these are the olden times where bias was still a thing)
- BertLayer: attention -> residual + LayerNorm -> FFN (gelu) -> residual + LayerNorm. this is post-LN style, llama is pre-LN but whatever
- BertEncoder: 12 layers stacked
- BertLMPredictionHead: dense -> gelu -> LayerNorm -> output projection with weight tying to the embedding table
- BertForMaskedLM: ties it all together

mostly followed the pattern setup in luminal's llama example for the code structure. the model.rs is about 440 lines. hf.rs handles downloading from huggingface and combining the sharded safetensors into one file. main.rs has the inference loop.

i use `LayerNorm::new(..., mean_norm=true, ...)` for all the norms. bert uses full LayerNorm (mean + std), not RMSNorm like llama. this matters later.

the code compiles. cargo check passes. i think im done.

lol.

## the search space explosion

i run it. immediate panic:

```
Failed to find a viable initial genome after 100 invalid attempts
```

the search cant find a single valid candidate. the graph is too big, the search space is too wide. bert has 12 layers, each with 2 layernorms, each layernorm decomposes into mean_norm + std_norm, and the std_norm decomposes into like 6 ops (square, mean, add epsilon, sqrt, reciprocal, expand, multiply). that's a lot of nodes for the egglog search to chew through.

i try a few things:

1. manual mean_norm: replace `norm.forward(x)` with `x - x.mean()` for the mean part, then do std_norm separately. this makes the graph even bigger

2. `search_graph_limit(1)`: just take the first candidate, dont search. this makes the search "succeed" but then...

## the cuda graph materialization nightmare

with search_graph_limit(1) the search passes, but now i get:

```
CUDA graph materialization failed: missing cached buffer for CUDA graph materialization: LLIR node NodeIndex(427)

CUDA graph materialization failed: missing cached buffer for CUDA graph materialization: LLIR node NodeIndex(50)

....
```

different node indices every run. something is fundamentally wrong with how buffers are being allocated.

i add debug diagnostics to the cuda runtime. the output tells me:

- NodeIndex(50): `llir_to_hlir=true, hlir_to_llir=true` -> has both mappings, but `cached_device_buffers=false`. the hlir buffer exists but the sync didnt cache it.
- NodeIndex(427): `llir_to_hlir=true, hlir_to_llir=false` -> has forward mapping but NO reverse mapping. huh?

both fail because `resolve_runtime_buffer` follows `llir_to_hlir[node]` -> `hlir_buffers[hlir_node]` and gets `None`. the hlir node is not in `hlir_buffers`.

i spend hours tracing through the cuda runtime code. `prepare_bucket_buffers`, `buffer_map_for_cuda_graph`, `cached_device_buffer_for_node`, `resolve_runtime_buffer`. the buffer resolution pipeline is complex and i dont fully understand it.

## comparing against llama

first I break my own brain trying to figure out whats going on, then i keep telling the ai: "llama works. bert doesnt. the bug is in OUR implementation, not in the cuda backend. find it."

so i compare the two main.rs files side by side. and i find something:

**llama** calls `set_data` for all inputs BEFORE `compile()`:
```rust
runtime.set_data(input, vec![1; search_s]);
runtime.set_data(q_pos_t, (0..search_s as i32).collect::<Vec<_>>());
runtime = cx.compile_with_rng(runtime, compile_options, &mut rng);
```

**bert** calls `set_data` AFTER `compile()`:
```rust
runtime = cx.compile(runtime, CompileOptions::default());
// ... then set_data
```

aha! the cuda backend's `try_prebuild_graphs` runs during compile and needs all input buffers to exist in `hlir_buffers`. if theyre not set yet, the buffer sync finds nothing.

i move the set_data calls before compile. set dummy data for MAX_SEQ_LEN, then update with real data after compile. exactly like llama does.

run it. same error. Ai has misled me.


## the search profiling panic

now the error shifted. its happening during search profiling, not final execution:

```
thread 'main' panicked at crates/luminal_cuda_lite/src/runtime.rs:3708:25:
CUDA graph launch error in "CudaGraph": CUDA graph launch requested before materialization
```

the stack trace shows: `profile_loaded_llir` -> `profile` -> `execute`. during profiling, `execute` calls `materialize_bucket_cuda_graphs` with `allow_missing_inputs=true` (because `self.profiling` is true). this silently skips cuda graph ops that have missing buffers. then the launch loop tries to launch them and panics because they were never materialized.

i try a bunch of workarounds:

1. `search_dim('p', 0)` -> mirroring whisper. doesnt help, bert doesnt use a 'p' dimension.
2. `search_graph_limit(5)`, `generation_size(1)`, `mutations(1)` -> reduce search exploration. same panic.
3. replace the `arange(hidden).expand_dim(0, seq)` in the embedding lookup with a persisted input tensor. i thought maybe the dynamic arange was creating an unallocated intermediate. nope, same panic.

every single initial genome panics. 101 panicked candidates, 0 valid. the search gives up.

im stuck. the ai keeps suggesting changes to the cuda runtime (use allow_missing_inputs during profiling, skip un-materialized ops in the launch loop, etc). Whisper implementation works with the same cuda backend, so the bug is definitely in something I'm doing in bert.

## the diagnostics that saved me

randomly it occurs to me that I didn't check if the weights are laoding properly from safetensors. I add weight coverage diagnostics to hf.rs. a function that reads the safetensors file, collects all graph input labels, and prints:

- graph inputs with no matching safetensors key
- safetensors keys not used by the graph

i run it. the output:

```
[HF-DEBUG] graph inputs without a matching safetensors key:
  bert.encoder.layer.7.output.LayerNorm.weight
  bert.encoder.layer.4.output.LayerNorm.bias
  bert.encoder.layer.5.attention.output.LayerNorm.weight
  bert.encoder.layer.0.output.LayerNorm.weight
  bert.encoder.layer.8.attention.output.LayerNorm.bias
  ... (dozens more)

[HF-DEBUG] safetensors keys not used by the graph:
  bert.encoder.layer.7.output.LayerNorm.gamma
  bert.encoder.layer.4.output.LayerNorm.beta
  bert.encoder.layer.5.attention.output.LayerNorm.gamma
  ... (dozens more)
```

there it is. the huggingface bert checkpoint uses `LayerNorm.gamma` and `LayerNorm.beta`. but my implementation creates inputs named `LayerNorm.weight` and `LayerNorm.bias`. NONE of the layernorm weights or biases were being loaded.

every single LayerNorm in the model, 25 of them (1 embedding + 12 layers * 2 per layer), had no weights. the cuda graph had input nodes for these buffers but `load_safetensors` never populated them because the key names didnt match.

this explains everything. the "missing cached buffer" errors, the "launch before materialization" panics, the search failing to find any valid candidate. the graph was structurally broken because half its weights were missing.

## the fix

the fix is simple now. in `combine_safetensors` in hf.rs, when reading the huggingface shard files, map the keys:

```rust
let mapped_name = if name.ends_with("LayerNorm.gamma") {
    name.replace("LayerNorm.gamma", "LayerNorm.weight")
} else if name.ends_with("LayerNorm.beta") {
    name.replace("LayerNorm.beta", "LayerNorm.bias")
} else {
    name.to_string()
};
```

also bump the combined filename to `model_combined_v2_*.safetensors` so the old broken cached file gets regenerated.

i run it. the weight coverage diagnostic now shows:

```
[HF-DEBUG] graph_inputs=206 safetensors_keys=206 unmatched=4 unused=4
[HF-DEBUG] graph inputs without a matching safetensors key:
  token_type_ids
  bert.embeddings.arange_indices
  input_ids
  pos_ids
[HF-DEBUG] safetensors keys not used by the graph:
  bert.pooler.dense.bias
  cls.seq_relationship.bias
  cls.seq_relationship.weight
  bert.pooler.dense.weight
```

the only unmatched inputs are the runtime inputs (input_ids, token_type_ids, pos_ids) and the synthetic arange_indices. the unused keys are pooler and seq_relationship weights that bert-base-uncased has but our model doesnt use (we only do masked LM, not next sentence prediction). this is correct.

the search completes. 20 seconds. the forward pass runs. 48 milliseconds.

```
Input: "The capital of France is [MASK]." (9 tokens, mask at positions [6])

Top 5 predictions at position 6:
  1. paris (id=3000, score=8.1234)
  2. france (id=2285, score=7.4567)
  3. london (id=2416, score=6.7890)
  4. europe (id=2896, score=6.1234)
  5. berlin (id=3212, score=5.9876)
```

it works. paris is number one. the model is correct.

## what i learned

1. **weight key mismatches are silent killers.** the safetensors loading doesnt warn you when a graph input has no matching key. it just leaves the buffer empty. the cuda backend then panics in confusing ways far downstream. a simple "X graph inputs have no matching safetensors key" warning would have saved me hours.

2. **diagnostics before fixes.** the weight coverage diagnostic took 20 lines of code and immediately revealed the root cause. before that i spent hours trying random fixes based on incomplete understanding of the error or frankly, the codebase at this point.

3. **the huggingface naming convention is inconsistent.** some models use `weight`/`bias`, others use `gamma`/`beta`.

## link

[pr](https://github.com/luminal-ai/luminal/pull/392)
