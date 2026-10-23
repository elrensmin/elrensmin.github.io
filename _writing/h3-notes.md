---
title: Hungry Hungry Hippos (H3) notes
date: 2024-12-13
description: "two fixed SSMs (shift + diag) wired into an attention-like layer, built to do associative recall"
tags: [ssm, study-notes, paper]
---

[Hungry Hungry Hippos](https://arxiv.org/abs/2212.14052) (dao, fu, saab, arora, rudra, ré) is the SSM-family paper that *stops* treating the SSM as one blob and instead rebuilds attention's function out of two specialized SSMs - explicitly because time-invariant SSMs fail at language modeling's nastiest requirement, **associative recall** (my [S4 notes](/o/s4-notes/) and [Mamba notes](/o/mamba-notes/) touch this). so despite how it reads in my vault's first line ("marks a departure from the ssm based model"), H3 is *in* the SSM family - what it departs from is the one-SSM-per-layer design. the implementation sketch below is transcribed from a multi-part annotated-SSM tutorial i worked from (the same series the code cells in the [S4 note](/o/s4-notes/) come from).

## the two SSMs

every H3 layer contains two SSMs, distinguished purely by the structure of their $A$ matrices.

### SSM-shift - a queue

$A$ is fixed (never learned): the sub-diagonal ones matrix.

![ssm shift matrix](/images/h3-notes/pasted-image-20241213171010.png)

multiplying by this matrix shifts every element of the state one slot - watch it in action:

```python
import torch

def shift_matrix(state_size):
    return torch.diag(torch.ones(state_size - 1), diagonal=-1)

shift_A = shift_matrix(5)
state = torch.arange(1, 6, dtype=torch.float32)
print("Initial state vector:", state)

for _ in range(5):
    state = shift_A @ state
    print("state = shift_A @ state -", state)
```

```
Shift matrix:
tensor([[0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0.],
        [0., 0., 1., 0., 0.],
        [0., 0., 0., 1., 0.]])

Initial state vector: tensor([1., 2., 3., 4., 5.])
state = shift_A @ state - tensor([0., 1., 2., 3., 4.])
state = shift_A @ state - tensor([0., 0., 1., 2., 3.])
state = shift_A @ state - tensor([0., 0., 0., 1., 2.])
state = shift_A @ state - tensor([0., 0., 0., 0., 1.])
state = shift_A @ state - tensor([0., 0., 0., 0., 0.])
```

nothing here learns, nothing decays - the state literally *holds the last $N$ inputs verbatim*. that's the "hungry" part: it eats tokens and stores them exactly. this is memory you need for recall: you can't retrieve a key you never stored precisely.

### SSM-diag - the compressor

$A$ initialized per the **S4D-Lin** recipe from my [S4 notes](/o/s4-notes/) ($A_n = -\tfrac12 + \pi i n$: decaying real part, linearly spaced oscillation frequencies):

![s4d lin init](/images/h3-notes/pasted-image-20241213165401.png)

a stable diagonal SSM; this one *aggregates* - it mixes stored pairs with decay and superposition, i.e. it does the "retrieval math" over what shift has stored.

## the attention-like layer

the H3 layer projects the input into query/key/value, runs the two SSMs in an attention-shaped pipeline:

![qkv flow through ssms](/images/h3-notes/pasted-image-20241213171312.png)

- $k$ goes through **SSM-shift** - store the keys
- then $kv = k \odot v$ goes through **SSM-diag** - let the compressed state accumulate key→value associations with decay
- finally $qkv = q \odot kv$ - use the query to read the association out

```python
class H3(torch.nn.Module):

    def __init__(self, config: H3Config):
        super().__init__()

        self.state_size = config.state_size
        self.hidden_size = config.hidden_size

        self.ssm_shift = SSM(config, init_A="shift")
        self.ssm_diag = SSM(config, init_A="s4d-lin")

        self.q_proj = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.k_proj = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.v_proj = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.out_proj = torch.nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, input, mode='recurrent'):
        q = self.q_proj(input)
        k = self.k_proj(input)
        v = self.v_proj(input)

        k = self.ssm_shift(k, mode)
        kv = k * v
        kv = self.ssm_diag(kv, mode)

        qkv = q * kv
        return self.out_proj(qkv)
```

with the config the tutorial hand-waved (adding so the above actually runs):

```python
@dataclass
class H3Config:
    state_size: int     # N - SSM state size
    hidden_size: int    # D - embedding dim
```

the generic `SSM` class backs both - the only difference from the [S4D block](/o/s4-notes/) is the initialization logic:

```python
class SSM(torch.nn.Module):
    """generic SSM to support both shift and diag models."""

    def __init__(self, config: H3Config, init_A="s4d-lin"):
        ...
        self.init_A = init_A
        if self.init_A == "s4d-lin":
            # S4D-Lin init. note: real part stored POSITIVE (+1/2) and the
            # sign is applied at forward time (-exp(log)), so the effective
            # decay is still -1/2 - same trick as the S4 note's
            # log(-A.real), just with the sign living in the forward.
            A = torch.tensor([
                0.5 + (math.pi * n * 1j) for n in range(self.state_size // 2)
            ]).repeat(self.hidden_size, 1)
            self.A_imag = torch.nn.Parameter(A.imag)
            A = torch.log(A.real)

            B = torch.randn(self.hidden_size, self.state_size // 2, dtype=torch.cfloat)
            B = torch.view_as_real(B)

            C = torch.randn(self.hidden_size, self.state_size // 2, dtype=torch.cfloat)
            C = torch.view_as_real(C)

        elif self.init_A == "shift":
            # 1s just below the diagonal. no parameters, no log trick
            A = torch.diag(torch.ones(self.state_size - 1), diagonal=-1)
            B = torch.randn(self.hidden_size, self.state_size)
            C = torch.randn(self.hidden_size, self.state_size)
        ...
```

## h3 is a transformer wearing an SSM

architecturally these models are transformer-like; the only real difference: a transformer block's **time mixing** is self-attention, an H3 model's time mixing is the H3 layer.

![h3 vs transformer block](/images/h3-notes/pasted-image-20241213171440.png)

(time mixing = move information across positions; channel mixing = the MLP that computes within a position. the [Mamba notes](/o/mamba-notes/)' "communication vs computation" framing is this same split.)

## why two SSMs is the right decomposition (my reading of the paper's motivation)

associative recall needs two things at once, and attention provides both: an exact, *unchanging* record of what keys appeared (its KV cache), and content-based retrieval (the query match). a single time-invariant SSM has to do both with one fixed operator - the same decay either blurs stored keys or forgets them, and it can't tell "store this" from "compare this". H3's move is to give the two jobs to two operators with *fixed* structure:

- **SSM-shift = the memory**: no decay, no mixing - a sliding queue of recent keys, exact.
- **SSM-diag = the retrieval compute**: decayed superposition over the stored pairs, so a product/lookup structure can emerge over time, which the outer $q \odot (\cdot)$ then probes.

that's why the layer looks like attention: it *emulates* $QK^\top$-then-$KV$ with recurrences instead of a materialized matrix. (and [Mamba](/o/mamba-notes/) later said: you don't need the shift half at all if you make *every* parameter selective - SSM-Shift is replaced by a plain 1D conv there.)

## papers

- H3: <https://arxiv.org/abs/2212.14052>

## todos

- why exactly $-\tfrac12 + \pi i n$ on the diagonal: the S4D paper derives S4D-Lin as a diagonal approximation of HiPPO-LegS - re-derive that link instead of reciting it.
- the paper's associative-recall experiments (induction heads, MQAR-style): pull one concrete result into this note - right now the motivation is all assertion.
- pin down the tutorial source (it's the same annotated-SSM series the [S4](/o/s4-notes/) code cells came from) and credit it properly.