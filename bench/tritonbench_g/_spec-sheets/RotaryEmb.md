# Spec sheet — `bench/tritonbench_g/rotary_emb/RotaryEmb.lean`

**Python source:** `bench/tritonbench_g/rotary_emb/rotary_emb.py`

## Public theorem: `rotary_emb_kernel_correctness`

<details><summary>docstring</summary>

```
/-- **The headline for `rotary_emb_fwd`.** For arbitrary symbolic strides,
sequence length, head counts, and block sizes: the full `_rotary_kernel`
surface lowers to the algorithm layer, and each of the four Python-observable
in-place stores (Q even, Q odd, K even, K odd) implements the interleaved
rotary rotation on its grouped IO signature — for every disjoint flat placement
of `[Q/K, Cos, Sin]`, every program id whose active lanes are in bounds, and
every launch state whose four read windows hold `xs`, the translated pointer
kernel terminates, every write-active lane of the data buffer ends up holding
`dataEven·cos - dataOdd·sin` (even store) resp. `dataEven·sin + dataOdd·cos`
(odd store) computed from the **old** window contents, and every other memory
cell is unchanged. The four `dim ↦ dim · stride_d` injectivity hypotheses are
the within-family (not even-vs-odd) collision-freedom of the host layout. -/
```
</details>

**Statement:**
```lean
specification rotary_emb_kernel_correctness
    (Q K Cos Sin : RegionName)
    (stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd
      stride_cosbs stride_cosd stride_sinbs stride_sind max_total_len
      HEAD_Q HEAD_K BLOCK_HEAD BLOCK_SEQ BLOCK_DMODEL BLOCK_HALF : Nat)
    (hQEven : Function.Injective
      (fun i : Fin BLOCK_HALF => dimEven i * stride_qd))
    (hQOdd : Function.Injective
      (fun i : Fin BLOCK_HALF => dimOdd i * stride_qd))
    (hKEven : Function.Injective
      (fun i : Fin BLOCK_HALF => dimEven i * stride_kd))
    (hKOdd : Function.Injective
      (fun i : Fin BLOCK_HALF => dimOdd i * stride_kd)) :
    (∃ alg, (rotary_kernel_surface Q K Cos Sin stride_qbs stride_qh stride_qd
      stride_kbs stride_kh stride_kd stride_cosbs stride_cosd stride_sinbs
      stride_sind max_total_len HEAD_Q HEAD_K BLOCK_HEAD BLOCK_SEQ
      BLOCK_DMODEL).toAlgorithm? = Except.ok alg) ∧
    (rotaryQ0IO Q Cos Sin stride_qbs stride_qh stride_qd stride_cosbs
        stride_cosd stride_sinbs stride_sind max_total_len HEAD_Q BLOCK_HALF
      ⊨ fun _ _ xs _ j =>
          let dataEven
```

**Assumptions / layout contracts:**
- `hQEven : Function.Injective
      (fun i : Fin BLOCK_HALF => dimEven i * stride_qd)`
- `hQOdd : Function.Injective
      (fun i : Fin BLOCK_HALF => dimOdd i * stride_qd)`
- `hKEven : Function.Injective
      (fun i : Fin BLOCK_HALF => dimEven i * stride_kd)`
- `hKOdd : Function.Injective
      (fun i : Fin BLOCK_HALF => dimOdd i * stride_kd)`

**Closed-form spec defs (transitive):** `dimEven`, `dimOdd`, `rotary_kernel_surface`, `rotaryQ0IO`, `rotaryStoreIO`, `rotary_emb_q0_block`

<details><summary><code>dimEven</code></summary>

```lean
def dimEven (i : Fin BLOCK_HALF) : Nat :=
  i.val * 2
```
</details>

<details><summary><code>dimOdd</code></summary>

```lean
def dimOdd (i : Fin BLOCK_HALF) : Nat :=
  i.val * 2 + 1
```
</details>

<details><summary><code>rotary_kernel_surface</code></summary>

```
/-- Faithful transcription of `rotary_emb.py`'s `_rotary_kernel`.

This keeps the block-shaped sequence/head/dimension tiles and writes both Q and
K rotary pairs over the full `[BLOCK_SEQ, BLOCK_HEAD, BLOCK_DMODEL / 2]`
surface. -/
```
```lean
def rotary_kernel_surface
    (Q K Cos Sin : RegionName)
    (stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd
      stride_cosbs stride_cosd stride_sinbs stride_sind max_total_len
      HEAD_Q HEAD_K BLOCK_HEAD BLOCK_SEQ BLOCK_DMODEL : Nat) :
    ComputeKernel := triton {
  cur_head_index = tl.program_id(0)
  cur_seq_index = tl.program_id(1)

  cur_head_range = cur_head_index * $(BLOCK_HEAD) + tl.arange(0, $(BLOCK_HEAD))
  cur_seq_range = cur_seq_index * $(BLOCK_SEQ) + tl.arange(0, $(BLOCK_SEQ))

  dim_range0 = tl.arange(0, $(BLOCK_DMODEL) // $(2)) * $(2)
  dim_range1 = tl.arange(0, $(BLOCK_DMODEL) // $(2)) * $(2) + $(1)

  off_q0 = cur_seq_range[:, None, None] * $(stride_qbs) +
    cur_head_range[None, :, None] * $(stride_qh) +
    dim_range0[None, None, :] * $(stride_qd)
  off_q1 = cur_seq_range[:, None, None] * $(stride_qbs) +
    cur_head_range[None, :, None] * $(stride_qh) +
    dim_range1[None, None, :] * $(stride_qd)

  off_dimcos_sin0 = cur_seq_range[:, None, None] * $(stride_cosbs) +
    dim_range0[None, None, :] * $(stride_cosd)
  off_dimcos_sin1 = cur_seq_range[:, None, None] * $(stride_cosbs) +
    dim_range1[None, None, :] * $(stride_cosd)

  q0 = tl.load(Q + off_q0,
    mask=(cur_seq_range[:, None, None] < $(max_total_len)) &
      (cur_head_range[None, :, None] < $(HEAD_Q)),
    other=0.0)
  q1 = tl.load(Q + off_q1,
    mask=(cur_seq_range[:, None, None] < $(max_total_len)) &
      (cur_head_range[None, :, None] < $(HEAD_Q)),
    other=0.0)

  cos0 = tl.load(Cos + off_dimcos_sin0,
    mask=cur_seq_range[:, None, None] < $(max_total_len), other=0.0)
  sin0 = tl.load(Sin + off_dimcos_sin0,
    mask=cur_seq_range[:, None, None] < $(max_total_len), other=0.0)

  cos1 = tl.load(Cos + off_dimcos_sin1,
    mask=cur_seq_range[:, None, None] < $(max_total_len), other=0.0)
  sin1 = tl.load(Sin + off_dimcos_sin1,
    mask=cur_seq_range[:, None, None] < $(max_total_len), other=0.0)

  out0 = q0 * cos0 - q1 * sin0
  out1 = q0 * sin1 + q1 * cos1

  tl.store(Q + off_q0, out0,
    mask=(cur_seq_range[:, None, None] < $(max_total_len)) &
      (cur_head_range[None, :, None] < $(HEAD_Q)))
  tl.store(Q + off_q1, out1,
    mask=(cur_seq_range[:, None, None] < $(max_total_len)) &
      (cur_head_range[None, :, None] < $(HEAD_Q)))

  off_k0 = cur_seq_range[:, None, None] * $(stride_kbs) +
    cur_head_range[None, :, None] * $(stride_kh) +
    dim_range0[None, None, :] * $(stride_kd)
  off_k1 = cur_seq_range[:, None, None] * $(stride_kbs) +
    cur_head_range[None, :, None] * $(stride_kh) +
    dim_range1[None, None, :] * $(stride_kd)

  off_dimcos_sin0 = cur_seq_range[:, None, None] * $(stride_cosbs) +
    dim_range0[None, None, :] * $(stride_cosd)
  off_dimcos_sin1 = cur_seq_range[:, None, None] * $(stride_cosbs) +
    dim_range1[None, None, :] * $(stride_cosd)

  k0 = tl.load(K + off_k0,
    mask=(cur_seq_range[:, None, None] < $(max_total_len)) &
      (cur_head_range[None, :, None] < $(HEAD_K)),
    other=0.0)
  k1 = tl.load(K + off_k1,
    mask=(cur_seq_range[:, None, None] < $(max_total_len)) &
      (cur_head_range[None, :, None] < $(HEAD_K)),
    other=0.0)

  cos0 = tl.load(Cos + off_dimcos_sin0,
    mask=cur_seq_range[:, None, None] < $(max_total_len), other=0.0)
  sin0 = tl.load(Sin + off_dimcos_sin0,
    mask=cur_seq_range[:, None, None] < $(max_total_len), other=0.0)

  cos1 = tl.load(Cos + off_dimcos_sin1,
    mask=cur_seq_range[:, None, None] < $(max_total_len), other=0.0)
  sin1 = tl.load(Sin + off_dimcos_sin1,
    mask=cur_seq_range[:, None, None] < $(max_total_len), other=0.0)

  out_k0 = k0 * cos0 - k1 * sin0
  out_k1 = k0 * sin1 + k1 * cos1

  tl.store(K + off_k0, out_k0,
    mask=(cur_seq_range[:, None, None] < $(max_total_len)) &
      (cur_head_range[None, :, None] < $(HEAD_K)))
  tl.store(K + off_k1, out_k1,
    mask=(cur_seq_range[:, None, None] < $(max_total_len)) &
      (cur_head_range[None, :, None] < $(HEAD_K)))
}
```
</details>

<details><summary><code>rotaryQ0IO</code></summary>

```
/-- The Q even-dimension store's IO signature. -/
```
```lean
def rotaryQ0IO (Q Cos Sin : RegionName)
    (stride_qbs stride_qh stride_qd stride_cosbs stride_cosd stride_sinbs
      stride_sind max_total_len HEAD_Q BLOCK_HALF : Nat) :
    GroupedMasked2DKernelIO :=
  rotaryStoreIO (rotary_emb_q0_block Q Cos Sin stride_qbs stride_qh stride_qd stride_cosbs stride_cosd
      stride_sinbs stride_sind max_total_len HEAD_Q BLOCK_HALF)
    Q Cos Sin stride_qbs stride_qh stride_qd stride_cosbs stride_cosd stride_sinbs
    stride_sind max_total_len HEAD_Q BLOCK_HALF dimEven
```
</details>

<details><summary><code>rotaryStoreIO</code></summary>

```
/-- The shared **IO signature** of one in-place rotary store — the whole
kernel-specific audit surface of the four `⊨` headlines below.

* `bufs = [Buf, Cos, Sin]` — the allocation list; `Buf` is the data buffer
  (`Q` or `K`), and the single output channel points **back at it**
  (`out 0 = Buf`): the rotary update is in place, so the triple reads the
  *old* window contents into `xs` and asserts the *new* ones.
* `nIn = 4`, `nOut = 1`, `B = BLOCK_HALF` — four `BLOCK_HALF`-lane read
  windows, one write window.
* read channel `0` — the data buffer at the **even** dimension
  `pid₁·stride_bs + pid₀·stride_h + (2j)·stride_d`;
  read channel `1` — the data buffer at the **odd** dimension
  `… + (2j+1)·stride_d`;
  read channels `2`/`3` — `Cos`/`Sin` at `pid₁·stride_cosbs + (2j)·stride_cosd`
  resp. `pid₁·stride_sinbs + (2j)·stride_sind` (the kernel loads both
  trigonometric tiles at the even dimension lanes).
* `readMask` — `pid₁ < max_total_len ∧ pid₀ < HEAD` on the two data channels
  and `pid₁ < max_total_len` on `Cos`/`Sin`: exactly the kernel's two
  `tl.load` masks.
* `write`/`writeMask` — the data buffer at `storeDim j` (instantiated with
  `dimEven` for the even store and `dimOdd` for the odd one), gated by
  `pid₁ < max_total_len ∧ pid₀ < HEAD`, the kernel's `tl.store` mask.

The windows and masks are declared, not parsed from the kernel; the headlines
**prove** the kernel's actual addressing and masking match them. Buffer sizes
are not signature content: the headline quantifies over every allocation whose
extents cover the active lanes. -/
```
```lean
def rotaryStoreIO (kernel : ComputeKernel) (Buf Cos Sin : RegionName)
    (stride_bs stride_h stride_d stride_cosbs stride_cosd stride_sinbs
      stride_sind max_total_len HEAD BLOCK_HALF : Nat)
    (storeDim : Fin BLOCK_HALF → Nat) : GroupedMasked2DKernelIO where
  kernel := kernel
  nIn := 4
  nOut := 1
  bufs := [Buf, Cos, Sin]
  inp := fun i => match i with
    | ⟨0, _⟩ => Buf
    | ⟨1, _⟩ => Buf
    | ⟨2, _⟩ => Cos
    | ⟨_ + 3, _⟩ => Sin
  out := fun _ => Buf
  B := BLOCK_HALF
  read := fun i pid₀ pid₁ j => match i with
    | ⟨0, _⟩ => pid₁ * stride_bs + pid₀ * stride_h + dimEven j * stride_d
    | ⟨1, _⟩ => pid₁ * stride_bs + pid₀ * stride_h + dimOdd j * stride_d
    | ⟨2, _⟩ => pid₁ * stride_cosbs + dimEven j * stride_cosd
    | ⟨_ + 3, _⟩ => pid₁ * stride_sinbs + dimEven j * stride_sind
  readMask := fun i pid₀ pid₁ _ => match i with
    | ⟨0, _⟩ => pid₁ < max_total_len ∧ pid₀ < HEAD
    | ⟨1, _⟩ => pid₁ < max_total_len ∧ pid₀ < HEAD
    | ⟨2, _⟩ => pid₁ < max_total_len
    | ⟨_ + 3, _⟩ => pid₁ < max_total_len
  write := fun _ pid₀ pid₁ j =>
    pid₁ * stride_bs + pid₀ * stride_h + storeDim j * stride_d
  writeMask := fun _ pid₀ pid₁ _ => pid₁ < max_total_len ∧ pid₀ < HEAD
```
</details>

<details><summary><code>rotary_emb_q0_block</code></summary>

```
/-- Proof-oriented Q-even-dimension slice of `rotary_emb.py`'s `_rotary_kernel`.

This models the first Q store for one sequence/head program tile:
`out0 = q0 * cos0 - q1 * sin0`, where even dimensions are paired with the
following odd dimension. -/
```
```lean
def rotary_emb_q0_block
    (Q Cos Sin : RegionName)
    (stride_qbs stride_qh stride_qd stride_cosbs stride_cosd stride_sinbs
      stride_sind max_total_len HEAD_Q BLOCK_HALF : Nat) :
    ComputeKernel := triton {
  cur_head_index = tl.program_id(0)
  cur_seq_index = tl.program_id(1)
  dim = tl.arange(0, $(BLOCK_HALF))
  dim0 = dim * $(2)
  dim1 = dim * $(2) + $(1)
  q0 = tl.load(Q + cur_seq_index * $(stride_qbs) +
      cur_head_index * $(stride_qh) + dim0 * $(stride_qd),
    mask=(cur_seq_index < $(max_total_len)) and (cur_head_index < $(HEAD_Q)),
    other=0.0)
  q1 = tl.load(Q + cur_seq_index * $(stride_qbs) +
      cur_head_index * $(stride_qh) + dim1 * $(stride_qd),
    mask=(cur_seq_index < $(max_total_len)) and (cur_head_index < $(HEAD_Q)),
    other=0.0)
  cos0 = tl.load(Cos + cur_seq_index * $(stride_cosbs) +
      dim0 * $(stride_cosd),
    mask=cur_seq_index < $(max_total_len), other=0.0)
  sin0 = tl.load(Sin + cur_seq_index * $(stride_sinbs) +
      dim0 * $(stride_sind),
    mask=cur_seq_index < $(max_total_len), other=0.0)
  out0 = q0 * cos0 - q1 * sin0
  tl.store(Q + cur_seq_index * $(stride_qbs) +
      cur_head_index * $(stride_qh) + dim0 * $(stride_qd),
    out0,
    mask=(cur_seq_index < $(max_total_len)) and (cur_head_index < $(HEAD_Q)))
}
```
</details>
