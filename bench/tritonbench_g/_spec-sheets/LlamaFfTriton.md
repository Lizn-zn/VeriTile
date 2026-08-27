# Spec sheet — `bench/tritonbench_g/llama_ff_triton/LlamaFfTriton.lean`

**Python source:** `bench/tritonbench_g/llama_ff_triton/llama_ff_triton.py`

## Public theorem: `ff_llama_closed_form_correct`

<details><summary>docstring</summary>

```
/-- **Closed-form correctness for `llama_ff_triton` (general statement).**

For arbitrary linear program id `pid`, tile dims `BM`/`BN`, K-block size `BK`,
and K-block count `numKBlocks` (so the contracted dimension is
`K = BK · numKBlocks`; the kernel's loads are unmasked, so this exact-multiple
presentation is required), every **active** output cell `(i, j)` of the
computed `BM × BN` tile equals

  `fp16( silu(n_i · S1[i,j]) * (n_i · S2[i,j]) )`

over ℝ — where `S1[i,j] = Σ_{t<numKBlocks} Σ_{e<BK} A[r(i), t·BK+e] ·
RMS[t·BK+e] · W1[t·BK+e, c(j)]` (similarly `S2` with `W3`),
`n_i = rsqrt((Σ_{t,e} A[r(i), t·BK+e]²)/K + EPS)`, `silu x = x · sigmoid x`,
`r(i) = (pid_m·BM + i) % M`, `c(j) = (pid_n·BN + j) % N` — the genuine
RMSNorm-fused SwiGLU closed form of the loaded `A`/`RMS`/`W1`/`W3` cells, NOT
the kernel's own executed value; inactive lanes are left untouched.

Layout: `A[i,k]` at `A + offs_am(i)·sam + k·sak`, `W[k,j]` at
`W + k·swk + offs_bn(j)·swn`, `RMS[k]` at `RMS + k·srms`, `OUT[i,j]` at
`OUT + soutm·offs_outm(i) + soutn·offs_outn(j)`, with `pid_m = pid //
cdiv(N,BN)`, `pid_n = pid % cdiv(N,BN)`. Preconditions: the row-major output
bounds `soutn = 1` / `BN ≤ soutm` (give output-offset injectivity) and clean
initial `undef`. -/
```
</details>

**Statement:**
```lean
specification ff_llama_closed_form_correct
    (A W1 W3 OUT RMS : RegionName) (s : BlockState)
    (M N sam sak sw1k sw1n sw3k sw3n soutm soutn srms BM BN BK numKBlocks : Nat) (K : Nat)
    (hK : K = BK * numKBlocks) (EPS : ℝ)
    (hcn : soutn = 1) (hbnle : BN ≤ soutm)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := ff_llama_surface A W1 W3 OUT RMS M N K sam sak sw1k sw1n sw3k sw3n
        soutm soutn srms BM BN BK numKBlocks EPS)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s M N BM BN)
        (fun idx => (OUT, outOffset s N BM BN soutm soutn idx)))
      (expected := fun idx : TileIndex [BM, BN] =>
        MemCell.of .fp16
          (FloatDType.real.cast FloatDType.fp16
            (some (ffSpec s A W1 W3 RMS M N BM BN sam sak sw1k sw1n sw3k sw3n srms
              BK numKBlocks K EPS idx.1 idx.2.1))))
```

**Assumptions / layout contracts:**
- `hK : K = BK * numKBlocks`
- `hcn : soutn = 1`
- `hbnle : BN ≤ soutm`
- `hundef : ∀ rg o, s.undef rg o = 0`

**Closed-form spec defs (transitive):** `ff_llama_surface`, `active`, `outOffset`, `ffSpec`, `rowGlobal`, `colGlobal`, `normSpec`, `accPartial`, `pidM`, `pidN`, `aElem`, `rmsElem`, `wElem`, `rowIndex`, `colIndex`

<details><summary><code>ff_llama_surface</code></summary>

```
/-- Faithful transcription of `llama_ff_triton.py`'s `ff_llama`
(`USE_FP8 = False` arm; see the Translation-surface blocker in the module
docstring for the dropped constexpr arm, the `numKBlocks` presentation, the
spelled-out store cast, and the `_i` loop counter). -/
```
```lean
def ff_llama_surface
    (A W1 W3 OUT RMS : RegionName)
    (M N K sam sak sw1k sw1n sw3k sw3n soutm soutn srms
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K numKBlocks : Nat) (EPS : ℝ) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  pid_m = pid // tl.cdiv($(N), $(BLOCK_SIZE_N))
  pid_n = pid % tl.cdiv($(N), $(BLOCK_SIZE_N))
  offs_am = (pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))) % $(M)
  offs_bn = (pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))) % $(N)
  offs_k = tl.arange(0, $(BLOCK_SIZE_K))
  a_ptrs = A + offs_am[:, None] * $(sam) + offs_k[None, :] * $(sak)
  w1_ptrs = W1 + offs_k[:, None] * $(sw1k) + offs_bn[None, :] * $(sw1n)
  w3_ptrs = W3 + offs_k[:, None] * $(sw3k) + offs_bn[None, :] * $(sw3n)
  acc1 = tl.zeros([$(BLOCK_SIZE_M), $(BLOCK_SIZE_N)], dtype=tl.float32)
  acc2 = tl.zeros([$(BLOCK_SIZE_M), $(BLOCK_SIZE_N)], dtype=tl.float32)
  rms_w_ptrs = RMS + tl.arange(0, $(BLOCK_SIZE_K))[None, :] * $(srms)
  a_sum = tl.zeros([$(BLOCK_SIZE_M), $(BLOCK_SIZE_K)], dtype=tl.float32)
  for _i in range($(0), $(numKBlocks), $(1)) {
    a = tl.load(a_ptrs)
    a_sum += tl.extra.cuda.libdevice.pow((a).to(tl.float32), $((2 : ℝ)))
    rms_w = tl.load(rms_w_ptrs)
    a = a * rms_w
    b = tl.load(w1_ptrs)
    acc1 += tl.dot(a, b)
    c = tl.load(w3_ptrs)
    acc2 += tl.dot(a, c)
    a_ptrs += $(BLOCK_SIZE_K) * $(sak)
    w1_ptrs += $(BLOCK_SIZE_K) * $(sw1k)
    w3_ptrs += $(BLOCK_SIZE_K) * $(sw3k)
    rms_w_ptrs += $(BLOCK_SIZE_K) * $(srms)
  }
  a_mean = tl.sum(a_sum, axis=1) / $(K) + $(EPS)
  a_norm = tl.math.rsqrt(a_mean)
  acc1 = acc1 * a_norm[:, None]
  acc2 = acc2 * a_norm[:, None]
  accumulator = (acc1 * tl.sigmoid(acc1)) * acc2
  offs_outm = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
  offs_outn = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
  out_ptrs = OUT + $(soutm) * offs_outm[:, None] + $(soutn) * offs_outn[None, :]
  out_mask = (offs_outm[:, None] < $(M)) & (offs_outn[None, :] < $(N))
  tl.store(out_ptrs, (accumulator).to(tl.float16), mask=out_mask)
}
```
</details>

<details><summary><code>active</code></summary>

```
/-- The boundary predicate `(row < M) & (col < N)` for tile lane `(i,j)`. -/
```
```lean
def active (s : BlockState) (M N BM BN : Nat) (idx : TileIndex [BM, BN]) : Prop :=
  rowGlobal s N BM BN idx.1 < M ∧ colGlobal s N BN idx.2.1 < N
```
</details>

<details><summary><code>outOffset</code></summary>

```
/-- The output store address for tile lane `(i,j)`:
`soutm · offs_outm i + soutn · offs_outn j` (un-wrapped global indices). -/
```
```lean
def outOffset (s : BlockState) (N BM BN soutm soutn : Nat) (idx : TileIndex [BM, BN]) : Nat :=
  soutm * rowGlobal s N BM BN idx.1 + soutn * colGlobal s N BN idx.2.1
```
</details>

<details><summary><code>ffSpec</code></summary>

```
/-- **Genuine RMSNorm-fused SwiGLU spec** (over ℝ):
`OUT[i,j] = silu(n_i · S1[i,j]) * (n_i · S2[i,j])` with `S1`/`S2` the full-loop
gated-GEMM sums against `W1`/`W3` and `n_i` the RMS factor. -/
```
```lean
noncomputable def ffSpec (s : BlockState) (A W1 W3 RMS : RegionName)
    (M N BM BN sam sak sw1k sw1n sw3k sw3n srms BK numKBlocks K : Nat) (EPS : ℝ)
    (i : Fin BM) (j : Fin BN) : ℝ :=
  TiledActivation.silu
      (normSpec s A M N BM BN sam sak BK numKBlocks K EPS i
        * accPartial s A W1 RMS M N BM BN sam sak sw1k sw1n srms BK i j numKBlocks)
    * (normSpec s A M N BM BN sam sak BK numKBlocks K EPS i
        * accPartial s A W3 RMS M N BM BN sam sak sw3k sw3n srms BK i j numKBlocks)
```
</details>

<details><summary><code>rowGlobal</code></summary>

```
/-- Global output row of tile lane `i`: `pid_m · BLOCK_M + i`, **before** the
`% M` wrap (the kernel's `offs_outm`). -/
```
```lean
def rowGlobal (s : BlockState) (N BM BN : Nat) (i : Fin BM) : Nat :=
  pidM (s.pids 0) N BN * BM + i.val
```
</details>

<details><summary><code>colGlobal</code></summary>

```
/-- Global output column of tile lane `j`: `pid_n · BLOCK_N + j`, before wrap. -/
```
```lean
def colGlobal (s : BlockState) (N BN : Nat) (j : Fin BN) : Nat :=
  pidN (s.pids 0) N BN * BN + j.val
```
</details>

<details><summary><code>normSpec</code></summary>

```
/-- **The RMS normalization factor** (over ℝ):
`n_i = rsqrt( (Σ_{t<numKBlocks} Σ_{e<BK} A[r(i), t·BK+e]²) / K + EPS )`. -/
```
```lean
noncomputable def normSpec (s : BlockState) (A : RegionName)
    (M N BM BN sam sak BK numKBlocks K : Nat) (EPS : ℝ) (i : Fin BM) : ℝ :=
  1 / Real.sqrt
    ((∑ t : Fin numKBlocks, ∑ e : Fin BK,
        aElem s A M N BM BN sam sak i (t.val * BK + e.val) ^ 2) / (K : ℝ) + EPS)
```
</details>

<details><summary><code>accPartial</code></summary>

```
/-- Partial gated-GEMM accumulator after `c` K-blocks (shared by `acc1`/`acc2`):
`Σ_{t<c} Σ_{e<BK} A[r(i), t·BK+e] · RMS[t·BK+e] · W[t·BK+e, c(j)]`. -/
```
```lean
noncomputable def accPartial (s : BlockState) (A W RMS : RegionName)
    (M N BM BN sam sak swk swn srms BK : Nat) (i : Fin BM) (j : Fin BN) (c : Nat) : ℝ :=
  ∑ t : Fin c, ∑ e : Fin BK,
    aElem s A M N BM BN sam sak i (t.val * BK + e.val)
      * rmsElem s RMS srms (t.val * BK + e.val)
      * wElem s W N BN swk swn j (t.val * BK + e.val)
```
</details>

<details><summary><code>pidM</code></summary>

```
/-- The kernel's `pid_m` derivation: `pid // cdiv(N, BLOCK_N)`. -/
```
```lean
def pidM (pid N BN : Nat) : Nat := pid / cdiv N BN
```
</details>

<details><summary><code>pidN</code></summary>

```
/-- The kernel's `pid_n` derivation: `pid % cdiv(N, BLOCK_N)`. -/
```
```lean
def pidN (pid N BN : Nat) : Nat := pid % cdiv N BN
```
</details>

<details><summary><code>aElem</code></summary>

```
/-- `A[i, k] = readMem A (offs_am i · sam + k · sak)`. -/
```
```lean
noncomputable def aElem (s : BlockState) (A : RegionName) (M N BM BN sam sak : Nat)
    (i : Fin BM) (k : Nat) : ℝ :=
  s.readMem A (rowIndex s M N BM BN i * sam + k * sak)
```
</details>

<details><summary><code>rmsElem</code></summary>

```
/-- `RMS[k] = readMem RMS (k · srms)`. -/
```
```lean
noncomputable def rmsElem (s : BlockState) (RMS : RegionName) (srms : Nat) (k : Nat) : ℝ :=
  s.readMem RMS (k * srms)
```
</details>

<details><summary><code>wElem</code></summary>

```
/-- `W[k, j] = readMem W (k · swk + offs_bn j · swn)` (shared by `W1`/`W3`). -/
```
```lean
noncomputable def wElem (s : BlockState) (W : RegionName) (N BN swk swn : Nat)
    (j : Fin BN) (k : Nat) : ℝ :=
  s.readMem W (k * swk + colIndex s N BN j * swn)
```
</details>

<details><summary><code>rowIndex</code></summary>

```
/-- The `% M`-wrapped A-row index of tile lane `i` (the kernel's `offs_am`). -/
```
```lean
def rowIndex (s : BlockState) (M N BM BN : Nat) (i : Fin BM) : Nat :=
  rowGlobal s N BM BN i % M
```
</details>

<details><summary><code>colIndex</code></summary>

```
/-- The `% N`-wrapped W-column index of tile lane `j` (the kernel's `offs_bn`). -/
```
```lean
def colIndex (s : BlockState) (N BN : Nat) (j : Fin BN) : Nat :=
  colGlobal s N BN j % N
```
</details>
