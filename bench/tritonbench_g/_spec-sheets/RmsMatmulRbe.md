# Spec sheet — `bench/tritonbench_g/rms_matmul_rbe/RmsMatmulRbe.lean`

**Python source:** `bench/tritonbench_g/rms_matmul_rbe/rms_matmul_rbe.py`

## Public theorem: `rms_matmul_rbe_closed_form_correct`

<details><summary>docstring</summary>

```
/-- **Closed-form correctness for `rms_matmul_rbe` (general statement).**

For arbitrary batch id (`s.pids 0`), linear tile pid (`s.pids 1`), tile dims
`BM`/`BN`, K-block size `BK`, and K-block count `numKBlocks` (so the
contracted dimension is `K = BK · numKBlocks`; the kernel's loads are
unmasked, so this exact-multiple presentation is required), every **active**
output cell `(i, j)` of the computed `BM × BN` tile equals

  `fp16( n_i · S[i,j] )`

over ℝ — where `S[i,j] = Σ_{t<numKBlocks} Σ_{e<BK} X[b, r(i), t·BK+e] ·
RMS[t·BK+e] · W[t·BK+e, c(j)]`,
`n_i = rsqrt((Σ_{t,e} X[b, r(i), t·BK+e]²)/K + EPS)`,
`r(i) = (pid_m·BM + i) % M`, `c(j) = (pid_n·BN + j) % N`, and every X read
carries the batch offset `b·sxb` (b = `pid_batch = s.pids 0`) — the genuine
batched RMSNorm-fused GEMM closed form of the loaded `X`/`RMS`/`W` cells, NOT
the kernel's own executed value; inactive lanes are left untouched.

Layout: `X[b,i,k]` at `X + b·sxb + offs_m(i)·sxm + k·sxk`, `W[k,j]` at
`W + k·swk + offs_n(j)·swn`, `RMS[k]` at `RMS + k·srms`, `OUT[b,i,j]` at
`OUT + b·sob + offs_m(i)·som + offs_n(j)·son`, with `pid_m = pid //
cdiv(N,BN)`, `pid_n = pid % cdiv(N,BN)` over `pid = s.pids 1`. Preconditions:
the row-major output bounds `son = 1` / `BN ≤ som` (give output-offset
injectivity; the batch term is a per-program constant) and clean initial
`undef`. -/
```
</details>

**Statement:**
```lean
specification rms_matmul_rbe_closed_form_correct
    (X W RMS OUT : RegionName) (s : BlockState)
    (M N sxb sxm sxk swk swn srms sob som son BM BN BK numKBlocks : Nat) (K : Nat)
    (hK : K = BK * numKBlocks) (EPS : ℝ)
    (hcn : son = 1) (hbnle : BN ≤ som)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := rms_matmul_rbe_surface X W RMS OUT M N K sxb sxm sxk swk swn srms
        sob som son BM BN BK numKBlocks EPS)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s M N BM BN)
        (fun idx => (OUT, outOffset s N BM BN sob som son idx)))
      (expected := fun idx : TileIndex [BM, BN] =>
        MemCell.of .fp16
          (FloatDType.real.cast FloatDType.fp16
            (some (rmsSpec s X W RMS M N BM BN sxb sxm sxk swk swn srms
              BK numKBlocks K EPS idx.1 idx.2.1))))
```

**Assumptions / layout contracts:**
- `hK : K = BK * numKBlocks`
- `hcn : son = 1`
- `hbnle : BN ≤ som`
- `hundef : ∀ rg o, s.undef rg o = 0`

**Closed-form spec defs (transitive):** `rms_matmul_rbe_surface`, `active`, `outOffset`, `rmsSpec`, `rowGlobal`, `colGlobal`, `normSpec`, `accPartial`, `pidM`, `pidN`, `xElem`, `rmsElem`, `wElem`, `rowIndex`, `colIndex`

<details><summary><code>rms_matmul_rbe_surface</code></summary>

```
/-- Faithful transcription of `rms_matmul_rbe.py`'s FIRST kernel
`rms_matmul_rbe` (`USE_FP8 = False` arm; see the Translation-surface blocker
in the module docstring for the dropped constexpr arm, the dropped unused
parameters, the `numKBlocks` presentation, the spelled-out store cast, and
the `_i` loop counter). Byte-identical to the `rms_rbe_matmul` port's target
JIT. -/
```
```lean
def rms_matmul_rbe_surface
    (X W RMS OUT : RegionName)
    (M N K sxb sxm sxk swk swn srms sob som son
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K numKBlocks : Nat) (EPS : ℝ) :
    ComputeKernel := triton {
  pid_batch = tl.program_id(axis=0)
  pid = tl.program_id(axis=1)
  pid_m = pid // tl.cdiv($(N), $(BLOCK_SIZE_N))
  pid_n = pid % tl.cdiv($(N), $(BLOCK_SIZE_N))
  offs_m = (pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))) % $(M)
  offs_n = (pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))) % $(N)
  offs_k = tl.arange(0, $(BLOCK_SIZE_K))
  x_ptrs = X + pid_batch * $(sxb) + offs_m[:, None] * $(sxm) + offs_k[None, :] * $(sxk)
  w_ptrs = W + offs_k[:, None] * $(swk) + offs_n[None, :] * $(swn)
  accumulator = tl.zeros([$(BLOCK_SIZE_M), $(BLOCK_SIZE_N)], dtype=tl.float32)
  rms_w_ptrs = RMS + tl.arange(0, $(BLOCK_SIZE_K))[None, :] * $(srms)
  x_sum = tl.zeros([$(BLOCK_SIZE_M), $(BLOCK_SIZE_K)], dtype=tl.float32)
  for _i in range($(0), $(numKBlocks), $(1)) {
    x = tl.load(x_ptrs)
    x_sum += tl.extra.cuda.libdevice.pow((x).to(tl.float32), $((2 : ℝ)))
    rms_w = tl.load(rms_w_ptrs)
    x = x * rms_w
    w = tl.load(w_ptrs)
    accumulator += tl.dot(x, w)
    x_ptrs += $(BLOCK_SIZE_K) * $(sxk)
    w_ptrs += $(BLOCK_SIZE_K) * $(swk)
    rms_w_ptrs += $(BLOCK_SIZE_K) * $(srms)
  }
  x_mean = tl.sum(x_sum, axis=1) / $(K) + $(EPS)
  x_norm = tl.math.rsqrt(x_mean)
  accumulator = accumulator * x_norm[:, None]
  offs_m = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
  offs_n = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
  out_ptrs = OUT + pid_batch * $(sob) + offs_m[:, None] * $(som) + offs_n[None, :] * $(son)
  out_mask = (offs_m[:, None] < $(M)) & (offs_n[None, :] < $(N))
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
`pid_batch · sob + offs_m i · som + offs_n j · son` (un-wrapped global indices). -/
```
```lean
def outOffset (s : BlockState) (N BM BN sob som son : Nat) (idx : TileIndex [BM, BN]) : Nat :=
  s.pids 0 * sob + rowGlobal s N BM BN idx.1 * som + colGlobal s N BN idx.2.1 * son
```
</details>

<details><summary><code>rmsSpec</code></summary>

```
/-- **Genuine batched RMSNorm-fused GEMM spec** (over ℝ):
`OUT[b,i,j] = n_i · S[i,j]` with `S` the full-loop RMS-scaled GEMM sum and
`n_i` the RMS factor. -/
```
```lean
noncomputable def rmsSpec (s : BlockState) (X W RMS : RegionName)
    (M N BM BN sxb sxm sxk swk swn srms BK numKBlocks K : Nat) (EPS : ℝ)
    (i : Fin BM) (j : Fin BN) : ℝ :=
  normSpec s X M N BM BN sxb sxm sxk BK numKBlocks K EPS i
    * accPartial s X W RMS M N BM BN sxb sxm sxk swk swn srms BK i j numKBlocks
```
</details>

<details><summary><code>rowGlobal</code></summary>

```
/-- Global output row of tile lane `i`: `pid_m · BLOCK_M + i`, **before** the
`% M` wrap (the kernel's tail `offs_m`). The tile pid is `s.pids 1` (axis 1;
axis 0 is the batch). -/
```
```lean
def rowGlobal (s : BlockState) (N BM BN : Nat) (i : Fin BM) : Nat :=
  pidM (s.pids 1) N BN * BM + i.val
```
</details>

<details><summary><code>colGlobal</code></summary>

```
/-- Global output column of tile lane `j`: `pid_n · BLOCK_N + j`, before wrap. -/
```
```lean
def colGlobal (s : BlockState) (N BN : Nat) (j : Fin BN) : Nat :=
  pidN (s.pids 1) N BN * BN + j.val
```
</details>

<details><summary><code>normSpec</code></summary>

```
/-- **The RMS normalization factor** (over ℝ):
`n_i = rsqrt( (Σ_{t<numKBlocks} Σ_{e<BK} X[b, r(i), t·BK+e]²) / K + EPS )`. -/
```
```lean
noncomputable def normSpec (s : BlockState) (X : RegionName)
    (M N BM BN sxb sxm sxk BK numKBlocks K : Nat) (EPS : ℝ) (i : Fin BM) : ℝ :=
  1 / Real.sqrt
    ((∑ t : Fin numKBlocks, ∑ e : Fin BK,
        xElem s X M N BM BN sxb sxm sxk i (t.val * BK + e.val) ^ 2) / (K : ℝ) + EPS)
```
</details>

<details><summary><code>accPartial</code></summary>

```
/-- Partial RMS-scaled GEMM accumulator after `c` K-blocks:
`Σ_{t<c} Σ_{e<BK} X[b, r(i), t·BK+e] · RMS[t·BK+e] · W[t·BK+e, c(j)]`. -/
```
```lean
noncomputable def accPartial (s : BlockState) (X W RMS : RegionName)
    (M N BM BN sxb sxm sxk swk swn srms BK : Nat) (i : Fin BM) (j : Fin BN) (c : Nat) : ℝ :=
  ∑ t : Fin c, ∑ e : Fin BK,
    xElem s X M N BM BN sxb sxm sxk i (t.val * BK + e.val)
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

<details><summary><code>xElem</code></summary>

```
/-- `X[b, i, k] = readMem X (pid_batch · sxb + offs_m i · sxm + k · sxk)`
(the batch offset `s.pids 0 · sxb` is in every X read). -/
```
```lean
noncomputable def xElem (s : BlockState) (X : RegionName) (M N BM BN sxb sxm sxk : Nat)
    (i : Fin BM) (k : Nat) : ℝ :=
  s.readMem X (s.pids 0 * sxb + rowIndex s M N BM BN i * sxm + k * sxk)
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
/-- `W[k, j] = readMem W (k · swk + offs_n j · swn)`. -/
```
```lean
noncomputable def wElem (s : BlockState) (W : RegionName) (N BN swk swn : Nat)
    (j : Fin BN) (k : Nat) : ℝ :=
  s.readMem W (k * swk + colIndex s N BN j * swn)
```
</details>

<details><summary><code>rowIndex</code></summary>

```
/-- The `% M`-wrapped X-row index of tile lane `i` (the kernel's loop `offs_m`). -/
```
```lean
def rowIndex (s : BlockState) (M N BM BN : Nat) (i : Fin BM) : Nat :=
  rowGlobal s N BM BN i % M
```
</details>

<details><summary><code>colIndex</code></summary>

```
/-- The `% N`-wrapped W-column index of tile lane `j` (the kernel's loop `offs_n`). -/
```
```lean
def colIndex (s : BlockState) (N BN : Nat) (j : Fin BN) : Nat :=
  colGlobal s N BN j % N
```
</details>

## Public theorem: `rms_matmul_rbe_qkv_closed_form_correct`

<details><summary>docstring</summary>

```
/-- **Closed-form correctness for `rms_matmul_rbe_qkv` (general statement).**

The qkv kernel is the callee `rms_matmul_rbe` inlined three times — at
`(q_weight, q)`, `(k_weight, k)`, `(v_weight, v)` — all sharing `X`, `RMS`,
and the launch geometry. The conclusion is the standard multi-output bundle:
a conjunction of three `Realizes_without_Rounding`, one per stored output.
For arbitrary batch id (`s.pids 0`) and linear tile pid (`s.pids 1`), every
**active** output cell `(i, j)` of the computed `BM × BN` tile of each
output equals

  `fp16( n_i · S_w[i,j] )`

over ℝ — where `S_w[i,j] = Σ_{t<numKBlocks} Σ_{e<BK} X[b, r(i), t·BK+e] ·
RMS[t·BK+e] · W[t·BK+e, c(j)]` against the pass's OWN weight matrix
(`W = QW`/`KW`/`VW` with its own strides) and
`n_i = rsqrt((Σ_{t,e} X[b, r(i), t·BK+e]²)/K + EPS)` is the SHARED RMS
normalizer; `r(i) = (pid_m·BM + i) % M`, `c(j) = (pid_n·BN + j) % N`, batch
offset `b·sxb` in every X read — the genuine closed form of the loaded
`X`/`RMS`/weight cells, NOT the kernel's own executed value; inactive lanes
impose no obligation.

Preconditions: `K = BK · numKBlocks` (unmasked loads); per-output row-major
bounds `sqn = 1`/`BN ≤ sqm`, `skn = 1`/`BN ≤ skm`, `svn = 1`/`BN ≤ svm`
(output-offset injectivity); pairwise distinctness of the written regions
`Q`/`KOut`/`V` from each other and from the read regions each later pass
consumes (`X`, `RMS`, and the pass's weight region — the host passes eight
distinct buffers), which is exactly what the cross-pass frame transport
needs; and clean initial `undef`. -/
```
</details>

**Statement:**
```lean
specification rms_matmul_rbe_qkv_closed_form_correct
    (X QW KW VW RMS Q KOut V : RegionName) (s : BlockState)
    (M N sxb sxm sxk sqwk sqwn skwk skwn svwk svwn srms
      sqb sqm sqn skb skm skn svb svm svn BM BN BK numKBlocks : Nat) (K : Nat)
    (hK : K = BK * numKBlocks) (EPS : ℝ)
    (hqn : sqn = 1) (hqm : BN ≤ sqm)
    (hkn : skn = 1) (hkm : BN ≤ skm)
    (hvn : svn = 1) (hvm : BN ≤ svm)
    (hXQ : X ≠ Q) (hRMSQ : RMS ≠ Q) (hKWQ : KW ≠ Q) (hVWQ : VW ≠ Q)
    (hXK : X ≠ KOut) (hRMSK : RMS ≠ KOut) (hVWK : VW ≠ KOut)
    (hQK : Q ≠ KOut) (hQV : Q ≠ V) (hKV : KOut ≠ V)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := rms_matmul_rbe_qkv_surface X QW KW VW RMS Q KOut V M N K sxb sxm sxk
        sqwk sqwn skwk skwn svwk svwn srms sqb sqm sqn skb skm skn svb svm svn
        BM BN BK numKBlocks EPS)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s M N BM BN)
        (fun idx => (Q, outOffset s N BM BN sqb sqm sqn idx)))
      (expected := fun idx : TileIndex [BM, BN] =>
        MemCell.of .fp16
          (FloatDType.real.cast FloatDType.fp16
            (some (rmsSpec s X QW RMS M N BM BN sxb sxm sxk sqwk sqwn srms
              BK numKBlocks K EPS idx.1 idx.2.1))))
    ∧ ComputeCorrect.Realizes_without_Rounding
        (kernel := rms_matmul_rbe_qkv_surface X QW KW VW RMS Q KOut V M N K sxb sxm sxk
          sqwk sqwn skwk skwn svwk svwn srms sqb sqm sqn skb skm skn svb svm svn
          BM BN BK numKBlocks EPS)
        (initialState := s)
        (write := ComputeCorrect.WriteMap.writeIf
          (active s M N BM BN)
          (fun idx => (KOut, outOffset s N BM BN skb skm skn idx)))
        (expected := fun idx : TileIndex [BM, BN] =>
          MemCell.of .fp16
            (FloatDType.real.cast FloatDType.fp16
              (some (rmsSpec s X KW RMS M N BM BN sxb sxm sxk skwk skwn srms
                BK numKBlocks K EPS idx.1 idx.2.1))))
    ∧ ComputeCorrect.Realizes_without_Rounding
        (kernel := rms_matmul_rbe_qkv_surface X QW KW VW RMS Q KOut V M N K sxb sxm sxk
          sqwk sqwn skwk skwn svwk svwn srms sqb sqm sqn skb skm skn svb svm svn
          BM BN BK numKBlocks EPS)
        (initialState := s)
        (write := ComputeCorrect.WriteMap.writeIf
          (active s M N BM BN)
          (fun idx => (V, outOffset s N BM BN svb svm svn idx)))
        (expected := fun idx : TileIndex [BM, BN] =>
          MemCell.of .fp16
            (FloatDType.real.cast FloatDType.fp16
              (some (rmsSpec s X VW RMS M N BM BN sxb sxm sxk svwk svwn srms
                BK numKBlocks K EPS idx.1 idx.2.1))))
```

**Assumptions / layout contracts:**
- `hK : K = BK * numKBlocks`
- `hqn : sqn = 1`
- `hqm : BN ≤ sqm`
- `hkn : skn = 1`
- `hkm : BN ≤ skm`
- `hvn : svn = 1`
- `hvm : BN ≤ svm`
- `hXQ : X ≠ Q`
- `hRMSQ : RMS ≠ Q`
- `hKWQ : KW ≠ Q`
- `hVWQ : VW ≠ Q`
- `hXK : X ≠ KOut`
- `hRMSK : RMS ≠ KOut`
- `hVWK : VW ≠ KOut`
- `hQK : Q ≠ KOut`
- `hQV : Q ≠ V`
- `hKV : KOut ≠ V`
- `hundef : ∀ rg o, s.undef rg o = 0`

**Closed-form spec defs (transitive):** `rms_matmul_rbe_qkv_surface`, `active`, `outOffset`, `rmsSpec`, `rowGlobal`, `colGlobal`, `normSpec`, `accPartial`, `pidM`, `pidN`, `xElem`, `rmsElem`, `wElem`, `rowIndex`, `colIndex`

<details><summary><code>rms_matmul_rbe_qkv_surface</code></summary>

```
/-- Faithful transcription of `rms_matmul_rbe_qkv` (the file's SECOND kernel
and the only one launched by a host function): the callee body of
`rms_matmul_rbe` inlined three times — pass 1 at `(QW, Q)` with strides
`sqwk`/`sqwn` and `sqb`/`sqm`/`sqn`, pass 2 at `(KW, KOut)` with
`skwk`/`skwn` and `skb`/`skm`/`skn`, pass 3 at `(VW, V)` with `svwk`/`svwn`
and `svb`/`svm`/`svn`; `X`/`RMS` and the X strides `sxb`/`sxm`/`sxk` and
`srms` are shared. See the Translation-surface blocker for the cross-JIT
inline, the dropped `USE_FP8` arm, and the vanished unused parameters. -/
```
```lean
def rms_matmul_rbe_qkv_surface
    (X QW KW VW RMS Q KOut V : RegionName)
    (M N K sxb sxm sxk sqwk sqwn skwk skwn svwk svwn srms
      sqb sqm sqn skb skm skn svb svm svn BM BN BK numKBlocks : Nat) (EPS : ℝ) :
    ComputeKernel := triton {
  pid_batch = tl.program_id(axis=0)
  pid = tl.program_id(axis=1)
  pid_m = pid // tl.cdiv($(N), $(BN))
  pid_n = pid % tl.cdiv($(N), $(BN))
  offs_m = (pid_m * $(BM) + tl.arange(0, $(BM))) % $(M)
  offs_n = (pid_n * $(BN) + tl.arange(0, $(BN))) % $(N)
  offs_k = tl.arange(0, $(BK))
  x_ptrs = X + pid_batch * $(sxb) + offs_m[:, None] * $(sxm) + offs_k[None, :] * $(sxk)
  w_ptrs = QW + offs_k[:, None] * $(sqwk) + offs_n[None, :] * $(sqwn)
  accumulator = tl.zeros([$(BM), $(BN)], dtype=tl.float32)
  rms_w_ptrs = RMS + tl.arange(0, $(BK))[None, :] * $(srms)
  x_sum = tl.zeros([$(BM), $(BK)], dtype=tl.float32)
  for _i in range($(0), $(numKBlocks), $(1)) {
    x = tl.load(x_ptrs)
    x_sum += tl.extra.cuda.libdevice.pow((x).to(tl.float32), $((2 : ℝ)))
    rms_w = tl.load(rms_w_ptrs)
    x = x * rms_w
    w = tl.load(w_ptrs)
    accumulator += tl.dot(x, w)
    x_ptrs += $(BK) * $(sxk)
    w_ptrs += $(BK) * $(sqwk)
    rms_w_ptrs += $(BK) * $(srms)
  }
  x_mean = tl.sum(x_sum, axis=1) / $(K) + $(EPS)
  x_norm = tl.math.rsqrt(x_mean)
  accumulator = accumulator * x_norm[:, None]
  offs_m = pid_m * $(BM) + tl.arange(0, $(BM))
  offs_n = pid_n * $(BN) + tl.arange(0, $(BN))
  out_ptrs = Q + pid_batch * $(sqb) + offs_m[:, None] * $(sqm) + offs_n[None, :] * $(sqn)
  out_mask = (offs_m[:, None] < $(M)) & (offs_n[None, :] < $(N))
  tl.store(out_ptrs, (accumulator).to(tl.float16), mask=out_mask)
  pid_batch = tl.program_id(axis=0)
  pid = tl.program_id(axis=1)
  pid_m = pid // tl.cdiv($(N), $(BN))
  pid_n = pid % tl.cdiv($(N), $(BN))
  offs_m = (pid_m * $(BM) + tl.arange(0, $(BM))) % $(M)
  offs_n = (pid_n * $(BN) + tl.arange(0, $(BN))) % $(N)
  offs_k = tl.arange(0, $(BK))
  x_ptrs = X + pid_batch * $(sxb) + offs_m[:, None] * $(sxm) + offs_k[None, :] * $(sxk)
  w_ptrs = KW + offs_k[:, None] * $(skwk) + offs_n[None, :] * $(skwn)
  accumulator = tl.zeros([$(BM), $(BN)], dtype=tl.float32)
  rms_w_ptrs = RMS + tl.arange(0, $(BK))[None, :] * $(srms)
  x_sum = tl.zeros([$(BM), $(BK)], dtype=tl.float32)
  for _i in range($(0), $(numKBlocks), $(1)) {
    x = tl.load(x_ptrs)
    x_sum += tl.extra.cuda.libdevice.pow((x).to(tl.float32), $((2 : ℝ)))
    rms_w = tl.load(rms_w_ptrs)
    x = x * rms_w
    w = tl.load(w_ptrs)
    accumulator += tl.dot(x, w)
    x_ptrs += $(BK) * $(sxk)
    w_ptrs += $(BK) * $(skwk)
    rms_w_ptrs += $(BK) * $(srms)
  }
  x_mean = tl.sum(x_sum, axis=1) / $(K) + $(EPS)
  x_norm = tl.math.rsqrt(x_mean)
  accumulator = accumulator * x_norm[:, None]
  offs_m = pid_m * $(BM) + tl.arange(0, $(BM))
  offs_n = pid_n * $(BN) + tl.arange(0, $(BN))
  out_ptrs = KOut + pid_batch * $(skb) + offs_m[:, None] * $(skm) + offs_n[None, :] * $(skn)
  out_mask = (offs_m[:, None] < $(M)) & (offs_n[None, :] < $(N))
  tl.store(out_ptrs, (accumulator).to(tl.float16), mask=out_mask)
  pid_batch = tl.program_id(axis=0)
  pid = tl.program_id(axis=1)
  pid_m = pid // tl.cdiv($(N), $(BN))
  pid_n = pid % tl.cdiv($(N), $(BN))
  offs_m = (pid_m * $(BM) + tl.arange(0, $(BM))) % $(M)
  offs_n = (pid_n * $(BN) + tl.arange(0, $(BN))) % $(N)
  offs_k = tl.arange(0, $(BK))
  x_ptrs = X + pid_batch * $(sxb) + offs_m[:, None] * $(sxm) + offs_k[None, :] * $(sxk)
  w_ptrs = VW + offs_k[:, None] * $(svwk) + offs_n[None, :] * $(svwn)
  accumulator = tl.zeros([$(BM), $(BN)], dtype=tl.float32)
  rms_w_ptrs = RMS + tl.arange(0, $(BK))[None, :] * $(srms)
  x_sum = tl.zeros([$(BM), $(BK)], dtype=tl.float32)
  for _i in range($(0), $(numKBlocks), $(1)) {
    x = tl.load(x_ptrs)
    x_sum += tl.extra.cuda.libdevice.pow((x).to(tl.float32), $((2 : ℝ)))
    rms_w = tl.load(rms_w_ptrs)
    x = x * rms_w
    w = tl.load(w_ptrs)
    accumulator += tl.dot(x, w)
    x_ptrs += $(BK) * $(sxk)
    w_ptrs += $(BK) * $(svwk)
    rms_w_ptrs += $(BK) * $(srms)
  }
  x_mean = tl.sum(x_sum, axis=1) / $(K) + $(EPS)
  x_norm = tl.math.rsqrt(x_mean)
  accumulator = accumulator * x_norm[:, None]
  offs_m = pid_m * $(BM) + tl.arange(0, $(BM))
  offs_n = pid_n * $(BN) + tl.arange(0, $(BN))
  out_ptrs = V + pid_batch * $(svb) + offs_m[:, None] * $(svm) + offs_n[None, :] * $(svn)
  out_mask = (offs_m[:, None] < $(M)) & (offs_n[None, :] < $(N))
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
`pid_batch · sob + offs_m i · som + offs_n j · son` (un-wrapped global indices). -/
```
```lean
def outOffset (s : BlockState) (N BM BN sob som son : Nat) (idx : TileIndex [BM, BN]) : Nat :=
  s.pids 0 * sob + rowGlobal s N BM BN idx.1 * som + colGlobal s N BN idx.2.1 * son
```
</details>

<details><summary><code>rmsSpec</code></summary>

```
/-- **Genuine batched RMSNorm-fused GEMM spec** (over ℝ):
`OUT[b,i,j] = n_i · S[i,j]` with `S` the full-loop RMS-scaled GEMM sum and
`n_i` the RMS factor. -/
```
```lean
noncomputable def rmsSpec (s : BlockState) (X W RMS : RegionName)
    (M N BM BN sxb sxm sxk swk swn srms BK numKBlocks K : Nat) (EPS : ℝ)
    (i : Fin BM) (j : Fin BN) : ℝ :=
  normSpec s X M N BM BN sxb sxm sxk BK numKBlocks K EPS i
    * accPartial s X W RMS M N BM BN sxb sxm sxk swk swn srms BK i j numKBlocks
```
</details>

<details><summary><code>rowGlobal</code></summary>

```
/-- Global output row of tile lane `i`: `pid_m · BLOCK_M + i`, **before** the
`% M` wrap (the kernel's tail `offs_m`). The tile pid is `s.pids 1` (axis 1;
axis 0 is the batch). -/
```
```lean
def rowGlobal (s : BlockState) (N BM BN : Nat) (i : Fin BM) : Nat :=
  pidM (s.pids 1) N BN * BM + i.val
```
</details>

<details><summary><code>colGlobal</code></summary>

```
/-- Global output column of tile lane `j`: `pid_n · BLOCK_N + j`, before wrap. -/
```
```lean
def colGlobal (s : BlockState) (N BN : Nat) (j : Fin BN) : Nat :=
  pidN (s.pids 1) N BN * BN + j.val
```
</details>

<details><summary><code>normSpec</code></summary>

```
/-- **The RMS normalization factor** (over ℝ):
`n_i = rsqrt( (Σ_{t<numKBlocks} Σ_{e<BK} X[b, r(i), t·BK+e]²) / K + EPS )`. -/
```
```lean
noncomputable def normSpec (s : BlockState) (X : RegionName)
    (M N BM BN sxb sxm sxk BK numKBlocks K : Nat) (EPS : ℝ) (i : Fin BM) : ℝ :=
  1 / Real.sqrt
    ((∑ t : Fin numKBlocks, ∑ e : Fin BK,
        xElem s X M N BM BN sxb sxm sxk i (t.val * BK + e.val) ^ 2) / (K : ℝ) + EPS)
```
</details>

<details><summary><code>accPartial</code></summary>

```
/-- Partial RMS-scaled GEMM accumulator after `c` K-blocks:
`Σ_{t<c} Σ_{e<BK} X[b, r(i), t·BK+e] · RMS[t·BK+e] · W[t·BK+e, c(j)]`. -/
```
```lean
noncomputable def accPartial (s : BlockState) (X W RMS : RegionName)
    (M N BM BN sxb sxm sxk swk swn srms BK : Nat) (i : Fin BM) (j : Fin BN) (c : Nat) : ℝ :=
  ∑ t : Fin c, ∑ e : Fin BK,
    xElem s X M N BM BN sxb sxm sxk i (t.val * BK + e.val)
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

<details><summary><code>xElem</code></summary>

```
/-- `X[b, i, k] = readMem X (pid_batch · sxb + offs_m i · sxm + k · sxk)`
(the batch offset `s.pids 0 · sxb` is in every X read). -/
```
```lean
noncomputable def xElem (s : BlockState) (X : RegionName) (M N BM BN sxb sxm sxk : Nat)
    (i : Fin BM) (k : Nat) : ℝ :=
  s.readMem X (s.pids 0 * sxb + rowIndex s M N BM BN i * sxm + k * sxk)
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
/-- `W[k, j] = readMem W (k · swk + offs_n j · swn)`. -/
```
```lean
noncomputable def wElem (s : BlockState) (W : RegionName) (N BN swk swn : Nat)
    (j : Fin BN) (k : Nat) : ℝ :=
  s.readMem W (k * swk + colIndex s N BN j * swn)
```
</details>

<details><summary><code>rowIndex</code></summary>

```
/-- The `% M`-wrapped X-row index of tile lane `i` (the kernel's loop `offs_m`). -/
```
```lean
def rowIndex (s : BlockState) (M N BM BN : Nat) (i : Fin BM) : Nat :=
  rowGlobal s N BM BN i % M
```
</details>

<details><summary><code>colIndex</code></summary>

```
/-- The `% N`-wrapped W-column index of tile lane `j` (the kernel's loop `offs_n`). -/
```
```lean
def colIndex (s : BlockState) (N BN : Nat) (j : Fin BN) : Nat :=
  colGlobal s N BN j % N
```
</details>
