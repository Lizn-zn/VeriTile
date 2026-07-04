# Spec sheet — `bench/tritonbench_g/matmul_triton_autotune/MatmulTritonAutotune.lean`

**Python source:** `bench/tritonbench_g/matmul_triton_autotune/matmul_triton_autotune.py`

## Public theorem: `matmul_autotune_closed_form_correct`

<details><summary>docstring</summary>

```
/-- **Closed-form correctness for `matmul_triton_autotune` (general statement).**

For arbitrary linear program id `pid`, tile dims `BM`/`BN`, K-block size `BLOCK_K`,
and K-block count `numKBlocks` (so the contracted dimension is
`K = BLOCK_K · numKBlocks`), every **active** output cell of the computed
`BM × BN` tile equals `fp16( act( Σ_{k < BLOCK_K·numKBlocks} A[i,k] · B[k,j] ) )` —
the genuine matrix product (over ℝ) of the loaded `A`/`B` tiles, optionally passed
through `leaky_relu`, cast to float16 — **not** the kernel's own executed value;
inactive lanes are left untouched.

Layout: `A[i,k]` at `A + offs_am(i)·stride_am + k·stride_ak`, `B[k,j]` at
`B + k·stride_bk + offs_bn(j)·stride_bn`, `C[i,j]` at
`C + stride_cm·offs_cm(i) + stride_cn·offs_cn(j)`, with `pid_m`/`pid_n` derived by
the kernel's L2-grouping schedule, `offs_am(i) = (pid_m·BM + i) % M`,
`offs_bn(j) = (pid_n·BN + j) % N` (the row-major pointer arithmetic with index
wrap). Preconditions: output-offset injectivity and clean initial `undef`. -/
```
</details>

**Statement:**
```lean
theorem matmul_autotune_closed_form_correct
    (A B C : RegionName) (s : BlockState)
    (M N BM BN GM sam sak sbk sbn scm scn BLOCK_K numKBlocks : Nat) (K : Nat)
    (hK : K = BLOCK_K * numKBlocks) (ACTIVATION : Bool)
    (hcn : scn = 1) (hbnle : BN ≤ scm)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ComputeCorrect.Realizes
      (kernel := matmul_autotune_surface A B C M N K sam sak sbk sbn scm scn
        BM BN BLOCK_K GM numKBlocks ACTIVATION)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s M N BM BN GM)
        (fun idx => (C, cOffset s M N BM BN GM scm scn idx)))
      (expected := fun idx : TileIndex [BM, BN] =>
        MemCell.of .fp16
          (FloatDType.real.cast FloatDType.fp16
            (some (matmulSpec s A B M N BM BN GM sam sak sbk sbn BLOCK_K numKBlocks ACTIVATION idx.1 idx.2.1))))
```

**Assumptions / layout contracts:**
- `hK : K = BLOCK_K * numKBlocks`
- `hcn : scn = 1`
- `hbnle : BN ≤ scm`
- `hundef : ∀ rg o, s.undef rg o = 0`

**Closed-form spec defs (transitive):** `matmul_autotune_surface`, `active`, `cOffset`, `matmulSpec`, `rowGlobal`, `colGlobal`, `act`, `aElem`, `bElem`, `pidM`, `pidN`, `leakyReLU`, `rowIndex`, `colIndex`, `kernelMin`

<details><summary><code>matmul_autotune_surface</code></summary>

```
/-- Faithful transcription of `matmul_triton_autotune.py`'s `matmul_kernel`.

The contracted dimension is presented as `K = BLOCK_SIZE_K · numKBlocks` so the
loop bound `tl.cdiv(K, BLOCK_SIZE_K) = numKBlocks` is exact; it is supplied as the
antiquoted `numKBlocks`. All other surface structure — the L2-grouping schedule,
the per-block `offs_k < K - k·BLOCK_K` load masks, the fused
`tl.dot(a, b, accumulator)`, the optional `leaky_relu`, the `float16` cast, and
the `(row<M)&(col<N)`-masked store — is transcribed verbatim. The Python string
constexpr `ACTIVATION == "leaky_relu"` is the Lean `Bool` parameter `ACTIVATION`. -/
```
```lean
def matmul_autotune_surface
    (A B C : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M numKBlocks : Nat)
    (ACTIVATION : Bool) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  num_pid_m = tl.cdiv($(M), $(BLOCK_SIZE_M))
  num_pid_n = tl.cdiv($(N), $(BLOCK_SIZE_N))
  num_pid_in_group = $(GROUP_SIZE_M) * num_pid_n
  group_id = pid // num_pid_in_group
  first_pid_m = group_id * $(GROUP_SIZE_M)
  group_size_m = min(num_pid_m - first_pid_m, $(GROUP_SIZE_M))
  pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
  pid_n = (pid % num_pid_in_group) // group_size_m
  offs_am = (pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))) % $(M)
  offs_bn = (pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))) % $(N)
  offs_k = tl.arange(0, $(BLOCK_SIZE_K))
  a_ptrs = A + offs_am[:, None] * $(stride_am) + offs_k[None, :] * $(stride_ak)
  b_ptrs = B + offs_k[:, None] * $(stride_bk) + offs_bn[None, :] * $(stride_bn)
  accumulator = tl.zeros([$(BLOCK_SIZE_M), $(BLOCK_SIZE_N)], dtype=tl.float32)
  for kk in range($(0), $(numKBlocks), $(1)) {
    a = tl.load(a_ptrs, mask=offs_k[None, :] < $(K) - kk * $(BLOCK_SIZE_K), other=0.0)
    b = tl.load(b_ptrs, mask=offs_k[:, None] < $(K) - kk * $(BLOCK_SIZE_K), other=0.0)
    accumulator = tl.dot(a, b, accumulator)
    a_ptrs += $(BLOCK_SIZE_K) * $(stride_ak)
    b_ptrs += $(BLOCK_SIZE_K) * $(stride_bk)
  }
  if ACTIVATION {
    accumulator = leaky_relu(accumulator)
  }
  c = (accumulator).to(tl.float16)
  offs_cm = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
  offs_cn = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
  c_ptrs = C + $(stride_cm) * offs_cm[:, None] + $(stride_cn) * offs_cn[None, :]
  c_mask = (offs_cm[:, None] < $(M)) & (offs_cn[None, :] < $(N))
  tl.store(c_ptrs, c, mask=c_mask)
}
```
</details>

<details><summary><code>active</code></summary>

```
/-- The boundary predicate `(row < M) & (col < N)` for tile lane `(i,j)`. -/
```
```lean
def active (s0 : BlockState) (M N BM BN GM : Nat) (idx : TileIndex [BM, BN]) : Prop :=
  rowGlobal s0 M N BM BN GM idx.1 < M ∧ colGlobal s0 M N BM BN GM idx.2.1 < N
```
</details>

<details><summary><code>cOffset</code></summary>

```
/-- The output store address for tile lane `(i,j)`: `scm · offs_cm i + scn · offs_cn j`
(the kernel's `c_ptrs`, using the **un-wrapped** global `offs_cm`/`offs_cn`). -/
```
```lean
def cOffset (s0 : BlockState) (M N BM BN GM scm scn : Nat) (idx : TileIndex [BM, BN]) : Nat :=
  scm * rowGlobal s0 M N BM BN GM idx.1 + scn * colGlobal s0 M N BM BN GM idx.2.1
```
</details>

<details><summary><code>matmulSpec</code></summary>

```
/-- **Genuine matmul+activation spec** (over ℝ):
`C[i,j] = act( Σ_{k < BLOCK_K·numKBlocks} A[i,k] · B[k,j] )`. -/
```
```lean
noncomputable def matmulSpec (s : BlockState) (A B : RegionName)
    (M N BM BN GM sam sak sbk sbn BLOCK_K numKBlocks : Nat) (ACTIVATION : Bool)
    (i : Fin BM) (j : Fin BN) : ℝ :=
  act ACTIVATION
    (gemmSum (aElem s A M N BM BN GM sam sak i) (bElem s B M N BM BN GM sbk sbn j)
      (BLOCK_K * numKBlocks))
```
</details>

<details><summary><code>rowGlobal</code></summary>

```
/-- Global output row of tile lane `i`: `pid_m · BLOCK_M + i`, **before** the
`% M` wrap (the kernel's `offs_cm`). -/
```
```lean
def rowGlobal (s : BlockState) (M N BM BN GM : Nat) (i : Fin BM) : Nat :=
  pidM (s.pids 0) M N BM BN GM * BM + i.val
```
</details>

<details><summary><code>colGlobal</code></summary>

```
/-- Global output column of tile lane `j`: `pid_n · BLOCK_N + j`, before wrap. -/
```
```lean
def colGlobal (s : BlockState) (M N BM BN GM : Nat) (j : Fin BN) : Nat :=
  pidN (s.pids 0) M N BM BN GM * BN + j.val
```
</details>

<details><summary><code>act</code></summary>

```
/-- The applied activation: `leakyReLU` when `ACTIVATION`, else the identity. -/
```
```lean
noncomputable def act (ACTIVATION : Bool) (x : ℝ) : ℝ :=
  if ACTIVATION then leakyReLU x else x
```
</details>

<details><summary><code>aElem</code></summary>

```
/-- `A[i, k] = readMem A (offs_am i · stride_am + k · stride_ak)`. -/
```
```lean
noncomputable def aElem (s : BlockState) (A : RegionName) (M N BM BN GM sam sak : Nat)
    (i : Fin BM) (k : Nat) : ℝ :=
  s.readMem A (rowIndex s M N BM BN GM i * sam + k * sak)
```
</details>

<details><summary><code>bElem</code></summary>

```
/-- `B[k, j] = readMem B (k · stride_bk + offs_bn j · stride_bn)`. -/
```
```lean
noncomputable def bElem (s : BlockState) (B : RegionName) (M N BM BN GM sbk sbn : Nat)
    (j : Fin BN) (k : Nat) : ℝ :=
  s.readMem B (k * sbk + colIndex s M N BM BN GM j * sbn)
```
</details>

<details><summary><code>pidM</code></summary>

```
/-- The kernel's L2-grouping derivation of `pid_m` from the linear `pid`. -/
```
```lean
def pidM (pid M N BM BN GM : Nat) : Nat :=
  let num_pid_m := cdiv M BM
  let num_pid_n := cdiv N BN
  let num_pid_in_group := GM * num_pid_n
  let group_id := pid / num_pid_in_group
  let first_pid_m := group_id * GM
  let group_size_m := kernelMin (num_pid_m - first_pid_m) GM
  first_pid_m + ((pid % num_pid_in_group) % group_size_m)
```
</details>

<details><summary><code>pidN</code></summary>

```
/-- The kernel's L2-grouping derivation of `pid_n` from the linear `pid`. -/
```
```lean
def pidN (pid M N BM BN GM : Nat) : Nat :=
  let num_pid_m := cdiv M BM
  let num_pid_n := cdiv N BN
  let num_pid_in_group := GM * num_pid_n
  let group_id := pid / num_pid_in_group
  let first_pid_m := group_id * GM
  let group_size_m := kernelMin (num_pid_m - first_pid_m) GM
  (pid % num_pid_in_group) / group_size_m
```
</details>

<details><summary><code>leakyReLU</code></summary>

```
/-- Real-valued leaky-ReLU activation (slope `0.01` below zero), matching the
kernel's `leaky_relu`. -/
```
```lean
noncomputable def leakyReLU (x : ℝ) : ℝ := if x ≥ 0 then x else 0.01 * x
```
</details>

<details><summary><code>rowIndex</code></summary>

```
/-- The `% M`-wrapped A-row index of tile lane `i` (the kernel's `offs_am`). -/
```
```lean
def rowIndex (s : BlockState) (M N BM BN GM : Nat) (i : Fin BM) : Nat :=
  rowGlobal s M N BM BN GM i % M
```
</details>

<details><summary><code>colIndex</code></summary>

```
/-- The `% N`-wrapped B-column index of tile lane `j` (the kernel's `offs_bn`). -/
```
```lean
def colIndex (s : BlockState) (M N BM BN GM : Nat) (j : Fin BN) : Nat :=
  colGlobal s M N BM BN GM j % N
```
</details>

<details><summary><code>kernelMin</code></summary>

```
/-- `min` as the kernel's `tl.where(a < b, a, b)` spells it. -/
```
```lean
def kernelMin (a b : Nat) : Nat := if a < b then a else b
```
</details>
