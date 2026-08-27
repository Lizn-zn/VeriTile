# Spec sheet — `bench/tritonbench_g/bmm_optimized/BmmOptimized.lean`

**Python source:** `bench/tritonbench_g/bmm_optimized/bmm_optimized.py`

## Public theorem: `bmm_o_exec_genuine`

<details><summary>docstring</summary>

```
/-- **★ Main theorem: the `o` store is the genuine batched GEMM closed
form.**

For every program `(pidx, pidy, pid_b)` on any launch grid (both quantified
through `s.pids` / `s.numPids`), executing the full surface succeeds and the
masked `o` store holds `Σ_{t < K} A[pid_b, m, t] · B[pid_b, t, n]` at every
in-window lane, where `(m, n)` are the global row/column of the
CTA-reordered tile (`bmmPidM`/`bmmPidN` — the `GROUP_M = 1` identity map or
the grouped swizzle with its runtime `GROUP_SIZE` boundary gate, both arms
proven).

Side conditions: `TILE_N ≤ N` (store-lane injectivity under the row-major
stride), `0 < TILE_K` (the kernel's own `tl.cdiv` trip count), and the
clean-input `hundef` (the masked loads carry no `other`, so masked-off
lanes read the `undef` channel — the `bmm_chunk_fwd` convention). **No**
divisibility hypotheses on `M`, `N`, or `K`: this is the fully-masked
`DIVISIBLE_* = False` arm, exact on arbitrary ragged shapes. -/
```
</details>

**Statement:**
```lean
specification bmm_o_exec_genuine
    (s : BlockState) (A B O : RegionName)
    (M N K TILE_M TILE_N TILE_K GROUP_M : Nat)
    (hTN : TILE_N ≤ N) (hTK : 0 < TILE_K)
    (hundef : ∀ rg off, s.undef rg off = 0) :
    ∃ sF, exec (bmm_surface A B O M N K TILE_M TILE_N TILE_K
        GROUP_M).toAlgKernel s = some sF
      ∧ (∀ idx : TileIndex [TILE_M, TILE_N],
          bmmOActive s M N TILE_M TILE_N GROUP_M idx →
          sF.readMem O (bmmOOffset s M N TILE_M TILE_N GROUP_M idx)
            = bmmOOut s A B M N K
                (bmmRowG s TILE_M GROUP_M idx.1.val)
                (bmmColG s TILE_N GROUP_M idx.2.1.val))
```

**Assumptions / layout contracts:**
- `hTN : TILE_N ≤ N`
- `hTK : 0 < TILE_K`
- `hundef : ∀ rg off, s.undef rg off = 0`
- `∀ idx : TileIndex [TILE_M, TILE_N],
          bmmOActive s M N TILE_M TILE_N GROUP_M idx →
          sF.readMem O (bmmOOffset s M N TILE_M TILE_N GROUP_M idx)
            = bmmOOut s A B M N K
                (bmmRowG s TILE_M GROUP_M idx.1.val)
                (bmmColG s TILE_N GROUP_M idx.2.1.val)`

**Closed-form spec defs (transitive):** `bmm_surface`, `bmmOActive`, `bmmOOffset`, `bmmOOut`, `bmmRowG`, `bmmColG`, `bmmAVal`, `bmmBVal`, `bmmPidM`, `bmmPidN`, `bmmGroupSize`

<details><summary><code>bmm_surface</code></summary>

```
/-- Faithful transcription of `bmm_kernel` at
`DIVISIBLE_M = DIVISIBLE_N = DIVISIBLE_K = False` (see the preamble), with
both `GROUP_M` CTA-reorder arms. -/
```
```lean
def bmm_surface
    (A B O : RegionName)
    (M N K TILE_M TILE_N TILE_K GROUP_M : Nat) :
    ComputeKernel := triton {
  pid_b = tl.program_id(2)
  pidx = tl.program_id(0)
  pidy = tl.program_id(1)
  if $(GROUP_M) == $(1) {
    pid_m = pidx
    pid_n = pidy
  } else {
    gridx = tl.num_programs(0)
    gridy = tl.num_programs(1)
    pid = pidx + pidy * gridx
    num_CTA_per_group = gridy * $(GROUP_M)
    group_id = pid // num_CTA_per_group
    inner_group_id = pid % num_CTA_per_group
    if (group_id * $(GROUP_M) + $(GROUP_M)) > gridx {
      GROUP_SIZE = gridx % $(GROUP_M)
    } else {
      GROUP_SIZE = $(GROUP_M)
    }
    pid_m = group_id * $(GROUP_M) + inner_group_id % GROUP_SIZE
    pid_n = inner_group_id // GROUP_SIZE
  }
  offs_m = pid_m * $(TILE_M) + tl.arange(0, $(TILE_M))
  offs_n = pid_n * $(TILE_N) + tl.arange(0, $(TILE_N))
  offs_k = tl.arange(0, $(TILE_K))
  mask_m = offs_m < $(M)
  mask_n = offs_n < $(N)
  a_ptrs = A + (pid_b * $(M) * $(K) + offs_m[:, None] * $(K) + offs_k[None, :])
  b_ptrs = B + (pid_b * $(K) * $(N) + offs_k[:, None] * $(N) + offs_n[None, :])
  o_ptrs = O + (pid_b * $(M) * $(N) + offs_m[:, None] * $(N) + offs_n[None, :])
  num_iters = tl.cdiv($(K), $(TILE_K))
  o = tl.zeros([$(TILE_M), $(TILE_N)], dtype=tl.float32)
  for _i in range($(0), num_iters, $(1)) {
    mask_k = offs_k < $(K)
    mask_a = mask_m[:, None] & mask_k[None, :]
    mask_b = mask_k[:, None] & mask_n[None, :]
    a = tl.load(a_ptrs, mask=mask_a)
    b = tl.load(b_ptrs, mask=mask_b)
    offs_k += $(TILE_K)
    a_ptrs += $(TILE_K)
    b_ptrs += $(TILE_K) * $(N)
    o += tl.dot(a, b, allow_tf32=false)
  }
  mask_c = mask_m[:, None] & mask_n[None, :]
  tl.store(o_ptrs, o, mask=mask_c)
}
```
</details>

<details><summary><code>bmmOActive</code></summary>

```
/-- An `o` store lane is *active* when it maps inside the `M × N` window. -/
```
```lean
def bmmOActive (s : BlockState) (M N TILE_M TILE_N GROUP_M : Nat)
    (idx : TileIndex [TILE_M, TILE_N]) : Prop :=
  bmmRowG s TILE_M GROUP_M idx.1.val < M ∧ bmmColG s TILE_N GROUP_M idx.2.1.val < N
```
</details>

<details><summary><code>bmmOOffset</code></summary>

```
/-- The `o` store address at lane `(i, j)`. -/
```
```lean
def bmmOOffset (s : BlockState) (M N TILE_M TILE_N GROUP_M : Nat)
    (idx : TileIndex [TILE_M, TILE_N]) : Nat :=
  s.pids 2 * M * N + bmmRowG s TILE_M GROUP_M idx.1.val * N
    + bmmColG s TILE_N GROUP_M idx.2.1.val
```
</details>

<details><summary><code>bmmOOut</code></summary>

```
/-- **The stored `o` lane** — the batched GEMM closed form
`Σ_{t < K} A[pid_b, m, t] · B[pid_b, t, n]` (raw reads; the store readback
is claimed at in-window lanes only). -/
```
```lean
noncomputable def bmmOOut (s : BlockState) (A B : RegionName)
    (M N K : Nat) (m n : Nat) : ℝ :=
  ∑ t ∈ Finset.range K, bmmAVal s A M K m t * bmmBVal s B K N t n
```
</details>

<details><summary><code>bmmRowG</code></summary>

```
/-- The global output row of tile lane `i`. -/
```
```lean
def bmmRowG (s : BlockState) (TILE_M GROUP_M : Nat) (i : Nat) : Nat :=
  bmmPidM (s.pids 0) (s.pids 1) (s.numPids 0) (s.numPids 1) GROUP_M * TILE_M + i
```
</details>

<details><summary><code>bmmColG</code></summary>

```
/-- The global output column of tile lane `j`. -/
```
```lean
def bmmColG (s : BlockState) (TILE_N GROUP_M : Nat) (j : Nat) : Nat :=
  bmmPidN (s.pids 0) (s.pids 1) (s.numPids 0) (s.numPids 1) GROUP_M * TILE_N + j
```
</details>

<details><summary><code>bmmAVal</code></summary>

```
/-- `A[pid_b, m, t]` (row-major `M × K` batch slab). -/
```
```lean
noncomputable def bmmAVal (s : BlockState) (A : RegionName) (M K m t : Nat) : ℝ :=
  s.readMem A (s.pids 2 * M * K + m * K + t)
```
</details>

<details><summary><code>bmmBVal</code></summary>

```
/-- `B[pid_b, t, n]` (row-major `K × N` batch slab). -/
```
```lean
noncomputable def bmmBVal (s : BlockState) (B : RegionName) (K N t n : Nat) : ℝ :=
  s.readMem B (s.pids 2 * K * N + t * N + n)
```
</details>

<details><summary><code>bmmPidM</code></summary>

```
/-- `pid_m` under the CTA reorder (identity when `GROUP_M = 1`). -/
```
```lean
def bmmPidM (pidx pidy gridx gridy GROUP_M : Nat) : Nat :=
  if GROUP_M = 1 then pidx
  else
    (pidx + pidy * gridx) / (gridy * GROUP_M) * GROUP_M
      + (pidx + pidy * gridx) % (gridy * GROUP_M)
          % bmmGroupSize gridx ((pidx + pidy * gridx) / (gridy * GROUP_M)) GROUP_M
```
</details>

<details><summary><code>bmmPidN</code></summary>

```
/-- `pid_n` under the CTA reorder (identity when `GROUP_M = 1`). -/
```
```lean
def bmmPidN (pidx pidy gridx gridy GROUP_M : Nat) : Nat :=
  if GROUP_M = 1 then pidy
  else
    (pidx + pidy * gridx) % (gridy * GROUP_M)
      / bmmGroupSize gridx ((pidx + pidy * gridx) / (gridy * GROUP_M)) GROUP_M
```
</details>

<details><summary><code>bmmGroupSize</code></summary>

```
/-- The runtime `GROUP_SIZE` boundary gate: `gridx % GROUP_M` on a ragged
final group, else `GROUP_M`. -/
```
```lean
def bmmGroupSize (gridx gid GROUP_M : Nat) : Nat :=
  if gid * GROUP_M + GROUP_M > gridx then gridx % GROUP_M else GROUP_M
```
</details>
