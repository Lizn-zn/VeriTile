import VeriTile.Triton

/-!
# `int_scaled_matmul` — strict per-kernel correctness (both JIT kernels)

This file is the DSL port of `int_scaled_matmul.py`'s
`matmul_kernel_with_block_pointers` (the audit anchor, the file's first JIT) —
a block-pointer int8×int8→int32 GEMM: three `tl.make_block_ptr` views with
dynamic per-`pid` offsets, `boundary_check=(0, 1)` zero-fill loads, an
`Op.dotInt` accumulation over the `range(0, K, BLOCK_K)` step-form loop with
in-loop `tl.advance`, and a boundary-checked block-pointer store of the int32
tile. The second JIT kernel, `scaled_matmul_kernel_with_block_pointers`
(launched by `int_scaled_matmul_kernel`), is modeled in full as well: a
classic pointer GEMM (no block pointers despite its name) over the
**descending** `range(K, 0, -BLOCK_K)` loop with the genuine `EVEN_K` Bool
constexpr (both arms), followed by the inductor-generated epilogue that
rescales the int32 accumulator by the per-row fp32 scale `s1` through
`tl.broadcast_to` flat-index addressing. Both kernels are launched by the
test (`test_matmul_kernel` drives `int_matmul_kernel` then
`int_scaled_matmul_kernel`); `A`/`B` are `torch.int8` signed inputs
(`randint(-128, 128)`) on the `.int` channel and `C` is `torch.int32`.

Translation-surface blocker: kernel 1 (`matmul_kernel_with_block_pointers`),
four disclosed surface deviations, none semantic. **(1)** the block-pointer
loads are spelled faithfully (`tl.load(bp, boundary_check=(0, 1))`, no
added kwargs): their `.int` channel is inherited from the **typed base
region** of `tl.make_block_ptr` (`base=$((a_ptr : Region .int))`, the
family's usual typed-binder ascription) — a block pointer knows its element
type, exactly as in Triton, via this port's inference rider
(`tl.make_block_ptr` arms in the DSL's pointer-element-dtype propagation;
untyped bases keep the historical `.real` default, so every existing
block-ptr port lowers unchanged). What *is* disclosed is that integer
**widths are erased** (the
`int8_quantization` fixed-width family) — the model keeps full launch-state
ℤ values where int8 hardware would wrap. **(2)** `tl.advance(ptr, (0, BLOCK_K))` tuple deltas are
respelled with brackets `[$(0), $(BLOCK_K)]` (the audit-known tuple→bracket
respell; same for `tl.zeros((M, N), …)` → `tl.zeros([$(M), $(N)], …)`).
**(3)** the constexpr parameter rebind `GROUP_M = min(num_pid_m -
first_pid_m, GROUP_M)` is spelled with the *register* `GROUP_M` on the left
and the antiquoted *parameter* `$(GROUP_M)` inside `min` — after this
statement every bare `GROUP_M` ident resolves to the register, exactly as
Python rebinds the name (probed: the register may shadow the Lean binder's
name with no clash, since the binder only ever appears antiquoted).
**(4)** the TMA `order=(1, 0)` tuples are scheduling metadata the DSL erases
into the same block-pointer AST (the `matmul_tma` precedent). Nothing else
deviates: the step-form loop `for k in range(0, K, BLOCK_K)` is spelled
directly (`Stmt.forRange` carries the step), so no trip-count binder is
introduced and raggedness in every axis is handled by the boundary checks.

Translation-surface blocker: kernel 2 (`scaled_matmul_kernel_with_block_pointers`),
five disclosed surface deviations, none semantic. **(1)** the **descending**
loop `for k in range(K, 0, -BLOCK_K)` is presented as the ascending
change of variable `for j in range(0, numKBlocks, 1)` with the remaining
count rematerialized as the first body statement `k = K - j*BLOCK_K`
(ℕ-truncated subtraction; the `parallel_retention_attention` descending-lever
precedent) — `numKBlocks` is the antiquoted trip count `tl.cdiv(K, BLOCK_K)`
with the per-arm hypothesis `hK` (exact form `K = numKBlocks·BLOCK_K` for
`EVEN_K = true`, ceil form `K ≤ numKBlocks·BLOCK_K` otherwise). **(2)**
`ACC_TYPE: tl.constexpr = tl.int32` is a dtype-valued constexpr fixed at its
only instantiation: `tl.zeros(…, dtype=ACC_TYPE)` is spelled
`dtype=tl.int32` (the `bmm_optimized` fixed-arm family). **(3)** the
inductor suffix's `tl.broadcast_to(…, mask.shape)` attribute argument is
respelled to the explicit literal dims `[BLOCK_M, BLOCK_N]` (attribute
spellings are not in the DSL; the target shape is statically known).
**(4)** `eviction_policy="evict_last"` on the `s1` load is kept and erased
as a pure cache hint; the positional `mask` arguments of the suffix's
`tl.load`/`tl.store` parse natively (no respell). **(5)** integer widths
are erased as in kernel 1, and `other=0.0` on the `.int`-channel masked
loads is the faithful `Op.constInt 0` (the 171st-port rider). The
`stride_s1m`/`stride_s1n` parameters are passed by the launcher but **dead
in the body** (the `s1` load address is the strideless
`tl.broadcast_to(idx_m, …)` — the `fused_recurrent_retention` dead-stride
precedent); they are kept as (unused) binders. `EVEN_K` is a genuine
`Bool` parameter with **both arms modeled** (the host passes `K % 2 == 0`;
the `has_bias` / `NO_GROUPS` precedent). The `acc * tmp0` store value is
`.int × .real` and auto-promotes through `Op.intToReal`: the stored cell is
the **exact ℝ** product `(Σ ℤ : ℝ) · s1(row)` in a `.real`-typed `MemCell`,
while the host allocates `C` as int32 — the hardware float→int32 truncation
at the container boundary is outside the model (the #154 fixed-width
family; the cell honestly carries the exact real, not a truncated int).

The textual py↔lean scans in `bench/audit_tritonbench_g.sh` exempt this port
on these markers (registered in `proof_blockers.md`).

## The specs

Kernel 1, for an in-range output cell `(row, col)` (`row < M`, `col < N`),
over exact ℤ:

`C[row, col] = ∑ kk < K, A[row, kk] · B[kk, col]`

with `A[row, kk]` read at `row·stride_am + kk·stride_ak` and `B[kk, col]` at
`kk·stride_bk + col·stride_bn` on the `.int` channel of the **launch** state
(the block pointers' offset-`(pid_m·BM, 0)` / `(0, pid_n·BN)` addresses; the
`boundary_check` zero-fill kills every out-of-range lane, so no divisibility
of `K` is assumed — only `0 < BLOCK_K`).

Kernel 2, same in-range cells, at the flat inductor address `col + N·row`:

`C[col + N·row] = MemCell.of .real ((∑ kk < K, A[row, kk]·B[kk, col] : ℤ→ℝ) · s1(row))`

with `s1` read at the strideless address `row`. The output region is never
read back into either spec: no part of the trust path is self-referential.

## Proof map

```
int_scaled_matmul_matmul_exec_genuine        kernel-1 headline
├─ mm_body_eq              13 top-level statements by `rfl`
├─ mmPreLoop_run           9 swizzle scalars + 2 block ptrs + zeros → mmInv 0
├─ mmLoop_collapse         `forRange_inv` at step BLOCK_K over `mmInv`
│  └─ mmBody_run           2 boundary-checked loads, dotInt, 2 advances
├─ mmAccVal_final          guarded running sum → the `Fin K` spec at active lanes
└─ mmPostLoop_run          c rebind, c_block_ptr, boundary-checked store
   └─ mm_store_props       `MemCell`-level `.int` scatter readback
mmCAddr_injective_rowMajor                   discharges the headline's `hInj`

int_scaled_matmul_scaled_exec_genuine        kernel-2 headline
├─ sm_body_eq              20 top-level statements by `rfl`
├─ smPreLoop_run           swizzle + wrapped offsets + ptr tiles + zeros → smInv 0
├─ smLoop_collapse         `forRange_inv` over `smInv` (ascending substitution)
│  └─ smBody_run           k remat + `EVEN_K` branch (both arms) + dotInt + advances
├─ smAccVal_final          guarded sum → the `Fin K` spec (per-arm `hK`)
└─ smPostLoop_run          rm/rn remat, inductor suffix, masked `.real` store
   ├─ sm_s1Load_eq         strideless broadcast_to load of the row scale
   └─ sm_store_props       `.real` scatter readback (flat-address injectivity
                           proven outright on the masked lanes — no hypothesis)
```

## Modeling boundary

Arithmetic is exact ℤ for both accumulations and exact ℝ for the scale
epilogue (no bit-accurate IEEE float); the host launches (1-D grids
`cdiv(M,BM)·cdiv(N,BN)`), `num_warps`/`num_stages`/`num_ctas`, and the
`Config` block sizes are the trusted boundary. Every dimension, stride and
block size stays a symbolic parameter.
-/

namespace VeriTile.Bench.TritonBenchG.IntScaledMatmul

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-! # ══════════ CORRECT — genuine / dimension-general (review this) ══════════ -/

section Correct_without_Rounding

/-! ## Kernel 1: `matmul_kernel_with_block_pointers` — surface -/

set_option linter.unusedVariables false in
/-- Faithful transcription of `matmul_kernel_with_block_pointers` (the
launcher `int_matmul_kernel`'s target). See the module docstring's kernel-1
blocker for the four disclosed respells (`dtype=` pins on the block-ptr
loads, tuple→bracket shape/delta lists, the `GROUP_M` register rebind, and
the erased `order` tuples). -/
def int_scaled_matmul_matmul_surface
    (a_ptr b_ptr : Region .int) (c_ptr : RegionName)
    (M N K : Nat)
    (stride_am stride_ak stride_bk stride_bn stride_cm stride_cn : Nat)
    (BLOCK_M BLOCK_N BLOCK_K GROUP_M : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  num_pid_m = tl.cdiv($(M), $(BLOCK_M))
  num_pid_n = tl.cdiv($(N), $(BLOCK_N))
  num_pid_in_group = $(GROUP_M) * num_pid_n
  group_id = pid // num_pid_in_group
  first_pid_m = group_id * $(GROUP_M)
  GROUP_M = min(num_pid_m - first_pid_m, $(GROUP_M))
  pid_m = first_pid_m + (pid % GROUP_M)
  pid_n = (pid % num_pid_in_group) // GROUP_M
  a_block_ptr = tl.make_block_ptr(base=$((a_ptr : Region .int)), shape=($(M), $(K)),
    strides=($(stride_am), $(stride_ak)), offsets=(pid_m * $(BLOCK_M), 0),
    block_shape=($(BLOCK_M), $(BLOCK_K)), order=(1, 0))
  b_block_ptr = tl.make_block_ptr(base=$((b_ptr : Region .int)), shape=($(K), $(N)),
    strides=($(stride_bk), $(stride_bn)), offsets=(0, pid_n * $(BLOCK_N)),
    block_shape=($(BLOCK_K), $(BLOCK_N)), order=(1, 0))
  accumulator = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.int32)
  for k in range(0, $(K), $(BLOCK_K)) {
    a = tl.load(a_block_ptr, boundary_check=(0, 1))
    b = tl.load(b_block_ptr, boundary_check=(0, 1))
    accumulator += tl.dot(a, b)
    a_block_ptr = tl.advance(a_block_ptr, [$(0), $(BLOCK_K)])
    b_block_ptr = tl.advance(b_block_ptr, [$(BLOCK_K), $(0)])
  }
  c = accumulator
  c_block_ptr = tl.make_block_ptr(base=c_ptr, shape=($(M), $(N)),
    strides=($(stride_cm), $(stride_cn)),
    offsets=(pid_m * $(BLOCK_M), pid_n * $(BLOCK_N)),
    block_shape=($(BLOCK_M), $(BLOCK_N)), order=(1, 0))
  tl.store(c_block_ptr, c, boundary_check=(0, 1))
}

/-- The kernel-1 surface lowers to a supported algorithm. -/
theorem int_scaled_matmul_matmul_surface_toAlgorithm_supported
    (a_ptr b_ptr : Region .int) (c_ptr : RegionName)
    (M N K : Nat)
    (stride_am stride_ak stride_bk stride_bn stride_cm stride_cn : Nat)
    (BM BN BK GM : Nat) :
    ∃ alg, (int_scaled_matmul_matmul_surface a_ptr b_ptr c_ptr M N K
      stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BM BN BK GM).toAlgorithm? = Except.ok alg := by
  simp [int_scaled_matmul_matmul_surface, ComputeExpr.toAlgorithm?]

/-! ## Kernel 2: `scaled_matmul_kernel_with_block_pointers` — surface -/

set_option linter.unusedVariables false in
/-- Faithful transcription of `scaled_matmul_kernel_with_block_pointers`
(the launcher `int_scaled_matmul_kernel`'s target; a classic pointer GEMM —
no block pointers despite the name — with the inductor store suffix). See
the module docstring's kernel-2 blocker for the five disclosed respells
(ascending change of variable with `k = K - j·BLOCK_K`, the fixed
`ACC_TYPE = tl.int32` constexpr, the `mask.shape` → literal-dims respell in
`tl.broadcast_to`, the erased `eviction_policy` hint, and the erased int
widths). `stride_s1m`/`stride_s1n` are passed by the launcher but dead in
the body (kept as unused binders). -/
def int_scaled_matmul_scaled_surface
    (a_ptr b_ptr : Region .int) (c_ptr s1_ptr : RegionName)
    (M N K : Nat)
    (stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      stride_s1m stride_s1n : Nat)
    (BLOCK_M BLOCK_N BLOCK_K GROUP_M : Nat) (EVEN_K : Bool)
    (numKBlocks : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0)
  grid_m = ($((M : Nat)) + $(BLOCK_M) - $(1)) // $(BLOCK_M)
  grid_n = ($((N : Nat)) + $(BLOCK_N) - $(1)) // $(BLOCK_N)
  width = $(GROUP_M) * grid_n
  group_id = pid // width
  group_size = min(grid_m - group_id * $(GROUP_M), $(GROUP_M))
  pid_m = group_id * $(GROUP_M) + (pid % group_size)
  pid_n = (pid % width) // (group_size)
  rm = pid_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  rn = pid_n * $(BLOCK_N) + tl.arange(0, $(BLOCK_N))
  ram = tl.max_contiguous(tl.multiple_of(rm % $(M), $(BLOCK_M)), $(BLOCK_M))
  rbn = tl.max_contiguous(tl.multiple_of(rn % $(N), $(BLOCK_N)), $(BLOCK_N))
  rk = tl.arange(0, $(BLOCK_K))
  A = $((a_ptr : Region .int)) + (ram[:, None] * $(stride_am) + rk[None, :] * $(stride_ak))
  B = $((b_ptr : Region .int)) + (rk[:, None] * $(stride_bk) + rbn[None, :] * $(stride_bn))
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.int32)
  for j in range(0, $(numKBlocks), $(1)) {
    k = $(K) - j * $(BLOCK_K)
    if EVEN_K {
      a = tl.load(A)
      b = tl.load(B)
    } else {
      a = tl.load(A, mask=rk[None, :] < k, other=0.0)
      b = tl.load(B, mask=rk[:, None] < k, other=0.0)
    }
    acc += tl.dot(a, b)
    A += $(BLOCK_K) * $(stride_ak)
    B += $(BLOCK_K) * $(stride_bk)
  }
  rm = pid_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  rn = pid_n * $(BLOCK_N) + tl.arange(0, $(BLOCK_N))
  idx_m = rm[:, None]
  idx_n = rn[None, :]
  mask = (idx_m < $(M)) & (idx_n < $(N))
  xindex = idx_n + ($(N) * idx_m)
  tmp0 = tl.load(s1_ptr + (tl.broadcast_to(idx_m, [$(BLOCK_M), $(BLOCK_N)])), mask, eviction_policy="evict_last")
  tl.store(c_ptr + (tl.broadcast_to(xindex, [$(BLOCK_M), $(BLOCK_N)])), acc * tmp0, mask)
}

/-- The kernel-2 surface lowers to a supported algorithm. -/
theorem int_scaled_matmul_scaled_surface_toAlgorithm_supported
    (a_ptr b_ptr : Region .int) (c_ptr s1_ptr : RegionName)
    (M N K : Nat)
    (stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      stride_s1m stride_s1n : Nat)
    (BM BN BK GM : Nat) (EVEN_K : Bool) (numKBlocks : Nat) :
    ∃ alg, (int_scaled_matmul_scaled_surface a_ptr b_ptr c_ptr s1_ptr M N K
      stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      stride_s1m stride_s1n BM BN BK GM EVEN_K numKBlocks).toAlgorithm?
        = Except.ok alg := by
  simp [int_scaled_matmul_scaled_surface, ComputeExpr.toAlgorithm?]


/-! ## Shared eval recipes (private copies — bench ports never import each other)

Scalar / tile `evalOp` reductions in the `int8_matmul_kernel` style, plus the
`MemCell`-level scatter machinery for the two output stores (`.int` for
kernel 1, `.real` for kernel 2). -/

section SharedRecipes

private theorem is_constInt_eval (n : Int) (t : BlockState) :
    evalOp (Op.constInt n) t = some (Tile.scalar n) := by
  simp [evalOp]

private theorem is_full_eval {dtype : TileDType} (sh : TileShape) (e : Op dtype [])
    (t : BlockState) (v : Tile dtype []) (hv : evalOp e t = some v) :
    evalOp (Op.full sh e) t
      = some (⟨fun _ => v.data PUnit.unit⟩ : Tile dtype sh) := by
  rw [evalOp_full, hv]
  rfl

private theorem is_mod_eval {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op .nat a) (y : Op .nat b) (t : BlockState)
    (vx : Tile .nat a) (vy : Tile .nat b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.mod IntegralDType.nat bc x y) t
      = some (Tile.bop (IntegralDType.mod IntegralDType.nat) bc vx vy) := by
  simp only [evalOp, hx, hy]
  rfl

private theorem is_floorDiv_eval {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op .nat a) (y : Op .nat b) (t : BlockState)
    (vx : Tile .nat a) (vy : Tile .nat b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.floorDiv IntegralDType.nat bc x y) t
      = some (Tile.bop (IntegralDType.floorDiv IntegralDType.nat) bc vx vy) := by
  simp only [evalOp, hx, hy]
  rfl

private theorem is_divTile_eval {dtype : TileDType} (h : NumericDType dtype)
    {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op dtype a) (y : Op dtype b) (t : BlockState)
    (vx : Tile dtype a) (vy : Tile dtype b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.div h bc x y) t = some (Tile.bop h.div bc vx vy) := by
  rw [evalOp_div, hx, hy]
  rfl

private theorem is_ltTile_eval {dtype : TileDType} (h : ComparableDType dtype)
    {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op dtype a) (y : Op dtype b) (t : BlockState)
    (vx : Tile dtype a) (vy : Tile dtype b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.lt h bc x y) t = some (Tile.cop h.lt bc vx vy) := by
  rw [evalOp_lt, hx, hy]
  rfl

private theorem is_where_eval {dtype : TileDType} {sh : TileShape}
    (c : Op .bool sh) (x y : Op dtype sh) (t : BlockState)
    (vc : Tile .bool sh) (vx vy : Tile dtype sh)
    (hc : evalOp c t = some vc) (hx : evalOp x t = some vx)
    (hy : evalOp y t = some vy) :
    evalOp (Op.where c x y) t = some (Tile.select vc vx vy) := by
  rw [evalOp_where, hc, hx, hy]
  rfl

private theorem is_ptrAdd_eval {a b : TileShape} {out : TileShape}
    (bc : Broadcast a b out)
    (pnm : RegName) (t : BlockState) (pt : Tile .ptr a) (off : Op .nat b)
    (ov : Tile .nat b)
    (hp : t.regs .ptr a pnm = some pt) (ho : evalOp off t = some ov) :
    evalOp (Op.ptrAdd bc (Op.ref .ptr a pnm) off) t
      = some (Tile.ptrAdd bc pt ov) := by
  simp only [evalOp, evalOp_ref, hp, ho]
  rfl

private theorem is_ptrAddBase_eval {d : TileDType} {b out : TileShape}
    (bc : Broadcast [] b out) (rg : Region d) (t : BlockState)
    (off : Op .nat b) (ov : Tile .nat b) (ho : evalOp off t = some ov) :
    evalOp (Op.ptrAdd bc (Op.ptrBase rg) off) t
      = some (Tile.ptrAdd bc (Tile.scalar ((Region.cast rg : RegionName), 0)) ov) := by
  simp only [evalOp, ho]
  rfl

private theorem is_mulTile_eval {dtype : TileDType} (h : NumericDType dtype)
    {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op dtype a) (y : Op dtype b) (t : BlockState)
    (vx : Tile dtype a) (vy : Tile dtype b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.mul h bc x y) t = some (Tile.bop h.mul bc vx vy) := by
  rw [evalOp_mul, hx, hy]
  rfl

private theorem is_addTile_eval {dtype : TileDType} (h : NumericDType dtype)
    {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op dtype a) (y : Op dtype b) (t : BlockState)
    (vx : Tile dtype a) (vy : Tile dtype b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.add h bc x y) t = some (Tile.bop h.add bc vx vy) := by
  rw [evalOp_add, hx, hy]
  rfl

private theorem is_subTile_eval {dtype : TileDType} (h : NumericDType dtype)
    {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op dtype a) (y : Op dtype b) (t : BlockState)
    (vx : Tile dtype a) (vy : Tile dtype b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.sub h bc x y) t = some (Tile.bop h.sub bc vx vy) := by
  rw [evalOp_sub, hx, hy]
  rfl

private theorem is_expandDim_eval {dtype : TileDType} {sh : TileShape}
    (ax : Fin (sh.length + 1)) (x : Op dtype sh) (t : BlockState)
    (v : Tile dtype sh) (hv : evalOp x t = some v) :
    evalOp (Op.expandDim ax x) t = some (Tile.expandDim ax v) := by
  rw [evalOp_expandDim, hv]
  rfl

private theorem is_boolAnd_eval {a b out : TileShape} (bc : Broadcast a b out)
    (x : Op .bool a) (y : Op .bool b) (t : BlockState)
    (vx : Tile .bool a) (vy : Tile .bool b)
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.boolAnd bc x y) t
      = some (Tile.bop (fun u v : Bool => u && v) bc vx vy) := by
  simp only [evalOp, hx, hy]
  rfl

private theorem is_remap_eval {dtype : TileDType} {sh : TileShape}
    (outShape : TileShape) (map : TileIndex outShape → TileIndex sh)
    (x : Op dtype sh) (t : BlockState) (v : Tile dtype sh)
    (hv : evalOp x t = some v) :
    evalOp (Op.remap outShape map x) t = some (Tile.remap map v) := by
  simp only [evalOp, hv]
  rfl

private theorem is_intToReal_eval {sh : TileShape} (x : Op .int sh)
    (t : BlockState) (v : Tile .int sh) (hv : evalOp x t = some v) :
    evalOp (Op.intToReal x) t = some (Tile.intToReal v) := by
  simp only [evalOp, hv]
  rfl

/-- `tl.dot` on the `.int` channel at rank 2. `erw`, not `rw`: the operand
shapes are `[] ++ [M, K]`, which does not unfold at reducible transparency. -/
private theorem is_dotInt_eval {M K N : Nat} (x : Op .int [M, K])
    (y : Op .int [K, N]) (t : BlockState)
    (vx : Tile .int [M, K]) (vy : Tile .int [K, N])
    (hx : evalOp x t = some vx) (hy : evalOp y t = some vy) :
    evalOp (Op.dotInt (batch := []) x y) t = some (Tile.dotInt [] vx vy) := by
  erw [evalOp_dotInt, hx, hy]
  rfl

/-- A masked-with-`other` `.ptr` load, fully general. -/
private theorem is_load_ptr_maskOther {dtype : TileDType} {sh : TileShape}
    (nm : RegName) (maskOp : Op .bool sh) (otherOp : Op dtype sh)
    (t : BlockState) (pt : Tile .ptr sh) (masks : Tile .bool sh)
    (others : Tile dtype sh)
    (hp : t.regs .ptr sh nm = some pt)
    (hm : evalOp maskOp t = some masks)
    (ho : evalOp otherOp t = some others) :
    evalOp (Op.load dtype (MemAccess.ptr (Op.ref .ptr sh nm))
        (MaskOpt.maskOther maskOp otherOp)) t
      = some (⟨fun i => if masks.data i
            then t.readMemValue dtype (pt.data i).1 (pt.data i).2
            else others.data i⟩ : Tile dtype sh) := by
  simp only [evalOp, evalOp_ref, hp, hm, ho]
  rfl

/-- An unmasked `.ptr` load: every lane reads memory. -/
private theorem is_load_ptr_plain {dtype : TileDType} {sh : TileShape}
    (nm : RegName) (t : BlockState) (pt : Tile .ptr sh)
    (hp : t.regs .ptr sh nm = some pt) :
    evalOp (Op.load dtype (MemAccess.ptr (Op.ref .ptr sh nm)) MaskOpt.none) t
      = some (⟨fun i => t.readMemValue dtype (pt.data i).1 (pt.data i).2⟩
          : Tile dtype sh) := by
  simp only [evalOp, evalOp_ref, hp]
  rfl

private theorem is_broadcastInt_eval (sh : TileShape) (t : BlockState) :
    evalOp ((Op.constInt (0 : Int)).broadcast sh) t
      = some (⟨fun _ => (0 : ℤ)⟩ : Tile .int sh) := by
  simp only [evalOp]
  rfl

/-! ### `nat` scalar shapes -/

private theorem is_mulScalarNat_eval (x y : Op .nat []) (t : BlockState) (u v : Nat)
    (hx : evalOp x t = some (Tile.scalar u))
    (hy : evalOp y t = some (Tile.scalar v)) :
    evalOp (Op.mul .nat Broadcast.nil x y) t = some (Tile.scalar (u * v)) := by
  rw [evalOp_mul, hx, hy]
  rfl

private theorem is_addScalarNat_eval (x y : Op .nat []) (t : BlockState) (u v : Nat)
    (hx : evalOp x t = some (Tile.scalar u))
    (hy : evalOp y t = some (Tile.scalar v)) :
    evalOp (Op.add .nat Broadcast.nil x y) t = some (Tile.scalar (u + v)) := by
  rw [evalOp_add, hx, hy]
  rfl

private theorem is_subScalarNat_eval (x y : Op .nat []) (t : BlockState) (u v : Nat)
    (hx : evalOp x t = some (Tile.scalar u))
    (hy : evalOp y t = some (Tile.scalar v)) :
    evalOp (Op.sub .nat Broadcast.nil x y) t = some (Tile.scalar (u - v)) := by
  rw [evalOp_sub, hx, hy]
  rfl

private theorem is_divScalarNat_eval (x y : Op .nat []) (t : BlockState) (u v : Nat)
    (hx : evalOp x t = some (Tile.scalar u))
    (hy : evalOp y t = some (Tile.scalar v)) :
    evalOp (Op.div .nat Broadcast.nil x y) t = some (Tile.scalar (u / v)) := by
  rw [is_divTile_eval NumericDType.nat Broadcast.nil x y t _ _ hx hy]
  rfl

private theorem is_modScalarNat_eval (x y : Op .nat []) (t : BlockState) (u v : Nat)
    (hx : evalOp x t = some (Tile.scalar u))
    (hy : evalOp y t = some (Tile.scalar v)) :
    evalOp (Op.mod IntegralDType.nat Broadcast.nil x y) t
      = some (Tile.scalar (u % v)) := by
  rw [is_mod_eval Broadcast.nil x y t _ _ hx hy]
  rfl

private theorem is_floorDivScalar_eval (x y : Op .nat []) (t : BlockState)
    (u v : Nat) (hx : evalOp x t = some (Tile.scalar u))
    (hy : evalOp y t = some (Tile.scalar v)) :
    evalOp (Op.floorDiv IntegralDType.nat Broadcast.nil x y) t
      = some (Tile.scalar (u / v)) := by
  rw [is_floorDiv_eval Broadcast.nil x y t _ _ hx hy]
  rfl

private theorem is_ltScalarNat_eval (x y : Op .nat []) (t : BlockState) (u v : Nat)
    (hx : evalOp x t = some (Tile.scalar u))
    (hy : evalOp y t = some (Tile.scalar v)) :
    evalOp (Op.lt ComparableDType.nat Broadcast.nil x y) t
      = some (Tile.scalar (decide (u < v))) := by
  rw [is_ltTile_eval ComparableDType.nat Broadcast.nil x y t _ _ hx hy]
  rfl

private theorem is_whereScalarNat_eval (c : Op .bool []) (x y : Op .nat [])
    (t : BlockState) (cv : Bool) (u v : Nat)
    (hc : evalOp c t = some (Tile.scalar cv))
    (hx : evalOp x t = some (Tile.scalar u))
    (hy : evalOp y t = some (Tile.scalar v)) :
    evalOp (Op.where c x y) t = some (Tile.scalar (if cv then u else v)) := by
  rw [is_where_eval c x y t _ _ _ hc hx hy]
  rfl

/-- `name * c` on a `nat` scalar register. -/
private theorem is_mulRef_eval (t : BlockState) (nm : RegName) (val c : Nat)
    (hr : t.regs .nat [] nm = some (Tile.scalar val)) :
    evalOp (Op.mul .nat Broadcast.nil (Op.ref .nat [] nm) (Op.constNat c)) t
      = some (Tile.scalar (val * c)) := by
  rw [evalOp_mul]
  simp only [evalOp_ref, evalOp_constNat, hr, Option.bind_eq_bind, Option.bind_some]
  rfl

/-- `min` is spelled as a `tl.where` by the DSL. -/
private theorem is_min_as_where (u v : Nat) :
    (if u < v then u else v) = min u v := by
  rcases Nat.lt_or_ge u v with h | h
  · rw [if_pos h]; omega
  · rw [if_neg (by omega)]; omega

/-- `setReg` leaves memory alone, at **function** level. -/
private theorem is_setReg_mem {dtype : TileDType} {sh : TileShape}
    (s : BlockState) (nm : RegName) (v : Tile dtype sh) :
    (s.setReg nm dtype sh v).mem = s.mem := rfl

/-- `tl.cdiv` (`Op.div` over `(X + BX - 1)`). -/
private theorem is_cdiv_eval (t : BlockState) (X BX : Nat) :
    evalOp (Op.div .nat Broadcast.nil
        (Op.sub .nat Broadcast.nil
          (Op.add .nat Broadcast.nil (Op.constNat X) (Op.constNat BX)) (Op.constNat 1))
        (Op.constNat BX)) t
      = some (Tile.scalar ((X + BX - 1) / BX)) :=
  is_divScalarNat_eval _ _ t (X + BX - 1) BX
    (is_subScalarNat_eval _ _ t (X + BX) 1
      (is_addScalarNat_eval _ _ t X BX (evalOp_constNat _ _) (evalOp_constNat _ _))
      (evalOp_constNat _ _))
    (evalOp_constNat _ _)

/-- `(X + BX - 1) // BX` (kernel 2's inline `grid_*` spelling — `Op.floorDiv`). -/
private theorem is_gridDiv_eval (t : BlockState) (X BX : Nat) :
    evalOp (Op.floorDiv IntegralDType.nat Broadcast.nil
        (Op.sub .nat Broadcast.nil
          (Op.add .nat Broadcast.nil (Op.constNat X) (Op.constNat BX)) (Op.constNat 1))
        (Op.constNat BX)) t
      = some (Tile.scalar ((X + BX - 1) / BX)) :=
  is_floorDivScalar_eval _ _ t (X + BX - 1) BX
    (is_subScalarNat_eval _ _ t (X + BX) 1
      (is_addScalarNat_eval _ _ t X BX (evalOp_constNat _ _) (evalOp_constNat _ _))
      (evalOp_constNat _ _))
    (evalOp_constNat _ _)

/-- `Stmt.ifThenElse` on a closed `Bool` guard picks the arm. -/
private theorem is_ifThenElse_step (cond : Bool) (tb eb : List Stmt)
    (t : BlockState) :
    stepStmt (Stmt.ifThenElse (Op.constBool cond) tb eb) t
      = if cond then stepStmts tb t else stepStmts eb t := by
  unfold stepStmt
  have hc : evalOp (Op.constBool cond) t = some (Tile.scalar cond) := by
    simp [evalOp]
  rw [hc]
  cases cond <;> rfl

/-! ### `MemCell`-level scatter machinery -/

private theorem is_writeMemTyped_mem_other {dtype : TileDType} (s : BlockState)
    (rg : RegionName) (a : Nat) (v : TileCarrier dtype) (ρ : RegionName) (o : Nat)
    (h : ¬(rg = ρ ∧ a = o)) :
    (s.writeMemTyped dtype rg a v).mem ρ o = s.mem ρ o := by
  cases dtype <;>
    · show (if ρ = rg ∧ o = a then _ else s.mem ρ o) = s.mem ρ o
      rw [if_neg (fun hc => h ⟨hc.1.symm, hc.2.symm⟩)]

private theorem is_writeMemTyped_int_mem_self (s : BlockState)
    (rg : RegionName) (a : Nat) (v : Int) :
    (s.writeMemTyped .int rg a v).mem rg a = MemCell.of .int v := by
  show (if rg = rg ∧ a = a then MemCell.of .int v else s.mem rg a)
    = MemCell.of .int v
  rw [if_pos ⟨rfl, rfl⟩]

private theorem is_writeMemTyped_real_mem_self (s : BlockState)
    (rg : RegionName) (a : Nat) (v : TileCarrier .real) :
    (s.writeMemTyped .real rg a v).mem rg a
      = MemCell.of .real (FloatDType.real.ofReal (FloatDType.real.storeValue v)) := by
  show (if rg = rg ∧ a = a then
      MemCell.of .real (FloatDType.real.ofReal (FloatDType.real.storeValue v))
    else s.mem rg a) = _
  rw [if_pos ⟨rfl, rfl⟩]

/-- Cell-level frame for a masked scatter foldl: every cell missed by all
active writes is unchanged. -/
private theorem is_foldl_write_mem_preserve {dtype : TileDType} {α : Type}
    (region : RegionName) (offsetFn : α → Nat) (valueFn : α → TileCarrier dtype)
    (P : α → Prop) [DecidablePred P] (ρ : RegionName) (o : Nat) (l : List α) :
    ∀ s : BlockState, (∀ k ∈ l, P k → ¬(region = ρ ∧ offsetFn k = o)) →
      ((l.foldl (fun acc k =>
          if P k then acc.writeMemTyped dtype region (offsetFn k) (valueFn k)
          else acc) s)).mem ρ o = s.mem ρ o := by
  induction l with
  | nil => intro s _; rfl
  | cons hd tl ih =>
      intro s h
      rw [List.foldl_cons]
      have htl : ∀ k ∈ tl, P k → ¬(region = ρ ∧ offsetFn k = o) :=
        fun k hk => h k (List.mem_cons_of_mem hd hk)
      by_cases hP : P hd
      · rw [if_pos hP, ih _ htl,
          is_writeMemTyped_mem_other _ _ _ _ _ _ (h hd List.mem_cons_self hP)]
      · rw [if_neg hP]
        exact ih _ htl

/-- `.int` scatter readback with mask-restricted injectivity. -/
private theorem is_scatter_int_mem {region : RegionName} {shape : TileShape}
    (s : BlockState) (offsetFn : TileIndex shape → Nat)
    (valueFn : TileIndex shape → Int) (P : TileIndex shape → Prop)
    [DecidablePred P]
    (h_inj : ∀ k₁ k₂, P k₁ → P k₂ → offsetFn k₁ = offsetFn k₂ → k₁ = k₂)
    (i : TileIndex shape) (hPi : P i) :
    ((TileShape.allIndices shape).foldl
       (fun acc k =>
         if P k then acc.writeMemTyped .int region (offsetFn k) (valueFn k)
         else acc) s).mem region (offsetFn i)
      = MemCell.of .int (valueFn i) := by
  obtain ⟨l₁, l₂, hl⟩ := List.append_of_mem (TileShape.mem_allIndices shape i)
  have h_nodup := TileShape.allIndices_nodup shape
  rw [hl] at h_nodup
  rw [List.nodup_append, List.nodup_cons] at h_nodup
  obtain ⟨_, ⟨hi_notin_l2, _⟩, hl1_disj⟩ := h_nodup
  rw [show TileShape.allIndices shape = l₁ ++ i :: l₂ from hl, List.foldl_append,
    List.foldl_cons]
  have h_l2 : ∀ k ∈ l₂, P k → ¬(region = region ∧ offsetFn k = offsetFn i) := by
    intro k hk hPk hc
    have hki : k = i := h_inj k i hPk hPi hc.2
    subst hki
    exact hi_notin_l2 hk
  rw [is_foldl_write_mem_preserve (dtype := .int) region offsetFn valueFn P
    region (offsetFn i) l₂ _ h_l2]
  rw [if_pos hPi]
  exact is_writeMemTyped_int_mem_self _ _ _ _

/-- `.real` scatter readback with mask-restricted injectivity (kernel 2's
flat store: the address map is injective only on the masked lanes). -/
private theorem is_scatter_real_mem {region : RegionName} {shape : TileShape}
    (s : BlockState) (offsetFn : TileIndex shape → Nat)
    (valueFn : TileIndex shape → TileCarrier .real) (P : TileIndex shape → Prop)
    [DecidablePred P]
    (h_inj : ∀ k₁ k₂, P k₁ → P k₂ → offsetFn k₁ = offsetFn k₂ → k₁ = k₂)
    (i : TileIndex shape) (hPi : P i) :
    ((TileShape.allIndices shape).foldl
       (fun acc k =>
         if P k then acc.writeMemTyped .real region (offsetFn k) (valueFn k)
         else acc) s).mem region (offsetFn i)
      = MemCell.of .real
          (FloatDType.real.ofReal (FloatDType.real.storeValue (valueFn i))) := by
  obtain ⟨l₁, l₂, hl⟩ := List.append_of_mem (TileShape.mem_allIndices shape i)
  have h_nodup := TileShape.allIndices_nodup shape
  rw [hl] at h_nodup
  rw [List.nodup_append, List.nodup_cons] at h_nodup
  obtain ⟨_, ⟨hi_notin_l2, _⟩, hl1_disj⟩ := h_nodup
  rw [show TileShape.allIndices shape = l₁ ++ i :: l₂ from hl, List.foldl_append,
    List.foldl_cons]
  have h_l2 : ∀ k ∈ l₂, P k → ¬(region = region ∧ offsetFn k = offsetFn i) := by
    intro k hk hPk hc
    have hki : k = i := h_inj k i hPk hPi hc.2
    subst hki
    exact hi_notin_l2 hk
  rw [is_foldl_write_mem_preserve (dtype := .real) region offsetFn valueFn P
    region (offsetFn i) l₂ _ h_l2]
  rw [if_pos hPi]
  exact is_writeMemTyped_real_mem_self _ _ _ _

end SharedRecipes

/-! ## Kernel 1 — the group-swizzled block coordinates

`pid` is decomposed exactly as the source decomposes it — including the
constexpr-name rebind `GROUP_M = min(num_pid_m - first_pid_m, GROUP_M)` — so
the spec is stated in the kernel's own coordinates. -/

/-- `num_pid_m = tl.cdiv(M, BLOCK_M)`. -/
def mmNumPidM (M BM : Nat) : Nat := (M + BM - 1) / BM

/-- `num_pid_n = tl.cdiv(N, BLOCK_N)`. -/
def mmNumPidN (N BN : Nat) : Nat := (N + BN - 1) / BN

/-- `num_pid_in_group = GROUP_M * num_pid_n`. -/
def mmNumPidInGroup (N BN GM : Nat) : Nat := GM * mmNumPidN N BN

/-- `group_id = pid // num_pid_in_group`. -/
def mmGroupId (s : BlockState) (N BN GM : Nat) : Nat :=
  s.pids 0 / mmNumPidInGroup N BN GM

/-- `first_pid_m = group_id * GROUP_M`. -/
def mmFirstPidM (s : BlockState) (N BN GM : Nat) : Nat :=
  mmGroupId s N BN GM * GM

/-- The rebound register `GROUP_M = min(num_pid_m - first_pid_m, GROUP_M)`. -/
def mmGroupSize (s : BlockState) (M N BM BN GM : Nat) : Nat :=
  min (mmNumPidM M BM - mmFirstPidM s N BN GM) GM

/-- `pid_m = first_pid_m + (pid % GROUP_M)` — `GROUP_M` here is the rebound
register (the group size), not the parameter. -/
def mmPidM (s : BlockState) (M N BM BN GM : Nat) : Nat :=
  mmFirstPidM s N BN GM + s.pids 0 % mmGroupSize s M N BM BN GM

/-- `pid_n = (pid % num_pid_in_group) // GROUP_M` (the rebound register). -/
def mmPidN (s : BlockState) (M N BM BN GM : Nat) : Nat :=
  s.pids 0 % mmNumPidInGroup N BN GM / mmGroupSize s M N BM BN GM

/-! ## Kernel 1 — element accessors and the GEMM spec -/

/-- `A[row, kk]` — a signed `.int`-channel read of the launch state (the
host's `torch.int8` input; widths erased, kernel-1 disclosure (1)). -/
def mmAElem (s : BlockState) (a_ptr : Region .int) (sam sak : Nat)
    (row kk : Nat) : ℤ :=
  s.readMemValue .int (Region.cast a_ptr) (row * sam + kk * sak)

/-- `B[kk, col]` — same on `B`. -/
def mmBElem (s : BlockState) (b_ptr : Region .int) (sbk sbn : Nat)
    (kk col : Nat) : ℤ :=
  s.readMemValue .int (Region.cast b_ptr) (kk * sbk + col * sbn)

/-- One K-lane's ℤ contribution to output cell `(row, col)`, **as the
boundary-checked loads produce it**: each factor is zero-filled whenever its
own `(0, 1)` boundary check fails. On in-range output cells the guards
reduce to `kk < K`. -/
def mmG (s : BlockState) (a_ptr b_ptr : Region .int)
    (M N K sam sak sbk sbn : Nat) (row col kk : Nat) : ℤ :=
  (if row < M ∧ kk < K then mmAElem s a_ptr sam sak row kk else 0)
    * (if kk < K ∧ col < N then mmBElem s b_ptr sbk sbn kk col else 0)

/-- The running accumulator after the loop counter has reached `c`: the sum
of all `c` swept K-lanes (out-of-range lanes contribute `0`). -/
def mmAccVal (s : BlockState) (a_ptr b_ptr : Region .int)
    (M N K sam sak sbk sbn : Nat) (row col c : Nat) : ℤ :=
  ∑ kk ∈ Finset.range c, mmG s a_ptr b_ptr M N K sam sak sbk sbn row col kk

/-- **The kernel-1 stored value**: the genuine ℤ GEMM
`∑ kk < K, A[row, kk] · B[kk, col]`. -/
def mmSpec (s : BlockState) (a_ptr b_ptr : Region .int)
    (K sam sak sbk sbn : Nat) (row col : Nat) : ℤ :=
  ∑ kk : Fin K, mmAElem s a_ptr sam sak row kk.val
    * mmBElem s b_ptr sbk sbn kk.val col

theorem mmAccVal_zero (s : BlockState) (a_ptr b_ptr : Region .int)
    (M N K sam sak sbk sbn row col : Nat) :
    mmAccVal s a_ptr b_ptr M N K sam sak sbk sbn row col 0 = 0 := by
  simp [mmAccVal]

/-- One loop iteration (counter `c`, one `BK`-wide slab of K-lanes) extends
the running sum. -/
theorem mmAccVal_step (s : BlockState) (a_ptr b_ptr : Region .int)
    (M N K sam sak sbk sbn row col c BK : Nat) :
    mmAccVal s a_ptr b_ptr M N K sam sak sbk sbn row col c
      + ∑ e : Fin BK, mmG s a_ptr b_ptr M N K sam sak sbk sbn row col (c + e.val)
      = mmAccVal s a_ptr b_ptr M N K sam sak sbk sbn row col (c + BK) := by
  unfold mmAccVal
  rw [Finset.sum_range_add]
  congr 1
  rw [Fin.sum_univ_eq_sum_range (fun e =>
    mmG s a_ptr b_ptr M N K sam sak sbk sbn row col (c + e)) BK]

/-- At any final counter `F ≥ K` and any in-range output cell, the guarded
running sum **is** the genuine GEMM: lanes `kk ≥ K` were zero-filled by the
boundary check, and the `row < M` / `col < N` guards hold. -/
theorem mmAccVal_final (s : BlockState) (a_ptr b_ptr : Region .int)
    (M N K sam sak sbk sbn row col F : Nat)
    (hrow : row < M) (hcol : col < N) (hKF : K ≤ F) :
    mmAccVal s a_ptr b_ptr M N K sam sak sbk sbn row col F
      = mmSpec s a_ptr b_ptr K sam sak sbk sbn row col := by
  unfold mmAccVal mmSpec
  rw [Fin.sum_univ_eq_sum_range (fun kk =>
    mmAElem s a_ptr sam sak row kk * mmBElem s b_ptr sbk sbn kk col) K]
  have hsub : Finset.range K ⊆ Finset.range F := fun x hx =>
    Finset.mem_range.mpr (lt_of_lt_of_le (Finset.mem_range.mp hx) hKF)
  rw [← Finset.sum_subset hsub
    (fun kk _ hkk => by
      unfold mmG
      rw [if_neg (fun hc => hkk (Finset.mem_range.mpr hc.2)), zero_mul])]
  refine Finset.sum_congr rfl fun kk hkk => ?_
  have hk : kk < K := Finset.mem_range.mp hkk
  unfold mmG
  rw [if_pos ⟨hrow, hk⟩, if_pos ⟨hk, hcol⟩]

/-- The `C` store address for output lane `idx` — the `c_block_ptr` address
(base offset `0`, offsets `(pid_m·BM, pid_n·BN)`). -/
def mmCAddr (scm scn BM BN pm pn : Nat) (idx : TileIndex [BM, BN]) : Nat :=
  (pm * BM + idx.1.val) * scm + (pn * BN + idx.2.1.val) * scn

/-- `hInj` discharge for a row-major `C`: distinct output lanes get distinct
addresses when the column stride is positive and one block row fits inside
the row stride. -/
theorem mmCAddr_injective_rowMajor (scm scn BM BN pm pn : Nat)
    (hcn : 0 < scn) (hfit : BN * scn ≤ scm) :
    Function.Injective
      (fun i : TileIndex [BM, BN] => mmCAddr scm scn BM BN pm pn i) := by
  intro i j hij
  obtain ⟨r₁, c₁, u₁⟩ := i
  obtain ⟨r₂, c₂, u₂⟩ := j
  simp only [mmCAddr] at hij
  have hexp : ∀ r c : Nat, (pm * BM + r) * scm + (pn * BN + c) * scn
      = pm * BM * scm + r * scm + (pn * BN * scn + c * scn) := by
    intro r c
    ring
  rw [hexp, hexp] at hij
  have key : r₁.val * scm + c₁.val * scn = r₂.val * scm + c₂.val * scn := by
    omega
  have hrow : ∀ c : Fin BN, c.val * scn < scm := fun c =>
    lt_of_lt_of_le (Nat.mul_lt_mul_of_lt_of_le c.isLt (le_refl _) hcn) hfit
  have hr : r₁.val = r₂.val := by
    rcases Nat.lt_trichotomy r₁.val r₂.val with h | h | h
    · have : r₁.val * scm + scm ≤ r₂.val * scm := by
        rw [← Nat.succ_mul]
        exact Nat.mul_le_mul_right _ h
      have := hrow c₁
      omega
    · exact h
    · have : r₂.val * scm + scm ≤ r₁.val * scm := by
        rw [← Nat.succ_mul]
        exact Nat.mul_le_mul_right _ h
      have := hrow c₂
      omega
  have hc : c₁.val = c₂.val := by
    have : c₁.val * scn = c₂.val * scn := by
      rw [hr] at key; omega
    exact Nat.eq_of_mul_eq_mul_right hcn this
  simp only [Prod.mk.injEq]
  exact ⟨Fin.ext hr, Fin.ext hc, trivial⟩

/-! ## Kernel 1 — compiled body decomposition

Checked against the macro output by `rfl`. Lowerings worth naming:
`tl.cdiv` expands to `Op.div .nat` (while the `//` operator is
`Op.floorDiv`); `min(a, b)` is `Op.where (Op.lt …) a b`; the rebound
`GROUP_M` is an ordinary `.nat` register named `"GROUP_M"`; the two
`tl.make_block_ptr` with runtime offsets lower to
`Op.makeBlockPtrDynOffsets` with per-axis offset *ops*; the
`boundary_check=(0, 1)` loads are `.int`-typed `MemAccess.blockPtr … [0, 1]`
accesses with `MaskOpt.none` (zero-fill is the block-ptr semantics, not a
mask); `tl.advance` deltas are `Int`-cast; the store is the `.int`-typed
boundary-checked block-ptr store. -/

/-- The prologue: nine swizzle scalars, the two input block pointers, and
the zeroed `.int` accumulator. -/
def mmPreLoop (a_ptr b_ptr : Region .int)
    (M N K sam sak sbk sbn BM BN BK GM : Nat) : List Stmt :=
  [ Stmt.assign .nat [] "pid" (Op.programId 0),
    Stmt.assign .nat [] "num_pid_m"
      (Op.div .nat Broadcast.nil
        (Op.sub .nat Broadcast.nil
          (Op.add .nat Broadcast.nil (Op.constNat M) (Op.constNat BM)) (Op.constNat 1))
        (Op.constNat BM)),
    Stmt.assign .nat [] "num_pid_n"
      (Op.div .nat Broadcast.nil
        (Op.sub .nat Broadcast.nil
          (Op.add .nat Broadcast.nil (Op.constNat N) (Op.constNat BN)) (Op.constNat 1))
        (Op.constNat BN)),
    Stmt.assign .nat [] "num_pid_in_group"
      (Op.mul .nat Broadcast.nil (Op.constNat GM) (Op.ref .nat [] "num_pid_n")),
    Stmt.assign .nat [] "group_id"
      (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.ref .nat [] "pid")
        (Op.ref .nat [] "num_pid_in_group")),
    Stmt.assign .nat [] "first_pid_m"
      (Op.mul .nat Broadcast.nil (Op.ref .nat [] "group_id") (Op.constNat GM)),
    Stmt.assign .nat [] "GROUP_M"
      (Op.where
        (Op.lt ComparableDType.nat Broadcast.nil
          (Op.sub .nat Broadcast.nil (Op.ref .nat [] "num_pid_m")
            (Op.ref .nat [] "first_pid_m"))
          (Op.constNat GM))
        (Op.sub .nat Broadcast.nil (Op.ref .nat [] "num_pid_m")
          (Op.ref .nat [] "first_pid_m"))
        (Op.constNat GM)),
    Stmt.assign .nat [] "pid_m"
      (Op.add .nat Broadcast.nil (Op.ref .nat [] "first_pid_m")
        (Op.mod IntegralDType.nat Broadcast.nil (Op.ref .nat [] "pid")
          (Op.ref .nat [] "GROUP_M"))),
    Stmt.assign .nat [] "pid_n"
      (Op.floorDiv IntegralDType.nat Broadcast.nil
        (Op.mod IntegralDType.nat Broadcast.nil (Op.ref .nat [] "pid")
          (Op.ref .nat [] "num_pid_in_group"))
        (Op.ref .nat [] "GROUP_M")),
    Stmt.assign .blockPtr [BM, BK] "a_block_ptr"
      (Op.makeBlockPtrDynOffsets (Region.cast a_ptr) (Op.constNat 0) [M, K] [BM, BK] [sam, sak]
        [Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_m") (Op.constNat BM),
         Op.constNat 0]),
    Stmt.assign .blockPtr [BK, BN] "b_block_ptr"
      (Op.makeBlockPtrDynOffsets (Region.cast b_ptr) (Op.constNat 0) [K, N] [BK, BN] [sbk, sbn]
        [Op.constNat 0,
         Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_n") (Op.constNat BN)]),
    Stmt.assign .int [BM, BN] "accumulator" (Op.full [BM, BN] (Op.constInt 0)) ]

/-- The loop body: two boundary-checked `.int` block-ptr loads, the
`Op.dotInt` accumulation, and the two `tl.advance`s. -/
def mmLoopBody (BM BN BK : Nat) : List Stmt :=
  [ Stmt.assign .int [BM, BK] "a"
      (Op.load .int
        (MemAccess.blockPtr (Op.ref .blockPtr [BM, BK] "a_block_ptr") [0, 1])
        MaskOpt.none),
    Stmt.assign .int [BK, BN] "b"
      (Op.load .int
        (MemAccess.blockPtr (Op.ref .blockPtr [BK, BN] "b_block_ptr") [0, 1])
        MaskOpt.none),
    Stmt.assign .int [BM, BN] "accumulator"
      (Op.add .int (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .int [BM, BN] "accumulator")
        (Op.dotInt (batch := []) (Op.ref .int [BM, BK] "a")
          (Op.ref .int [BK, BN] "b"))),
    Stmt.assign .blockPtr [BM, BK] "a_block_ptr"
      (Op.advanceBlockPtr (Op.ref .blockPtr [BM, BK] "a_block_ptr")
        [((0 : Nat) : Int), ((BK : Nat) : Int)]),
    Stmt.assign .blockPtr [BK, BN] "b_block_ptr"
      (Op.advanceBlockPtr (Op.ref .blockPtr [BK, BN] "b_block_ptr")
        [((BK : Nat) : Int), ((0 : Nat) : Int)]) ]

/-- The tail: the bare `c = accumulator` re-assign, the output block
pointer, and the boundary-checked `.int` store. -/
def mmPostLoop (c_ptr : RegionName) (M N scm scn BM BN : Nat) : List Stmt :=
  [ Stmt.assign .int [BM, BN] "c" (Op.ref .int [BM, BN] "accumulator"),
    Stmt.assign .blockPtr [BM, BN] "c_block_ptr"
      (Op.makeBlockPtrDynOffsets c_ptr (Op.constNat 0) [M, N] [BM, BN] [scm, scn]
        [Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_m") (Op.constNat BM),
         Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_n") (Op.constNat BN)]),
    Stmt.store .int [BM, BN]
      (MemAccess.blockPtr (Op.ref .blockPtr [BM, BN] "c_block_ptr") [0, 1])
      (Op.ref .int [BM, BN] "c") MaskOpt.none ]

set_option maxRecDepth 20000 in
/-- **Kernel-1 body split (by `rfl`).** 13 top-level statements. -/
theorem mm_body_eq (a_ptr b_ptr : Region .int) (c_ptr : RegionName)
    (M N K sam sak sbk sbn scm scn BM BN BK GM : Nat) :
    (int_scaled_matmul_matmul_surface a_ptr b_ptr c_ptr M N K
        sam sak sbk sbn scm scn BM BN BK GM).toAlgKernel.body
      = mmPreLoop a_ptr b_ptr M N K sam sak sbk sbn BM BN BK GM
        ++ [Stmt.forRange "k" 0 K BK (mmLoopBody BM BN BK)]
        ++ mmPostLoop c_ptr M N scm scn BM BN := by
  rfl

/-! ## Kernel 1 — block-pointer tiles and eval recipes -/

/-- The constant block-pointer tile every lane of a `tl.make_block_ptr`
result carries (base offset `0`). -/
def mmBPTile (R : RegionName) (pR pC sR sC BR BC o0 o1 : Nat) :
    Tile .blockPtr [BR, BC] :=
  ⟨fun _ => { region := R, baseOffset := 0, parentShape := [pR, pC],
              blockShape := [BR, BC], strides := [sR, sC],
              offsets := [o0, o1] }⟩

/-- `tl.make_block_ptr` with two dynamic offset ops. -/
private theorem mm_makeBP_eval {BR BC : Nat} (R : RegionName) (t : BlockState)
    (pR pC sR sC : Nat) (off0 off1 : Op .nat []) (v0 v1 : Nat)
    (h0 : evalOp off0 t = some (Tile.scalar v0))
    (h1 : evalOp off1 t = some (Tile.scalar v1)) :
    evalOp (Op.makeBlockPtrDynOffsets R (Op.constNat 0) [pR, pC]
        ([BR, BC] : TileShape) [sR, sC] [off0, off1]) t
      = some (mmBPTile R pR pC sR sC BR BC v0 v1) := by
  rw [makeBlockPtr2_eval]
  simp only [evalOp_constNat, h0, h1, List.mapM_cons, List.mapM_nil,
    Option.bind_some, Option.pure_def, Option.bind_eq_bind]
  rfl

/-- A boundary-checked `.int` block-ptr load: in-bounds lanes read the
`.int` channel at `(o0+i)·sR + (o1+j)·sC`, out-of-bounds lanes zero-fill. -/
private theorem mm_load_int_eval {BR BC : Nat} (nm : RegName) (t : BlockState)
    (R : RegionName) (pR pC sR sC o0 o1 : Nat)
    (hbp : t.regs .blockPtr [BR, BC] nm = some (mmBPTile R pR pC sR sC BR BC o0 o1)) :
    evalOp (Op.load .int (MemAccess.blockPtr (Op.ref .blockPtr [BR, BC] nm) [0, 1])
        MaskOpt.none) t
      = some (⟨fun idx => if o0 + idx.1.val < pR ∧ o1 + idx.2.1.val < pC
            then t.readMemValue .int R
              ((o0 + idx.1.val) * sR + (o1 + idx.2.1.val) * sC)
            else 0⟩ : Tile .int [BR, BC]) := by
  simp only [evalOp, evalOp_ref, hbp, Option.bind_some, Option.bind, Option.map]
  refine congrArg some ?_
  ext idx
  simp only [mmBPTile, TileShape.blockPtr_inBounds_2d_offsets_index,
    TileShape.blockPtr_address_2d_offsets_index, Nat.zero_add,
    decide_eq_true_eq]
  rfl

/-- One `tl.advance` of a block-pointer register by `Nat`-cast deltas. -/
private theorem mm_advance_eval {BR BC : Nat} (nm : RegName) (t : BlockState)
    (R : RegionName) (pR pC sR sC o0 o1 d0 d1 : Nat)
    (hbp : t.regs .blockPtr [BR, BC] nm = some (mmBPTile R pR pC sR sC BR BC o0 o1)) :
    evalOp (Op.advanceBlockPtr (Op.ref .blockPtr [BR, BC] nm)
        [((d0 : Nat) : Int), ((d1 : Nat) : Int)]) t
      = some (mmBPTile R pR pC sR sC BR BC (o0 + d0) (o1 + d1)) := by
  rw [advanceBlockPtr_eval]
  simp only [evalOp_ref, hbp, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  simp only [mmBPTile, BlockPtr.advance_2d_offsets]

/-! ## Kernel 1 — loaded tiles and the accumulator tile -/

/-- The loaded `a` tile at loop counter `c`: the boundary check zero-fills
every lane outside the `M × K` parent. -/
def mmATile (s : BlockState) (a_ptr : Region .int)
    (M K sam sak BM BK pm c : Nat) : Tile .int [BM, BK] :=
  ⟨fun idx => if pm * BM + idx.1.val < M ∧ c + idx.2.1.val < K
      then mmAElem s a_ptr sam sak (pm * BM + idx.1.val) (c + idx.2.1.val)
      else 0⟩

/-- The loaded `b` tile at loop counter `c`. -/
def mmBTile (s : BlockState) (b_ptr : Region .int)
    (K N sbk sbn BK BN pn c : Nat) : Tile .int [BK, BN] :=
  ⟨fun idx => if c + idx.1.val < K ∧ pn * BN + idx.2.1.val < N
      then mmBElem s b_ptr sbk sbn (c + idx.1.val) (pn * BN + idx.2.1.val)
      else 0⟩

/-- `accumulator` at loop counter `c`. -/
def mmAccTile (s : BlockState) (a_ptr b_ptr : Region .int)
    (M N K sam sak sbk sbn BM BN pm pn c : Nat) : Tile .int [BM, BN] :=
  ⟨fun idx => mmAccVal s a_ptr b_ptr M N K sam sak sbk sbn
      (pm * BM + idx.1.val) (pn * BN + idx.2.1.val) c⟩

theorem mmAccTile_zero (s : BlockState) (a_ptr b_ptr : Region .int)
    (M N K sam sak sbk sbn BM BN pm pn : Nat) :
    mmAccTile s a_ptr b_ptr M N K sam sak sbk sbn BM BN pm pn 0
      = (⟨fun _ => 0⟩ : Tile .int [BM, BN]) := by
  apply Tile.ext
  intro idx
  simp [mmAccTile, mmAccVal_zero]

/-- **The `Op.dotInt` accumulator step.** `accumulator += tl.dot(a, b)`
extends the running sum by one `BK`-slab: the per-lane product of the two
zero-filled loads is exactly the guarded summand `mmG`. -/
theorem mmAccTile_dotInt_step (s : BlockState) (a_ptr b_ptr : Region .int)
    (M N K sam sak sbk sbn BM BN BK pm pn c : Nat) :
    Tile.bop NumericDType.int.add
        (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (mmAccTile s a_ptr b_ptr M N K sam sak sbk sbn BM BN pm pn c)
        (Tile.dotInt [] (mmATile s a_ptr M K sam sak BM BK pm c)
          (mmBTile s b_ptr K N sbk sbn BK BN pn c))
      = mmAccTile s a_ptr b_ptr M N K sam sak sbk sbn BM BN pm pn (c + BK) := by
  apply Tile.ext
  intro idx
  obtain ⟨r, cc, u⟩ := idx
  simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex,
    mmAccTile, NumericDType.int_add]
  erw [Tile.dotInt_nil_data]
  rw [← mmAccVal_step s a_ptr b_ptr M N K sam sak sbk sbn
    (pm * BM + r.val) (pn * BN + cc.val) c BK]
  -- the per-lane product of the two zero-filled loads is *definitionally*
  -- the guarded summand `mmG`, so `congr 1` closes the sum.
  congr 1

/-! ## Kernel 1 — the loop invariant and walks -/

/-- The state carried across loop iterations (`c` is the raw counter — a
multiple of `BK`). -/
noncomputable def mmInv (s0 : BlockState) (a_ptr b_ptr : Region .int)
    (M N K sam sak sbk sbn BM BN BK GM : Nat) (c : Nat) (s : BlockState) : Prop :=
  s.mem = s0.mem
  ∧ s.pids = s0.pids
  ∧ s.regs .nat [] "pid_m" = some (Tile.scalar (mmPidM s0 M N BM BN GM))
  ∧ s.regs .nat [] "pid_n" = some (Tile.scalar (mmPidN s0 M N BM BN GM))
  ∧ s.regs .blockPtr [BM, BK] "a_block_ptr"
      = some (mmBPTile (Region.cast a_ptr) M K sam sak BM BK (mmPidM s0 M N BM BN GM * BM) c)
  ∧ s.regs .blockPtr [BK, BN] "b_block_ptr"
      = some (mmBPTile (Region.cast b_ptr) K N sbk sbn BK BN c (mmPidN s0 M N BM BN GM * BN))
  ∧ s.regs .int [BM, BN] "accumulator"
      = some (mmAccTile s0 a_ptr b_ptr M N K sam sak sbk sbn BM BN
          (mmPidM s0 M N BM BN GM) (mmPidN s0 M N BM BN GM) c)

/-- The loop combinator writes the counter register `"k"` (unused by the
body); `mmInv` constrains no register named `"k"`. -/
theorem mmInv_setReg_k (s0 : BlockState) (a_ptr b_ptr : Region .int)
    (M N K sam sak sbk sbn BM BN BK GM c v : Nat) (s : BlockState)
    (h : mmInv s0 a_ptr b_ptr M N K sam sak sbk sbn BM BN BK GM c s) :
    mmInv s0 a_ptr b_ptr M N K sam sak sbk sbn BM BN BK GM c
      (s.setReg "k" .nat [] (Tile.scalar v)) := by
  obtain ⟨hmem, hpids, hpm, hpn, hA, hB, hacc⟩ := h
  exact ⟨hmem, hpids, by simpa using hpm, by simpa using hpn,
    by simpa using hA, by simpa using hB, by simpa using hacc⟩

/-- One loop iteration preserves the invariant, advancing the counter by
`BK`. -/
theorem mmBody_run (s0 : BlockState) (a_ptr b_ptr : Region .int)
    (M N K sam sak sbk sbn BM BN BK GM c : Nat) (s : BlockState)
    (hinv : mmInv s0 a_ptr b_ptr M N K sam sak sbk sbn BM BN BK GM c s) :
    ∃ s', stepStmts (mmLoopBody BM BN BK) s = some s'
      ∧ mmInv s0 a_ptr b_ptr M N K sam sak sbk sbn BM BN BK GM (c + BK) s' := by
  obtain ⟨hmem, hpids, hpm, hpn, hA, hB, hacc⟩ := hinv
  set pm := mmPidM s0 M N BM BN GM with hpmDef
  set pn := mmPidN s0 M N BM BN GM with hpnDef
  unfold mmLoopBody
  -- 1. `a = tl.load(a_block_ptr, boundary_check=(0, 1))`
  have haLoad : evalOp (Op.load .int
      (MemAccess.blockPtr (Op.ref .blockPtr [BM, BK] "a_block_ptr") [0, 1])
      MaskOpt.none) s = some (mmATile s0 a_ptr M K sam sak BM BK pm c) := by
    rw [mm_load_int_eval "a_block_ptr" s (Region.cast a_ptr) M K sam sak (pm * BM) c hA]
    refine congrArg some ?_
    ext idx
    simp only [mmATile, mmAElem, BlockState.readMemValue, BlockState.readMemTyped,
      hmem]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some haLoad)]
  -- 2. `b = tl.load(b_block_ptr, boundary_check=(0, 1))`
  have hbLoad : evalOp (Op.load .int
      (MemAccess.blockPtr (Op.ref .blockPtr [BK, BN] "b_block_ptr") [0, 1])
      MaskOpt.none) (s.setReg "a" .int [BM, BK] (mmATile s0 a_ptr M K sam sak BM BK pm c))
      = some (mmBTile s0 b_ptr K N sbk sbn BK BN pn c) := by
    rw [mm_load_int_eval "b_block_ptr" _ (Region.cast b_ptr) K N sbk sbn c (pn * BN)
      (by simpa using hB)]
    refine congrArg some ?_
    ext idx
    simp only [mmBTile, mmBElem, BlockState.readMemValue, BlockState.readMemTyped,
      is_setReg_mem, hmem]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some hbLoad)]
  -- 3. `accumulator += tl.dot(a, b)`
  have haccStep : evalOp (Op.add .int
      (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
      (Op.ref .int [BM, BN] "accumulator")
      (Op.dotInt (batch := []) (Op.ref .int [BM, BK] "a")
        (Op.ref .int [BK, BN] "b")))
      ((s.setReg "a" .int [BM, BK] (mmATile s0 a_ptr M K sam sak BM BK pm c)).setReg
        "b" .int [BK, BN] (mmBTile s0 b_ptr K N sbk sbn BK BN pn c))
      = some (mmAccTile s0 a_ptr b_ptr M N K sam sak sbk sbn BM BN pm pn (c + BK)) := by
    rw [← mmAccTile_dotInt_step s0 a_ptr b_ptr M N K sam sak sbk sbn BM BN BK pm pn c]
    exact is_addTile_eval NumericDType.int _ _ _ _ _ _
      (by rw [evalOp_ref]; simpa using hacc)
      (is_dotInt_eval _ _ _ _ _
        (by rw [evalOp_ref]; simp) (by rw [evalOp_ref]; simp))
  rw [stepStmts.cons_some (stepStmt_assign_eq_some haccStep)]
  -- 4. `a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))`
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.advanceBlockPtr (Op.ref .blockPtr [BM, BK] "a_block_ptr")
        [((0 : Nat) : Int), ((BK : Nat) : Int)]) _
      = some (mmBPTile (Region.cast a_ptr) M K sam sak BM BK (pm * BM + 0) (c + BK)) from
      mm_advance_eval "a_block_ptr" _ (Region.cast a_ptr) M K sam sak (pm * BM) c 0 BK
        (by simpa using hA)))]
  -- 5. `b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))`
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.advanceBlockPtr (Op.ref .blockPtr [BK, BN] "b_block_ptr")
        [((BK : Nat) : Int), ((0 : Nat) : Int)]) _
      = some (mmBPTile (Region.cast b_ptr) K N sbk sbn BK BN (c + BK) (pn * BN + 0)) from
      mm_advance_eval "b_block_ptr" _ (Region.cast b_ptr) K N sbk sbn c (pn * BN) BK 0
        (by simpa using hB)))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp only [is_setReg_mem]
    exact hmem
  · simp only [BlockState.setReg_pids]
    exact hpids
  · simpa using hpm
  · simpa using hpn
  · simp [hpmDef]
  · simp [hpnDef]
  · simp [hpmDef, hpnDef]

/-- The collapsed K-loop (`forRange` at step `BLOCK_K`): the final counter
is the first multiple of `BK` at or above `K`. -/
theorem mmLoop_collapse (s0 : BlockState) (a_ptr b_ptr : Region .int)
    (M N K sam sak sbk sbn BM BN BK GM : Nat) (s : BlockState)
    (hBK : 0 < BK)
    (h0 : mmInv s0 a_ptr b_ptr M N K sam sak sbk sbn BM BN BK GM 0 s) :
    ∃ F sF, stepStmt (Stmt.forRange "k" 0 K BK (mmLoopBody BM BN BK)) s = some sF
      ∧ K ≤ F
      ∧ mmInv s0 a_ptr b_ptr M N K sam sak sbk sbn BM BN BK GM F sF := by
  obtain ⟨F, sF, hrun, hfinal, hP⟩ :=
    forRange_inv (idx := "k") (start := 0) (stop := K) (step := BK)
      (P := fun c t => mmInv s0 a_ptr b_ptr M N K sam sak sbk sbn BM BN BK GM c t)
      (Nat.pos_iff_ne_zero.mp hBK) h0
      (fun c t hc hinv => by
        obtain ⟨s', hs', hinv'⟩ :=
          mmBody_run s0 a_ptr b_ptr M N K sam sak sbk sbn BM BN BK GM c _
            (mmInv_setReg_k s0 a_ptr b_ptr M N K sam sak sbk sbn BM BN BK GM
              c c t hinv)
        exact ⟨s', hs', hinv'⟩)
  exact ⟨F, sF, hrun, hfinal, hP⟩

/-! ## Kernel 1 — the prologue walk -/

private theorem mm_width_eval (t : BlockState) (N BN GM : Nat)
    (hgn : t.regs .nat [] "num_pid_n" = some (Tile.scalar (mmNumPidN N BN))) :
    evalOp (Op.mul .nat Broadcast.nil (Op.constNat GM)
        (Op.ref .nat [] "num_pid_n")) t
      = some (Tile.scalar (mmNumPidInGroup N BN GM)) := by
  rw [is_mulScalarNat_eval _ _ t GM (mmNumPidN N BN) (evalOp_constNat _ _)
    (by rw [evalOp_ref]; exact hgn)]
  rfl

private theorem mm_groupId_eval (s t : BlockState) (N BN GM : Nat)
    (hpid : t.regs .nat [] "pid" = some (Tile.scalar (s.pids 0)))
    (hwid : t.regs .nat [] "num_pid_in_group"
      = some (Tile.scalar (mmNumPidInGroup N BN GM))) :
    evalOp (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.ref .nat [] "pid")
        (Op.ref .nat [] "num_pid_in_group")) t
      = some (Tile.scalar (mmGroupId s N BN GM)) := by
  rw [is_floorDivScalar_eval _ _ t (s.pids 0) (mmNumPidInGroup N BN GM)
    (by rw [evalOp_ref]; exact hpid) (by rw [evalOp_ref]; exact hwid)]
  rfl

private theorem mm_firstPidM_eval (s t : BlockState) (N BN GM : Nat)
    (hgid : t.regs .nat [] "group_id"
      = some (Tile.scalar (mmGroupId s N BN GM))) :
    evalOp (Op.mul .nat Broadcast.nil (Op.ref .nat [] "group_id")
        (Op.constNat GM)) t
      = some (Tile.scalar (mmFirstPidM s N BN GM)) := by
  rw [is_mulRef_eval t "group_id" (mmGroupId s N BN GM) GM hgid]
  rfl

/-- The `GROUP_M` register rebind: `min(num_pid_m - first_pid_m, GROUP_M)`. -/
private theorem mm_groupSize_eval (s t : BlockState) (M N BM BN GM : Nat)
    (hgm : t.regs .nat [] "num_pid_m" = some (Tile.scalar (mmNumPidM M BM)))
    (hfp : t.regs .nat [] "first_pid_m"
      = some (Tile.scalar (mmFirstPidM s N BN GM))) :
    evalOp (Op.where
        (Op.lt ComparableDType.nat Broadcast.nil
          (Op.sub .nat Broadcast.nil (Op.ref .nat [] "num_pid_m")
            (Op.ref .nat [] "first_pid_m"))
          (Op.constNat GM))
        (Op.sub .nat Broadcast.nil (Op.ref .nat [] "num_pid_m")
          (Op.ref .nat [] "first_pid_m"))
        (Op.constNat GM)) t
      = some (Tile.scalar (mmGroupSize s M N BM BN GM)) := by
  have hsub : evalOp (Op.sub .nat Broadcast.nil (Op.ref .nat [] "num_pid_m")
      (Op.ref .nat [] "first_pid_m")) t
      = some (Tile.scalar (mmNumPidM M BM - mmFirstPidM s N BN GM)) :=
    is_subScalarNat_eval _ _ t _ _ (by rw [evalOp_ref]; exact hgm)
      (by rw [evalOp_ref]; exact hfp)
  rw [is_whereScalarNat_eval _ _ _ t
    (decide (mmNumPidM M BM - mmFirstPidM s N BN GM < GM))
    (mmNumPidM M BM - mmFirstPidM s N BN GM) GM
    (is_ltScalarNat_eval _ _ t _ _ hsub (evalOp_constNat _ _)) hsub
    (evalOp_constNat _ _)]
  simp only [decide_eq_true_eq, is_min_as_where, mmGroupSize]

private theorem mm_pidM_eval (s t : BlockState) (M N BM BN GM : Nat)
    (hfp : t.regs .nat [] "first_pid_m"
      = some (Tile.scalar (mmFirstPidM s N BN GM)))
    (hpid : t.regs .nat [] "pid" = some (Tile.scalar (s.pids 0)))
    (hgs : t.regs .nat [] "GROUP_M"
      = some (Tile.scalar (mmGroupSize s M N BM BN GM))) :
    evalOp (Op.add .nat Broadcast.nil (Op.ref .nat [] "first_pid_m")
        (Op.mod IntegralDType.nat Broadcast.nil (Op.ref .nat [] "pid")
          (Op.ref .nat [] "GROUP_M"))) t
      = some (Tile.scalar (mmPidM s M N BM BN GM)) := by
  rw [is_addScalarNat_eval _ _ t (mmFirstPidM s N BN GM)
    (s.pids 0 % mmGroupSize s M N BM BN GM)
    (by rw [evalOp_ref]; exact hfp)
    (is_modScalarNat_eval _ _ t _ _ (by rw [evalOp_ref]; exact hpid)
      (by rw [evalOp_ref]; exact hgs))]
  rfl

private theorem mm_pidN_eval (s t : BlockState) (M N BM BN GM : Nat)
    (hpid : t.regs .nat [] "pid" = some (Tile.scalar (s.pids 0)))
    (hwid : t.regs .nat [] "num_pid_in_group"
      = some (Tile.scalar (mmNumPidInGroup N BN GM)))
    (hgs : t.regs .nat [] "GROUP_M"
      = some (Tile.scalar (mmGroupSize s M N BM BN GM))) :
    evalOp (Op.floorDiv IntegralDType.nat Broadcast.nil
        (Op.mod IntegralDType.nat Broadcast.nil (Op.ref .nat [] "pid")
          (Op.ref .nat [] "num_pid_in_group"))
        (Op.ref .nat [] "GROUP_M")) t
      = some (Tile.scalar (mmPidN s M N BM BN GM)) := by
  rw [is_floorDivScalar_eval _ _ t (s.pids 0 % mmNumPidInGroup N BN GM)
    (mmGroupSize s M N BM BN GM)
    (is_modScalarNat_eval _ _ t _ _ (by rw [evalOp_ref]; exact hpid)
      (by rw [evalOp_ref]; exact hwid))
    (by rw [evalOp_ref]; exact hgs)]
  rfl

/-- The nine swizzle scalars, then the two input block pointers and the
zeroed accumulator: lands on `mmInv … 0`. -/
theorem mmPreLoop_run (s : BlockState) (a_ptr b_ptr : Region .int)
    (M N K sam sak sbk sbn BM BN BK GM : Nat) :
    ∃ t, stepStmts (mmPreLoop a_ptr b_ptr M N K sam sak sbk sbn BM BN BK GM) s
        = some t
      ∧ mmInv s a_ptr b_ptr M N K sam sak sbk sbn BM BN BK GM 0 t := by
  unfold mmPreLoop
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 0 s))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (is_cdiv_eval _ M BM))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (is_cdiv_eval _ N BN))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (mm_width_eval _ N BN GM (by simp [mmNumPidN])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (mm_groupId_eval s _ N BN GM (by simp) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (mm_firstPidM_eval s _ N BN GM (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (mm_groupSize_eval s _ M N BM BN GM (by simp [mmNumPidM]) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (mm_pidM_eval s _ M N BM BN GM (by simp) (by simp) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (mm_pidN_eval s _ M N BM BN GM (by simp) (by simp) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (mm_makeBP_eval (Region.cast a_ptr) _ M K sam sak _ _ (mmPidM s M N BM BN GM * BM) 0
      (is_mulRef_eval _ "pid_m" (mmPidM s M N BM BN GM) BM (by simp))
      (evalOp_constNat _ _)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (mm_makeBP_eval (Region.cast b_ptr) _ K N sbk sbn _ _ 0 (mmPidN s M N BM BN GM * BN)
      (evalOp_constNat _ _)
      (is_mulRef_eval _ "pid_n" (mmPidN s M N BM BN GM) BN (by simp))))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.full [BM, BN] (Op.constInt 0)) _
        = some (mmAccTile s a_ptr b_ptr M N K sam sak sbk sbn BM BN
            (mmPidM s M N BM BN GM) (mmPidN s M N BM BN GM) 0) from by
      rw [mmAccTile_zero]
      exact is_full_eval [BM, BN] (Op.constInt 0) _ _ (is_constInt_eval 0 _)))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;> simp [is_setReg_mem]

/-! ## Kernel 1 — the store tail -/

/-- The post-store state: the boundary-checked block-ptr store is a
bounds-guarded `.int` scatter over the output tile. -/
noncomputable def mmStoreState (c_ptr : RegionName)
    (M N scm scn BM BN pm pn : Nat) (f : TileIndex [BM, BN] → ℤ)
    (t : BlockState) : BlockState :=
  (TileShape.allIndices [BM, BN]).foldl
    (fun acc i => if pm * BM + i.1.val < M ∧ pn * BN + i.2.1.val < N then
        acc.writeMemTyped .int c_ptr (mmCAddr scm scn BM BN pm pn i) (f i)
      else acc) t

/-- The boundary-checked `.int` block-ptr store steps to the named scatter
state. -/
private theorem mm_store_eq (c_ptr : RegionName)
    (M N scm scn BM BN pm pn : Nat) (t : BlockState)
    (vt : Tile .int [BM, BN]) (f : TileIndex [BM, BN] → ℤ)
    (hfv : ∀ i, vt.data i = f i)
    (hcp : t.regs .blockPtr [BM, BN] "c_block_ptr"
      = some (mmBPTile c_ptr M N scm scn BM BN (pm * BM) (pn * BN)))
    (hv : t.regs .int [BM, BN] "c" = some vt) :
    stepStmt (Stmt.store .int [BM, BN]
        (MemAccess.blockPtr (Op.ref .blockPtr [BM, BN] "c_block_ptr") [0, 1])
        (Op.ref .int [BM, BN] "c") MaskOpt.none) t
      = some (mmStoreState c_ptr M N scm scn BM BN pm pn f t) := by
  unfold stepStmt mmStoreState
  simp only [evalOp_ref, hv, hcp, Option.bind_some, Option.map_some, bind,
    Option.bind_some]
  refine congrArg some
    (congrArg (fun F => List.foldl F t (TileShape.allIndices [BM, BN])) ?_)
  funext acc i
  obtain ⟨r, cc, u⟩ := i
  simp only [mmBPTile, Bool.true_and,
    TileShape.blockPtr_inBounds_2d_offsets_index,
    TileShape.blockPtr_address_2d_offsets_index, Nat.zero_add]
  by_cases hb : pm * BM + r.val < M ∧ pn * BN + cc.val < N
  · rw [if_pos (by simpa using hb), if_pos hb, hfv]
    rfl
  · rw [if_neg (by simpa using hb), if_neg hb]

/-- `MemCell`-level readback of the bounds-guarded `.int` scatter. -/
private theorem mm_store_props (c_ptr : RegionName)
    (M N scm scn BM BN pm pn : Nat) (t : BlockState)
    (f : TileIndex [BM, BN] → ℤ)
    (hInj : Function.Injective
      (fun i : TileIndex [BM, BN] => mmCAddr scm scn BM BN pm pn i)) :
    ∀ i : TileIndex [BM, BN],
      (pm * BM + i.1.val < M ∧ pn * BN + i.2.1.val < N) →
      (mmStoreState c_ptr M N scm scn BM BN pm pn f t).mem c_ptr
          (mmCAddr scm scn BM BN pm pn i)
        = MemCell.of .int (f i) := by
  classical
  intro i hi
  unfold mmStoreState
  exact is_scatter_int_mem t
    (fun j : TileIndex [BM, BN] => mmCAddr scm scn BM BN pm pn j) f
    (fun j : TileIndex [BM, BN] =>
      pm * BM + j.1.val < M ∧ pn * BN + j.2.1.val < N)
    (fun k₁ k₂ _ _ h => hInj h) i hi

/-- The tail: `c = accumulator`, the output block pointer, and the
boundary-checked store; every in-range lane holds the genuine GEMM value. -/
theorem mmPostLoop_run (s0 : BlockState) (a_ptr b_ptr : Region .int) (c_ptr : RegionName)
    (M N K sam sak sbk sbn scm scn BM BN BK GM F : Nat) (t : BlockState)
    (hKF : K ≤ F)
    (hInj : Function.Injective
      (fun i : TileIndex [BM, BN] => mmCAddr scm scn BM BN
        (mmPidM s0 M N BM BN GM) (mmPidN s0 M N BM BN GM) i))
    (hinv : mmInv s0 a_ptr b_ptr M N K sam sak sbk sbn BM BN BK GM F t) :
    ∃ sF, stepStmts (mmPostLoop c_ptr M N scm scn BM BN) t = some sF
      ∧ ∀ idx : TileIndex [BM, BN],
          (mmPidM s0 M N BM BN GM * BM + idx.1.val < M
            ∧ mmPidN s0 M N BM BN GM * BN + idx.2.1.val < N) →
          sF.mem c_ptr (mmCAddr scm scn BM BN (mmPidM s0 M N BM BN GM)
              (mmPidN s0 M N BM BN GM) idx)
            = MemCell.of .int (mmSpec s0 a_ptr b_ptr K sam sak sbk sbn
                (mmPidM s0 M N BM BN GM * BM + idx.1.val)
                (mmPidN s0 M N BM BN GM * BN + idx.2.1.val)) := by
  obtain ⟨hmem, hpids, hpm, hpn, hA, hB, hacc⟩ := hinv
  set pm := mmPidM s0 M N BM BN GM with hpmDef
  set pn := mmPidN s0 M N BM BN GM with hpnDef
  unfold mmPostLoop
  -- 1. `c = accumulator`
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ref .int [BM, BN] "accumulator") t
        = some (mmAccTile s0 a_ptr b_ptr M N K sam sak sbk sbn BM BN pm pn F) from by
      rw [evalOp_ref]; exact hacc))]
  -- 2. `c_block_ptr`
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (mm_makeBP_eval c_ptr _ M N scm scn _ _ (pm * BM) (pn * BN)
      (is_mulRef_eval _ "pid_m" pm BM (by simpa using hpm))
      (is_mulRef_eval _ "pid_n" pn BN (by simpa using hpn))))]
  -- 3. the boundary-checked store
  rw [stepStmts.cons_some
    (mm_store_eq c_ptr M N scm scn BM BN pm pn _
      (mmAccTile s0 a_ptr b_ptr M N K sam sak sbk sbn BM BN pm pn F)
      (fun ix => mmAccVal s0 a_ptr b_ptr M N K sam sak sbk sbn
        (pm * BM + ix.1.val) (pn * BN + ix.2.1.val) F)
      (fun _ => rfl) (by simp) (by simp))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  intro idx hidx
  rw [mm_store_props c_ptr M N scm scn BM BN pm pn _ _ hInj idx hidx]
  rw [mmAccVal_final s0 a_ptr b_ptr M N K sam sak sbk sbn _ _ F hidx.1 hidx.2 hKF]

/-! ## Kernel 1 — main theorem -/

set_option maxHeartbeats 1000000 in
set_option linter.unusedVariables false in
/-- **Genuine, dimension-general correctness of
`matmul_kernel_with_block_pointers`.** For every launch state, the kernel
runs to completion and every in-range output lane of `C` holds the `.int`
memory cell carrying the exact ℤ GEMM

`mmSpec = ∑ kk < K, A[row, kk] · B[kk, col]`

at the block-pointer address `row·stride_cm + col·stride_cn`. No
divisibility of `K` is assumed: the `boundary_check=(0, 1)` loads zero-fill
every lane past `K` (and past `M`/`K` in the row axes), so the ragged tail
contributes nothing — the only loop hypothesis is `0 < BLOCK_K` (a
`range(0, K, BLOCK_K)` step must be positive). `hInj` says distinct output
lanes get distinct `C` addresses — `mmCAddr_injective_rowMajor` discharges
it for a row-major `C`. The grid is 1-D, so there is no auxiliary
program-id hypothesis. -/
specification int_scaled_matmul_matmul_exec_genuine
    (a_ptr b_ptr : Region .int) (c_ptr : RegionName)
    (M N K : Nat)
    (stride_am stride_ak stride_bk stride_bn stride_cm stride_cn : Nat)
    (BM BN BK GM : Nat) (s : BlockState)
    (hBK : 0 < BK)
    (hInj : Function.Injective
      (fun i : TileIndex [BM, BN] => mmCAddr stride_cm stride_cn BM BN
        (mmPidM s M N BM BN GM) (mmPidN s M N BM BN GM) i)) :
    ∃ sF, exec (int_scaled_matmul_matmul_surface a_ptr b_ptr c_ptr M N K
        stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
        BM BN BK GM).toAlgKernel s = some sF
      ∧ ∀ idx : TileIndex [BM, BN],
          (mmPidM s M N BM BN GM * BM + idx.1.val < M
            ∧ mmPidN s M N BM BN GM * BN + idx.2.1.val < N) →
          sF.mem c_ptr (mmCAddr stride_cm stride_cn BM BN (mmPidM s M N BM BN GM)
              (mmPidN s M N BM BN GM) idx)
            = MemCell.of .int (mmSpec s a_ptr b_ptr K stride_am stride_ak
                stride_bk stride_bn
                (mmPidM s M N BM BN GM * BM + idx.1.val)
                (mmPidN s M N BM BN GM * BN + idx.2.1.val)) := by
  rw [exec, mm_body_eq]
  obtain ⟨t1, hrun1, h1inv⟩ :=
    mmPreLoop_run s a_ptr b_ptr M N K stride_am stride_ak stride_bk stride_bn
      BM BN BK GM
  simp only [List.append_assoc]
  rw [stepStmts.append_some hrun1]
  obtain ⟨F, t2, hrun2, hKF, h2inv⟩ :=
    mmLoop_collapse s a_ptr b_ptr M N K stride_am stride_ak stride_bk stride_bn
      BM BN BK GM t1 hBK h1inv
  rw [show [Stmt.forRange "k" 0 K BK (mmLoopBody BM BN BK)]
        ++ mmPostLoop c_ptr M N stride_cm stride_cn BM BN
      = Stmt.forRange "k" 0 K BK (mmLoopBody BM BN BK)
        :: mmPostLoop c_ptr M N stride_cm stride_cn BM BN from rfl]
  rw [stepStmts.cons_some hrun2]
  obtain ⟨sF, hpost, hout⟩ :=
    mmPostLoop_run s a_ptr b_ptr c_ptr M N K stride_am stride_ak stride_bk
      stride_bn stride_cm stride_cn BM BN BK GM F t2 hKF hInj h2inv
  exact ⟨sF, hpost, hout⟩

/-! ## Kernel 2 — the group-swizzled block coordinates

The `triton.ops.matmul` swizzle (the `int8_dequant_matmul` twin): `grid_*`
are computed with the inline `(X + B - 1) // B` spelling (`Op.floorDiv`,
not `tl.cdiv`'s `Op.div`). -/

/-- `grid_m = (M + BLOCK_M - 1) // BLOCK_M`. -/
def smGridM (M BM : Nat) : Nat := (M + BM - 1) / BM

/-- `grid_n = (N + BLOCK_N - 1) // BLOCK_N`. -/
def smGridN (N BN : Nat) : Nat := (N + BN - 1) / BN

/-- `width = GROUP_M * grid_n`. -/
def smWidth (N BN GM : Nat) : Nat := GM * smGridN N BN

/-- `group_id = pid // width`. -/
def smGroupId (s : BlockState) (N BN GM : Nat) : Nat :=
  s.pids 0 / smWidth N BN GM

/-- `group_size = min(grid_m - group_id * GROUP_M, GROUP_M)`. -/
def smGroupSize (s : BlockState) (M N BM BN GM : Nat) : Nat :=
  min (smGridM M BM - smGroupId s N BN GM * GM) GM

/-- `pid_m = group_id * GROUP_M + (pid % group_size)`. -/
def smPidM (s : BlockState) (M N BM BN GM : Nat) : Nat :=
  smGroupId s N BN GM * GM + s.pids 0 % smGroupSize s M N BM BN GM

/-- `pid_n = (pid % width) // group_size`. -/
def smPidN (s : BlockState) (M N BM BN GM : Nat) : Nat :=
  s.pids 0 % smWidth N BN GM / smGroupSize s M N BM BN GM

/-! ## Kernel 2 — element accessors and the scaled-GEMM spec -/

/-- `A[row, kk]` — the `.int` channel of the launch state (`torch.int8`,
widths erased). -/
def smAElem (s : BlockState) (a_ptr : Region .int) (sam sak : Nat)
    (row kk : Nat) : ℤ :=
  s.readMemValue .int (Region.cast a_ptr) (row * sam + kk * sak)

/-- `B[kk, col]`. -/
def smBElem (s : BlockState) (b_ptr : Region .int) (sbk sbn : Nat)
    (kk col : Nat) : ℤ :=
  s.readMemValue .int (Region.cast b_ptr) (kk * sbk + col * sbn)

/-- `s1[row]` — the per-row fp32 scale, read at the **strideless** address
`row` (the `tl.broadcast_to(idx_m, …)` load; `stride_s1m`/`stride_s1n` are
dead parameters). -/
noncomputable def smS1 (s : BlockState) (s1_ptr : RegionName) (row : Nat) : ℝ :=
  s.readMem s1_ptr row

/-- One K-lane's ℤ contribution: in the masked (`EVEN_K = false`) arm both
loads are masked on the *same* `rk < k` remaining-count condition, so the
lane product is the single-guard `if kk < K` form; in the unmasked arm the
guard is always true (`hEven`). -/
def smG (s : BlockState) (a_ptr b_ptr : Region .int)
    (K sam sak sbk sbn : Nat) (row col kk : Nat) : ℤ :=
  if kk < K then smAElem s a_ptr sam sak row kk * smBElem s b_ptr sbk sbn kk col
  else 0

/-- The running accumulator after `c` swept K-lanes. -/
def smAccVal (s : BlockState) (a_ptr b_ptr : Region .int)
    (K sam sak sbk sbn : Nat) (row col c : Nat) : ℤ :=
  ∑ kk ∈ Finset.range c, smG s a_ptr b_ptr K sam sak sbk sbn row col kk

/-- **The kernel-2 integer accumulator value**: the genuine ℤ GEMM. -/
def smSpec (s : BlockState) (a_ptr b_ptr : Region .int)
    (K sam sak sbk sbn : Nat) (row col : Nat) : ℤ :=
  ∑ kk : Fin K, smAElem s a_ptr sam sak row kk.val
    * smBElem s b_ptr sbk sbn kk.val col

theorem smAccVal_zero (s : BlockState) (a_ptr b_ptr : Region .int)
    (K sam sak sbk sbn row col : Nat) :
    smAccVal s a_ptr b_ptr K sam sak sbk sbn row col 0 = 0 := by
  simp [smAccVal]

theorem smAccVal_step (s : BlockState) (a_ptr b_ptr : Region .int)
    (K sam sak sbk sbn row col c BK : Nat) :
    smAccVal s a_ptr b_ptr K sam sak sbk sbn row col c
      + ∑ e : Fin BK, smG s a_ptr b_ptr K sam sak sbk sbn row col (c + e.val)
      = smAccVal s a_ptr b_ptr K sam sak sbk sbn row col (c + BK) := by
  unfold smAccVal
  rw [Finset.sum_range_add]
  congr 1
  rw [Fin.sum_univ_eq_sum_range (fun e =>
    smG s a_ptr b_ptr K sam sak sbk sbn row col (c + e)) BK]

/-- At any final counter `F ≥ K` the guarded running sum is the genuine
GEMM (lanes `kk ≥ K` contribute `0`). -/
theorem smAccVal_final (s : BlockState) (a_ptr b_ptr : Region .int)
    (K sam sak sbk sbn row col F : Nat) (hKF : K ≤ F) :
    smAccVal s a_ptr b_ptr K sam sak sbk sbn row col F
      = smSpec s a_ptr b_ptr K sam sak sbk sbn row col := by
  unfold smAccVal smSpec
  rw [Fin.sum_univ_eq_sum_range (fun kk =>
    smAElem s a_ptr sam sak row kk * smBElem s b_ptr sbk sbn kk col) K]
  have hsub : Finset.range K ⊆ Finset.range F := fun x hx =>
    Finset.mem_range.mpr (lt_of_lt_of_le (Finset.mem_range.mp hx) hKF)
  rw [← Finset.sum_subset hsub
    (fun kk _ hkk => by
      unfold smG
      rw [if_neg (fun hc => hkk (Finset.mem_range.mpr hc))])]
  refine Finset.sum_congr rfl fun kk hkk => ?_
  unfold smG
  rw [if_pos (Finset.mem_range.mp hkk)]

/-- The flat inductor store address `col + N·row` (absolute, unwrapped
coordinates). -/
def smCAddr (N BM BN pm pn : Nat) (idx : TileIndex [BM, BN]) : Nat :=
  pn * BN + idx.2.1.val + N * (pm * BM + idx.1.val)

/-- **Mask-restricted injectivity of the flat address** — proven outright
(no headline hypothesis): on lanes the store mask lets through, `col < N`,
so `col + N·row` is a base-`N` encoding. -/
theorem smCAddr_inj_active (M N BM BN pm pn : Nat) :
    ∀ i₁ i₂ : TileIndex [BM, BN],
      (pm * BM + i₁.1.val < M ∧ pn * BN + i₁.2.1.val < N) →
      (pm * BM + i₂.1.val < M ∧ pn * BN + i₂.2.1.val < N) →
      smCAddr N BM BN pm pn i₁ = smCAddr N BM BN pm pn i₂ → i₁ = i₂ := by
  rintro ⟨r₁, c₁, u₁⟩ ⟨r₂, c₂, u₂⟩ ⟨_, hc₁⟩ ⟨_, hc₂⟩ heq
  simp only [smCAddr] at heq hc₁ hc₂
  have hr : pm * BM + r₁.val = pm * BM + r₂.val := by
    rcases Nat.lt_trichotomy (pm * BM + r₁.val) (pm * BM + r₂.val) with h | h | h
    · have : N * (pm * BM + r₁.val) + N ≤ N * (pm * BM + r₂.val) := by
        rw [← Nat.mul_succ]
        exact Nat.mul_le_mul_left N h
      omega
    · exact h
    · have : N * (pm * BM + r₂.val) + N ≤ N * (pm * BM + r₁.val) := by
        rw [← Nat.mul_succ]
        exact Nat.mul_le_mul_left N h
      omega
  have hr' : r₁.val = r₂.val := by omega
  have hc' : c₁.val = c₂.val := by
    rw [hr] at heq
    omega
  simp only [Prod.mk.injEq]
  exact ⟨Fin.ext hr', Fin.ext hc', trivial⟩

/-! ## Kernel 2 — compiled body decomposition -/

/-- The prologue: eight swizzle scalars, the index vectors (`ram`/`rbn` are
the `%`-wrapped rows/cols behind the value-erased
`tl.max_contiguous(tl.multiple_of(…))`), the two typed pointer tiles, and
the zeroed `.int` accumulator (the fixed `ACC_TYPE = tl.int32`). -/
def smPreLoop (a_ptr b_ptr : Region .int)
    (M N sam sak sbk sbn BM BN BK GM : Nat) : List Stmt :=
  [ Stmt.assign .nat [] "pid" (Op.programId 0),
    Stmt.assign .nat [] "grid_m"
      (Op.floorDiv IntegralDType.nat Broadcast.nil
        (Op.sub .nat Broadcast.nil
          (Op.add .nat Broadcast.nil (Op.constNat M) (Op.constNat BM)) (Op.constNat 1))
        (Op.constNat BM)),
    Stmt.assign .nat [] "grid_n"
      (Op.floorDiv IntegralDType.nat Broadcast.nil
        (Op.sub .nat Broadcast.nil
          (Op.add .nat Broadcast.nil (Op.constNat N) (Op.constNat BN)) (Op.constNat 1))
        (Op.constNat BN)),
    Stmt.assign .nat [] "width"
      (Op.mul .nat Broadcast.nil (Op.constNat GM) (Op.ref .nat [] "grid_n")),
    Stmt.assign .nat [] "group_id"
      (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.ref .nat [] "pid")
        (Op.ref .nat [] "width")),
    Stmt.assign .nat [] "group_size"
      (Op.where
        (Op.lt ComparableDType.nat Broadcast.nil
          (Op.sub .nat Broadcast.nil (Op.ref .nat [] "grid_m")
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "group_id")
              (Op.constNat GM)))
          (Op.constNat GM))
        (Op.sub .nat Broadcast.nil (Op.ref .nat [] "grid_m")
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "group_id")
            (Op.constNat GM)))
        (Op.constNat GM)),
    Stmt.assign .nat [] "pid_m"
      (Op.add .nat Broadcast.nil
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "group_id") (Op.constNat GM))
        (Op.mod IntegralDType.nat Broadcast.nil (Op.ref .nat [] "pid")
          (Op.ref .nat [] "group_size"))),
    Stmt.assign .nat [] "pid_n"
      (Op.floorDiv IntegralDType.nat Broadcast.nil
        (Op.mod IntegralDType.nat Broadcast.nil (Op.ref .nat [] "pid")
          (Op.ref .nat [] "width"))
        (Op.ref .nat [] "group_size")),
    Stmt.assign .nat [BM] "rm"
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_m") (Op.constNat BM))
        (Op.arange BM)),
    Stmt.assign .nat [BN] "rn"
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_n") (Op.constNat BN))
        (Op.arange BN)),
    Stmt.assign .nat [BM] "ram"
      (Op.mod IntegralDType.nat Broadcast.scalarR (Op.ref .nat [BM] "rm")
        (Op.constNat M)),
    Stmt.assign .nat [BN] "rbn"
      (Op.mod IntegralDType.nat Broadcast.scalarR (Op.ref .nat [BN] "rn")
        (Op.constNat N)),
    Stmt.assign .nat [BK] "rk" (Op.arange BK),
    Stmt.assign .ptr [BM, BK] "A"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase a_ptr)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "ram"))
            (Op.constNat sam))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BK] "rk"))
            (Op.constNat sak)))),
    Stmt.assign .ptr [BK, BN] "B"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase b_ptr)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BK] "rk"))
            (Op.constNat sbk))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "rbn"))
            (Op.constNat sbn)))),
    Stmt.assign .int [BM, BN] "acc" (Op.full [BM, BN] (Op.constInt 0)) ]

/-- The loop body: the remaining-count rematerialization `k = K - j·BK`
(the ascending change of variable), the `EVEN_K` branch with both load
arms, the `Op.dotInt` accumulation, and the two pointer advances. -/
def smLoopBody (K sak sbk BM BN BK : Nat) (EVEN_K : Bool) : List Stmt :=
  [ Stmt.assign .nat [] "k"
      (Op.sub .nat Broadcast.nil (Op.constNat K)
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "j") (Op.constNat BK))),
    Stmt.ifThenElse (Op.constBool EVEN_K)
      [ Stmt.assign .int [BM, BK] "a"
          (Op.load .int (MemAccess.ptr (Op.ref .ptr [BM, BK] "A")) MaskOpt.none),
        Stmt.assign .int [BK, BN] "b"
          (Op.load .int (MemAccess.ptr (Op.ref .ptr [BK, BN] "B")) MaskOpt.none) ]
      [ Stmt.assign .int [BM, BK] "a"
          (Op.load .int (MemAccess.ptr (Op.ref .ptr [BM, BK] "A"))
            (MaskOpt.maskOther
              (Op.remap [BM, BK]
                (Broadcast.consL (Broadcast.consSame Broadcast.nil)).leftIndex
                (Op.lt ComparableDType.nat Broadcast.scalarR
                  (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BK] "rk"))
                  (Op.ref .nat [] "k")))
              ((Op.constInt 0).broadcast [BM, BK]))),
        Stmt.assign .int [BK, BN] "b"
          (Op.load .int (MemAccess.ptr (Op.ref .ptr [BK, BN] "B"))
            (MaskOpt.maskOther
              (Op.remap [BK, BN]
                (Broadcast.consSame (Broadcast.consL Broadcast.nil)).leftIndex
                (Op.lt ComparableDType.nat Broadcast.scalarR
                  (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BK] "rk"))
                  (Op.ref .nat [] "k")))
              ((Op.constInt 0).broadcast [BK, BN]))) ],
    Stmt.assign .int [BM, BN] "acc"
      (Op.add .int (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .int [BM, BN] "acc")
        (Op.dotInt (batch := []) (Op.ref .int [BM, BK] "a")
          (Op.ref .int [BK, BN] "b"))),
    Stmt.assign .ptr [BM, BK] "A"
      (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BM, BK] "A")
        (Op.mul .nat Broadcast.nil (Op.constNat BK) (Op.constNat sak))),
    Stmt.assign .ptr [BK, BN] "B"
      (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BK, BN] "B")
        (Op.mul .nat Broadcast.nil (Op.constNat BK) (Op.constNat sbk))) ]

/-- The inductor suffix: the rematerialized `rm`/`rn`, the unit-axis index
tiles, the two-axis mask, the flat index, the strideless
`tl.broadcast_to(idx_m, …)` scale load (its `[BM, 1] → [BM, BN]` lowering
is `Op.remap`; the already-`[BM, BN]`-shaped `tl.broadcast_to(xindex, …)`
is the identity), and the masked `.real` store of `acc * tmp0` (the
`Op.intToReal` promotion). -/
def smPostLoop (c_ptr s1_ptr : RegionName) (M N BM BN : Nat) : List Stmt :=
  [ Stmt.assign .nat [BM] "rm"
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_m") (Op.constNat BM))
        (Op.arange BM)),
    Stmt.assign .nat [BN] "rn"
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_n") (Op.constNat BN))
        (Op.arange BN)),
    Stmt.assign .nat [BM, 1] "idx_m"
      (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "rm")),
    Stmt.assign .nat [1, BN] "idx_n"
      (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "rn")),
    Stmt.assign .bool [BM, BN] "mask"
      (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.ref .nat [BM, 1] "idx_m") (Op.constNat M))
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.ref .nat [1, BN] "idx_n") (Op.constNat N))),
    Stmt.assign .nat [BM, BN] "xindex"
      (Op.add .nat (Broadcast.consL (Broadcast.consR Broadcast.nil))
        (Op.ref .nat [1, BN] "idx_n")
        (Op.mul .nat Broadcast.scalarL (Op.constNat N)
          (Op.ref .nat [BM, 1] "idx_m"))),
    Stmt.assign .real [BM, BN] "tmp0"
      (Op.load .real
        (MemAccess.region s1_ptr
          (Op.remap [BM, BN]
            (Broadcast.consSame (Broadcast.consL Broadcast.nil)).leftIndex
            (Op.ref .nat [BM, 1] "idx_m")))
        (MaskOpt.mask (Op.ref .bool [BM, BN] "mask"))),
    Stmt.store .real [BM, BN]
      (MemAccess.region c_ptr (Op.ref .nat [BM, BN] "xindex"))
      (Op.mul .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.intToReal (Op.ref .int [BM, BN] "acc"))
        (Op.ref .real [BM, BN] "tmp0"))
      (MaskOpt.mask (Op.ref .bool [BM, BN] "mask")) ]

set_option maxRecDepth 20000 in
/-- **Kernel-2 body split (by `rfl`).** 17 top-level statements. -/
theorem sm_body_eq (a_ptr b_ptr : Region .int) (c_ptr s1_ptr : RegionName)
    (M N K sam sak sbk sbn scm scn ss1m ss1n BM BN BK GM : Nat)
    (EVEN_K : Bool) (numKBlocks : Nat) :
    (int_scaled_matmul_scaled_surface a_ptr b_ptr c_ptr s1_ptr M N K
        sam sak sbk sbn scm scn ss1m ss1n BM BN BK GM EVEN_K
        numKBlocks).toAlgKernel.body
      = smPreLoop a_ptr b_ptr M N sam sak sbk sbn BM BN BK GM
        ++ [Stmt.forRange "j" 0 numKBlocks 1 (smLoopBody K sak sbk BM BN BK EVEN_K)]
        ++ smPostLoop c_ptr s1_ptr M N BM BN := by
  rfl

/-! ## Kernel 2 — index vectors, pointer tiles and value tiles -/

/-- `pid_* * BLOCK + tl.arange(0, BLOCK)`. -/
def smOffs (base BD : Nat) : Tile .nat [BD] := ⟨fun idx => base + idx.1.val⟩

/-- The wrapped vector `(base + e) % Mm` — `ram` / `rbn`. -/
def smWrapOffs (base BD Mm : Nat) : Tile .nat [BD] :=
  ⟨fun idx => (base + idx.1.val) % Mm⟩

/-- `A` lane `(r, e)` at iteration `j`. -/
def smAAddr (M sam sak BM BK pm j : Nat) (idx : TileIndex [BM, BK]) : Nat :=
  (pm * BM + idx.1.val) % M * sam + idx.2.1.val * sak + j * (BK * sak)

/-- `B` lane `(e, c)` at iteration `j`. -/
def smBAddr (N sbk sbn BK BN pn j : Nat) (idx : TileIndex [BK, BN]) : Nat :=
  idx.1.val * sbk + (pn * BN + idx.2.1.val) % N * sbn + j * (BK * sbk)

noncomputable def smAPtrs (a_ptr : Region .int)
    (M sam sak BM BK pm j : Nat) : Tile .ptr [BM, BK] :=
  ⟨fun idx => (Region.cast a_ptr, smAAddr M sam sak BM BK pm j idx)⟩

noncomputable def smBPtrs (b_ptr : Region .int)
    (N sbk sbn BK BN pn j : Nat) : Tile .ptr [BK, BN] :=
  ⟨fun idx => (Region.cast b_ptr, smBAddr N sbk sbn BK BN pn j idx)⟩

theorem smAPtrs_succ (a_ptr : Region .int) (M sam sak BM BK pm j : Nat) :
    Tile.ptrAdd Broadcast.scalarR (smAPtrs a_ptr M sam sak BM BK pm j)
        (Tile.scalar (BK * sak))
      = smAPtrs a_ptr M sam sak BM BK pm (j + 1) := by
  apply Tile.ext
  intro idx
  simp only [Tile.ptrAdd_data, smAPtrs, smAAddr, Tile.scalar,
    Broadcast.leftIndex, Prod.mk.injEq]
  refine ⟨trivial, ?_⟩
  ring

theorem smBPtrs_succ (b_ptr : Region .int) (N sbk sbn BK BN pn j : Nat) :
    Tile.ptrAdd Broadcast.scalarR (smBPtrs b_ptr N sbk sbn BK BN pn j)
        (Tile.scalar (BK * sbk))
      = smBPtrs b_ptr N sbk sbn BK BN pn (j + 1) := by
  apply Tile.ext
  intro idx
  simp only [Tile.ptrAdd_data, smBPtrs, smBAddr, Tile.scalar,
    Broadcast.leftIndex, Prod.mk.injEq]
  refine ⟨trivial, ?_⟩
  ring

theorem smAAddr_eq (M sam sak BM BK pm j : Nat) (idx : TileIndex [BM, BK]) :
    smAAddr M sam sak BM BK pm j idx
      = (pm * BM + idx.1.val) % M * sam + (j * BK + idx.2.1.val) * sak := by
  simp only [smAAddr]
  ring

theorem smBAddr_eq (N sbk sbn BK BN pn j : Nat) (idx : TileIndex [BK, BN]) :
    smBAddr N sbk sbn BK BN pn j idx
      = (j * BK + idx.1.val) * sbk + (pn * BN + idx.2.1.val) % N * sbn := by
  simp only [smBAddr]
  ring

/-- The loaded `a` tile at iteration `j` — the single-guard form both arms
land on. -/
def smATile (s : BlockState) (a_ptr : Region .int)
    (M K sam sak BM BK pm j : Nat) : Tile .int [BM, BK] :=
  ⟨fun idx => if j * BK + idx.2.1.val < K
      then smAElem s a_ptr sam sak ((pm * BM + idx.1.val) % M)
        (j * BK + idx.2.1.val)
      else 0⟩

/-- The loaded `b` tile at iteration `j`. -/
def smBTile (s : BlockState) (b_ptr : Region .int)
    (N K sbk sbn BK BN pn j : Nat) : Tile .int [BK, BN] :=
  ⟨fun idx => if j * BK + idx.1.val < K
      then smBElem s b_ptr sbk sbn (j * BK + idx.1.val)
        ((pn * BN + idx.2.1.val) % N)
      else 0⟩

/-- `acc` at iteration `j`, at the wrapped lane coordinates. -/
def smAccTile (s : BlockState) (a_ptr b_ptr : Region .int)
    (M N K sam sak sbk sbn BM BN BK pm pn j : Nat) : Tile .int [BM, BN] :=
  ⟨fun idx => smAccVal s a_ptr b_ptr K sam sak sbk sbn
      ((pm * BM + idx.1.val) % M) ((pn * BN + idx.2.1.val) % N) (j * BK)⟩

theorem smAccTile_zero (s : BlockState) (a_ptr b_ptr : Region .int)
    (M N K sam sak sbk sbn BM BN BK pm pn : Nat) :
    smAccTile s a_ptr b_ptr M N K sam sak sbk sbn BM BN BK pm pn 0
      = (⟨fun _ => 0⟩ : Tile .int [BM, BN]) := by
  apply Tile.ext
  intro idx
  simp [smAccTile, smAccVal_zero]

/-- **The `Op.dotInt` accumulator step**: the per-lane product of the two
single-guard loads is the guarded summand `smG` (`(if c then x else 0) ·
(if c then y else 0) = if c then x·y else 0`). -/
theorem smAccTile_dotInt_step (s : BlockState) (a_ptr b_ptr : Region .int)
    (M N K sam sak sbk sbn BM BN BK pm pn j : Nat) :
    Tile.bop NumericDType.int.add
        (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (smAccTile s a_ptr b_ptr M N K sam sak sbk sbn BM BN BK pm pn j)
        (Tile.dotInt [] (smATile s a_ptr M K sam sak BM BK pm j)
          (smBTile s b_ptr N K sbk sbn BK BN pn j))
      = smAccTile s a_ptr b_ptr M N K sam sak sbk sbn BM BN BK pm pn (j + 1) := by
  apply Tile.ext
  intro idx
  obtain ⟨r, cc, u⟩ := idx
  simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex,
    smAccTile, NumericDType.int_add]
  erw [Tile.dotInt_nil_data]
  rw [show (j + 1) * BK = j * BK + BK from by ring]
  rw [← smAccVal_step s a_ptr b_ptr K sam sak sbk sbn
    ((pm * BM + r.val) % M) ((pn * BN + cc.val) % N) (j * BK) BK]
  congr 1
  refine Finset.sum_congr rfl fun e _ => ?_
  simp only [smATile, smBTile, smG]
  by_cases h : j * BK + e.val < K
  · rw [if_pos h, if_pos h, if_pos h]
  · rw [if_neg h, if_neg h, if_neg h, mul_zero]

/-! ## Kernel 2 — per-statement eval recipes -/

/-- `pid_* * BLOCK + tl.arange(0, BLOCK)` from a scalar register. -/
private theorem sm_offs_eval (nm : RegName) (t : BlockState) (BD base c : Nat)
    (hr : t.regs .nat [] nm = some (Tile.scalar base)) :
    evalOp (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] nm) (Op.constNat c))
        (Op.arange BD)) t
      = some (smOffs (base * c) BD) := by
  rw [is_addTile_eval NumericDType.nat Broadcast.scalarL _ _ t
    (Tile.scalar (base * c)) (Tile.vec (fun i => (i.val : Nat)))
    (is_mulScalarNat_eval _ _ t base c (by rw [evalOp_ref]; exact hr)
      (evalOp_constNat _ _))
    (evalOp_arange _ _)]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp [smOffs, Tile.vec, Broadcast.rightIndex, NumericDType.add]

/-- Bare `tl.arange(0, BK)` — `rk`. -/
private theorem sm_arange_eval (t : BlockState) (BD : Nat) :
    evalOp (Op.arange BD) t = some (smOffs 0 BD) := by
  rw [evalOp_arange]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp [smOffs, Tile.vec]

/-- `ram = rm % M` (behind the value-erased contiguity hints). -/
private theorem sm_wrap_eval (nm : RegName) (t : BlockState) (BD base Mm : Nat)
    (hr : t.regs .nat [BD] nm = some (smOffs base BD)) :
    evalOp (Op.mod IntegralDType.nat Broadcast.scalarR (Op.ref .nat [BD] nm)
        (Op.constNat Mm)) t
      = some (smWrapOffs base BD Mm) := by
  rw [is_mod_eval Broadcast.scalarR _ _ t (smOffs base BD) (Tile.scalar Mm)
    (by rw [evalOp_ref]; exact hr) (evalOp_constNat _ _)]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp [smWrapOffs, smOffs, Tile.bop_data, Broadcast.leftIndex,
    Broadcast.rightIndex]

/-- `A = a_ptr + (ram[:, None] * sam + rk[None, :] * sak)` — iteration 0. -/
private theorem sm_aPtrsInit_eval (a_ptr : Region .int) (t : BlockState)
    (M sam sak BM BK pm : Nat)
    (hram : t.regs .nat [BM] "ram" = some (smWrapOffs (pm * BM) BM M))
    (hrk : t.regs .nat [BK] "rk" = some (smOffs 0 BK)) :
    evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase a_ptr)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "ram"))
            (Op.constNat sam))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BK] "rk"))
            (Op.constNat sak)))) t
      = some (smAPtrs a_ptr M sam sak BM BK pm 0) := by
  rw [is_ptrAddBase_eval _ _ t _ _
    (is_addTile_eval NumericDType.nat _ _ _ t _ _
      (is_mulTile_eval NumericDType.nat Broadcast.scalarR _ _ t _
        (Tile.scalar sam)
        (is_expandDim_eval _ _ t _ (by rw [evalOp_ref]; exact hram))
        (evalOp_constNat _ _))
      (is_mulTile_eval NumericDType.nat Broadcast.scalarR _ _ t _
        (Tile.scalar sak)
        (is_expandDim_eval _ _ t _ (by rw [evalOp_ref]; exact hrk))
        (evalOp_constNat _ _)))]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp [smAPtrs, smAAddr, smWrapOffs, smOffs, Tile.ptrAdd_data, Tile.bop_data,
    Tile.expandDim_data, TileShape.dropInsertedIndex, Broadcast.leftIndex,
    Broadcast.rightIndex, NumericDType.add, NumericDType.mul]

/-- `B = b_ptr + (rk[:, None] * sbk + rbn[None, :] * sbn)` — iteration 0. -/
private theorem sm_bPtrsInit_eval (b_ptr : Region .int) (t : BlockState)
    (N sbk sbn BK BN pn : Nat)
    (hrk : t.regs .nat [BK] "rk" = some (smOffs 0 BK))
    (hrbn : t.regs .nat [BN] "rbn" = some (smWrapOffs (pn * BN) BN N)) :
    evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase b_ptr)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BK] "rk"))
            (Op.constNat sbk))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "rbn"))
            (Op.constNat sbn)))) t
      = some (smBPtrs b_ptr N sbk sbn BK BN pn 0) := by
  rw [is_ptrAddBase_eval _ _ t _ _
    (is_addTile_eval NumericDType.nat _ _ _ t _ _
      (is_mulTile_eval NumericDType.nat Broadcast.scalarR _ _ t _
        (Tile.scalar sbk)
        (is_expandDim_eval _ _ t _ (by rw [evalOp_ref]; exact hrk))
        (evalOp_constNat _ _))
      (is_mulTile_eval NumericDType.nat Broadcast.scalarR _ _ t _
        (Tile.scalar sbn)
        (is_expandDim_eval _ _ t _ (by rw [evalOp_ref]; exact hrbn))
        (evalOp_constNat _ _)))]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp [smBPtrs, smBAddr, smWrapOffs, smOffs, Tile.ptrAdd_data, Tile.bop_data,
    Tile.expandDim_data, TileShape.dropInsertedIndex, Broadcast.leftIndex,
    Broadcast.rightIndex, NumericDType.add, NumericDType.mul]

/-- The masked arm's `a`-load mask: `rk[None, :] < k` at the remaining
count `kv`. -/
private theorem sm_aMask_eval (t : BlockState) (BM BK kv : Nat)
    (hk : t.regs .nat [BK] "rk" = some (smOffs 0 BK))
    (hkk : t.regs .nat [] "k" = some (Tile.scalar kv)) :
    evalOp (Op.remap [BM, BK]
        (Broadcast.consL (Broadcast.consSame Broadcast.nil)).leftIndex
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BK] "rk"))
          (Op.ref .nat [] "k"))) t
      = some (⟨fun idx => decide (idx.2.1.val < kv)⟩ : Tile .bool [BM, BK]) := by
  rw [is_remap_eval _ _ _ t _
    (is_ltTile_eval ComparableDType.nat Broadcast.scalarR _ _ t _ _
      (is_expandDim_eval _ _ t _ (by rw [evalOp_ref]; exact hk))
      (by rw [evalOp_ref]; exact hkk))]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  obtain ⟨r, e, u⟩ := idx
  simp [Tile.remap, Tile.cop_data, Tile.expandDim_data, smOffs,
    TileShape.dropInsertedIndex, Broadcast.leftIndex, Broadcast.rightIndex,
    ComparableDType.lt]

/-- The masked arm's `b`-load mask: `rk[:, None] < k`. -/
private theorem sm_bMask_eval (t : BlockState) (BK BN kv : Nat)
    (hk : t.regs .nat [BK] "rk" = some (smOffs 0 BK))
    (hkk : t.regs .nat [] "k" = some (Tile.scalar kv)) :
    evalOp (Op.remap [BK, BN]
        (Broadcast.consSame (Broadcast.consL Broadcast.nil)).leftIndex
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BK] "rk"))
          (Op.ref .nat [] "k"))) t
      = some (⟨fun idx => decide (idx.1.val < kv)⟩ : Tile .bool [BK, BN]) := by
  rw [is_remap_eval _ _ _ t _
    (is_ltTile_eval ComparableDType.nat Broadcast.scalarR _ _ t _ _
      (is_expandDim_eval _ _ t _ (by rw [evalOp_ref]; exact hk))
      (by rw [evalOp_ref]; exact hkk))]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  obtain ⟨e, cc, u⟩ := idx
  simp [Tile.remap, Tile.cop_data, Tile.expandDim_data, smOffs,
    TileShape.dropInsertedIndex, Broadcast.leftIndex, Broadcast.rightIndex,
    ComparableDType.lt]

/-- The **unmasked** (`EVEN_K = true`) `a` load lands on the single-guard
tile: under `hcov` every lane is in K-range. -/
private theorem sm_aLoad_plain_eq (s0 : BlockState) (a_ptr : Region .int)
    (t : BlockState) (M K sam sak BM BK pm j : Nat)
    (hcov : (j + 1) * BK ≤ K)
    (hmem : t.mem = s0.mem)
    (hA : t.regs .ptr [BM, BK] "A" = some (smAPtrs a_ptr M sam sak BM BK pm j)) :
    evalOp (Op.load .int (MemAccess.ptr (Op.ref .ptr [BM, BK] "A"))
        MaskOpt.none) t
      = some (smATile s0 a_ptr M K sam sak BM BK pm j) := by
  rw [is_load_ptr_plain "A" t _ hA]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  obtain ⟨r, e, u⟩ := idx
  simp only [smATile, smAPtrs]
  rw [if_pos (show j * BK + e.val < K from by
    have : j * BK + e.val < (j + 1) * BK := by
      have := e.isLt
      calc j * BK + e.val < j * BK + BK := by omega
        _ = (j + 1) * BK := by ring
    omega)]
  rw [smAAddr_eq]
  simp only [smAElem, BlockState.readMemValue, BlockState.readMemTyped, hmem]

/-- The **masked** (`EVEN_K = false`) `a` load lands on the same
single-guard tile: `e < K - j·BK ↔ j·BK + e < K` (ℕ subtraction), and
masked-off lanes read the faithful `other = 0`. -/
private theorem sm_aLoad_masked_eq (s0 : BlockState) (a_ptr : Region .int)
    (t : BlockState) (M K sam sak BM BK pm j : Nat)
    (hmem : t.mem = s0.mem)
    (hA : t.regs .ptr [BM, BK] "A" = some (smAPtrs a_ptr M sam sak BM BK pm j))
    (hk : t.regs .nat [BK] "rk" = some (smOffs 0 BK))
    (hkk : t.regs .nat [] "k" = some (Tile.scalar (K - j * BK))) :
    evalOp (Op.load .int (MemAccess.ptr (Op.ref .ptr [BM, BK] "A"))
        (MaskOpt.maskOther
          (Op.remap [BM, BK]
            (Broadcast.consL (Broadcast.consSame Broadcast.nil)).leftIndex
            (Op.lt ComparableDType.nat Broadcast.scalarR
              (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BK] "rk"))
              (Op.ref .nat [] "k")))
          ((Op.constInt 0).broadcast [BM, BK]))) t
      = some (smATile s0 a_ptr M K sam sak BM BK pm j) := by
  rw [is_load_ptr_maskOther "A" _ _ t _ _ _ hA
    (sm_aMask_eval t BM BK (K - j * BK) hk hkk)
    (is_broadcastInt_eval [BM, BK] t)]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  obtain ⟨r, e, u⟩ := idx
  simp only [smATile, smAPtrs, decide_eq_true_eq]
  by_cases h : j * BK + e.val < K
  · rw [if_pos (show e.val < K - j * BK from by omega), if_pos h, smAAddr_eq]
    simp only [smAElem, BlockState.readMemValue, BlockState.readMemTyped, hmem]
  · rw [if_neg (show ¬ e.val < K - j * BK from by omega), if_neg h]

/-- Unmasked `b` load. -/
private theorem sm_bLoad_plain_eq (s0 : BlockState) (b_ptr : Region .int)
    (t : BlockState) (N K sbk sbn BK BN pn j : Nat)
    (hcov : (j + 1) * BK ≤ K)
    (hmem : t.mem = s0.mem)
    (hB : t.regs .ptr [BK, BN] "B" = some (smBPtrs b_ptr N sbk sbn BK BN pn j)) :
    evalOp (Op.load .int (MemAccess.ptr (Op.ref .ptr [BK, BN] "B"))
        MaskOpt.none) t
      = some (smBTile s0 b_ptr N K sbk sbn BK BN pn j) := by
  rw [is_load_ptr_plain "B" t _ hB]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  obtain ⟨e, cc, u⟩ := idx
  simp only [smBTile, smBPtrs]
  rw [if_pos (show j * BK + e.val < K from by
    have : j * BK + e.val < (j + 1) * BK := by
      have := e.isLt
      calc j * BK + e.val < j * BK + BK := by omega
        _ = (j + 1) * BK := by ring
    omega)]
  rw [smBAddr_eq]
  simp only [smBElem, BlockState.readMemValue, BlockState.readMemTyped, hmem]

/-- Masked `b` load. -/
private theorem sm_bLoad_masked_eq (s0 : BlockState) (b_ptr : Region .int)
    (t : BlockState) (N K sbk sbn BK BN pn j : Nat)
    (hmem : t.mem = s0.mem)
    (hB : t.regs .ptr [BK, BN] "B" = some (smBPtrs b_ptr N sbk sbn BK BN pn j))
    (hk : t.regs .nat [BK] "rk" = some (smOffs 0 BK))
    (hkk : t.regs .nat [] "k" = some (Tile.scalar (K - j * BK))) :
    evalOp (Op.load .int (MemAccess.ptr (Op.ref .ptr [BK, BN] "B"))
        (MaskOpt.maskOther
          (Op.remap [BK, BN]
            (Broadcast.consSame (Broadcast.consL Broadcast.nil)).leftIndex
            (Op.lt ComparableDType.nat Broadcast.scalarR
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BK] "rk"))
              (Op.ref .nat [] "k")))
          ((Op.constInt 0).broadcast [BK, BN]))) t
      = some (smBTile s0 b_ptr N K sbk sbn BK BN pn j) := by
  rw [is_load_ptr_maskOther "B" _ _ t _ _ _ hB
    (sm_bMask_eval t BK BN (K - j * BK) hk hkk)
    (is_broadcastInt_eval [BK, BN] t)]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  obtain ⟨e, cc, u⟩ := idx
  simp only [smBTile, smBPtrs, decide_eq_true_eq]
  by_cases h : j * BK + e.val < K
  · rw [if_pos (show e.val < K - j * BK from by omega), if_pos h, smBAddr_eq]
    simp only [smBElem, BlockState.readMemValue, BlockState.readMemTyped, hmem]
  · rw [if_neg (show ¬ e.val < K - j * BK from by omega), if_neg h]

/-! ## Kernel 2 — the loop invariant and walks -/

/-- The state carried across iterations (`j` is the ascending counter of
the changed variable). -/
noncomputable def smInv (s0 : BlockState) (a_ptr b_ptr : Region .int)
    (M N K sam sak sbk sbn BM BN BK GM numKBlocks : Nat)
    (j : Nat) (s : BlockState) : Prop :=
  j ≤ numKBlocks
  ∧ s.mem = s0.mem
  ∧ s.pids = s0.pids
  ∧ s.regs .nat [] "pid_m" = some (Tile.scalar (smPidM s0 M N BM BN GM))
  ∧ s.regs .nat [] "pid_n" = some (Tile.scalar (smPidN s0 M N BM BN GM))
  ∧ s.regs .nat [BK] "rk" = some (smOffs 0 BK)
  ∧ s.regs .ptr [BM, BK] "A"
      = some (smAPtrs a_ptr M sam sak BM BK (smPidM s0 M N BM BN GM) j)
  ∧ s.regs .ptr [BK, BN] "B"
      = some (smBPtrs b_ptr N sbk sbn BK BN (smPidN s0 M N BM BN GM) j)
  ∧ s.regs .int [BM, BN] "acc"
      = some (smAccTile s0 a_ptr b_ptr M N K sam sak sbk sbn BM BN BK
          (smPidM s0 M N BM BN GM) (smPidN s0 M N BM BN GM) j)

/-- The loop combinator writes `"j"`; `smInv` constrains no register named
`"j"`. -/
theorem smInv_setReg_j (s0 : BlockState) (a_ptr b_ptr : Region .int)
    (M N K sam sak sbk sbn BM BN BK GM numKBlocks j v : Nat) (s : BlockState)
    (h : smInv s0 a_ptr b_ptr M N K sam sak sbk sbn BM BN BK GM numKBlocks j s) :
    smInv s0 a_ptr b_ptr M N K sam sak sbk sbn BM BN BK GM numKBlocks j
      (s.setReg "j" .nat [] (Tile.scalar v)) := by
  obtain ⟨hle, hmem, hpids, hpm, hpn, hrk, hA, hB, hacc⟩ := h
  exact ⟨hle, hmem, hpids, by simpa using hpm, by simpa using hpn,
    by simpa using hrk, by simpa using hA, by simpa using hB,
    by simpa using hacc⟩

/-- One loop iteration preserves the invariant — both `EVEN_K` arms. -/
theorem smBody_run (s0 : BlockState) (a_ptr b_ptr : Region .int)
    (M N K sam sak sbk sbn BM BN BK GM numKBlocks j : Nat) (EVEN_K : Bool)
    (s : BlockState)
    (hEven : EVEN_K = Bool.true → K = numKBlocks * BK)
    (hnext : j + 1 ≤ numKBlocks)
    (hjreg : s.regs .nat [] "j" = some (Tile.scalar j))
    (hinv : smInv s0 a_ptr b_ptr M N K sam sak sbk sbn BM BN BK GM numKBlocks j s) :
    ∃ s', stepStmts (smLoopBody K sak sbk BM BN BK EVEN_K) s = some s'
      ∧ smInv s0 a_ptr b_ptr M N K sam sak sbk sbn BM BN BK GM numKBlocks
          (j + 1) s' := by
  obtain ⟨-, hmem, hpids, hpm, hpn, hrk, hA, hB, hacc⟩ := hinv
  set pm := smPidM s0 M N BM BN GM with hpmDef
  set pn := smPidN s0 M N BM BN GM with hpnDef
  unfold smLoopBody
  -- 1. `k = K - j * BLOCK_K` (the remaining count)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (is_subScalarNat_eval _ _ s K (j * BK) (evalOp_constNat _ _)
      (is_mulRef_eval s "j" j BK hjreg)))]
  -- 2. the `EVEN_K` branch: both arms land on the same guarded tiles
  have harms : stepStmt (Stmt.ifThenElse (Op.constBool EVEN_K)
      [ Stmt.assign .int [BM, BK] "a"
          (Op.load .int (MemAccess.ptr (Op.ref .ptr [BM, BK] "A")) MaskOpt.none),
        Stmt.assign .int [BK, BN] "b"
          (Op.load .int (MemAccess.ptr (Op.ref .ptr [BK, BN] "B")) MaskOpt.none) ]
      [ Stmt.assign .int [BM, BK] "a"
          (Op.load .int (MemAccess.ptr (Op.ref .ptr [BM, BK] "A"))
            (MaskOpt.maskOther
              (Op.remap [BM, BK]
                (Broadcast.consL (Broadcast.consSame Broadcast.nil)).leftIndex
                (Op.lt ComparableDType.nat Broadcast.scalarR
                  (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BK] "rk"))
                  (Op.ref .nat [] "k")))
              ((Op.constInt 0).broadcast [BM, BK]))),
        Stmt.assign .int [BK, BN] "b"
          (Op.load .int (MemAccess.ptr (Op.ref .ptr [BK, BN] "B"))
            (MaskOpt.maskOther
              (Op.remap [BK, BN]
                (Broadcast.consSame (Broadcast.consL Broadcast.nil)).leftIndex
                (Op.lt ComparableDType.nat Broadcast.scalarR
                  (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BK] "rk"))
                  (Op.ref .nat [] "k")))
              ((Op.constInt 0).broadcast [BK, BN]))) ])
      (s.setReg "k" .nat [] (Tile.scalar (K - j * BK)))
      = some (((s.setReg "k" .nat [] (Tile.scalar (K - j * BK))).setReg
          "a" .int [BM, BK] (smATile s0 a_ptr M K sam sak BM BK pm j)).setReg
          "b" .int [BK, BN] (smBTile s0 b_ptr N K sbk sbn BK BN pn j)) := by
    rw [is_ifThenElse_step]
    cases EVEN_K
    · -- the masked (`EVEN_K = false`) arm
      rw [if_neg (by simp)]
      rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (sm_aLoad_masked_eq s0 a_ptr _ M K sam sak BM BK pm j
          (by simpa [is_setReg_mem] using hmem) (by simpa using hA)
          (by simpa using hrk) (by simp)))]
      rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (sm_bLoad_masked_eq s0 b_ptr _ N K sbk sbn BK BN pn j
          (by simpa [is_setReg_mem] using hmem) (by simpa using hB)
          (by simpa using hrk) (by simp)))]
      rw [stepStmts.nil]
    · -- the unmasked (`EVEN_K = true`) arm
      have hK := hEven rfl
      have hcov : (j + 1) * BK ≤ K := by
        rw [hK]
        exact Nat.mul_le_mul_right BK hnext
      rw [if_pos rfl]
      rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (sm_aLoad_plain_eq s0 a_ptr _ M K sam sak BM BK pm j hcov
          (by simpa [is_setReg_mem] using hmem) (by simpa using hA)))]
      rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (sm_bLoad_plain_eq s0 b_ptr _ N K sbk sbn BK BN pn j hcov
          (by simpa [is_setReg_mem] using hmem) (by simpa using hB)))]
      rw [stepStmts.nil]
  rw [stepStmts.cons_some harms]
  -- 3. `acc += tl.dot(a, b)`
  have haccStep : evalOp (Op.add .int
      (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
      (Op.ref .int [BM, BN] "acc")
      (Op.dotInt (batch := []) (Op.ref .int [BM, BK] "a")
        (Op.ref .int [BK, BN] "b")))
      (((s.setReg "k" .nat [] (Tile.scalar (K - j * BK))).setReg
          "a" .int [BM, BK] (smATile s0 a_ptr M K sam sak BM BK pm j)).setReg
          "b" .int [BK, BN] (smBTile s0 b_ptr N K sbk sbn BK BN pn j))
      = some (smAccTile s0 a_ptr b_ptr M N K sam sak sbk sbn BM BN BK pm pn
          (j + 1)) := by
    rw [← smAccTile_dotInt_step s0 a_ptr b_ptr M N K sam sak sbk sbn BM BN BK
      pm pn j]
    exact is_addTile_eval NumericDType.int _ _ _ _ _ _
      (by rw [evalOp_ref]; simpa using hacc)
      (is_dotInt_eval _ _ _ _ _
        (by rw [evalOp_ref]; simp) (by rw [evalOp_ref]; simp))
  rw [stepStmts.cons_some (stepStmt_assign_eq_some haccStep)]
  -- 4. `A += BLOCK_K * stride_ak`
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BM, BK] "A")
        (Op.mul .nat Broadcast.nil (Op.constNat BK) (Op.constNat sak))) _
      = some (smAPtrs a_ptr M sam sak BM BK pm (j + 1)) from by
      rw [← smAPtrs_succ]
      exact is_ptrAdd_eval Broadcast.scalarR "A" _ _ _
        (Tile.scalar (BK * sak)) (by simpa using hA)
        (is_mulScalarNat_eval _ _ _ BK sak (evalOp_constNat _ _)
          (evalOp_constNat _ _))))]
  -- 5. `B += BLOCK_K * stride_bk`
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BK, BN] "B")
        (Op.mul .nat Broadcast.nil (Op.constNat BK) (Op.constNat sbk))) _
      = some (smBPtrs b_ptr N sbk sbn BK BN pn (j + 1)) from by
      rw [← smBPtrs_succ]
      exact is_ptrAdd_eval Broadcast.scalarR "B" _ _ _
        (Tile.scalar (BK * sbk)) (by simpa using hB)
        (is_mulScalarNat_eval _ _ _ BK sbk (evalOp_constNat _ _)
          (evalOp_constNat _ _))))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, hnext, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp only [is_setReg_mem]
    exact hmem
  · simp only [BlockState.setReg_pids]
    exact hpids
  · simpa using hpm
  · simpa using hpn
  · simpa using hrk
  · simp [hpmDef]
  · simp [hpnDef]
  · simp [hpmDef, hpnDef]

/-- The collapsed loop: exactly `numKBlocks` iterations. -/
theorem smLoop_collapse (s0 : BlockState) (a_ptr b_ptr : Region .int)
    (M N K sam sak sbk sbn BM BN BK GM numKBlocks : Nat) (EVEN_K : Bool)
    (s : BlockState)
    (hEven : EVEN_K = Bool.true → K = numKBlocks * BK)
    (h0 : smInv s0 a_ptr b_ptr M N K sam sak sbk sbn BM BN BK GM numKBlocks 0 s) :
    ∃ sF, stepStmt (Stmt.forRange "j" 0 numKBlocks 1
          (smLoopBody K sak sbk BM BN BK EVEN_K)) s = some sF
      ∧ smInv s0 a_ptr b_ptr M N K sam sak sbk sbn BM BN BK GM numKBlocks
          numKBlocks sF := by
  obtain ⟨F, sF, hrun, hfinal, hP⟩ :=
    forRange_inv (idx := "j") (start := 0) (stop := numKBlocks) (step := 1)
      (P := fun j t => smInv s0 a_ptr b_ptr M N K sam sak sbk sbn BM BN BK GM
        numKBlocks j t)
      one_ne_zero h0
      (fun j t hj hinv => by
        obtain ⟨s', hs', hinv'⟩ :=
          smBody_run s0 a_ptr b_ptr M N K sam sak sbk sbn BM BN BK GM
            numKBlocks j EVEN_K _ hEven (by omega) (by simp)
            (smInv_setReg_j s0 a_ptr b_ptr M N K sam sak sbk sbn BM BN BK GM
              numKBlocks j j t hinv)
        exact ⟨s', hs', hinv'⟩)
  -- `rw … at hP`, not `subst`: `subst` would eliminate the *binder*
  -- `numKBlocks`, breaking every later mention of it.
  have hEq : F = numKBlocks := le_antisymm hP.1 hfinal
  rw [hEq] at hP
  exact ⟨sF, hrun, hP⟩

/-! ## Kernel 2 — the prologue walk -/

private theorem sm_width_eval (t : BlockState) (N BN GM : Nat)
    (hgn : t.regs .nat [] "grid_n" = some (Tile.scalar (smGridN N BN))) :
    evalOp (Op.mul .nat Broadcast.nil (Op.constNat GM)
        (Op.ref .nat [] "grid_n")) t
      = some (Tile.scalar (smWidth N BN GM)) := by
  rw [is_mulScalarNat_eval _ _ t GM (smGridN N BN) (evalOp_constNat _ _)
    (by rw [evalOp_ref]; exact hgn)]
  rfl

private theorem sm_groupId_eval (s t : BlockState) (N BN GM : Nat)
    (hpid : t.regs .nat [] "pid" = some (Tile.scalar (s.pids 0)))
    (hwid : t.regs .nat [] "width" = some (Tile.scalar (smWidth N BN GM))) :
    evalOp (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.ref .nat [] "pid")
        (Op.ref .nat [] "width")) t
      = some (Tile.scalar (smGroupId s N BN GM)) := by
  rw [is_floorDivScalar_eval _ _ t (s.pids 0) (smWidth N BN GM)
    (by rw [evalOp_ref]; exact hpid) (by rw [evalOp_ref]; exact hwid)]
  rfl

private theorem sm_groupSize_eval (s t : BlockState) (M N BM BN GM : Nat)
    (hgm : t.regs .nat [] "grid_m" = some (Tile.scalar (smGridM M BM)))
    (hgid : t.regs .nat [] "group_id"
      = some (Tile.scalar (smGroupId s N BN GM))) :
    evalOp (Op.where
        (Op.lt ComparableDType.nat Broadcast.nil
          (Op.sub .nat Broadcast.nil (Op.ref .nat [] "grid_m")
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "group_id")
              (Op.constNat GM)))
          (Op.constNat GM))
        (Op.sub .nat Broadcast.nil (Op.ref .nat [] "grid_m")
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "group_id")
            (Op.constNat GM)))
        (Op.constNat GM)) t
      = some (Tile.scalar (smGroupSize s M N BM BN GM)) := by
  have hsub : evalOp (Op.sub .nat Broadcast.nil (Op.ref .nat [] "grid_m")
      (Op.mul .nat Broadcast.nil (Op.ref .nat [] "group_id") (Op.constNat GM))) t
      = some (Tile.scalar (smGridM M BM - smGroupId s N BN GM * GM)) :=
    is_subScalarNat_eval _ _ t _ _ (by rw [evalOp_ref]; exact hgm)
      (is_mulRef_eval t "group_id" (smGroupId s N BN GM) GM hgid)
  rw [is_whereScalarNat_eval _ _ _ t
    (decide (smGridM M BM - smGroupId s N BN GM * GM < GM))
    (smGridM M BM - smGroupId s N BN GM * GM) GM
    (is_ltScalarNat_eval _ _ t _ _ hsub (evalOp_constNat _ _)) hsub
    (evalOp_constNat _ _)]
  simp only [decide_eq_true_eq, is_min_as_where, smGroupSize]

private theorem sm_pidM_eval (s t : BlockState) (M N BM BN GM : Nat)
    (hgid : t.regs .nat [] "group_id"
      = some (Tile.scalar (smGroupId s N BN GM)))
    (hpid : t.regs .nat [] "pid" = some (Tile.scalar (s.pids 0)))
    (hgs : t.regs .nat [] "group_size"
      = some (Tile.scalar (smGroupSize s M N BM BN GM))) :
    evalOp (Op.add .nat Broadcast.nil
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "group_id") (Op.constNat GM))
        (Op.mod IntegralDType.nat Broadcast.nil (Op.ref .nat [] "pid")
          (Op.ref .nat [] "group_size"))) t
      = some (Tile.scalar (smPidM s M N BM BN GM)) := by
  rw [is_addScalarNat_eval _ _ t (smGroupId s N BN GM * GM)
    (s.pids 0 % smGroupSize s M N BM BN GM)
    (is_mulRef_eval t "group_id" (smGroupId s N BN GM) GM hgid)
    (is_modScalarNat_eval _ _ t _ _ (by rw [evalOp_ref]; exact hpid)
      (by rw [evalOp_ref]; exact hgs))]
  rfl

private theorem sm_pidN_eval (s t : BlockState) (M N BM BN GM : Nat)
    (hpid : t.regs .nat [] "pid" = some (Tile.scalar (s.pids 0)))
    (hwid : t.regs .nat [] "width" = some (Tile.scalar (smWidth N BN GM)))
    (hgs : t.regs .nat [] "group_size"
      = some (Tile.scalar (smGroupSize s M N BM BN GM))) :
    evalOp (Op.floorDiv IntegralDType.nat Broadcast.nil
        (Op.mod IntegralDType.nat Broadcast.nil (Op.ref .nat [] "pid")
          (Op.ref .nat [] "width"))
        (Op.ref .nat [] "group_size")) t
      = some (Tile.scalar (smPidN s M N BM BN GM)) := by
  rw [is_floorDivScalar_eval _ _ t (s.pids 0 % smWidth N BN GM)
    (smGroupSize s M N BM BN GM)
    (is_modScalarNat_eval _ _ t _ _ (by rw [evalOp_ref]; exact hpid)
      (by rw [evalOp_ref]; exact hwid))
    (by rw [evalOp_ref]; exact hgs)]
  rfl

/-- The eight swizzle scalars, the index vectors, the two pointer tiles and
the zeroed accumulator: lands on `smInv … 0`. -/
theorem smPreLoop_run (s : BlockState) (a_ptr b_ptr : Region .int)
    (M N K sam sak sbk sbn BM BN BK GM numKBlocks : Nat) :
    ∃ t, stepStmts (smPreLoop a_ptr b_ptr M N sam sak sbk sbn BM BN BK GM) s
        = some t
      ∧ smInv s a_ptr b_ptr M N K sam sak sbk sbn BM BN BK GM numKBlocks 0 t := by
  unfold smPreLoop
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 0 s))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (is_gridDiv_eval _ M BM))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (is_gridDiv_eval _ N BN))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (sm_width_eval _ N BN GM (by simp [smGridN])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (sm_groupId_eval s _ N BN GM (by simp) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (sm_groupSize_eval s _ M N BM BN GM (by simp [smGridM]) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (sm_pidM_eval s _ M N BM BN GM (by simp) (by simp) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (sm_pidN_eval s _ M N BM BN GM (by simp) (by simp) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (sm_offs_eval "pid_m" _ BM (smPidM s M N BM BN GM) BM (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (sm_offs_eval "pid_n" _ BN (smPidN s M N BM BN GM) BN (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (sm_wrap_eval "rm" _ BM (smPidM s M N BM BN GM * BM) M (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (sm_wrap_eval "rn" _ BN (smPidN s M N BM BN GM * BN) N (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (sm_arange_eval _ BK))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (sm_aPtrsInit_eval a_ptr _ M sam sak BM BK (smPidM s M N BM BN GM)
      (by simp) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (sm_bPtrsInit_eval b_ptr _ N sbk sbn BK BN (smPidN s M N BM BN GM)
      (by simp) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.full [BM, BN] (Op.constInt 0)) _
        = some (smAccTile s a_ptr b_ptr M N K sam sak sbk sbn BM BN BK
            (smPidM s M N BM BN GM) (smPidN s M N BM BN GM) 0) from by
      rw [smAccTile_zero]
      exact is_full_eval [BM, BN] (Op.constInt 0) _ _ (is_constInt_eval 0 _)))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, Nat.zero_le _, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;>
    simp [is_setReg_mem]

/-! ## Kernel 2 — the inductor suffix -/

/-- `idx_m = rm[:, None]`. -/
def smIdxMTile (BM pm : Nat) : Tile .nat [BM, 1] :=
  ⟨fun idx => pm * BM + idx.1.val⟩

/-- `idx_n = rn[None, :]`. -/
def smIdxNTile (BN pn : Nat) : Tile .nat [1, BN] :=
  ⟨fun idx => pn * BN + idx.2.1.val⟩

/-- `mask = (idx_m < M) & (idx_n < N)` — absolute (unwrapped) coordinates. -/
def smMaskTile (M N BM BN pm pn : Nat) : Tile .bool [BM, BN] :=
  ⟨fun idx => decide (pm * BM + idx.1.val < M)
      && decide (pn * BN + idx.2.1.val < N)⟩

/-- `xindex = idx_n + (N * idx_m)` — the flat store index. -/
def smXIndexTile (N BM BN pm pn : Nat) : Tile .nat [BM, BN] :=
  ⟨fun idx => pn * BN + idx.2.1.val + N * (pm * BM + idx.1.val)⟩

private theorem sm_idxM_eval (t : BlockState) (BM pm : Nat)
    (hrm : t.regs .nat [BM] "rm" = some (smOffs (pm * BM) BM)) :
    evalOp (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "rm")) t
      = some (smIdxMTile BM pm) := by
  rw [is_expandDim_eval _ _ t _ (by rw [evalOp_ref]; exact hrm)]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp [smIdxMTile, smOffs, Tile.expandDim_data, TileShape.dropInsertedIndex]

private theorem sm_idxN_eval (t : BlockState) (BN pn : Nat)
    (hrn : t.regs .nat [BN] "rn" = some (smOffs (pn * BN) BN)) :
    evalOp (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "rn")) t
      = some (smIdxNTile BN pn) := by
  rw [is_expandDim_eval _ _ t _ (by rw [evalOp_ref]; exact hrn)]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp [smIdxNTile, smOffs, Tile.expandDim_data, TileShape.dropInsertedIndex]

private theorem sm_mask_eval (t : BlockState) (M N BM BN pm pn : Nat)
    (him : t.regs .nat [BM, 1] "idx_m" = some (smIdxMTile BM pm))
    (hin : t.regs .nat [1, BN] "idx_n" = some (smIdxNTile BN pn)) :
    evalOp (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.ref .nat [BM, 1] "idx_m") (Op.constNat M))
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.ref .nat [1, BN] "idx_n") (Op.constNat N))) t
      = some (smMaskTile M N BM BN pm pn) := by
  rw [is_boolAnd_eval (Broadcast.consR (Broadcast.consL Broadcast.nil)) _ _ t _ _
    (is_ltTile_eval ComparableDType.nat Broadcast.scalarR _ _ t _
      (Tile.scalar M) (by rw [evalOp_ref]; exact him) (evalOp_constNat _ _))
    (is_ltTile_eval ComparableDType.nat Broadcast.scalarR _ _ t _
      (Tile.scalar N) (by rw [evalOp_ref]; exact hin) (evalOp_constNat _ _))]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp [smMaskTile, smIdxMTile, smIdxNTile, Tile.bop_data, Tile.cop_data,
    Broadcast.leftIndex, Broadcast.rightIndex, ComparableDType.lt]

private theorem sm_xindex_eval (t : BlockState) (N BM BN pm pn : Nat)
    (him : t.regs .nat [BM, 1] "idx_m" = some (smIdxMTile BM pm))
    (hin : t.regs .nat [1, BN] "idx_n" = some (smIdxNTile BN pn)) :
    evalOp (Op.add .nat (Broadcast.consL (Broadcast.consR Broadcast.nil))
        (Op.ref .nat [1, BN] "idx_n")
        (Op.mul .nat Broadcast.scalarL (Op.constNat N)
          (Op.ref .nat [BM, 1] "idx_m"))) t
      = some (smXIndexTile N BM BN pm pn) := by
  rw [is_addTile_eval NumericDType.nat _ _ _ t _ _
    (by rw [evalOp_ref]; exact hin)
    (is_mulTile_eval NumericDType.nat Broadcast.scalarL _ _ t
      (Tile.scalar N) _ (evalOp_constNat _ _)
      (by rw [evalOp_ref]; exact him))]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  simp [smXIndexTile, smIdxMTile, smIdxNTile, Tile.bop_data,
    Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add,
    NumericDType.mul]

/-- The masked `.real` region load without `other`: active lanes read
memory, masked-off lanes read the (never-stored) `undef` value. -/
private theorem sm_load_region_masked_real {sh : TileShape}
    (rg : RegionName) (offOp : Op .nat sh) (maskOp : Op .bool sh)
    (t : BlockState) (offs : Tile .nat sh) (masks : Tile .bool sh)
    (ho : evalOp offOp t = some offs) (hm : evalOp maskOp t = some masks) :
    evalOp (Op.load .real (MemAccess.region rg offOp) (MaskOpt.mask maskOp)) t
      = some (⟨fun i => if masks.data i
            then t.readMemValue .real rg (offs.data i)
            else some (t.undef rg (offs.data i))⟩
          : Tile .real sh) := by
  simp only [evalOp, ho, hm]
  rfl

/-- The strideless `tl.broadcast_to(idx_m, …)` scale load: lane `(r, c)`
reads `s1[pid_m·BM + r]` when the mask lets it through. -/
private theorem sm_s1Load_eq (s0 : BlockState) (s1_ptr : RegionName)
    (t : BlockState) (M N BM BN pm pn : Nat)
    (hmem : t.mem = s0.mem)
    (him : t.regs .nat [BM, 1] "idx_m" = some (smIdxMTile BM pm))
    (hmk : t.regs .bool [BM, BN] "mask" = some (smMaskTile M N BM BN pm pn)) :
    evalOp (Op.load .real
        (MemAccess.region s1_ptr
          (Op.remap [BM, BN]
            (Broadcast.consSame (Broadcast.consL Broadcast.nil)).leftIndex
            (Op.ref .nat [BM, 1] "idx_m")))
        (MaskOpt.mask (Op.ref .bool [BM, BN] "mask"))) t
      = some (⟨fun idx =>
            if pm * BM + idx.1.val < M ∧ pn * BN + idx.2.1.val < N
            then some (smS1 s0 s1_ptr (pm * BM + idx.1.val))
            else some (t.undef s1_ptr (pm * BM + idx.1.val))⟩
          : Tile .real [BM, BN]) := by
  have hread : ∀ o, t.readMem s1_ptr o = s0.readMem s1_ptr o := by
    intro o
    unfold BlockState.readMem
    rw [hmem]
  rw [sm_load_region_masked_real s1_ptr _ _ t
    (Tile.remap ((Broadcast.consSame (Broadcast.consL Broadcast.nil)).leftIndex)
      (smIdxMTile BM pm))
    (smMaskTile M N BM BN pm pn)
    (is_remap_eval _ _ _ t _ (by rw [evalOp_ref]; exact him)) (by
      rw [evalOp_ref]; exact hmk)]
  refine congrArg some ?_
  apply Tile.ext
  intro idx
  obtain ⟨r, cc, u⟩ := idx
  simp only [smMaskTile, smIdxMTile, Tile.remap, Broadcast.leftIndex,
    Bool.and_eq_true, decide_eq_true_eq, BlockState.readMemValue_real, smS1]
  by_cases hb : pm * BM + r.val < M ∧ pn * BN + cc.val < N
  · rw [if_pos hb, if_pos hb, hread]
  · rw [if_neg hb, if_neg hb]

/-! ## Kernel 2 — the masked flat store -/

/-- The post-store state: a mask-guarded `.real` scatter at the flat
addresses. -/
noncomputable def smStoreState (c_ptr : RegionName)
    (M N BM BN pm pn : Nat) (v : TileIndex [BM, BN] → TileCarrier .real)
    (t : BlockState) : BlockState :=
  (TileShape.allIndices [BM, BN]).foldl
    (fun acc i => if pm * BM + i.1.val < M ∧ pn * BN + i.2.1.val < N then
        acc.writeMemTyped .real c_ptr (smCAddr N BM BN pm pn i) (v i)
      else acc) t

/-- The masked `.real` store of `acc * tmp0` steps to the named scatter
state (the value is consulted only on active lanes). -/
private theorem sm_store_eq (c_ptr : RegionName)
    (M N BM BN pm pn : Nat) (t : BlockState)
    (valueOp : Op .real [BM, BN]) (vt : Tile .real [BM, BN])
    (v : TileIndex [BM, BN] → TileCarrier .real)
    (hval : evalOp valueOp t = some vt)
    (hfv : ∀ i : TileIndex [BM, BN],
      (pm * BM + i.1.val < M ∧ pn * BN + i.2.1.val < N) → vt.data i = v i)
    (hx : t.regs .nat [BM, BN] "xindex" = some (smXIndexTile N BM BN pm pn))
    (hmk : t.regs .bool [BM, BN] "mask" = some (smMaskTile M N BM BN pm pn)) :
    stepStmt (Stmt.store .real [BM, BN]
        (MemAccess.region c_ptr (Op.ref .nat [BM, BN] "xindex"))
        valueOp (MaskOpt.mask (Op.ref .bool [BM, BN] "mask"))) t
      = some (smStoreState c_ptr M N BM BN pm pn v t) := by
  unfold stepStmt smStoreState
  simp only [evalOp_ref, hval, hx, hmk, Option.bind_some, Option.map_some, bind]
  refine congrArg some
    (congrArg (fun F => List.foldl F t (TileShape.allIndices [BM, BN])) ?_)
  funext acc i
  obtain ⟨r, cc, u⟩ := i
  by_cases hb : pm * BM + r.val < M ∧ pn * BN + cc.val < N
  · rw [if_pos (show (smMaskTile M N BM BN pm pn).data (r, cc, u) = Bool.true from by
      simp only [smMaskTile, Bool.and_eq_true, decide_eq_true_eq]
      exact ⟨hb.1, hb.2⟩)]
    rw [if_pos hb, hfv _ hb]
    rfl
  · rw [if_neg (show ¬((smMaskTile M N BM BN pm pn).data (r, cc, u) = Bool.true) from by
      simp only [smMaskTile, Bool.and_eq_true, decide_eq_true_eq]
      exact fun hc => hb hc)]
    rw [if_neg hb]

/-- `MemCell`-level readback of the masked `.real` scatter on every active
lane — the flat-address injectivity is the *proven*
`smCAddr_inj_active`. -/
private theorem sm_store_props (c_ptr : RegionName)
    (M N BM BN pm pn : Nat) (t : BlockState)
    (v : TileIndex [BM, BN] → TileCarrier .real) :
    ∀ i : TileIndex [BM, BN],
      (pm * BM + i.1.val < M ∧ pn * BN + i.2.1.val < N) →
      (smStoreState c_ptr M N BM BN pm pn v t).mem c_ptr
          (smCAddr N BM BN pm pn i)
        = MemCell.of .real
            (FloatDType.real.ofReal (FloatDType.real.storeValue (v i))) := by
  classical
  intro i hi
  unfold smStoreState
  exact is_scatter_real_mem t
    (fun j : TileIndex [BM, BN] => smCAddr N BM BN pm pn j) v
    (fun j : TileIndex [BM, BN] =>
      pm * BM + j.1.val < M ∧ pn * BN + j.2.1.val < N)
    (fun k₁ k₂ h₁ h₂ h => smCAddr_inj_active M N BM BN pm pn k₁ k₂ h₁ h₂ h) i hi

/-- The inductor suffix: `rm`/`rn` remat, the index tiles, mask, flat
index, the strideless scale load, and the masked `.real` store — every
active lane holds the exact real `(∑ ℤ : ℝ) · s1(row)`. -/
theorem smPostLoop_run (s0 : BlockState) (a_ptr b_ptr : Region .int)
    (c_ptr s1_ptr : RegionName)
    (M N K sam sak sbk sbn BM BN BK GM numKBlocks : Nat) (t : BlockState)
    (hCeil : K ≤ numKBlocks * BK)
    (hinv : smInv s0 a_ptr b_ptr M N K sam sak sbk sbn BM BN BK GM numKBlocks
      numKBlocks t) :
    ∃ sF, stepStmts (smPostLoop c_ptr s1_ptr M N BM BN) t = some sF
      ∧ ∀ idx : TileIndex [BM, BN],
          (smPidM s0 M N BM BN GM * BM + idx.1.val < M
            ∧ smPidN s0 M N BM BN GM * BN + idx.2.1.val < N) →
          sF.mem c_ptr (smCAddr N BM BN (smPidM s0 M N BM BN GM)
              (smPidN s0 M N BM BN GM) idx)
            = MemCell.of .real (some
                (((smSpec s0 a_ptr b_ptr K sam sak sbk sbn
                    (smPidM s0 M N BM BN GM * BM + idx.1.val)
                    (smPidN s0 M N BM BN GM * BN + idx.2.1.val) : ℤ) : ℝ)
                  * smS1 s0 s1_ptr (smPidM s0 M N BM BN GM * BM + idx.1.val))) := by
  obtain ⟨-, hmem, hpids, hpm, hpn, hrk, hA, hB, hacc⟩ := hinv
  set pm := smPidM s0 M N BM BN GM with hpmDef
  set pn := smPidN s0 M N BM BN GM with hpnDef
  unfold smPostLoop
  -- 1–2. `rm` / `rn` rematerialized
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (sm_offs_eval "pid_m" _ BM pm BM (by simpa using hpm)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (sm_offs_eval "pid_n" _ BN pn BN (by simpa using hpn)))]
  -- 3–4. the unit-axis index tiles (`@`-pinned: the `expandDim` result
  -- shape `insertAxis [BM] ⟨1,_⟩ 1` must be read at the literal `[BM, 1]`)
  rw [stepStmts.cons_some (@stepStmt_assign_eq_some .nat [BM, 1] "idx_m" _ _ _
    (sm_idxM_eval _ BM pm (by simp)))]
  rw [stepStmts.cons_some (@stepStmt_assign_eq_some .nat [1, BN] "idx_n" _ _ _
    (sm_idxN_eval _ BN pn (by simp)))]
  -- 5. the mask
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (sm_mask_eval _ M N BM BN pm pn (by simp) (by simp)))]
  -- 6. the flat index
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (sm_xindex_eval _ N BM BN pm pn (by simp) (by simp)))]
  -- 7. the strideless scale load
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (sm_s1Load_eq s0 s1_ptr _ M N BM BN pm pn
      (by simpa [is_setReg_mem] using hmem) (by simp) (by simp)))]
  -- 8. the masked `.real` store of `acc * tmp0`
  set tmp0T : Tile .real [BM, BN] :=
    ⟨fun idx => if pm * BM + idx.1.val < M ∧ pn * BN + idx.2.1.val < N
        then some (smS1 s0 s1_ptr (pm * BM + idx.1.val))
        else some ((((((((t.setReg "rm" .nat [BM] (smOffs (pm * BM) BM)).setReg
            "rn" .nat [BN] (smOffs (pn * BN) BN)).setReg
            "idx_m" .nat [BM, 1] (smIdxMTile BM pm)).setReg
            "idx_n" .nat [1, BN] (smIdxNTile BN pn)).setReg
            "mask" .bool [BM, BN] (smMaskTile M N BM BN pm pn)).setReg
            "xindex" .nat [BM, BN] (smXIndexTile N BM BN pm pn))).undef
            s1_ptr (pm * BM + idx.1.val))⟩ with htmp0T
  rw [stepStmts.cons_some
    (sm_store_eq c_ptr M N BM BN pm pn _
      (Op.mul .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.intToReal (Op.ref .int [BM, BN] "acc"))
        (Op.ref .real [BM, BN] "tmp0"))
      (Tile.bop NumericDType.real.mul
        (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Tile.intToReal (smAccTile s0 a_ptr b_ptr M N K sam sak sbk sbn
          BM BN BK pm pn numKBlocks))
        tmp0T)
      (fun i => some
        (((smAccVal s0 a_ptr b_ptr K sam sak sbk sbn
            ((pm * BM + i.1.val) % M) ((pn * BN + i.2.1.val) % N)
            (numKBlocks * BK) : ℤ) : ℝ)
          * smS1 s0 s1_ptr (pm * BM + i.1.val)))
      (is_mulTile_eval NumericDType.real _ _ _ _ _ _
        (is_intToReal_eval _ _ _ (by rw [evalOp_ref]; simpa using hacc))
        (by rw [evalOp_ref]; simp [htmp0T]))
      (by
        intro i hb
        obtain ⟨r, cc, u⟩ := i
        simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex,
          Tile.intToReal_data, smAccTile, htmp0T]
        rw [if_pos hb]
        rfl)
      (by simp) (by simp))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  intro idx hidx
  rw [sm_store_props c_ptr M N BM BN pm pn _ _ idx hidx]
  rw [Nat.mod_eq_of_lt hidx.1, Nat.mod_eq_of_lt hidx.2]
  rw [smAccVal_final s0 a_ptr b_ptr K sam sak sbk sbn _ _ _ hCeil]
  simp only [FloatDType.real_ofReal, FloatDType.real_storeValue,
    WithBot.unbotD_some]

/-! ## Kernel 2 — main theorem -/

set_option maxHeartbeats 1000000 in
set_option linter.unusedVariables false in
/-- **Genuine, dimension-general correctness of
`scaled_matmul_kernel_with_block_pointers`.** For every launch state and
both `EVEN_K` arms, the kernel runs to completion and every in-range output
lane of `C` holds the **`.real`-typed** memory cell carrying the exact real

`(∑ kk < K, A[row, kk] · B[kk, col] : ℤ → ℝ) · s1[row]`

at the flat inductor address `col + N·row` (`xindex = idx_n + N·idx_m`).
The `s1` scale is read at the strideless address `row` — the
`tl.broadcast_to(idx_m, …)` load; `stride_s1m`/`stride_s1n` are dead. The
host allocates `C` as int32: the float→int32 container truncation at the
store boundary is outside the model (the #154 fixed-width family — the cell
carries the exact real; see the module docstring).

Loop hypotheses: `hCeil` pins `numKBlocks` as an upper trip count covering
all of `K` (`K ≤ numKBlocks·BLOCK_K`, the ceil form — extra iterations are
all-masked and contribute nothing in the masked arm), and `hEven` demands
the exact form `K = numKBlocks·BLOCK_K` **only** of the unmasked
`EVEN_K = true` arm, whose loads have no masks to protect a ragged tail.
No `hInj` hypothesis: on the masked lanes `col < N`, so the flat address is
injective outright (`smCAddr_inj_active`). The `% M`/`% N` wraps of
`ram`/`rbn` disappear on exactly the lanes the store mask lets through
(`Nat.mod_eq_of_lt`). -/
specification int_scaled_matmul_scaled_exec_genuine
    (a_ptr b_ptr : Region .int) (c_ptr s1_ptr : RegionName)
    (M N K : Nat)
    (stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      stride_s1m stride_s1n : Nat)
    (BM BN BK GM : Nat) (EVEN_K : Bool) (numKBlocks : Nat) (s : BlockState)
    (hCeil : K ≤ numKBlocks * BK)
    (hEven : EVEN_K = Bool.true → K = numKBlocks * BK) :
    ∃ sF, exec (int_scaled_matmul_scaled_surface a_ptr b_ptr c_ptr s1_ptr M N K
        stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
        stride_s1m stride_s1n BM BN BK GM EVEN_K numKBlocks).toAlgKernel s
        = some sF
      ∧ ∀ idx : TileIndex [BM, BN],
          (smPidM s M N BM BN GM * BM + idx.1.val < M
            ∧ smPidN s M N BM BN GM * BN + idx.2.1.val < N) →
          sF.mem c_ptr (smCAddr N BM BN (smPidM s M N BM BN GM)
              (smPidN s M N BM BN GM) idx)
            = MemCell.of .real (some
                (((smSpec s a_ptr b_ptr K stride_am stride_ak stride_bk
                    stride_bn
                    (smPidM s M N BM BN GM * BM + idx.1.val)
                    (smPidN s M N BM BN GM * BN + idx.2.1.val) : ℤ) : ℝ)
                  * smS1 s s1_ptr (smPidM s M N BM BN GM * BM + idx.1.val))) := by
  rw [exec, sm_body_eq]
  obtain ⟨t1, hrun1, h1inv⟩ :=
    smPreLoop_run s a_ptr b_ptr M N K stride_am stride_ak stride_bk stride_bn
      BM BN BK GM numKBlocks
  simp only [List.append_assoc]
  rw [stepStmts.append_some hrun1]
  obtain ⟨t2, hrun2, h2inv⟩ :=
    smLoop_collapse s a_ptr b_ptr M N K stride_am stride_ak stride_bk stride_bn
      BM BN BK GM numKBlocks EVEN_K t1 hEven h1inv
  rw [show [Stmt.forRange "j" 0 numKBlocks 1
          (smLoopBody K stride_ak stride_bk BM BN BK EVEN_K)]
        ++ smPostLoop c_ptr s1_ptr M N BM BN
      = Stmt.forRange "j" 0 numKBlocks 1
          (smLoopBody K stride_ak stride_bk BM BN BK EVEN_K)
        :: smPostLoop c_ptr s1_ptr M N BM BN from rfl]
  rw [stepStmts.cons_some hrun2]
  obtain ⟨sF, hpost, hout⟩ :=
    smPostLoop_run s a_ptr b_ptr c_ptr s1_ptr M N K stride_am stride_ak
      stride_bk stride_bn BM BN BK GM numKBlocks t2 hCeil h2inv
  exact ⟨sF, hpost, hout⟩

end Correct_without_Rounding

end VeriTile.Bench.TritonBenchG.IntScaledMatmul


