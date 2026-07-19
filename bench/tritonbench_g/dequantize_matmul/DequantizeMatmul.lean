import VeriTile.Triton

/-!
# `dequantize_matmul` — strict per-kernel correctness

`dequantize_kernel` is the dequantization prologue of an int8 matmul: program
`(k_block_idx, n_block_idx)` loads a `[BLOCK_SIZE_K, BLOCK_SIZE_N]` tile of the
int8 weight `b`, loads the per-column scale `b_scale`, and stores the
dequantized `int_b * scale_b` into `fpb`, masked by
`(k·BSK + offs_k < K) & (n·BSN + offs_n < N)`. The downstream `torch.mm(a, fp_b)`
is host code, not part of this kernel.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`dequantize_kernel[grid](...)`, the 2-D grid
`(cdiv(K, BSK), cdiv(N, BSN))`, the strides, the subsequent `torch.mm`, and the
runtime composition of per-program writes) is the *trusted boundary*. Both
program ids (`k_block_idx`, `n_block_idx`) are universally quantified, so the
per-program statement covers every program of the grid.

The headline is stated on the kernel's masked **IO signature** `dequantizeIO`
(`Masked2DKernelIO₂`): which buffer is which argument (weight `b_ptr` = `in1`,
column scale `b_scale_ptr` = `in2`, dequantized output `fpb_ptr` = `out`), the
`[BLOCK_SIZE_K, BLOCK_SIZE_N]` tile flattened row-major into
`B = BLOCK_SIZE_K * BLOCK_SIZE_N` lanes (`laneIdx`: lane `j` = tile cell
`(j / BSN, j % BSN)`), the strided per-lane windows, and — the reason this
kernel needs the `read2Mask` field — **two different read gates**: the weight
tile is read under the full mask `k·BSK + i < K ∧ n·BSN + j < N` while the
column scale is read under the wider column-only gate `n·BSN + j < N`.

`⊨` (`Masked2DKernelIO₂.Implements`) is the audit-once masked Hoare-triple
combinator: for **every** disjoint flat placement of the three buffers,
**every** program-id pair whose gated lanes are in bounds, and **every** launch
state whose windows hold `xs` (weight) and `ys` (scale) at their respective
gates, the translated pointer kernel terminates, every write-active lane of
`fpb_ptr` holds `xs j * ys j`, and every other memory cell is unchanged.

## Proof architecture

```
dequantize_kernel_correctness            ← TOP SPECIFICATION (dequantizeIO ⊨ xs * ys)
  ├─ dequantize_kernel_flattenOk         bridge fragment membership
  ├─ dequantize_kernel_traceSafe         per-execution lane-wise safety walk
  └─ dequantize_kernel_region_run        region-model masked Hoare triple
       ├─ dequantize_kernel_exec_isSome  termination
       ├─ dequantize_kernel_correct      ← algorithm-layer readback per lane
       └─ dequantize_kernel_frame        masked scatter-store cell frame
```

## Modeling boundary

Arithmetic is over `ℝ`, not bit-accurate IEEE float. The quantization semantics
here are the **dequantization direction only**: the proof models `int_b *
scale_b` as a plain real product, where `int_b` is the value loaded from the
int8 weight region and `scale_b` the loaded float scale. There is no rounding or
int cast on the output side, so this kernel has no `to(tl.int8)` / `llrint`
blocker — `int_b` is taken as the loaded real value and the store is exact. The
masked `other=0.0` defaults are modeled (they are invisible in the headline:
the conclusion only speaks about write-active lanes, where both loads are
gated on). `@triton.autotune` is not modeled — the headline is general in
`BLOCK_SIZE_N`/`BLOCK_SIZE_K`, so it covers every autotune config at once.

Two honest side hypotheses, both **required for truth**:

* `hOutInj` — distinct lanes of a program's output tile must not alias the same
  `fp_b` cell; otherwise only the last store survives and the per-lane claim is
  false. The host's row-major `fp_b` satisfies it.
* `0 < BLOCK_SIZE_K` — the column-scale load has tile shape `[1, BLOCK_SIZE_N]`
  and happens *unconditionally on the K axis*, so with `BLOCK_SIZE_K = 0` the
  lane set `Fin (BSK * BSN)` is empty and the signature can place **no** bounds
  hypothesis on that load, leaving the flat-memory bridge's trace-safety
  obligation unprovable. Every real autotune config has `BLOCK_SIZE_K ≥ 32`.
-/

namespace VeriTile.Bench.TritonBenchG.DequantizeMatmul

open VeriTile.Triton
open scoped VeriTile.Triton.Masked2DKernelIO₂

/-- Faithful 1:1 transcription of `dequantize_matmul.py`'s `dequantize_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE_N: tl.constexpr` / `BLOCK_SIZE_K: tl.constexpr` → Lean
  `Nat` parameters. -/
def dequantize_kernel
    (b_ptr b_scale_ptr fpb_ptr : RegionName)
    (K N stride_bk stride_bn stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K : Nat) :
    ComputeKernel := triton {
  k_block_idx = tl.program_id(axis=0)
  n_block_idx = tl.program_id(axis=1)
  offs_k = tl.arange(0, $(BLOCK_SIZE_K))
  offs_n = tl.arange(0, $(BLOCK_SIZE_N))
  b_offs = (k_block_idx * $(BLOCK_SIZE_K) + offs_k[:, None]) * $(stride_bk) +
    (n_block_idx * $(BLOCK_SIZE_N) + offs_n[None, :]) * $(stride_bn)
  fpb_offs = (k_block_idx * $(BLOCK_SIZE_K) + offs_k[:, None]) * $(stride_fpbk) +
    (n_block_idx * $(BLOCK_SIZE_N) + offs_n[None, :]) * $(stride_fpbn)
  bs_offs = n_block_idx * $(BLOCK_SIZE_N) + offs_n[None, :]
  n_mask = n_block_idx * $(BLOCK_SIZE_N) + offs_n[None, :] < $(N)
  mask = (k_block_idx * $(BLOCK_SIZE_K) + offs_k[:, None] < $(K)) & n_mask
  int_b = tl.load(b_ptr + b_offs, mask=mask, other=0.0)
  scale_b = tl.load(b_scale_ptr + bs_offs, mask=n_mask, other=0.0)
  tl.store(fpb_ptr + fpb_offs, int_b * scale_b, mask=mask)
}

/-! ### Row-major flattening of the 2-D tile into `⊨` lanes -/

/-- Lane `j` of the row-major flattening of an `R × C` tile. -/
def laneIdx (R C : Nat) (j : Fin (R * C)) : TileIndex [R, C] :=
  (⟨j.val / C, Nat.div_lt_of_lt_mul
      (Nat.lt_of_lt_of_le j.isLt (Nat.le_of_eq (Nat.mul_comm R C)))⟩,
   ⟨j.val % C, Nat.mod_lt _ (Nat.pos_of_ne_zero (by
      rintro rfl
      exact absurd j.isLt (by simp)))⟩,
   PUnit.unit)

/-- The lane of a logical `R × C` tile index — inverse of `laneIdx`. -/
def laneOf (R C : Nat) (idx : TileIndex [R, C]) : Fin (R * C) :=
  ⟨idx.1.val * C + idx.2.1.val, by
    have h1 : idx.1.val + 1 ≤ R := idx.1.isLt
    have h2 : idx.1.val * C + idx.2.1.val < idx.1.val * C + C :=
      Nat.add_lt_add_left idx.2.1.isLt _
    refine Nat.lt_of_lt_of_le h2 ?_
    rw [← Nat.succ_mul]
    exact Nat.mul_le_mul_right C h1⟩

@[simp] theorem laneIdx_laneOf (R C : Nat) (idx : TileIndex [R, C]) :
    laneIdx R C (laneOf R C idx) = idx := by
  obtain ⟨a, b, u⟩ := idx
  have hC : 0 < C := Nat.lt_of_le_of_lt (Nat.zero_le _) b.isLt
  have hdiv : (a.val * C + b.val) / C = a.val := by
    rw [Nat.mul_comm, Nat.mul_add_div hC, Nat.div_eq_of_lt b.isLt, Nat.add_zero]
  have hmod : (a.val * C + b.val) % C = b.val := by
    rw [Nat.mul_comm, Nat.mul_add_mod, Nat.mod_eq_of_lt b.isLt]
  simp only [laneIdx, laneOf, hdiv, hmod]

theorem laneOf_injective (R C : Nat) : Function.Injective (laneOf R C) :=
  Function.LeftInverse.injective (laneIdx_laneOf R C)

/-- Tile-index injectivity from the lane-indexed injectivity the `⊨` headline
states (`laneIdx` is onto the tile indices). -/
theorem inj_of_lane_inj {R C : Nat} {F : TileIndex [R, C] → Nat}
    (h : Function.Injective (fun j : Fin (R * C) => F (laneIdx R C j))) :
    Function.Injective F := by
  intro a b hab
  have hcomp : F (laneIdx R C (laneOf R C a)) = F (laneIdx R C (laneOf R C b)) := by
    rw [laneIdx_laneOf, laneIdx_laneOf]; exact hab
  exact laneOf_injective R C (h hcomp)

/-! ### Windows, gates and the per-lane spec -/

/-- Weight-tile read address of tile cell `idx` for program `(p₀, p₁)`. -/
def bOffset (p₀ p₁ stride_bk stride_bn BLOCK_SIZE_N BLOCK_SIZE_K : Nat)
    (idx : TileIndex [BLOCK_SIZE_K, BLOCK_SIZE_N]) : Nat :=
  (p₀ * BLOCK_SIZE_K + idx.1.val) * stride_bk +
    (p₁ * BLOCK_SIZE_N + idx.2.1.val) * stride_bn

/-- Output write address of tile cell `idx` for program `(p₀, p₁)`. -/
def fpbOffset (p₀ p₁ stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K : Nat)
    (idx : TileIndex [BLOCK_SIZE_K, BLOCK_SIZE_N]) : Nat :=
  (p₀ * BLOCK_SIZE_K + idx.1.val) * stride_fpbk +
    (p₁ * BLOCK_SIZE_N + idx.2.1.val) * stride_fpbn

/-- Column-scale read address of tile column `j` for program `(_, p₁)`. -/
def nOffset (p₁ BLOCK_SIZE_N : Nat) (j : Fin BLOCK_SIZE_N) : Nat :=
  p₁ * BLOCK_SIZE_N + j.val

/-- The kernel's store gate `mask` = `(k·BSK + i < K) & (n·BSN + j < N)`. -/
def dequantizeActive (p₀ p₁ K N BLOCK_SIZE_N BLOCK_SIZE_K : Nat)
    (idx : TileIndex [BLOCK_SIZE_K, BLOCK_SIZE_N]) : Prop :=
  p₀ * BLOCK_SIZE_K + idx.1.val < K ∧ p₁ * BLOCK_SIZE_N + idx.2.1.val < N

instance dequantizeActiveDecidable
    (p₀ p₁ K N BLOCK_SIZE_N BLOCK_SIZE_K : Nat)
    (idx : TileIndex [BLOCK_SIZE_K, BLOCK_SIZE_N]) :
    Decidable (dequantizeActive p₀ p₁ K N BLOCK_SIZE_N BLOCK_SIZE_K idx) := by
  unfold dequantizeActive
  infer_instance

/-- State-coupled readback spec (internal to `dequantize_kernel_correct`): the
exact dequantized value written at active tile lane `idx`. The `⊨` headline
re-expresses it as the pure product `xs j * ys j` of the loaded tiles. -/
noncomputable def dequantizeSpec
    (s : BlockState) (b_ptr b_scale_ptr : RegionName)
    (p₀ p₁ stride_bk stride_bn BLOCK_SIZE_N BLOCK_SIZE_K : Nat)
    (idx : TileIndex [BLOCK_SIZE_K, BLOCK_SIZE_N]) : ℝ :=
  s.readMem b_ptr (bOffset p₀ p₁ stride_bk stride_bn BLOCK_SIZE_N BLOCK_SIZE_K idx) *
    s.readMem b_scale_ptr (nOffset p₁ BLOCK_SIZE_N idx.2.1)

private noncomputable def dequantizeStoreBase
    (s : BlockState) (b_ptr b_scale_ptr : RegionName)
    (K N stride_bk stride_bn stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K : Nat) :
    BlockState :=
  ((((((((((s.setReg "k_block_idx" .nat [] (Tile.scalar (s.pids 0))).setReg
    "n_block_idx" .nat [] (Tile.scalar (s.pids 1))).setReg
    "offs_k" .nat [BLOCK_SIZE_K] (Tile.vec fun i => i.val)).setReg
    "offs_n" .nat [BLOCK_SIZE_N] (Tile.vec fun i => i.val)).setReg
    "b_offs" .nat [BLOCK_SIZE_K, BLOCK_SIZE_N]
      { data := fun i => bOffset (s.pids 0) (s.pids 1) stride_bk stride_bn
          BLOCK_SIZE_N BLOCK_SIZE_K i }).setReg
    "fpb_offs" .nat [BLOCK_SIZE_K, BLOCK_SIZE_N]
      { data := fun i => fpbOffset (s.pids 0) (s.pids 1) stride_fpbk stride_fpbn
          BLOCK_SIZE_N BLOCK_SIZE_K i }).setReg
    "bs_offs" .nat [1, BLOCK_SIZE_N]
      { data := fun i => nOffset (s.pids 1) BLOCK_SIZE_N i.2.1 }).setReg
    "n_mask" .bool [1, BLOCK_SIZE_N]
      { data := fun i => decide (nOffset (s.pids 1) BLOCK_SIZE_N i.2.1 < N) }).setReg
    "mask" .bool [BLOCK_SIZE_K, BLOCK_SIZE_N]
      { data := fun i => decide (dequantizeActive (s.pids 0) (s.pids 1) K N
          BLOCK_SIZE_N BLOCK_SIZE_K i) }).setReg
    "int_b" .real [BLOCK_SIZE_K, BLOCK_SIZE_N]
      { data := fun i =>
          if dequantizeActive (s.pids 0) (s.pids 1) K N BLOCK_SIZE_N BLOCK_SIZE_K i then
            some (s.readMem b_ptr (bOffset (s.pids 0) (s.pids 1) stride_bk stride_bn
              BLOCK_SIZE_N BLOCK_SIZE_K i))
          else some (0.0 : ℝ) }).setReg
    "scale_b" .real [1, BLOCK_SIZE_N]
      { data := fun i =>
          if nOffset (s.pids 1) BLOCK_SIZE_N i.2.1 < N then
            some (s.readMem b_scale_ptr (nOffset (s.pids 1) BLOCK_SIZE_N i.2.1))
          else some (0.0 : ℝ) }

/-- Algorithm-layer correctness for `dequantize_kernel`. -/
theorem dequantize_kernel_correct
    (b_ptr b_scale_ptr fpb_ptr : RegionName)
    (K N stride_bk stride_bn stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fpbOffset (s.pids 0) (s.pids 1) stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K))
    (hExec : exec (dequantize_kernel b_ptr b_scale_ptr fpb_ptr K N stride_bk stride_bn
          stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K) s = some s') :
    ∀ idx : TileIndex [BLOCK_SIZE_K, BLOCK_SIZE_N],
      s'.readMem fpb_ptr
          (fpbOffset (s.pids 0) (s.pids 1) stride_fpbk stride_fpbn
            BLOCK_SIZE_N BLOCK_SIZE_K idx)
        = if dequantizeActive (s.pids 0) (s.pids 1) K N BLOCK_SIZE_N BLOCK_SIZE_K idx then
          dequantizeSpec s b_ptr b_scale_ptr (s.pids 0) (s.pids 1) stride_bk stride_bn
            BLOCK_SIZE_N BLOCK_SIZE_K idx
        else
          s.readMem fpb_ptr
            (fpbOffset (s.pids 0) (s.pids 1) stride_fpbk stride_fpbn
              BLOCK_SIZE_N BLOCK_SIZE_K idx) := by
  intro idx
  simp [exec, dequantize_kernel, stepStmts, stepStmt, evalOp.eq_def, Option.bind,
        Tile.bop, Tile.cop, Tile.expandDim,
        NumericDType.add, NumericDType.mul, ComparableDType.lt,
        TileShape.dropInsertedIndex] at hExec
  rw [← hExec]
  let offsetFn : TileIndex [BLOCK_SIZE_K, BLOCK_SIZE_N] → Nat :=
    fun idx =>
      (s.pids 0 * BLOCK_SIZE_K + idx.1.val) * stride_fpbk +
        (s.pids 1 * BLOCK_SIZE_N + idx.2.1.val) * stride_fpbn
  let active : TileIndex [BLOCK_SIZE_K, BLOCK_SIZE_N] → Prop :=
    fun idx =>
      s.pids 0 * BLOCK_SIZE_K + idx.1.val < K ∧
        s.pids 1 * BLOCK_SIZE_N + idx.2.1.val < N
  let valueFn : TileIndex [BLOCK_SIZE_K, BLOCK_SIZE_N] → ℝ :=
    fun idx =>
      WithBot.unbotD 0
        (Option.map₂ (fun x1 x2 => x1 * x2)
          (if active idx then
            some
              (s.readMem b_ptr
                ((s.pids 0 * BLOCK_SIZE_K + idx.1.val) * stride_bk +
                  (s.pids 1 * BLOCK_SIZE_N + idx.2.1.val) * stride_bn))
          else some (0.0 : ℝ))
          (if s.pids 1 * BLOCK_SIZE_N + idx.2.1.val < N then
            some
              (s.readMem b_scale_ptr
                (s.pids 1 * BLOCK_SIZE_N + idx.2.1.val))
          else some (0.0 : ℝ)))
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, fpbOffset] using hOutInj
  have hscatter := BlockState.scatter_readback_prop_masked_nd
    (region := fpb_ptr)
    (s := dequantizeStoreBase s b_ptr b_scale_ptr K N stride_bk stride_bn
      stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K)
    (offsetFn := offsetFn) (valueFn := valueFn) (P := active)
    hOffsetInj idx
  by_cases hActive : active idx
  · rcases hActive with ⟨hk, hn⟩
    simpa [offsetFn, active, valueFn, dequantizeActive, dequantizeSpec,
      bOffset, fpbOffset, nOffset, hk, hn,
      dequantizeStoreBase, TileShape.dropInsertedIndex, Bool.and_eq_true] using hscatter
  · simpa [offsetFn, active, valueFn, dequantizeActive, dequantizeSpec,
      bOffset, fpbOffset, nOffset, hActive,
      dequantizeStoreBase, TileShape.dropInsertedIndex] using hscatter

/-- A masked scatter-store `foldl` leaves every memory cell it does not
actively hit unchanged (cell-level frame for the masked store). -/
private theorem foldl_store_preserve_cell {α : Type} {region : RegionName}
    (offsetFn : α → Nat) (valueFn : α → ℝ) (P : α → Prop) [DecidablePred P]
    (r : RegionName) (o : Nat) (l : List α) (s : BlockState)
    (hnot : ∀ k ∈ l, P k → ¬(region = r ∧ offsetFn k = o)) :
    (l.foldl (fun acc k =>
        if P k then acc.writeMem region (offsetFn k) (valueFn k) else acc)
      s).mem r o = s.mem r o := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
      rw [List.foldl_cons]
      by_cases hP : P hd
      · rw [if_pos hP,
          ih _ (fun k hk => hnot k (List.mem_cons_of_mem hd hk)),
          BlockState.writeMem_mem]
        exact if_neg (fun hc =>
          hnot hd List.mem_cons_self hP ⟨hc.1.symm, hc.2.symm⟩)
      · rw [if_neg hP]
        exact ih _ (fun k hk => hnot k (List.mem_cons_of_mem hd hk))

set_option maxHeartbeats 1600000 in
/-- Frame half: every memory cell not actively written by the masked output
store — every cell of every region other than `fpb_ptr`, and the masked-off
lanes of the output window itself — is preserved by the run. -/
private theorem dequantize_kernel_frame
    (b_ptr b_scale_ptr fpb_ptr : RegionName)
    (K N stride_bk stride_bn stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K : Nat)
    (s s1 : BlockState)
    (hExec : exec ((dequantize_kernel b_ptr b_scale_ptr fpb_ptr K N stride_bk stride_bn
        stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K).toAlgKernel) s = some s1)
    (r : RegionName) (o : Nat)
    (hmiss : ∀ idx : TileIndex [BLOCK_SIZE_K, BLOCK_SIZE_N],
      dequantizeActive (s.pids 0) (s.pids 1) K N BLOCK_SIZE_N BLOCK_SIZE_K idx →
      ¬(fpb_ptr = r ∧ fpbOffset (s.pids 0) (s.pids 1) stride_fpbk stride_fpbn
          BLOCK_SIZE_N BLOCK_SIZE_K idx = o)) :
    s1.mem r o = s.mem r o := by
  simp [exec, dequantize_kernel, ComputeKernel.toAlgKernel, stepStmts, stepStmt,
        evalOp.eq_def, Option.bind, ComputeExpr.toAlgorithm?,
        Tile.bop, Tile.cop, Tile.expandDim,
        NumericDType.add, NumericDType.mul, ComparableDType.lt,
        TileShape.dropInsertedIndex] at hExec
  subst hExec
  refine Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_) rfl
  intro k _ hmk hc
  refine hmiss k ?_ (by simpa [fpbOffset] using hc)
  simpa [dequantizeActive, Bool.and_eq_true] using hmk

set_option maxHeartbeats 1600000 in
/-- Termination: the kernel executes to completion from any state (a
straight-line masked load / load / multiply / masked store). -/
private theorem dequantize_kernel_exec_isSome
    (b_ptr b_scale_ptr fpb_ptr : RegionName)
    (K N stride_bk stride_bn stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K : Nat)
    (s : BlockState) :
    ∃ s1, exec ((dequantize_kernel b_ptr b_scale_ptr fpb_ptr K N stride_bk stride_bn
        stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K).toAlgKernel) s = some s1 := by
  simp [exec, dequantize_kernel, ComputeKernel.toAlgKernel, stepStmts, stepStmt,
        evalOp.eq_def, Option.bind, ComputeExpr.toAlgorithm?,
        Tile.bop, Tile.cop, Tile.expandDim,
        NumericDType.add, NumericDType.mul, ComparableDType.lt,
        TileShape.dropInsertedIndex]

/-- The kernel sits inside the flat-memory bridge's covered fragment (pointer
arithmetic, two masked loads and a masked store). -/
theorem dequantize_kernel_flattenOk
    (b_ptr b_scale_ptr fpb_ptr : RegionName)
    (K N stride_bk stride_bn stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K : Nat) :
    ((dequantize_kernel b_ptr b_scale_ptr fpb_ptr K N stride_bk stride_bn
        stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [dequantize_kernel, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

set_option maxHeartbeats 1600000 in
/-- Per-execution safety walk: the two `program_id`s, the two `arange`s, the
three offset tiles and the two mask tiles are memory-silent; the masked weight
load, the column-scale load (gated by the wider `n_mask`) and the masked store
reduce to the lane-wise bounds hypotheses. The `0 < BLOCK_SIZE_K` hypothesis is
what lets an arbitrary scale column `n` be exhibited as a lane of the flattened
`[BLOCK_SIZE_K, BLOCK_SIZE_N]` tile. -/
theorem dequantize_kernel_traceSafe
    (b_ptr b_scale_ptr fpb_ptr : RegionName)
    (K N stride_bk stride_bn stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K : Nat)
    (hBK : 0 < BLOCK_SIZE_K)
    (bounds : RegionBounds) (s : BlockState)
    (h1 : ∀ j : Fin (BLOCK_SIZE_K * BLOCK_SIZE_N),
      dequantizeActive (s.pids 0) (s.pids 1) K N BLOCK_SIZE_N BLOCK_SIZE_K
        (laneIdx BLOCK_SIZE_K BLOCK_SIZE_N j) →
      bOffset (s.pids 0) (s.pids 1) stride_bk stride_bn BLOCK_SIZE_N BLOCK_SIZE_K
        (laneIdx BLOCK_SIZE_K BLOCK_SIZE_N j) < bounds b_ptr)
    (h2 : ∀ j : Fin (BLOCK_SIZE_K * BLOCK_SIZE_N),
      nOffset (s.pids 1) BLOCK_SIZE_N (laneIdx BLOCK_SIZE_K BLOCK_SIZE_N j).2.1 < N →
      nOffset (s.pids 1) BLOCK_SIZE_N (laneIdx BLOCK_SIZE_K BLOCK_SIZE_N j).2.1
        < bounds b_scale_ptr)
    (h3 : ∀ j : Fin (BLOCK_SIZE_K * BLOCK_SIZE_N),
      dequantizeActive (s.pids 0) (s.pids 1) K N BLOCK_SIZE_N BLOCK_SIZE_K
        (laneIdx BLOCK_SIZE_K BLOCK_SIZE_N j) →
      fpbOffset (s.pids 0) (s.pids 1) stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K
        (laneIdx BLOCK_SIZE_K BLOCK_SIZE_N j) < bounds fpb_ptr) :
    Kernel.TraceSafe bounds
      ((dequantize_kernel b_ptr b_scale_ptr fpb_ptr K N stride_bk stride_bn
        stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K).toAlgKernel) s := by
  have h1' : ∀ (a : Fin BLOCK_SIZE_K) (b : Fin BLOCK_SIZE_N),
      s.pids 0 * BLOCK_SIZE_K + a.val < K →
      s.pids 1 * BLOCK_SIZE_N + b.val < N →
      (s.pids 0 * BLOCK_SIZE_K + a.val) * stride_bk +
        (s.pids 1 * BLOCK_SIZE_N + b.val) * stride_bn < bounds b_ptr := by
    intro a b ha hb
    have := h1 (laneOf BLOCK_SIZE_K BLOCK_SIZE_N (a, b, PUnit.unit))
      (by simpa [dequantizeActive] using And.intro ha hb)
    simpa [bOffset] using this
  have h2' : ∀ b : Fin BLOCK_SIZE_N,
      s.pids 1 * BLOCK_SIZE_N + b.val < N →
      s.pids 1 * BLOCK_SIZE_N + b.val < bounds b_scale_ptr := by
    intro b hb
    have := h2 (laneOf BLOCK_SIZE_K BLOCK_SIZE_N (⟨0, hBK⟩, b, PUnit.unit))
      (by simpa [nOffset] using hb)
    simpa [nOffset] using this
  have h3' : ∀ (a : Fin BLOCK_SIZE_K) (b : Fin BLOCK_SIZE_N),
      s.pids 0 * BLOCK_SIZE_K + a.val < K →
      s.pids 1 * BLOCK_SIZE_N + b.val < N →
      (s.pids 0 * BLOCK_SIZE_K + a.val) * stride_fpbk +
        (s.pids 1 * BLOCK_SIZE_N + b.val) * stride_fpbn < bounds fpb_ptr := by
    intro a b ha hb
    have := h3 (laneOf BLOCK_SIZE_K BLOCK_SIZE_N (a, b, PUnit.unit))
      (by simpa [dequantizeActive] using And.intro ha hb)
    simpa [fpbOffset] using this
  unfold Kernel.TraceSafe
  simp [dequantize_kernel, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
    MaskOpt.SafeAt, MemAccess.SafeAt,
    stepStmt, evalOp.eq_def, Option.bind, MemAccess.ActiveAddressSafe,
    memAccessActiveAddressSafe, MaskOpt.Active, BlockState.setReg,
    Tile.bop, Tile.cop, Tile.expandDim,
    NumericDType.add, NumericDType.mul, ComparableDType.lt,
    TileShape.dropInsertedIndex]
  refine ⟨?_, ?_, ?_⟩ <;> intros <;>
    first
      | (apply h1' <;> assumption)
      | (apply h2' <;> assumption)
      | (apply h3' <;> assumption)

set_option maxHeartbeats 1600000 in
/-- **The region-model masked Hoare triple** — termination, write-active-lane
output values, and frame off the write-active output lanes, from any launch
state whose weight window holds `xs` at the full mask and whose scale window
holds `ys` at the column gate. This is the `hrun` obligation of the `⊨`
headline. -/
theorem dequantize_kernel_region_run
    (b_ptr b_scale_ptr fpb_ptr : RegionName)
    (K N stride_bk stride_bn stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K : Nat)
    (s₀ : BlockState) (xs ys : Fin (BLOCK_SIZE_K * BLOCK_SIZE_N) → ℝ)
    (hOutInj : Function.Injective
      (fpbOffset (s₀.pids 0) (s₀.pids 1) stride_fpbk stride_fpbn
        BLOCK_SIZE_N BLOCK_SIZE_K))
    (hx : ∀ j : Fin (BLOCK_SIZE_K * BLOCK_SIZE_N),
      dequantizeActive (s₀.pids 0) (s₀.pids 1) K N BLOCK_SIZE_N BLOCK_SIZE_K
        (laneIdx BLOCK_SIZE_K BLOCK_SIZE_N j) →
      s₀.readMem b_ptr (bOffset (s₀.pids 0) (s₀.pids 1) stride_bk stride_bn
        BLOCK_SIZE_N BLOCK_SIZE_K (laneIdx BLOCK_SIZE_K BLOCK_SIZE_N j)) = xs j)
    (hy : ∀ j : Fin (BLOCK_SIZE_K * BLOCK_SIZE_N),
      nOffset (s₀.pids 1) BLOCK_SIZE_N (laneIdx BLOCK_SIZE_K BLOCK_SIZE_N j).2.1 < N →
      s₀.readMem b_scale_ptr (nOffset (s₀.pids 1) BLOCK_SIZE_N
        (laneIdx BLOCK_SIZE_K BLOCK_SIZE_N j).2.1) = ys j) :
    ∃ s1, exec ((dequantize_kernel b_ptr b_scale_ptr fpb_ptr K N stride_bk stride_bn
        stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K).toAlgKernel) s₀ = some s1
      ∧ (∀ j : Fin (BLOCK_SIZE_K * BLOCK_SIZE_N),
          dequantizeActive (s₀.pids 0) (s₀.pids 1) K N BLOCK_SIZE_N BLOCK_SIZE_K
            (laneIdx BLOCK_SIZE_K BLOCK_SIZE_N j) →
          s1.readMem fpb_ptr (fpbOffset (s₀.pids 0) (s₀.pids 1) stride_fpbk stride_fpbn
            BLOCK_SIZE_N BLOCK_SIZE_K (laneIdx BLOCK_SIZE_K BLOCK_SIZE_N j))
            = xs j * ys j)
      ∧ (∀ r o,
          (r ≠ fpb_ptr ∨ ∀ j : Fin (BLOCK_SIZE_K * BLOCK_SIZE_N),
            dequantizeActive (s₀.pids 0) (s₀.pids 1) K N BLOCK_SIZE_N BLOCK_SIZE_K
              (laneIdx BLOCK_SIZE_K BLOCK_SIZE_N j) →
            o ≠ fpbOffset (s₀.pids 0) (s₀.pids 1) stride_fpbk stride_fpbn
              BLOCK_SIZE_N BLOCK_SIZE_K (laneIdx BLOCK_SIZE_K BLOCK_SIZE_N j)) →
          s1.mem r o = s₀.mem r o) := by
  obtain ⟨s1, hs1⟩ := dequantize_kernel_exec_isSome b_ptr b_scale_ptr fpb_ptr K N
    stride_bk stride_bn stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K s₀
  have hs1' : exec (dequantize_kernel b_ptr b_scale_ptr fpb_ptr K N stride_bk
      stride_bn stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K) s₀ = some s1 := hs1
  refine ⟨s1, hs1, fun j hj => ?_, fun r o hcond => ?_⟩
  · have h := dequantize_kernel_correct b_ptr b_scale_ptr fpb_ptr K N stride_bk
      stride_bn stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K s₀ s1 hOutInj hs1'
      (laneIdx BLOCK_SIZE_K BLOCK_SIZE_N j)
    rw [if_pos hj] at h
    rw [h, dequantizeSpec, hx j hj, hy j (by simpa [nOffset] using hj.2)]
  · refine dequantize_kernel_frame b_ptr b_scale_ptr fpb_ptr K N stride_bk stride_bn
      stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K s₀ s1 hs1 r o
      (fun idx hact ⟨hr, ho⟩ => ?_)
    rcases hcond with hne | hno
    · exact hne hr.symm
    · exact hno (laneOf BLOCK_SIZE_K BLOCK_SIZE_N idx)
        (by rw [laneIdx_laneOf]; exact hact)
        (by rw [laneIdx_laneOf]; exact ho.symm)

/-- `dequantize_kernel`'s masked **IO signature** — the whole kernel-specific
audit surface of the `⊨` headline:

* `in1`/`in2`/`out` — which buffer is which argument (weight `b_ptr`, column
  scale `b_scale_ptr`, dequantized output `fpb_ptr`);
* `B = BLOCK_SIZE_K * BLOCK_SIZE_N` — the tile flattened row-major into lanes,
  lane `j` = tile cell `(j / BSN, j % BSN)` (`laneIdx`);
* `read1`/`write` — the strided weight / output windows at row
  `pid₀ * BSK + i` and column `pid₁ * BSN + j`;
* `read2` — the column-scale window `pid₁ * BSN + j` (row-independent: the
  scale is broadcast down the K axis);
* `mask` — the store gate `pid₀ * BSK + i < K ∧ pid₁ * BSN + j < N`, which is
  also the weight-load gate; `writeMask` keeps the struct default;
* `read2Mask` — the **wider** column-only gate `pid₁ * BSN + j < N`, exactly
  the kernel's `n_mask`.

The windows and gates are declared, not parsed from the kernel; the headline
**proves** the kernel's actual addressing and masking match them. -/
def dequantizeIO (b_ptr b_scale_ptr fpb_ptr : RegionName)
    (K N stride_bk stride_bn stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K : Nat) :
    Masked2DKernelIO₂ where
  kernel := dequantize_kernel b_ptr b_scale_ptr fpb_ptr K N stride_bk stride_bn
    stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K
  in1 := b_ptr
  in2 := b_scale_ptr
  out := fpb_ptr
  B := BLOCK_SIZE_K * BLOCK_SIZE_N
  read1 := fun p₀ p₁ j =>
    bOffset p₀ p₁ stride_bk stride_bn BLOCK_SIZE_N BLOCK_SIZE_K
      (laneIdx BLOCK_SIZE_K BLOCK_SIZE_N j)
  read2 := fun _ p₁ j =>
    nOffset p₁ BLOCK_SIZE_N (laneIdx BLOCK_SIZE_K BLOCK_SIZE_N j).2.1
  write := fun p₀ p₁ j =>
    fpbOffset p₀ p₁ stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K
      (laneIdx BLOCK_SIZE_K BLOCK_SIZE_N j)
  mask := fun p₀ p₁ j =>
    dequantizeActive p₀ p₁ K N BLOCK_SIZE_N BLOCK_SIZE_K
      (laneIdx BLOCK_SIZE_K BLOCK_SIZE_N j)
  read2Mask := fun _ p₁ j =>
    nOffset p₁ BLOCK_SIZE_N (laneIdx BLOCK_SIZE_K BLOCK_SIZE_N j).2.1 < N

/-- **The headline**: `dequantize_kernel` implements the elementwise
dequantization `int_b * scale_b` on its masked IO signature — for every
disjoint flat placement of the three buffers, every program-id pair whose
gated lanes are in bounds, and every launch state whose weight window holds
`xs` at the full mask and whose column-scale window holds `ys` at the wider
column gate, the translated pointer kernel terminates, every write-active lane
`j` of `fp_b` holds `xs j * ys j`, and every other memory cell — including the
masked-off lanes of the output window — is unchanged.

Two honest, **truth-required** side hypotheses: `hOutInj` (distinct lanes of a
program's output tile must not alias the same `fp_b` cell — otherwise only the
last store survives) and `0 < BLOCK_SIZE_K` (with an empty K block the lane set
is empty, so the signature can state no bounds hypothesis at all for the
`[1, BLOCK_SIZE_N]`-shaped column-scale load, which the kernel performs
regardless). -/
specification dequantize_kernel_correctness
    (b_ptr b_scale_ptr fpb_ptr : RegionName)
    (K N stride_bk stride_bn stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K : Nat)
    (hBK : 0 < BLOCK_SIZE_K)
    (hOutInj : ∀ p₀ p₁ : Nat, Function.Injective
      (fun j : Fin (BLOCK_SIZE_K * BLOCK_SIZE_N) =>
        fpbOffset p₀ p₁ stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K
          (laneIdx BLOCK_SIZE_K BLOCK_SIZE_N j))) :
    dequantizeIO b_ptr b_scale_ptr fpb_ptr K N stride_bk stride_bn
        stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K
      ⊨ fun _ _ xs ys j => xs j * ys j := by
  refine Masked2DKernelIO₂.Implements.intro _ ?_ ?_ ?_
  · exact dequantize_kernel_flattenOk b_ptr b_scale_ptr fpb_ptr K N stride_bk
      stride_bn stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K
  · intro bounds s hb hs hout _
    exact dequantize_kernel_traceSafe b_ptr b_scale_ptr fpb_ptr K N stride_bk
      stride_bn stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K hBK bounds s
      hb hs hout
  · intro s₀ xs ys hx hy
    obtain ⟨s1, hs1, hval, hframe⟩ :=
      dequantize_kernel_region_run b_ptr b_scale_ptr fpb_ptr K N stride_bk
        stride_bn stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K s₀ xs ys
        (inj_of_lane_inj (hOutInj (s₀.pids 0) (s₀.pids 1))) hx hy
    exact ⟨s1, hs1, hval, fun r o hcond _ => hframe r o hcond⟩

end VeriTile.Bench.TritonBenchG.DequantizeMatmul
