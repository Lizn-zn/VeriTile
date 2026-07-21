import VeriTile.Triton

/-!
# `log_softmax` — strict per-kernel correctness

`log_softmax_kernel` is a row-wise log-softmax over a 3-D `[M, N, K]` tensor:
program `(pid_m, pid_k)` loads the `[BLOCK_M, BLOCK_N]` tile at stride
`m·N·K + n·K + pid_k`, computes `log(exp(inp − rowmax) / ∑ exp(inp − rowmax))`
along the `N` (axis 1) dimension, and stores the result, masked by
`m_offset < M ∧ n_offset < N`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body, for both the forward `log_softmax_kernel` and the
`log_softmax_backward_kernel`. The host launch (the autotuned/heuristic grid
`(cdiv(M, BLOCK_M), K)`, the `BLOCK_N` heuristic, and how the runtime composes
per-program writes into one buffer) is the *trusted boundary*. Because the
program ids are universally quantified, the per-program statement covers every
program of the grid.

The **forward** headline is stated on the kernel's masked **IO signature**
`logSoftmaxIO` (`Masked2DKernelIO₁`): which buffer is which argument, the
`[BLOCK_M, BLOCK_N]` tile flattened row-major into `B = BLOCK_M * BLOCK_N`
lanes (`laneIdx`: lane `j` = tile cell `(j / BLOCK_N, j % BLOCK_N)`), the
shared read/write window `(pid₀·BLOCK_M + m)·N·K + n·K + pid₁`, and the active
lane predicate `pid₀·BLOCK_M + m < M ∧ n < N`. `⊨`
(`Masked2DKernelIO₁.Implements`) is the audit-once masked Hoare-triple
combinator: for **every** disjoint flat placement of the two buffers, **every**
program-id pair whose active lanes are in bounds, and **every** launch state
whose input window holds the tile `xs` at the active lanes, the translated
pointer kernel terminates, every active output lane holds the row-wise
log-softmax of `xs` (`logSoftmaxOf` applied to the tile rebuilt from `xs`, with
the masked-off lanes carrying the kernel's own `other = -inf` padding), and
every other memory cell is unchanged.

## Proof architecture

```
log_softmax_kernel_correctness                       ← TOP SPECIFICATION (logSoftmaxIO ⊨ row-wise log-softmax)
  ├─ log_softmax_kernel_flattenOk                    bridge fragment membership
  ├─ log_softmax_kernel_traceSafe                    per-execution lane-wise safety walk
  └─ log_softmax_kernel_region_run                   region-model masked Hoare triple
       ├─ log_softmax_kernel_exec_isSome             termination (needs `0 < BLOCK_N`)
       ├─ log_softmax_kernel_correct                 ← algorithm-layer readback per cell
       ├─ log_softmax_inputTile_eq                   memory tile = tile rebuilt from `xs`
       └─ log_softmax_kernel_frame                   masked scatter-store cell frame
log_softmax_backward_kernel_correctness              ← BACKWARD SPECIFICATION (logSoftmaxBackwardIO ⊨ log-softmax VJP)
  ├─ log_softmax_backward_kernel_flattenOk           bridge fragment membership
  ├─ log_softmax_backward_kernel_traceSafe           per-execution lane-wise safety walk
  └─ log_softmax_backward_kernel_region_run          region-model masked Hoare triple (needs the `undef` pin)
       ├─ log_softmax_backward_kernel_exec_isSome      termination (needs NO `0 < BLOCK_N`)
       ├─ log_softmax_backward_kernel_correct          ← algorithm-layer readback per cell
       ├─ log_softmax_backward_{out,grad}Tile_eq       memory tile = tile rebuilt from `xs`/`ys` (masked lanes = pinned `undef = 0`)
       └─ log_softmax_backward_kernel_frame            masked scatter-store cell frame
```

The reduction core `logSoftmaxOf` is built inline from the tiled primitives
`Tile.reduceMax`/`reduceSum`/`bop`/`uop` rather than from a
`VeriTile.Triton.Math.Softmax` oracle — the softmax math is inline-duplicated in
this file. The backward spec (`logSoftmaxBackwardSpec`) is likewise inline.

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); `@triton.autotune` and
`@triton.heuristics` are not modeled — the statements are general in
`BLOCK_M`/`BLOCK_N`, so they cover every config at once. The `.to(tl.float32)`
casts reduce to the identity at the algorithm layer (post-erasure all dtypes
unify to `ℝ`). Out-of-tile lanes are loaded with `other = -float("inf")`
(forward); the masked store leaves inactive cells (`¬(m < M ∧ n < N)`)
untouched, so the *stored* result never depends on padded-block values.

Two honest side hypotheses on the forward headline, both **required for
truth**:

* `hOutInj` — distinct lanes of a program's tile must not alias the same output
  cell; otherwise only the last store survives and the per-lane claim is false.
  The host's contiguous `[M, N, K]` layout satisfies it.
* `0 < BLOCK_N` — `tl.max(inp, axis=1)` over a **zero-length** axis has no
  value, and the kernel's execution genuinely gets stuck there
  (`exec … = none`), so termination fails. Every real `BLOCK_N` comes from
  `heur_block_n = next_power_of_2(N) ≥ 1`.

### Backward kernel: the `undef`-pinned `⊨` route

`log_softmax_backward_kernel` is on the `⊨` surface via
`Masked2DKernelIO₂` (`logSoftmaxBackwardIO ⊨ log-softmax VJP`, headline
`log_softmax_backward_kernel_correctness`). Its two loads are masked *without*
an `other=` default, so the masked-off lanes of a block enter the register
tiles as `s.undef …`; the row reduction `scale = tl.sum(out_grad, 1)` then
consumes them, so the value stored at an **active** lane depends on `s.undef` at
the block's *inactive* lanes. `Masked2DKernelIO₂.Implements` pins
`s₀.undef = fun _ _ => 0`, and the assembly lemma
`Masked2DKernelIO₂.Implements.intro_undef` — the `undef`-threading sibling of
`intro` — hands that pin to its `hrun` obligation. The honest per-lane spec
therefore reduces over the tiles rebuilt from the interface inputs `xs`/`ys`
with the **masked-off lanes fixed to `0`** (`logSoftmaxBackwardPure{Out,Grad}Tile`),
matching what the kernel actually computes. No `other=` default is added to the
model, and no `0 < BLOCK_N` is needed: the backward kernel's only reduction is
`tl.sum`, which is a total `Finset.sum` (`= 0` on an empty axis), so termination
never fails.
-/

namespace VeriTile.Bench.TritonBenchG.LogSoftmax

open VeriTile.Triton
open scoped VeriTile.Triton.Masked2DKernelIO₁

set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

/-- Faithful transcription of `log_softmax.py`'s `log_softmax_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_M: tl.constexpr` / `BLOCK_N: tl.constexpr` -> Lean `Nat`
  parameters. -/
def log_softmax_kernel
    (output_ptr input_ptr : RegionName)
    (M N K BLOCK_M BLOCK_N : Nat) :
    ComputeKernel := triton {
  pid_m = tl.program_id(0)
  pid_k = tl.program_id(1)
  m_offset = pid_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  n_offset = tl.arange(0, $(BLOCK_N))
  offset = m_offset[:, None] * $(N) * $(K) + n_offset[None, :] * $(K) + pid_k
  mask = m_offset[:, None] < $(M) and n_offset[None, :] < $(N)
  input_ptrs = input_ptr + offset
  inp = (tl.load(input_ptrs, mask=mask, other=-float("inf"))).to(tl.float32)
  row_minus_max = inp - tl.max(inp, axis=1)[:, None]
  numerator = tl.exp(row_minus_max)
  denominator = tl.sum(numerator, axis=1)[:, None]
  softmax_output = tl.log(numerator / denominator)
  output_ptrs = output_ptr + offset
  tl.store(output_ptrs, softmax_output, mask=mask)
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

/-! ### Windows, gates and the reduction core -/

/-- The shared load/store address of tile cell `idx` for program `(p₀, p₁)`. -/
def outOffset (p₀ p₁ N K BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_N]) : Nat :=
  (p₀ * BLOCK_M + idx.1.val) * N * K + idx.2.1.val * K + p₁

/-- The kernel's mask `m_offset < M ∧ n_offset < N`. -/
def active (p₀ M N BLOCK_M : Nat) (idx : TileIndex [BLOCK_M, BLOCK_N]) : Prop :=
  p₀ * BLOCK_M + idx.1.val < M ∧ idx.2.1.val < N

instance activeDecidable (p₀ M N BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_N]) :
    Decidable (active p₀ M N BLOCK_M idx) := by
  unfold active
  infer_instance

/-- The kernel's reduction core as a function of the *loaded tile*:
`log(exp(inp − rowmax) / ∑ exp(inp − rowmax))` along axis 1. This is the pure
content of the `⊨` headline; the two tile-builders below feed it either the
memory contents (readback lemma) or the `⊨` interface's `xs` (headline). -/
noncomputable def logSoftmaxOf (BLOCK_M BLOCK_N : Nat)
    (inp : Tile .real [BLOCK_M, BLOCK_N])
    (idx : TileIndex [BLOCK_M, BLOCK_N]) : ℝ :=
  match Tile.reduceMax (shape := [BLOCK_M, BLOCK_N]) ⟨1, by simp⟩ Bool.true inp with
  | some rowMax =>
      let rowBroadcast : Broadcast [BLOCK_M, BLOCK_N] [BLOCK_M, 1] [BLOCK_M, BLOCK_N] :=
        Broadcast.consSame (Broadcast.consR Broadcast.nil)
      let shifted := Tile.bop (NumericDType.sub .real) rowBroadcast inp rowMax
      let numerator := Tile.uop WithBot.realExp shifted
      let denominator := Tile.reduceSum (shape := [BLOCK_M, BLOCK_N]) ⟨1, by simp⟩ Bool.true numerator
      WithBot.unbotD 0
        ((Tile.uop WithBot.realLog
          (Tile.bop (NumericDType.div .real) rowBroadcast numerator denominator)).data idx)
  | none => 0

/-- The tile the kernel actually loads: memory at the active lanes, the
`other = -float("inf")` padding (`none` = `⊥`) elsewhere. -/
noncomputable def logSoftmaxInputTile
    (s : BlockState) (input_ptr : RegionName)
    (p₀ p₁ M N K BLOCK_M BLOCK_N : Nat) :
    Tile .real [BLOCK_M, BLOCK_N] :=
  { data := fun idx =>
      if active p₀ M N BLOCK_M idx then
        some (s.readMem input_ptr (outOffset p₀ p₁ N K BLOCK_M idx))
      else none }

/-- The same tile rebuilt from the `⊨` interface's per-lane input `xs`. -/
noncomputable def logSoftmaxPureTile
    (p₀ M N BLOCK_M BLOCK_N : Nat) (xs : Fin (BLOCK_M * BLOCK_N) → ℝ) :
    Tile .real [BLOCK_M, BLOCK_N] :=
  { data := fun idx =>
      if active p₀ M N BLOCK_M idx then
        some (xs (laneOf BLOCK_M BLOCK_N idx))
      else none }

/-- State-coupled readback spec, internal to `log_softmax_kernel_correct`. -/
noncomputable def logSoftmaxSpec
    (s : BlockState) (input_ptr : RegionName)
    (p₀ p₁ M N K BLOCK_M BLOCK_N : Nat) (idx : TileIndex [BLOCK_M, BLOCK_N]) : ℝ :=
  logSoftmaxOf BLOCK_M BLOCK_N
    (logSoftmaxInputTile s input_ptr p₀ p₁ M N K BLOCK_M BLOCK_N) idx

/-- Bridge: under the `⊨` input contract the loaded tile *is* the tile rebuilt
from `xs` (they agree at the active lanes by hypothesis, and both carry the
`-inf` padding elsewhere). -/
theorem log_softmax_inputTile_eq
    (s : BlockState) (input_ptr : RegionName)
    (p₀ p₁ M N K BLOCK_M BLOCK_N : Nat) (xs : Fin (BLOCK_M * BLOCK_N) → ℝ)
    (hx : ∀ j : Fin (BLOCK_M * BLOCK_N),
      active p₀ M N BLOCK_M (laneIdx BLOCK_M BLOCK_N j) →
      s.readMem input_ptr (outOffset p₀ p₁ N K BLOCK_M
        (laneIdx BLOCK_M BLOCK_N j)) = xs j) :
    logSoftmaxInputTile s input_ptr p₀ p₁ M N K BLOCK_M BLOCK_N
      = logSoftmaxPureTile p₀ M N BLOCK_M BLOCK_N xs := by
  simp only [logSoftmaxInputTile, logSoftmaxPureTile]
  congr 1
  funext idx
  by_cases h : active p₀ M N BLOCK_M idx
  · have hj := hx (laneOf BLOCK_M BLOCK_N idx) (by rw [laneIdx_laneOf]; exact h)
    rw [laneIdx_laneOf] at hj
    rw [if_pos h, if_pos h, hj]
  · rw [if_neg h, if_neg h]

/-- Algorithm-layer cellwise correctness for `log_softmax_kernel`. -/
theorem log_softmax_kernel_correct
    (output_ptr input_ptr : RegionName)
    (M N K BLOCK_M BLOCK_N : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
        outOffset (s.pids 0) (s.pids 1) N K BLOCK_M idx))
    (hExec : exec (log_softmax_kernel output_ptr input_ptr M N K BLOCK_M BLOCK_N) s = some s') :
    ∀ idx : TileIndex [BLOCK_M, BLOCK_N],
      s'.readMem output_ptr (outOffset (s.pids 0) (s.pids 1) N K BLOCK_M idx) =
        if active (s.pids 0) M N BLOCK_M idx then
          logSoftmaxSpec s input_ptr (s.pids 0) (s.pids 1) M N K BLOCK_M BLOCK_N idx
        else s.readMem output_ptr (outOffset (s.pids 0) (s.pids 1) N K BLOCK_M idx) := by
  intro idx
  by_cases hBM : 0 < BLOCK_M
  · by_cases hBN : 0 < BLOCK_N
    · simp [exec, log_softmax_kernel, stepStmts, stepStmt, evalOp, evalOp.eq_def,
            Option.bind, Option.map,
            Tile.bop, Tile.cop, Tile.ptrAdd, Tile.expandDim, Tile.uop,
            Tile.reduceMax, Tile.reduceMaxDrop, Tile.reduceMaxKeep,
            Tile.reduceSum, Tile.reduceSumDrop, Tile.reduceSumKeep,
            TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
            TileShape.dropInsertedIndex, NumericDType.add, NumericDType.mul,
            NumericDType.sub, NumericDType.div, ComparableDType.lt,
            FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot,
            ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, hBN] at hExec
      rw [← hExec]
      have hOffsetInj : Function.Injective
          (fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
            (s.pids 0 * BLOCK_M + idx.1.val) * N * K + idx.2.1.val * K + s.pids 1) := by
        intro a b h
        apply hOutInj
        simpa [outOffset] using h
      simp only [outOffset]
      rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj idx]
      by_cases hActive : active (s.pids 0) M N BLOCK_M idx
      · rcases hActive with ⟨hM, hN⟩
        simp [active, hM, hN, logSoftmaxSpec, logSoftmaxOf,
              logSoftmaxInputTile, outOffset, Tile.reduceMax, Tile.reduceMaxDrop,
              Tile.reduceMaxKeep, Tile.reduceSum, Tile.reduceSumDrop,
              Tile.reduceSumKeep, TileShape.axisDim,
              TileShape.eraseAxis, TileShape.insertAxisIndex,
              TileShape.replaceAxisIndex,
              TileShape.dropInsertedIndex, NumericDType.sub, hBN]
        congr
      · have hInactive :
            ¬ (s.pids 0 * BLOCK_M + idx.1.val < M ∧ idx.2.1.val < N) := by
          simpa [active] using hActive
        simp [hInactive]
        intro h
        exact False.elim (hActive h)
    · simp [exec, log_softmax_kernel, stepStmts, stepStmt, evalOp, evalOp.eq_def,
            Option.bind, Option.map,
            Tile.bop, Tile.cop, Tile.ptrAdd, Tile.expandDim, Tile.uop,
            Tile.reduceMax, Tile.reduceMaxDrop, Tile.reduceMaxKeep,
            Tile.reduceSum, Tile.reduceSumDrop, Tile.reduceSumKeep,
            TileShape.axisDim, TileShape.eraseAxis, TileShape.dropInsertedIndex,
            NumericDType.add, NumericDType.mul, NumericDType.sub,
            NumericDType.div, ComparableDType.lt,
            ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, hBN] at hExec
  · exact False.elim (hBM (Nat.lt_of_le_of_lt (Nat.zero_le _) idx.1.isLt))

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

/-- Termination: with a non-degenerate reduction axis the kernel executes to
completion from any state. `0 < BLOCK_N` is genuinely needed — `tl.max` over a
zero-length axis has no value and `exec` returns `none`. -/
private theorem log_softmax_kernel_exec_isSome
    (output_ptr input_ptr : RegionName)
    (M N K BLOCK_M BLOCK_N : Nat) (hBN : 0 < BLOCK_N)
    (s : BlockState) :
    ∃ s1, exec ((log_softmax_kernel output_ptr input_ptr M N K BLOCK_M
        BLOCK_N).toAlgKernel) s = some s1 := by
  simp [exec, log_softmax_kernel, ComputeKernel.toAlgKernel, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map,
        Tile.bop, Tile.cop, Tile.ptrAdd, Tile.expandDim, Tile.uop,
        Tile.reduceMax, Tile.reduceMaxDrop, Tile.reduceMaxKeep,
        Tile.reduceSum, Tile.reduceSumDrop, Tile.reduceSumKeep,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        TileShape.dropInsertedIndex, NumericDType.add, NumericDType.mul,
        NumericDType.sub, NumericDType.div, ComparableDType.lt,
        FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, hBN]

/-- Frame half: every memory cell not actively written by the masked output
store — every cell of every region other than `output_ptr`, and the
masked-off lanes of the output window itself — is preserved by the run. -/
private theorem log_softmax_kernel_frame
    (output_ptr input_ptr : RegionName)
    (M N K BLOCK_M BLOCK_N : Nat) (hBN : 0 < BLOCK_N)
    (s s1 : BlockState)
    (hExec : exec ((log_softmax_kernel output_ptr input_ptr M N K BLOCK_M
        BLOCK_N).toAlgKernel) s = some s1)
    (r : RegionName) (o : Nat)
    (hmiss : ∀ idx : TileIndex [BLOCK_M, BLOCK_N],
      active (s.pids 0) M N BLOCK_M idx →
      ¬(output_ptr = r ∧ outOffset (s.pids 0) (s.pids 1) N K BLOCK_M idx = o)) :
    s1.mem r o = s.mem r o := by
  simp [exec, log_softmax_kernel, ComputeKernel.toAlgKernel, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map,
        Tile.bop, Tile.cop, Tile.ptrAdd, Tile.expandDim, Tile.uop,
        Tile.reduceMax, Tile.reduceMaxDrop, Tile.reduceMaxKeep,
        Tile.reduceSum, Tile.reduceSumDrop, Tile.reduceSumKeep,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        TileShape.dropInsertedIndex, NumericDType.add, NumericDType.mul,
        NumericDType.sub, NumericDType.div, ComparableDType.lt,
        FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, hBN] at hExec
  subst hExec
  refine Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_) rfl
  intro k _ hmk hc
  refine hmiss k ?_ (by simpa [outOffset] using hc)
  simpa [active] using hmk

/-- The kernel sits inside the flat-memory bridge's covered fragment (pointer
arithmetic, a masked load, two axis-1 reductions and a masked store). -/
theorem log_softmax_kernel_flattenOk
    (output_ptr input_ptr : RegionName) (M N K BLOCK_M BLOCK_N : Nat) :
    ((log_softmax_kernel output_ptr input_ptr M N K BLOCK_M
        BLOCK_N).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [log_softmax_kernel, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, StmtList.FlattenOk, Stmt.FlattenOk,
    Op.FlattenOk.eq_def]

/-- Per-execution safety walk: the two `program_id`s, the two `arange`s, the
offset/mask tiles and the whole reduction chain are memory-silent; the masked
load and the masked store reduce to the lane-wise bounds hypotheses. -/
theorem log_softmax_kernel_traceSafe
    (output_ptr input_ptr : RegionName)
    (M N K BLOCK_M BLOCK_N : Nat) (hBN : 0 < BLOCK_N)
    (bounds : RegionBounds) (s : BlockState)
    (h1 : ∀ j : Fin (BLOCK_M * BLOCK_N),
      active (s.pids 0) M N BLOCK_M (laneIdx BLOCK_M BLOCK_N j) →
      outOffset (s.pids 0) (s.pids 1) N K BLOCK_M (laneIdx BLOCK_M BLOCK_N j)
        < bounds input_ptr)
    (h2 : ∀ j : Fin (BLOCK_M * BLOCK_N),
      active (s.pids 0) M N BLOCK_M (laneIdx BLOCK_M BLOCK_N j) →
      outOffset (s.pids 0) (s.pids 1) N K BLOCK_M (laneIdx BLOCK_M BLOCK_N j)
        < bounds output_ptr) :
    Kernel.TraceSafe bounds
      ((log_softmax_kernel output_ptr input_ptr M N K BLOCK_M
        BLOCK_N).toAlgKernel) s := by
  have h1' : ∀ (a : Fin BLOCK_M) (b : Fin BLOCK_N),
      s.pids 0 * BLOCK_M + a.val < M → b.val < N →
      (s.pids 0 * BLOCK_M + a.val) * N * K + b.val * K + s.pids 1
        < bounds input_ptr := by
    intro a b ha hb
    have := h1 (laneOf BLOCK_M BLOCK_N (a, b, PUnit.unit))
      (by simpa [active] using And.intro ha hb)
    simpa [outOffset] using this
  have h2' : ∀ (a : Fin BLOCK_M) (b : Fin BLOCK_N),
      s.pids 0 * BLOCK_M + a.val < M → b.val < N →
      (s.pids 0 * BLOCK_M + a.val) * N * K + b.val * K + s.pids 1
        < bounds output_ptr := by
    intro a b ha hb
    have := h2 (laneOf BLOCK_M BLOCK_N (a, b, PUnit.unit))
      (by simpa [active] using And.intro ha hb)
    simpa [outOffset] using this
  unfold Kernel.TraceSafe
  simp [log_softmax_kernel, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
    Op.PointerAddressesSafeOn, Op.MemorySafe, MaskOpt.SafeAt, MemAccess.SafeAt, stepStmt, evalOp, evalOp.eq_def,
    Option.bind, Option.map, MemAccess.ActiveAddressSafe,
    memAccessActiveAddressSafe, MaskOpt.Active, BlockState.setReg,
    Tile.bop, Tile.cop, Tile.ptrAdd, Tile.expandDim, Tile.uop,
    Tile.reduceMax, Tile.reduceMaxDrop, Tile.reduceMaxKeep,
    Tile.reduceSum, Tile.reduceSumDrop, Tile.reduceSumKeep,
    TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
    TileShape.dropInsertedIndex, NumericDType.add, NumericDType.mul,
    NumericDType.sub, NumericDType.div, ComparableDType.lt,
    FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot, hBN]
  refine ⟨?_, ?_⟩ <;> intros <;>
    first
      | (apply h1' <;> assumption)
      | (apply h2' <;> assumption)

/-- **The region-model masked Hoare triple** — termination, active-lane output
values, and frame off the active output lanes, from any launch state whose
input window holds `xs` at the active lanes. This is the `hrun` obligation of
the `⊨` headline; the value half reuses `log_softmax_kernel_correct`, whose
state-coupled tile collapses to the `xs`-rebuilt tile by
`log_softmax_inputTile_eq`. -/
theorem log_softmax_kernel_region_run
    (output_ptr input_ptr : RegionName)
    (M N K BLOCK_M BLOCK_N : Nat) (hBN : 0 < BLOCK_N)
    (s₀ : BlockState) (xs : Fin (BLOCK_M * BLOCK_N) → ℝ)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
        outOffset (s₀.pids 0) (s₀.pids 1) N K BLOCK_M idx))
    (hx : ∀ j : Fin (BLOCK_M * BLOCK_N),
      active (s₀.pids 0) M N BLOCK_M (laneIdx BLOCK_M BLOCK_N j) →
      s₀.readMem input_ptr (outOffset (s₀.pids 0) (s₀.pids 1) N K BLOCK_M
        (laneIdx BLOCK_M BLOCK_N j)) = xs j) :
    ∃ s1, exec ((log_softmax_kernel output_ptr input_ptr M N K BLOCK_M
        BLOCK_N).toAlgKernel) s₀ = some s1
      ∧ (∀ j : Fin (BLOCK_M * BLOCK_N),
          active (s₀.pids 0) M N BLOCK_M (laneIdx BLOCK_M BLOCK_N j) →
          s1.readMem output_ptr (outOffset (s₀.pids 0) (s₀.pids 1) N K BLOCK_M
            (laneIdx BLOCK_M BLOCK_N j))
            = logSoftmaxOf BLOCK_M BLOCK_N
                (logSoftmaxPureTile (s₀.pids 0) M N BLOCK_M BLOCK_N xs)
                (laneIdx BLOCK_M BLOCK_N j))
      ∧ (∀ r o,
          (r ≠ output_ptr ∨ ∀ j : Fin (BLOCK_M * BLOCK_N),
            active (s₀.pids 0) M N BLOCK_M (laneIdx BLOCK_M BLOCK_N j) →
            o ≠ outOffset (s₀.pids 0) (s₀.pids 1) N K BLOCK_M
              (laneIdx BLOCK_M BLOCK_N j)) →
          s1.mem r o = s₀.mem r o) := by
  obtain ⟨s1, hs1⟩ := log_softmax_kernel_exec_isSome output_ptr input_ptr M N K
    BLOCK_M BLOCK_N hBN s₀
  have hs1' : exec (log_softmax_kernel output_ptr input_ptr M N K BLOCK_M
      BLOCK_N) s₀ = some s1 := hs1
  refine ⟨s1, hs1, fun j hj => ?_, fun r o hcond => ?_⟩
  · have h := log_softmax_kernel_correct output_ptr input_ptr M N K BLOCK_M
      BLOCK_N s₀ s1 hOutInj hs1' (laneIdx BLOCK_M BLOCK_N j)
    rw [if_pos hj] at h
    rw [h, logSoftmaxSpec,
      log_softmax_inputTile_eq s₀ input_ptr (s₀.pids 0) (s₀.pids 1) M N K
        BLOCK_M BLOCK_N xs hx]
  · refine log_softmax_kernel_frame output_ptr input_ptr M N K BLOCK_M BLOCK_N
      hBN s₀ s1 hs1 r o (fun idx hact ⟨hr, ho⟩ => ?_)
    rcases hcond with hne | hno
    · exact hne hr.symm
    · exact hno (laneOf BLOCK_M BLOCK_N idx)
        (by rw [laneIdx_laneOf]; exact hact)
        (by rw [laneIdx_laneOf]; exact ho.symm)

/-! ### The companion backward kernel (`⊨` surface via the `undef`-pinned
`Masked2DKernelIO₂.Implements.intro_undef` — see the module docstring) -/

/-- Faithful transcription of `log_softmax.py`'s
`log_softmax_backward_kernel`. -/
def log_softmax_backward_kernel
    (out_ptr out_grad_ptr in_grad_ptr : RegionName)
    (M N K BLOCK_M BLOCK_N : Nat) :
    ComputeKernel := triton {
  pid_m = tl.program_id(0)
  pid_k = tl.program_id(1)
  m_offset = pid_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  n_offset = tl.arange(0, $(BLOCK_N))
  offsets = m_offset[:, None] * $(N) * $(K) + n_offset[None, :] * $(K) + pid_k
  mask = m_offset[:, None] < $(M) and n_offset[None, :] < $(N)
  out_ptrs = out_ptr + offsets
  out = tl.load(out_ptrs, mask=mask).to(tl.float32)
  out_grad_ptrs = out_grad_ptr + offsets
  out_grad = tl.load(out_grad_ptrs, mask=mask).to(tl.float32)
  scale = tl.sum(out_grad, 1)
  in_grad = out_grad - tl.exp((out).to(tl.float32)) * scale[:, None]
  in_grad_ptrs = in_grad_ptr + offsets
  tl.store(in_grad_ptrs, in_grad, mask=mask)
}

noncomputable def logSoftmaxBackwardOutTile
    (s : BlockState) (out_ptr : RegionName)
    (M N K BLOCK_M BLOCK_N : Nat) :
    Tile .real [BLOCK_M, BLOCK_N] :=
  { data := fun idx =>
      if active (s.pids 0) M N BLOCK_M idx then
        some (s.readMem out_ptr (outOffset (s.pids 0) (s.pids 1) N K BLOCK_M idx))
      else some (s.undef out_ptr (outOffset (s.pids 0) (s.pids 1) N K BLOCK_M idx)) }

noncomputable def logSoftmaxBackwardGradTile
    (s : BlockState) (out_grad_ptr : RegionName)
    (M N K BLOCK_M BLOCK_N : Nat) :
    Tile .real [BLOCK_M, BLOCK_N] :=
  { data := fun idx =>
      if active (s.pids 0) M N BLOCK_M idx then
        some (s.readMem out_grad_ptr (outOffset (s.pids 0) (s.pids 1) N K BLOCK_M idx))
      else some (s.undef out_grad_ptr (outOffset (s.pids 0) (s.pids 1) N K BLOCK_M idx)) }

/-- The backward kernel's pointwise VJP core as a function of the two *loaded
tiles* (`out`, `outGrad`): `outGrad − exp(out) · (Σ_row outGrad)` along axis 1.
This is the pure content of the `⊨` headline; the tile-builders below feed it
either the memory contents (readback lemma) or the `⊨` interface's `xs`/`ys`
(headline). -/
noncomputable def logSoftmaxBackwardOf (BLOCK_M BLOCK_N : Nat)
    (out outGrad : Tile .real [BLOCK_M, BLOCK_N])
    (idx : TileIndex [BLOCK_M, BLOCK_N]) : ℝ :=
  let scale := Tile.reduceSum (shape := [BLOCK_M, BLOCK_N]) ⟨1, by simp⟩ Bool.true outGrad
  let rowBroadcast : Broadcast [BLOCK_M, BLOCK_N] [BLOCK_M, 1] [BLOCK_M, BLOCK_N] :=
    Broadcast.consSame (Broadcast.consR Broadcast.nil)
  let sameBroadcast : Broadcast [BLOCK_M, BLOCK_N] [BLOCK_M, BLOCK_N] [BLOCK_M, BLOCK_N] :=
    Broadcast.consSame (Broadcast.consSame Broadcast.nil)
  WithBot.unbotD 0
    ((Tile.bop (NumericDType.sub .real) sameBroadcast
      outGrad
      (Tile.bop (NumericDType.mul .real) rowBroadcast
        (Tile.uop WithBot.realExp out) scale)).data idx)

/-- The `out`/`outGrad` tiles rebuilt from the `⊨` interface's per-lane inputs
`xs`/`ys`, with the masked-off lanes carrying the pinned `undef = 0` (the
backward loads have no `other=` default, so masked lanes enter as `s.undef`,
which `Masked2DKernelIO₂.Implements` pins to `0`). -/
noncomputable def logSoftmaxBackwardPureOutTile
    (p₀ M N BLOCK_M BLOCK_N : Nat) (xs : Fin (BLOCK_M * BLOCK_N) → ℝ) :
    Tile .real [BLOCK_M, BLOCK_N] :=
  { data := fun idx =>
      if active p₀ M N BLOCK_M idx then
        some (xs (laneOf BLOCK_M BLOCK_N idx))
      else some 0 }

/-- Companion of `logSoftmaxBackwardPureOutTile` for the second input. -/
noncomputable def logSoftmaxBackwardPureGradTile
    (p₀ M N BLOCK_M BLOCK_N : Nat) (ys : Fin (BLOCK_M * BLOCK_N) → ℝ) :
    Tile .real [BLOCK_M, BLOCK_N] :=
  { data := fun idx =>
      if active p₀ M N BLOCK_M idx then
        some (ys (laneOf BLOCK_M BLOCK_N idx))
      else some 0 }

/-- State-coupled readback spec, internal to `log_softmax_backward_kernel_correct`. -/
noncomputable def logSoftmaxBackwardSpec
    (s : BlockState) (out_ptr out_grad_ptr : RegionName)
    (M N K BLOCK_M BLOCK_N : Nat) (idx : TileIndex [BLOCK_M, BLOCK_N]) : ℝ :=
  logSoftmaxBackwardOf BLOCK_M BLOCK_N
    (logSoftmaxBackwardOutTile s out_ptr M N K BLOCK_M BLOCK_N)
    (logSoftmaxBackwardGradTile s out_grad_ptr M N K BLOCK_M BLOCK_N) idx

/-- Algorithm-layer cellwise correctness for `log_softmax_backward_kernel`. -/
theorem log_softmax_backward_kernel_correct
    (out_ptr out_grad_ptr in_grad_ptr : RegionName)
    (M N K BLOCK_M BLOCK_N : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
        outOffset (s.pids 0) (s.pids 1) N K BLOCK_M idx))
    (hExec : exec (log_softmax_backward_kernel out_ptr out_grad_ptr in_grad_ptr
        M N K BLOCK_M BLOCK_N) s = some s') :
    ∀ idx : TileIndex [BLOCK_M, BLOCK_N],
      s'.readMem in_grad_ptr (outOffset (s.pids 0) (s.pids 1) N K BLOCK_M idx) =
        if active (s.pids 0) M N BLOCK_M idx then
          logSoftmaxBackwardSpec s out_ptr out_grad_ptr M N K BLOCK_M BLOCK_N idx
        else s.readMem in_grad_ptr (outOffset (s.pids 0) (s.pids 1) N K BLOCK_M idx) := by
  intro idx
  by_cases hBM : 0 < BLOCK_M
  · by_cases hBN : 0 < BLOCK_N
    · simp [exec, log_softmax_backward_kernel, stepStmts, stepStmt, evalOp, evalOp.eq_def,
            Option.bind, Option.map,
            Tile.bop, Tile.cop, Tile.ptrAdd, Tile.expandDim, Tile.uop,
            Tile.reduceSum, Tile.reduceSumDrop, Tile.reduceSumKeep,
            TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
            TileShape.dropInsertedIndex, NumericDType.add, NumericDType.mul,
            NumericDType.sub, ComparableDType.lt, FloatDType.cast,
            FloatDType.ofWithBot, FloatDType.toWithBot,
            ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, hBN] at hExec
      rw [← hExec]
      have hOffsetInj : Function.Injective
          (fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
            (s.pids 0 * BLOCK_M + idx.1.val) * N * K + idx.2.1.val * K + s.pids 1) := by
        intro a b h
        apply hOutInj
        simpa [outOffset] using h
      simp only [outOffset]
      rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj idx]
      by_cases hActive : active (s.pids 0) M N BLOCK_M idx
      · rcases hActive with ⟨hM, hN⟩
        simp [active, hM, hN, logSoftmaxBackwardSpec, logSoftmaxBackwardOf,
              logSoftmaxBackwardOutTile, logSoftmaxBackwardGradTile, outOffset,
              Tile.reduceSum, Tile.reduceSumDrop, Tile.reduceSumKeep,
              TileShape.axisDim, TileShape.eraseAxis,
              TileShape.insertAxisIndex, TileShape.replaceAxisIndex,
              TileShape.dropInsertedIndex,
              FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot,
              NumericDType.sub, NumericDType.mul, hBN]
        congr
      · have hInactive :
            ¬ (s.pids 0 * BLOCK_M + idx.1.val < M ∧ idx.2.1.val < N) := by
          simpa [active] using hActive
        simp [hInactive]
        intro h
        exact False.elim (hActive h)
    · exact False.elim (hBN (Nat.lt_of_le_of_lt (Nat.zero_le _) idx.2.1.isLt))
  · exact False.elim (hBM (Nat.lt_of_le_of_lt (Nat.zero_le _) idx.1.isLt))

/-- Bridge: under the `undef` pin and the two `⊨` input contracts the loaded
`out` tile *is* the tile rebuilt from `xs` (they agree at the active lanes by
hypothesis; both carry the pinned `undef = 0` elsewhere). -/
theorem log_softmax_backward_outTile_eq
    (s : BlockState) (out_ptr : RegionName)
    (M N K BLOCK_M BLOCK_N : Nat) (xs : Fin (BLOCK_M * BLOCK_N) → ℝ)
    (hundef : s.undef = (fun _ _ => 0))
    (hx : ∀ j : Fin (BLOCK_M * BLOCK_N),
      active (s.pids 0) M N BLOCK_M (laneIdx BLOCK_M BLOCK_N j) →
      s.readMem out_ptr (outOffset (s.pids 0) (s.pids 1) N K BLOCK_M
        (laneIdx BLOCK_M BLOCK_N j)) = xs j) :
    logSoftmaxBackwardOutTile s out_ptr M N K BLOCK_M BLOCK_N
      = logSoftmaxBackwardPureOutTile (s.pids 0) M N BLOCK_M BLOCK_N xs := by
  simp only [logSoftmaxBackwardOutTile, logSoftmaxBackwardPureOutTile]
  congr 1
  funext idx
  by_cases h : active (s.pids 0) M N BLOCK_M idx
  · have hj := hx (laneOf BLOCK_M BLOCK_N idx) (by rw [laneIdx_laneOf]; exact h)
    rw [laneIdx_laneOf] at hj
    rw [if_pos h, if_pos h, hj]
  · rw [if_neg h, if_neg h, hundef]

/-- Bridge for the second input — companion of `log_softmax_backward_outTile_eq`. -/
theorem log_softmax_backward_gradTile_eq
    (s : BlockState) (out_grad_ptr : RegionName)
    (M N K BLOCK_M BLOCK_N : Nat) (ys : Fin (BLOCK_M * BLOCK_N) → ℝ)
    (hundef : s.undef = (fun _ _ => 0))
    (hy : ∀ j : Fin (BLOCK_M * BLOCK_N),
      active (s.pids 0) M N BLOCK_M (laneIdx BLOCK_M BLOCK_N j) →
      s.readMem out_grad_ptr (outOffset (s.pids 0) (s.pids 1) N K BLOCK_M
        (laneIdx BLOCK_M BLOCK_N j)) = ys j) :
    logSoftmaxBackwardGradTile s out_grad_ptr M N K BLOCK_M BLOCK_N
      = logSoftmaxBackwardPureGradTile (s.pids 0) M N BLOCK_M BLOCK_N ys := by
  simp only [logSoftmaxBackwardGradTile, logSoftmaxBackwardPureGradTile]
  congr 1
  funext idx
  by_cases h : active (s.pids 0) M N BLOCK_M idx
  · have hj := hy (laneOf BLOCK_M BLOCK_N idx) (by rw [laneIdx_laneOf]; exact h)
    rw [laneIdx_laneOf] at hj
    rw [if_pos h, if_pos h, hj]
  · rw [if_neg h, if_neg h, hundef]

/-- The backward kernel sits inside the flat-memory bridge's covered fragment
(pointer arithmetic, two masked loads, one axis-1 reduction and a masked
store). -/
theorem log_softmax_backward_kernel_flattenOk
    (out_ptr out_grad_ptr in_grad_ptr : RegionName)
    (M N K BLOCK_M BLOCK_N : Nat) :
    ((log_softmax_backward_kernel out_ptr out_grad_ptr in_grad_ptr M N K BLOCK_M
        BLOCK_N).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [log_softmax_backward_kernel, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, StmtList.FlattenOk, Stmt.FlattenOk,
    Op.FlattenOk.eq_def]

/-- Termination: the backward kernel executes to completion from any state.
Unlike the forward kernel this needs **no** `0 < BLOCK_N`: its only reduction is
`tl.sum` (`Tile.reduceSum` = a total `Finset.sum`, `= 0` on an empty axis), so
there is no zero-length-axis `none`. -/
private theorem log_softmax_backward_kernel_exec_isSome
    (out_ptr out_grad_ptr in_grad_ptr : RegionName)
    (M N K BLOCK_M BLOCK_N : Nat) (s : BlockState) :
    ∃ s1, exec ((log_softmax_backward_kernel out_ptr out_grad_ptr in_grad_ptr
        M N K BLOCK_M BLOCK_N).toAlgKernel) s = some s1 := by
  simp [exec, log_softmax_backward_kernel, ComputeKernel.toAlgKernel, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map,
        Tile.bop, Tile.cop, Tile.ptrAdd, Tile.expandDim, Tile.uop,
        Tile.reduceSum, Tile.reduceSumDrop, Tile.reduceSumKeep,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        TileShape.dropInsertedIndex, NumericDType.add, NumericDType.mul,
        NumericDType.sub, ComparableDType.lt,
        FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-- Frame half: every memory cell not actively written by the masked
`in_grad` store is preserved by the run. -/
private theorem log_softmax_backward_kernel_frame
    (out_ptr out_grad_ptr in_grad_ptr : RegionName)
    (M N K BLOCK_M BLOCK_N : Nat)
    (s s1 : BlockState)
    (hExec : exec ((log_softmax_backward_kernel out_ptr out_grad_ptr in_grad_ptr
        M N K BLOCK_M BLOCK_N).toAlgKernel) s = some s1)
    (r : RegionName) (o : Nat)
    (hmiss : ∀ idx : TileIndex [BLOCK_M, BLOCK_N],
      active (s.pids 0) M N BLOCK_M idx →
      ¬(in_grad_ptr = r ∧ outOffset (s.pids 0) (s.pids 1) N K BLOCK_M idx = o)) :
    s1.mem r o = s.mem r o := by
  simp [exec, log_softmax_backward_kernel, ComputeKernel.toAlgKernel, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map,
        Tile.bop, Tile.cop, Tile.ptrAdd, Tile.expandDim, Tile.uop,
        Tile.reduceSum, Tile.reduceSumDrop, Tile.reduceSumKeep,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        TileShape.dropInsertedIndex, NumericDType.add, NumericDType.mul,
        NumericDType.sub, ComparableDType.lt,
        FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
  subst hExec
  refine Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_) rfl
  intro k _ hmk hc
  refine hmiss k ?_ (by simpa [outOffset] using hc)
  simpa [active] using hmk

/-- Per-execution safety walk: the two `program_id`s, the two `arange`s, the
offset/mask tiles and the reduction are memory-silent; the two masked loads and
the masked store reduce to the lane-wise bounds hypotheses. -/
theorem log_softmax_backward_kernel_traceSafe
    (out_ptr out_grad_ptr in_grad_ptr : RegionName)
    (M N K BLOCK_M BLOCK_N : Nat)
    (bounds : RegionBounds) (s : BlockState)
    (h1 : ∀ j : Fin (BLOCK_M * BLOCK_N),
      active (s.pids 0) M N BLOCK_M (laneIdx BLOCK_M BLOCK_N j) →
      outOffset (s.pids 0) (s.pids 1) N K BLOCK_M (laneIdx BLOCK_M BLOCK_N j)
        < bounds out_ptr)
    (h2 : ∀ j : Fin (BLOCK_M * BLOCK_N),
      active (s.pids 0) M N BLOCK_M (laneIdx BLOCK_M BLOCK_N j) →
      outOffset (s.pids 0) (s.pids 1) N K BLOCK_M (laneIdx BLOCK_M BLOCK_N j)
        < bounds out_grad_ptr)
    (h3 : ∀ j : Fin (BLOCK_M * BLOCK_N),
      active (s.pids 0) M N BLOCK_M (laneIdx BLOCK_M BLOCK_N j) →
      outOffset (s.pids 0) (s.pids 1) N K BLOCK_M (laneIdx BLOCK_M BLOCK_N j)
        < bounds in_grad_ptr) :
    Kernel.TraceSafe bounds
      ((log_softmax_backward_kernel out_ptr out_grad_ptr in_grad_ptr M N K BLOCK_M
        BLOCK_N).toAlgKernel) s := by
  have mk : ∀ (region : RegionName),
      (∀ j : Fin (BLOCK_M * BLOCK_N),
        active (s.pids 0) M N BLOCK_M (laneIdx BLOCK_M BLOCK_N j) →
        outOffset (s.pids 0) (s.pids 1) N K BLOCK_M (laneIdx BLOCK_M BLOCK_N j)
          < bounds region) →
      ∀ (a : Fin BLOCK_M) (b : Fin BLOCK_N),
        s.pids 0 * BLOCK_M + a.val < M → b.val < N →
        (s.pids 0 * BLOCK_M + a.val) * N * K + b.val * K + s.pids 1
          < bounds region := by
    intro region hh a b ha hb
    have := hh (laneOf BLOCK_M BLOCK_N (a, b, PUnit.unit))
      (by simpa [active] using And.intro ha hb)
    simpa [outOffset] using this
  have h1' := mk out_ptr h1
  have h2' := mk out_grad_ptr h2
  have h3' := mk in_grad_ptr h3
  unfold Kernel.TraceSafe
  simp [log_softmax_backward_kernel, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
    Op.PointerAddressesSafeOn, Op.MemorySafe, MaskOpt.SafeAt, MemAccess.SafeAt, stepStmt, evalOp, evalOp.eq_def,
    Option.bind, Option.map, MemAccess.ActiveAddressSafe,
    memAccessActiveAddressSafe, MaskOpt.Active, BlockState.setReg,
    Tile.bop, Tile.cop, Tile.ptrAdd, Tile.expandDim, Tile.uop,
    Tile.reduceSum, Tile.reduceSumDrop, Tile.reduceSumKeep,
    TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
    TileShape.dropInsertedIndex, NumericDType.add, NumericDType.mul,
    NumericDType.sub, ComparableDType.lt,
    FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot]
  refine ⟨?_, ?_, ?_⟩ <;> intros <;>
    first
      | (apply h1' <;> assumption)
      | (apply h2' <;> assumption)
      | (apply h3' <;> assumption)

/-- **The region-model masked Hoare triple** for the backward kernel —
termination, active-lane output values, and frame off the active output lanes,
from any launch state whose `undef` is pinned to `0` and whose input windows
hold `xs`/`ys` at the active lanes. This is the `hrun` obligation of the `⊨`
headline; the value half reuses `log_softmax_backward_kernel_correct`, whose
state-coupled tiles collapse to the `xs`/`ys`-rebuilt tiles (masked-off lanes
`= 0` under the pin) by `log_softmax_backward_{out,grad}Tile_eq`. -/
theorem log_softmax_backward_kernel_region_run
    (out_ptr out_grad_ptr in_grad_ptr : RegionName)
    (M N K BLOCK_M BLOCK_N : Nat)
    (s₀ : BlockState) (xs ys : Fin (BLOCK_M * BLOCK_N) → ℝ)
    (hundef : s₀.undef = (fun _ _ => 0))
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
        outOffset (s₀.pids 0) (s₀.pids 1) N K BLOCK_M idx))
    (hx : ∀ j : Fin (BLOCK_M * BLOCK_N),
      active (s₀.pids 0) M N BLOCK_M (laneIdx BLOCK_M BLOCK_N j) →
      s₀.readMem out_ptr (outOffset (s₀.pids 0) (s₀.pids 1) N K BLOCK_M
        (laneIdx BLOCK_M BLOCK_N j)) = xs j)
    (hy : ∀ j : Fin (BLOCK_M * BLOCK_N),
      active (s₀.pids 0) M N BLOCK_M (laneIdx BLOCK_M BLOCK_N j) →
      s₀.readMem out_grad_ptr (outOffset (s₀.pids 0) (s₀.pids 1) N K BLOCK_M
        (laneIdx BLOCK_M BLOCK_N j)) = ys j) :
    ∃ s1, exec ((log_softmax_backward_kernel out_ptr out_grad_ptr in_grad_ptr
        M N K BLOCK_M BLOCK_N).toAlgKernel) s₀ = some s1
      ∧ (∀ j : Fin (BLOCK_M * BLOCK_N),
          active (s₀.pids 0) M N BLOCK_M (laneIdx BLOCK_M BLOCK_N j) →
          s1.readMem in_grad_ptr (outOffset (s₀.pids 0) (s₀.pids 1) N K BLOCK_M
            (laneIdx BLOCK_M BLOCK_N j))
            = logSoftmaxBackwardOf BLOCK_M BLOCK_N
                (logSoftmaxBackwardPureOutTile (s₀.pids 0) M N BLOCK_M BLOCK_N xs)
                (logSoftmaxBackwardPureGradTile (s₀.pids 0) M N BLOCK_M BLOCK_N ys)
                (laneIdx BLOCK_M BLOCK_N j))
      ∧ (∀ r o,
          (r ≠ in_grad_ptr ∨ ∀ j : Fin (BLOCK_M * BLOCK_N),
            active (s₀.pids 0) M N BLOCK_M (laneIdx BLOCK_M BLOCK_N j) →
            o ≠ outOffset (s₀.pids 0) (s₀.pids 1) N K BLOCK_M
              (laneIdx BLOCK_M BLOCK_N j)) →
          s1.mem r o = s₀.mem r o) := by
  obtain ⟨s1, hs1⟩ := log_softmax_backward_kernel_exec_isSome out_ptr out_grad_ptr
    in_grad_ptr M N K BLOCK_M BLOCK_N s₀
  have hs1' : exec (log_softmax_backward_kernel out_ptr out_grad_ptr in_grad_ptr
      M N K BLOCK_M BLOCK_N) s₀ = some s1 := hs1
  refine ⟨s1, hs1, fun j hj => ?_, fun r o hcond => ?_⟩
  · have h := log_softmax_backward_kernel_correct out_ptr out_grad_ptr in_grad_ptr
      M N K BLOCK_M BLOCK_N s₀ s1 hOutInj hs1' (laneIdx BLOCK_M BLOCK_N j)
    rw [if_pos hj] at h
    rw [h, logSoftmaxBackwardSpec,
      log_softmax_backward_outTile_eq s₀ out_ptr M N K BLOCK_M BLOCK_N xs hundef hx,
      log_softmax_backward_gradTile_eq s₀ out_grad_ptr M N K BLOCK_M BLOCK_N ys hundef hy]
  · refine log_softmax_backward_kernel_frame out_ptr out_grad_ptr in_grad_ptr M N K
      BLOCK_M BLOCK_N s₀ s1 hs1 r o (fun idx hact ⟨hr, ho⟩ => ?_)
    rcases hcond with hne | hno
    · exact hne hr.symm
    · exact hno (laneOf BLOCK_M BLOCK_N idx)
        (by rw [laneIdx_laneOf]; exact hact)
        (by rw [laneIdx_laneOf]; exact ho.symm)

/-- `log_softmax_backward_kernel`'s masked **IO signature** — the two-input
sibling of `logSoftmaxIO`:

* `in1`/`in2`/`out` — `out_ptr`, `out_grad_ptr`, `in_grad_ptr`;
* `B = BLOCK_M * BLOCK_N` — the row-major flattening (`laneIdx`);
* `read1 = read2 = write` — the shared window
  `(pid₀·BLOCK_M + m)·N·K + n·K + pid₁` (the two loads and the store use the
  same offsets in the three buffers);
* `mask` — the kernel's `m_offset < M ∧ n_offset < N`; `read2Mask`/`writeMask`
  keep the struct default (both loads and the store carry the *same* mask). -/
def logSoftmaxBackwardIO (out_ptr out_grad_ptr in_grad_ptr : RegionName)
    (M N K BLOCK_M BLOCK_N : Nat) : Masked2DKernelIO₂ where
  kernel := log_softmax_backward_kernel out_ptr out_grad_ptr in_grad_ptr M N K BLOCK_M BLOCK_N
  in1 := out_ptr
  in2 := out_grad_ptr
  out := in_grad_ptr
  B := BLOCK_M * BLOCK_N
  read1 := fun p₀ p₁ j => outOffset p₀ p₁ N K BLOCK_M (laneIdx BLOCK_M BLOCK_N j)
  read2 := fun p₀ p₁ j => outOffset p₀ p₁ N K BLOCK_M (laneIdx BLOCK_M BLOCK_N j)
  write := fun p₀ p₁ j => outOffset p₀ p₁ N K BLOCK_M (laneIdx BLOCK_M BLOCK_N j)
  mask := fun p₀ _ j => active p₀ M N BLOCK_M (laneIdx BLOCK_M BLOCK_N j)

/-- **The backward headline**: `log_softmax_backward_kernel` implements the
log-softmax VJP `in_grad = out_grad − exp(out) · (Σ_row out_grad)` on its masked
IO signature — for every disjoint flat placement of the three buffers, every
program-id pair whose active lanes are in bounds, and every launch state whose
input windows hold `xs`/`ys` at the active lanes, the translated pointer kernel
terminates, every active output lane `j` holds `logSoftmaxBackwardOf` applied to
the `out`/`out_grad` tiles rebuilt from `xs`/`ys`, and every other memory cell,
including the masked-off lanes of the output window, is unchanged.

The kernel's two loads are masked **without** an `other=` default, so at the
masked-off lanes the register tiles carry `s₀.undef` — which
`Masked2DKernelIO₂.Implements` pins to `0`. The row reduction
`scale = tl.sum(out_grad, 1)` genuinely consumes those pinned zeros, so the
honest per-lane spec sums the tile with masked-off lanes set to `0`
(`logSoftmaxBackwardPure{Out,Grad}Tile`). Because the pin is a hypothesis of the
`⊨` obligation, this headline is discharged via
`Masked2DKernelIO₂.Implements.intro_undef` (the `undef`-threading sibling of
`intro`), not the plain `intro`.

One honest, **truth-required** side hypothesis: `hOutInj` (distinct lanes of a
program's tile must not alias the same output cell — otherwise only the last
store survives). Unlike the forward headline, **no** `0 < BLOCK_N` is needed:
the backward kernel's only reduction is `tl.sum`, which is total on an empty
axis (`= 0`), so termination never fails. -/
specification log_softmax_backward_kernel_correctness
    (out_ptr out_grad_ptr in_grad_ptr : RegionName)
    (M N K BLOCK_M BLOCK_N : Nat)
    (hOutInj : ∀ p₀ p₁ : Nat, Function.Injective
      (fun j : Fin (BLOCK_M * BLOCK_N) =>
        outOffset p₀ p₁ N K BLOCK_M (laneIdx BLOCK_M BLOCK_N j))) :
    Masked2DKernelIO₂.Implements
      (logSoftmaxBackwardIO out_ptr out_grad_ptr in_grad_ptr M N K BLOCK_M BLOCK_N)
      (fun p₀ _ xs ys j =>
          logSoftmaxBackwardOf BLOCK_M BLOCK_N
            (logSoftmaxBackwardPureOutTile p₀ M N BLOCK_M BLOCK_N xs)
            (logSoftmaxBackwardPureGradTile p₀ M N BLOCK_M BLOCK_N ys)
            (laneIdx BLOCK_M BLOCK_N j)) := by
  refine Masked2DKernelIO₂.Implements.intro_undef _ ?_ ?_ ?_
  · exact log_softmax_backward_kernel_flattenOk out_ptr out_grad_ptr in_grad_ptr
      M N K BLOCK_M BLOCK_N
  · intro bounds s hin1 hin2 hout _
    exact log_softmax_backward_kernel_traceSafe out_ptr out_grad_ptr in_grad_ptr
      M N K BLOCK_M BLOCK_N bounds s hin1 hin2 hout
  · intro s₀ xs ys hundef hx hy
    obtain ⟨s1, hs1, hval, hframe⟩ :=
      log_softmax_backward_kernel_region_run out_ptr out_grad_ptr in_grad_ptr
        M N K BLOCK_M BLOCK_N s₀ xs ys hundef
        (inj_of_lane_inj (hOutInj (s₀.pids 0) (s₀.pids 1))) hx hy
    exact ⟨s1, hs1, hval, fun r o hcond _ => hframe r o hcond⟩

/-- `log_softmax_kernel`'s masked **IO signature** — the whole kernel-specific
audit surface of the `⊨` headline:

* `inp`/`out` — which buffer is which argument (`input_ptr`, `output_ptr`);
* `B = BLOCK_M * BLOCK_N` — the tile flattened row-major into lanes, lane `j` =
  tile cell `(j / BLOCK_N, j % BLOCK_N)` (`laneIdx`);
* `read = write` — the shared window
  `(pid₀·BLOCK_M + m)·N·K + n·K + pid₁` (the kernel reads and writes the same
  offsets in two different buffers);
* `mask` — the kernel's `m_offset < M ∧ n_offset < N`; `writeMask` keeps the
  struct default (the load and the store carry the *same* mask).

Both program-id axes are used (`pid_m` tiles the `M` axis, `pid_k` is the
innermost coordinate). -/
def logSoftmaxIO (output_ptr input_ptr : RegionName)
    (M N K BLOCK_M BLOCK_N : Nat) : Masked2DKernelIO₁ where
  kernel := log_softmax_kernel output_ptr input_ptr M N K BLOCK_M BLOCK_N
  inp := input_ptr
  out := output_ptr
  B := BLOCK_M * BLOCK_N
  read := fun p₀ p₁ j => outOffset p₀ p₁ N K BLOCK_M (laneIdx BLOCK_M BLOCK_N j)
  write := fun p₀ p₁ j => outOffset p₀ p₁ N K BLOCK_M (laneIdx BLOCK_M BLOCK_N j)
  mask := fun p₀ _ j => active p₀ M N BLOCK_M (laneIdx BLOCK_M BLOCK_N j)

/-- **The headline**: `log_softmax_kernel` implements the row-wise log-softmax
on its masked IO signature — for every disjoint flat placement of
`input_ptr`/`output_ptr`, every program-id pair whose active lanes are in
bounds, and every launch state whose input window holds the tile `xs` at the
active lanes, the translated pointer kernel terminates, every active output
lane `j` holds `log(exp(x − rowmax) / ∑ exp(x − rowmax))` — the reduction core
`logSoftmaxOf` applied to the tile rebuilt from `xs` (with the kernel's own
`other = -inf` padding on the masked-off lanes, which is what makes the
row reduction independent of memory outside the window) — and every other
memory cell, including the masked-off lanes of the output window, is
unchanged.

Two honest, **truth-required** side hypotheses: `hOutInj` (distinct lanes of a
program's tile must not alias the same output cell — otherwise only the last
store survives) and `hBN : 0 < BLOCK_N` (`tl.max` over a zero-length reduction
axis has no value and `exec` returns `none`, so termination genuinely fails). -/
specification log_softmax_kernel_correctness
    (output_ptr input_ptr : RegionName)
    (M N K BLOCK_M BLOCK_N : Nat) (hBN : 0 < BLOCK_N)
    (hOutInj : ∀ p₀ p₁ : Nat, Function.Injective
      (fun j : Fin (BLOCK_M * BLOCK_N) =>
        outOffset p₀ p₁ N K BLOCK_M (laneIdx BLOCK_M BLOCK_N j))) :
    logSoftmaxIO output_ptr input_ptr M N K BLOCK_M BLOCK_N
      ⊨ fun p₀ _ xs j =>
          logSoftmaxOf BLOCK_M BLOCK_N
            (logSoftmaxPureTile p₀ M N BLOCK_M BLOCK_N xs)
            (laneIdx BLOCK_M BLOCK_N j) := by
  refine Masked2DKernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact log_softmax_kernel_flattenOk output_ptr input_ptr M N K BLOCK_M BLOCK_N
  · intro bounds s hin hout _
    exact log_softmax_kernel_traceSafe output_ptr input_ptr M N K BLOCK_M BLOCK_N
      hBN bounds s hin hout
  · intro s₀ xs hx
    obtain ⟨s1, hs1, hval, hframe⟩ :=
      log_softmax_kernel_region_run output_ptr input_ptr M N K BLOCK_M BLOCK_N
        hBN s₀ xs (inj_of_lane_inj (hOutInj (s₀.pids 0) (s₀.pids 1))) hx
    exact ⟨s1, hs1, hval, fun r o hcond _ => hframe r o hcond⟩

end VeriTile.Bench.TritonBenchG.LogSoftmax
