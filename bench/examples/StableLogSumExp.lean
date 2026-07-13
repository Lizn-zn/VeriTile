import VeriTile.Triton
import VeriTile.Examples.Common
import VeriTile.Meta.StatementAudit

/-!
# Log-sum-exp: direct vs shift-trick — rounding-invariant writes-equality

Self-contained showcase, read top to bottom: **kernels** first (what we
built), then the **theorem** (one public headline
`log_sum_exp_refinement_view`; the proof is direct, so there is no `private`
lemma section), then a compile-time **trust audit**. The three real sections
below are `LogSumExp.kernels`, `LogSumExp.theorems`, and `LogSumExp.exact`
(the exact-ℝ companion layer, dissolved from the former library example
`VeriTile.Examples.LogSumExpEq`).

Two block-parallel log-sum-exp kernels compute `y = log Σ exp(x)` per row and
**store the result rounded to bf16** (`(y).to(tl.bfloat16)`): `directLSEKernel`
sums `exp(x)` raw; `stableLSEKernel` subtracts the row max first
(`m + log Σ exp(x − m)`), the numerically-stable rewrite. The reductions run in
ℝ (no intermediate rounding), so both kernels produce the **same** ℝ value —
the shift-invariance of log-sum-exp (`log_sum_exp_shift_invariant`) — and the
only rounding is the shared bf16 output store. Any rounding model therefore
quantizes the two equal ℝ values identically, so the writes agree.

## The public result (bottom of file)

The single public headline is **`log_sum_exp_refinement_view`** — a
kernel-vs-kernel refinement on `ComputeRefine.Refines` (the rounding-model
surface): for the fixed rounding model `R`, from the same state the direct and
shift-trick kernels perform the same writes (no scratch regions, so the scratch
list is `[]`). Its statement mentions only the two kernels, the rounding-model
surface, and the state/region types — **no spec, and no input precondition**
(the `#stmtSurfaceSubset` gate below enforces this). This is the boundary-
rounded analogue of the exact-ℝ refinement; the compositional rounding pattern
is `bench/examples/FusedSwiglu.lean`.

Alongside it, the `Exact` namespace (section `LogSumExp.exact`) keeps the
exact-store layer of the dissolved library example: the same two kernels
minus the bf16 output cast, their per-kernel closed-form correctness
(`Exact.direct_lse_correct`, `Exact.stable_lse_correct`), the exact-ℝ
refinement `Exact.log_sum_exp_refinement`, and its `TensorView` exec view.
-/

namespace VeriTile.Bench.Examples.LogSumExp

open VeriTile.Triton VeriTile.Triton.TiledLogSumExp VeriTile.Triton.TiledSoftmax
open VeriTile.Examples

/-! ## Kernels -/
section LogSumExp.kernels

/-- Direct log-sum-exp: `y = log Σ exp(x)`, summed raw (no max shift), stored
rounded to bf16. -/
def directLSEKernel (xReg yReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(blockSize) + tl.arange(0, $(blockSize))
  x    := tl.load($(xReg) + offs)
  e    := tl.exp(x)
  s    := tl.sum(e, axis=0)
  y    := tl.log(s)
  tl.store($(yReg) + pid, (y).to(tl.bfloat16))
}

/-- Numerically-stable log-sum-exp: subtract the row max first,
`y = m + log Σ exp(x − m)`, stored rounded to bf16. -/
def stableLSEKernel (xReg yReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(blockSize) + tl.arange(0, $(blockSize))
  x    := tl.load($(xReg) + offs)
  m    := tl.max(x, axis=0)
  e    := tl.exp(x - m)
  s    := tl.sum(e, axis=0)
  y    := m + tl.log(s)
  tl.store($(yReg) + pid, (y).to(tl.bfloat16))
}

end LogSumExp.kernels

/-! ## The headline theorem -/
section LogSumExp.theorems

/- Shared parameters of the headline, hoisted to a `variable` block: the two
regions, the block size (nonempty), the state, and the rounding model `R`. -/
variable (xReg yReg : RegionName) (N : Nat) (hN : 0 < N) (s : BlockState)
variable (R : RoundingModel)

include hN in
/-- **direct refines shift-trick** (`ComputeRefine.Refines R`, no scratch): for
the rounding model `R`, from the same initial state the direct and shift-trick
LSE kernels perform the same writes — their final memories agree at every cell.
Both compute the same ℝ log-sum-exp (shift-invariance) and round it at the same
bf16 store, so `R` quantizes equal values equally. -/
specification log_sum_exp_refinement_view :
    ComputeRefine.Refines R
      (directLSEKernel xReg yReg N)
      (stableLSEKernel xReg yReg N) s [] := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hN.ne'
  apply ComputeKernel.computeRefineR_of_toAlgKernel rfl rfl
  intro s0 lhs' rhs' hL hR hs0
  subst s0
  intro r hr o
  simp [execR, directLSEKernel, stepStmtsR, stepStmtR, evalOpR.eq_def,
        Tile.bop, Tile.uop, NumericDType.add, NumericDType.mul,
        ComputeExpr.toAlgorithm?] at hL
  repeat unfold evalOp at hL
  simp [Tile.reduceSum, Tile.reduceSumDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex] at hL
  simp [execR, stableLSEKernel, stepStmtsR, stepStmtR, evalOpR.eq_def,
        Tile.bop, Tile.uop, NumericDType.add, NumericDType.mul, NumericDType.sub,
        ComputeExpr.toAlgorithm?] at hR
  repeat unfold evalOp at hR
  simp [Tile.reduceSum, Tile.reduceSumDrop,
        Tile.reduceMax, Tile.reduceMaxDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex] at hR
  subst lhs'
  subst rhs'
  rw [BlockState.writeMemAsR_mem, BlockState.writeMemAsR_mem]
  by_cases hc : r = yReg ∧ o = s.pids 0
  · rw [if_pos hc, if_pos hc]
    -- both store bf16-rounded copies of the SAME ℝ value (shift-invariance)
    exact congrArg (fun v : ℝ => MemCell.of _ (FloatDType.bf16.ofReal (R.storeValue .bf16 (FloatDType.bf16.ofReal (R.round .bf16 v)))))
      (log_sum_exp_shift_invariant hN _ _)
  · rw [if_neg hc, if_neg hc]
    rfl

/-! ## Trust audit (compile-time gate)

These commands re-audit the public result every time the file is elaborated —
if either gate fails (a smuggled axiom / `sorry`, or a foreign constant in the
trusted statement) the file stops compiling. See
`VeriTile.Meta.StatementAudit`. -/

-- (1) No `sorry`, no smuggled axiom, in the public theorem's transitive proof.
#axiomsClean log_sum_exp_refinement_view

-- (2) The headline is a *kernel-vs-kernel* refinement: its statement may mention
-- ONLY the two kernels, the rounding-model surface, and the state/region types.
#stmtSurfaceSubset log_sum_exp_refinement_view ⊆
  [directLSEKernel, stableLSEKernel, ComputeRefine.Refines, RoundingModel,
   BlockState, RegionName]

end LogSumExp.theorems

/-! ## Exact-ℝ layer (dissolved library example)

The former `VeriTile.Examples.LogSumExpEq` proved the same refinement for the
**exact-store** kernel variants — identical programs except the output store
writes the raw ℝ value (no `(y).to(tl.bfloat16)` cast). That layer lives on
here under the `Exact` namespace: single-cell observation `observeLSE`,
per-kernel correctness against the closed forms `LSE` /
`stableLSEWithShift`, the exact-ℝ refinement, and its `TensorView`
exec-level view. The math identity `log_sum_exp_shift_invariant` is the
load-bearing fact, exactly as in the rounded headline above. -/
section LogSumExp.exact

namespace Exact

/-- Direct log-sum-exp kernel, exact-store variant: `y = log(Σ exp(x))`,
stored unrounded. -/
def directLSEKernel (xReg yReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(blockSize) + tl.arange(0, $(blockSize))
  x    := tl.load($(xReg) + offs)
  e    := tl.exp(x)
  s    := tl.sum(e, axis=0)
  y    := tl.log(s)
  tl.store($(yReg) + pid, y)
}

/-- Shift-trick LSE kernel, exact-store variant: `y = m + log(Σ exp(x - m))`
where `m = max(x)`, stored unrounded. -/
def stableLSEKernel (xReg yReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(blockSize) + tl.arange(0, $(blockSize))
  x    := tl.load($(xReg) + offs)
  m    := tl.max(x, axis=0)
  e    := tl.exp(x - m)
  s    := tl.sum(e, axis=0)
  y    := m + tl.log(s)
  tl.store($(yReg) + pid, y)
}

/-- Read region `region` at offset `basePid` (single scalar per program_id).
    Unlike softmax which writes a tile, LSE writes one cell per pid. -/
noncomputable def observeLSE
    (sf : Option BlockState) (region : RegionName) (basePid : Nat) : Option ℝ :=
  sf.map (·.readMem region basePid)

/-- **Direct LSE kernel correctness.** Single-cell observation at Y[pid]. -/
theorem direct_lse_correct
    (xReg yReg : RegionName)
    (N : Nat) (_hN : 0 < N) (s : BlockState) (xs : Fin N → ℝ)
    (_h_x : InputLoadedAt s xReg N xs) :
    observeLSE (exec (directLSEKernel xReg yReg N) s) yReg s.pid
      = some (LSE xs Bool.false 0) := by
  simp [observeLSE, exec, directLSEKernel, stepStmts, stepStmt, evalOp,
        Tile.bop, Tile.uop, Tile.reduceSum, Tile.reduceSumDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        NumericDType.add, NumericDType.mul, LSE]
  repeat unfold evalOp
  simp [observeLSE, exec, directLSEKernel, stepStmts, stepStmt,
        Tile.bop, Tile.uop, Tile.reduceSum, Tile.reduceSumDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        NumericDType.add, NumericDType.mul, LSE]
  unfold InputLoadedAt at _h_x
  simp_rw [_h_x]
  rfl

/-- **Stable LSE kernel correctness.** Single-cell observation; closed form
    is `m + log(Σ exp(x - m))` where `m = tileMax xs`. -/
theorem stable_lse_correct
    (xReg yReg : RegionName)
    (N : Nat) (hN : 0 < N) (s : BlockState) (xs : Fin N → ℝ)
    (_h_x : InputLoadedAt s xReg N xs) :
    observeLSE (exec (stableLSEKernel xReg yReg N) s) yReg s.pid
      = some (stableLSEWithShift xs (tileMax hN xs)) := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hN.ne'
  simp [observeLSE, exec, stableLSEKernel, stepStmts, stepStmt, evalOp,
        Tile.bop, Tile.uop, Tile.reduceSum, Tile.reduceSumDrop,
        Tile.reduceMax, Tile.reduceMaxDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        NumericDType.add, NumericDType.mul, NumericDType.sub,
        stableLSEWithShift, tileMax]
  repeat unfold evalOp
  simp [observeLSE, exec, stableLSEKernel, stepStmts, stepStmt,
        Tile.bop, Tile.uop, Tile.reduceSum, Tile.reduceSumDrop,
        Tile.reduceMax, Tile.reduceMaxDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        NumericDType.add, NumericDType.mul, NumericDType.sub,
        stableLSEWithShift, tileMax]
  unfold InputLoadedAt at _h_x
  simp_rw [_h_x]
  rfl

/-- **Refinement: `Exact.directLSEKernel` and `Exact.stableLSEKernel` produce
    the same `Y[pid]` value.** Composes the two correctness lemmas via the
    math identity `log_sum_exp_shift_invariant`. -/
theorem log_sum_exp_refinement
    (xReg yReg : RegionName)
    (N : Nat) (hN : 0 < N) (s : BlockState) (xs : Fin N → ℝ)
    (h_x : InputLoadedAt s xReg N xs) :
    observeLSE (exec (directLSEKernel xReg yReg N) s) yReg s.pid =
    observeLSE (exec (stableLSEKernel xReg yReg N) s) yReg s.pid := by
  rw [direct_lse_correct xReg yReg N hN s xs h_x,
      stable_lse_correct xReg yReg N hN s xs h_x]
  congr 1
  unfold LSE stableLSEWithShift
  simp
  exact log_sum_exp_shift_invariant hN xs (tileMax hN xs)

/-- View-level surface for `Exact.log_sum_exp_refinement`. -/
theorem log_sum_exp_refinement_exec_view
    (xReg yReg : RegionName)
    (N : Nat) (hN : 0 < N) (s : BlockState) (xs : Fin N → ℝ)
    (h_x : TensorView.loaded s (programTileView s xReg N)
      (fun idx : TileIndex [N] => xs idx.1)) :
    TensorView.observe (exec (directLSEKernel xReg yReg N) s)
        (scalarCellView yReg s.pid) PUnit.unit =
    TensorView.observe (exec (stableLSEKernel xReg yReg N) s)
        (scalarCellView yReg s.pid) PUnit.unit := by
  have hx := inputLoadedAt_of_programTileView_loaded (s := s) (region := xReg)
    (N := N) (xs := xs) h_x
  simpa [TensorView.observe, observeTileAt, scalarCellView,
         TensorView.offset, Offset.strided, observeLSE]
    using log_sum_exp_refinement xReg yReg N hN s xs hx

end Exact

/-! ### Trust audit for the exact-ℝ layer -/

-- No `sorry`, no smuggled axiom, anywhere in the exact layer's proofs.
#axiomsClean Exact.direct_lse_correct
#axiomsClean Exact.stable_lse_correct
#axiomsClean Exact.log_sum_exp_refinement
#axiomsClean Exact.log_sum_exp_refinement_exec_view

end LogSumExp.exact

end VeriTile.Bench.Examples.LogSumExp
