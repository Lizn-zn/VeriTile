import VeriTile.Triton
import VeriTile.Examples.Common
import VeriTile.Meta.StatementAudit

/-!
# Softmax: naive vs stable — rounding-invariant writes-equality

Self-contained showcase, read top to bottom: **kernels** first, the
**supporting lemma** in the middle (`private` plumbing — a bf16-scatter
congruence), the **theorem** last (one public headline
`softmax_kernels_refinement_view`), then a compile-time **trust audit**. The
three real sections are `Softmax.kernels`, `Softmax.lemmas`, `Softmax.theorems`.

Two block-parallel softmax kernels compute `y = exp(x) / Σ exp(x)` per row and
**store the result rounded to bf16** (`(y).to(tl.bfloat16)`):
`naiveSoftmaxKernel` exponentiates `x` raw; `stableSoftmaxKernel` subtracts the
row max first. The reductions run in ℝ (no intermediate rounding), so both
kernels produce the **same** per-lane ℝ value — the shift-cancellation of
softmax (`naive_eq_stable`) — and the only rounding is the shared bf16 output
store, which quantizes equal values identically.

## The public result (bottom of file)

The single public headline is **`softmax_kernels_refinement_view`** — a
kernel-vs-kernel refinement on `ComputeRefine.Refines R` (the rounding-model
surface): for the rounding model `R`, from the same state the naive and stable
kernels perform the same writes (no scratch, no input precondition). The
compositional rounding pattern is `bench/examples/FusedSwiglu.lean`.
-/

namespace VeriTile.Bench.Examples.Softmax

open VeriTile.Triton VeriTile.Triton.TiledSoftmax

/-! ## Kernels -/
section Softmax.kernels

/-- Naive softmax: `y = exp(x) / Σ exp(x)`, exponentiated raw, stored bf16. -/
def naiveSoftmaxKernel (xReg yReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(blockSize) + tl.arange(0, $(blockSize))
  x    := tl.load($(xReg) + offs)
  e    := tl.exp(x)
  s    := tl.sum(e, axis=0)
  y    := e / s
  tl.store($(yReg) + offs, (y).to(tl.bfloat16))
}

/-- Numerically-stable softmax: subtract the row max first,
`y = exp(x − m) / Σ exp(x − m)`, stored bf16. -/
def stableSoftmaxKernel (xReg yReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(blockSize) + tl.arange(0, $(blockSize))
  x    := tl.load($(xReg) + offs)
  m    := tl.max(x, axis=0)
  e    := tl.exp(x - m)
  s    := tl.sum(e, axis=0)
  y    := e / s
  tl.store($(yReg) + offs, (y).to(tl.bfloat16))
}

end Softmax.kernels

/-! ## Supporting lemma (private plumbing) -/
section Softmax.lemmas

/-- Two `writeMemAsR` scatters over the same offsets agree cell-by-cell when
their per-lane values agree — the rounding-store analogue of
`BlockState.foldl_writeMem_mem_congr`. -/
private theorem foldl_writeMemAsR_mem_congr {α : Type} (R : RoundingModel)
    (dtype : FloatDType) {region : RegionName} (l : List α) (offsetFn : α → Nat)
    (vL vR : α → TileCarrier dtype.toTileDType)
    (hv : ∀ k ∈ l, vL k = vR k) (r : RegionName) (o : Nat) :
    ∀ sL sR : BlockState, sL.mem r o = sR.mem r o →
      (l.foldl (fun acc k => acc.writeMemAsR R dtype region (offsetFn k) (vL k)) sL).mem r o
        = (l.foldl (fun acc k => acc.writeMemAsR R dtype region (offsetFn k) (vR k)) sR).mem r o := by
  induction l with
  | nil => intro sL sR h; exact h
  | cons hd tl ih =>
      intro sL sR h
      refine ih (fun k hk => hv k (List.mem_cons_of_mem _ hk)) _ _ ?_
      rw [BlockState.writeMemAsR_mem, BlockState.writeMemAsR_mem,
          hv hd List.mem_cons_self]
      by_cases hc : r = region ∧ o = offsetFn hd
      · rw [if_pos hc, if_pos hc]
      · rw [if_neg hc, if_neg hc]; exact h

end Softmax.lemmas

/-! ## The headline theorem -/
section Softmax.theorems

/- Shared parameters of the headline: the two regions, the block size
(nonempty), the state, and the rounding model `R`. -/
variable (xReg yReg : RegionName) (blockSize : Nat) (hN : 0 < blockSize) (s : BlockState)
variable (R : RoundingModel)

include hN in
/-- **naive refines stable** (`ComputeRefine.Refines R`, no scratch): for the
rounding model `R`, from the same initial state the naive and stable softmax
kernels perform the same writes. Both compute the same per-lane ℝ softmax value
(shift-cancellation) and round it at the same bf16 store. -/
specification softmax_kernels_refinement_view :
    ComputeRefine.Refines R
      (naiveSoftmaxKernel xReg yReg blockSize)
      (stableSoftmaxKernel xReg yReg blockSize) s [] := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hN.ne'
  apply ComputeKernel.computeRefineR_of_toAlgKernel rfl rfl
  intro s0 lhs' rhs' hL hR hs0
  subst s0
  intro r hr o
  simp [execR, naiveSoftmaxKernel, stepStmtsR, stepStmtR, evalOpR.eq_def,
        Tile.bop, Tile.uop, NumericDType.add, NumericDType.mul, NumericDType.div,
        ComputeExpr.toAlgorithm?] at hL
  simp [Tile.reduceSumDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex] at hL
  simp [execR, stableSoftmaxKernel, stepStmtsR, stepStmtR, evalOpR.eq_def,
        Tile.bop, Tile.uop, NumericDType.add, NumericDType.mul, NumericDType.sub,
        NumericDType.div, ComputeExpr.toAlgorithm?] at hR
  simp [Tile.reduceSumDrop, Tile.reduceMaxDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex] at hR
  subst lhs'
  subst rhs'
  refine foldl_writeMemAsR_mem_congr R .bf16 _ _ _ _ ?_ r o _ _ rfl
  intro k _
  exact congrArg (fun v : ℝ => R.cast FloatDType.real FloatDType.bf16 (some v))
    (congrFun (naive_eq_stable
      (fun i : Fin (n + 1) => s.readMem xReg (s.pids 0 * (n + 1) + i.val)) _) k.1)

/-! ## Trust audit (compile-time gate)

If either gate fails (a smuggled axiom / `sorry`, or a foreign constant in the
trusted statement) the file stops compiling. See `VeriTile.Meta.StatementAudit`. -/

-- (1) No `sorry`, no smuggled axiom, in the public theorem's transitive proof.
#axiomsClean softmax_kernels_refinement_view

-- (2) The headline is a *kernel-vs-kernel* refinement: its statement may mention
-- ONLY the two kernels, the rounding-model surface, and the state/region types.
#stmtSurfaceSubset softmax_kernels_refinement_view ⊆
  [naiveSoftmaxKernel, stableSoftmaxKernel, ComputeRefine.Refines, RoundingModel,
   BlockState, RegionName]

end Softmax.theorems

end VeriTile.Bench.Examples.Softmax
