import bench.examples.RowWiseSum.Kernels
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Data.Fin.Rev
import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL
import VeriTile.Triton.Math.Reduction
import VeriTile.Triton.Memory.KernelSpec
import VeriTile.Examples.Common
import VeriTile.Meta.Specification
import VeriTile.Meta.StatementAudit

/-! Real execution, arithmetic and memory lemmas for RowWiseSum.
The public IO contracts and mathematical specifications are in ../Correct.lean. -/

namespace VeriTile.Bench.Examples.RowWiseSum
open VeriTile.Bench.Examples.RowWiseSum.Kernels

attribute [local simp] VeriTile.Triton.ComputeExpr.toAlgorithm?
  VeriTile.Triton.ComputeOp.toAlgorithm? VeriTile.Triton.ComputeDType.eraseDType

open VeriTile.Triton
open VeriTile.Triton.KernelIO₁ (Implements)
open scoped VeriTile.Triton.KernelIO₁
open VeriTile.Examples

/-- Value half: after the run on a state where row `s.pid` of `xReg` holds
the tile `xs`, the cell `yReg[s.pid]` equals `∑ xs`. -/
theorem rowWiseSum_correct
    (xReg yReg : RegionName) (nCol blockSize : Nat)
    (s : BlockState) (xs : Fin blockSize → ℝ)
    (h_x : InputRowLoadedAt s xReg nCol blockSize xs) :
    observeRowAt (exec (rowWiseSumKernel xReg yReg nCol blockSize) s) yReg s.pid
      = some (∑ k, xs k) := by
  simp [observeRowAt, exec, rowWiseSumKernel, stepStmts, stepStmt,
        Tile.bop, TileShape.eraseAxis, NumericDType.mul, NumericDType.add]
  unfold InputRowLoadedAt at h_x
  simp_rw [h_x]
  apply Exists.intro
  constructor
  · unfold evalOp
    simp [Tile.reduceSum, Tile.reduceSumDrop,
      TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex]
    rfl
  · simp [BlockState.writeMem_readMem]
    rfl

/-- Frame half: the single-cell store at `y[pid]` leaves every other memory
cell unchanged. -/
private theorem rowWiseSum_frame (xReg yReg : RegionName)
    (nCol B : Nat) (s s1 : BlockState)
    (hExec : exec ((rowWiseSumKernel xReg yReg nCol B).toAlgKernel) s
      = some s1)
    (r : RegionName) (o : Nat) (hmiss : ¬(yReg = r ∧ s.pid = o)) :
    s1.mem r o = s.mem r o := by
  simp [exec, rowWiseSumKernel, ComputeKernel.toAlgKernel, stepStmts,
    stepStmt, Tile.bop, NumericDType.add, NumericDType.mul] at hExec
  repeat unfold evalOp at hExec
  simp [Tile.reduceSum, Tile.reduceSumDrop, TileShape.axisDim,
    TileShape.eraseAxis, TileShape.insertAxisIndex] at hExec
  subst hExec
  rw [BlockState.writeMem_mem]
  exact if_neg (fun hc => hmiss ⟨hc.1.symm, hc.2.symm⟩)

/-- **The region-model Hoare triple** — termination, the output-cell value,
and frame, from any launch state whose input row window is loaded. This is
what the `⊨` headline transports to flat memory. -/
theorem rowWiseSum_region_run (nCol B : Nat)
    (s₀ : BlockState) (xs : Fin B → ℝ)
    (hx : ∀ j : Fin B, s₀.readMem ⟨"x"⟩ (s₀.pid * nCol + j.val) = xs j) :
    ∃ s1, exec ((rowWiseSumKernel ⟨"x"⟩ ⟨"y"⟩ nCol B).toAlgKernel) s₀
        = some s1
      ∧ (∀ j : Fin 1, s1.readMem ⟨"y"⟩ (s₀.pid + j.val) = ∑ k, xs k)
      ∧ (∀ r o,
          (r ≠ ⟨"y"⟩ ∨ ∀ j : Fin 1, o ≠ s₀.pid + j.val) →
          s1.mem r o = s₀.mem r o) := by
  have hobs := rowWiseSum_correct ⟨"x"⟩ ⟨"y"⟩ nCol B s₀ xs hx
  rw [show exec (rowWiseSumKernel ⟨"x"⟩ ⟨"y"⟩ nCol B) s₀
      = exec ((rowWiseSumKernel ⟨"x"⟩ ⟨"y"⟩ nCol B).toAlgKernel) s₀ from rfl]
    at hobs
  cases hsrc : exec ((rowWiseSumKernel ⟨"x"⟩ ⟨"y"⟩ nCol B).toAlgKernel) s₀ with
  | none =>
      rw [hsrc] at hobs
      simp [observeRowAt] at hobs
  | some s1 =>
      rw [hsrc] at hobs
      refine ⟨s1, rfl, fun j => ?_, fun r o hcond => ?_⟩
      · have hj : (j : Nat) = 0 := Nat.lt_one_iff.mp j.isLt
        simpa [observeRowAt, hj] using hobs
      · refine rowWiseSum_frame ⟨"x"⟩ ⟨"y"⟩ nCol B s₀ s1 hsrc r o
          (fun ⟨hr, ho⟩ => ?_)
        rcases hcond with hne | hno
        · exact hne hr.symm
        · exact hno ⟨0, Nat.one_pos⟩ (by simpa using ho.symm)

/-- Inversion for a successful `assign` step. -/
private theorem stepStmt_assign_inv {d : TileDType} {sh : TileShape}
    {nm : RegName} {e : Op d sh} {s s' : BlockState}
    (h : stepStmt (.assign d sh nm e) s = some s') :
    ∃ v, evalOp e s = some v ∧ s' = s.setReg nm d sh v := by
  simp only [stepStmt] at h
  cases hv : evalOp e s with
  | none => rw [hv] at h; exact absurd h (by simp)
  | some v =>
      rw [hv] at h
      replace h : some (s.setReg nm d sh v) = some s' := h
      exact ⟨v, rfl, (Option.some_inj.mp h).symm⟩

/-- Bounds discharge for the row-stride load: in any state whose `row`
register holds `pid₀` and whose `cols` register holds `arange B`, all
addressed cells `pid₀ * nCol + i` lie below `pid₀ * nCol + B ≤ bounds reg`. -/
private theorem rowCols_activeAddressSafe (nCol B : Nat)
    (bounds : RegionBounds) (pid₀ : Nat) (t : BlockState)
    (active : TileIndex [B] → Prop)
    (hrow : t.regs .nat [] "row" = some (Tile.scalar (dtype := .nat) pid₀))
    (hcols : t.regs .nat [B] "cols" = some (Tile.vec fun i => i.val))
    (reg : RegionName) (hreg : pid₀ * nCol + B ≤ bounds reg) :
    MemAccess.ActiveAddressSafe bounds
      (MemAccess.region reg (Op.add .nat .scalarL
        (Op.mul .nat .nil (Op.ref .nat [] "row") (Op.constNat nCol))
        (Op.ref .nat [B] "cols"))) t active := by
  simp only [MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe]
  intro offsets hoffs i _
  rw [show evalOp (Op.add .nat .scalarL
      (Op.mul .nat .nil (Op.ref .nat [] "row") (Op.constNat nCol))
      (Op.ref .nat [B] "cols")) t
      = some ⟨fun i => pid₀ * nCol + i.1.val⟩ from by
    simp [hrow, hcols, Tile.bop, NumericDType.nat_add, NumericDType.nat_mul,
      Tile.vec]] at hoffs
  obtain rfl := Option.some_inj.mp hoffs
  simp only [Region.cast_self]
  exact lt_of_lt_of_le (Nat.add_lt_add_left i.1.isLt _) hreg

/-- Bounds discharge for the single-cell store through the scalar `row`
register: the one addressed cell is `pid₀`, in bounds when
`pid₀ + 1 ≤ bounds reg`. -/
private theorem row_activeAddressSafe (bounds : RegionBounds)
    (pid₀ : Nat) (t : BlockState) (active : TileIndex [] → Prop)
    (hrow : t.regs .nat [] "row" = some (Tile.scalar (dtype := .nat) pid₀))
    (reg : RegionName) (hreg : pid₀ + 1 ≤ bounds reg) :
    MemAccess.ActiveAddressSafe bounds
      (MemAccess.region reg (Op.ref .nat [] "row")) t active := by
  simp only [MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe]
  intro offsets hoffs i _
  rw [show evalOp (Op.ref .nat [] "row") t
      = some (Tile.scalar (dtype := .nat) pid₀) from by simp [hrow]] at hoffs
  obtain rfl := Option.some_inj.mp hoffs
  simp only [Region.cast_self]
  exact hreg

set_option maxHeartbeats 1600000 in
/-- The row-wise sum kernel is trace-safe: of its five statements, only the
row-stride load and the single-cell store touch memory; the `sum` reduction
is memory-silent. -/
theorem rowWiseSum_traceSafe (xReg yReg : RegionName)
    (nCol B : Nat) (bounds : RegionBounds) (s : BlockState)
    (hx : s.pid * nCol + B ≤ bounds xReg) (hy : s.pid + 1 ≤ bounds yReg) :
    Kernel.TraceSafe bounds
      ((rowWiseSumKernel xReg yReg nCol B).toAlgKernel)
      s := by
  unfold Kernel.TraceSafe
  -- statement 1: row := program_id(0)
  refine Stmt.TraceSafeList.cons_intro (by simp [Stmt.TraceSafe, Op.SafeAt]) ?_
  intro s1 hs1
  obtain ⟨v1, hv1, rfl⟩ := stepStmt_assign_inv hs1
  rw [show evalOp (Op.programId 0) s
      = some (Tile.scalar (dtype := .nat) (s.pids 0)) from by simp] at hv1
  obtain rfl := Option.some_inj.mp hv1
  -- statement 2: cols := arange B
  refine Stmt.TraceSafeList.cons_intro (by simp [Stmt.TraceSafe, Op.SafeAt]) ?_
  intro s2 hs2
  obtain ⟨v2, hv2, rfl⟩ := stepStmt_assign_inv hs2
  rw [show evalOp (Op.arange B)
      (s.setReg "row" .nat [] (Tile.scalar (dtype := .nat) (s.pids 0)))
      = some (Tile.vec fun i => i.val) from by simp] at hv2
  obtain rfl := Option.some_inj.mp hv2
  -- statement 3: values := load(xReg + row * nCol + cols)   (unmasked)
  refine Stmt.TraceSafeList.cons_intro ?_ ?_
  · simp only [ComputeDType.eraseDType, Stmt.TraceSafe, Op.SafeAt]
    exact ⟨by simp, trivial,
      rowCols_activeAddressSafe nCol B bounds (s.pids 0) _ _
        (by simp [BlockState.setReg]) (by simp [BlockState.setReg]) xReg hx⟩
  intro s3 hs3
  obtain ⟨v3, hv3, rfl⟩ := stepStmt_assign_inv hs3
  -- statement 4: result := sum(values, axis=0)   (register op)
  refine Stmt.TraceSafeList.cons_intro
    (by simp [Stmt.TraceSafe, Op.SafeAt.eq_def]) ?_
  intro s4 hs4
  obtain ⟨v4, hv4, rfl⟩ := stepStmt_assign_inv hs4
  -- statement 5: store(yReg + row, result)   (single cell, unmasked)
  refine Stmt.TraceSafeList.cons_intro ?_ (fun _ _ => .nil_intro)
  simp only [Stmt.TraceSafe, MemAccess.SafeAt, MaskOpt.SafeAt]
  refine ⟨by simp [Op.SafeAt], by simp [Op.SafeAt], trivial,
    row_activeAddressSafe bounds (s.pids 0) _ _ ?_ yReg hy⟩
  simp [BlockState.setReg]

/-- The kernel sits inside the bridge's covered fragment. -/
theorem rowWiseSum_flattenOk (xReg yReg : RegionName) (nCol B : Nat) :
    ((rowWiseSumKernel xReg yReg nCol B).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [rowWiseSumKernel, ComputeKernel.toAlgKernel,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

section Optimized

set_option maxHeartbeats 1600000

theorem reversed_region_run (nCol B : Nat) (s : BlockState) (xs : Fin B → ℝ)
    (hx : ∀ i : Fin B, s.readMem "x" (s.pid * nCol + i.val) = xs i) :
    ∃ t, exec (reversedKernel "x" "y" nCol B).toAlgKernel s = some t ∧
      (∀ i : Fin 1, t.readMem "y" (s.pid + i.val) = ∑ k, xs k) ∧
      (∀ (r : RegionName) o, (r ≠ "y" ∨ ∀ i : Fin 1, o ≠ s.pid + i.val) →
        t.mem r o = s.mem r o) := by
  have hrev (i : Fin B) : s.readMem "x" (s.pid * nCol + (B - 1 - i.val)) = xs i.rev := by
    simpa only [Fin.val_rev, Nat.sub_sub, Nat.add_comm] using hx i.rev
  simp [reversedKernel, exec, stepStmts, stepStmt, evalOp.eq_def,
    Region.cast, Tile.bop, NumericDType.add, NumericDType.mul, NumericDType.sub,
    Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis,
    TileShape.insertAxisIndex]
  refine ⟨?_, ?_⟩
  · simpa [← BlockState.pid_eq, hrev] using Equiv.sum_comp Fin.revPerm xs
  · intro r o hmiss
    rw [BlockState.writeMem_mem]
    apply if_neg
    rintro ⟨hr, ho⟩
    rcases hmiss with hr' | ho'
    · exact hr' hr
    · exact ho' ho

end Optimized

end VeriTile.Bench.Examples.RowWiseSum
