import bench.tritonbench_g.bgmv_shrink_kernel.Proofs.RealTail

/-! Full real execution of the actual signed source, including both output
branches and every split-K iteration. The formula reads only initial memory.
Allocation bounds and flat-memory transport are separate obligations. -/
namespace VeriTile.Bench.TritonBenchG.BgmvShrinkKernel.RealProgram
open VeriTile.Triton RealSetupSource RealSetup
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

/-- This program's rank-slice GEMV contribution. Tail lanes outside K
contribute zero; K need not be divisible by the loop step. -/
def contribution (p : Parameters) (initial : BlockState) (slot n : Nat) : ℝ :=
  p.scaling * ∑ c ∈ Finset.range ((p.K + p.BK * p.splits - 1) / (p.BK * p.splits)),
    ∑ e : Fin p.BK,
      let k := c * (p.BK * p.splits) + (e.val + initial.pids 0 * p.BK)
      if k < p.K then
        initial.readMem p.input (initial.pids 1 * p.inputBatchStride + k) *
        initial.readMem p.weights (p.slotStride * slot + n * p.rowStride + k * p.colStride)
      else 0

/-- A single split overwrites output; multiple splits use atomic addition
and retain the cell's value from before this program. -/
def formula (p : Parameters) (initial : BlockState) (slot n : Nat) : ℝ :=
  if p.splits = 1 then contribution p initial slot n
  else initial.readMem p.output (RealTail.offset p (initial.pids 1) n) + contribution p initial slot n

def Misses (p : Parameters) (initial : BlockState) (r : RegionName) (o : Nat) : Prop :=
  ∀ n : Fin p.BN, n.val < p.N → ¬(p.output = r ∧ RealTail.offset p (initial.pids 1) n.val = o)

theorem contribution_eq (p : Parameters) (initial : BlockState) (slot n : Nat) :
    contribution p initial slot n = RealLoop.scaledContribution initial p.input p.weights slot p.scaling
      p.K p.inputBatchStride p.slotStride p.rowStride p.colStride p.BK p.splits n := rfl

theorem body_active (p : Parameters) (swapped : Bool) (initial : BlockState) (slot : Nat)
    (hm : initial.readMemValue .int (Region.cast p.indices) (initial.pids 1) = Int.ofNat slot)
    (hS : 0 < p.BK * p.splits) (hstride : 0 < p.outputStride) :
    ∃ final, stepStmts (p.body swapped) initial = some final ∧
      (∀ n : Fin p.BN, n.val < p.N →
        final.readMem p.output (RealTail.offset p (initial.pids 1) n.val) = formula p initial slot n.val) ∧
      (∀ r o, Misses p initial r o → final.mem r o = initial.mem r o) := by
  obtain ⟨counter, st, he, hc, hi⟩ := loop_exec p initial slot hS
  obtain ⟨hpids, hic, hbound, hcb, hon, hok, hap, hbp, hacc, hmem⟩ := hi
  let g : Fin p.BN → ℝ := fun n => if n.val < p.N then
    RealLoop.accPartial initial p.input p.weights slot p.K p.inputBatchStride p.slotStride
      p.rowStride p.colStride p.BK p.splits (counter / (p.BK * p.splits)) n.val else 0
  have ht := RealTail.tail_exec p swapped st (initial.pids 1) g hcb hacc
  refine ⟨RealTail.result p st (initial.pids 1) g, ?_, ?_, ?_⟩
  · unfold Parameters.body
    rw [stepStmts.append_some (prelude_exec p initial (Int.ofNat slot) hm)]
    have hg : evalOp Parameters.guard (afterPrelude initial (Int.ofNat slot)) =
        some (Tile.scalar Bool.true) := by
      have hh := guard_eval (afterPrelude initial (Int.ofNat slot)) (Int.ofNat slot)
        (by simp [afterPrelude])
      have hn : Int.ofNat slot ≠ -1 := by
        change (slot : ℤ) ≠ -1
        omega
      simpa only [decide_eq_true hn] using hh
    have hb : stepStmt (.ifThen Parameters.guard (p.setup ++ p.loop :: p.tail swapped))
        (afterPrelude initial (Int.ofNat slot)) = some (RealTail.result p st (initial.pids 1) g) := by
      simp only [stepStmt, hg, Option.bind_some, Tile.scalar_data, if_true]
      rw [stepStmts.append_some (setup_exec p initial slot), stepStmts.cons_some he]
      exact ht
    rw [stepStmts.cons_some hb, stepStmts.nil]
  · intro n hn
    rw [RealTail.tail_readback p st (initial.pids 1) g hstride n hn]
    have hread : st.readMem = initial.readMem := by
      funext r o
      unfold BlockState.readMem
      rw [hmem]
    simp only [g, if_pos hn, hc, formula, contribution_eq, RealLoop.scaledContribution, hread]
    rw [mul_comm]
  · intro r o hmiss
    rw [RealTail.tail_frame p st (initial.pids 1) g r o hmiss, hmem]

/-- The entire source successfully executes and produces every N output
lane, provided the launch tile covers N. All cells outside active writes are
preserved. Inputs, weights and output may alias: the loop reads before writes. -/
theorem source_executes (p : Parameters) (storage : TypedSource.Storage)
    (out_dtype : FloatDType) (swapped : Bool) (initial : BlockState) (slot : Nat)
    (hm : initial.readMemValue .int (Region.cast p.indices) (initial.pids 1) = Int.ofNat slot)
    (hS : 0 < p.BK * p.splits) (hstride : 0 < p.outputStride) (hN : p.N ≤ p.BN) :
    ∃ final, exec (p.source storage out_dtype swapped).eraseDType.toAlgKernel initial = some final ∧
      (∀ n, n < p.N → final.readMem p.output (RealTail.offset p (initial.pids 1) n) =
        formula p initial slot n) ∧
      (∀ r o, Misses p initial r o → final.mem r o = initial.mem r o) := by
  obtain ⟨final, he, hv, hf⟩ := body_active p swapped initial slot hm hS hstride
  refine ⟨final, ?_, ?_, hf⟩
  · simpa only [exec, Parameters.source_body] using he
  · intro n hn
    exact hv ⟨n, lt_of_lt_of_le hn hN⟩ hn

/-- Signed -1 skips all data loads, loop steps and output writes. This also
covers degenerate launch parameters since the active branch never runs. -/
theorem sentinel_skips (p : Parameters) (storage : TypedSource.Storage)
    (out_dtype : FloatDType) (swapped : Bool) (initial : BlockState)
    (hm : initial.readMemValue .int (Region.cast p.indices) (initial.pids 1) = -1) :
    ∃ final, exec (p.source storage out_dtype swapped).eraseDType.toAlgKernel initial = some final ∧
      final.mem = initial.mem := by
  refine ⟨afterPrelude initial (-1), ?_, rfl⟩
  simp only [exec, Parameters.source_body, Parameters.body]
  rw [stepStmts.append_some (prelude_exec p initial (-1) hm)]
  have hg : evalOp Parameters.guard (afterPrelude initial (-1)) = some (Tile.scalar Bool.false) :=
    guard_eval _ (-1) (by simp [afterPrelude])
  have hb : stepStmt (.ifThen Parameters.guard (p.setup ++ p.loop :: p.tail swapped))
      (afterPrelude initial (-1)) = some (afterPrelude initial (-1)) := by
    simp only [stepStmt, hg, Option.bind_some, Tile.scalar_data, Bool.false_eq_true, if_false]
    rfl
  rw [stepStmts.cons_some hb, stepStmts.nil]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BgmvShrinkKernel.RealProgram
