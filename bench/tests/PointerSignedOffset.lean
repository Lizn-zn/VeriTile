import VeriTile.Triton.DSL
import VeriTile.Triton.Semantics.Step
import VeriTile.Meta.StatementAudit

/-! A static region's total signed offset must be converted to a natural
address only after all addends are combined. Negative metadata is exercised
with a nonnegative final address, including scalar/tile broadcasting. -/
namespace PointerSignedOffsetTests
open VeriTile.Triton
noncomputable section
set_option linter.unusedSimpArgs false

def kernel (index data : Region .int) : ComputeKernel := triton {
  tl.region index = tl.int32, data = tl.int32
  i = (tl.load(index)).to(tl.int32)
  flat = tl.load(data + $((13 : Nat)) + i)
  grouped = tl.load(data + ($((13 : Nat)) + i))
  recovered = tl.load(data + i + $((13 : Nat)))
  chained = tl.load(data + i + $((10 : Nat)) + $(3))
  offsets = tl.arange(0, $(3))
  tile = tl.load(data + $((13 : Nat)) + i + offsets)
}

def initial : BlockState :=
  { mem := fun r o => MemCell.of .int (if r = "index" then -2 else (o : Int)),
    regs := fun _ _ _ => none, pids := fun _ => 0 }

def result : BlockState :=
  let s := initial.setReg "i" .int [] (Tile.scalar (-2))
  let s := s.setReg "flat" .int [] (Tile.scalar 11)
  let s := s.setReg "grouped" .int [] (Tile.scalar 11)
  let s := s.setReg "recovered" .int [] (Tile.scalar 11)
  let s := s.setReg "chained" .int [] (Tile.scalar 11)
  let s := s.setReg "offsets" .nat [3] (Tile.vec Fin.val)
  s.setReg "tile" .int [3] (Tile.vec (fun i => (11 : Int) + (i.val : Int)))

set_option maxHeartbeats 1500000 in
theorem actual_loads_use_total_offset :
    stepStmts (kernel ⟨"index"⟩ ⟨"data"⟩).toAlgKernel.body initial = some result := by
  simp [kernel, ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?,
    ComputeKernel.fromKernelBody, ComputeKernel.surfaceBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    stepStmts, stepStmt, evalOp.eq_def, bind, Option.bind, Except.bind, initial, result,
    BlockState.readMemValue, BlockState.readMemTyped, MemCell.readAs, MemCell.of,
    NumericDType.add, NumericDType.mul, Tile.bop, Tile.scalar, Tile.vec,
    Region.cast, Int.toNat, -Tile.scalar_eta]
  congr 2

theorem grouped_and_ungrouped_agree :
    result.regs .int [] "flat" = result.regs .int [] "grouped" ∧
    result.regs .int [] "flat" = result.regs .int [] "recovered" ∧
    result.regs .int [] "flat" = result.regs .int [] "chained" := by
  simp [result]

theorem negative_addend_is_not_clipped :
    result.regs .int [] "flat" = some (Tile.scalar 11) ∧
    result.regs .int [] "flat" ≠ some (Tile.scalar 13) := by
  have h11 : result.regs .int [] "flat" = some (Tile.scalar 11) := by simp [result]
  refine ⟨h11, ?_⟩
  intro h13
  have ht := Option.some.inj (h11.symm.trans h13)
  have he : (11 : Int) = 13 := congrArg (fun t : Tile .int [] => t.data PUnit.unit) ht
  omega

/-- These two naturals differ, so the test cannot be passed by clipping i
before adding the layout offset. -/
theorem early_conversion_changes_address : (13 : Nat) + (-2 : Int).toNat ≠ (13 + (-2 : Int)).toNat := by
  decide

#guard_msgs (drop info) in
#auditModuleAxioms
end
end PointerSignedOffsetTests
