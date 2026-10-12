import VeriTile.Triton.DSL
import VeriTile.Triton.Semantics.Step
import VeriTile.Meta.StatementAudit

/-! Expression subtraction keeps pointer storage types and accumulates a
static region's signed offset before natural-address conversion. -/
namespace PointerSubtractionTests
open VeriTile.Triton
noncomputable section
set_option linter.unusedSimpArgs false

def naturalArithmetic (a b : Nat) : ComputeKernel := triton {
  x = $(a) - $(b)
}

def realArithmetic (a b : ℝ) : ComputeKernel := triton {
  x = $((a : ℝ)) - $((b : ℝ))
}

/-- Ordinary subtraction must not be misclassified as a pointer operation. -/
theorem natural_subtraction (a b : Nat) :
    (naturalArithmetic a b).toAlgKernel.body =
      [.assign .nat [] "x" (.sub .nat .nil (.constNat a) (.constNat b))] := rfl

theorem real_subtraction (a b : ℝ) :
    (realArithmetic a b).toAlgKernel.body =
      [.assign .real [] "x" (.sub .real .nil (.const a) (.const b))] := rfl

def kernel (index data : Region .int) : ComputeKernel := triton {
  tl.region index = tl.int32, data = tl.int32
  i = (tl.load(index)).to(tl.int32)
  positive = tl.load(data + $(13) - $(2))
  negative = tl.load(data + $(13) - i)
  recovered = tl.load(data + $(2) - $(5) + $(7))
  offsets = tl.arange(0, $(3))
  tile = tl.load(data + $(13) + offsets - i)
  p = data + $(13) + offsets
  -- The DSL explicitly annotates the element type of a pointer register.
  dynamic = tl.load(p - $(2), dtype=tl.int32)
}

def initial : BlockState :=
  { mem := fun r o => MemCell.of .int (if r = "index" then -2 else (o : Int)),
    regs := fun _ _ _ => none, pids := fun _ => 0 }

def result : BlockState :=
  let s := initial.setReg "i" .int [] (Tile.scalar (-2))
  let s := s.setReg "positive" .int [] (Tile.scalar 11)
  let s := s.setReg "negative" .int [] (Tile.scalar 15)
  let s := s.setReg "recovered" .int [] (Tile.scalar 4)
  let s := s.setReg "offsets" .nat [3] (Tile.vec Fin.val)
  let s := s.setReg "tile" .int [3] (Tile.vec (fun i => (15 : Int) + (i.val : Int)))
  let s := s.setReg "p" .ptr [3] (Tile.vec (fun i => ("data",13 + i.val)))
  s.setReg "dynamic" .int [3] (Tile.vec (fun i => (11 : Int) + (i.val : Int)))

set_option maxHeartbeats 3000000 in
theorem complete_execution :
    stepStmts (kernel ⟨"index"⟩ ⟨"data"⟩).toAlgKernel.body initial = some result := by
  simp [kernel, ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?,
    ComputeKernel.fromKernelBody, ComputeKernel.surfaceBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    stepStmts, stepStmt, evalOp.eq_def, bind, Option.bind, Except.bind, initial, result,
    BlockState.readMemValue, BlockState.readMemTyped, MemCell.readAs, MemCell.of,
    NumericDType.add, NumericDType.sub, NumericDType.mul, Tile.bop, Tile.scalar, Tile.vec,
    Region.cast, -Tile.scalar_eta]
  congr 2
  · congr 1
    apply Tile.ext
    intro i
    change max ((13 : Int) + i.1.val + 2) 0 = 15 + i.1.val
    omega
  · simp [Tile.ptrAdd, Broadcast.leftIndex, Broadcast.rightIndex]
  · funext i
    omega

#guard_msgs (drop info) in
#auditModuleAxioms
end
end PointerSubtractionTests
