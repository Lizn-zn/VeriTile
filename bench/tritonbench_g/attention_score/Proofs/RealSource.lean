import bench.tritonbench_g.attention_score.Proofs.RealSafety
import bench.tritonbench_g.attention_score.Proofs.RealExecution

/-! Source parameters used by the real IO assembly. The Python wrapper uses
one common query/key block size B; both complete typed variants remain visible
in TypedSource. The unused batch-count and inner output stride stay explicit. -/
namespace VeriTile.Bench.TritonBenchG.AttentionScore.RealSource
open VeriTile.Triton
set_option maxHeartbeats 1000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false
noncomputable section

structure Parameters where
  (Q K M Out : RegionName)
  (queryBatch queryHead queryRow queryDim : Nat)
  (keyBatch keyHead keyRow keyDim : Nat)
  (outputBatch outputHead outputInner : Nat)
  (batches heads kvHeads queryContext roundedContext keyContext : Nat)
  (offset size : Int)
  scale : ℝ
  (window complement evenM evenN : Bool)

def source (B D : Nat) (p : Parameters) (candidate : Bool) : ComputeKernel :=
  (if candidate then TypedSource.scalarLeft else TypedSource.original) p.Q p.K p.M p.Out
    p.queryBatch p.queryHead p.queryRow p.queryDim p.keyBatch p.keyHead p.keyRow p.keyDim
    p.outputBatch p.outputHead p.outputInner p.batches p.heads p.kvHeads p.queryContext p.roundedContext
    p.keyContext B D B p.offset p.size p.scale p.window p.complement p.evenM p.evenN rfl

def kernel (B D : Nat) (p : Parameters) (candidate : Bool) : ComputeKernel :=
  (source B D p candidate).eraseDType

def config (B D : Nat) (p : Parameters) (candidate : Bool) (s : BlockState) : RealLoop.Config :=
  RealPrelude.config s p.Q p.K p.M p.heads p.kvHeads p.queryContext p.roundedContext p.keyContext B D
    p.queryBatch p.queryHead p.queryRow p.queryDim p.keyBatch p.keyHead p.keyRow p.keyDim
    p.offset p.size p.scale p.window p.complement p.evenM p.evenN candidate

def blocks (B : Nat) (p : Parameters) : Nat := RealLoop.tripCount p.roundedContext B
def rows (B : Nat) (p : Parameters) : Nat := blocks B p * B
def outputBase (p : Parameters) (headBatch : Nat) : Nat :=
  headBatch / p.heads * p.outputBatch + headBatch % p.heads * p.outputHead

theorem source_projection (B D : Nat) (p : Parameters) (candidate : Bool) :
    (source B D p candidate).toAlgorithm? = .ok (source B D p candidate).toAlgKernel := by
  cases candidate <;> rfl

theorem projection (B D : Nat) (p : Parameters) (candidate : Bool) :
    (kernel B D p candidate).toAlgorithm? = .ok (kernel B D p candidate).toAlgKernel := by
  simp only [kernel, ComputeKernel.eraseDType, source_projection]
  simp [ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?_fromKernelBody]

theorem body (B D : Nat) (p : Parameters) (candidate : Bool) :
    (kernel B D p candidate).toAlgKernel.body = RealProgram.program p.Q p.K p.M p.Out
      p.queryBatch p.queryHead p.queryRow p.queryDim p.keyBatch p.keyHead p.keyRow p.keyDim
      p.outputBatch p.outputHead p.heads p.kvHeads p.queryContext p.roundedContext p.keyContext B D
      p.offset p.size p.scale p.window p.complement p.evenM p.evenN candidate := by
  cases candidate
  · exact RealProgram.original_source p.Q p.K p.M p.Out
      p.queryBatch p.queryHead p.queryRow p.queryDim p.keyBatch p.keyHead p.keyRow p.keyDim
      p.outputBatch p.outputHead p.outputInner p.batches p.heads p.kvHeads p.queryContext p.roundedContext
      p.keyContext B D B p.offset p.size p.scale p.window p.complement p.evenM p.evenN rfl
  · exact RealProgram.candidate_source p.Q p.K p.M p.Out
      p.queryBatch p.queryHead p.queryRow p.queryDim p.keyBatch p.keyHead p.keyRow p.keyDim
      p.outputBatch p.outputHead p.outputInner p.batches p.heads p.kvHeads p.queryContext p.roundedContext
      p.keyContext B D B p.offset p.size p.scale p.window p.complement p.evenM p.evenN rfl

private theorem dynamic_offsets_ok (region : RegionName) (base : Op .nat [])
    (parent shape strides : List Nat) (a b : Op .nat []) :
    (Op.makeBlockPtrDynOffsets region base parent shape strides [a,b]).FlattenOk ↔
      base.FlattenOk ∧ a.FlattenOk ∧ b.FlattenOk := by
  rw [Op.FlattenOk.eq_def]
  simp

theorem flatten_ok (B D : Nat) (p : Parameters) (candidate : Bool) :
    (kernel B D p candidate).toAlgKernel.FlattenOk := by
  rw [Kernel.FlattenOk, body]
  cases candidate
  all_goals
    simp only [RealProgram.program, RealProgram.prelude, RealProgram.postlude,
      RealBody.body, RealBody.loadPrefix, RealBody.loadQuery, RealBody.loadKey, RealBody.suffix,
      RealArithmetic.body, List.cons_append, List.nil_append, StmtList.FlattenOk,
      Stmt.FlattenOk, dynamic_offsets_ok, Bool.false_eq_true, ↓reduceIte]
    simp [RealExpressions.distanceExpr, Op.FlattenOk.eq_def]

theorem safe (bounds : RegionBounds) (s : BlockState) (B D : Nat) (p : Parameters)
    (candidate : Bool) (hB : 0 < B)
    (hb : RealSafety.Bounds bounds (config B D p candidate s) p.roundedContext p.Out (outputBase p (s.pids 1))) :
    (kernel B D p candidate).toAlgKernel.TraceSafe bounds s := by
  rw [Kernel.TraceSafe, body]
  exact RealSafety.correct bounds s p.Q p.K p.M p.Out
    p.queryBatch p.queryHead p.queryRow p.queryDim p.keyBatch p.keyHead p.keyRow p.keyDim
    p.outputBatch p.outputHead p.heads p.kvHeads p.queryContext p.roundedContext p.keyContext B D
    p.offset p.size p.scale p.window p.complement p.evenM p.evenN candidate hB hb

theorem executes (s : BlockState) (B D : Nat) (p : Parameters) (candidate : Bool) (hB : 0 < B) :
    ∃ t, exec (kernel B D p candidate).toAlgKernel s = some t ∧
      (∀ i : Fin B, t.readMem p.Out (outputBase p (s.pids 1)+s.pids 0*B+i.val) =
        if s.pids 0*B+i.val < p.keyContext then (config B D p candidate s).column s (blocks B p) i
        else s.readMem p.Out (outputBase p (s.pids 1)+s.pids 0*B+i.val)) ∧
      (∀ r a, (r ≠ p.Out ∨ ∀ i : Fin B, s.pids 0*B+i.val < p.keyContext →
        a ≠ outputBase p (s.pids 1)+s.pids 0*B+i.val) → t.mem r a = s.mem r a) := by
  simp only [exec, body]
  exact RealExecution.correct s p.Q p.K p.M p.Out
    p.queryBatch p.queryHead p.queryRow p.queryDim p.keyBatch p.keyHead p.keyRow p.keyDim
    p.outputBatch p.outputHead p.heads p.kvHeads p.queryContext p.roundedContext p.keyContext B D
    p.offset p.size p.scale p.window p.complement p.evenM p.evenN candidate hB

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttentionScore.RealSource
