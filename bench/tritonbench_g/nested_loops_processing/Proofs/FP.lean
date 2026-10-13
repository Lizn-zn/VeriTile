import bench.tritonbench_g.nested_loops_processing.Kernels
import VeriTile.Triton.Float.ProfiledSoundness

namespace VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.FullFP
open VeriTile.Triton FP.Structural TypedSource
set_option maxHeartbeats 1000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false

def initialPointer (v : Bool) (src dst : RegionName) (sm sn : Int) : ComputeStmt :=
  ((kernel .int32 v src dst sm sn).surfaceBody[2]?).getD (.forRange "" 0 0 1 [])

theorem pointer_step {α : Type} [Inhabited α] (M : Algebra α)
    (src dst : RegionName) (sm sn : Int) (s : State α) :
    step M (initialPointer Bool.false src dst sm sn) s =
      step M (initialPointer Bool.true src dst sm sn) s := by
  simp only [initialPointer, kernel, ComputeKernel.surfaceBody, 
    List.getElem?_cons_succ, List.getElem?_cons_zero, Option.getD_some]
  simp [step, evalExpr, evalComputeOp, evalOp_unfold, numeric]
  unfold FP.Structural.bop
  simp only [Broadcast.leftIndex, Broadcast.rightIndex, Int.mul_comm]

theorem source_decomposition (storage : Storage) (v : Bool)
    (src dst : RegionName) (sm sn : Int) :
    (kernel storage v src dst sm sn).surfaceBody =
      (kernel .int32 Bool.false src dst sm sn).surfaceBody.take 2 ++
      [initialPointer v src dst sm sn] ++
      (kernel storage Bool.false src dst sm sn).surfaceBody.drop 3 := by
  cases storage <;> cases v <;> rfl

/-- Complete optional-state equality includes every nested iteration, alias, and failure. -/
theorem source_equivalence {α : Type} [Inhabited α] (M : Algebra α)
    (storage : Storage) (src dst : RegionName) (sm sn : Int) (s : State α) :
    exec M (kernel storage Bool.false src dst sm sn) s =
      exec M (kernel storage Bool.true src dst sm sn) s := by
  change run M (kernel storage Bool.false src dst sm sn).surfaceBody s =
    run M (kernel storage Bool.true src dst sm sn).surfaceBody s
  conv_lhs => rw [source_decomposition storage Bool.false]
  conv_rhs => rw [source_decomposition storage Bool.true]
  simp only [run_append, run, pointer_step]

end VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.FullFP

#guard_msgs (drop info) in
#auditModuleAxioms
