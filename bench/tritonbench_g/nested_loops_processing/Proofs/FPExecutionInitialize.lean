import bench.tritonbench_g.nested_loops_processing.Proofs.FPExecutionStep

namespace VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.TypedExecution
open VeriTile.Triton FP.Structural TypedSource
set_option maxHeartbeats 1000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def prefixBody (src dst : RegionName) (sm sn : Int) : List ComputeStmt :=
  (kernel .int32 Bool.false src dst sm sn).surfaceBody.take 6

def initialA (src : RegionName) (sm sn : Int) : PointerTile :=
  fun i ↦ (src, ((i.1.val : Int) * sm + (i.2.1.val : Int) * sn).toNat)

def initialC (dst : RegionName) (sm sn : Int) : PointerTile :=
  fun i ↦ (dst, (sm * (i.1.val : Int) + sn * (i.2.1.val : Int)).toNat)

def prefixState (s : State α) (src dst : RegionName) (sm sn : Int) : State α :=
  let s := s.setReg "offs_am" .nat [2] (fun i ↦ i.1.val)
  let s := s.setReg "offs_an" .nat [2] (fun i ↦ i.1.val)
  let s := s.setReg "a_ptrs" .ptr [2,2] (initialA src sm sn)
  let s := s.setReg "offs_cm" .nat [2] (fun i ↦ i.1.val)
  let s := s.setReg "offs_cn" .nat [2] (fun i ↦ i.1.val)
  s.setReg "c_ptrs" .ptr [2,2] (initialC dst sm sn)

theorem prefix_step (M : Algebra α) (src dst : RegionName) (sm sn : Int) (s : State α) :
    run M (prefixBody src dst sm sn) s = some (prefixState s src dst sm sn) := by
  simp [ComputeDType.eraseDType, Option.bind, prefixBody, kernel, ComputeKernel.surfaceBody, run, step,
    evalExpr, evalComputeOp, evalOp_unfold, numeric, prefixState, initialA, initialC]
  unfold FP.Structural.bop
  simp [ComputeDType.eraseDType, Option.bind, Broadcast.leftIndex, Broadcast.rightIndex, prefixState, initialA, initialC]
  all_goals rfl

theorem source_decomposition (storage : Storage) (src dst : RegionName) (sm sn : Int) :
    (kernel storage Bool.false src dst sm sn).surfaceBody = prefixBody src dst sm sn ++
      [.forRange "i" 0 2 1 (SourceChecks.outer storage Bool.false src dst sm sn)] := by
  cases storage <;> rfl

end VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.TypedExecution

#guard_msgs (drop info) in
#auditModuleAxioms
