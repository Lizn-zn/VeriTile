import bench.tritonbench_g.nested_loops_processing.Proofs.FPExecutionMiddle

namespace VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.TypedExecution
open VeriTile.Triton FP.Structural TypedSource
set_option maxHeartbeats 1000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def advances : Nat → PointerTile → Nat → PointerTile
  | 0, p, _ => p
  | n + 1, p, delta => bump (advances n p delta) delta

theorem middleLoop_registers (M : Algebra α) (storage : Storage) (s : State α)
    (a c : PointerTile) (v1 : Payload α storage) (delta : Nat) :
    (middleLoopState M storage s a c v1 delta).regs .ptr [2,2] "a_ptrs" = some (advances 6 a delta) ∧
    (middleLoopState M storage s a c v1 delta).regs .ptr [2,2] "c_ptrs" = some (advances 12 c delta) := by
  simp [middleLoopState, middleState, middlePrefixState, innerLoopState, innerState,
    scatter, advances]

def outerPrefix (storage : Storage) (src dst : RegionName) (sm sn : Int) : List ComputeStmt :=
  (SourceChecks.outer storage Bool.false src dst sm sn).take 1

def outerSuffix (storage : Storage) (src dst : RegionName) (sm sn : Int) : List ComputeStmt :=
  (SourceChecks.outer storage Bool.false src dst sm sn).drop 2

def outerState (M : Algebra α) (storage : Storage) (s : State α)
    (a c : PointerTile) (delta : Nat) : State α :=
  let t := s.setReg "a1" (SourceChecks.dtype storage) [2,2] (loaded M storage s a)
  let t := middleLoopState M storage t a c (loaded M storage s a) delta
  t.setReg "a_ptrs" .ptr [2,2] (advances 7 a delta)

theorem outer_decomposition (storage : Storage) (src dst : RegionName) (sm sn : Int) :
    SourceChecks.outer storage Bool.false src dst sm sn = outerPrefix storage src dst sm sn ++
      [.forRange "j" 0 2 1 (SourceChecks.middle storage Bool.false src dst sm sn)] ++
      outerSuffix storage src dst sm sn := by
  cases storage <;> rfl

theorem outer_prefix_step (M : Algebra α) (storage : Storage)
    (src dst : RegionName) (sm sn : Int) (s : State α) (a : PointerTile)
    (ha : s.regs .ptr [2,2] "a_ptrs" = some a) :
    run M (outerPrefix storage src dst sm sn) s =
      some (s.setReg "a1" (SourceChecks.dtype storage) [2,2] (loaded M storage s a)) := by
  cases storage
  all_goals simp [ComputeDType.eraseDType, Option.bind, outerPrefix, SourceChecks.outer, kernel, ComputeKernel.surfaceBody,
    run, step, evalExpr, evalComputeOp, evalOp_unfold,
    ha, loaded, SourceChecks.dtype]
  all_goals rfl

theorem outer_suffix_step (M : Algebra α) (storage : Storage)
    (src dst : RegionName) (sm sn : Int) (s : State α) (a : PointerTile)
    (ha : s.regs .ptr [2,2] "a_ptrs" = some a) :
    run M (outerSuffix storage src dst sm sn) s =
      some (s.setReg "a_ptrs" .ptr [2,2] (bump a (2 * sn).toNat)) := by
  cases storage
  all_goals simp [ComputeDType.eraseDType, Option.bind, outerSuffix, SourceChecks.outer, kernel, ComputeKernel.surfaceBody,
    run, step, evalExpr, evalComputeOp, evalOp_unfold, numeric, ha, bump]
  all_goals unfold FP.Structural.bop
  all_goals simp [ComputeDType.eraseDType, Option.bind, Broadcast.leftIndex, Broadcast.rightIndex, ha, bump]
  all_goals rfl

theorem outer_step (M : Algebra α) (storage : Storage)
    (src dst : RegionName) (sm sn : Int) (s : State α) (a c : PointerTile)
    (ha : s.regs .ptr [2,2] "a_ptrs" = some a)
    (hc : s.regs .ptr [2,2] "c_ptrs" = some c) :
    run M (SourceChecks.outer storage Bool.false src dst sm sn) s =
      some (outerState M storage s a c (2 * sn).toNat) := by
  have hp := outer_prefix_step M storage src dst sm sn s a ha
  have hl := middle_loop M storage src dst sm sn
    (s.setReg "a1" (SourceChecks.dtype storage) [2,2] (loaded M storage s a))
    a c (loaded M storage s a) (by simpa using ha) (by simpa using hc) (by simp)
  have regs := middleLoop_registers M storage
    (s.setReg "a1" (SourceChecks.dtype storage) [2,2] (loaded M storage s a))
    a c (loaded M storage s a) (2 * sn).toNat
  have hs := outer_suffix_step M storage src dst sm sn _ _ regs.1
  rw [outer_decomposition]
  simp only [run_append, hp, run, step, hl, Option.bind_eq_bind, Option.bind_some]
  exact hs

end VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.TypedExecution

#guard_msgs (drop info) in
#auditModuleAxioms
