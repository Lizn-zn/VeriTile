import bench.tritonbench_g.nested_loops_processing.Proofs.FPExecutionLoop

namespace VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.TypedExecution
open VeriTile.Triton FP.Structural TypedSource
set_option maxHeartbeats 1000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def middlePrefix (storage : Storage) (src dst : RegionName) (sm sn : Int) : List ComputeStmt :=
  (SourceChecks.middle storage Bool.false src dst sm sn).take 2

def middlePrefixState (M : Algebra α) (storage : Storage)
    (s : State α) (a : PointerTile) (delta : Nat) : State α :=
  let t := s.setReg "a_ptrs" .ptr [2,2] (bump a delta)
  t.setReg "a2" (SourceChecks.dtype storage) [2,2] (loaded M storage s (bump a delta))

def middleState (M : Algebra α) (storage : Storage) (s : State α)
    (a c : PointerTile) (v1 : Payload α storage) (delta : Nat) : State α :=
  innerLoopState M storage (middlePrefixState M storage s a delta) (bump a delta) c v1
    (loaded M storage s (bump a delta)) delta

theorem middle_decomposition (storage : Storage) (src dst : RegionName) (sm sn : Int) :
    SourceChecks.middle storage Bool.false src dst sm sn = middlePrefix storage src dst sm sn ++
      [.forRange "k" 0 2 1 (SourceChecks.inner storage Bool.false src dst sm sn)] := by
  cases storage <;> rfl

theorem middle_prefix_step (M : Algebra α) (storage : Storage)
    (src dst : RegionName) (sm sn : Int) (s : State α) (a : PointerTile)
    (ha : s.regs .ptr [2,2] "a_ptrs" = some a) :
    run M (middlePrefix storage src dst sm sn) s =
      some (middlePrefixState M storage s a (2 * sn).toNat) := by
  cases storage
  all_goals simp only [middlePrefix, SourceChecks.middle, SourceChecks.outer,
    kernel, ComputeKernel.surfaceBody, List.getElem?_cons_zero, List.getElem?_cons_succ,
    List.take_succ_cons, List.take_zero, run, step, evalExpr, evalComputeOp]
  all_goals simp [ComputeDType.eraseDType, Option.bind, evalOp_unfold, numeric, SourceChecks.dtype, ha,
    middlePrefixState, loaded, bump]
  all_goals unfold FP.Structural.bop
  all_goals simp [ComputeDType.eraseDType, Option.bind, Broadcast.leftIndex, Broadcast.rightIndex, ha,
    middlePrefixState, loaded, bump, SourceChecks.dtype]
  all_goals rfl

theorem middle_step (M : Algebra α) (storage : Storage)
    (src dst : RegionName) (sm sn : Int) (s : State α) (a c : PointerTile)
    (v1 : Payload α storage)
    (ha : s.regs .ptr [2,2] "a_ptrs" = some a)
    (hc : s.regs .ptr [2,2] "c_ptrs" = some c)
    (h1 : s.regs (SourceChecks.dtype storage) [2,2] "a1" = some v1) :
    run M (SourceChecks.middle storage Bool.false src dst sm sn) s =
      some (middleState M storage s a c v1 (2 * sn).toNat) := by
  rw [middle_decomposition, run_append, middle_prefix_step M storage src dst sm sn s a ha]
  have h := inner_loop M storage src dst sm sn (middlePrefixState M storage s a (2 * sn).toNat)
    (bump a (2 * sn).toNat) c v1 (loaded M storage s (bump a (2 * sn).toNat))
    (by simp [middlePrefixState]) (by simpa [middlePrefixState] using hc)
    (by simpa [middlePrefixState] using h1) (by simp [middlePrefixState])
  simpa only [run, step, h, middleState, Option.bind_eq_bind, Option.bind_some] using h

theorem middle_registers (M : Algebra α) (storage : Storage) (s : State α)
    (a c : PointerTile) (v1 : Payload α storage) (delta : Nat)
    (h1 : s.regs (SourceChecks.dtype storage) [2,2] "a1" = some v1) :
    (middleState M storage s a c v1 delta).regs .ptr [2,2] "a_ptrs" =
      some (bump (bump (bump a delta) delta) delta) ∧
    (middleState M storage s a c v1 delta).regs .ptr [2,2] "c_ptrs" =
      some (bump (bump (bump (bump (bump (bump c delta) delta) delta) delta) delta) delta) ∧
    (middleState M storage s a c v1 delta).regs (SourceChecks.dtype storage) [2,2] "a1" = some v1 := by
  simp [middleState, middlePrefixState, innerLoopState, innerState, scatter, h1]

def middleLoopState (M : Algebra α) (storage : Storage) (s : State α)
    (a c : PointerTile) (v1 : Payload α storage) (delta : Nat) : State α :=
  let t := middleState M storage (s.setReg "j" .nat [] (fun _ ↦ 0)) a c v1 delta
  middleState M storage (t.setReg "j" .nat [] (fun _ ↦ 1))
    (bump (bump (bump a delta) delta) delta)
    (bump (bump (bump (bump (bump (bump c delta) delta) delta) delta) delta) delta) v1 delta

theorem middle_loop (M : Algebra α) (storage : Storage)
    (src dst : RegionName) (sm sn : Int) (s : State α) (a c : PointerTile)
    (v1 : Payload α storage)
    (ha : s.regs .ptr [2,2] "a_ptrs" = some a)
    (hc : s.regs .ptr [2,2] "c_ptrs" = some c)
    (h1 : s.regs (SourceChecks.dtype storage) [2,2] "a1" = some v1) :
    range M "j" 0 2 1 (SourceChecks.middle storage Bool.false src dst sm sn) s =
      some (middleLoopState M storage s a c v1 (2 * sn).toNat) := by
  rw [range_two]
  have e0 := middle_step M storage src dst sm sn (s.setReg "j" .nat [] (fun _ ↦ 0)) a c v1
    (by simpa using ha) (by simpa using hc) (by simpa using h1)
  rw [e0]
  have regs := middle_registers M storage (s.setReg "j" .nat [] (fun _ ↦ 0)) a c v1
    (2 * sn).toNat (by simpa using h1)
  exact middle_step M storage src dst sm sn _ _ _ _
    (by simpa using regs.1) (by simpa using regs.2.1) (by simpa using regs.2.2)

end VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.TypedExecution

#guard_msgs (drop info) in
#auditModuleAxioms
