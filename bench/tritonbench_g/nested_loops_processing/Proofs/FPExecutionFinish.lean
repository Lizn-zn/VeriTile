import bench.tritonbench_g.nested_loops_processing.Proofs.FPExecutionOuter
import bench.tritonbench_g.nested_loops_processing.Proofs.FPExecutionInitialize
import bench.tritonbench_g.nested_loops_processing.Proofs.FP

namespace VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.TypedExecution
open VeriTile.Triton FP.Structural TypedSource
set_option maxHeartbeats 1000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem outer_registers (M : Algebra α) (storage : Storage) (s : State α)
    (a c : PointerTile) (delta : Nat) :
    (outerState M storage s a c delta).regs .ptr [2,2] "a_ptrs" = some (advances 7 a delta) ∧
    (outerState M storage s a c delta).regs .ptr [2,2] "c_ptrs" = some (advances 12 c delta) := by
  have h := middleLoop_registers M storage
    (s.setReg "a1" (SourceChecks.dtype storage) [2,2] (loaded M storage s a)) a c
    (loaded M storage s a) delta
  constructor
  · simp [outerState]
  · simpa [outerState] using h.2

def outerLoopState (M : Algebra α) (storage : Storage) (s : State α)
    (a c : PointerTile) (delta : Nat) : State α :=
  let t := outerState M storage (s.setReg "i" .nat [] (fun _ ↦ 0)) a c delta
  outerState M storage (t.setReg "i" .nat [] (fun _ ↦ 1))
    (advances 7 a delta) (advances 12 c delta) delta

theorem outer_loop (M : Algebra α) (storage : Storage)
    (src dst : RegionName) (sm sn : Int) (s : State α) (a c : PointerTile)
    (ha : s.regs .ptr [2,2] "a_ptrs" = some a)
    (hc : s.regs .ptr [2,2] "c_ptrs" = some c) :
    range M "i" 0 2 1 (SourceChecks.outer storage Bool.false src dst sm sn) s =
      some (outerLoopState M storage s a c (2 * sn).toNat) := by
  rw [range_two]
  have e0 := outer_step M storage src dst sm sn (s.setReg "i" .nat [] (fun _ ↦ 0)) a c
    (by simpa using ha) (by simpa using hc)
  rw [e0]
  have regs := outer_registers M storage (s.setReg "i" .nat [] (fun _ ↦ 0)) a c (2 * sn).toNat
  exact outer_step M storage src dst sm sn _ _ _ (by simpa using regs.1) (by simpa using regs.2)

def finalState (M : Algebra α) (storage : Storage) (s : State α)
    (src dst : RegionName) (sm sn : Int) : State α :=
  outerLoopState M storage (prefixState s src dst sm sn) (initialA src sm sn) (initialC dst sm sn)
    (2 * sn).toNat

/-- Complete successful typed execution for every payload profile and backend.
All actual loads, cached values, stores, and nested loop indices remain in the state. -/
theorem source_run (M : Algebra α) (storage : Storage)
    (src dst : RegionName) (sm sn : Int) (s : State α) :
    exec M (kernel storage Bool.false src dst sm sn) s =
      some (finalState M storage s src dst sm sn) := by
  change run M (kernel storage Bool.false src dst sm sn).surfaceBody s = _
  rw [source_decomposition, run_append, prefix_step]
  have h := outer_loop M storage src dst sm sn (prefixState s src dst sm sn)
    (initialA src sm sn) (initialC dst sm sn)
    (by simp [prefixState]) (by simp [prefixState])
  simpa only [run, step, h, finalState, Option.bind_eq_bind, Option.bind_some] using h

theorem candidate_run (M : Algebra α) (storage : Storage)
    (src dst : RegionName) (sm sn : Int) (s : State α) :
    exec M (kernel storage Bool.true src dst sm sn) s =
      some (finalState M storage s src dst sm sn) := by
  rw [← FullFP.source_equivalence M storage src dst sm sn s]
  exact source_run M storage src dst sm sn s

end VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.TypedExecution

#guard_msgs (drop info) in
#auditModuleAxioms
