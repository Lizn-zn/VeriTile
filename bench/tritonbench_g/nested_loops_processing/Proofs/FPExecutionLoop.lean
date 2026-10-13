import bench.tritonbench_g.nested_loops_processing.Proofs.FPExecutionStep

namespace VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.TypedExecution
open VeriTile.Triton FP.Structural TypedSource
set_option maxHeartbeats 1000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

 theorem range_two (M : Algebra α) (idx : RegName) (body : List ComputeStmt) (s : State α) :
    range M idx 0 2 1 body s =
      (run M body (s.setReg idx .nat [] (fun _ ↦ 0)) >>= fun t ↦
        run M body (t.setReg idx .nat [] (fun _ ↦ 1))) := by
  rw [range]
  simp only [Nat.one_ne_zero, Nat.reduceLT, ↓reduceIte]
  cases h0 : run M body (s.setReg idx .nat [] (fun _ ↦ 0)) with
  | none => rfl
  | some t =>
    change range M idx 1 2 1 body t = run M body (t.setReg idx .nat [] (fun _ ↦ 1))
    rw [range]
    simp only [Nat.one_ne_zero, Nat.reduceLT, ↓reduceIte]
    cases h1 : run M body (t.setReg idx .nat [] (fun _ ↦ 1)) with
    | none => simp [h1]
    | some u => simp [range, h1]

theorem inner_registers (M : Algebra α) (storage : Storage) (s : State α)
    (a c : PointerTile) (v1 v2 : Payload α storage) (delta : Nat)
    (h1 : s.regs (SourceChecks.dtype storage) [2,2] "a1" = some v1)
    (h2 : s.regs (SourceChecks.dtype storage) [2,2] "a2" = some v2) :
    (innerState M storage s a c v1 v2 delta).regs .ptr [2,2] "a_ptrs" = some (bump a delta) ∧
    (innerState M storage s a c v1 v2 delta).regs .ptr [2,2] "c_ptrs" =
      some (bump (bump (bump c delta) delta) delta) ∧
    (innerState M storage s a c v1 v2 delta).regs (SourceChecks.dtype storage) [2,2] "a1" = some v1 ∧
    (innerState M storage s a c v1 v2 delta).regs (SourceChecks.dtype storage) [2,2] "a2" = some v2 := by
  simp [innerState, scatter, h1, h2]

def innerLoopState (M : Algebra α) (storage : Storage) (s : State α)
    (a c : PointerTile) (v1 v2 : Payload α storage) (delta : Nat) : State α :=
  let t := innerState M storage (s.setReg "k" .nat [] (fun _ ↦ 0)) a c v1 v2 delta
  innerState M storage (t.setReg "k" .nat [] (fun _ ↦ 1))
    (bump a delta) (bump (bump (bump c delta) delta) delta) v1 v2 delta

theorem inner_loop (M : Algebra α) (storage : Storage) (src dst : RegionName) (sm sn : Int)
    (s : State α) (a c : PointerTile) (v1 v2 : Payload α storage)
    (ha : s.regs .ptr [2,2] "a_ptrs" = some a)
    (hc : s.regs .ptr [2,2] "c_ptrs" = some c)
    (h1 : s.regs (SourceChecks.dtype storage) [2,2] "a1" = some v1)
    (h2 : s.regs (SourceChecks.dtype storage) [2,2] "a2" = some v2) :
    range M "k" 0 2 1 (SourceChecks.inner storage Bool.false src dst sm sn) s =
      some (innerLoopState M storage s a c v1 v2 (2 * sn).toNat) := by
  rw [range_two]
  have e0 := inner_step M storage src dst sm sn (s.setReg "k" .nat [] (fun _ ↦ 0)) a c v1 v2
    (by simpa using ha) (by simpa using hc) (by simpa using h1) (by simpa using h2)
  rw [e0]
  have regs := inner_registers M storage (s.setReg "k" .nat [] (fun _ ↦ 0)) a c v1 v2
    (2 * sn).toNat (by simpa using h1) (by simpa using h2)
  exact inner_step M storage src dst sm sn _ _ _ _ _
    (by simpa using regs.1) (by simpa using regs.2.1)
    (by simpa using regs.2.2.1) (by simpa using regs.2.2.2)

end VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.TypedExecution

#guard_msgs (drop info) in
#auditModuleAxioms
