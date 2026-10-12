import bench.tritonbench_g.attention_fwd_triton1.Proofs.RealCandidate

/-! A nonzero two-chunk execution witness. Unlike a zero-dot consistency
model, this real execution distinguishes the old and new carry: O is [1,2]
and STORE writes H=[0,1], not [1,2]. The general proof has symbolic sizes. -/
namespace VeriTile.Bench.TritonBenchG.AttentionFwdTriton1.RealExecution
open VeriTile.Triton
set_option linter.unusedSimpArgs false

def initial : BlockState where
  mem := fun r _ => .real (if r = "q" ∨ r = "k" ∨ r = "v" then 1 else 17)
  regs := fun _ _ _ => none
  pids := fun _ => 0

@[simp] private theorem read_query (off : Nat) : initial.readMem "q" off = 1 := rfl
@[simp] private theorem read_key (off : Nat) : initial.readMem "k" off = 1 := rfl
@[simp] private theorem read_value (off : Nat) : initial.readMem "v" off = 1 := rfl

theorem nonzero_all_flags (STORE IFCOND : Bool) :
    ∃ st,
      exec (TypedSource.original "q" "k" "v" "h" "o" 2 1 1 2 1 2 1
        1 1 2 STORE IFCOND).eraseDType.toAlgKernel initial = some st ∧
      exec (TypedSource.scalarLeft "q" "k" "v" "h" "o" 2 1 1 2 1 2 1
        1 1 2 STORE IFCOND).eraseDType.toAlgKernel initial = some st ∧
      st.readMem "o" 0 = 1 ∧ st.readMem "o" 1 = 2 ∧
      st.readMem "h" 0 = (if STORE then 0 else 17) ∧
      st.readMem "h" 1 = (if STORE then 1 else 17) ∧
      st.readMem "o" 2 = 17 ∧ st.readMem "q" 0 = 1 := by
  obtain ⟨st, hrun, h⟩ := RealLoop.original_executes "q" "k" "v" "h" "o"
    2 2 1 1 1 1 2 STORE IFCOND initial (by decide) (by decide)
    (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)
  refine ⟨st, hrun, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · exact (RealCandidate.same_execution "q" "k" "v" "h" "o" 2 2 1 1
      1 1 2 STORE IFCOND initial).trans hrun
  · have ho := h.output 0 (by decide) 0 0
    change st.readMem "o" 0 = _ at ho
    simpa [aft1OutG, aft1LocalOutG, aft1RecOutG, aft1RecStateG,
      aft1QCellG, aft1KCellG, aft1VCellG] using ho
  · have ho := h.output 1 (by decide) 0 0
    change st.readMem "o" 1 = _ at ho
    norm_num [aft1OutG, aft1LocalOutG, aft1RecOutG, aft1RecStateG,
      aft1QCellG, aft1KCellG, aft1VCellG] at ho
    exact ho
  · cases STORE with
    | «false» =>
      have hf := h.frame "h" 0 (Or.inl (by decide)) (Or.inl rfl)
      simp only [BlockState.readMem, hf]
      rfl
    | «true» =>
      simpa [initial, aft1RecStateG] using h.snapshots rfl 0 (by decide) 0 0
  · cases STORE with
    | «false» =>
      have hf := h.frame "h" 1 (Or.inl (by decide)) (Or.inl rfl)
      simp only [BlockState.readMem, hf]
      rfl
    | «true» =>
      have hh := h.snapshots rfl 1 (by decide) 0 0
      change st.readMem "h" 1 = _ at hh
      simpa [aft1RecStateG, aft1KCellG, aft1VCellG] using hh
  · have hf := h.frame "o" 2 (Or.inr (by
        intro c hc i j
        have hi : i.val = 0 := by omega
        have hj : j.val = 0 := by omega
        simp only [initial, hi, hj, Nat.zero_mul, Nat.mul_one, Nat.add_zero, Nat.zero_add]
        omega)) (Or.inr (Or.inl (by decide)))
    simp only [BlockState.readMem, hf]
    rfl
  · rw [h.query]
    rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionFwdTriton1.RealExecution
