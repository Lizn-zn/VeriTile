import bench.tritonbench_g.cosine_compute.Proofs.Real
namespace VeriTile.Bench.TritonBenchG.CosineCompute.RealWitness
open VeriTile.Triton RealProof
noncomputable section
set_option maxHeartbeats 1000000

def initial : BlockState where
  mem := fun _ _ => .real 0
  regs := fun _ _ _ => none
  pids := fun _ => 1
  numPids := fun _ => 2
  undef := fun _ _ => 0

def allocation : FlatAlloc where
  flat := "flat"
  regions := ["a","b"]
  base := fun r => if r = "a" then 0 else 8
  extent := fun r => if r = "a" ∨ r = "b" then 3 else 0

theorem disjoint : allocation.Disjoint := by
  intro r hr s hs hne
  simp only [allocation,List.mem_cons,List.not_mem_nil,or_false] at hr hs
  rcases hr with rfl | rfl <;> rcases hs with rfl | rfl
  all_goals first | exact False.elim (hne rfl) | decide

theorem coverage : ∀ r, r ∉ allocation.regions → allocation.extent r = 0 := by
  intro r hr
  simpa [allocation] using hr

theorem input_bounds (j : Fin 2) (h : initial.pid*2+j.val < 3) :
    initial.pid*2+j.val < allocation.extent "a" := by simpa [allocation] using h
theorem output_bounds (j : Fin 2) (h : initial.pid*2+j.val < 3) :
    initial.pid*2+j.val < allocation.extent "b" := by simpa [allocation] using h

/-- A nonempty partial block with finite allocation, empty registers and an
actual active write. Both versions and every storage branch execute. -/
theorem executes (storage : Storage) (swapped : Bool) :
    ∃ final, exec (allocation.flattenKernel
      (kernel storage swapped "a" "b" 3 2).eraseDType.toAlgKernel)
      (allocation.flattenState initial) = some final := by
  obtain ⟨u,hu,_,_⟩ := cos_func_region_run "a" "b" 3 2 initial (fun _ => 0) (by intros; rfl)
  have he : exec (realKernel swapped "a" "b" 3 2).toAlgKernel initial = some u := by
    cases swapped
    · exact hu
    · rw [commuted_exec]; exact hu
  have ht : Kernel.TraceSafe allocation.extent
      (realKernel swapped "a" "b" 3 2).toAlgKernel initial := by
    cases swapped
    · exact cos_func_traceSafe _ _ _ _ _ _ input_bounds output_bounds
    · exact commuted_traceSafe _ _ _ _ _ _ input_bounds output_bounds
  have hok : (realKernel swapped "a" "b" 3 2).toAlgKernel.FlattenOk := by
    cases swapped
    · exact cos_func_flattenOk _ _ _ _
    · exact commuted_flattenOk _ _ _ _
  have hf := allocation.exec_flatten disjoint coverage _ initial ht hok rfl
  rw [he,Option.map_some] at hf
  refine ⟨allocation.flattenState u,?_⟩
  rw [projects]
  exact hf

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.CosineCompute.RealWitness
