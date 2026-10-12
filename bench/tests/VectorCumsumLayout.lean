import bench.tritonbench_g.chunk_cumsum_vector.Correct
import VeriTile.Meta.StatementAudit

/-! The active-address condition admits ordinary contiguous matrices and rejects
zero time stride, which would overwrite earlier chunks. -/
namespace VectorCumsumLayoutTests
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkCumsumVector.Correctness
theorem contiguous_layout (h T S BT BS : Nat) : noOutputOverlap ⟨h,S,1,T,S,BT,BS⟩ := by
  intro feature head t u i j ht hu heq
  dsimp only at ht hu heq
  simp only [Nat.mul_one] at heq
  have rows : t = u := by
    by_contra hn
    rcases lt_or_gt_of_ne hn with hlt | hgt
    · have hx := Nat.mul_le_mul_right S (show t + 1 ≤ u by omega)
      nlinarith
    · have hx := Nat.mul_le_mul_right S (show u + 1 ≤ t by omega)
      nlinarith
  refine ⟨rows, Fin.ext ?_⟩
  subst u
  omega
theorem aliased_time_stride_rejected : ¬ noOutputOverlap ⟨0,0,1,2,1,1,1⟩ := by
  intro h
  have hn := h 0 0 0 1 ⟨0,by decide⟩ ⟨0,by decide⟩ (by decide) (by decide) (by decide)
  omega

#guard_msgs (drop info) in
#auditModuleAxioms
end VectorCumsumLayoutTests
