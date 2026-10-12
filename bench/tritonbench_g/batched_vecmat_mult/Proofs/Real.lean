import bench.tritonbench_g.batched_vecmat_mult.Proofs.RealSource
import VeriTile.Triton.Memory.KernelSpec.StreamReal

/-! Both complete sources implement the independent double sum in flat memory.
The original uses the existing real loop invariant and access-safety proof.
The candidate transfers the entire successful execution and frame through
real addition commutativity at every iteration; no FP assumptions are used. -/
namespace VeriTile.Bench.TritonBenchG.BatchedVecmatMult.RealProofs
open VeriTile.Triton
open scoped VeriTile.Triton.StreamMasked2DKernelIO₂
noncomputable section

section
variable (A B out : RegionName) (M N BM BN BK T : Nat)

def io (candidate : Bool) : StreamMasked2DKernelIO₂ :=
  { batchedVecmatIO A B out M N BM BN BK T with
    kernel := RealSource.kernel A B out M N (BK*T) BM BN BK candidate
    projection := RealSource.projection A B out M N (BK*T) BM BN BK candidate }

def expected (xs : Fin T → Fin (BM*BK) → ℝ)
    (ys : Fin T → Fin (BN*BM*BK) → ℝ) (l : Fin (BM*BN)) : ℝ :=
  ∑ t : Fin T, ∑ e : Fin BK,
    xs t (aLane BM BN BK l e) * ys t (bLane BM BN BK l e)

theorem original_correct (hBK : 0 < BK) (hBN : BN ≤ N) :
    io A B out M N BM BN BK T Bool.false ⊨ fun _ _ xs ys l => expected BM BN BK T xs ys l := by
  have h := StreamMasked2DKernelIO₂.Implements.of_identityRounding
    (batchedVecmatIO A B out M N BM BN BK T) rfl _
    (batched_vecmat_io_correctness .triv A B out M N BM BN BK T hBK hBN)
  simpa only [io, StreamMasked2DKernelIO₂.Implements, RealSource.original_flat_source] using h

theorem candidate_correct (hBK : 0 < BK) (hBN : BN ≤ N) :
    io A B out M N BM BN BK T Bool.true ⊨ fun _ _ xs ys l => expected BM BN BK T xs ys l := by
  have h := original_correct A B out M N BM BN BK T hBK hBN
  simpa only [io, StreamMasked2DKernelIO₂.Implements, RealSource.candidate_flat_execution] using h

end
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BatchedVecmatMult.RealProofs
