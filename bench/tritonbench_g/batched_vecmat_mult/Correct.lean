import bench.tritonbench_g.batched_vecmat_mult.Kernels
import bench.tritonbench_g.batched_vecmat_mult.Proofs.Real
import VeriTile.Meta.StatementAudit

/-! Both complete kernels implement the real batched GEMV formula
out[m,n] = ∑k A[m,k]*B[m,n,k]. The source has unmasked loads and stores;
K is a whole number of key blocks. The IO requires disjoint allocations,
bounds for every displayed input/output address, matching input values and
zero undef. BK>0 resolves the dynamic loop count; BN≤N separates output lanes.
There is no scratch allocation, precomputed accumulator or output equation.
Dimensions remain symbolic. Host launch composition and fixed-width address
overflow are outside this per-program mathematical contract. -/
namespace VeriTile.Bench.TritonBenchG.BatchedVecmatMult.Correctness
open VeriTile.Triton
open scoped VeriTile.Triton.StreamMasked2DKernelIO₂
noncomputable section

/-- At iteration t, read A[m,t*BK+e] and B[m,n,t*BK+e], with source B tile
order [BN,BM,BK]; store output[m,n]. All these accesses are unmasked. M is
source-unused; the actual windows and allocation bounds determine validity. -/
def originalIO (A B out : RegionName) (M N BM BN BK T : Nat) : StreamMasked2DKernelIO₂ where
  kernel := (TypedSource.original A B out M N (BK*T) BM BN BK).eraseDType
  projection := RealSource.projection A B out M N (BK*T) BM BN BK Bool.false
  inp1 := A
  inp2 := B
  out := out
  T := T
  B1 := BM*BK
  B2 := BN*BM*BK
  C := BM*BN
  outDType := .real
  read1 := fun pm _ t l => (pm*BM+l.val/BK)*(BK*T) + (t.val*BK+l.val%BK)
  read2 := fun pm pn t l => (pm*BM+l.val/BK%BM)*N*(BK*T) +
    (pn*BN+l.val/BK/BM)*(BK*T) + (t.val*BK+l.val%BK)
  write := fun pm pn l => (pm*BM+l.val/BN)*N + (pn*BN+l.val%BN)
  mask1 := fun _ _ _ _ => True
  mask2 := fun _ _ _ _ => True
  writeMask := fun _ _ _ => True

def blockFirstIO (A B out : RegionName) (M N BM BN BK T : Nat) : StreamMasked2DKernelIO₂ :=
  { originalIO A B out M N BM BN BK T with
    kernel := (TypedSource.blockFirst A B out M N (BK*T) BM BN BK).eraseDType
    projection := RealSource.projection A B out M N (BK*T) BM BN BK Bool.true }

/-- For output lane (i,j), A contributes (i,e) and B contributes (j,i,e). -/
def inputALane (BM BN BK : Nat) (l : Fin (BM*BN)) (e : Fin BK) : Fin (BM*BK) :=
  Lane2D.encode ((Lane2D.decode l).1, e, PUnit.unit)

def inputBLane (BM BN BK : Nat) (l : Fin (BM*BN)) (e : Fin BK) : Fin (BN*BM*BK) :=
  Lane2D.encode (Lane2D.encode ((Lane2D.decode l).2.1, (Lane2D.decode l).1, PUnit.unit), e, PUnit.unit)

/-- Independent mathematical sum over all K=T*BK input elements. -/
def expected (BM BN BK T : Nat) (xs : Fin T → Fin (BM*BK) → ℝ)
    (ys : Fin T → Fin (BN*BM*BK) → ℝ) (l : Fin (BM*BN)) : ℝ :=
  ∑ t : Fin T, ∑ e : Fin BK,
    xs t (inputALane BM BN BK l e) * ys t (inputBLane BM BN BK l e)

specification vecmat_correctness (A B out : RegionName) (M N BM BN BK T : Nat)
    (hBK : 0 < BK) (hBN : BN ≤ N) :
    Spec.Real (originalIO A B out M N BM BN BK T ⊨ fun _ _ xs ys l => expected BM BN BK T xs ys l) := by
  exact RealProofs.original_correct A B out M N BM BN BK T hBK hBN

specification vecmat_block_first_correctness (A B out : RegionName) (M N BM BN BK T : Nat)
    (hBK : 0 < BK) (hBN : BN ≤ N) :
    Spec.Real (blockFirstIO A B out M N BM BN BK T ⊨ fun _ _ xs ys l => expected BM BN BK T xs ys l) := by
  exact RealProofs.candidate_correct A B out M N BM BN BK T hBK hBN

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BatchedVecmatMult.Correctness
