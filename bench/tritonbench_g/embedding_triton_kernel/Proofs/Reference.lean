import bench.tritonbench_g.embedding_triton_kernel.Proofs.Erasure
import VeriTile.Triton.Semantics.State
namespace VeriTile.Bench.TritonBenchG.EmbeddingTritonKernel.Reference
open VeriTile.Triton
noncomputable section

def row (pid N NN t lane : Nat) : Nat := pid*N+t*NN+lane

def token (u : BlockState) (ids : RegionName) (r : Nat) : Int := u.readMemValue .int ids r

/-- Independent mathematical lookup, before any output stores. Negative IDs
are valid exactly when they lie in the declared signed vocabulary interval. -/
def value (u : BlockState) (weight ids : RegionName) (first last : Int)
    (weightStride r d : Nat) : ℝ :=
  let id := token u ids r
  if first≤id ∧ id<last then u.readMem weight ((id-first).toNat*weightStride+d) else 0

/-- All source store locations, including a possible final overshooting chunk. -/
def Written (pid N NN D n hidden outStride : Nat) (o : Nat) : Prop :=
  ∃ t, t*NN<N ∧ ∃ r, r<NN ∧ ∃ d, d<D ∧
    row pid N NN t r<n ∧ d<hidden ∧ o=row pid N NN t r*outStride+d

/-- The mathematical lookup depends only on the two input regions. -/
theorem value_congr (u v : BlockState) (weight ids : RegionName) (first last : Int)
    (sw r d : Nat) (hw : ∀ o, v.mem weight o=u.mem weight o) (hi : ∀ o, v.mem ids o=u.mem ids o) :
    value v weight ids first last sw r d=value u weight ids first last sw r d := by
  simp [value,token,BlockState.readMemValue,BlockState.readMemTyped,BlockState.readMem,hw,hi]

/-- Row-major output addressing is injective on the columns actually stored.
Padding may exceed the stride because inactive columns do not write. -/
theorem row_major_injective (so D h a b : Nat) (d e : Fin D)
    (hstride : min D h≤so) (hd : d.val<h) (he : e.val<h)
    (heq : a*so+d.val=b*so+e.val) : a=b ∧ d=e := by
  have hdso : d.val<so := lt_of_lt_of_le (lt_min d.isLt hd) hstride
  have heso : e.val<so := lt_of_lt_of_le (lt_min e.isLt he) hstride
  have hs : 0<so := by omega
  have hdiv := congrArg (fun x => x/so) heq
  have hab : a=b := by
    simpa [Nat.mul_comm a so,Nat.mul_comm b so,Nat.mul_add_div hs,
      Nat.div_eq_of_lt hdso,Nat.div_eq_of_lt heso] using hdiv
  refine ⟨hab,Fin.ext ?_⟩
  subst b
  omega

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.EmbeddingTritonKernel.Reference
