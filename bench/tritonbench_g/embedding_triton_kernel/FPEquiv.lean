import bench.tritonbench_g.embedding_triton_kernel.Proofs.FP
/-! Full typed source equivalence, including all loop iterations, signed-ID
comparisons, masked loads/stores and failure. Only the natural-number start_n
product is commuted. No floating-point numerical atom is required. -/
namespace VeriTile.Bench.TritonBenchG.EmbeddingTritonKernel.Equivalence
open VeriTile Triton FP.ProfiledRewrite TypedSource
open scoped VeriTile.Spec

def assumptions : Spec.Assumptions FP.GuardedFragment := []

def original (storage : Storage) (weight ids out : RegionName)
    (first last : Int) (sw so n h D N NN : Nat) : Program where
  kernel := source storage Bool.false weight ids out first last sw so n h D N NN
  domain := []

def indexCommuted (storage : Storage) (weight ids out : RegionName)
    (first last : Int) (sw so n h D N NN : Nat) : Program where
  kernel := source storage Bool.true weight ids out first last sw so n h D N NN
  domain := []

specification embedding_equiv (storage : Storage) (weight ids out : RegionName)
    (first last : Int) (sw so n h D N NN : Nat) :
    original storage weight ids out first last sw so n h D N NN ≡[assumptions]
      indexCommuted storage weight ids out first last sw so n h D N NN := by
  exact FPProof.equivalent storage weight ids out first last sw so n h D N NN

#print_fp_assumptions embedding_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.EmbeddingTritonKernel.Equivalence
