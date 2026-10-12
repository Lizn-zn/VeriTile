import bench.tritonbench_g.dropout_triton.Proofs.FP
/-! Complete typed dropout equivalence. Integer block-index commutation
preserves both mask storage variants, all floating arithmetic (including the
eager divide in tl.where), omitted padding, stores, full states and failure.
No numerical atom or operand-domain restriction is used. -/
namespace VeriTile.Bench.TritonBenchG.DropoutTriton.Equivalence
open VeriTile Triton FP.ProfiledRewrite TypedSource
open scoped VeriTile.Spec

def assumptions : Spec.Assumptions FP.GuardedFragment := []

def original (storage : Storage) (keep : KeepStorage) (x k y : RegionName) (n B : Nat) (p : ℝ) : Program where
  kernel := source storage keep Bool.false x k y n B p
  domain := []

def indexCommuted (storage : Storage) (keep : KeepStorage) (x k y : RegionName) (n B : Nat) (p : ℝ) : Program where
  kernel := source storage keep Bool.true x k y n B p
  domain := []

specification dropout_equiv (storage : Storage) (keep : KeepStorage) (x k y : RegionName) (n B : Nat) (p : ℝ) :
    original storage keep x k y n B p ≡[assumptions] indexCommuted storage keep x k y n B p := by
  exact FPProof.equivalent storage keep x k y n B p

#print_fp_assumptions dropout_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.DropoutTriton.Equivalence
