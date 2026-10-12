import bench.tritonbench_g.fast_rope_embedding.Proofs.FP
/-! Complete typed-state/failure equivalence for both rotation directions and
all 27 input/cosine/sine storage triples. Q loads convert to sine storage,
products follow their source precision, and stores convert to Q storage.
The candidate only commutes the natural-number head-group product. The
input domain and list of numerical atoms are both empty. -/
namespace VeriTile.Bench.TritonBenchG.FastRopeEmbedding.Equivalence
open VeriTile Triton FP.ProfiledRewrite TypedSource
open scoped VeriTile.Spec

def assumptions : Spec.Assumptions FP.GuardedFragment := []

def program (input cosine sine : Storage) (swapped : Bool) (Q C S : RegionName)
    (QR CR SR seq H heads B : Nat) (backwards : Bool) : Program where
  kernel := rope input cosine sine swapped Q C S QR CR SR seq H heads B backwards
  domain := []

specification rope_equiv (input cosine sine : Storage) (Q C S : RegionName)
    (QR CR SR seq H heads B : Nat) (backwards : Bool) :
    program input cosine sine Bool.false Q C S QR CR SR seq H heads B backwards ≡[assumptions]
    program input cosine sine Bool.true Q C S QR CR SR seq H heads B backwards := by
  exact FPProof.equivalent input cosine sine Q C S QR CR SR seq H heads B backwards

#print_fp_assumptions rope_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.FastRopeEmbedding.Equivalence
