import bench.tritonbench_g.fast_ce_loss.Proofs.FP
/-! Whole typed equivalence for the three original JITs and integer row-index
commutation, preserving storage casts, signed labels, libdevice backend,
constexpr branches, complete states and failure. Half/bfloat scaling precedes
fp32 conversion; reductions and loss/LSE outputs use fp32, and the backward
store converts to the input storage type. The libdevice tanh backend stays
distinct from the intrinsic tanh. No numerical FP atom is used. -/
namespace VeriTile.Bench.TritonBenchG.FastCeLoss.Equivalence
open VeriTile Triton FP.ProfiledRewrite TypedSource
open scoped VeriTile.Spec

def assumptions : Spec.Assumptions FP.GuardedFragment := []

def forwardProgram (storage : Storage) (cap scale swapped : Bool)
    (X Y Z : RegionName) (L : Region .int) (V R B : Nat) (SC SL : ℝ) : Program where
  kernel := forward storage cap scale swapped X Y Z L V R B SC SL
  domain := []

specification forward_equiv (storage : Storage) (cap scale : Bool)
    (X Y Z : RegionName) (L : Region .int) (V R B : Nat) (SC SL : ℝ) :
    forwardProgram storage cap scale Bool.false X Y Z L V R B SC SL ≡[assumptions]
    forwardProgram storage cap scale Bool.true X Y Z L V R B SC SL := by
  exact FPProof.forward_equivalent storage cap scale X Y Z L V R B SC SL

def chunkedProgram (storage : Storage) (cap scale swapped : Bool)
    (X Y Z : RegionName) (L : Region .int) (V C R B : Nat) (SC SL : ℝ) : Program where
  kernel := chunked storage cap scale swapped X Y Z L V C R B SC SL
  domain := []

specification chunked_equiv (storage : Storage) (cap scale : Bool)
    (X Y Z : RegionName) (L : Region .int) (V C R B : Nat) (SC SL : ℝ) :
    chunkedProgram storage cap scale Bool.false X Y Z L V C R B SC SL ≡[assumptions]
    chunkedProgram storage cap scale Bool.true X Y Z L V C R B SC SL := by
  exact FPProof.chunked_equivalent storage cap scale X Y Z L V R B SC SL C

def backwardProgram (storage : Storage) (cap scale swapped : Bool)
    (X Y Z : RegionName) (L : Region .int) (V R DR B : Nat) (SC SL : ℝ) : Program where
  kernel := backward storage cap scale swapped X Y Z L V R DR B SC SL
  domain := []

specification backward_equiv (storage : Storage) (cap scale : Bool)
    (X Y Z : RegionName) (L : Region .int) (V R DR B : Nat) (SC SL : ℝ) :
    backwardProgram storage cap scale Bool.false X Y Z L V R DR B SC SL ≡[assumptions]
    backwardProgram storage cap scale Bool.true X Y Z L V R DR B SC SL := by
  exact FPProof.backward_equivalent storage cap scale X Y Z L V R B SC SL DR
#print_fp_assumptions forward_equiv
#print_fp_assumptions chunked_equiv
#print_fp_assumptions backward_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.FastCeLoss.Equivalence
