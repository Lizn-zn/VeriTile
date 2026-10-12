import FifthSphDraft.FP
/-! Whole typed-state/failure equivalence. The only change commutes the natural
coordinate-block product. Original and candidate retain all source statements,
fp16/bf16/fp32 input/output choices and independent backward gradient/upstream
types. Input powers retain input precision, named decimal coefficients are fp32
scalar tensors, coefficient expressions promote to fp32, and each store casts
to its destination storage. Backward retains load-then-add of the old gradient.
The input domain and numerical FP atom list are empty. -/
namespace VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.Equivalence
open VeriTile Triton FP.ProfiledRewrite TypedSource
open scoped VeriTile.Spec
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 30000

def assumptions : Spec.Assumptions FP.GuardedFragment := []

def forwardProgram (input output : Storage) (swapped : Bool) (X Y : RegionName) (B N M off stride : Nat) : Program where
  kernel := forward input output swapped X Y B N M off stride
  domain := []

specification forward_equiv (input output : Storage) (X Y : RegionName) (B N M off stride : Nat) :
    forwardProgram input output Bool.false X Y B N M off stride ≡[assumptions]
    forwardProgram input output Bool.true X Y B N M off stride := by
  exact FPProof.forward_equivalent input output X Y B N M off stride

#print_fp_assumptions forward_equiv

def backwardProgram (input gradient upstream : Storage) (swapped : Bool) (X dX g : RegionName) (B N M off stride : Nat) : Program where
  kernel := backward input gradient upstream swapped X dX g B N M off stride
  domain := []

specification backward_equiv (input gradient upstream : Storage) (X dX g : RegionName) (B N M off stride : Nat) :
    backwardProgram input gradient upstream Bool.false X dX g B N M off stride ≡[assumptions]
    backwardProgram input gradient upstream Bool.true X dX g B N M off stride := by
  exact FPProof.backward_equivalent input gradient upstream X dX g B N M off stride

#print_fp_assumptions backward_equiv

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.Equivalence
