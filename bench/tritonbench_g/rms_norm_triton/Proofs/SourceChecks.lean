import bench.tritonbench_g.rms_norm_triton.Kernels
namespace VeriTile.Bench.TritonBenchG.RmsNormTriton
open VeriTile.Triton

def Storage.float : Storage → FloatDType
  | .fp32 | .fp64 => .real
  | .fp16 => .fp16
  | .bf16 => .bf16

private def normalizedProduct : Op d sh → Bool
  | .mul _ _ (.ref _ _ x) (.ref _ _ r) => x == "x" && r == "rrms"
  | .withPrecision _ _ e => normalizedProduct e
  | _ => Bool.false

/-- Traverse only the left operand's casts, preserving their order. -/
private def hasPreWeightCast (wanted : FloatDType) : Op d sh → Bool
  | .castFloat .real dst e => (dst == wanted && normalizedProduct e) || hasPreWeightCast wanted e
  | .castFloat _ _ e => hasPreWeightCast wanted e
  | .withPrecision _ _ e => hasPreWeightCast wanted e
  | _ => Bool.false

private def weightRef : Op d sh → Bool
  | .ref _ _ n => n == "w"
  | .castFloat _ _ e => weightRef e
  | .withPrecision _ _ e => weightRef e
  | _ => Bool.false

private def weightedProduct (wanted : FloatDType) : Op d sh → Bool
  | .mul _ _ left right => hasPreWeightCast wanted left && weightRef right
  | _ => Bool.false

private def weightedExpr (wanted : FloatDType) : ComputeExpr d sh → Bool
  | .alg e => weightedProduct wanted e
  | .compute (.alg _ e) => weightedProduct wanted e
  | _ => Bool.false

/-- The source cast to Y's element type is inside the weight multiplication,
for every storage pairing and either address-expression ordering. -/
theorem pre_weight_cast_preserved (storage weight : Storage) (v : Bool)
    (Y X W : RegionName) (ysr ysc xsr xsc N : Nat) (eps : ℝ) (B : Nat) :
    ((typed storage weight v Y X W ysr ysc xsr xsc N eps B).surfaceBody[9]?).any
      (fun st => match st with
        | .assign _ _ name e => name == "y" && weightedExpr storage.float e
        | _ => Bool.false) = Bool.true := by
  cases storage <;> cases weight <;> cases v <;> rfl

theorem statement_count (storage weight : Storage) (v : Bool)
    (Y X W : RegionName) (ysr ysc xsr xsc N : Nat) (eps : ℝ) (B : Nat) :
    (typed storage weight v Y X W ysr ysc xsr xsc N eps B).surfaceBody.length = 11 := by
  cases storage <;> cases weight <;> cases v <;> rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RmsNormTriton
