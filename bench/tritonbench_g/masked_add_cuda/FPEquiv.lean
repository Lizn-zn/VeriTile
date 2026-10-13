import bench.tritonbench_g.masked_add_cuda.Kernels
import VeriTile.Triton.Float.RewriteTactics
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.MaskedAddCuda.FPEquiv
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open scoped VeriTile.Spec

def additionSite (storage : MaskStorage) (grad p mask : RegionName)
    (n : Nat) (alpha : ℝ) (B : Nat) : Site :=
  .addFP32 (Source.prefixBefore
    (integer_kernel storage Bool.false grad p mask n alpha B).surfaceBody "grad" 1) [B]
    (.ref .real [B] "grad")
    (.mul .real .scalarR (.ref .real [B] "p") (.castFloat .real .real (.const alpha)))

def original (storage : MaskStorage) (grad p mask : RegionName)
    (n : Nat) (alpha : ℝ) (B : Nat) : Program where
  kernel := integer_kernel storage Bool.false grad p mask n alpha B
  domain := [additionSite storage grad p mask n alpha B]

def commuted (storage : MaskStorage) (grad p mask : RegionName)
    (n : Nat) (alpha : ℝ) (B : Nat) : Program :=
  { original storage grad p mask n alpha B with
    kernel := integer_kernel storage Bool.true grad p mask n alpha B }

specification masked_add_commuted_equiv (storage : MaskStorage) (grad p mask : RegionName)
    (n : Nat) (alpha : ℝ) (B : Nat) (R : FP.ScalarArithmetic.Rules) :
    original storage grad p mask n alpha B ≡[R] commuted storage grad p mask n alpha B := by
  cases storage <;> equiv_decompose
  all_goals fp_prove

def storedOperand (storage : FloatStorage) (name : RegName) (B : Nat) : Op .real [B] :=
  match storage with
  | .fp32 => .ref .real [B] name
  | .fp16 => .castFloat .fp16 .real (.ref .fp16 [B] name)
  | .bf16 => .castFloat .bf16 .real (.ref .bf16 [B] name)

def typedAdditionSite (gt pt : FloatStorage) (mt : MaskStorage) (g p m : RegionName)
    (n : Nat) (a : ℝ) (B : Nat) : Site :=
  .addFP32 (Source.prefixBefore
    (typed_integer_kernel gt pt mt Bool.false g p m n a B).surfaceBody "grad" 1) [B]
    (storedOperand gt "grad" B)
    (.mul .real .scalarR (storedOperand pt "p" B) (.castFloat .real .real (.const a)))

def typedOriginal (gt pt : FloatStorage) (mt : MaskStorage) (g p m : RegionName)
    (n : Nat) (a : ℝ) (B : Nat) : Program where
  kernel := typed_integer_kernel gt pt mt Bool.false g p m n a B
  domain := [typedAdditionSite gt pt mt g p m n a B]

def typedCommuted (gt pt : FloatStorage) (mt : MaskStorage) (g p m : RegionName)
    (n : Nat) (a : ℝ) (B : Nat) : Program :=
  { typedOriginal gt pt mt g p m n a B with
    kernel := typed_integer_kernel gt pt mt Bool.true g p m n a B }

specification masked_add_typed_commuted_equiv (gt pt : FloatStorage) (mt : MaskStorage)
    (g p m : RegionName) (n : Nat) (a : ℝ) (B : Nat) (R : FP.ScalarArithmetic.Rules) :
    typedOriginal gt pt mt g p m n a B ≡[R] typedCommuted gt pt mt g p m n a B := by
  cases gt <;> cases pt <;> cases mt <;> equiv_decompose
  all_goals fp_prove

#print_fp_assumptions masked_add_typed_commuted_equiv

/-- Legacy bool-buffer specialization, with the same fp32 arithmetic guard. -/
def boolOriginal (grad p mask : RegionName) (n : Nat) (alpha : ℝ) (B : Nat) : Program where
  kernel := masked_add_kernel grad p mask n alpha B
  domain := [.add (Source.prefixBefore
    (masked_add_kernel grad p mask n alpha B).surfaceBody "grad" 1) [B]
    (.ref .real [B] "grad")
    (.mul .real .scalarR (.ref .real [B] "p") (.const alpha))]

def boolCommuted (grad p mask : RegionName) (n : Nat) (alpha : ℝ) (B : Nat) : Program :=
  { boolOriginal grad p mask n alpha B with
    kernel := masked_add_commuted_kernel grad p mask n alpha B }

specification masked_add_bool_commuted_equiv (grad p mask : RegionName)
    (n : Nat) (alpha : ℝ) (B : Nat) (R : FP.ScalarArithmetic.Rules) :
    boolOriginal grad p mask n alpha B ≡[R] boolCommuted grad p mask n alpha B := by
  equiv_decompose
  fp_prove

#print_fp_assumptions masked_add_bool_commuted_equiv
#print_fp_assumptions masked_add_commuted_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.MaskedAddCuda.FPEquiv
