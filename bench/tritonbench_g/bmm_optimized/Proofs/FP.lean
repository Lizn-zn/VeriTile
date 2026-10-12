import bench.tritonbench_g.bmm_optimized.Kernels
import VeriTile.Triton.Float.RewriteTactics
import VeriTile.Meta.StatementAudit

/-! Each reached iteration commutes the fp32 carry and dot result. All masks,
CTA ordering, pointer updates, omitted `other`, typed storage and TF32 option
are retained in context. Guards refer to the actual loaded operands. -/
namespace VeriTile.Bench.TritonBenchG.BmmOptimized.FPEquivProofs
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open scoped VeriTile.Spec
set_option maxHeartbeats 8000000
set_option maxRecDepth 18000

def dotDType : TypedSource.Storage → FloatDType
  | .fp32 => .fp32 | .fp16 => .fp16 | .bf16 => .bf16

def blockDot (dtype : TypedSource.Storage) (BM BN BK : Nat) : Op .real [BM, BN] :=
  match dotDType dtype with
  | .fp32 => .dot (options := {allowTF32 := some Bool.false}) (batch := [])
      (.ref .real [BM, BK] "a") (.ref .real [BK, BN] "b")
  | d => .castFloat .fp32 .real
      (.dotTyped d d .fp32 (options := {allowTF32 := some Bool.false}) (batch := [])
        (.ref d.toTileDType [BM, BK] "a") (.ref d.toTileDType [BK, BN] "b"))

def requirementsFor (kernel : ComputeKernel) (dtype : TypedSource.Storage)
    (BM BN BK : Nat) : List Site :=
  (Source.dynamicRange? kernel.surfaceBody "_i").toList.map fun loop =>
    .range loop.before "_i" loop.start loop.stop loop.stride loop.body
      (.addFP32 (Source.prefixBefore loop.body "o") [BM,BN]
        (.ref .real [BM,BN] "o") (blockDot dtype BM BN BK))

section
variable (dtype : TypedSource.Storage) (divM divN divK : Bool)
    (A B O : RegionName) (M N K BM BN BK GROUP_M : Nat)

def original : Program where
  kernel := TypedSource.original dtype divM divN divK A B O M N K BM BN BK GROUP_M
  domain := requirementsFor (TypedSource.original dtype divM divN divK A B O M N K BM BN BK GROUP_M) dtype BM BN BK

def dotFirst : Program :=
  { original dtype divM divN divK A B O M N K BM BN BK GROUP_M with
    kernel := TypedSource.dotFirst dtype divM divN divK A B O M N K BM BN BK GROUP_M }

theorem requirements_present :
    (original dtype divM divN divK A B O M N K BM BN BK GROUP_M).domain.length = 1 := by
  cases dtype <;> cases divM <;> cases divN <;> cases divK <;> rfl

theorem dot_first_equiv (R : FP.ScalarArithmetic.Rules) :
    original dtype divM divN divK A B O M N K BM BN BK GROUP_M ≡[R]
    dotFirst dtype divM divN divK A B O M N K BM BN BK GROUP_M := by
  cases dtype <;> cases divM <;> cases divN <;> cases divK <;> equiv_decompose <;> fp_prove

end
#print_fp_assumptions dot_first_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.BmmOptimized.FPEquivProofs
