import VeriTile.Triton.DSL
import VeriTile.Triton.Float.BroadcastRewrite
import VeriTile.Triton.Float.RewriteTactics
import VeriTile.Meta.StatementAudit

/-! A non-scalar broadcast rewrite under explicit fp32 evaluation. The common
prefix loads both operands and the common suffix scans and stores the result.
Only reached finite operand checks justify the swap; scan is kept opaque. -/
namespace VeriTile.Tests.FPBroadcastMultiplication
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open scoped VeriTile.Spec

private def tile : Op .real [3,4] := .ref .real [3,4] "x"
private def row : Op .real [3,1] := .expandDim ⟨1,by simp⟩ (.ref .real [3] "a")

private def originalKernel : ComputeKernel :=
  let x : RegionName := "x"
  let a : RegionName := "a"
  let y : RegionName := "y"
  triton {
    tl.region x = tl.float16, a = tl.bfloat16, y = tl.float32
    h = tl.arange(0, 3)
    t = tl.arange(0, 4)
    x = tl.load(x + h[:, None] * 4 + t[None, :]).to(tl.float32)
    a = tl.load(a + h).to(tl.float32)
    product = x * a[:, None]
    total = tl.cumsum(product, axis=1)
    tl.store(y + h[:, None] * 4 + t[None, :], total)
  }

private def scaleFirstKernel : ComputeKernel :=
  let x : RegionName := "x"
  let a : RegionName := "a"
  let y : RegionName := "y"
  triton {
    tl.region x = tl.float16, a = tl.bfloat16, y = tl.float32
    h = tl.arange(0, 3)
    t = tl.arange(0, 4)
    x = tl.load(x + h[:, None] * 4 + t[None, :]).to(tl.float32)
    a = tl.load(a + h).to(tl.float32)
    product = a[:, None] * x
    total = tl.cumsum(product, axis=1)
    tl.store(y + h[:, None] * 4 + t[None, :], total)
  }

private def original : Program where
  kernel := originalKernel
  domain := [.mulBroadcastFP32 (Source.prefixBefore (originalKernel).surfaceBody "product") [3,4] [3,1] tile row]

private def scaleFirst : Program := { original with kernel := scaleFirstKernel }

theorem broadcast_equiv (R : FP.ScalarArithmetic.Rules) : original ≡[R] scaleFirst := by
  equiv_decompose
  fp_prove

private def originalDefaultKernel (dtype out_dtype : FloatDType) : ComputeKernel :=
  let x : RegionName := "x"
  let a : RegionName := "a"
  let y : RegionName := "y"
  triton {
    tl.region x = DTYPE, a = OUT_DTYPE, y = tl.float32
    h = tl.arange(0, 3)
    t = tl.arange(0, 4)
    x = tl.load(x + h[:, None] * 4 + t[None, :]).to(tl.float32)
    a = tl.load(a + h).to(tl.float32)
    product = x * a[:, None]
    total = tl.cumsum(product, axis=1)
    tl.store(y + h[:, None] * 4 + t[None, :], total)
  }

private def scaleFirstDefaultKernel (dtype out_dtype : FloatDType) : ComputeKernel :=
  let x : RegionName := "x"
  let a : RegionName := "a"
  let y : RegionName := "y"
  triton {
    tl.region x = DTYPE, a = OUT_DTYPE, y = tl.float32
    h = tl.arange(0, 3)
    t = tl.arange(0, 4)
    x = tl.load(x + h[:, None] * 4 + t[None, :]).to(tl.float32)
    a = tl.load(a + h).to(tl.float32)
    product = a[:, None] * x
    total = tl.cumsum(product, axis=1)
    tl.store(y + h[:, None] * 4 + t[None, :], total)
  }

private def originalDefault (dtype out_dtype : FloatDType) : Program where
  kernel := originalDefaultKernel dtype out_dtype
  domain := [.mulBroadcast (Source.prefixBefore (originalDefaultKernel dtype out_dtype).surfaceBody "product") [3,4] [3,1] tile row]

private def scaleFirstDefault (dtype out_dtype : FloatDType) : Program :=
  { originalDefault dtype out_dtype with kernel := scaleFirstDefaultKernel dtype out_dtype }

theorem broadcast_default_equiv (dtype out_dtype : FloatDType) (R : FP.ScalarArithmetic.Rules) :
    originalDefault dtype out_dtype ≡[R] scaleFirstDefault dtype out_dtype := by
  equiv_decompose
  fp_prove

#print_fp_assumptions broadcast_default_equiv

#print_fp_assumptions broadcast_equiv
#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Tests.FPBroadcastMultiplication
