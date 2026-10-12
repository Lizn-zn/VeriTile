import bench.tritonbench_g.chunk_gla_fwd.Proofs.FPOutput
import bench.tritonbench_g.chunk_gla_fwd.Proofs.FPInter
import bench.tritonbench_g.chunk_gla_fwd.Proofs.FPIntra
import bench.tritonbench_g.chunk_gla_fwd.Proofs.FPSplit
import bench.tritonbench_g.chunk_gla_fwd.Proofs.FPMerge

/-! Floating-point equivalence of all five complete kernels in `Kernels.lean`.
Each candidate commutes exactly one fp32 operation. Dimensions and launch IDs
remain symbolic. Domains check the actual operands after the original prefix,
inside every reachable branch/iteration; they do not constrain final outputs.
All casts, intrinsic `tl.exp`, dot options, masks, padding and stores remain.
These are conditional model proofs under admitted atoms, not IEEE identities. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGlaFwd.FPEquiv
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open scoped VeriTile.Spec
set_option maxRecDepth 18000

namespace Output

/-- Finite checks cover the final carry and A·V dot after the complete K loop. -/
def intraDot (storage : TypedOutput.Storage) (BT BV : Nat) : Op .real [BT,BV] :=
  match storage with
  | .fp32 => .dot (batch := []) (.ref .real [BT,BT] "b_A") (.ref .real [BT,BV] "b_v")
      {allowTF32 := some Bool.false}
  | .fp16 => .castFloat .fp32 .real (.dotTyped .fp16 .fp16 .fp32 (batch := [])
      (.ref .fp16 [BT,BT] "b_A") (.ref .fp16 [BT,BV] "b_v") {allowTF32 := some Bool.false})
  | .bf16 => .castFloat .fp32 .real (.dotTyped .bf16 .bf16 .fp32 (batch := [])
      (.ref .bf16 [BT,BT] "b_A") (.ref .bf16 [BT,BV] "b_v") {allowTF32 := some Bool.false})

def outputRequirements (kernel : ComputeKernel) (storage : TypedOutput.Storage) (BT BV : Nat) : List Site :=
  [.addFP32 (Source.prefixBefore kernel.surfaceBody "b_o" 1) [BT,BV]
    (.ref .real [BT,BV] "b_o") (intraDot storage BT BV)]

section
variable (queryStorage valueStorage gateStorage : TypedOutput.Storage)
    (dtype out_dtype : FloatDType) (q v g h o A : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat) (scale : ℝ) (T K V BT BK BV : Nat)

def outputOriginal : Program where
  kernel := TypedOutput.original queryStorage valueStorage gateStorage dtype out_dtype q v g h o A
    s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV
  domain := outputRequirements (TypedOutput.original queryStorage valueStorage gateStorage dtype out_dtype q v g h o A
    s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV) valueStorage BT BV

def outputSumLeft : Program :=
  { outputOriginal queryStorage valueStorage gateStorage dtype out_dtype q v g h o A
      s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV with
    kernel := TypedOutput.sumLeft queryStorage valueStorage gateStorage dtype out_dtype q v g h o A
      s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV }

specification output_equiv (R : FP.ScalarArithmetic.Rules) :
    outputOriginal queryStorage valueStorage gateStorage dtype out_dtype q v g h o A
      s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV ≡[R]
    outputSumLeft queryStorage valueStorage gateStorage dtype out_dtype q v g h o A
      s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV := by
  exact FPOutput.output_equiv queryStorage valueStorage gateStorage dtype out_dtype q v g h o A
    s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV R
end

#print_fp_assumptions output_equiv
end Output

namespace Inter

/-- Finite checks cover the carry and dot in each reached K iteration after both early-return tests. -/
def interRequirements (kernel : ComputeKernel) (BC BK : Nat) : List Site :=
  (do
    let outer ← Source.conditional? kernel.surfaceBody (fun code =>
      (Source.conditional? code (fun body => (Source.dynamicRange? body "i_k").isSome)).isSome)
    let inner ← Source.conditional? outer.yes (fun body => (Source.dynamicRange? body "i_k").isSome)
    let loop ← Source.dynamicRange? inner.yes "i_k"
    return Site.branch outer.before outer.condition Bool.true
      (.branch inner.before inner.condition Bool.true
        (.range loop.before "i_k" loop.start loop.stop loop.stride loop.body
          (.addFP32 (Source.prefixBefore loop.body "b_A") [BC,BC]
            (.ref .real [BC,BC] "b_A")
            (.dot (batch := []) (.ref .real [BC,BK] "b_qg") (.ref .real [BK,BC] "b_kg") {}))))).toList
section
variable (qs ks gs : TypedInter.Storage) (out : FloatDType) (q k g A : RegionName)
  (s_k_h s_k_t : Nat) (scale : ℝ) (T K BT BC BK NC : Nat)
def interOriginal : Program where
  kernel := TypedInter.original qs ks gs out q k g A s_k_h s_k_t scale T K BT BC BK NC
  domain := interRequirements (TypedInter.original qs ks gs out q k g A s_k_h s_k_t scale T K BT BC BK NC) BC BK
def interDotFirst : Program :=
  { interOriginal qs ks gs out q k g A s_k_h s_k_t scale T K BT BC BK NC with
    kernel := TypedInter.dotFirst qs ks gs out q k g A s_k_h s_k_t scale T K BT BC BK NC }
specification inter_equiv (R : FP.ScalarArithmetic.Rules) :
    interOriginal qs ks gs out q k g A s_k_h s_k_t scale T K BT BC BK NC ≡[R]
    interDotFirst qs ks gs out q k g A s_k_h s_k_t scale T K BT BC BK NC := by
  exact FPInter.inter_equiv qs ks gs out q k g A s_k_h s_k_t scale T K BT BC BK NC R
end
#print_fp_assumptions inter_equiv
end Inter

namespace Intra

/-- Finite checks cover the sum and scale for each reached column, on all lanes of where. -/
def intraRequirements (kernel : ComputeKernel) (BC : Nat) (scale : ℝ) : List Site :=
  (do
    let active ← Source.conditional? kernel.surfaceBody (fun code => (Source.dynamicRange? code "j").isSome)
    let loop ← Source.dynamicRange? active.yes "j"
    return Site.branch active.before active.condition Bool.true
      (.range loop.before "j" loop.start loop.stop loop.stride loop.body
        (.mulScalar (Source.prefixBefore loop.body "b_A" 1) BC []
          (.ref .real [BC] "b_A") (.const scale)))).toList
section
variable (qs ks gs : TypedIntra.Storage) (q k g A : RegionName)
  (s_k_h s_k_t : Nat) (scale : ℝ) (T K BT BC BK : Nat)
def intraOriginal : Program where
  kernel := TypedIntra.original qs ks gs q k g A s_k_h s_k_t scale T K BT BC BK
  domain := intraRequirements (TypedIntra.original qs ks gs q k g A s_k_h s_k_t scale T K BT BC BK) BC scale
def intraScaleFirst : Program :=
  { intraOriginal qs ks gs q k g A s_k_h s_k_t scale T K BT BC BK with
    kernel := TypedIntra.scaleFirst qs ks gs q k g A s_k_h s_k_t scale T K BT BC BK }
specification intra_equiv (R : FP.ScalarArithmetic.Rules) :
    intraOriginal qs ks gs q k g A s_k_h s_k_t scale T K BT BC BK ≡[R]
    intraScaleFirst qs ks gs q k g A s_k_h s_k_t scale T K BT BC BK := by
  exact FPIntra.intra_equiv qs ks gs q k g A s_k_h s_k_t scale T K BT BC BK R
end
#print_fp_assumptions intra_equiv
end Intra

namespace Split

/-- Finite checks cover the sum after the source zero/add steps and scale, on all lanes of where. -/
def splitRequirements (kernel : ComputeKernel) (BC : Nat) (scale : ℝ) : List Site :=
  (do
    let active ← Source.conditional? kernel.surfaceBody (fun code => (Source.dynamicRange? code "j").isSome)
    let loop ← Source.dynamicRange? active.yes "j"
    return Site.branch active.before active.condition Bool.true
      (.range loop.before "j" loop.start loop.stop loop.stride loop.body
        (.mulScalar (Source.prefixBefore loop.body "b_A" 2) BC []
          (.ref .real [BC] "b_A") (.const scale)))).toList
section
variable (qs ks gs : TypedSplit.Storage) (q k g A : RegionName)
  (s_k_h s_k_t : Nat) (scale : ℝ) (T K BT BC BK NC : Nat)
def splitOriginal : Program where
  kernel := TypedSplit.original qs ks gs q k g A s_k_h s_k_t scale T K BT BC BK NC
  domain := splitRequirements (TypedSplit.original qs ks gs q k g A s_k_h s_k_t scale T K BT BC BK NC) BC scale
def splitScaleFirst : Program :=
  { splitOriginal qs ks gs q k g A s_k_h s_k_t scale T K BT BC BK NC with
    kernel := TypedSplit.scaleFirst qs ks gs q k g A s_k_h s_k_t scale T K BT BC BK NC }
specification split_equiv (R : FP.ScalarArithmetic.Rules) :
    splitOriginal qs ks gs q k g A s_k_h s_k_t scale T K BT BC BK NC ≡[R]
    splitScaleFirst qs ks gs q k g A s_k_h s_k_t scale T K BT BC BK NC := by
  exact FPSplit.split_equiv qs ks gs q k g A s_k_h s_k_t scale T K BT BC BK NC R
end
#print_fp_assumptions split_equiv
end Split

namespace Merge

/-- Finite checks cover the carry and checked load in every reached NK iteration. -/
def mergeRequirements (kernel : ComputeKernel) (BC : Nat) : List Site :=
  (do
    let branch ← Source.conditional? kernel.surfaceBody (fun body => (Source.dynamicRange? body "i_k").isSome)
    let loop ← Source.dynamicRange? branch.yes "i_k"
    return Site.branch branch.before branch.condition Bool.true
      (.range loop.before "i_k" loop.start loop.stop loop.stride loop.body
        (.addFP32 (Source.prefixBefore loop.body "b_A") [BC,BC]
          (.ref .real [BC,BC] "b_A")
          (.load .real (.blockPtr (.ref .blockPtr [BC,BC] "p_A") [0,1]) .none .unspecified)))).toList

def mergeOriginal (dtype : FloatDType) (A A2 : RegionName) (T BT BC NK : Nat) : Program where
  kernel := TypedMerge.original dtype A A2 T BT BC NK
  domain := mergeRequirements (TypedMerge.original dtype A A2 T BT BC NK) BC

def mergeLoadFirst (dtype : FloatDType) (A A2 : RegionName) (T BT BC NK : Nat) : Program :=
  { mergeOriginal dtype A A2 T BT BC NK with kernel := TypedMerge.loadFirst dtype A A2 T BT BC NK }

specification merge_equiv (dtype : FloatDType) (A A2 : RegionName) (T BT BC NK : Nat)
    (R : FP.ScalarArithmetic.Rules) :
    mergeOriginal dtype A A2 T BT BC NK ≡[R] mergeLoadFirst dtype A A2 T BT BC NK := by
  exact FPMerge.merge_equiv dtype A A2 T BT BC NK R

#print_fp_assumptions merge_equiv
end Merge

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ChunkGlaFwd.FPEquiv
