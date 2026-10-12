import bench.tritonbench_g.context_attn_bloom.Proofs.Formula

namespace VeriTile.Bench.TritonBenchG.ContextAttnBloom.FiniteMaskFormula
open VeriTile.Triton Math Formula
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 1500000
set_option maxRecDepth 10000

def layout : Layout where
  BM := 16
  BN := 16
  D := 16
  headDim := 16
  kvGroup := 1
  qRow := 16
  qHead := 256
  qDim := 1
  kRow := 16
  kHead := 256
  kDim := 1
  vRow := 16
  vHead := 256
  vDim := 1
  outRow := 16
  outHead := 256
  outDim := 1
  requestRow := 16
  requestToken := 1

def launch : Launch := ⟨0,0,0⟩
def arrays : Inputs where
  q := fun _ => 0
  k := fun _ => 0
  v := fun offset => if offset=16 then 1 else 0
  start := fun _ => 0
  sequence := fun _ => 16
  prompt := fun _ => 0
  request := fun _ => 0
  tokens := Int.ofNat

/-- Query zero has zero value; the next token has value one in column zero.
All scores before causal masking are zero. -/
def idealCausal : ℝ :=
  (∑ j : Fin 16, if j.val ≤ 0 then Real.exp 0*arrays.v (j.val*16) else 0) /
    (∑ j : Fin 16, if j.val ≤ 0 then Real.exp 0 else 0)

theorem ideal_zero : idealCausal = 0 := by
  norm_num [idealCausal,arrays,Fin.sum_univ_succ]

theorem one_block : count layout launch arrays = 1 := by decide
theorem cardinality : count layout launch arrays*layout.BN = 16 := by decide

theorem denominator_one : denominator layout launch arrays (1/4) 0 1 =
    1+15*Real.exp (-100000000) := by
  change (∑ j : Fin 16, if j.val < 1*16 then
    Real.exp (score layout launch arrays (1/4) 0 j.val) else 0) = _
  norm_num [score,query,key,queryRow,Math.sequence,Math.limit,token,
    layout,launch,arrays,Fin.sum_univ_succ]

theorem numerator_one : numerator layout launch arrays (1/4) 0 0 1 =
    Real.exp (-100000000) := by
  change (∑ j : Fin 16, if j.val < 1*16 then
    Real.exp (score layout launch arrays (1/4) 0 j.val)*value layout launch arrays j.val 0 else 0) = _
  norm_num [score,query,key,value,queryRow,Math.sequence,Math.limit,token,
    layout,launch,arrays,Fin.sum_univ_succ,Int.toNat]

/-- The exact finite-sentinel formula retains a positive future-token term.
This concerns real arithmetic, not the underflow behavior of GPU exponentials. -/
theorem finite_formula :
    expected layout launch arrays (1/4) 0 0 (count layout launch arrays) =
      Real.exp (-100000000)/(1+15*Real.exp (-100000000)) := by
  rw [one_block]
  simp only [expected,zero_mul,zero_add,numerator_one,denominator_one]
  simp [numerator]

theorem finite_formula_ne_ideal :
    expected layout launch arrays (1/4) 0 0 (count layout launch arrays) ≠ idealCausal := by
  rw [finite_formula,ideal_zero]
  apply ne_of_gt
  positivity

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnBloom.FiniteMaskFormula
