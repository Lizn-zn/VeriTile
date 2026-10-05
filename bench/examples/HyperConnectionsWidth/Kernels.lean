import VeriTile.Triton.DSL

/-!
HyperConnectionsWidth: source programs shared by the real and FP proof files.
The definitions retain their original operation order, precision, casts, masks
and memory effects. Correct.lean interprets the mathematical projection;
FPEquiv.lean establishes equivalence under the indicated atomic assumptions.
-/

namespace VeriTile.Bench.Examples.HyperConnectionsWidth.Kernels
open VeriTile Triton

/-- Full matrix source, including every normalization iteration. -/
def matrixKernel (resReg hResReg hPreReg resMixReg branchInReg : RegionName)
    (S T D numIters : Nat) (tau : ℝ) : ComputeKernel := triton {
  b      := tl.program_id(0)
  offs_s := tl.arange(0, $(S))
  offs_s2 := tl.arange(0, $(S))
  offs_t := tl.arange(0, $(T))
  offs_d := tl.arange(0, $(D))

  res_ptrs := b * $(S * D) + offs_s[:, None] * $(D) + offs_d[None, :]
  residuals := tl.load($(resReg) + res_ptrs)

  h_res_ptrs := offs_s[:, None] * $(S) + offs_s2[None, :]
  h_res_logits := tl.load($(hResReg) + h_res_ptrs)
  z := h_res_logits / $(tau)
  u := tl.zeros([$(S)])
  v := tl.zeros([$(S)])
  tl.static_range iter in $(numIters) {
    u := 0 - tl.log(tl.sum(tl.exp(z + v[None, :]), axis = 1))
    v := 0 - tl.log(tl.sum(tl.exp(z + u[:, None]), axis = 0))
  }
  res_weights := tl.exp(z + u[:, None] + v[None, :])
  res_mix := tl.dot(res_weights, residuals)

  h_pre_ptrs := offs_s[:, None] * $(T) + offs_t[None, :]
  h_pre_logits := tl.load($(hPreReg) + h_pre_ptrs)
  z := h_pre_logits / $(tau)
  u := tl.zeros([$(S)])
  v := tl.zeros([$(T)])
  tl.static_range iter in $(numIters) {
    u := 0 - tl.log(tl.sum(tl.exp(z + v[None, :]), axis = 1))
    v := 0 - tl.log(tl.sum(tl.exp(z + u[:, None]), axis = 0))
  }
  pre_weights := tl.exp(z + u[:, None] + v[None, :])
  branch_in := tl.dot(tl.trans(pre_weights), residuals)

  res_mix_ptrs := b * $(S * D) + offs_s[:, None] * $(D) + offs_d[None, :]
  branch_ptrs := b * $(T * D) + offs_t[:, None] * $(D) + offs_d[None, :]
  tl.store($(resMixReg) + res_mix_ptrs, res_mix)
  tl.store($(branchInReg) + branch_ptrs, branch_in)
}

/-- Width-side mHC core.

It computes two normalized maps from logits:

- `resWeights : [S, S]`, used to mix the residual streams;
- `preWeights : [S, T]`, used to produce the branch input views.

The kernel stores:

- `resMixReg[b, s, d] = (resWeights @ residuals)[s, d]`
- `branchInReg[b, t, d] = (preWeightsᵀ @ residuals)[t, d]`

The first match arm is the scalar fixed-rank slice (`S = T = D = 1`, zero
Sinkhorn iterations) — retained as a separately proved scalar source:
for program id `b`, the residual and both outputs live at address `b`, both
logits are scalar cells at address `0`, and with zero iterations the
normalized weights degenerate to `exp(h/τ)`. The fallback arm is the faithful
generic-rank transcription; matrixOriginal/matrixOptimized are proved for all
dimensions and iteration counts in Correct.lean and FPEquiv.lean. -/
def mhcWidthConnectionKernel
    (resReg hResReg hPreReg resMixReg branchInReg : RegionName)
    (S T D numIters : Nat) (tau : ℝ) : ComputeKernel :=
  match S, T, D, numIters with
  | 1, 1, 1, 0 => triton {
  b := tl.program_id(0)
  residual := tl.load($(resReg) + b, dtype=tl.float32)
  h_res := tl.load($(hResReg), dtype=tl.float32)
  res_mix := tl.exp(h_res / $(tau)) * residual
  h_pre := tl.load($(hPreReg), dtype=tl.float32)
  branch_in := tl.exp(h_pre / $(tau)) * residual
  tl.store($(resMixReg) + b, res_mix)
  tl.store($(branchInReg) + b, branch_in)
}
  | _, _, _, _ => matrixKernel resReg hResReg hPreReg resMixReg branchInReg S T D numIters tau

def originalKernel (tau : ℝ) : ComputeKernel :=
  mhcWidthConnectionKernel "res" "h_res" "h_pre" "res_mix" "branch_in" 1 1 1 0 tau

def middleKernel (tau : ℝ) : ComputeKernel :=
  let resReg : Region .fp32 := ⟨"res"⟩
  let hResReg : Region .fp32 := ⟨"h_res"⟩
  let hPreReg : Region .fp32 := ⟨"h_pre"⟩
  let resMixReg : Region .fp32 := ⟨"res_mix"⟩
  let branchInReg : Region .fp32 := ⟨"branch_in"⟩
  triton {
  b := tl.program_id(0)
  residual := tl.load(resReg + b)
  h_res := tl.load(hResReg)
  res_mix := residual * tl.exp(h_res / $(tau))
  h_pre := tl.load(hPreReg)
  branch_in := tl.exp(h_pre / $(tau)) * residual
  tl.store(resMixReg + b, res_mix)
  tl.store(branchInReg + b, branch_in)
}

def optimizedKernel (tau : ℝ) : ComputeKernel :=
  let resReg : Region .fp32 := ⟨"res"⟩
  let hResReg : Region .fp32 := ⟨"h_res"⟩
  let hPreReg : Region .fp32 := ⟨"h_pre"⟩
  let resMixReg : Region .fp32 := ⟨"res_mix"⟩
  let branchInReg : Region .fp32 := ⟨"branch_in"⟩
  triton {
  b := tl.program_id(0)
  residual := tl.load(resReg + b)
  h_res := tl.load(hResReg)
  res_mix := residual * tl.exp(h_res / $(tau))
  h_pre := tl.load(hPreReg)
  branch_in := residual * tl.exp(h_pre / $(tau))
  tl.store(resMixReg + b, res_mix)
  tl.store(branchInReg + b, branch_in)
}

/-- General dimensions and iteration count. Matrix products keep their order. -/
def matrixOriginal (S T D numIters : Nat) (tau : ℝ) : ComputeKernel :=
  matrixKernel "res" "h_res" "h_pre" "res_mix" "branch_in" S T D numIters tau

/-- Intermediate source: rewrite only the residual logits. -/
def matrixMiddle (resReg hResReg hPreReg resMixReg branchInReg : RegionName)
    (S T D numIters : Nat) (tau : ℝ) : ComputeKernel := triton {
  b      := tl.program_id(0)
  offs_s := tl.arange(0, $(S))
  offs_s2 := tl.arange(0, $(S))
  offs_t := tl.arange(0, $(T))
  offs_d := tl.arange(0, $(D))

  res_ptrs := b * $(S * D) + offs_s[:, None] * $(D) + offs_d[None, :]
  residuals := tl.load($(resReg) + res_ptrs)

  h_res_ptrs := offs_s[:, None] * $(S) + offs_s2[None, :]
  h_res_logits := tl.load($(hResReg) + h_res_ptrs)
  z := h_res_logits * (1 / $(tau))
  u := tl.zeros([$(S)])
  v := tl.zeros([$(S)])
  tl.static_range iter in $(numIters) {
    u := 0 - tl.log(tl.sum(tl.exp(z + v[None, :]), axis = 1))
    v := 0 - tl.log(tl.sum(tl.exp(z + u[:, None]), axis = 0))
  }
  res_weights := tl.exp(z + u[:, None] + v[None, :])
  res_mix := tl.dot(res_weights, residuals)

  h_pre_ptrs := offs_s[:, None] * $(T) + offs_t[None, :]
  h_pre_logits := tl.load($(hPreReg) + h_pre_ptrs)
  z := h_pre_logits / $(tau)
  u := tl.zeros([$(S)])
  v := tl.zeros([$(T)])
  tl.static_range iter in $(numIters) {
    u := 0 - tl.log(tl.sum(tl.exp(z + v[None, :]), axis = 1))
    v := 0 - tl.log(tl.sum(tl.exp(z + u[:, None]), axis = 0))
  }
  pre_weights := tl.exp(z + u[:, None] + v[None, :])
  branch_in := tl.dot(tl.trans(pre_weights), residuals)

  res_mix_ptrs := b * $(S * D) + offs_s[:, None] * $(D) + offs_d[None, :]
  branch_ptrs := b * $(T * D) + offs_t[:, None] * $(D) + offs_d[None, :]
  tl.store($(resMixReg) + res_mix_ptrs, res_mix)
  tl.store($(branchInReg) + branch_ptrs, branch_in)
}

/-- Replace logit division by scalar reciprocal multiplication. -/
def matrixOptimized (resReg hResReg hPreReg resMixReg branchInReg : RegionName)
    (S T D numIters : Nat) (tau : ℝ) : ComputeKernel := triton {
  b      := tl.program_id(0)
  offs_s := tl.arange(0, $(S))
  offs_s2 := tl.arange(0, $(S))
  offs_t := tl.arange(0, $(T))
  offs_d := tl.arange(0, $(D))

  res_ptrs := b * $(S * D) + offs_s[:, None] * $(D) + offs_d[None, :]
  residuals := tl.load($(resReg) + res_ptrs)

  h_res_ptrs := offs_s[:, None] * $(S) + offs_s2[None, :]
  h_res_logits := tl.load($(hResReg) + h_res_ptrs)
  z := h_res_logits * (1 / $(tau))
  u := tl.zeros([$(S)])
  v := tl.zeros([$(S)])
  tl.static_range iter in $(numIters) {
    u := 0 - tl.log(tl.sum(tl.exp(z + v[None, :]), axis = 1))
    v := 0 - tl.log(tl.sum(tl.exp(z + u[:, None]), axis = 0))
  }
  res_weights := tl.exp(z + u[:, None] + v[None, :])
  res_mix := tl.dot(res_weights, residuals)

  h_pre_ptrs := offs_s[:, None] * $(T) + offs_t[None, :]
  h_pre_logits := tl.load($(hPreReg) + h_pre_ptrs)
  z := h_pre_logits * (1 / $(tau))
  u := tl.zeros([$(S)])
  v := tl.zeros([$(T)])
  tl.static_range iter in $(numIters) {
    u := 0 - tl.log(tl.sum(tl.exp(z + v[None, :]), axis = 1))
    v := 0 - tl.log(tl.sum(tl.exp(z + u[:, None]), axis = 0))
  }
  pre_weights := tl.exp(z + u[:, None] + v[None, :])
  branch_in := tl.dot(tl.trans(pre_weights), residuals)

  res_mix_ptrs := b * $(S * D) + offs_s[:, None] * $(D) + offs_d[None, :]
  branch_ptrs := b * $(T * D) + offs_t[:, None] * $(D) + offs_d[None, :]
  tl.store($(resMixReg) + res_mix_ptrs, res_mix)
  tl.store($(branchInReg) + branch_ptrs, branch_in)
}

end VeriTile.Bench.Examples.HyperConnectionsWidth.Kernels
