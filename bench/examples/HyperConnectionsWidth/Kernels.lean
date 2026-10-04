import VeriTile.Triton.DSL

/-!
HyperConnectionsWidth: source programs shared by the real and FP proof files.
The definitions retain their original operation order, precision, casts, masks
and memory effects. Correct.lean interprets the mathematical projection;
FPEquiv.lean establishes equivalence under the indicated atomic assumptions.
-/

namespace VeriTile.Bench.Examples.HyperConnectionsWidth.Kernels
open VeriTile Triton

/-- Width-side mHC core.

It computes two normalized maps from logits:

- `resWeights : [S, S]`, used to mix the residual streams;
- `preWeights : [S, T]`, used to produce the branch input views.

The kernel stores:

- `resMixReg[b, s, d] = (resWeights @ residuals)[s, d]`
- `branchInReg[b, t, d] = (preWeightsᵀ @ residuals)[t, d]`

The first match arm is the scalar fixed-rank slice (`S = T = D = 1`, zero
Sinkhorn iterations) — the proof-covered case, written out in scalar form:
for program id `b`, the residual and both outputs live at address `b`, both
logits are scalar cells at address `0`, and with zero iterations the
normalized weights degenerate to `exp(h/τ)`. The fallback arm is the faithful
generic-rank transcription; its proof remains future work. -/
def mhcWidthConnectionKernel
    (resReg hResReg hPreReg resMixReg branchInReg : RegionName)
    (S T D numIters : Nat) (tau : ℝ) : ComputeKernel :=
  match S, T, D, numIters with
  | 1, 1, 1, 0 => triton {
  b := tl.program_id(0)
  residual := tl.load($(resReg) + b)
  h_res := tl.load($(hResReg))
  res_mix := tl.exp(h_res / $(tau)) * residual
  h_pre := tl.load($(hPreReg))
  branch_in := tl.exp(h_pre / $(tau)) * residual
  tl.store($(resMixReg) + b, res_mix)
  tl.store($(branchInReg) + b, branch_in)
}
  | _, _, _, _ => triton {
  b      := tl.program_id(0)
  offs_s := tl.arange(0, $(S))
  offs_s2 := tl.arange(0, $(S))
  offs_t := tl.arange(0, $(T))
  offs_d := tl.arange(0, $(D))

  res_ptrs := b * $(S * D) + offs_s[:, None] * $(D) + offs_d[None, :]
  residuals := tl.load($(resReg) + res_ptrs)

  h_res_ptrs := offs_s[:, None] * $(S) + offs_s2[None, :]
  h_res_logits := tl.load($(hResReg) + h_res_ptrs)
  z_res := h_res_logits / $(tau)
  u_res := tl.zeros([$(S)])
  v_res := tl.zeros([$(S)])
  tl.static_range iter in $(numIters) {
    u_res := 0 - tl.log(tl.sum(tl.exp(z_res + v_res[None, :]), axis = 1))
    v_res := 0 - tl.log(tl.sum(tl.exp(z_res + u_res[:, None]), axis = 0))
  }
  res_weights := tl.exp(z_res + u_res[:, None] + v_res[None, :])
  res_mix := tl.dot(res_weights, residuals)

  h_pre_ptrs := offs_s[:, None] * $(T) + offs_t[None, :]
  h_pre_logits := tl.load($(hPreReg) + h_pre_ptrs)
  z_pre := h_pre_logits / $(tau)
  u_pre := tl.zeros([$(S)])
  v_pre := tl.zeros([$(T)])
  tl.static_range iter in $(numIters) {
    u_pre := 0 - tl.log(tl.sum(tl.exp(z_pre + v_pre[None, :]), axis = 1))
    v_pre := 0 - tl.log(tl.sum(tl.exp(z_pre + u_pre[:, None]), axis = 0))
  }
  pre_weights := tl.exp(z_pre + u_pre[:, None] + v_pre[None, :])
  branch_in := tl.dot(tl.trans(pre_weights), residuals)

  res_mix_ptrs := b * $(S * D) + offs_s[:, None] * $(D) + offs_d[None, :]
  branch_ptrs := b * $(T * D) + offs_t[:, None] * $(D) + offs_d[None, :]
  tl.store($(resMixReg) + res_mix_ptrs, res_mix)
  tl.store($(branchInReg) + branch_ptrs, branch_in)
}

def originalKernel (tau : ℝ) : ComputeKernel :=
  let resReg : Region .fp32 := ⟨"res"⟩
  let hResReg : Region .fp32 := ⟨"h_res"⟩
  let hPreReg : Region .fp32 := ⟨"h_pre"⟩
  let resMixReg : Region .fp32 := ⟨"res_mix"⟩
  let branchInReg : Region .fp32 := ⟨"branch_in"⟩
  triton {
  b := tl.program_id(0)
  residual := tl.load(resReg + b)
  h_res := tl.load(hResReg)
  res_mix := tl.exp(h_res / $(tau)) * residual
  h_pre := tl.load(hPreReg)
  branch_in := tl.exp(h_pre / $(tau)) * residual
  tl.store(resMixReg + b, res_mix)
  tl.store(branchInReg + b, branch_in)
}

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

end VeriTile.Bench.Examples.HyperConnectionsWidth.Kernels
