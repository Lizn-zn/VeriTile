/-
VeriTile.Examples.HyperConnections.MHC

Fixed-rank, inference-only DSL surface for the core tensor flow of
manifold-constrained hyper-connections.

This file intentionally does not model the PyTorch module wrapper, dropout,
autograd, or an arbitrary user branch.  The branch is split at the DSL boundary:
`mhcWidthConnectionKernel` prepares the branch input and residual stream, and
`mhcDepthConnectionKernel` consumes a separately supplied branch output.
-/

import VeriTile.Triton.DSL

namespace VeriTile.Examples.HyperConnections

open VeriTile.Triton

/-!
## Layout convention

For one program id `b`, tensors are stored row-major:

- `resReg` / `resMixReg` / `outReg`: `[B, S, D]`
- `branchInReg` / `branchOutReg`: `[B, T, D]`
- `hResReg`: `[S, S]`
- `hPreReg`: `[S, T]`
- `hPostReg`: `[T, S]`

`S` is the residual-stream count, `T` is the branch-view count, and `D` is the
feature dimension.  `numIters` controls the fixed Sinkhorn-style log-domain
normalization count.
-/

/-- Width-side mHC core.

It computes two normalized maps from logits:

- `resWeights : [S, S]`, used to mix the residual streams;
- `preWeights : [S, T]`, used to produce the branch input views.

The kernel stores:

- `resMixReg[b, s, d] = (resWeights @ residuals)[s, d]`
- `branchInReg[b, t, d] = (preWeightsᵀ @ residuals)[t, d]`

The exact Real spec/proof is intentionally deferred; this definition is the
DSL artifact for issue #87. -/
def mhcWidthConnectionKernel
    (resReg hResReg hPreReg resMixReg branchInReg : RegionName)
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
  z_res := h_res_logits / $ℝ(tau)
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
  z_pre := h_pre_logits / $ℝ(tau)
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

/-- Depth-side mHC core.

This consumes the output of an external branch, mixes it back into the residual
streams through a normalized post map, and adds it to the width-side residual
mixture. -/
def mhcDepthConnectionKernel
    (resMixReg branchOutReg hPostReg outReg : RegionName)
    (S T D numIters : Nat) (tau : ℝ) : ComputeKernel := triton {
  b      := tl.program_id(0)
  offs_s := tl.arange(0, $(S))
  offs_t := tl.arange(0, $(T))
  offs_t2 := tl.arange(0, $(T))
  offs_d := tl.arange(0, $(D))

  res_mix_ptrs := b * $(S * D) + offs_s[:, None] * $(D) + offs_d[None, :]
  branch_ptrs := b * $(T * D) + offs_t[:, None] * $(D) + offs_d[None, :]
  res_mix := tl.load($(resMixReg) + res_mix_ptrs)
  branch_out := tl.load($(branchOutReg) + branch_ptrs)

  h_post_ptrs := offs_t[:, None] * $(S) + offs_s[None, :]
  h_post_logits := tl.load($(hPostReg) + h_post_ptrs)
  z_post := h_post_logits / $ℝ(tau)
  u_post := tl.zeros([$(T)])
  v_post := tl.zeros([$(S)])
  tl.static_range iter in $(numIters) {
    u_post := 0 - tl.log(tl.sum(tl.exp(z_post + v_post[None, :]), axis = 1))
    v_post := 0 - tl.log(tl.sum(tl.exp(z_post + u_post[:, None]), axis = 0))
  }
  post_weights := tl.exp(z_post + u_post[:, None] + v_post[None, :])
  branch_mix := tl.dot(tl.trans(post_weights), branch_out)
  out := res_mix + branch_mix

  out_ptrs := b * $(S * D) + offs_s[:, None] * $(D) + offs_d[None, :]
  tl.store($(outReg) + out_ptrs, out)
}

end VeriTile.Examples.HyperConnections
