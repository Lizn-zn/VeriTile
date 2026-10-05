import VeriTile.Triton.DSL

/-!
HyperConnectionsDepth: source programs shared by the real and FP proof files.
The definitions retain their original operation order, precision, casts, masks
and memory effects. Correct.lean interprets the mathematical projection;
FPEquiv.lean establishes equivalence under the indicated atomic assumptions.
-/

namespace VeriTile.Bench.Examples.HyperConnectionsDepth.Kernels
open VeriTile Triton

/-- Full matrix source, including every normalization iteration. -/
def matrixKernel (resMixReg branchOutReg hPostReg outReg : RegionName)
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
  z := h_post_logits / $(tau)
  u := tl.zeros([$(T)])
  v := tl.zeros([$(S)])
  tl.static_range iter in $(numIters) {
    u := 0 - tl.log(tl.sum(tl.exp(z + v[None, :]), axis = 1))
    v := 0 - tl.log(tl.sum(tl.exp(z + u[:, None]), axis = 0))
  }
  post_weights := tl.exp(z + u[:, None] + v[None, :])
  branch_mix := tl.dot(tl.trans(post_weights), branch_out)
  out := res_mix + branch_mix

  out_ptrs := b * $(S * D) + offs_s[:, None] * $(D) + offs_d[None, :]
  tl.store($(outReg) + out_ptrs, out)
}

/-- Depth-side mHC core.

This consumes the output of an external branch, mixes it back into the residual
streams through a normalized post map, and adds it to the width-side residual
mixture.

The first match arm is the scalar fixed-rank slice (`S = T = D = 1`, zero
Sinkhorn iterations) — retained as a separately proved scalar source:
for program id `b`, it consumes the width-side residual mix at `resMixReg[b]`
and the external branch output at `branchOutReg[b]`, weights the latter by
`exp(hPost/τ)` (the zero-iteration degeneration of the normalized post map,
read from the scalar cell `hPostReg[0]`), and writes `outReg[b]`. The
fallback arm uses matrixKernel, whose general real correctness and FP
equivalence are proved alongside the scalar specialization. -/
def mhcDepthConnectionKernel
    (resMixReg branchOutReg hPostReg outReg : RegionName)
    (S T D numIters : Nat) (tau : ℝ) : ComputeKernel :=
  match S, T, D, numIters with
  | 1, 1, 1, 0 => triton {
  b := tl.program_id(0)
  res_mix := tl.load($(resMixReg) + b, dtype=tl.float32)
  branch_out := tl.load($(branchOutReg) + b, dtype=tl.float32)
  h_post := tl.load($(hPostReg), dtype=tl.float32)
  branch_mix := tl.exp(h_post / $(tau)) * branch_out
  out := res_mix + branch_mix
  tl.store($(outReg) + b, out)
}
  | _, _, _, _ => matrixKernel resMixReg branchOutReg hPostReg outReg S T D numIters tau

def originalKernel (tau : ℝ) : ComputeKernel :=
  mhcDepthConnectionKernel "res_mix" "branch_out" "h_post" "out" 1 1 1 0 tau

def optimizedKernel (tau : ℝ) : ComputeKernel :=
  let resMixReg : Region .fp32 := ⟨"res_mix"⟩
  let branchOutReg : Region .fp32 := ⟨"branch_out"⟩
  let hPostReg : Region .fp32 := ⟨"h_post"⟩
  let outReg : Region .fp32 := ⟨"out"⟩
  triton {
  b := tl.program_id(0)
  res_mix := tl.load(resMixReg + b)
  branch_out := tl.load(branchOutReg + b)
  h_post := tl.load(hPostReg)
  branch_mix := tl.exp(h_post / $(tau)) * branch_out
  out := branch_mix + res_mix
  tl.store(outReg + b, out)
}

/-- General dimensions and iteration count. Matrix products keep their order. -/
def matrixOriginal (S T D numIters : Nat) (tau : ℝ) : ComputeKernel :=
  matrixKernel "res_mix" "branch_out" "h_post" "out" S T D numIters tau

/-- Commute the final pointwise addition. -/
def matrixOptimized (resMixReg branchOutReg hPostReg outReg : RegionName)
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
  z := h_post_logits / $(tau)
  u := tl.zeros([$(T)])
  v := tl.zeros([$(S)])
  tl.static_range iter in $(numIters) {
    u := 0 - tl.log(tl.sum(tl.exp(z + v[None, :]), axis = 1))
    v := 0 - tl.log(tl.sum(tl.exp(z + u[:, None]), axis = 0))
  }
  post_weights := tl.exp(z + u[:, None] + v[None, :])
  branch_mix := tl.dot(tl.trans(post_weights), branch_out)
  out := branch_mix + res_mix

  out_ptrs := b * $(S * D) + offs_s[:, None] * $(D) + offs_d[None, :]
  tl.store($(outReg) + out_ptrs, out)
}

end VeriTile.Bench.Examples.HyperConnectionsDepth.Kernels
