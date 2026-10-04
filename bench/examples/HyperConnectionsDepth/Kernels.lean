import VeriTile.Triton.DSL

/-!
HyperConnectionsDepth: source programs shared by the real and FP proof files.
The definitions retain their original operation order, precision, casts, masks
and memory effects. Correct.lean interprets the mathematical projection;
FPEquiv.lean establishes equivalence under the indicated atomic assumptions.
-/

namespace VeriTile.Bench.Examples.HyperConnectionsDepth.Kernels
open VeriTile Triton

/-- Depth-side mHC core.

This consumes the output of an external branch, mixes it back into the residual
streams through a normalized post map, and adds it to the width-side residual
mixture.

The first match arm is the scalar fixed-rank slice (`S = T = D = 1`, zero
Sinkhorn iterations) — the proof-covered case, written out in scalar form:
for program id `b`, it consumes the width-side residual mix at `resMixReg[b]`
and the external branch output at `branchOutReg[b]`, weights the latter by
`exp(hPost/τ)` (the zero-iteration degeneration of the normalized post map,
read from the scalar cell `hPostReg[0]`), and writes `outReg[b]`. The
fallback arm is the faithful generic-rank transcription; its proof remains
future work. -/
def mhcDepthConnectionKernel
    (resMixReg branchOutReg hPostReg outReg : RegionName)
    (S T D numIters : Nat) (tau : ℝ) : ComputeKernel :=
  match S, T, D, numIters with
  | 1, 1, 1, 0 => triton {
  b := tl.program_id(0)
  res_mix := tl.load($(resMixReg) + b)
  branch_out := tl.load($(branchOutReg) + b)
  h_post := tl.load($(hPostReg))
  branch_mix := tl.exp(h_post / $(tau)) * branch_out
  out := res_mix + branch_mix
  tl.store($(outReg) + b, out)
}
  | _, _, _, _ => triton {
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
  z_post := h_post_logits / $(tau)
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

def originalKernel (tau : ℝ) : ComputeKernel :=
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
  out := res_mix + branch_mix
  tl.store(outReg + b, out)
}

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

end VeriTile.Bench.Examples.HyperConnectionsDepth.Kernels
