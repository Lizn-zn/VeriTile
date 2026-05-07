/-
VeriTile.Examples.HyperConnections.MHC

Fixed-rank, inference-only DSL surface for the core tensor flow of
manifold-constrained hyper-connections.

This file intentionally does not model the PyTorch module wrapper, dropout,
autograd, or an arbitrary user branch.  The branch is split at the DSL boundary:
`mhcWidthConnectionKernel` prepares the branch input and residual stream, and
`mhcDepthConnectionKernel` consumes a separately supplied branch output.
-/

import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
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

/-- Scalar fixed-rank width-side mHC core (`S = T = D = 1`, zero Sinkhorn
iterations).

This is the proof-facing specialization of `mhcWidthConnectionKernel`. It keeps
the same layout convention: for program id `b`, the residual and both outputs
live at address `b`, while both logits are scalar cells at address `0`.
-/
def mhcWidthConnectionKernel1
    (resReg hResReg hPreReg resMixReg branchInReg : RegionName)
    (tau : ℝ) : ComputeKernel := triton {
  b := tl.program_id(0)
  residual := tl.load($(resReg) + b)
  h_res := tl.load($(hResReg))
  res_mix := tl.exp(h_res / $(tau)) * residual
  h_pre := tl.load($(hPreReg))
  branch_in := tl.exp(h_pre / $(tau)) * residual
  tl.store($(resMixReg) + b, res_mix)
  tl.store($(branchInReg) + b, branch_in)
}

/-- Width-side mHC core.

It computes two normalized maps from logits:

- `resWeights : [S, S]`, used to mix the residual streams;
- `preWeights : [S, T]`, used to produce the branch input views.

The kernel stores:

- `resMixReg[b, s, d] = (resWeights @ residuals)[s, d]`
- `branchInReg[b, t, d] = (preWeightsᵀ @ residuals)[t, d]`

The fixed-rank `S = T = D = 1`, `numIters = 0` case is definitionally routed to
`mhcWidthConnectionKernel1`, which is the first proof-covered slice.  The
generic rank / iterative Sinkhorn proof remains future work. -/
def mhcWidthConnectionKernel
    (resReg hResReg hPreReg resMixReg branchInReg : RegionName)
    (S T D numIters : Nat) (tau : ℝ) : ComputeKernel :=
  match S, T, D, numIters with
  | 1, 1, 1, 0 =>
      mhcWidthConnectionKernel1 resReg hResReg hPreReg resMixReg branchInReg tau
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

/-- Fixed-rank (`S = T = D = 1`, zero Sinkhorn iterations) width-side residual
mixing spec. -/
noncomputable def mhcWidthConnectionResMixSpec1 (residual hRes tau : ℝ) : ℝ :=
  Real.exp (hRes / tau) * residual

/-- Fixed-rank (`S = T = D = 1`, zero Sinkhorn iterations) width-side branch
input spec. -/
noncomputable def mhcWidthConnectionBranchInSpec1 (residual hPre tau : ℝ) : ℝ :=
  Real.exp (hPre / tau) * residual

/-- Fixed-rank (`S = T = D = 1`, zero Sinkhorn iterations) depth-side output
spec. -/
noncomputable def mhcDepthConnectionOutSpec1
    (resMix branchOut hPost tau : ℝ) : ℝ :=
  resMix + Real.exp (hPost / tau) * branchOut

theorem mhcWidthConnection_exec_view
    (resReg hResReg hPreReg resMixReg branchInReg : RegionName)
    (tau : ℝ) (s : BlockState)
    (hOutNe : resMixReg ≠ branchInReg) :
    let result := exec
      (mhcWidthConnectionKernel resReg hResReg hPreReg resMixReg branchInReg
        1 1 1 0 tau).toAlgKernel s
    result.map (fun s' =>
      (s'.readMem resMixReg s.pid, s'.readMem branchInReg s.pid)) =
      some
        (mhcWidthConnectionResMixSpec1
          (s.readMem resReg s.pid) (s.readMem hResReg 0) tau,
         mhcWidthConnectionBranchInSpec1
          (s.readMem resReg s.pid) (s.readMem hPreReg 0) tau) := by
  simp [mhcWidthConnectionKernel, mhcWidthConnectionKernel1, exec, stepStmts, stepStmt, evalOp,
    mhcWidthConnectionResMixSpec1, mhcWidthConnectionBranchInSpec1,
    NumericDType.mul, NumericDType.div, WithBot.realMul, WithBot.realDiv,
    BlockState.writeMem, BlockState.readMem, hOutNe]

/-- First fixed-rank mHC width-side correctness theorem.

For `S = T = D = 1` and zero Sinkhorn iterations, the fixed-rank kernel writes the two
Real-valued width-side outputs:

* `resMixReg[b, 0, 0] = exp(hRes[0, 0] / tau) * residual[b, 0, 0]`;
* `branchInReg[b, 0, 0] = exp(hPre[0, 0] / tau) * residual[b, 0, 0]`.
-/
theorem mhcWidthConnection_correct_view
    (resReg hResReg hPreReg resMixReg branchInReg : RegionName)
    (tau : ℝ) (s : BlockState)
    (hOutNe : resMixReg ≠ branchInReg) :
    ComputeKernel.ComputeCorrect
      (mhcWidthConnectionKernel resReg hResReg hPreReg resMixReg branchInReg
        1 1 1 0 tau)
      (fun s0 s' =>
        s0 = s →
          s'.readMem resMixReg s.pid =
            mhcWidthConnectionResMixSpec1
              (s.readMem resReg s.pid) (s.readMem hResReg 0) tau
          ∧
          s'.readMem branchInReg s.pid =
            mhcWidthConnectionBranchInSpec1
              (s.readMem resReg s.pid) (s.readMem hPreReg 0) tau) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel rfl
  intro s0 s' hExec hs0
  subst s0
  have hview := mhcWidthConnection_exec_view
    resReg hResReg hPreReg resMixReg branchInReg tau s hOutNe
  rw [hExec] at hview
  simpa using hview

/-- Scalar fixed-rank depth-side mHC core (`S = T = D = 1`, zero Sinkhorn
iterations).

For program id `b`, this consumes the width-side residual mix at `resMixReg[b]`,
the external branch output at `branchOutReg[b]`, and the scalar post logit at
`hPostReg[0]`, then writes `outReg[b]`.
-/
def mhcDepthConnectionKernel1
    (resMixReg branchOutReg hPostReg outReg : RegionName)
    (tau : ℝ) : ComputeKernel := triton {
  b := tl.program_id(0)
  res_mix := tl.load($(resMixReg) + b)
  branch_out := tl.load($(branchOutReg) + b)
  h_post := tl.load($(hPostReg))
  branch_mix := tl.exp(h_post / $(tau)) * branch_out
  out := res_mix + branch_mix
  tl.store($(outReg) + b, out)
}

/-- Depth-side mHC core.

This consumes the output of an external branch, mixes it back into the residual
streams through a normalized post map, and adds it to the width-side residual
mixture.

The fixed-rank `S = T = D = 1`, `numIters = 0` case is definitionally routed to
`mhcDepthConnectionKernel1`, matching the proof-covered width-side slice.  The
generic rank / iterative Sinkhorn proof remains future work. -/
def mhcDepthConnectionKernel
    (resMixReg branchOutReg hPostReg outReg : RegionName)
    (S T D numIters : Nat) (tau : ℝ) : ComputeKernel :=
  match S, T, D, numIters with
  | 1, 1, 1, 0 =>
      mhcDepthConnectionKernel1 resMixReg branchOutReg hPostReg outReg tau
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

theorem mhcDepthConnection_exec_view
    (resMixReg branchOutReg hPostReg outReg : RegionName)
    (tau : ℝ) (s : BlockState) :
    let result := exec
      (mhcDepthConnectionKernel resMixReg branchOutReg hPostReg outReg
        1 1 1 0 tau).toAlgKernel s
    result.map (fun s' => s'.readMem outReg s.pid) =
      some
        (mhcDepthConnectionOutSpec1
          (s.readMem resMixReg s.pid)
          (s.readMem branchOutReg s.pid)
          (s.readMem hPostReg 0)
          tau) := by
  simp [mhcDepthConnectionKernel, mhcDepthConnectionKernel1, exec, stepStmts, stepStmt, evalOp,
    mhcDepthConnectionOutSpec1, NumericDType.add, NumericDType.mul, NumericDType.div,
    WithBot.realAdd, WithBot.realMul, WithBot.realDiv,
    BlockState.writeMem, BlockState.readMem]

/-- First fixed-rank mHC depth-side correctness theorem.

For `S = T = D = 1` and zero Sinkhorn iterations, the fixed-rank depth kernel
writes:

`outReg[b, 0, 0] = resMixReg[b, 0, 0] + exp(hPost[0, 0] / tau) * branchOutReg[b, 0, 0]`.
-/
theorem mhcDepthConnection_correct_view
    (resMixReg branchOutReg hPostReg outReg : RegionName)
    (tau : ℝ) (s : BlockState) :
    ComputeKernel.ComputeCorrect
      (mhcDepthConnectionKernel resMixReg branchOutReg hPostReg outReg
        1 1 1 0 tau)
      (fun s0 s' =>
        s0 = s →
          s'.readMem outReg s.pid =
            mhcDepthConnectionOutSpec1
              (s.readMem resMixReg s.pid)
              (s.readMem branchOutReg s.pid)
              (s.readMem hPostReg 0)
              tau) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel rfl
  intro s0 s' hExec hs0
  subst s0
  have hview := mhcDepthConnection_exec_view
    resMixReg branchOutReg hPostReg outReg tau s
  rw [hExec] at hview
  simpa using hview

end VeriTile.Examples.HyperConnections
