/-
bench/examples/HyperConnectionsWidth

**Width-side hyper-connections (mHC)**: fixed-rank, inference-only DSL
surface for the residual-mixing / branch-input half of manifold-constrained
hyper-connections. The depth-side sibling (`mhcDepthConnectionKernel`) lives
in `bench/examples/HyperConnectionsDepth.lean` — the old two-kernel
`HyperConnections.lean` was split so that each showcased kernel is
self-contained in its own file.

This file intentionally does not model the PyTorch module wrapper, dropout,
autograd, or an arbitrary user branch. The branch is split at the DSL
boundary: `mhcWidthConnectionKernel` prepares the branch input and residual
stream, and the depth-side kernel consumes a separately supplied branch
output.

Four parts, following the canonical KernelIO showcase
`bench/examples/VectorAdd.lean`:

1. **The kernel** — `mhcWidthConnectionKernel`; its `S = T = D = 1,
   numIters = 0` match arm is the scalar fixed-rank slice the proof covers
   (generic rank still future work).
2. **Region-model Hoare triple** — value view `mhcWidth_exec_view`,
   two-store frame `mhcWidth_frame`, and their package
   `mhcWidth_region_run`.
3. **Flat-memory bridge side conditions** — the eight-statement `TraceSafe`
   walk (per-program residual cell at `pid`, the two **shared** scalar logit
   cells at `0`, two single-cell stores at `pid`) and `FlattenOk`.
4. **The spec** — the file's single `specification`:

       mhc_width_correctness : mhcWidthIO tau ⊨ fun res hRes hPre =>
         (fun _ => Real.exp (hRes 0 / tau) * res 0,
          fun _ => Real.exp (hPre 0 / tau) * res 0)

   the first **two-output** (`KernelIO₃ₓ₂`) instance: both output windows
   get their values asserted, and the frame covers every flat cell outside
   the **union** of the two output windows. Spelled out: for every disjoint
   flat placement of the five buffers, every program id whose windows are
   in bounds, and every launch state whose input windows hold
   `res`/`hRes`/`hPre` — everything else arbitrary — the translated pointer
   kernel terminates, `res_mix[pid]` holds `exp(hRes/τ)·res`,
   `branch_in[pid]` holds `exp(hPre/τ)·res`, and every other flat cell is
   unchanged.

## Layout convention

For one program id `b`, tensors are stored row-major:

- `resReg` / `resMixReg`: `[B, S, D]`
- `branchInReg`: `[B, T, D]`
- `hResReg`: `[S, S]`
- `hPreReg`: `[S, T]`

`S` is the residual-stream count, `T` is the branch-view count, and `D` is
the feature dimension. `numIters` controls the fixed Sinkhorn-style
log-domain normalization count. In the proven fixed-rank slice every tensor
degenerates to one cell per program (the residual and both outputs live at
address `b`, both logits are scalar cells at address `0`).
-/

import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL
import VeriTile.Triton.Memory.KernelSpec
import VeriTile.Meta.Specification
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Examples.HyperConnectionsWidth

open VeriTile.Triton
open VeriTile.Triton.KernelIO₃ₓ₂ (Implements)
open scoped VeriTile.Triton.KernelIO₃ₓ₂

/-! ## Part 1 — the kernel -/

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

/-! ## Part 2 — the region-model Hoare triple

The mathematical core, proved against the region model (named buffers, no
pointer arithmetic): a value view (`mhcWidth_exec_view`), a frame lemma over
the two single-cell stores (`mhcWidth_frame`), and their package
`mhcWidth_region_run` — exactly the `hrun` obligation of
`KernelIO₃ₓ₂.Implements.intro`. -/

/-- Value half: the run terminates, `res_mix[pid]` holds `exp(hRes/τ)·res`
and `branch_in[pid]` holds `exp(hPre/τ)·res` (in terms of the launch state's
own memory cells). -/
private theorem mhcWidth_exec_view (tau : ℝ) (s : BlockState) :
    (exec ((mhcWidthConnectionKernel ⟨"res"⟩ ⟨"h_res"⟩ ⟨"h_pre"⟩ ⟨"res_mix"⟩
        ⟨"branch_in"⟩ 1 1 1 0 tau).toAlgKernel) s).map
      (fun s' => (s'.readMem ⟨"res_mix"⟩ s.pid, s'.readMem ⟨"branch_in"⟩ s.pid))
      = some
        (Real.exp (s.readMem ⟨"h_res"⟩ 0 / tau) * s.readMem ⟨"res"⟩ s.pid,
         Real.exp (s.readMem ⟨"h_pre"⟩ 0 / tau) * s.readMem ⟨"res"⟩ s.pid) := by
  simp [mhcWidthConnectionKernel, exec, stepStmts,
    stepStmt, NumericDType.mul, NumericDType.div,
    WithBot.realMul, WithBot.realDiv, BlockState.writeMem, BlockState.readMem]

/-- Frame half: every memory cell other than the two output cells
`res_mix[pid]` / `branch_in[pid]` is preserved by the run. -/
private theorem mhcWidth_frame (tau : ℝ) (s s1 : BlockState)
    (hExec : exec ((mhcWidthConnectionKernel ⟨"res"⟩ ⟨"h_res"⟩ ⟨"h_pre"⟩
        ⟨"res_mix"⟩ ⟨"branch_in"⟩ 1 1 1 0 tau).toAlgKernel) s = some s1)
    (r : RegionName) (o : Nat)
    (h1 : ¬(r = ⟨"res_mix"⟩ ∧ o = s.pid))
    (h2 : ¬(r = ⟨"branch_in"⟩ ∧ o = s.pid)) :
    s1.mem r o = s.mem r o := by
  simp [exec, mhcWidthConnectionKernel,
    ComputeKernel.toAlgKernel, stepStmts, stepStmt,
    NumericDType.mul, NumericDType.div] at hExec
  subst hExec
  rw [BlockState.writeMem_mem, if_neg h2, BlockState.writeMem_mem, if_neg h1]
  rfl

/-- **The region-model Hoare triple** — termination, both output-cell values,
and the two-window frame, from any launch state whose input windows are
loaded. This is what the `⊨` headline transports to flat memory. -/
theorem mhcWidth_region_run (tau : ℝ) (s₀ : BlockState)
    (res hRes hPre : Fin 1 → ℝ)
    (hres : ∀ j : Fin 1, s₀.readMem ⟨"res"⟩ (s₀.pid + j.val) = res j)
    (hhres : ∀ j : Fin 1, s₀.readMem ⟨"h_res"⟩ (0 + j.val) = hRes j)
    (hhpre : ∀ j : Fin 1, s₀.readMem ⟨"h_pre"⟩ (0 + j.val) = hPre j) :
    ∃ s1,
      exec ((mhcWidthConnectionKernel ⟨"res"⟩ ⟨"h_res"⟩ ⟨"h_pre"⟩ ⟨"res_mix"⟩
          ⟨"branch_in"⟩ 1 1 1 0 tau).toAlgKernel) s₀ = some s1
      ∧ (∀ j : Fin 1, s1.readMem ⟨"res_mix"⟩ (s₀.pid + j.val)
          = Real.exp (hRes 0 / tau) * res 0)
      ∧ (∀ j : Fin 1, s1.readMem ⟨"branch_in"⟩ (s₀.pid + j.val)
          = Real.exp (hPre 0 / tau) * res 0)
      ∧ (∀ r o,
          (r ≠ ⟨"res_mix"⟩ ∨ ∀ j : Fin 1, o ≠ s₀.pid + j.val) →
          (r ≠ ⟨"branch_in"⟩ ∨ ∀ j : Fin 1, o ≠ s₀.pid + j.val) →
          s1.mem r o = s₀.mem r o) := by
  have h0res : s₀.readMem ⟨"res"⟩ s₀.pid = res 0 := by simpa using hres 0
  have h0hres : s₀.readMem ⟨"h_res"⟩ 0 = hRes 0 := by simpa using hhres 0
  have h0hpre : s₀.readMem ⟨"h_pre"⟩ 0 = hPre 0 := by simpa using hhpre 0
  have hview := mhcWidth_exec_view tau s₀
  cases hsrc : exec ((mhcWidthConnectionKernel ⟨"res"⟩ ⟨"h_res"⟩ ⟨"h_pre"⟩
      ⟨"res_mix"⟩ ⟨"branch_in"⟩ 1 1 1 0 tau).toAlgKernel) s₀ with
  | none => rw [hsrc] at hview; simp at hview
  | some s1 =>
      rw [hsrc] at hview
      simp only [Option.map_some, Option.some_inj, Prod.mk.injEq] at hview
      obtain ⟨hv1, hv2⟩ := hview
      refine ⟨s1, rfl, fun j => ?_, fun j => ?_, fun r o hc1 hc2 => ?_⟩
      · have hj : (j : Nat) = 0 := Nat.lt_one_iff.mp j.isLt
        rw [hj, Nat.add_zero, hv1, h0hres, h0res]
      · have hj : (j : Nat) = 0 := Nat.lt_one_iff.mp j.isLt
        rw [hj, Nat.add_zero, hv2, h0hpre, h0res]
      · refine mhcWidth_frame tau s₀ s1 hsrc r o ?_ ?_
        · rintro ⟨hr, ho⟩
          rcases hc1 with hne | hno
          · exact hne hr
          · exact hno 0 (by simpa using ho)
        · rintro ⟨hr, ho⟩
          rcases hc2 with hne | hno
          · exact hne hr
          · exact hno 0 (by simpa using ho)

/-! ## Part 3 — flat-memory bridge side conditions

The kernel is register-indirect (the scalar `b` register feeds four of the
five accesses), so no ∀-state contract covers it; the flat-memory bridge
(v1.2) takes the per-execution `Kernel.TraceSafe` contract instead,
discharged below by walking the actual eight-statement execution. All five
accesses are **unmasked** single cells: the residual load and both stores at
`pid` (in bounds when `pid + 1 ≤ bounds reg`), and the two shared logit
loads at the constant cell `0` (in bounds when `0 + 1 ≤ bounds reg`). -/

/-- Inversion for a successful `assign` step. -/
private theorem stepStmt_assign_inv {d : TileDType} {sh : TileShape}
    {nm : RegName} {e : Op d sh} {s s' : BlockState}
    (h : stepStmt (.assign d sh nm e) s = some s') :
    ∃ v, evalOp e s = some v ∧ s' = s.setReg nm d sh v := by
  simp only [stepStmt] at h
  cases hv : evalOp e s with
  | none => rw [hv] at h; exact absurd h (by simp)
  | some v =>
      rw [hv] at h
      replace h : some (s.setReg nm d sh v) = some s' := h
      exact ⟨v, rfl, (Option.some_inj.mp h).symm⟩

/-- Bounds discharge for a single-cell access through the scalar `b`
register: the one addressed cell is `pid₀`, in bounds when
`pid₀ + 1 ≤ bounds reg`. -/
private theorem scalarB_activeAddressSafe (bounds : RegionBounds)
    (pid₀ : Nat) (t : BlockState) (active : TileIndex [] → Prop)
    (hb : t.regs .nat [] "b" = some (Tile.scalar (dtype := .nat) pid₀))
    (reg : RegionName) (hreg : pid₀ + 1 ≤ bounds reg) :
    MemAccess.ActiveAddressSafe bounds
      (MemAccess.region reg (Op.ref .nat [] "b")) t active := by
  simp only [MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe]
  intro offsets hoffs i _
  rw [show evalOp (Op.ref .nat [] "b") t
      = some (Tile.scalar (dtype := .nat) pid₀) from by simp [hb]] at hoffs
  obtain rfl := Option.some_inj.mp hoffs
  simp only [Region.cast_self]
  exact hreg

/-- Bounds discharge for the shared scalar-logit loads: the one addressed
cell is the constant `0`, in bounds when the buffer is nonempty. -/
private theorem sharedScalar_activeAddressSafe (bounds : RegionBounds)
    (t : BlockState) (active : TileIndex [] → Prop)
    (reg : RegionName) (hreg : 0 + 1 ≤ bounds reg) :
    MemAccess.ActiveAddressSafe bounds
      (MemAccess.region reg (Op.constNat 0)) t active := by
  simp only [MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe]
  intro offsets hoffs i _
  rw [show evalOp (Op.constNat 0) t
      = some (Tile.scalar (dtype := .nat) 0) from by simp] at hoffs
  obtain rfl := Option.some_inj.mp hoffs
  simp only [Region.cast_self]
  exact hreg

set_option maxHeartbeats 1600000 in
/-- The fixed-rank width kernel is trace-safe: of its eight statements, five
touch memory — three single-cell loads (`res[pid]`, the shared `h_res[0]` /
`h_pre[0]`) and two single-cell stores (`res_mix[pid]`, `branch_in[pid]`);
the `program_id` assign and the two `exp`/`div`/`mul` assigns are
memory-silent. -/
theorem mhcWidth_traceSafe (tau : ℝ) (bounds : RegionBounds) (s : BlockState)
    (hres : s.pid + 1 ≤ bounds ⟨"res"⟩)
    (hhres : 0 + 1 ≤ bounds ⟨"h_res"⟩)
    (hhpre : 0 + 1 ≤ bounds ⟨"h_pre"⟩)
    (hmix : s.pid + 1 ≤ bounds ⟨"res_mix"⟩)
    (hbin : s.pid + 1 ≤ bounds ⟨"branch_in"⟩) :
    Kernel.TraceSafe bounds
      ((mhcWidthConnectionKernel ⟨"res"⟩ ⟨"h_res"⟩ ⟨"h_pre"⟩ ⟨"res_mix"⟩
        ⟨"branch_in"⟩ 1 1 1 0 tau).toAlgKernel) s := by
  unfold Kernel.TraceSafe
  -- statement 1: b := program_id(0)
  refine Stmt.TraceSafeList.cons_intro (by simp [Stmt.TraceSafe, Op.SafeAt]) ?_
  intro s1 hs1
  obtain ⟨v1, hv1, rfl⟩ := stepStmt_assign_inv hs1
  rw [show evalOp (Op.programId 0) s
      = some (Tile.scalar (dtype := .nat) (s.pids 0)) from by simp] at hv1
  obtain rfl := Option.some_inj.mp hv1
  -- statement 2: residual := load(res + b)   (single cell, unmasked)
  refine Stmt.TraceSafeList.cons_intro ?_ ?_
  · simp only [Stmt.TraceSafe, Op.SafeAt]
    exact ⟨trivial, trivial,
      scalarB_activeAddressSafe bounds (s.pids 0) _ _
        (by simp [BlockState.setReg]) _ hres⟩
  intro s2 hs2
  obtain ⟨v2, hv2, rfl⟩ := stepStmt_assign_inv hs2
  -- statement 3: h_res := load(h_res)   (shared scalar cell 0)
  refine Stmt.TraceSafeList.cons_intro ?_ ?_
  · simp only [Stmt.TraceSafe, Op.SafeAt]
    exact ⟨trivial, trivial, sharedScalar_activeAddressSafe bounds _ _ _ hhres⟩
  intro s3 hs3
  obtain ⟨v3, hv3, rfl⟩ := stepStmt_assign_inv hs3
  -- statement 4: res_mix := exp(h_res / tau) * residual   (register op)
  refine Stmt.TraceSafeList.cons_intro
    (by simp [Stmt.TraceSafe, Op.SafeAt.eq_def]) ?_
  intro s4 hs4
  obtain ⟨v4, hv4, rfl⟩ := stepStmt_assign_inv hs4
  -- statement 5: h_pre := load(h_pre)   (shared scalar cell 0)
  refine Stmt.TraceSafeList.cons_intro ?_ ?_
  · simp only [Stmt.TraceSafe, Op.SafeAt]
    exact ⟨trivial, trivial, sharedScalar_activeAddressSafe bounds _ _ _ hhpre⟩
  intro s5 hs5
  obtain ⟨v5, hv5, rfl⟩ := stepStmt_assign_inv hs5
  -- statement 6: branch_in := exp(h_pre / tau) * residual   (register op)
  refine Stmt.TraceSafeList.cons_intro
    (by simp [Stmt.TraceSafe, Op.SafeAt.eq_def]) ?_
  intro s6 hs6
  obtain ⟨v6, hv6, rfl⟩ := stepStmt_assign_inv hs6
  -- statement 7: store(res_mix + b, res_mix)   (single cell, unmasked)
  refine Stmt.TraceSafeList.cons_intro ?_ ?_
  · simp only [Stmt.TraceSafe, MemAccess.SafeAt, MaskOpt.SafeAt]
    refine ⟨by simp [Op.SafeAt], by simp [Op.SafeAt], trivial,
      scalarB_activeAddressSafe bounds (s.pids 0) _ _ ?_ _ hmix⟩
    simp [BlockState.setReg]
  intro s7 hs7
  simp [stepStmt, BlockState.setReg, TileShape.allIndices] at hs7
  subst hs7
  -- statement 8: store(branch_in + b, branch_in)   (single cell, unmasked)
  refine Stmt.TraceSafeList.cons_intro ?_ (fun _ _ => .nil_intro)
  simp only [Stmt.TraceSafe, MemAccess.SafeAt, MaskOpt.SafeAt]
  refine ⟨by simp [Op.SafeAt], by simp [Op.SafeAt], trivial,
    scalarB_activeAddressSafe bounds (s.pids 0) _ _ ?_ _ hbin⟩
  simp

/-- The fixed-rank width kernel sits inside the bridge's covered fragment. -/
theorem mhcWidth_flattenOk (tau : ℝ) :
    ((mhcWidthConnectionKernel ⟨"res"⟩ ⟨"h_res"⟩ ⟨"h_pre"⟩ ⟨"res_mix"⟩
      ⟨"branch_in"⟩ 1 1 1 0 tau).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [mhcWidthConnectionKernel,
    ComputeKernel.toAlgKernel, StmtList.FlattenOk, Stmt.FlattenOk,
    Op.FlattenOk.eq_def]

/-! ## Part 4 — the spec: `mhcWidthIO ⊨` the width-side mixing pair -/

/-- The fixed-rank width kernel's **IO signature** — the whole
kernel-specific audit surface of the headline: the residual stream `res` and
the two logits `h_res` / `h_pre` in, the residual mix `res_mix` and the
branch input `branch_in` out; all five tile lengths `1`. Program `pid` reads
its own residual cell (`read1 pid = pid`), while **all** programs read the
same shared scalar logit cell (`read2 = read3 = fun _ => 0`); each program
writes its own cell of both outputs. The windows are declared, not parsed
from the kernel: the headline **proves** the kernel's actual addressing
matches them. Buffer sizes are not signature content.

`@[reducible]` is load-bearing: the headline writes `res 0` / `hRes 0` /
`hPre 0`, and the `OfNat (Fin _)` instances behind those literals only fire
if elaboration can see through this def to the literal tile lengths `1`. -/
@[reducible] def mhcWidthIO (tau : ℝ) : KernelIO₃ₓ₂ where
  kernel := mhcWidthConnectionKernel ⟨"res"⟩ ⟨"h_res"⟩ ⟨"h_pre"⟩ ⟨"res_mix"⟩
    ⟨"branch_in"⟩ 1 1 1 0 tau
  in1 := ⟨"res"⟩
  in2 := ⟨"h_res"⟩
  in3 := ⟨"h_pre"⟩
  out1 := ⟨"res_mix"⟩
  out2 := ⟨"branch_in"⟩
  B1 := 1
  B2 := 1
  B3 := 1
  Bout1 := 1
  Bout2 := 1
  read1 := fun pid => pid        -- each program reads its own residual cell
  read2 := fun _ => 0            -- ALL programs read the shared scalar logit
  read3 := fun _ => 0
  write1 := fun pid => pid
  write2 := fun pid => pid

/-- **The headline**: the fixed-rank width kernel implements the mHC
width-side pair — residual mix `exp(hRes/τ)·res` and branch input
`exp(hPre/τ)·res` — on its IO signature; see the module docstring for the
full two-output Hoare triple `⊨` unfolds to. Proof: `Implements.intro`
assembles the region-model triple (Part 2) with the bridge side conditions
(Part 3). -/
specification mhc_width_correctness (tau : ℝ) :
    mhcWidthIO tau ⊨ fun res hRes hPre =>
      (fun _ => Real.exp (hRes 0 / tau) * res 0,
       fun _ => Real.exp (hPre 0 / tau) * res 0) := by
  refine KernelIO₃ₓ₂.Implements.intro _ ?_ ?_ ?_
  · exact mhcWidth_flattenOk tau
  · intro bounds s h1 h2 h3 h4 h5
    exact mhcWidth_traceSafe tau bounds s h1 h2 h3 h4 h5
  · intro s₀ res hRes hPre h1 h2 h3
    exact mhcWidth_region_run tau s₀ res hRes hPre h1 h2 h3

/-! ## Trust gates -/

-- No `sorry`, no smuggled axiom, in the headline's transitive proof.
#axiomsClean mhc_width_correctness

/- The headline's statement surface is the IO signature plus the audit-once
two-output Hoare-triple combinator — no other project constant. -/
#stmtSurfaceSubset mhc_width_correctness ⊆
  [mhcWidthIO, VeriTile.Triton.KernelIO₃ₓ₂.Implements,
   VeriTile.Triton.KernelIO₃ₓ₂.B1, VeriTile.Triton.KernelIO₃ₓ₂.B2,
   VeriTile.Triton.KernelIO₃ₓ₂.B3, VeriTile.Triton.KernelIO₃ₓ₂.Bout1,
   VeriTile.Triton.KernelIO₃ₓ₂.Bout2]

end VeriTile.Bench.Examples.HyperConnectionsWidth
