import bench.examples.HyperConnectionsWidth.Kernels
import VeriTile.Triton.Math.Sinkhorn
import VeriTile.Triton.Math.MatrixRewrite

/- The matrix specifications at the end cover symbolic S/T/D and every finite
normalization count, with region-memory outputs and frames. The earlier scalar
KernelIO proofs additionally retain their flat-memory bridge. -/
/-
bench/examples/HyperConnectionsWidth

**Width-side hyper-connections (mHC)**: fixed-rank, inference-only DSL
surface for the residual-mixing / branch-input half of manifold-constrained
hyper-connections. The depth-side sibling (`mhcDepthConnectionKernel`) lives
in `bench/examples/HyperConnectionsDepth/Correct.lean` — the old two-kernel
`HyperConnections.lean` was split so that each showcased kernel is
self-contained in its example directory.

This file intentionally does not model the PyTorch module wrapper, dropout,
autograd, or an arbitrary user branch. The branch is split at the DSL
boundary: `mhcWidthConnectionKernel` prepares the branch input and residual
stream, and the depth-side kernel consumes a separately supplied branch
output.

Four parts, following the canonical KernelIO showcase
`bench/examples/VectorAdd/Correct.lean`:

1. **The kernel** — `mhcWidthConnectionKernel`; its `S = T = D = 1,
   numIters = 0` match arm is the scalar fixed-rank slice the proof covers
   (the general matrix specifications follow at the end).
2. **Region-model Hoare triple** — value view `mhcWidth_exec_view`,
   two-store frame `mhcWidth_frame`, and their package
   `mhcWidth_region_run`.
3. **Flat-memory bridge side conditions** — the eight-statement `TraceSafe`
   walk (per-program residual cell at `pid`, the two **shared** scalar logit
   cells at `0`, two single-cell stores at `pid`) and `FlattenOk`.
4. **The spec** — the original kernel's `specification`:

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

The optimized implementation in Kernels.lean has its own real specification
below, with the same mathematical formula and IO contract. Its proof uses
real arithmetic and execution semantics independently of FPEquiv.lean.
-/

import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL
import VeriTile.Triton.Memory.KernelSpec
import VeriTile.Meta.Specification
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Examples.HyperConnectionsWidth
open VeriTile.Bench.Examples.HyperConnectionsWidth.Kernels

-- Correctness interprets the shared typed source through its real projection.
attribute [local simp] VeriTile.Triton.ComputeExpr.toAlgorithm?
  VeriTile.Triton.ComputeOp.toAlgorithm? VeriTile.Triton.ComputeDType.eraseDType
open VeriTile.Triton
open VeriTile.Triton.KernelIO₃ₓ₂ (Implements)
open scoped VeriTile.Triton.KernelIO₃ₓ₂

/-! ## Part 1 — the kernel -/


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
  · simp only [ComputeDType.eraseDType, Stmt.TraceSafe, Op.SafeAt]
    exact ⟨trivial, trivial,
      scalarB_activeAddressSafe bounds (s.pids 0) _ _
        (by simp [BlockState.setReg]) _ hres⟩
  intro s2 hs2
  obtain ⟨v2, hv2, rfl⟩ := stepStmt_assign_inv hs2
  -- statement 3: h_res := load(h_res)   (shared scalar cell 0)
  refine Stmt.TraceSafeList.cons_intro ?_ ?_
  · simp only [ComputeDType.eraseDType, Stmt.TraceSafe, Op.SafeAt]
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
  · simp only [ComputeDType.eraseDType, Stmt.TraceSafe, Op.SafeAt]
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
    Spec.Real (mhcWidthIO tau ⊨ fun res hRes hPre =>
      (fun _ => Real.exp (hRes 0 / tau) * res 0,
       fun _ => Real.exp (hPre 0 / tau) * res 0)) := by
  refine KernelIO₃ₓ₂.Implements.intro _ ?_ ?_ ?_
  · exact mhcWidth_flattenOk tau
  · intro bounds s h1 h2 h3 h4 h5
    exact mhcWidth_traceSafe tau bounds s h1 h2 h3 h4 h5
  · intro s₀ res hRes hPre h1 h2 h3
    exact mhcWidth_region_run tau s₀ res hRes hPre h1 h2 h3

/-! ## Optimized implementation: real correctness -/

section Optimized
set_option maxHeartbeats 5000000
attribute [local simp] ComputeExpr.toAlgorithm? ComputeOp.toAlgorithm? ComputeDType.eraseDType

-- This equality is derived in the real interpreter; it uses no FP assumptions.
private theorem optimized_exec (tau : ℝ) (s : BlockState) :
    exec (optimizedKernel tau).toAlgKernel s =
      exec (mhcWidthConnectionKernel "res" "h_res" "h_pre" "res_mix" "branch_in" 1 1 1 0 tau).toAlgKernel s := by
  simp [optimizedKernel, mhcWidthConnectionKernel, exec, stepStmts, stepStmt, evalOp.eq_def,
    Region.cast, Tile.bop, Tile.uop, NumericDType.mul, NumericDType.div, mul_comm]

/-- The optimized source retains the original IO windows and memory contract. -/
@[reducible] def optimizedIO (tau : ℝ) : KernelIO₃ₓ₂ :=
  { mhcWidthIO tau with kernel := optimizedKernel tau, projection := by rfl }

/-- The optimized implementation computes the same independent real formula. -/
specification mhc_width_optimized_correctness (tau : ℝ) :
    Spec.Real (optimizedIO tau ⊨ fun res hRes hPre =>
      (fun _ => Real.exp (hRes 0 / tau) * res 0,
       fun _ => Real.exp (hPre 0 / tau) * res 0)) := by
  refine KernelIO₃ₓ₂.Implements.intro _
    ?_ ?_ ?_
  · simp [optimizedKernel, Kernel.FlattenOk,
      StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]
  · intro bounds s h1 h2 h3 h4 h5
    simpa [optimizedIO, mhcWidthIO, Kernel.TraceSafe, optimizedKernel, mhcWidthConnectionKernel,
      Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def, MaskOpt.SafeAt,
      MemAccess.SafeAt, stepStmt, evalOp.eq_def,
      Region.cast, Tile.bop, Tile.uop, NumericDType.mul, NumericDType.div, mul_comm]
      using mhcWidth_traceSafe tau bounds s h1 h2 h3 h4 h5
  · intro s res hRes hPre h1 h2 h3
    change ∃ s1, exec (optimizedKernel tau).toAlgKernel s = some s1 ∧ _
    rw [optimized_exec]
    exact mhcWidth_region_run tau s res hRes hPre h1 h2 h3

#axiomsClean mhc_width_optimized_correctness
#stmtSurfaceSubset mhc_width_optimized_correctness ⊆
  [Spec.Real, optimizedIO, VeriTile.Triton.KernelIO₃ₓ₂.Implements,
   VeriTile.Triton.KernelIO₃ₓ₂.B1, VeriTile.Triton.KernelIO₃ₓ₂.B2,
   VeriTile.Triton.KernelIO₃ₓ₂.B3, VeriTile.Triton.KernelIO₃ₓ₂.Bout1,
   VeriTile.Triton.KernelIO₃ₓ₂.Bout2]

end Optimized

/-! ## Trust gates -/

-- No `sorry`, no smuggled axiom, in the headline's transitive proof.
#axiomsClean mhc_width_correctness

/- The headline's statement surface is the IO signature plus the audit-once
two-output Hoare-triple combinator — no other project constant. -/
#stmtSurfaceSubset mhc_width_correctness ⊆
  [Spec.Real, mhcWidthIO, VeriTile.Triton.KernelIO₃ₓ₂.Implements,
   VeriTile.Triton.KernelIO₃ₓ₂.B1, VeriTile.Triton.KernelIO₃ₓ₂.B2,
   VeriTile.Triton.KernelIO₃ₓ₂.B3, VeriTile.Triton.KernelIO₃ₓ₂.Bout1,
   VeriTile.Triton.KernelIO₃ₓ₂.Bout2]

end VeriTile.Bench.Examples.HyperConnectionsWidth


/-! General width connection: Sinkhorn(H_res / tau) R and
Sinkhorn(H_pre / tau)ᵀ R. Both normalization loops and both output stores
are part of the proved source. The recurrence is defined in Math/Sinkhorn. -/
namespace VeriTile.Bench.Examples.HyperConnectionsWidth.MatrixCorrect
open VeriTile Triton LogSinkhorn
open VeriTile.Bench.Examples.HyperConnectionsWidth.Kernels
set_option maxHeartbeats 6400000
set_option linter.unusedSimpArgs false

noncomputable def residualFormula (res : Tile .real [S, D]) (logits : Tile .real [S, S])
    (tau : ℝ) (iters : Nat) : Tile .real [S, D] := Tile.dot [] (weights (scale logits tau) iters) res

noncomputable def branchFormula (res : Tile .real [S, D]) (logits : Tile .real [S, T])
    (tau : ℝ) (iters : Nat) : Tile .real [T, D] :=
  Tile.dot [] (Tile.transpose [] (weights (scale logits tau) iters)) res

private theorem prefix_run (S T D iters : Nat) (tau : ℝ) (s : BlockState) :
    ∃ a, stepStmts ((matrixOriginal S T D iters tau).body.take 12) s = some a ∧
      a.regs .real [S, S] "z" = some (scale (readMatrix s "h_res" 0 S S) tau) ∧
      a.regs .real [S] "u" = some (iterates (scale (readMatrix s "h_res" 0 S S) tau) 0).1 ∧
      a.regs .real [S] "v" = some (iterates (scale (readMatrix s "h_res" 0 S S) tau) 0).2 ∧
      a.regs .real [S, D] "residuals" = some (readMatrix s "res" (s.pid * (S * D)) S D) ∧
      a.regs .nat [] "b" = some (Tile.scalar s.pid) ∧
      a.regs .nat [S] "offs_s" = some (Tile.vec (fun i => i.val)) ∧
      a.regs .nat [T] "offs_t" = some (Tile.vec (fun i => i.val)) ∧
      a.regs .nat [D] "offs_d" = some (Tile.vec (fun i => i.val)) ∧
      a.mem = s.mem ∧ a.pids = s.pids := by
  simp [matrixOriginal, matrixKernel, ComputeKernel.body, ComputeKernel.toAlgKernel,
    List.take, stepStmts, stepStmt, evalOp.eq_def, BlockState.setReg,
    NumericDType.add, NumericDType.mul, NumericDType.div, readMatrix, scale, iterates,
    BlockState.readMem, Region.cast, Option.bind, Tile.bop, Tile.expandDim, TileShape.dropInsertedIndex, TileShape.insertAxis]
  repeat' first | exact rfl | apply And.intro

specification mhc_width_matrix_correct (S T D iters : Nat) (tau : ℝ) (s : BlockState) :
    Spec.Real (∃ t, exec (matrixOriginal S T D iters tau) s = some t ∧
      (∀ i : TileIndex [S, D], t.readMem "res_mix" (s.pid * (S * D) + i.1.val * D + i.2.1.val) =
        ((residualFormula (readMatrix s "res" (s.pid * (S * D)) S D)
          (readMatrix s "h_res" 0 S S) tau iters).data i).unbotD 0) ∧
      (∀ i : TileIndex [T, D], t.readMem "branch_in" (s.pid * (T * D) + i.1.val * D + i.2.1.val) =
        ((branchFormula (readMatrix s "res" (s.pid * (S * D)) S D)
          (readMatrix s "h_pre" 0 S T) tau iters).data i).unbotD 0) ∧
      (∀ (r : RegionName) o,
        (r ≠ "res_mix" ∨ ∀ i : TileIndex [S, D], o ≠ s.pid * (S * D) + i.1.val * D + i.2.1.val) →
        (r ≠ "branch_in" ∨ ∀ i : TileIndex [T, D], o ≠ s.pid * (T * D) + i.1.val * D + i.2.1.val) →
        t.mem r o = s.mem r o)) := by
  obtain ⟨a, ha, hz, hu, hv, hres, hb, hs, ht, hd, hm, hp⟩ := prefix_run S T D iters tau s
  obtain ⟨t, hloop, hz', hu', hv', hmem, hpids, hr⟩ :=
    loop_run (scale (readMatrix s "h_res" 0 S S) tau) iters a hz hu hv
  have hres' := (hr .real [S, D] "residuals" (by decide) (by decide) (by decide)).trans hres
  have hb' := (hr .nat [] "b" (by decide) (by decide) (by decide)).trans hb
  have hs' := (hr .nat [S] "offs_s" (by decide) (by decide) (by decide)).trans hs
  have ht' := (hr .nat [T] "offs_t" (by decide) (by decide) (by decide)).trans ht
  have hd' := (hr .nat [D] "offs_d" (by decide) (by decide) (by decide)).trans hd
  have hmem' : t.mem = s.mem := hmem.trans hm
  have hmid : ∃ q, stepStmts (((matrixOriginal S T D iters tau).body.drop 13).take 7) t = some q ∧
      q.regs .real [S, T] "z" = some (scale (readMatrix s "h_pre" 0 S T) tau) ∧
      q.regs .real [S] "u" = some (iterates (scale (readMatrix s "h_pre" 0 S T) tau) 0).1 ∧
      q.regs .real [T] "v" = some (iterates (scale (readMatrix s "h_pre" 0 S T) tau) 0).2 ∧
      q.regs .real [S, D] "res_mix" = some (residualFormula
        (readMatrix s "res" (s.pid * (S * D)) S D) (readMatrix s "h_res" 0 S S) tau iters) ∧
      q.regs .real [S, D] "residuals" = some (readMatrix s "res" (s.pid * (S * D)) S D) ∧
      q.regs .nat [] "b" = some (Tile.scalar s.pid) ∧
      q.regs .nat [S] "offs_s" = some (Tile.vec (fun i => i.val)) ∧
      q.regs .nat [T] "offs_t" = some (Tile.vec (fun i => i.val)) ∧
      q.regs .nat [D] "offs_d" = some (Tile.vec (fun i => i.val)) ∧ q.mem = s.mem := by
    simp [matrixOriginal, matrixKernel, ComputeKernel.body, ComputeKernel.toAlgKernel,
      List.drop, List.take, stepStmts, stepStmt, evalOp.eq_def, BlockState.setReg,
      hz', hu', hv', hres', hb', hs', ht', hd', hmem', Option.bind,
      NumericDType.add, NumericDType.mul, NumericDType.div, weights, residualFormula, scale, readMatrix, iterates,
      BlockState.readMem, Region.cast, Tile.bop, Tile.uop, Tile.expandDim, TileShape.dropInsertedIndex, TileShape.insertAxis]
    repeat' first | exact rfl | apply And.intro
  obtain ⟨q, hq, hqz, hqu, hqv, hmix, hqr, hqb, hqs, hqt, hqd, hqm⟩ := hmid
  obtain ⟨u, hloop2, huz, huu, huv, hum, _, hur⟩ :=
    loop_run (scale (readMatrix s "h_pre" 0 S T) tau) iters q hqz hqu hqv
  have hmix' := (hur .real [S, D] "res_mix" (by decide) (by decide) (by decide)).trans hmix
  have hur' := (hur .real [S, D] "residuals" (by decide) (by decide) (by decide)).trans hqr
  have hub := (hur .nat [] "b" (by decide) (by decide) (by decide)).trans hqb
  have hus := (hur .nat [S] "offs_s" (by decide) (by decide) (by decide)).trans hqs
  have hut := (hur .nat [T] "offs_t" (by decide) (by decide) (by decide)).trans hqt
  have hud := (hur .nat [D] "offs_d" (by decide) (by decide) (by decide)).trans hqd
  have hbody : (matrixOriginal S T D iters tau).toAlgKernel.body =
      (matrixOriginal S T D iters tau).body.take 12 ++
        [.forLoop "iter" iters (LogSinkhorn.body S S)] ++
        ((matrixOriginal S T D iters tau).body.drop 13).take 7 ++
        .forLoop "iter" iters (LogSinkhorn.body S T) :: (matrixOriginal S T D iters tau).body.drop 21 := rfl
  change ∃ v, stepStmts _ s = some v ∧ _
  rw [hbody, List.append_assoc, List.append_assoc, stepStmts.append_some ha]
  change ∃ v, stepStmts (.forLoop "iter" iters (LogSinkhorn.body S S) :: _) a = some v ∧ _
  rw [stepStmts.cons_some hloop]
  change ∃ v, stepStmts (((matrixOriginal S T D iters tau).body.drop 13).take 7 ++
    .forLoop "iter" iters (LogSinkhorn.body S T) :: (matrixOriginal S T D iters tau).body.drop 21) t = some v ∧ _
  rw [stepStmts.append_some hq, stepStmts.cons_some hloop2]
  simp [matrixOriginal, matrixKernel, ComputeKernel.body, ComputeKernel.toAlgKernel,
    List.drop, stepStmts, stepStmt, evalOp.eq_def, BlockState.setReg,
    huz, huu, huv, hmix', hur', hub, hus, hut, hud, Option.bind,
    NumericDType.add, NumericDType.mul, weights, residualFormula, branchFormula]
  refine ⟨?_, ?_, ?_⟩
  · intro i j
    rw [scatter_read_other _ _ _ _ _ _ _ (by decide)]
    rw [BlockState.scatter_readback_nd _ _ _ (address_injective _ S D) (i, j, PUnit.unit)]
    rintro ⟨⟩
    rfl
  · intro i j
    rw [BlockState.scatter_readback_nd _ _ _ (address_injective _ T D) (i, j, PUnit.unit)]
    rintro ⟨⟩
    rfl
  · intro r o hR hB
    rw [scatter_frame _ _ _ _ _ r o (by simpa using hB), scatter_frame _ _ _ _ _ r o (by simpa using hR)]
    exact congrFun (congrFun (hum.trans hqm) r) o

#axiomsClean mhc_width_matrix_correct

specification mhc_width_matrix_optimized_correct (S T D iters : Nat) (tau : ℝ) (s : BlockState) :
    Spec.Real (∃ t, exec (matrixOptimized "res" "h_res" "h_pre" "res_mix" "branch_in" S T D iters tau) s = some t ∧
      (∀ i : TileIndex [S, D], t.readMem "res_mix" (s.pid * (S * D) + i.1.val * D + i.2.1.val) =
        ((residualFormula (readMatrix s "res" (s.pid * (S * D)) S D)
          (readMatrix s "h_res" 0 S S) tau iters).data i).unbotD 0) ∧
      (∀ i : TileIndex [T, D], t.readMem "branch_in" (s.pid * (T * D) + i.1.val * D + i.2.1.val) =
        ((branchFormula (readMatrix s "res" (s.pid * (S * D)) S D)
          (readMatrix s "h_pre" 0 S T) tau iters).data i).unbotD 0) ∧
      (∀ (r : RegionName) o,
        (r ≠ "res_mix" ∨ ∀ i : TileIndex [S, D], o ≠ s.pid * (S * D) + i.1.val * D + i.2.1.val) →
        (r ≠ "branch_in" ∨ ∀ i : TileIndex [T, D], o ≠ s.pid * (T * D) + i.1.val * D + i.2.1.val) →
        t.mem r o = s.mem r o)) := by
  obtain ⟨t, ht, post⟩ := mhc_width_matrix_correct S T D iters tau s
  refine ⟨t, ?_, post⟩
  have hmid : exec (matrixMiddle "res" "h_res" "h_pre" "res_mix" "branch_in" S T D iters tau) s = some t :=
    MatrixRewrite.div_mul_rcp ((matrixOriginal S T D iters tau).body.take 9)
      ((matrixOriginal S T D iters tau).body.drop 10) "z" _ tau s t ht
  exact MatrixRewrite.div_mul_rcp
    ((matrixMiddle "res" "h_res" "h_pre" "res_mix" "branch_in" S T D iters tau).body.take 17)
    ((matrixMiddle "res" "h_res" "h_pre" "res_mix" "branch_in" S T D iters tau).body.drop 18)
    "z" _ tau s t hmid

#axiomsClean mhc_width_matrix_optimized_correct

end VeriTile.Bench.Examples.HyperConnectionsWidth.MatrixCorrect
