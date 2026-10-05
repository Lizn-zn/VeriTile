import bench.examples.HyperConnectionsDepth.Kernels
import VeriTile.Triton.Math.Sinkhorn
import VeriTile.Triton.Math.MatrixRewrite

/- The matrix specifications at the end cover symbolic S/T/D and every finite
normalization count, with region-memory outputs and frames. The earlier scalar
KernelIO proofs additionally retain their flat-memory bridge. -/
/-
bench/examples/HyperConnectionsDepth

**Depth-side hyper-connections (mHC)**: fixed-rank, inference-only DSL
surface for the branch-reintegration half of manifold-constrained
hyper-connections. The width-side sibling (`mhcWidthConnectionKernel`) lives
in `bench/examples/HyperConnectionsWidth/Correct.lean` — the old two-kernel
`HyperConnections.lean` was split so that each showcased kernel is
self-contained in its example directory.

This file intentionally does not model the PyTorch module wrapper, dropout,
autograd, or an arbitrary user branch. The branch is split at the DSL
boundary: the width-side kernel prepares the branch input and residual
stream, and `mhcDepthConnectionKernel` consumes a separately supplied branch
output.

Four parts, following the canonical KernelIO showcase
`bench/examples/VectorAdd/Correct.lean`:

1. **The kernel** — `mhcDepthConnectionKernel`; its `S = T = D = 1,
   numIters = 0` match arm is the scalar fixed-rank slice the proof covers
   (the general matrix specifications follow at the end).
2. **Region-model Hoare triple** — value view `mhcDepth_exec_view`,
   single-cell frame `mhcDepth_frame`, and their package
   `mhcDepth_region_run`.
3. **Flat-memory bridge side conditions** — the seven-statement `TraceSafe`
   walk (per-program cells at `pid`, the **shared** scalar logit cell at
   `0`, one single-cell store at `pid`) and `FlattenOk`.
4. **The spec** — the original kernel's `specification`:

       mhc_depth_correctness : mhcDepthIO tau ⊨ fun resMix branchOut hPost _ =>
         resMix 0 + Real.exp (hPost 0 / tau) * branchOut 0

   a three-input (`KernelIO₃`) instance mixing per-program tiles with a
   shared buffer. Spelled out: for every disjoint flat placement of the
   four buffers, every program id whose windows are in bounds, and every
   launch state whose input windows hold `resMix`/`branchOut`/`hPost` —
   everything else arbitrary — the translated pointer kernel terminates,
   `out[pid]` holds `resMix + exp(hPost/τ)·branchOut`, and every other flat
   cell is unchanged.

## Layout convention

For one program id `b`, tensors are stored row-major:

- `resMixReg` / `outReg`: `[B, S, D]`
- `branchOutReg`: `[B, T, D]`
- `hPostReg`: `[T, S]`

`S` is the residual-stream count, `T` is the branch-view count, and `D` is
the feature dimension. `numIters` controls the fixed Sinkhorn-style
log-domain normalization count. In the proven fixed-rank slice every tensor
degenerates to one cell per program (the three per-program cells live at
address `b`, the post logit is the scalar cell at address `0`).

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

namespace VeriTile.Bench.Examples.HyperConnectionsDepth
open VeriTile.Bench.Examples.HyperConnectionsDepth.Kernels

-- Correctness interprets the shared typed source through its real projection.
attribute [local simp] VeriTile.Triton.ComputeExpr.toAlgorithm?
  VeriTile.Triton.ComputeOp.toAlgorithm? VeriTile.Triton.ComputeDType.eraseDType
open VeriTile.Triton
open VeriTile.Triton.KernelIO₃ (Implements)
open scoped VeriTile.Triton.KernelIO₃

/-! ## Part 1 — the kernel -/


/-! ## Part 2 — the region-model Hoare triple

The mathematical core, proved against the region model (named buffers, no
pointer arithmetic): a value view (`mhcDepth_exec_view`), a single-cell frame
(`mhcDepth_frame`), and their package `mhcDepth_region_run` — exactly the
`hrun` obligation of `KernelIO₃.Implements.intro`. -/

/-- Value half: the run terminates and `out[pid]` holds
`resMix + exp(hPost/τ)·branchOut` (in terms of the launch state's own memory
cells). -/
private theorem mhcDepth_exec_view (tau : ℝ) (s : BlockState) :
    (exec ((mhcDepthConnectionKernel ⟨"res_mix"⟩ ⟨"branch_out"⟩ ⟨"h_post"⟩
        ⟨"out"⟩ 1 1 1 0 tau).toAlgKernel) s).map
      (fun s' => s'.readMem ⟨"out"⟩ s.pid)
      = some
        (s.readMem ⟨"res_mix"⟩ s.pid
          + Real.exp (s.readMem ⟨"h_post"⟩ 0 / tau)
            * s.readMem ⟨"branch_out"⟩ s.pid) := by
  simp [mhcDepthConnectionKernel, exec, stepStmts,
    stepStmt, NumericDType.add, NumericDType.mul, NumericDType.div,
    WithBot.realAdd, WithBot.realMul, WithBot.realDiv,
    BlockState.writeMem, BlockState.readMem]

/-- Frame half: every memory cell other than the output cell `out[pid]` is
preserved by the run. -/
private theorem mhcDepth_frame (tau : ℝ) (s s1 : BlockState)
    (hExec : exec ((mhcDepthConnectionKernel ⟨"res_mix"⟩ ⟨"branch_out"⟩
        ⟨"h_post"⟩ ⟨"out"⟩ 1 1 1 0 tau).toAlgKernel) s = some s1)
    (r : RegionName) (o : Nat)
    (hmiss : ¬(r = ⟨"out"⟩ ∧ o = s.pid)) :
    s1.mem r o = s.mem r o := by
  simp [exec, mhcDepthConnectionKernel,
    ComputeKernel.toAlgKernel, stepStmts, stepStmt,
    NumericDType.add, NumericDType.mul, NumericDType.div] at hExec
  subst hExec
  rw [BlockState.writeMem_mem, if_neg hmiss]
  rfl

/-- **The region-model Hoare triple** — termination, the output-cell value,
and frame, from any launch state whose input windows are loaded. This is
what the `⊨` headline transports to flat memory. -/
theorem mhcDepth_region_run (tau : ℝ) (s₀ : BlockState)
    (resMix branchOut hPost : Fin 1 → ℝ)
    (hmix : ∀ j : Fin 1, s₀.readMem ⟨"res_mix"⟩ (s₀.pid + j.val) = resMix j)
    (hbout : ∀ j : Fin 1, s₀.readMem ⟨"branch_out"⟩ (s₀.pid + j.val) = branchOut j)
    (hhpost : ∀ j : Fin 1, s₀.readMem ⟨"h_post"⟩ (0 + j.val) = hPost j) :
    ∃ s1,
      exec ((mhcDepthConnectionKernel ⟨"res_mix"⟩ ⟨"branch_out"⟩ ⟨"h_post"⟩
          ⟨"out"⟩ 1 1 1 0 tau).toAlgKernel) s₀ = some s1
      ∧ (∀ j : Fin 1, s1.readMem ⟨"out"⟩ (s₀.pid + j.val)
          = resMix 0 + Real.exp (hPost 0 / tau) * branchOut 0)
      ∧ (∀ r o,
          (r ≠ ⟨"out"⟩ ∨ ∀ j : Fin 1, o ≠ s₀.pid + j.val) →
          s1.mem r o = s₀.mem r o) := by
  have h0mix : s₀.readMem ⟨"res_mix"⟩ s₀.pid = resMix 0 := by simpa using hmix 0
  have h0bout : s₀.readMem ⟨"branch_out"⟩ s₀.pid = branchOut 0 := by
    simpa using hbout 0
  have h0hpost : s₀.readMem ⟨"h_post"⟩ 0 = hPost 0 := by simpa using hhpost 0
  have hview := mhcDepth_exec_view tau s₀
  cases hsrc : exec ((mhcDepthConnectionKernel ⟨"res_mix"⟩ ⟨"branch_out"⟩
      ⟨"h_post"⟩ ⟨"out"⟩ 1 1 1 0 tau).toAlgKernel) s₀ with
  | none => rw [hsrc] at hview; simp at hview
  | some s1 =>
      rw [hsrc] at hview
      simp only [Option.map_some, Option.some_inj] at hview
      refine ⟨s1, rfl, fun j => ?_, fun r o hcond => ?_⟩
      · have hj : (j : Nat) = 0 := Nat.lt_one_iff.mp j.isLt
        rw [hj, Nat.add_zero, hview, h0mix, h0hpost, h0bout]
      · refine mhcDepth_frame tau s₀ s1 hsrc r o ?_
        rintro ⟨hr, ho⟩
        rcases hcond with hne | hno
        · exact hne hr
        · exact hno 0 (by simpa using ho)

/-! ## Part 3 — flat-memory bridge side conditions

The kernel is register-indirect (the scalar `b` register feeds three of the
four accesses), so no ∀-state contract covers it; the flat-memory bridge
(v1.2) takes the per-execution `Kernel.TraceSafe` contract instead,
discharged below by walking the actual seven-statement execution. All four
accesses are **unmasked** single cells: the two per-program loads and the
store at `pid` (in bounds when `pid + 1 ≤ bounds reg`), and the shared logit
load at the constant cell `0` (in bounds when `0 + 1 ≤ bounds reg`). -/

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

/-- Bounds discharge for the shared scalar-logit load: the one addressed
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
/-- The fixed-rank depth kernel is trace-safe: of its seven statements, four
touch memory — three single-cell loads (`res_mix[pid]`, `branch_out[pid]`,
the shared `h_post[0]`) and one single-cell store (`out[pid]`); the
`program_id`, `exp`/`div`/`mul`, and `add` assigns are memory-silent. -/
theorem mhcDepth_traceSafe (tau : ℝ) (bounds : RegionBounds) (s : BlockState)
    (hmix : s.pid + 1 ≤ bounds ⟨"res_mix"⟩)
    (hbout : s.pid + 1 ≤ bounds ⟨"branch_out"⟩)
    (hhpost : 0 + 1 ≤ bounds ⟨"h_post"⟩)
    (hout : s.pid + 1 ≤ bounds ⟨"out"⟩) :
    Kernel.TraceSafe bounds
      ((mhcDepthConnectionKernel ⟨"res_mix"⟩ ⟨"branch_out"⟩ ⟨"h_post"⟩
        ⟨"out"⟩ 1 1 1 0 tau).toAlgKernel) s := by
  unfold Kernel.TraceSafe
  -- statement 1: b := program_id(0)
  refine Stmt.TraceSafeList.cons_intro (by simp [Stmt.TraceSafe, Op.SafeAt]) ?_
  intro s1 hs1
  obtain ⟨v1, hv1, rfl⟩ := stepStmt_assign_inv hs1
  rw [show evalOp (Op.programId 0) s
      = some (Tile.scalar (dtype := .nat) (s.pids 0)) from by simp] at hv1
  obtain rfl := Option.some_inj.mp hv1
  -- statement 2: res_mix := load(res_mix + b)   (single cell, unmasked)
  refine Stmt.TraceSafeList.cons_intro ?_ ?_
  · simp only [ComputeDType.eraseDType, Stmt.TraceSafe, Op.SafeAt]
    exact ⟨trivial, trivial,
      scalarB_activeAddressSafe bounds (s.pids 0) _ _
        (by simp [BlockState.setReg]) _ hmix⟩
  intro s2 hs2
  obtain ⟨v2, hv2, rfl⟩ := stepStmt_assign_inv hs2
  -- statement 3: branch_out := load(branch_out + b)   (single cell, unmasked)
  refine Stmt.TraceSafeList.cons_intro ?_ ?_
  · simp only [ComputeDType.eraseDType, Stmt.TraceSafe, Op.SafeAt]
    exact ⟨trivial, trivial,
      scalarB_activeAddressSafe bounds (s.pids 0) _ _
        (by simp [BlockState.setReg]) _ hbout⟩
  intro s3 hs3
  obtain ⟨v3, hv3, rfl⟩ := stepStmt_assign_inv hs3
  -- statement 4: h_post := load(h_post)   (shared scalar cell 0)
  refine Stmt.TraceSafeList.cons_intro ?_ ?_
  · simp only [ComputeDType.eraseDType, Stmt.TraceSafe, Op.SafeAt]
    exact ⟨trivial, trivial, sharedScalar_activeAddressSafe bounds _ _ _ hhpost⟩
  intro s4 hs4
  obtain ⟨v4, hv4, rfl⟩ := stepStmt_assign_inv hs4
  -- statement 5: branch_mix := exp(h_post / tau) * branch_out   (register op)
  refine Stmt.TraceSafeList.cons_intro
    (by simp [Stmt.TraceSafe, Op.SafeAt.eq_def]) ?_
  intro s5 hs5
  obtain ⟨v5, hv5, rfl⟩ := stepStmt_assign_inv hs5
  -- statement 6: out := res_mix + branch_mix   (register op)
  refine Stmt.TraceSafeList.cons_intro
    (by simp [Stmt.TraceSafe, Op.SafeAt.eq_def]) ?_
  intro s6 hs6
  obtain ⟨v6, hv6, rfl⟩ := stepStmt_assign_inv hs6
  -- statement 7: store(out + b, out)   (single cell, unmasked)
  refine Stmt.TraceSafeList.cons_intro ?_ (fun _ _ => .nil_intro)
  simp only [Stmt.TraceSafe, MemAccess.SafeAt, MaskOpt.SafeAt]
  refine ⟨by simp [Op.SafeAt], by simp [Op.SafeAt], trivial,
    scalarB_activeAddressSafe bounds (s.pids 0) _ _ ?_ _ hout⟩
  simp [BlockState.setReg]

/-- The fixed-rank depth kernel sits inside the bridge's covered fragment. -/
theorem mhcDepth_flattenOk (tau : ℝ) :
    ((mhcDepthConnectionKernel ⟨"res_mix"⟩ ⟨"branch_out"⟩ ⟨"h_post"⟩
      ⟨"out"⟩ 1 1 1 0 tau).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [mhcDepthConnectionKernel,
    ComputeKernel.toAlgKernel, StmtList.FlattenOk, Stmt.FlattenOk,
    Op.FlattenOk.eq_def]

/-! ## Part 4 — the spec: `mhcDepthIO ⊨` the depth-side reintegration -/

/-- The fixed-rank depth kernel's **IO signature** — the whole
kernel-specific audit surface of the headline: the width-side residual mix
`res_mix`, the external branch output `branch_out`, and the post logit
`h_post` in, the updated residual stream `out` out; all four tile lengths
`1`. Program `pid` reads its own `res_mix` / `branch_out` cells
(`read1 pid = read2 pid = pid`), while **all** programs read the same shared
scalar logit cell (`read3 = fun _ => 0`); each program writes its own output
cell. The windows are declared, not parsed from the kernel: the headline
**proves** the kernel's actual addressing matches them. Buffer sizes are not
signature content.

`@[reducible]` is load-bearing: the headline writes `resMix 0` / `hPost 0` /
`branchOut 0`, and the `OfNat (Fin _)` instances behind those literals only
fire if elaboration can see through this def to the literal tile lengths
`1`. -/
@[reducible] def mhcDepthIO (tau : ℝ) : KernelIO₃ where
  kernel := mhcDepthConnectionKernel ⟨"res_mix"⟩ ⟨"branch_out"⟩ ⟨"h_post"⟩
    ⟨"out"⟩ 1 1 1 0 tau
  in1 := ⟨"res_mix"⟩
  in2 := ⟨"branch_out"⟩
  in3 := ⟨"h_post"⟩
  out := ⟨"out"⟩
  B1 := 1
  B2 := 1
  B3 := 1
  Bout := 1
  read1 := fun pid => pid        -- each program reads its own residual-mix cell
  read2 := fun pid => pid        -- … and its own branch-output cell
  read3 := fun _ => 0            -- ALL programs read the shared scalar logit
  write := fun pid => pid

/-- **The headline**: the fixed-rank depth kernel implements the mHC
depth-side reintegration `resMix + exp(hPost/τ)·branchOut` on its IO
signature; see the module docstring for the full Hoare triple `⊨` unfolds
to. Proof: `Implements.intro` assembles the region-model triple (Part 2)
with the bridge side conditions (Part 3). -/
specification mhc_depth_correctness (tau : ℝ) :
    Spec.Real (mhcDepthIO tau ⊨ fun resMix branchOut hPost _ =>
      resMix 0 + Real.exp (hPost 0 / tau) * branchOut 0) := by
  refine KernelIO₃.Implements.intro _ ?_ ?_ ?_
  · exact mhcDepth_flattenOk tau
  · intro bounds s h1 h2 h3 h4 _
    exact mhcDepth_traceSafe tau bounds s h1 h2 h3 h4
  · intro s₀ resMix branchOut hPost h1 h2 h3
    obtain ⟨s1, hexec, hval, hframe⟩ :=
      mhcDepth_region_run tau s₀ resMix branchOut hPost h1 h2 h3
    -- scratch is empty, so its frame side condition is vacuous
    exact ⟨s1, hexec, hval, fun r o hout _ => hframe r o hout⟩

/-! ## Optimized implementation: real correctness -/

section Optimized
set_option maxHeartbeats 5000000
attribute [local simp] ComputeExpr.toAlgorithm? ComputeOp.toAlgorithm? ComputeDType.eraseDType

-- This equality is derived in the real interpreter; it uses no FP assumptions.
private theorem optimized_exec (tau : ℝ) (s : BlockState) :
    exec (optimizedKernel tau).toAlgKernel s =
      exec (mhcDepthConnectionKernel "res_mix" "branch_out" "h_post" "out" 1 1 1 0 tau).toAlgKernel s := by
  simp [optimizedKernel, mhcDepthConnectionKernel, exec, stepStmts, stepStmt, evalOp.eq_def,
    Region.cast, Tile.bop, Tile.uop, NumericDType.add, NumericDType.mul,
      NumericDType.div, add_comm]

/-- The optimized source retains the original IO windows and memory contract. -/
@[reducible] def optimizedIO (tau : ℝ) : KernelIO₃ :=
  { mhcDepthIO tau with kernel := optimizedKernel tau, projection := by rfl }

/-- The optimized implementation computes the same independent real formula. -/
specification mhc_depth_optimized_correctness (tau : ℝ) :
    Spec.Real (optimizedIO tau ⊨ fun resMix branchOut hPost _ =>
      resMix 0 + Real.exp (hPost 0 / tau) * branchOut 0) := by
  refine KernelIO₃.Implements.intro _
    ?_ ?_ ?_
  · simp [optimizedKernel, Kernel.FlattenOk,
      StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]
  · intro bounds s h1 h2 h3 h4 _
    simpa [optimizedIO, mhcDepthIO, Kernel.TraceSafe, optimizedKernel, mhcDepthConnectionKernel,
      Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def, MaskOpt.SafeAt,
      MemAccess.SafeAt, stepStmt, evalOp.eq_def,
      Region.cast, Tile.bop, Tile.uop, NumericDType.add, NumericDType.mul,
      NumericDType.div, add_comm]
      using mhcDepth_traceSafe tau bounds s h1 h2 h3 h4
  · intro s resMix branchOut hPost h1 h2 h3
    change ∃ s1, exec (optimizedKernel tau).toAlgKernel s = some s1 ∧ _
    rw [optimized_exec]
    obtain ⟨s1, he, hv, hf⟩ := mhcDepth_region_run tau s resMix branchOut hPost h1 h2 h3
    exact ⟨s1, he, hv, fun r o hmiss _ => hf r o hmiss⟩

#axiomsClean mhc_depth_optimized_correctness
#stmtSurfaceSubset mhc_depth_optimized_correctness ⊆
  [Spec.Real, optimizedIO, VeriTile.Triton.KernelIO₃.Implements,
   VeriTile.Triton.KernelIO₃.B1, VeriTile.Triton.KernelIO₃.B2,
   VeriTile.Triton.KernelIO₃.B3, VeriTile.Triton.KernelIO₃.Bout]

end Optimized

/-! ## Trust gates -/

-- No `sorry`, no smuggled axiom, in the headline's transitive proof.
#axiomsClean mhc_depth_correctness

/- The headline's statement surface is the IO signature plus the audit-once
Hoare-triple combinator — no other project constant. -/
#stmtSurfaceSubset mhc_depth_correctness ⊆
  [Spec.Real, mhcDepthIO, VeriTile.Triton.KernelIO₃.Implements,
   VeriTile.Triton.KernelIO₃.B1, VeriTile.Triton.KernelIO₃.B2,
   VeriTile.Triton.KernelIO₃.B3, VeriTile.Triton.KernelIO₃.Bout]

end VeriTile.Bench.Examples.HyperConnectionsDepth


/-! General depth connection: R + Sinkhorn(H / tau)ᵀ B, with a finite,
user-selected number of normalization iterations. All dimensions are symbolic. -/
namespace VeriTile.Bench.Examples.HyperConnectionsDepth.MatrixCorrect
open VeriTile Triton LogSinkhorn
open VeriTile.Bench.Examples.HyperConnectionsDepth.Kernels
set_option maxHeartbeats 3200000

noncomputable def formula (res : Tile .real [S, D]) (branch : Tile .real [T, D])
    (logits : Tile .real [T, S]) (tau : ℝ) (iters : Nat) : Tile .real [S, D] :=
  Tile.bop WithBot.realAdd (.consSame (.consSame .nil)) res
    (Tile.dot [] (Tile.transpose [] (weights (scale logits tau) iters)) branch)

private theorem prefix_run (S T D iters : Nat) (tau : ℝ) (s : BlockState) :
    ∃ a, stepStmts ((matrixOriginal S T D iters tau).body.take 14) s = some a ∧
      a.regs .real [T, S] "z" = some (scale (readMatrix s "h_post" 0 T S) tau) ∧
      a.regs .real [T] "u" = some (iterates (scale (readMatrix s "h_post" 0 T S) tau) 0).1 ∧
      a.regs .real [S] "v" = some (iterates (scale (readMatrix s "h_post" 0 T S) tau) 0).2 ∧
      a.regs .real [S, D] "res_mix" = some (readMatrix s "res_mix" (s.pid * (S * D)) S D) ∧
      a.regs .real [T, D] "branch_out" = some (readMatrix s "branch_out" (s.pid * (T * D)) T D) ∧
      a.regs .nat [] "b" = some (Tile.scalar s.pid) ∧
      a.regs .nat [S] "offs_s" = some (Tile.vec (fun i => i.val)) ∧
      a.regs .nat [D] "offs_d" = some (Tile.vec (fun i => i.val)) ∧
      a.mem = s.mem ∧ a.pids = s.pids := by
  simp [matrixOriginal, matrixKernel, ComputeKernel.body, ComputeKernel.toAlgKernel,
    List.take, stepStmts, stepStmt, evalOp.eq_def, BlockState.setReg,
    NumericDType.add, NumericDType.mul, NumericDType.div, readMatrix, scale, iterates,
    BlockState.readMem, Region.cast, Option.bind, Tile.bop, Tile.expandDim, TileShape.dropInsertedIndex, TileShape.insertAxis]
  exact ⟨rfl, rfl, rfl, rfl, rfl⟩


specification mhc_depth_matrix_correct (S T D iters : Nat) (tau : ℝ) (s : BlockState) :
    Spec.Real (∃ t, exec (matrixOriginal S T D iters tau) s = some t ∧
      (∀ i : TileIndex [S, D], t.readMem "out" (s.pid * (S * D) + i.1.val * D + i.2.1.val) =
        ((formula (readMatrix s "res_mix" (s.pid * (S * D)) S D)
          (readMatrix s "branch_out" (s.pid * (T * D)) T D)
          (readMatrix s "h_post" 0 T S) tau iters).data i).unbotD 0) ∧
      (∀ (r : RegionName) o, (r ≠ "out" ∨ ∀ i : TileIndex [S, D],
        o ≠ s.pid * (S * D) + i.1.val * D + i.2.1.val) → t.mem r o = s.mem r o)) := by
  obtain ⟨a, ha, hz, hu, hv, hres, hbranch, hb, hs, hd, hm, hp⟩ := prefix_run S T D iters tau s
  obtain ⟨t, ht, hz', hu', hv', hmem, hpids, hr⟩ :=
    loop_run (scale (readMatrix s "h_post" 0 T S) tau) iters a hz hu hv
  have hres' := (hr .real [S, D] "res_mix" (by decide) (by decide) (by decide)).trans hres
  have hbranch' := (hr .real [T, D] "branch_out" (by decide) (by decide) (by decide)).trans hbranch
  have hb' := (hr .nat [] "b" (by decide) (by decide) (by decide)).trans hb
  have hs' := (hr .nat [S] "offs_s" (by decide) (by decide) (by decide)).trans hs
  have hd' := (hr .nat [D] "offs_d" (by decide) (by decide) (by decide)).trans hd
  have hbody : (matrixOriginal S T D iters tau).toAlgKernel.body =
      (matrixOriginal S T D iters tau).body.take 14 ++
        .forLoop "iter" iters (LogSinkhorn.body T S) :: (matrixOriginal S T D iters tau).body.drop 15 := rfl
  change ∃ u, stepStmts _ s = some u ∧ _
  rw [hbody, stepStmts.append_some ha, stepStmts.cons_some ht]
  simp [matrixOriginal, matrixKernel, ComputeKernel.body, ComputeKernel.toAlgKernel,
    List.drop, stepStmts, stepStmt, evalOp.eq_def, BlockState.setReg,
    hz', hu', hv', hres', hbranch', hb', hs', hd', Option.bind,
    NumericDType.add, NumericDType.mul, weights, formula]
  refine ⟨?_, ?_⟩
  · intro i j
    rw [BlockState.scatter_readback_nd _ _ _ (address_injective _ S D) (i, j, PUnit.unit)]
    rfl
  · intro r o hmiss
    rcases hmiss with hr | ho
    · exact (BlockState.foldl_writeMem_mem_preserve_other_region _ _ _ r hr o _).trans
        (congrFun (congrFun (hmem.trans hm) r) o)
    · by_cases hro : r = "out"
      · subst r
        exact (BlockState.foldl_writeMem_mem_preserve_unhit _ _ _ o
          (fun i _ he => ho i.1 i.2.1 he.symm) _).trans (congrFun (congrFun (hmem.trans hm) _) o)
      · exact (BlockState.foldl_writeMem_mem_preserve_other_region _ _ _ r hro o _).trans
          (congrFun (congrFun (hmem.trans hm) r) o)

#axiomsClean mhc_depth_matrix_correct

specification mhc_depth_matrix_optimized_correct (S T D iters : Nat) (tau : ℝ) (s : BlockState) :
    Spec.Real (∃ t, exec (matrixOptimized "res_mix" "branch_out" "h_post" "out" S T D iters tau) s = some t ∧
      (∀ i : TileIndex [S, D], t.readMem "out" (s.pid * (S * D) + i.1.val * D + i.2.1.val) =
        ((formula (readMatrix s "res_mix" (s.pid * (S * D)) S D)
          (readMatrix s "branch_out" (s.pid * (T * D)) T D)
          (readMatrix s "h_post" 0 T S) tau iters).data i).unbotD 0) ∧
      (∀ (r : RegionName) o, (r ≠ "out" ∨ ∀ i : TileIndex [S, D],
        o ≠ s.pid * (S * D) + i.1.val * D + i.2.1.val) → t.mem r o = s.mem r o)) := by
  obtain ⟨t, ht, post⟩ := mhc_depth_matrix_correct S T D iters tau s
  refine ⟨t, ?_, post⟩
  exact MatrixRewrite.add_commute
    ((matrixOriginal S T D iters tau).body.take 17)
    ((matrixOriginal S T D iters tau).body.drop 18) "out" _ _ s t ht

#axiomsClean mhc_depth_matrix_optimized_correct

end VeriTile.Bench.Examples.HyperConnectionsDepth.MatrixCorrect
