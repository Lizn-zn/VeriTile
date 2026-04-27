/-
VeriTile.Examples.FusedSiLU

Worked equivalence example: a fused MLP block as a *single kernel* vs
the same computation split across three *separate kernels* with scratch
buffers materializing the intermediates. Both compute, elementwise per
program_id,

    y[i] = residual[i] + (x[i] * gate[i]) * sigmoid(x[i] * gate[i])

The fused version (`fusedSiLUKernel`) does it all in one pass, keeping
intermediates in registers. The unfused pipeline runs three separate
kernels back-to-back, with scratch regions `zReg` and `siluReg` holding
the intermediate tile values:

```
     [siluStepGate]      [siluStepSilu]         [siluStepResidual]
xReg ─┐                                                              ┐
gateReg ┼─→  zReg  ──→ zReg ──→ siluReg  ──→  siluReg ─┐            │
                                                       ├─→ outReg
                                  residualReg ─────────┘            │
```

This is the canonical "fused vs unfused" trade-off in GPU kernel
authoring: fusion saves memory traffic on intermediates at the cost of
register pressure and (sometimes) compositional reuse. Verifying the
two are equivalent shows the fused rewrite preserves observable
semantics.

Pre-conditions on the unfused pipeline include light disjointness:
the scratch regions `zReg`, `siluReg` must be distinct from
`residualReg` (otherwise step 1 / 2 would overwrite the data step 3
needs to read). No other distinctness is needed — under RP1's
named-region model, regions automatically don't alias when their
String names differ.

Status:
* `fusedSiLUKernel` correctness against `fusedSiLUSpec` — closed.
* Multi-kernel pipeline (`siluStepGate` / `siluStepSilu` /
  `siluStepResidual` and `execUnfusedSiLU`) — defined.
* `unfused_silu_correct` and the headline `silu_kernels_refinement`
  — closed. The per-step post-state pattern threads `s.pid`, output
  region values, and frame preservation across `Option.bind`
  boundaries by packaging each step's reduced post-state as an explicit
  `set s_i := ...` term (the `foldl … writeMem … {mem, regs, pid}`
  shape produced by `simp` on the kernel body), then reading off the
  `pid` / output region / preserved region facts via the existing
  `BlockState.scatter_readback` and `BlockState.scatter_preserves_other_region`
  lemmas. The `Option.bind` chain stitches together by `rw [h_step] ;
  simp only` per step, which collapses the outer `match some _` once each
  intermediate `exec _ = some _` is rewritten. This sidesteps the
  `∃ s', P(s')` elaboration issue (where `P` mentions a still-metavariable
  `s'`) by giving `s'` an explicit, well-typed term up front.

Per RP1 (`Notes/research_problem_pointer_vs_named_region.md`) buffers are
named regions, parameterized via the `tl.load($(reg) + offs)` antiquote.
Per RP2 (`Notes/research_problem_address_typing.md`) address arithmetic
stays in `Nat`. Single-block aligned case
(`BLOCK_SIZE = blockSize = problem length`).
-/

import Mathlib.Analysis.SpecialFunctions.Sigmoid
import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.DSL
import VeriTile.Examples.Common

namespace VeriTile.Examples

open VeriTile.Triton

/-! ## (a) Embedded Triton ASTs -/

/-- Fused MLP block, all in one kernel: `y = residual + (x * gate) *
    sigmoid(x * gate)`. -/
def fusedSiLUKernel (xReg gateReg residualReg outReg : RegionName)
    (blockSize : Nat) : Kernel := triton {
  pid      := tl.program_id(0)
  offsets  := pid * $(blockSize) + tl.arange($(blockSize))
  x        := tl.load($(xReg) + offsets)
  gate     := tl.load($(gateReg) + offsets)
  residual := tl.load($(residualReg) + offsets)
  z        := x * gate
  silu     := z * tl.sigmoid(z)
  y        := residual + silu
  tl.store($(outReg) + offsets, y)
}

/-- Step 1 of the unfused pipeline: load `x` and `gate`, compute
    `z = x * gate`, store `z` to scratch region `zReg`. -/
def siluStepGate (xReg gateReg zReg : RegionName) (blockSize : Nat) : Kernel := triton {
  pid     := tl.program_id(0)
  offsets := pid * $(blockSize) + tl.arange($(blockSize))
  x       := tl.load($(xReg) + offsets)
  gate    := tl.load($(gateReg) + offsets)
  z       := x * gate
  tl.store($(zReg) + offsets, z)
}

/-- Step 2 of the unfused pipeline: load `z` from `zReg`, apply SiLU
    (`silu = z * sigmoid(z)`), store to scratch region `siluReg`. -/
def siluStepSilu (zReg siluReg : RegionName) (blockSize : Nat) : Kernel := triton {
  pid     := tl.program_id(0)
  offsets := pid * $(blockSize) + tl.arange($(blockSize))
  z       := tl.load($(zReg) + offsets)
  silu    := z * tl.sigmoid(z)
  tl.store($(siluReg) + offsets, silu)
}

/-- Step 3 of the unfused pipeline: load `silu` from `siluReg` and
    `residual` from `residualReg`, compute `y = residual + silu`, store
    to `outReg`. -/
def siluStepResidual (siluReg residualReg outReg : RegionName)
    (blockSize : Nat) : Kernel := triton {
  pid      := tl.program_id(0)
  offsets  := pid * $(blockSize) + tl.arange($(blockSize))
  silu     := tl.load($(siluReg) + offsets)
  residual := tl.load($(residualReg) + offsets)
  y        := residual + silu
  tl.store($(outReg) + offsets, y)
}

/-- Sequential composition of the three step kernels via `Option.bind`.
    `none` if any step fails. -/
noncomputable def execUnfusedSiLU
    (xReg gateReg residualReg zReg siluReg outReg : RegionName)
    (blockSize : Nat) (s : BlockState) : Option BlockState :=
  match exec (siluStepGate xReg gateReg zReg blockSize) s with
  | none => none
  | some s1 =>
    match exec (siluStepSilu zReg siluReg blockSize) s1 with
    | none => none
    | some s2 => exec (siluStepResidual siluReg residualReg outReg blockSize) s2

/-! ## (b) Math denotation -/

/-- Elementwise spec: `out[i] = residuals[i] + (xs[i] * gates[i]) *
    sigmoid(xs[i] * gates[i])`. Both fused and unfused kernels match this. -/
noncomputable def fusedSiLUSpec {blockSize : Nat}
    (xs gates residuals : Fin blockSize → ℝ) (i : Fin blockSize) : ℝ :=
  residuals i + (xs i * gates i) * Real.sigmoid (xs i * gates i)

/-! ## (c) Per-kernel correctness -/

/-- **`fusedSiLUKernel` correctness against `fusedSiLUSpec`.** Single
    kernel, scatter_readback closes. -/
theorem fused_silu_correct
    (xReg gateReg residualReg outReg : RegionName)
    (blockSize : Nat) (_hN : 0 < blockSize) (s : BlockState)
    (xs gates residuals : Fin blockSize → ℝ)
    (_h_x   : InputLoadedAt s xReg blockSize xs)
    (_h_g   : InputLoadedAt s gateReg blockSize gates)
    (_h_res : InputLoadedAt s residualReg blockSize residuals) :
    ∀ i : Fin blockSize,
      observeAt (exec (fusedSiLUKernel xReg gateReg residualReg outReg blockSize) s)
                outReg blockSize s.pid i
        = some (fusedSiLUSpec xs gates residuals i) := by
  intro i
  have h_inj :
      Function.Injective (fun k : Fin blockSize => s.pid * blockSize + k.val) := by
    intro a b hab
    exact Fin.ext (Nat.add_left_cancel hab)
  simp [observeAt, exec, fusedSiLUKernel, stepStmts, stepStmt, evalOp,
        Value.bop, Value.uop,
        BlockState.setReg, BlockState.readMem, fusedSiLUSpec]
  unfold InputLoadedAt at _h_x _h_g _h_res
  simp_rw [_h_x, _h_g, _h_res]
  exact BlockState.scatter_readback _ _ _ h_inj i

/-- **`execUnfusedSiLU` correctness against `fusedSiLUSpec`.**

    The proof reduces each of the three kernels to an explicit post-state
    (`s1`, `s2`, `s3`), then stitches the `Option`-valued execution chain
    together. `scatter_readback` proves that each step's written region has
    the intended values; `scatter_preserves_other_region` proves that scratch
    writes do not disturb `residualReg`, under the stated disjointness
    assumptions. -/
theorem unfused_silu_correct
    (xReg gateReg residualReg zReg siluReg outReg : RegionName)
    (blockSize : Nat) (_hN : 0 < blockSize) (s : BlockState)
    (xs gates residuals : Fin blockSize → ℝ)
    (_h_x   : InputLoadedAt s xReg blockSize xs)
    (_h_g   : InputLoadedAt s gateReg blockSize gates)
    (_h_res : InputLoadedAt s residualReg blockSize residuals)
    (_h_zRes    : zReg ≠ residualReg)
    (_h_siluRes : siluReg ≠ residualReg) :
    ∀ i : Fin blockSize,
      observeAt
        (execUnfusedSiLU xReg gateReg residualReg zReg siluReg outReg blockSize s)
        outReg blockSize s.pid i
        = some (fusedSiLUSpec xs gates residuals i) := by
  intro i
  have h_inj :
      Function.Injective (fun k : Fin blockSize => s.pid * blockSize + k.val) := by
    intro a b hab
    exact Fin.ext (Nat.add_left_cancel hab)
  -- Local helper: a `foldl` of `writeMem` updates only `mem`; `pid` is preserved.
  have foldl_pid : ∀ {α : Type} (region : RegionName)
      (offs : α → Nat) (vals : α → ℝ) (l : List α) (s' : BlockState),
      (l.foldl (fun acc k => acc.writeMem region (offs k) (vals k)) s').pid = s'.pid := by
    intro α region offs vals l
    induction l with
    | nil => intros; rfl
    | cons hd tl ih =>
      intro s'
      rw [List.foldl_cons, ih]
      rfl
  unfold InputLoadedAt at _h_x _h_g _h_res
  -- Step 1's post-state, written explicitly as a foldl over the final register
  -- file. `simp` reduces `siluStepGate`'s body to this term; we package it as
  -- `s1` so subsequent steps can refer to it by name and so we sidestep the
  -- `∃ s', P s'` elaboration issue (P depending on a metavariable).
  set s1 : BlockState :=
    (List.finRange blockSize).foldl
        (fun acc i => acc.writeMem zReg (s.pid * blockSize + i.val) (xs i * gates i))
        { mem := s.mem,
          regs := fun n =>
            if n = "z" then some (Value.tile blockSize (fun i => xs i * gates i))
            else if n = "gate" then some (Value.tile blockSize (fun i => gates i))
            else if n = "x" then some (Value.tile blockSize (fun i => xs i))
            else if n = "offsets" then
                  some (Value.tileNat blockSize (fun i => s.pid * blockSize + i.val))
            else if n = "pid" then some (Value.scalarNat s.pid) else s.regs n,
          pid := s.pid } with hs1_def
  have h1 : exec (siluStepGate xReg gateReg zReg blockSize) s = some s1 := by
    simp [exec, siluStepGate, stepStmts, stepStmt, evalOp, Value.bop,
          BlockState.setReg, BlockState.readMem]
    simp_rw [_h_x, _h_g]
    rfl
  have h_s1_pid : s1.pid = s.pid := by
    rw [hs1_def]; exact foldl_pid _ _ _ _ _
  have h_s1_zReg : ∀ k : Fin blockSize,
      s1.mem zReg (s1.pid * blockSize + k.val) = xs k * gates k := by
    intro k; rw [h_s1_pid, hs1_def]
    exact BlockState.scatter_readback _ _ _ h_inj k
  have h_s1_other : ∀ R off, R ≠ zReg → s1.mem R off = s.mem R off := by
    intro R off hR
    rw [hs1_def]
    exact BlockState.scatter_preserves_other_region zReg
      (fun k : Fin blockSize => s.pid * blockSize + k.val)
      (fun k => xs k * gates k) R hR off (List.finRange blockSize) _
  -- Step 2's post-state from `s1`. Same packaging trick.
  set s2 : BlockState :=
    (List.finRange blockSize).foldl
        (fun acc i => acc.writeMem siluReg (s1.pid * blockSize + i.val)
            (xs i * gates i * (xs i * gates i).sigmoid))
        { mem := s1.mem,
          regs := fun n =>
            if n = "silu" then some (Value.tile blockSize
                (fun i => xs i * gates i * (xs i * gates i).sigmoid))
            else if n = "z" then some (Value.tile blockSize (fun i => xs i * gates i))
            else if n = "offsets" then
                  some (Value.tileNat blockSize (fun i => s1.pid * blockSize + i.val))
            else if n = "pid" then some (Value.scalarNat s1.pid) else s1.regs n,
          pid := s1.pid } with hs2_def
  have h2 : exec (siluStepSilu zReg siluReg blockSize) s1 = some s2 := by
    simp [exec, siluStepSilu, stepStmts, stepStmt, evalOp, Value.bop, Value.uop,
          BlockState.setReg, BlockState.readMem]
    simp_rw [h_s1_zReg]
    rfl
  have h_s2_pid : s2.pid = s.pid := by
    rw [hs2_def, foldl_pid]; exact h_s1_pid
  have h_s2_siluReg : ∀ k : Fin blockSize,
      s2.mem siluReg (s2.pid * blockSize + k.val) =
        xs k * gates k * (xs k * gates k).sigmoid := by
    intro k
    rw [h_s2_pid, ← h_s1_pid, hs2_def]
    exact BlockState.scatter_readback _ _ _ (by
      intro a b hab
      apply Fin.ext
      exact Nat.add_left_cancel hab) k
  -- residualReg is preserved through both step 1 (zReg ≠ residualReg) and
  -- step 2 (siluReg ≠ residualReg), so it still holds the original `residuals` tile.
  have h_s2_residual : ∀ k : Fin blockSize,
      s2.mem residualReg (s2.pid * blockSize + k.val) = residuals k := by
    intro k
    rw [h_s2_pid, hs2_def]
    rw [BlockState.scatter_preserves_other_region siluReg _ _ residualReg
          (Ne.symm _h_siluRes) _ (List.finRange blockSize)]
    show s1.mem residualReg (s.pid * blockSize + k.val) = residuals k
    rw [h_s1_other residualReg _ (Ne.symm _h_zRes)]
    exact _h_res k
  -- Step 3's post-state from `s2`.
  set s3 : BlockState :=
    (List.finRange blockSize).foldl
        (fun acc i => acc.writeMem outReg (s2.pid * blockSize + i.val)
            (residuals i + xs i * gates i * (xs i * gates i).sigmoid))
        { mem := s2.mem,
          regs := fun n =>
            if n = "y" then some (Value.tile blockSize
                (fun i => residuals i + xs i * gates i * (xs i * gates i).sigmoid))
            else if n = "residual" then some (Value.tile blockSize (fun i => residuals i))
            else if n = "silu" then some (Value.tile blockSize
                (fun i => xs i * gates i * (xs i * gates i).sigmoid))
            else if n = "offsets" then
                  some (Value.tileNat blockSize (fun i => s2.pid * blockSize + i.val))
            else if n = "pid" then some (Value.scalarNat s2.pid) else s2.regs n,
          pid := s2.pid } with hs3_def
  have h3 : exec (siluStepResidual siluReg residualReg outReg blockSize) s2 = some s3 := by
    simp [exec, siluStepResidual, stepStmts, stepStmt, evalOp, Value.bop,
          BlockState.setReg, BlockState.readMem]
    simp_rw [h_s2_siluReg, h_s2_residual]
    rfl
  -- Stitch: chained `match` on `Option` reduces once each `exec` is rewritten.
  have h_total :
      execUnfusedSiLU xReg gateReg residualReg zReg siluReg outReg blockSize s = some s3 := by
    unfold execUnfusedSiLU
    rw [h1]; simp only; rw [h2]; simp only; exact h3
  rw [observeAt, h_total]
  show some _ = some _
  congr 1
  show s3.readMem outReg (s.pid * blockSize + i.val) = fusedSiLUSpec xs gates residuals i
  rw [BlockState.readMem, hs3_def]
  rw [show s.pid * blockSize + i.val = s2.pid * blockSize + i.val from by rw [h_s2_pid]]
  rw [BlockState.scatter_readback _ _ _ (by
        intro a b hab
        apply Fin.ext
        exact Nat.add_left_cancel hab) i]
  rfl

/-! ## (d) Refinement: fused ≡ unfused -/

/-- **`fusedSiLUKernel` and `execUnfusedSiLU` produce the same `outReg`
    memory.** Compose `fused_silu_correct` and `unfused_silu_correct`. -/
theorem silu_kernels_refinement
    (xReg gateReg residualReg zReg siluReg outReg : RegionName)
    (blockSize : Nat) (hN : 0 < blockSize) (s : BlockState)
    (xs gates residuals : Fin blockSize → ℝ)
    (h_x   : InputLoadedAt s xReg blockSize xs)
    (h_g   : InputLoadedAt s gateReg blockSize gates)
    (h_res : InputLoadedAt s residualReg blockSize residuals)
    (h_zRes    : zReg ≠ residualReg)
    (h_siluRes : siluReg ≠ residualReg) :
    ∀ i : Fin blockSize,
      observeAt (exec (fusedSiLUKernel xReg gateReg residualReg outReg blockSize) s)
                outReg blockSize s.pid i =
      observeAt
        (execUnfusedSiLU xReg gateReg residualReg zReg siluReg outReg blockSize s)
        outReg blockSize s.pid i := by
  intro i
  rw [fused_silu_correct
        xReg gateReg residualReg outReg blockSize hN s xs gates residuals h_x h_g h_res i,
      unfused_silu_correct
        xReg gateReg residualReg zReg siluReg outReg blockSize hN s xs gates residuals
        h_x h_g h_res h_zRes h_siluRes i]

end VeriTile.Examples
