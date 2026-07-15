import Mathlib.Analysis.SpecialFunctions.Sigmoid
import VeriTile.Triton
import VeriTile.Examples.Common
import VeriTile.Meta.StatementAudit

/-!
# Fused SiLU ≡ three-kernel pipeline — kernel equivalence `≡[R]` on the IO surface

Self-contained showcase, read top to bottom: **kernels** first (what we
built), the **supporting lemmas** in the middle (`private` plumbing — the
grind), the region-level refinement core next
(`silu_kernels_refinement_view`), then the **flat-memory bridge side
conditions**, the **specification** last (one public headline `silu_equiv` on
the `≡[R]` surface), and a compile-time **trust audit**. The real sections
below are `FusedSiLUEquiv.kernels`, `FusedSiLUEquiv.lemmas`,
`FusedSiLUEquiv.theorems`, `FusedSiLUEquiv.bridge`,
`FusedSiLUEquiv.spec`.

The **boundary-rounding** SiLU story (the exact-ℝ template is
`bench/examples/FusedSiLUReal.lean`), on the `⊨`-grade kernel-equivalence
surface. Only the **output** store is rounded to bf16 (`(y).to(tl.bfloat16)`);
the pipeline's scratch materializations (`z`/`silu`) stay in ℝ and are declared
as private `scratch` buffers of the pipeline's IO signature. Both the fused
kernel and the unfused pipeline compute the **same** per-lane ℝ output
`residual + silu(x·gate)`, so any rounding model `R` quantizes the two equal
values identically at the shared bf16 output store — the writes agree outside
the scratch temporaries.

Contrast: `bench/examples/FusedSwigluEquiv.lean` rounds the materialized
**intermediate** too (bf16 scratch), needing idempotence; here the scratch is
ℝ, so only the boundary store matters and no idempotence is used.

## The public result (bottom of file)

The single public headline is **`silu_equiv`** — kernel equivalence on the
shared three-input IO signature:

    siluFusedIO B ≡[R] siluUnfusedIO B

`≡[R]` is the audit-once kernel-equivalence combinator (`KernelIO₃.Equiv`,
`VeriTile.Triton.Memory.KernelSpec`), the `⊨`-grade form of the refinement
surface. Spelled out, the headline says: for **every** disjoint flat placement
of the interface buffers `x`/`gate`/`residual`/`out` **plus the pipeline's
private staging buffers `z` and `silu`** (∀ base pointers, ∀ buffer sizes),
**every** program id whose windows `[pid * B, pid * B + B)` land inside their
buffers, and **every** launch state — **no input hypotheses at all**, not even
loaded inputs: "equal inputs" is simply "the same `s₀`" — both translated
pointer kernels terminate under `execR R`, their `out` windows hold equal
values, and each kernel leaves every cell outside the output window and its
**own** scratch windows untouched (the fused kernel has no scratch, so it
frames outside `out` alone; the pipeline may additionally write its `z` and
`silu` windows, and `≡` never compares them).

The statement mentions only the two IO signatures and the library equivalence
surface — **no spec** (the `#stmtSurfaceSubset` gate below enforces this).
Everything else — the region-level refinement core
`silu_kernels_refinement_view` (formerly the file's headline, now the theorem
the `≡[R]` region-model obligation `hrun` repackages: its memory agreement
outside `[zReg, siluReg]`, read at the output window, is exactly the
`out`-agreement leg), the frame lemmas, and the flat-memory bridge side
conditions — is scaffolding. The generic unpacker that turns the `Refines`
fact into `hrun`'s per-execution memory agreement is kernel-agnostic and lives
in the library (`ComputeRefine.Refines.out`, `VeriTile.Triton.Float.Refine`).

**Honest boundary**: `unfusedSiLUKernel` is the `ComputeKernel.seq`
*single-launch concatenation* of the three step bodies — register bindings
flow across the stage seams, which no real three-launch execution allows. The
register-leakage gap is bridged once in the library
(`ComputeKernel.execR_seq_rel_execPipelineR`, `VeriTile.Triton.Float.Pipeline`);
this file's headline is about the concatenated kernel.
-/

namespace VeriTile.Bench.Examples.FusedSiLUEquiv

open VeriTile.Triton VeriTile.Examples
open scoped VeriTile.Triton.KernelIO₃

/-! ## Kernels -/
section FusedSiLUEquiv.kernels

/-- Fused SiLU with a bf16-rounded output store. -/
def fusedSiLUKernel (xReg gateReg residualReg outReg : RegionName)
    (blockSize : Nat) : ComputeKernel := triton {
  pid      := tl.program_id(0)
  offsets  := pid * $(blockSize) + tl.arange($(blockSize))
  x        := tl.load($(xReg) + offsets)
  gate     := tl.load($(gateReg) + offsets)
  residual := tl.load($(residualReg) + offsets)
  z        := x * gate
  silu     := z * tl.sigmoid(z)
  y        := residual + silu
  tl.store($(outReg) + offsets, (y).to(tl.bfloat16))
}

/-- Step A: materialize `z = x * gate` into the ℝ scratch tensor `z`. -/
def siluStepGate (xReg gateReg zReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid     := tl.program_id(0)
  offsets := pid * $(blockSize) + tl.arange($(blockSize))
  x       := tl.load($(xReg) + offsets)
  gate    := tl.load($(gateReg) + offsets)
  z       := x * gate
  tl.store($(zReg) + offsets, z)
}

/-- Step B: materialize `silu = z * sigmoid(z)` into the ℝ scratch tensor `silu`. -/
def siluStepSilu (zReg siluReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid     := tl.program_id(0)
  offsets := pid * $(blockSize) + tl.arange($(blockSize))
  z       := tl.load($(zReg) + offsets)
  silu    := z * tl.sigmoid(z)
  tl.store($(siluReg) + offsets, silu)
}

/-- Step C: `out = residual + silu`, with a bf16-rounded output store. -/
def siluStepResidual (siluReg residualReg outReg : RegionName)
    (blockSize : Nat) : ComputeKernel := triton {
  pid      := tl.program_id(0)
  offsets  := pid * $(blockSize) + tl.arange($(blockSize))
  silu     := tl.load($(siluReg) + offsets)
  residual := tl.load($(residualReg) + offsets)
  y        := residual + silu
  tl.store($(outReg) + offsets, (y).to(tl.bfloat16))
}

/-- The unfused pipeline as one kernel: `ComputeKernel.seq` of the three step
kernels — the concatenation of their bodies (registers flow across the seams).
The staged execution is recovered by `execR_unfusedSiLU_split` below. -/
def unfusedSiLUKernel
    (xReg gateReg residualReg zReg siluReg outReg : RegionName)
    (blockSize : Nat) : ComputeKernel :=
  ComputeKernel.seq [xReg, gateReg, residualReg] [outReg]
    [siluStepGate xReg gateReg zReg blockSize,
     siluStepSilu zReg siluReg blockSize,
     siluStepResidual siluReg residualReg outReg blockSize]

end FusedSiLUEquiv.kernels

/- Shared parameters of every lemma and the headline — the pipeline's regions in
data-flow order (`keepReg` is the generic frame region), the block size, the
state, the three per-lane input vectors, and the rounding model `R`. -/
variable (xReg gateReg zReg siluReg residualReg outReg keepReg : RegionName)
variable (blockSize : Nat) (s : BlockState)
variable (xs gates residuals : Fin blockSize → ℝ)
variable (R : RoundingModel)

/-! ## Supporting lemmas (private plumbing) -/
section FusedSiLUEquiv.lemmas

/-- Two `writeMemAsR` scatters over the same offsets agree cell-by-cell when
their per-lane values agree — the rounding-store analogue of
`BlockState.foldl_writeMem_mem_congr`. -/
private theorem foldl_writeMemAsR_mem_congr {α : Type} (R : RoundingModel)
    (dtype : FloatDType) {region : RegionName} (l : List α) (offsetFn : α → Nat)
    (vL vR : α → TileCarrier dtype.toTileDType)
    (hv : ∀ k ∈ l, vL k = vR k) (r : RegionName) (o : Nat) :
    ∀ sL sR : BlockState, sL.mem r o = sR.mem r o →
      (l.foldl (fun acc k => acc.writeMemAsR R dtype region (offsetFn k) (vL k)) sL).mem r o
        = (l.foldl (fun acc k => acc.writeMemAsR R dtype region (offsetFn k) (vR k)) sR).mem r o := by
  induction l with
  | nil => intro sL sR h; exact h
  | cons hd tl ih =>
      intro sL sR h
      refine ih (fun k hk => hv k (List.mem_cons_of_mem _ hk)) _ _ ?_
      rw [BlockState.writeMemAsR_mem, BlockState.writeMemAsR_mem,
          hv hd List.mem_cons_self]
      by_cases hc : r = region ∧ o = offsetFn hd
      · rw [if_pos hc, if_pos hc]
      · rw [if_neg hc, if_neg hc]; exact h

/-- The two ℝ scratch-storing step kernels contain no `castFloat`, so they step
identically under `execR R` and `exec` (state universally quantified so the
degeneration rewrites under the `Option.bind` binders of the pipeline split). -/
private theorem execR_siluStepGate (t : BlockState) :
    execR R (siluStepGate xReg gateReg zReg blockSize) t
      = exec (siluStepGate xReg gateReg zReg blockSize) t := by
  simp [execR, exec, siluStepGate, stepStmtsR, stepStmts, stepStmtR, stepStmt,
        evalOpR.eq_def, Tile.bop, NumericDType.add, NumericDType.mul,
        BlockState.writeMemTypedR, ComputeExpr.toAlgorithm?]

private theorem execR_siluStepSilu (t : BlockState) :
    execR R (siluStepSilu zReg siluReg blockSize) t
      = exec (siluStepSilu zReg siluReg blockSize) t := by
  simp [execR, exec, siluStepSilu, stepStmtsR, stepStmts, stepStmtR, stepStmt,
        evalOpR.eq_def, evalOp, Tile.bop, Tile.uop, NumericDType.add, NumericDType.mul,
        BlockState.writeMemTypedR, ComputeExpr.toAlgorithm?]

/-- The unfused pipeline's `execR` splits into its three stages run in sequence
(the concatenated statement list splits by the library `stepStmtsR_append`);
the two ℝ scratch stages degenerate to `exec` (they carry no `castFloat`), while
the final residual stage keeps its bf16-rounded store under `execR R`. -/
private theorem execR_unfusedSiLU_split :
    execR R (unfusedSiLUKernel xReg gateReg residualReg zReg siluReg outReg blockSize) s =
      (exec (siluStepGate xReg gateReg zReg blockSize) s).bind (fun s1 =>
        (exec (siluStepSilu zReg siluReg blockSize) s1).bind (fun s2 =>
          execR R (siluStepResidual siluReg residualReg outReg blockSize) s2)) := by
  have hsplit :
      execR R (unfusedSiLUKernel xReg gateReg residualReg zReg siluReg outReg blockSize) s =
        (execR R (siluStepGate xReg gateReg zReg blockSize) s).bind (fun s1 =>
          (execR R (siluStepSilu zReg siluReg blockSize) s1).bind (fun s2 =>
            execR R (siluStepResidual siluReg residualReg outReg blockSize) s2)) := by
    simp [unfusedSiLUKernel, execR, stepStmtsR_append]
  rw [hsplit]
  simp only [execR_siluStepGate, execR_siluStepSilu]

/-! ### ℝ scratch-stage facts (reused verbatim from the exact-ℝ template,
`bench/examples/FusedSiLUReal.lean`) -/

private theorem silu_step_gate_correct
    (_h_x : InputLoadedAt s xReg blockSize xs)
    (_h_g : InputLoadedAt s gateReg blockSize gates) :
    ∀ i : Fin blockSize,
      observeAt (exec (siluStepGate xReg gateReg zReg blockSize) s)
                zReg blockSize s.pid i
        = some (xs i * gates i) := by
  intro i
  have h_inj :
      Function.Injective (fun idx : TileIndex [blockSize] => s.pid * blockSize + idx.1.val) :=
    injective_offset_singleton (s.pid * blockSize)
  simp [observeAt, exec, siluStepGate, stepStmts, stepStmt,
        Tile.bop, NumericDType.add, NumericDType.mul]
  unfold InputLoadedAt at _h_x _h_g
  rw [BlockState.scatter_readback_nd _ _ _ h_inj (i, PUnit.unit)]
  simp [_h_x, _h_g]

private theorem silu_step_gate_preserves
    (_h_x : InputLoadedAt s xReg blockSize xs)
    (_h_g : InputLoadedAt s gateReg blockSize gates)
    (h_keep : keepReg ≠ zReg) :
    ∀ i : Fin blockSize,
      observeAt (exec (siluStepGate xReg gateReg zReg blockSize) s)
                keepReg blockSize s.pid i
        = some (s.readMem keepReg (s.pid * blockSize + i.val)) := by
  intro i
  simp [observeAt, exec, siluStepGate, stepStmts, stepStmt,
        Tile.bop, NumericDType.add, NumericDType.mul]
  unfold InputLoadedAt at _h_x _h_g
  simp_rw [_h_x, _h_g]
  rw [BlockState.scatter_preserves_other_region zReg
        (fun k : TileIndex [blockSize] => s.pid * blockSize + k.1.val)
        (fun k : TileIndex [blockSize] => xs k.1 * gates k.1)
        keepReg h_keep (s.pid * blockSize + i.val)]
  simp [BlockState.pid_eq]

private theorem silu_step_silu_correct (zs : Fin blockSize → ℝ)
    (_h_z : InputLoadedAt s zReg blockSize zs) :
    ∀ i : Fin blockSize,
      observeAt (exec (siluStepSilu zReg siluReg blockSize) s)
                siluReg blockSize s.pid i
        = some (Triton.TiledActivation.silu (zs i)) := by
  intro i
  have h_inj :
      Function.Injective (fun idx : TileIndex [blockSize] => s.pid * blockSize + idx.1.val) :=
    injective_offset_singleton (s.pid * blockSize)
  simp [observeAt, exec, siluStepSilu, stepStmts, stepStmt, evalOp,
        Tile.bop, Tile.uop, NumericDType.add, NumericDType.mul,
        Triton.TiledActivation.silu]
  unfold InputLoadedAt at _h_z
  rw [BlockState.scatter_readback_nd _ _ _ h_inj (i, PUnit.unit)]
  simp [_h_z]

private theorem silu_step_silu_preserves (zs : Fin blockSize → ℝ)
    (_h_z : InputLoadedAt s zReg blockSize zs)
    (h_keep : keepReg ≠ siluReg) :
    ∀ i : Fin blockSize,
      observeAt (exec (siluStepSilu zReg siluReg blockSize) s)
                keepReg blockSize s.pid i
        = some (s.readMem keepReg (s.pid * blockSize + i.val)) := by
  intro i
  simp [observeAt, exec, siluStepSilu, stepStmts, stepStmt, evalOp,
        Tile.bop, Tile.uop, NumericDType.add, NumericDType.mul]
  unfold InputLoadedAt at _h_z
  simp_rw [_h_z]
  rw [BlockState.scatter_preserves_other_region siluReg
        (fun k : TileIndex [blockSize] => s.pid * blockSize + k.1.val)
        (fun k : TileIndex [blockSize] => zs k.1 * Real.sigmoid (zs k.1))
        keepReg h_keep (s.pid * blockSize + i.val)]
  simp [BlockState.pid_eq]

private theorem silu_step_gate_mem_frame {s' : BlockState}
    (h : exec (siluStepGate xReg gateReg zReg blockSize) s = some s')
    {r : RegionName} (hr : r ≠ zReg) (o : Nat) :
    s'.mem r o = s.mem r o := by
  simp [exec, siluStepGate, stepStmts, stepStmt, Tile.bop,
        NumericDType.add, NumericDType.mul] at h
  subst h
  exact BlockState.foldl_writeMem_mem_preserve_other_region _ _ _ r hr o _

private theorem silu_step_silu_mem_frame {s' : BlockState}
    (h : exec (siluStepSilu zReg siluReg blockSize) s = some s')
    {r : RegionName} (hr : r ≠ siluReg) (o : Nat) :
    s'.mem r o = s.mem r o := by
  simp [exec, siluStepSilu, stepStmts, stepStmt, evalOp, Tile.bop, Tile.uop,
        NumericDType.add, NumericDType.mul] at h
  subst h
  exact BlockState.foldl_writeMem_mem_preserve_other_region _ _ _ r hr o _

/-! ### bf16-rounded output-store frames (the two stores that carry a `castFloat`) -/

private theorem fusedR_silu_mem_frame {s' : BlockState}
    (h : execR R (fusedSiLUKernel xReg gateReg residualReg outReg blockSize) s = some s')
    {r : RegionName} (hr : r ≠ outReg) (o : Nat) :
    s'.mem r o = s.mem r o := by
  simp [execR, fusedSiLUKernel, stepStmtsR, stepStmtR, evalOpR.eq_def, Tile.bop, Tile.uop,
        NumericDType.add, NumericDType.mul, ComputeExpr.toAlgorithm?] at h
  subst h
  exact BlockState.foldl_writeMemAsR_preserve_cell R .bf16 _ _ r o _
    (fun k _ hc => hr hc.1.symm) _

private theorem siluR_step_residual_mem_frame {s' : BlockState}
    (h : execR R (siluStepResidual siluReg residualReg outReg blockSize) s = some s')
    {r : RegionName} (hr : r ≠ outReg) (o : Nat) :
    s'.mem r o = s.mem r o := by
  simp [execR, siluStepResidual, stepStmtsR, stepStmtR, evalOpR.eq_def, Tile.bop,
        NumericDType.add, NumericDType.mul, ComputeExpr.toAlgorithm?] at h
  subst h
  exact BlockState.foldl_writeMemAsR_preserve_cell R .bf16 _ _ r o _
    (fun k _ hc => hr hc.1.symm) _

/-! ### Unhit-cell frames (the offsets each store does NOT hit)

The `≡[R]` contract cedes only the per-program windows `[pid*B, pid*B + B)` of
`out` (both kernels) and of the pipeline's scratch `z`/`silu` to the writer —
same-region cells outside the window must be framed too. These are the
per-cell siblings of the whole-region frames above, via the library unmasked
scatter cell frames: `BlockState.foldl_writeMem_mem_preserve_unhit` for the
exact ℝ stores, `BlockState.foldl_writeMemAsR_preserve_cell` for the
bf16-rounded ones. -/

/-- An unmasked `writeMemTypedR` scatter `foldl` leaves the program ids
untouched (needed by the pipeline's `TraceSafeR` walk to carry the lane
addressing across the stage seams). Kept local: the library's pid frames
(`BlockState.writeMemAsR_pids`, `BlockState.foldl_writeMemAsR_masked_pids`,
`foldl_store_pids`) are stated for `writeMemAsR`/`writeMem` at a fixed
`FloatDType` and for *masked* folds — none matches the unmasked
`writeMemTypedR` fold over a generic `TileDType` that the concatenated
pipeline's safety walk produces (proved here by one `cases dtype`). -/
private theorem foldl_writeMemTypedR_pids {α : Type} (R : RoundingModel)
    (dtype : TileDType) {region : RegionName}
    (offsetFn : α → Nat) (vf : α → TileCarrier dtype) (l : List α) :
    ∀ s0 : BlockState,
      (l.foldl (fun acc k =>
        BlockState.writeMemTypedR R acc dtype region (offsetFn k) (vf k)) s0).pids
        = s0.pids := by
  induction l with
  | nil => intro s0; rfl
  | cons hd tl ih =>
      intro s0
      rw [List.foldl_cons, ih]
      cases dtype <;> simp [BlockState.writeMemTypedR]

/-- Cell-level frame for step A's exact `z` store: `zReg` offsets not hit by a
lane of the program window are untouched. -/
private theorem silu_step_gate_unhit {s' : BlockState} (o : Nat)
    (ho : ∀ i : Fin blockSize, s.pid * blockSize + i.val ≠ o)
    (h : exec (siluStepGate xReg gateReg zReg blockSize) s = some s') :
    s'.mem zReg o = s.mem zReg o := by
  simp only [BlockState.pid_eq] at ho
  simp [exec, siluStepGate, stepStmts, stepStmt, Tile.bop,
        NumericDType.add, NumericDType.mul] at h
  subst h
  exact BlockState.foldl_writeMem_mem_preserve_unhit _ _ _ o
    (fun k _ => ho k.1) _

/-- Cell-level frame for step B's exact `silu` store. -/
private theorem silu_step_silu_unhit {s' : BlockState} (o : Nat)
    (ho : ∀ i : Fin blockSize, s.pid * blockSize + i.val ≠ o)
    (h : exec (siluStepSilu zReg siluReg blockSize) s = some s') :
    s'.mem siluReg o = s.mem siluReg o := by
  simp only [BlockState.pid_eq] at ho
  simp [exec, siluStepSilu, stepStmts, stepStmt, evalOp, Tile.bop, Tile.uop,
        NumericDType.add, NumericDType.mul] at h
  subst h
  exact BlockState.foldl_writeMem_mem_preserve_unhit _ _ _ o
    (fun k _ => ho k.1) _

/-- Cell-level frame for the fused kernel's bf16-rounded `out` store. -/
private theorem fusedR_silu_unhit {s' : BlockState} (o : Nat)
    (ho : ∀ i : Fin blockSize, s.pid * blockSize + i.val ≠ o)
    (h : execR R (fusedSiLUKernel xReg gateReg residualReg outReg blockSize) s = some s') :
    s'.mem outReg o = s.mem outReg o := by
  simp only [BlockState.pid_eq] at ho
  simp [execR, fusedSiLUKernel, stepStmtsR, stepStmtR, evalOpR.eq_def, Tile.bop, Tile.uop,
        NumericDType.add, NumericDType.mul, ComputeExpr.toAlgorithm?] at h
  subst h
  exact BlockState.foldl_writeMemAsR_preserve_cell R .bf16 _ _ _ o _
    (fun k _ hc => ho k.1 hc.2) _

/-- Cell-level frame for step C's bf16-rounded `out` store. -/
private theorem siluR_step_residual_unhit {s' : BlockState} (o : Nat)
    (ho : ∀ i : Fin blockSize, s.pid * blockSize + i.val ≠ o)
    (h : execR R (siluStepResidual siluReg residualReg outReg blockSize) s = some s') :
    s'.mem outReg o = s.mem outReg o := by
  simp only [BlockState.pid_eq] at ho
  simp [execR, siluStepResidual, stepStmtsR, stepStmtR, evalOpR.eq_def, Tile.bop,
        NumericDType.add, NumericDType.mul, ComputeExpr.toAlgorithm?] at h
  subst h
  exact BlockState.foldl_writeMemAsR_preserve_cell R .bf16 _ _ _ o _
    (fun k _ hc => ho k.1 hc.2) _

/-! ### Termination (unconditional, for the `≡[R]` contract)

`≡[R]`'s region-model obligation starts from **any** state — no loaded inputs
— so termination cannot come from the realization plumbing (which takes
`ViewsLoaded`). It holds outright: all four bodies are straight-line statement
lists whose every step is total, so `execR R` (resp. `exec` for the ℝ scratch
stages) computes to `some _` by the same computational unfold the frames use. -/

/-- The fused kernel terminates under `execR R` from every state. -/
private theorem fusedSiLU_execR_isSome :
    ∃ s', execR R ((fusedSiLUKernel xReg gateReg residualReg outReg blockSize).toAlgKernel) s
      = some s' := by
  simp [execR, fusedSiLUKernel, stepStmtsR, stepStmtR, evalOpR.eq_def, Tile.bop, Tile.uop,
        NumericDType.add, NumericDType.mul, ComputeExpr.toAlgorithm?]

/-- Step A terminates under `exec` from every state. -/
private theorem silu_step_gate_exec_isSome :
    ∃ s', exec (siluStepGate xReg gateReg zReg blockSize) s = some s' := by
  simp [exec, siluStepGate, stepStmts, stepStmt, Tile.bop,
        NumericDType.add, NumericDType.mul]

/-- Step B terminates under `exec` from every state. -/
private theorem silu_step_silu_exec_isSome :
    ∃ s', exec (siluStepSilu zReg siluReg blockSize) s = some s' := by
  simp [exec, siluStepSilu, stepStmts, stepStmt, evalOp, Tile.bop, Tile.uop,
        NumericDType.add, NumericDType.mul]

/-- Step C terminates under `execR R` from every state. -/
private theorem siluR_step_residual_execR_isSome :
    ∃ s', execR R (siluStepResidual siluReg residualReg outReg blockSize) s = some s' := by
  simp [execR, siluStepResidual, stepStmtsR, stepStmtR, evalOpR.eq_def, Tile.bop,
        NumericDType.add, NumericDType.mul, ComputeExpr.toAlgorithm?]

end FusedSiLUEquiv.lemmas

/-! ## The region-level refinement core -/
section FusedSiLUEquiv.theorems

set_option maxHeartbeats 1600000 in
/-- **fused refines pipeline** (`ComputeRefine.Refines R`): for the rounding
model `R`, from the same initial state the fused kernel and the unfused pipeline
perform the same writes — the two final memories agree at every cell outside the
declared scratch regions `zReg`/`siluReg`. Both compute the same per-lane ℝ
output `residual + silu(x·gate)` and round it at the shared bf16 output store, so
`R` quantizes equal values equally.

Formerly the file's headline; now the mathematical core the `≡[R]`
specification's region-model obligation repackages (its `out`-agreement leg is
exactly this memory agreement read at the output window). -/
theorem silu_kernels_refinement_view
    (hin : TensorView.ViewsLoaded s
      [TensorView.slot (programTileView s xReg blockSize) xs,
       TensorView.slot (programTileView s gateReg blockSize) gates,
       TensorView.slot (programTileView s residualReg blockSize) residuals])
    (hscratch : residualReg ∉ [zReg, siluReg]) :
    ComputeRefine.Refines R
      (fusedSiLUKernel xReg gateReg residualReg outReg blockSize)
      (unfusedSiLUKernel xReg gateReg residualReg zReg siluReg outReg blockSize)
      s [zReg, siluReg] := by
  obtain ⟨h_x', h_g', h_res', -⟩ := hin
  have h_x := inputLoadedAt_of_programTileView_loaded h_x'
  have h_g := inputLoadedAt_of_programTileView_loaded h_g'
  have h_res := inputLoadedAt_of_programTileView_loaded h_res'
  simp only [List.mem_cons, List.not_mem_nil, or_false, not_or] at hscratch
  have h_zRes : zReg ≠ residualReg := fun h => hscratch.1 h.symm
  have h_siluRes : siluReg ≠ residualReg := fun h => hscratch.2 h.symm
  have hx := h_x
  have hg := h_g
  have hres := h_res
  apply ComputeKernel.computeRefineR_of_toAlgKernel rfl rfl
  intro s0 lhs' rhs' hL hR hs0
  subst s0
  intro r hr o
  simp only [List.mem_cons, List.not_mem_nil, or_false, not_or] at hr
  obtain ⟨hrz, hrsilu⟩ := hr
  have hRsplit := hR
  rw [execR_unfusedSiLU_split] at hRsplit
  cases h1 : exec (siluStepGate xReg gateReg zReg blockSize) s with
  | none => simp [h1] at hRsplit
  | some s1 =>
    simp only [h1, Option.bind_some] at hRsplit
    cases h2 : exec (siluStepSilu zReg siluReg blockSize) s1 with
    | none => simp [h2] at hRsplit
    | some s2 =>
      simp only [h2, Option.bind_some] at hRsplit
      -- hRsplit : execR R (siluStepResidual …) s2 = some rhs'
      have hpid1 : s1.pid = s.pid := exec_pid h1
      have hpid2 : s2.pid = s1.pid := exec_pid h2
      have hpid2s : s2.pid = s.pid := hpid2.trans hpid1
      by_cases hrOut : r = outReg
      · -- both sides scatter equal bf16-rounded values to the same `outReg` offsets
        subst hrOut
        have h_z : InputLoadedAt s1 zReg blockSize (fun j => xs j * gates j) := by
          intro j
          have hzj :=
            silu_step_gate_correct xReg gateReg zReg blockSize s xs gates hx hg j
          rw [h1] at hzj
          simp [observeAt] at hzj
          rw [hpid1]
          exact hzj
        have h_silu_loaded :
            InputLoadedAt s2 siluReg blockSize ((fun j => Triton.TiledActivation.silu (xs j * gates j))) := by
          intro j
          have hsj :=
            silu_step_silu_correct zReg siluReg blockSize s1 (fun j => xs j * gates j) h_z j
          rw [h2] at hsj
          simp [observeAt] at hsj
          rw [hpid2]
          exact hsj
        have h_res1 : InputLoadedAt s1 residualReg blockSize residuals := by
          intro j
          have hrj :=
            silu_step_gate_preserves xReg gateReg zReg residualReg blockSize s
              xs gates hx hg (fun h => h_zRes h.symm) j
          rw [h1] at hrj
          simp [observeAt] at hrj
          rw [hpid1, hrj]
          exact hres j
        have h_res2 : InputLoadedAt s2 residualReg blockSize residuals := by
          intro j
          have hrj :=
            silu_step_silu_preserves zReg siluReg residualReg blockSize s1
              (fun j => xs j * gates j) h_z (fun h => h_siluRes h.symm) j
          rw [h2] at hrj
          simp [observeAt] at hrj
          rw [hpid2, hrj]
          exact h_res1 j
        -- symbolic execution of the two rounded stores
        simp [execR, fusedSiLUKernel, stepStmtsR, stepStmtR, evalOpR.eq_def, Tile.bop, Tile.uop,
              NumericDType.add, NumericDType.mul, ComputeExpr.toAlgorithm?] at hL
        simp [execR, siluStepResidual, stepStmtsR, stepStmtR, evalOpR.eq_def, Tile.bop,
              NumericDType.add, NumericDType.mul, ComputeExpr.toAlgorithm?] at hRsplit
        subst hL
        subst hRsplit
        have hpids : s2.pids 0 = s.pids 0 := hpid2s
        rw [hpids]
        refine foldl_writeMemAsR_mem_congr R .bf16 _ _ _ _ ?_ r o _ _ ?_
        · -- per-lane written values agree (equal ℝ output, rounded identically)
          intro k _
          refine congrArg (fun v : ℝ => R.cast FloatDType.real FloatDType.bf16 (some v)) ?_
          have hxk := hx k.1
          have hgk := hg k.1
          have hresk := hres k.1
          have hsiluk := h_silu_loaded k.1
          have hres2k := h_res2 k.1
          rw [hpid2s] at hsiluk hres2k
          simp [hxk, hgk, hresk, hsiluk, hres2k,
                Triton.TiledActivation.silu]
        · -- the two scatter bases agree at the observed cell
          simp only [BlockState.setReg_mem]
          rw [silu_step_silu_mem_frame zReg siluReg blockSize s1 h2 hrsilu o,
              silu_step_gate_mem_frame xReg gateReg zReg blockSize s h1 hrz o]
      · -- `r` is none of `zReg`/`siluReg`/`outReg`: both sides preserve the cell
        rw [fusedR_silu_mem_frame xReg gateReg residualReg outReg blockSize s R hL hrOut o,
            siluR_step_residual_mem_frame siluReg residualReg outReg blockSize s2 R hRsplit hrOut o,
            silu_step_silu_mem_frame zReg siluReg blockSize s1 h2 hrsilu o,
            silu_step_gate_mem_frame xReg gateReg zReg blockSize s h1 hrz o]

end FusedSiLUEquiv.theorems

/-! ## Flat-memory bridge side conditions

Both kernels are register-indirect (`offsets := pid * B + tl.arange(B);
tl.load(xReg + offsets)`), so no ∀-state safety contract covers them; the
flat-memory bridge (v1.2, `execR` flavor) takes the per-execution
`Kernel.TraceSafeR` contract instead, plus `FlattenOk` (bridge fragment
membership). Every access on both sides is **unmasked** over the whole
per-program window, so the bounds obligations are whole-window: each buffer
must admit `[pid * B, pid * B + B)`. The fused kernel makes 4 accesses (loads
`x`/`gate`/`residual`, store `out`); the unfused pipeline — the concatenation
of the three step bodies — makes 8 (loads `x`/`gate`, store `z`, load `z`,
store `silu`, loads `silu`/`residual`, store `out`), and its `z`/`silu` bounds
come from the `≡[R]` scratch-window hypothesis. -/
section FusedSiLUEquiv.bridge

/-- The fused kernel sits inside the bridge's covered fragment. -/
theorem fusedSiLU_flattenOk :
    ((fusedSiLUKernel xReg gateReg residualReg outReg blockSize).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [fusedSiLUKernel, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

/-- The concatenated unfused pipeline sits inside the bridge's covered
fragment (concatenation adds no new statement forms). -/
theorem unfusedSiLU_flattenOk :
    ((unfusedSiLUKernel xReg gateReg residualReg zReg siluReg outReg blockSize).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [unfusedSiLUKernel, siluStepGate, siluStepSilu, siluStepResidual,
    ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

set_option maxHeartbeats 1600000 in
/-- Per-execution safety walk for the fused kernel under `R`: one
computational unfold walks all nine statements, discharging the five
memory-silent ones (`program_id`, the `arange` offset arithmetic, and the
`z`/`silu`/`y` register computes) and reducing the four unmasked accesses'
(loads `x`/`gate`/`residual`, store `out`) address obligations to the
whole-window bounds hypotheses. -/
theorem fusedSiLU_traceSafeR (bounds : RegionBounds)
    (hx : s.pid * blockSize + blockSize ≤ bounds xReg)
    (hg : s.pid * blockSize + blockSize ≤ bounds gateReg)
    (hres : s.pid * blockSize + blockSize ≤ bounds residualReg)
    (hout : s.pid * blockSize + blockSize ≤ bounds outReg) :
    Kernel.TraceSafeR R bounds
      ((fusedSiLUKernel xReg gateReg residualReg outReg blockSize).toAlgKernel) s := by
  unfold Kernel.TraceSafeR
  simp only [BlockState.pid_eq] at hx hg hres hout
  simp [fusedSiLUKernel, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
    Stmt.TraceSafeListR, Stmt.TraceSafeR, Op.SafeAtR.eq_def,
    MaskOpt.SafeAtR, MemAccess.SafeAtR, stepStmtR, evalOpR.eq_def,
    tile_elementwise,
    MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR,
    MaskOpt.ActiveR, BlockState.setReg,
    Tile.bop, Tile.uop, NumericDType.add, NumericDType.mul]
  refine ⟨fun a => ?_, fun a => ?_, fun a => ?_, fun a => ?_⟩ <;>
    exact Nat.lt_of_lt_of_le (Nat.add_lt_add_left a.isLt _) (by assumption)

set_option maxHeartbeats 1600000 in
/-- Per-execution safety walk for the concatenated unfused pipeline under
`R`: same computational unfold over the seventeen concatenated statements
(nine of them memory-silent: each stage's `program_id` and `arange` offset
arithmetic, plus the `z`/`silu`/`y` register computes); the eight unmasked
accesses (loads `x`/`gate`, store `z`, load `z`, store `silu`, loads
`silu`/`residual`, store `out`) all address the shared per-program window, and
the `z`/`silu` bounds come from the `≡[R]` scratch-window hypothesis. -/
theorem unfusedSiLU_traceSafeR (bounds : RegionBounds)
    (hx : s.pid * blockSize + blockSize ≤ bounds xReg)
    (hg : s.pid * blockSize + blockSize ≤ bounds gateReg)
    (hres : s.pid * blockSize + blockSize ≤ bounds residualReg)
    (hout : s.pid * blockSize + blockSize ≤ bounds outReg)
    (hz : s.pid * blockSize + blockSize ≤ bounds zReg)
    (hsilu : s.pid * blockSize + blockSize ≤ bounds siluReg) :
    Kernel.TraceSafeR R bounds
      ((unfusedSiLUKernel xReg gateReg residualReg zReg siluReg outReg blockSize).toAlgKernel) s := by
  unfold Kernel.TraceSafeR
  simp only [BlockState.pid_eq] at hx hg hres hout hz hsilu
  simp [unfusedSiLUKernel, siluStepGate, siluStepSilu, siluStepResidual,
    ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
    Stmt.TraceSafeListR, Stmt.TraceSafeR, Op.SafeAtR.eq_def,
    MaskOpt.SafeAtR, MemAccess.SafeAtR, stepStmtR, evalOpR.eq_def,
    tile_elementwise,
    MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR,
    MaskOpt.ActiveR, BlockState.setReg,
    foldl_writeMemTypedR_pids,
    Tile.bop, Tile.uop, NumericDType.add, NumericDType.mul]
  -- (six obligations, not eight: `simp` merges the z-store/z-load and
  -- silu-store/silu-load windows' identical whole-window bounds conjuncts)
  refine ⟨fun a => ?_, fun a => ?_, fun a => ?_, fun a => ?_,
    fun a => ?_, fun a => ?_⟩ <;>
    exact Nat.lt_of_lt_of_le (Nat.add_lt_add_left a.isLt _) (by assumption)

end FusedSiLUEquiv.bridge

/-! ## The spec: `siluFusedIO ≡[R] siluUnfusedIO` -/
section FusedSiLUEquiv.spec

/-- The unfused pipeline's **IO signature** — the whole kernel-specific audit
surface of the headline: interface `x`, `gate`, `residual` → `out`, each
program owning the window `[pid * B, pid * B + B)`, **plus its two private
staging buffers `z` and `silu`** (same per-program window), declared as
`scratch`: the pipeline may write its `z`/`silu` windows, but their post-state
is no part of the contract — `≡` never compares them. -/
def siluUnfusedIO (B : Nat) : KernelIO₃ where
  kernel := unfusedSiLUKernel ⟨"x"⟩ ⟨"gate"⟩ ⟨"residual"⟩ ⟨"z"⟩ ⟨"silu"⟩ ⟨"out"⟩ B
  in1 := ⟨"x"⟩
  in2 := ⟨"gate"⟩
  in3 := ⟨"residual"⟩
  out := ⟨"out"⟩
  B1 := B
  B2 := B
  B3 := B
  Bout := B
  read1 := fun pid => pid * B
  read2 := fun pid => pid * B
  read3 := fun pid => pid * B
  write := fun pid => pid * B
  scratch := [⟨⟨"z"⟩, fun pid => pid * B, B⟩, ⟨⟨"silu"⟩, fun pid => pid * B, B⟩]

/-- The fused kernel's IO signature: the **same interface by construction**
(structure update of `siluUnfusedIO` — buffers and windows shared verbatim),
with the fused kernel plugged in and **no scratch** (it stages nothing
through memory). -/
def siluFusedIO (B : Nat) : KernelIO₃ :=
  { siluUnfusedIO B with
    kernel := fusedSiLUKernel ⟨"x"⟩ ⟨"gate"⟩ ⟨"residual"⟩ ⟨"out"⟩ B
    scratch := [] }

/-- **The headline**: fused SiLU is equivalent to the unfused three-step
pipeline on their shared three-input IO signature, for **every** rounding
model — see the module docstring for the full contract `≡[R]` unfolds to
(both kernels terminate from any state, `out` windows agree, each frames
outside `out` ∪ its own scratch). Proof: `KernelIO₃.Equiv.intro` assembles
the region-model equivalence — termination is unconditional, the
`out`-agreement leg is the region-level refinement theorem's memory agreement
(equal ℝ outputs, rounded identically at the boundary store), the frames are
the unmasked scatter frame lemmas — with the flat-memory bridge side
conditions (`FlattenOk` + `TraceSafeR` per kernel). -/
specification silu_equiv (R : RoundingModel) (B : Nat) :
    siluFusedIO B ≡[R] siluUnfusedIO B := by
  refine KernelIO₃.Equiv.intro _ _ ?_ ?_ ?_ ?_ ?_
  · -- FlattenOk, fused
    exact fusedSiLU_flattenOk ⟨"x"⟩ ⟨"gate"⟩ ⟨"residual"⟩ ⟨"out"⟩ B
  · -- FlattenOk, unfused
    exact unfusedSiLU_flattenOk ⟨"x"⟩ ⟨"gate"⟩ ⟨"z"⟩ ⟨"silu"⟩ ⟨"residual"⟩ ⟨"out"⟩ B
  · -- TraceSafeR, fused (its accesses need no scratch bounds)
    intro bounds t h1 h2 h3 h4 _
    simp only [siluFusedIO, siluUnfusedIO] at h1 h2 h3 h4 ⊢
    exact fusedSiLU_traceSafeR ⟨"x"⟩ ⟨"gate"⟩ ⟨"residual"⟩ ⟨"out"⟩ B t R bounds
      h1 h2 h3 h4
  · -- TraceSafeR, unfused (z/silu bounds from the scratch-window hypothesis)
    intro bounds t h1 h2 h3 h4 hsc
    simp only [siluFusedIO, siluUnfusedIO] at h1 h2 h3 h4 hsc ⊢
    exact unfusedSiLU_traceSafeR ⟨"x"⟩ ⟨"gate"⟩ ⟨"z"⟩ ⟨"silu"⟩ ⟨"residual"⟩
      ⟨"out"⟩ B t R bounds h1 h2 h3 h4
      (hsc ⟨⟨"z"⟩, fun pid => pid * B, B⟩ (by simp))
      (hsc ⟨⟨"silu"⟩, fun pid => pid * B, B⟩ (by simp))
  · -- the region-model equivalence, from ANY state s₀ (no input hypotheses)
    intro s₀
    simp only [siluFusedIO, siluUnfusedIO]
    -- termination of both sides
    obtain ⟨s1, hexec1⟩ :=
      fusedSiLU_execR_isSome ⟨"x"⟩ ⟨"gate"⟩ ⟨"residual"⟩ ⟨"out"⟩ B s₀ R
    obtain ⟨sA, hexecA⟩ := silu_step_gate_exec_isSome ⟨"x"⟩ ⟨"gate"⟩ ⟨"z"⟩ B s₀
    obtain ⟨sB, hexecB⟩ := silu_step_silu_exec_isSome ⟨"z"⟩ ⟨"silu"⟩ B sA
    obtain ⟨s2, hexecC⟩ :=
      siluR_step_residual_execR_isSome ⟨"silu"⟩ ⟨"residual"⟩ ⟨"out"⟩ B sB R
    have hexec2 : execR R
        ((unfusedSiLUKernel ⟨"x"⟩ ⟨"gate"⟩ ⟨"residual"⟩ ⟨"z"⟩ ⟨"silu"⟩ ⟨"out"⟩ B).toAlgKernel) s₀
        = some s2 := by
      rw [execR_unfusedSiLU_split]
      simp only [hexecA, hexecB, Option.bind_some]
      exact hexecC
    have hpidA : sA.pid = s₀.pid := exec_pid hexecA
    have hpidB : sB.pid = sA.pid := exec_pid hexecB
    -- the inputs both kernels actually see: whatever s₀ holds (no hypotheses)
    have hin : TensorView.ViewsLoaded s₀
        [TensorView.slot (programTileView s₀ ⟨"x"⟩ B)
          (fun j => s₀.readMem ⟨"x"⟩ (s₀.pid * B + j.val)),
         TensorView.slot (programTileView s₀ ⟨"gate"⟩ B)
          (fun j => s₀.readMem ⟨"gate"⟩ (s₀.pid * B + j.val)),
         TensorView.slot (programTileView s₀ ⟨"residual"⟩ B)
          (fun j => s₀.readMem ⟨"residual"⟩ (s₀.pid * B + j.val))] := by
      refine ⟨?_, ?_, ?_, trivial⟩
      all_goals
        rintro ⟨i, -⟩
        simp [TensorView.offset, programTileView, Offset.strided]
    -- the region-level refinement: memories agree outside z/silu (unpacked at
    -- this execution pair by the library `ComputeRefine.Refines.out`)
    have hmem12 : ∀ r, r ∉ [(⟨"z"⟩ : RegionName), ⟨"silu"⟩] →
        ∀ o, s1.mem r o = s2.mem r o :=
      (silu_kernels_refinement_view ⟨"x"⟩ ⟨"gate"⟩ ⟨"z"⟩ ⟨"silu"⟩
          ⟨"residual"⟩ ⟨"out"⟩ B s₀ _ _ _ R hin (by decide)).out
        rfl rfl hexec1 hexec2
    refine ⟨s1, s2, hexec1, hexec2, ?_, ?_, ?_⟩
    · -- OUT windows agree (out ∉ [z, silu])
      intro j
      have hmem := hmem12 ⟨"out"⟩ (by decide) (s₀.pid * B + j.val)
      unfold BlockState.readMem
      rw [hmem]
    · -- fused frame: writes only the out window (no scratch)
      intro r o hcond _hscr
      by_cases hrOUT : r = ⟨"out"⟩
      · subst hrOUT
        rcases hcond with hne | hno
        · exact absurd rfl hne
        · exact fusedR_silu_unhit ⟨"x"⟩ ⟨"gate"⟩ ⟨"residual"⟩ ⟨"out"⟩ B s₀ R o
            (fun i heq => hno i heq.symm) hexec1
      · exact fusedR_silu_mem_frame ⟨"x"⟩ ⟨"gate"⟩ ⟨"residual"⟩ ⟨"out"⟩ B s₀ R
          hexec1 hrOUT o
    · -- unfused frame: writes only out window ∪ z/silu windows (its scratch)
      intro r o hcond hscr
      have hC : s2.mem r o = sB.mem r o := by
        by_cases hrOUT : r = ⟨"out"⟩
        · subst hrOUT
          rcases hcond with hne | hno
          · exact absurd rfl hne
          · refine siluR_step_residual_unhit ⟨"silu"⟩ ⟨"residual"⟩ ⟨"out"⟩ B sB
              R o (fun i => ?_) hexecC
            rw [hpidB, hpidA]
            exact fun heq => hno i heq.symm
        · exact siluR_step_residual_mem_frame ⟨"silu"⟩ ⟨"residual"⟩ ⟨"out"⟩ B
            sB R hexecC hrOUT o
      have hB : sB.mem r o = sA.mem r o := by
        by_cases hrS : r = ⟨"silu"⟩
        · subst hrS
          refine silu_step_silu_unhit ⟨"z"⟩ ⟨"silu"⟩ B sA o
            (fun i => ?_) hexecB
          rw [hpidA]
          exact fun heq =>
            hscr ⟨⟨"silu"⟩, fun pid => pid * B, B⟩ (by simp) rfl i heq.symm
        · exact silu_step_silu_mem_frame ⟨"z"⟩ ⟨"silu"⟩ B sA hexecB hrS o
      have hA : sA.mem r o = s₀.mem r o := by
        by_cases hrZ : r = ⟨"z"⟩
        · subst hrZ
          exact silu_step_gate_unhit ⟨"x"⟩ ⟨"gate"⟩ ⟨"z"⟩ B s₀ o
            (fun i heq =>
              hscr ⟨⟨"z"⟩, fun pid => pid * B, B⟩ (by simp) rfl i heq.symm)
            hexecA
        · exact silu_step_gate_mem_frame ⟨"x"⟩ ⟨"gate"⟩ ⟨"z"⟩ B s₀ hexecA hrZ o
      rw [hC, hB, hA]

end FusedSiLUEquiv.spec

/-! ## Trust audit (compile-time gate)

These commands re-audit the public result every time the file is elaborated —
if either gate fails (a smuggled axiom / `sorry`, or a foreign constant in the
trusted statement) the file stops compiling. See
`VeriTile.Meta.StatementAudit`. -/

-- (1) No `sorry`, no smuggled axiom — in the region-level core's and the
-- public headline's transitive proofs.
#axiomsClean silu_kernels_refinement_view
#axiomsClean silu_equiv

-- (2) The headline's statement surface is the two IO signatures plus the
-- audit-once kernel-equivalence combinator — NO spec (there are none), no
-- other project constant. If a spec-like definition ever creeps into the
-- statement, this fails.
#stmtSurfaceSubset silu_equiv ⊆
  [siluFusedIO, siluUnfusedIO, VeriTile.Triton.KernelIO₃.Equiv,
   RoundingModel]

-- (There is no `#specNonCircular` gate: the file defines no specs at all, so a
-- self-referential spec is impossible by construction.)

end VeriTile.Bench.Examples.FusedSiLUEquiv
