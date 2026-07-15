import VeriTile.Triton
import VeriTile.Examples.Common
import VeriTile.Meta.StatementAudit

/-!
# SwiGLU fused vs unfused — kernel equivalence `≡[R]` on the masked IO surface

Self-contained showcase, read top to bottom: **kernels** first (what we
built), the **supporting lemmas** in the middle (`private` plumbing — the
grind), the region-level refinement core next (`swiglu_fused_refines_unfused`),
then the **flat-memory bridge side conditions**, the **specification** last
(one public headline `swiglu_equiv` on the `≡[R]` surface), and a compile-time
**trust audit**. The real sections below are `Swiglu.kernels`, `Swiglu.lemmas`,
`Swiglu.theorems`, `Swiglu.bridge`, `Swiglu.spec`.

**Both kernels round at exactly the same sites.** `R.castTo .bf16 _` is one
bf16 quantization (`RoundingModel.castTo` = the ℝ-level core of a `.to(bf16)`
cast / a store into a bf16 buffer). Writing `c(·)` for it, per active output
lane both pipelines produce `c(c( c(silu(x)) · y ))`:

* **fused** rounds `silu(x)` to bf16 (the `.to(bf16)`), multiplies by `y` on
  ℝ, then the output cast + store.
* **unfused** = `silu_step` (materializes `c(c(silu(x)))` into a bf16 scratch
  tensor `S`) then `mul_step` (loads it back — tag-exact — multiplies by `y`,
  output cast + store).

The pipeline's intermediate carries a doubled cast `c(c(silu(x)))` (cast +
store) where the fused kernel has a single `c(silu(x))`. **Idempotence of `c`
(`round ∘ round = round`) is exactly what collapses that redundant re-round** so
the two agree — the honest formalization of "they round at the same places, so
re-rounding changes nothing". This needs no extra hypothesis: idempotence is a
*defining field* of `RoundingModel` (`round_idem`) — a non-idempotent function
is not a rounding — so it holds for **every** `R`. dtypes are always
distinguished: `castTo` takes the `FloatDType`, so bf16 and fp32 round by
genuinely different functions; only the per-dtype rounding *behaviour* is the
abstract model `R`.

Outputs are read at the `MemCell` layer (`floatCell .bf16 v`): bf16 stores are
tag-exact, so a narrow-float cell is only observable as its typed `MemCell` —
the classic fp16/bf16 convention (`rmsnorm_fused_llama`). The generic
fixed-`R` unpackers used below (`ComputeKernel.ExecCorrectR.out` for a
realization, `ComputeRefine.Refines.out` for a refinement) are
kernel-agnostic and live in the library (`VeriTile.Triton.Float.Refine`).

## The public result (bottom of file)

The single public headline is **`swiglu_equiv`** — kernel equivalence on the
shared masked IO signature:

    swigluFusedIO ncols B ≡[R] swigluUnfusedIO ncols B

`≡[R]` is the audit-once kernel-equivalence combinator
(`MaskedKernelIO₂.Equiv`, `VeriTile.Triton.Memory.KernelSpec`), the `⊨`-grade
form of the refinement surface. Spelled out, the headline says: for **every**
disjoint flat placement of the interface buffers `X`/`Y`/`OUT` **plus the
pipeline's private staging buffer `S`** (∀ base pointers, ∀ buffer sizes),
**every** program id all of whose *active* lanes (`pid * B + j < ncols`) land
inside their buffers, and **every** launch state (modulo the surface-wide
launch-clean `undef` bookkeeping channel — the modeling boundary documented
in `VeriTile.Triton.Memory.KernelSpec`) — **no input hypotheses at all**, not
even loaded inputs: "equal inputs" is simply "the same `s₀`" —
both translated pointer kernels terminate under `execR R`, their active `OUT`
lanes hold equal values, and each kernel leaves every cell outside the active
`OUT` window and its **own** active scratch windows untouched (the fused
kernel has no scratch, so it frames outside `OUT` alone; the pipeline may
additionally write the active `S` window, and `≡` never compares `S`).

The statement mentions only the two IO signatures and the library equivalence
surface — **no spec** (the `#stmtSurfaceSubset` gate below enforces this).
Everything else — the per-kernel realizations, frame lemmas, the pipeline
split, and the region-level refinement `swiglu_fused_refines_unfused` that
`hrun` repackages — is scaffolding. There are **no spec definitions**: each
kernel's expected per-lane output is written inline as a `castTo` closed
form, so a self-referential spec is impossible by construction (and there is
no `#specNonCircular` gate to run).

**Honest boundary**: `swiglu_unfused` is the `ComputeKernel.seq`
*single-launch concatenation* of the two step bodies — register bindings flow
across the stage seam, which no real two-launch execution allows. The honest
per-launch semantics is `execPipelineR`, and the register-leakage gap is
bridged once in the library (`ComputeKernel.execR_seq_rel_execPipelineR`,
`VeriTile.Triton.Float.Pipeline`); this file's headline is about the
concatenated kernel.
-/

namespace VeriTile.Bench.Examples.SwigluRounding

open VeriTile.Triton
open scoped VeriTile.Triton.MaskedKernelIO₂
open VeriTile.Examples (InputLoadedAt floatCell InputCellsLoadedAt programLaneOffset)

/-! ## Kernels -/
section Swiglu.kernels

/-- Fused SwiGLU: rounds the intermediate `silu(x)` to bf16 (the `.to(bf16)`
cast), multiplies by `y` on the ℝ channel, then the bf16 output store — so it
rounds at exactly the same places as the unfused pipeline. -/
def swiglu_fused (X Y OUT : RegionName) (ncols BLOCK_N : Nat) : ComputeKernel := triton {
  start_col = tl.program_id(0) * $(BLOCK_N)
  cols = start_col + tl.arange(0, $(BLOCK_N))
  x = tl.load(X + cols, mask=cols < $(ncols), other=0.0)
  y = tl.load(Y + cols, mask=cols < $(ncols), other=0.0)
  sil = (x * tl.sigmoid(x)).to(tl.bfloat16)
  out = sil * y
  tl.store(OUT + cols, (out).to(tl.bfloat16), mask=cols < $(ncols))
}

/-- Step A of the unfused pipeline: materialize `silu(x)` into the bf16
tensor `S`. -/
def silu_step (X S : RegionName) (ncols BLOCK_N : Nat) : ComputeKernel := triton {
  start_col = tl.program_id(0) * $(BLOCK_N)
  cols = start_col + tl.arange(0, $(BLOCK_N))
  x = tl.load(X + cols, mask=cols < $(ncols), other=0.0)
  s = x * tl.sigmoid(x)
  tl.store(S + cols, (s).to(tl.bfloat16), mask=cols < $(ncols))
}

/-- Step B of the unfused pipeline: load the bf16 intermediate back and
multiply by `y`. The `S` load is bf16-typed (it reads bf16 cells); the
mixed-dtype multiply upcasts it to ℝ exactly. -/
def mul_step (S Y OUT : RegionName) (ncols BLOCK_N : Nat) : ComputeKernel := triton {
  start_col = tl.program_id(0) * $(BLOCK_N)
  cols = start_col + tl.arange(0, $(BLOCK_N))
  z = tl.load(S + cols, mask=cols < $(ncols)).to(tl.bfloat16)
  y = tl.load(Y + cols, mask=cols < $(ncols), other=0.0)
  out = z * y
  tl.store(OUT + cols, (out).to(tl.bfloat16), mask=cols < $(ncols))
}

/-- The unfused pipeline `B ∘ A` as one kernel: `ComputeKernel.seq` of the
two step kernels — the concatenation of their bodies (registers flowing
across the seam; the honest per-launch semantics is `execPipelineR`, related
by the library launch bridge `ComputeKernel.execR_seq_rel_execPipelineR` —
see the module docstring's honest-boundary note). -/
def swiglu_unfused (X Y S OUT : RegionName) (ncols BLOCK_N : Nat) : ComputeKernel :=
  ComputeKernel.seq [X, Y] [OUT]
    [silu_step X S ncols BLOCK_N, mul_step S Y OUT ncols BLOCK_N]

end Swiglu.kernels

/- Variables shared by every lemma and theorem below — declared **once**, at
namespace scope, so the `Swiglu.lemmas`, `Swiglu.theorems`, and
`Swiglu.bridge` sections all inherit them (`end <section>` only clears
variables declared *inside* that section; `Swiglu.spec` instantiates the
regions concretely instead). Each
kernel's per-lane "expected output" is written inline as a closed form over
`TiledActivation` + `RoundingModel.castTo` (one `castTo .bf16` = one bf16
quantization) — there are deliberately **no** spec definitions. -/
variable (X Y S OUT : RegionName) (ncols BLOCK_N : Nat) (s : BlockState)
variable (xs ys : Fin BLOCK_N → ℝ)
variable (R : RoundingModel)

section Swiglu.lemmas

/-! ## Sequential decomposition of the pipeline -/

/-- Execution of the composed pipeline `A ; B` as the sequential composition
of the two step executions — the two-stage instance of the library split
`execR_toAlgKernel_seq` (the no-reset fold over the `ComputeKernel.seq`
stages). -/
private theorem exec_swiglu_unfusedR (R : RoundingModel)
    (X Y S OUT : RegionName) (ncols BLOCK_N : Nat) (s : BlockState) :
    execR R (swiglu_unfused X Y S OUT ncols BLOCK_N) s =
      (execR R (silu_step X S ncols BLOCK_N) s).bind
        (fun s1 => execR R (mul_step S Y OUT ncols BLOCK_N) s1) := by
  show execR R (ComputeKernel.seq [X, Y] [OUT]
      [silu_step X S ncols BLOCK_N, mul_step S Y OUT ncols BLOCK_N]).toAlgKernel s = _
  rw [execR_toAlgKernel_seq]
  simp only [List.foldl_cons, List.foldl_nil, Option.bind_some]

/-! ## Component realizations

Input hypotheses use the shared `VeriTile.Examples.InputLoadedAt` contract
(region holds `xs` at `s.pid * N + i`; `InputCellsLoadedAt` is its
`MemCell`-typed sibling, used for the bf16 intermediate `S`). Output claims
are stated directly through `ComputeCorrect.OutputReadable.read`.
`programLaneOffset` is that same lane address, named for the proofs. -/

/-- The fused kernel writes `c(c( c(silu x) · y ))` (three bf16 quantizations)
at each active `OUT` lane, under rounding model `R`. -/
private theorem swiglu_fused_realizesR
    (h_x : InputLoadedAt s X BLOCK_N xs) (h_y : InputLoadedAt s Y BLOCK_N ys) :
    ComputeKernel.ExecCorrectR R (swiglu_fused X Y OUT ncols BLOCK_N) s (fun s' =>
      ∀ i : Fin BLOCK_N, s.pid * BLOCK_N + i.val < ncols →
        ComputeCorrect.OutputReadable.read s' (OUT, s.pid * BLOCK_N + i.val)
          = floatCell .bf16 (R.castTo .bf16 (R.castTo .bf16 (R.castTo .bf16 (TiledActivation.silu (xs i)) * ys i)))) := by
  unfold InputLoadedAt at h_x h_y
  apply ComputeKernel.computeCorrectR_of_toAlgKernel
  · simp [swiglu_fused, ComputeExpr.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h_inj : Function.Injective
      (fun idx : TileIndex [BLOCK_N] => s.pids 0 * BLOCK_N + idx.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    have hab' : s.pids 0 * BLOCK_N + a.val = s.pids 0 * BLOCK_N + b.val := by
      simpa using hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab')
    rfl
  simp [execR, swiglu_fused, stepStmtsR, stepStmtR, evalOpR.eq_def,
        tile_elementwise, ComputeExpr.toAlgorithm?] at hExec
  subst s'
  simp only [ComputeCorrect.OutputReadable.read_memcell, BlockState.pid_eq]
  rw [BlockState.scatter_memcell_R_prop_masked_nd R .bf16 _ _ _ _ h_inj (i, PUnit.unit)]
  have hx := h_x i
  have hy := h_y i
  simp only [BlockState.pid_eq] at hx hy
  simp [hActive, floatCell, TiledActivation.silu,
        tile_elementwise, hx, hy, RoundingModel.cast, RoundingModel.storeValue,
        FloatDType.storeValue, FloatDType.ofReal]

/-- Component theorem for step A: under rounding model `R`, the active lanes of
`S` receive `c(c(silu x))` (the `.to(bf16)` cast + the store to the bf16
tensor) as bf16 cells. -/
private theorem silu_step_realizesR (h_x : InputLoadedAt s X BLOCK_N xs) :
    ComputeKernel.ExecCorrectR R (silu_step X S ncols BLOCK_N) s (fun s' =>
      ∀ i : Fin BLOCK_N, s.pid * BLOCK_N + i.val < ncols →
        ComputeCorrect.OutputReadable.read s' (S, s.pid * BLOCK_N + i.val)
          = floatCell .bf16 (R.castTo .bf16 (R.castTo .bf16 (TiledActivation.silu (xs i))))) := by
  unfold InputLoadedAt at h_x
  apply ComputeKernel.computeCorrectR_of_toAlgKernel
  · simp [silu_step, ComputeExpr.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h_inj : Function.Injective
      (fun idx : TileIndex [BLOCK_N] => s.pids 0 * BLOCK_N + idx.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    have hab' : s.pids 0 * BLOCK_N + a.val = s.pids 0 * BLOCK_N + b.val := by
      simpa using hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab')
    rfl
  simp [execR, silu_step, stepStmtsR, stepStmtR, evalOpR.eq_def,
        tile_elementwise, ComputeExpr.toAlgorithm?] at hExec
  subst s'
  simp only [ComputeCorrect.OutputReadable.read_memcell, BlockState.pid_eq]
  rw [BlockState.scatter_memcell_R_prop_masked_nd R .bf16 _ _ _ _ h_inj (i, PUnit.unit)]
  have hx := h_x i
  simp only [BlockState.pid_eq] at hx
  simp [hActive, floatCell, TiledActivation.silu,
        tile_elementwise, hx, RoundingModel.cast, RoundingModel.storeValue,
        FloatDType.storeValue, FloatDType.ofReal]

/-- Frame lemma for step A: every region other than `S` is untouched, for
every rounding model. -/
private theorem silu_step_preservesR (s' : BlockState)
    (keepReg : RegionName) (h_keep : keepReg ≠ S)
    (hExec : execR R (silu_step X S ncols BLOCK_N) s = some s') :
    ∀ offset : Nat, s'.mem keepReg offset = s.mem keepReg offset := by
  intro offset
  simp [execR, silu_step, stepStmtsR, stepStmtR, evalOpR.eq_def,
        tile_elementwise, ComputeExpr.toAlgorithm?] at hExec
  subst s'
  rw [BlockState.foldl_writeMemAsR_preserve_other_region R .bf16 _ _ _
        keepReg h_keep offset]
  rfl

/-- Step A leaves the program ids untouched (needed to carry the lane
addressing across the composition). -/
private theorem silu_step_execR_pids (s1 : BlockState)
    (hExec : execR R (silu_step X S ncols BLOCK_N) s = some s1) :
    s1.pids = s.pids := by
  simp [execR, silu_step, stepStmtsR, stepStmtR, evalOpR.eq_def,
        tile_elementwise, ComputeExpr.toAlgorithm?] at hExec
  subst s1
  rw [BlockState.foldl_writeMemAsR_masked_pids]
  rfl

/-- Component theorem for step B: given the bf16 intermediate `S` holding
payloads `zs`, the active lanes of `OUT` receive `c(c(z·y))` under `R`. -/
private theorem mul_step_realizesR (zs : Fin BLOCK_N → ℝ)
    (h_z : InputCellsLoadedAt s S ncols BLOCK_N .bf16 zs) (h_y : InputLoadedAt s Y BLOCK_N ys) :
    ComputeKernel.ExecCorrectR R (mul_step S Y OUT ncols BLOCK_N) s (fun s' =>
      ∀ i : Fin BLOCK_N, s.pid * BLOCK_N + i.val < ncols →
        ComputeCorrect.OutputReadable.read s' (OUT, s.pid * BLOCK_N + i.val)
          = floatCell .bf16 (R.castTo .bf16 (R.castTo .bf16 (zs i * ys i)))) := by
  unfold InputCellsLoadedAt floatCell at h_z
  unfold InputLoadedAt at h_y
  apply ComputeKernel.computeCorrectR_of_toAlgKernel
  · simp [mul_step, ComputeExpr.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h_inj : Function.Injective
      (fun idx : TileIndex [BLOCK_N] => s.pids 0 * BLOCK_N + idx.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    have hab' : s.pids 0 * BLOCK_N + a.val = s.pids 0 * BLOCK_N + b.val := by
      simpa using hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab')
    rfl
  simp [execR, mul_step, stepStmtsR, stepStmtR, evalOpR.eq_def,
        tile_elementwise, ComputeExpr.toAlgorithm?] at hExec
  subst s'
  simp only [ComputeCorrect.OutputReadable.read_memcell, BlockState.pid_eq]
  rw [BlockState.scatter_memcell_R_prop_masked_nd R .bf16 _ _ _ _ h_inj (i, PUnit.unit)]
  have hz := h_z i hActive
  have hy := h_y i
  simp only [BlockState.pid_eq] at hz hy
  simp [hActive, floatCell, BlockState.readMemValue_bf16_of_cell hz, hy,
        tile_elementwise, RoundingModel.cast, RoundingModel.storeValue,
        FloatDType.storeValue, FloatDType.ofReal]

/-- The unfused pipeline writes `c(c( c(c(silu x)) · y ))` (four bf16
quantizations — the intermediate `S` cast+store, then B's cast+store) at each
active `OUT` lane, for every rounding model. **Derived by composing**
`silu_step_realizesR` and `mul_step_realizesR` through the intermediate state
(the frame lemma keeps `Y` intact; `S ≠ Y` is the aliasing hypothesis). -/
private theorem swiglu_unfused_realizesR
    (h_x : InputLoadedAt s X BLOCK_N xs) (h_y : InputLoadedAt s Y BLOCK_N ys)
    (h_SY : S ≠ Y) :
    ComputeKernel.ExecCorrectR R (swiglu_unfused X Y S OUT ncols BLOCK_N) s (fun s' =>
      ∀ i : Fin BLOCK_N, s.pid * BLOCK_N + i.val < ncols →
        ComputeCorrect.OutputReadable.read s' (OUT, s.pid * BLOCK_N + i.val)
          = floatCell .bf16
            (R.castTo .bf16 (R.castTo .bf16
              (R.castTo .bf16 (R.castTo .bf16 (TiledActivation.silu (xs i))) * ys i)))) := by
  apply ComputeKernel.computeCorrectR_of_toAlgKernel
  · simp [swiglu_unfused]
  intro s0 s' hExec hs0
  subst s0
  rw [exec_swiglu_unfusedR] at hExec
  intro i hActive
  cases hA : execR R (ComputeKernel.toAlgKernel (silu_step X S ncols BLOCK_N)) s with
  | none => rw [hA] at hExec; exact absurd hExec (by simp)
  | some s1 =>
      rw [hA] at hExec
      have hExecB : execR R (ComputeKernel.toAlgKernel (mul_step S Y OUT ncols BLOCK_N)) s1
          = some s' := hExec
      have hpids : s1.pids = s.pids := silu_step_execR_pids X S ncols BLOCK_N s R s1 hA
      -- intermediate `S` cells from step A's realization
      have hAout := ComputeKernel.ExecCorrectR.out
        (silu_step_realizesR X S ncols BLOCK_N s xs R h_x)
        (by simp [silu_step, ComputeExpr.toAlgorithm?]) hA
      have h_z1 : InputCellsLoadedAt s1 S ncols BLOCK_N .bf16 (fun j => R.castTo .bf16 (R.castTo .bf16 (TiledActivation.silu (xs j)))) := by
        intro j hj
        rw [BlockState.pid_eq] at hj
        have hj' : s.pids 0 * BLOCK_N + j.val < ncols := by rwa [hpids] at hj
        have hout := hAout j (by rw [BlockState.pid_eq]; exact hj')
        simpa [BlockState.pid_eq, hpids] using hout
      -- `Y` survives step A (frame lemma; `S ≠ Y`)
      have h_y1 : InputLoadedAt s1 Y BLOCK_N ys := by
        intro j
        have hmem := silu_step_preservesR X S ncols BLOCK_N s R s1 Y (Ne.symm h_SY) hA
        have hread : s1.readMem Y (programLaneOffset s1 BLOCK_N j)
            = s.readMem Y (programLaneOffset s1 BLOCK_N j) := by
          unfold BlockState.readMem
          rw [hmem]
        show s1.readMem Y (programLaneOffset s1 BLOCK_N j) = ys j
        rw [hread]
        have hy := h_y j
        simp only [BlockState.pid_eq, programLaneOffset, hpids] at hy ⊢
        exact hy
      -- step B's realization at the intermediate state
      have hBout := ComputeKernel.ExecCorrectR.out
        (mul_step_realizesR Y S OUT ncols BLOCK_N s1 ys R (fun j => R.castTo .bf16 (R.castTo .bf16 (TiledActivation.silu (xs j)))) h_z1 h_y1)
        (by simp [mul_step, ComputeExpr.toAlgorithm?]) hExecB
      have hAct1 : s1.pids 0 * BLOCK_N + i.val < ncols := by rw [hpids]; exact hActive
      have hout := hBout i (by rw [BlockState.pid_eq]; exact hAct1)
      simpa [programLaneOffset, hpids] using hout

/-! ## Frame lemmas (the writes each kernel does NOT perform)

Every store in this file is lane-masked, so the cell-level frames go through
the library's **masked** scatter frame
`BlockState.foldl_writeMemAsR_preserve_masked_prop` (region-level:
`BlockState.foldl_writeMemAsR_preserve_other_region`); its unmasked sibling
`BlockState.foldl_writeMemAsR_preserve_cell` does not apply here. -/

/-- Frame lemma for the fused kernel: every region other than `OUT` is
untouched. -/
private theorem swiglu_fused_preservesR (s' : BlockState)
    (r : RegionName) (hrOUT : r ≠ OUT)
    (hExec : execR R (swiglu_fused X Y OUT ncols BLOCK_N) s = some s') :
    ∀ o : Nat, s'.mem r o = s.mem r o := by
  intro o
  simp [execR, swiglu_fused, stepStmtsR, stepStmtR, evalOpR.eq_def,
        tile_elementwise, ComputeExpr.toAlgorithm?] at hExec
  subst s'
  rw [BlockState.foldl_writeMemAsR_preserve_other_region R .bf16 _ _ _ r hrOUT o]
  rfl

/-- Frame lemma for the fused kernel: `OUT` offsets not hit by an active lane
are untouched. -/
private theorem swiglu_fused_preserves_unhitR (s' : BlockState)
    (o : Nat)
    (ho : ∀ i : Fin BLOCK_N, s.pids 0 * BLOCK_N + i.val < ncols →
      programLaneOffset s BLOCK_N i ≠ o)
    (hExec : execR R (swiglu_fused X Y OUT ncols BLOCK_N) s = some s') :
    s'.mem OUT o = s.mem OUT o := by
  simp [execR, swiglu_fused, stepStmtsR, stepStmtR, evalOpR.eq_def,
        tile_elementwise, ComputeExpr.toAlgorithm?] at hExec
  subst s'
  rw [BlockState.foldl_writeMemAsR_preserve_masked_prop R .bf16 _ _ _ o _
      (fun k _ hPk => ho k.1 hPk)]
  rfl

/-- Frame lemma for step B: every region other than `OUT` is untouched. -/
private theorem mul_step_preservesR (s' : BlockState)
    (r : RegionName) (hrOUT : r ≠ OUT)
    (hExec : execR R (mul_step S Y OUT ncols BLOCK_N) s = some s') :
    ∀ o : Nat, s'.mem r o = s.mem r o := by
  intro o
  simp [execR, mul_step, stepStmtsR, stepStmtR, evalOpR.eq_def,
        tile_elementwise, ComputeExpr.toAlgorithm?] at hExec
  subst s'
  rw [BlockState.foldl_writeMemAsR_preserve_other_region R .bf16 _ _ _ r hrOUT o]
  rfl

/-- Frame lemma for step B: `OUT` offsets not hit by an active lane are
untouched. -/
private theorem mul_step_preserves_unhitR (s' : BlockState)
    (o : Nat)
    (ho : ∀ i : Fin BLOCK_N, s.pids 0 * BLOCK_N + i.val < ncols →
      programLaneOffset s BLOCK_N i ≠ o)
    (hExec : execR R (mul_step S Y OUT ncols BLOCK_N) s = some s') :
    s'.mem OUT o = s.mem OUT o := by
  simp [execR, mul_step, stepStmtsR, stepStmtR, evalOpR.eq_def,
        tile_elementwise, ComputeExpr.toAlgorithm?] at hExec
  subst s'
  rw [BlockState.foldl_writeMemAsR_preserve_masked_prop R .bf16 _ _ _ o _
      (fun k _ hPk => ho k.1 hPk)]
  rfl

/-- Frame lemma for step A: `S` offsets not hit by an active lane are
untouched — the in-region complement of `silu_step_preservesR`, needed
because the `≡[R]` frame only cedes the *active* `S` window to the pipeline
(the inactive overhang of a partial block stays framed). -/
private theorem silu_step_preserves_unhitR (s' : BlockState)
    (o : Nat)
    (ho : ∀ i : Fin BLOCK_N, s.pids 0 * BLOCK_N + i.val < ncols →
      programLaneOffset s BLOCK_N i ≠ o)
    (hExec : execR R (silu_step X S ncols BLOCK_N) s = some s') :
    s'.mem S o = s.mem S o := by
  simp [execR, silu_step, stepStmtsR, stepStmtR, evalOpR.eq_def,
        tile_elementwise, ComputeExpr.toAlgorithm?] at hExec
  subst s'
  rw [BlockState.foldl_writeMemAsR_preserve_masked_prop R .bf16 _ _ _ o _
      (fun k _ hPk => ho k.1 hPk)]
  rfl

/-! ## Termination (unconditional, for the `≡[R]` contract)

`≡[R]`'s region-model obligation starts from **any** state — no loaded
inputs, no clean undef — so termination cannot come from the realization
theorems (which take `InputLoadedAt`). It holds outright: both bodies are
straight-line statement lists whose every step is total, so `execR R`
computes to `some _` by the same computational unfold the realizations use. -/

/-- The fused kernel terminates under `execR R` from every state. -/
private theorem swiglu_fused_execR_isSome :
    ∃ s', execR R ((swiglu_fused X Y OUT ncols BLOCK_N).toAlgKernel) s
      = some s' := by
  simp [execR, swiglu_fused, stepStmtsR, stepStmtR, evalOpR.eq_def,
        tile_elementwise, ComputeExpr.toAlgorithm?]

/-- Step A terminates under `execR R` from every state. -/
private theorem silu_step_execR_isSome :
    ∃ s', execR R ((silu_step X S ncols BLOCK_N).toAlgKernel) s = some s' := by
  simp [execR, silu_step, stepStmtsR, stepStmtR, evalOpR.eq_def,
        tile_elementwise, ComputeExpr.toAlgorithm?]

/-- Step B terminates under `execR R` from every state. -/
private theorem mul_step_execR_isSome :
    ∃ s', execR R ((mul_step S Y OUT ncols BLOCK_N).toAlgKernel) s
      = some s' := by
  simp [execR, mul_step, stepStmtsR, stepStmtR, evalOpR.eq_def,
        tile_elementwise, ComputeExpr.toAlgorithm?]

end Swiglu.lemmas

/-! ## The region-level refinement core -/
section Swiglu.theorems

/-- **fused refines unfused** (`ComputeRefine.Refines`, scratch `[S]`): for
every rounding model `R`, running the fused kernel and the unfused pipeline
from the same state performs THE SAME WRITES — the final memories agree at
every cell outside the scratch tensor `S`. Idempotence (`round ∘ round =
round`, a defining field of `RoundingModel`) is what makes the two kernels'
matched rounding sites coincide; no hypothesis is needed.

Formerly the file's headline; now the mathematical core the `≡[R]`
specification's region-model obligation repackages (its `OUT`-agreement leg
is exactly this memory agreement read at the active output lanes). -/
theorem swiglu_fused_refines_unfused
    (h_x : InputLoadedAt s X BLOCK_N xs) (h_y : InputLoadedAt s Y BLOCK_N ys)
    (h_SY : S ≠ Y) :
    ComputeRefine.Refines R
      (swiglu_fused X Y OUT ncols BLOCK_N)
      (swiglu_unfused X Y S OUT ncols BLOCK_N) s [S] := by
  apply ComputeKernel.computeRefineR_of_toAlgKernel
  · simp [swiglu_fused, ComputeExpr.toAlgorithm?]
  · simp [swiglu_unfused]
  intro s0 lhs' rhs' hL hR hs0
  subst s0
  intro r hr o
  have hrS : r ≠ S := by simpa using hr
  -- split the pipeline execution into A ; B (keeping the original `hR`)
  have hRsplit := hR
  rw [exec_swiglu_unfusedR] at hRsplit
  cases hA : execR R (ComputeKernel.toAlgKernel (silu_step X S ncols BLOCK_N)) s with
  | none => rw [hA] at hRsplit; exact absurd hRsplit (by simp)
  | some s1 =>
      rw [hA] at hRsplit
      have hRB : execR R (ComputeKernel.toAlgKernel (mul_step S Y OUT ncols BLOCK_N)) s1
          = some rhs' := hRsplit
      have hpids : s1.pids = s.pids := silu_step_execR_pids X S ncols BLOCK_N s R s1 hA
      -- step A only touches `S`, and `r ≠ S`
      have hA_mem : ∀ o' : Nat, s1.mem r o' = s.mem r o' :=
        silu_step_preservesR X S ncols BLOCK_N s R s1 r hrS hA
      by_cases hrOUT : r = OUT
      · rw [hrOUT] at hA_mem ⊢
        by_cases hhit : ∃ i : Fin BLOCK_N,
            (s.pids 0 * BLOCK_N + i.val < ncols) ∧ o = programLaneOffset s BLOCK_N i
        · -- an active lane writes this cell on both sides; the written values
          -- coincide because idempotence collapses the pipeline's doubled
          -- intermediate cast to the fused kernel's single one
          obtain ⟨i, hAct, rfl⟩ := hhit
          have hFout := ComputeKernel.ExecCorrectR.out
            (swiglu_fused_realizesR X Y OUT ncols BLOCK_N s xs ys R h_x h_y)
            (by simp [swiglu_fused, ComputeExpr.toAlgorithm?]) hL
          have hUout := ComputeKernel.ExecCorrectR.out
            (swiglu_unfused_realizesR X Y S OUT ncols BLOCK_N s xs ys R h_x h_y h_SY)
            (by simp [swiglu_unfused]) hR
          have hF := hFout i (by rw [BlockState.pid_eq]; exact hAct)
          have hU := hUout i (by rw [BlockState.pid_eq]; exact hAct)
          simp only [ComputeCorrect.OutputReadable.read_memcell, BlockState.pid_eq] at hF hU
          simp only [programLaneOffset]
          rw [hF, hU]
          simp [RoundingModel.castTo, R.round_idem]
        · -- no active lane hits this `OUT` cell: preserved on both sides
          have hmiss : ∀ i : Fin BLOCK_N, s.pids 0 * BLOCK_N + i.val < ncols →
              programLaneOffset s BLOCK_N i ≠ o :=
            fun i hi heq => hhit ⟨i, hi, heq.symm⟩
          have hLmem : lhs'.mem OUT o = s.mem OUT o :=
            swiglu_fused_preserves_unhitR X Y OUT ncols BLOCK_N s R lhs' o hmiss hL
          have hBmem : rhs'.mem OUT o = s1.mem OUT o :=
            mul_step_preserves_unhitR Y S OUT ncols BLOCK_N s1 R rhs' o
              (fun i hi => by
                simp only [programLaneOffset, BlockState.pid_eq, hpids]
                exact hmiss i (by rwa [hpids] at hi)) hRB
          rw [hLmem, hBmem, hA_mem o]
      · -- `r ∉ {S, OUT}`: untouched by every store on both sides
        have hLmem : lhs'.mem r o = s.mem r o :=
          swiglu_fused_preservesR X Y OUT ncols BLOCK_N s R lhs' r hrOUT hL o
        have hBmem : rhs'.mem r o = s1.mem r o :=
          mul_step_preservesR Y S OUT ncols BLOCK_N s1 R rhs' r hrOUT hRB o
        rw [hLmem, hBmem, hA_mem o]

end Swiglu.theorems

/-! ## Flat-memory bridge side conditions

Both kernels are register-indirect (`cols = start_col + tl.arange(…);
tl.load(X + cols, mask=cols < ncols)`), so no ∀-state safety contract covers
them; the flat-memory bridge (v1.2, `execR` flavor) takes the per-execution
`Kernel.TraceSafeR` contract instead, plus `FlattenOk` (bridge fragment
membership). Every access on both sides uses the same masked window: only the
*active* lanes (`pid * BLOCK_N + j < ncols`) touch memory, so the bounds
obligations are **lane-wise** — each active lane's address is below the bound
of the buffer it touches, with no whole-window `ncols ≤ bounds` contract in
sight. The fused kernel makes 3 accesses (load `X`, load `Y`, store `OUT`);
the unfused pipeline — the concatenation of the two step bodies — makes 5
(load `X`, store `S`, load `S`, load `Y`, store `OUT`), and its `S` bounds
come from the `≡[R]` scratch-window hypothesis. -/
section Swiglu.bridge

/-- The fused kernel sits inside the bridge's covered fragment. -/
theorem swiglu_fused_flattenOk :
    ((swiglu_fused X Y OUT ncols BLOCK_N).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [swiglu_fused, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

/-- The concatenated unfused pipeline sits inside the bridge's covered
fragment (concatenation adds no new statement forms). -/
theorem swiglu_unfused_flattenOk :
    ((swiglu_unfused X Y S OUT ncols BLOCK_N).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [swiglu_unfused, silu_step, mul_step, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

set_option maxHeartbeats 1600000 in
/-- Per-execution safety walk for the fused kernel under `R`: one
computational unfold walks all seven statements — four are memory-silent
(the `program_id` offset `start_col`, the lane vector `cols`, and the
register binds `sil`/`out`), so their `SafeAtR` discharges outright — and
reduces the three masked accesses (load `X`, load `Y`, store `OUT`) to the
lane-wise bounds hypotheses. -/
theorem swiglu_fused_traceSafeR (bounds : RegionBounds)
    (hx : ∀ j : Fin BLOCK_N, s.pid * BLOCK_N + j.val < ncols →
      s.pid * BLOCK_N + j.val < bounds X)
    (hy : ∀ j : Fin BLOCK_N, s.pid * BLOCK_N + j.val < ncols →
      s.pid * BLOCK_N + j.val < bounds Y)
    (hout : ∀ j : Fin BLOCK_N, s.pid * BLOCK_N + j.val < ncols →
      s.pid * BLOCK_N + j.val < bounds OUT) :
    Kernel.TraceSafeR R bounds
      ((swiglu_fused X Y OUT ncols BLOCK_N).toAlgKernel) s := by
  unfold Kernel.TraceSafeR
  simp only [BlockState.pid_eq] at hx hy hout
  simp [swiglu_fused, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
    Stmt.TraceSafeListR, Stmt.TraceSafeR, Op.SafeAtR.eq_def,
    MaskOpt.SafeAtR, MemAccess.SafeAtR, stepStmtR, evalOpR.eq_def,
    tile_elementwise,
    MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR,
    MaskOpt.ActiveR, BlockState.setReg,
    Tile.bop, Tile.cop, NumericDType.add, NumericDType.mul,
    ComparableDType.lt]
  exact ⟨fun a ha => hx a ha, fun a ha => hy a ha, fun a ha => hout a ha⟩

set_option maxHeartbeats 1600000 in
/-- Per-execution safety walk for the concatenated unfused pipeline under
`R`: same computational unfold over the eleven concatenated statements — six
are memory-silent (each step's `program_id` offset and lane vector, plus the
register binds `s`/`out`); the five masked accesses (load `X`, store `S`,
load `S`, load `Y`, store `OUT`) all address the shared active window. -/
theorem swiglu_unfused_traceSafeR (bounds : RegionBounds)
    (hx : ∀ j : Fin BLOCK_N, s.pid * BLOCK_N + j.val < ncols →
      s.pid * BLOCK_N + j.val < bounds X)
    (hy : ∀ j : Fin BLOCK_N, s.pid * BLOCK_N + j.val < ncols →
      s.pid * BLOCK_N + j.val < bounds Y)
    (hout : ∀ j : Fin BLOCK_N, s.pid * BLOCK_N + j.val < ncols →
      s.pid * BLOCK_N + j.val < bounds OUT)
    (hs : ∀ j : Fin BLOCK_N, s.pid * BLOCK_N + j.val < ncols →
      s.pid * BLOCK_N + j.val < bounds S) :
    Kernel.TraceSafeR R bounds
      ((swiglu_unfused X Y S OUT ncols BLOCK_N).toAlgKernel) s := by
  unfold Kernel.TraceSafeR
  simp only [BlockState.pid_eq] at hx hy hout hs
  simp [swiglu_unfused, silu_step, mul_step, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?,
    Stmt.TraceSafeListR, Stmt.TraceSafeR, Op.SafeAtR.eq_def,
    MaskOpt.SafeAtR, MemAccess.SafeAtR, stepStmtR, evalOpR.eq_def,
    tile_elementwise,
    MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR,
    MaskOpt.ActiveR, BlockState.setReg,
    BlockState.foldl_writeMemAsR_masked_pids,
    Tile.bop, Tile.cop, NumericDType.add, NumericDType.mul,
    ComparableDType.lt]
  -- (four obligations, not five: `simp` merges the S-store and S-load
  -- windows' identical lane-bounds conjuncts)
  exact ⟨fun a ha => hx a ha, fun a ha => hs a ha,
    fun a ha => hy a ha, fun a ha => hout a ha⟩

end Swiglu.bridge

/-! ## The spec: `swigluFusedIO ≡[R] swigluUnfusedIO` -/
section Swiglu.spec

/-- The unfused pipeline's masked **IO signature** — the whole kernel-specific
audit surface of the headline: interface `X`, `Y` → `OUT`, each program
owning the window `[pid * B, pid * B + B)` with active lanes
`pid * B + j < ncols` (partial blocks deactivate the overhang), **plus its
private staging buffer `S`** (same per-program window, lane-masked like the
output), declared as `scratch`: the pipeline may write its active `S` window,
but `S`'s post-state is no part of the contract — `≡` never compares it. -/
def swigluUnfusedIO (ncols B : Nat) : MaskedKernelIO₂ where
  kernel := swiglu_unfused ⟨"X"⟩ ⟨"Y"⟩ ⟨"S"⟩ ⟨"OUT"⟩ ncols B
  in1 := ⟨"X"⟩
  in2 := ⟨"Y"⟩
  out := ⟨"OUT"⟩
  B := B
  read1 := fun pid => pid * B
  read2 := fun pid => pid * B
  write := fun pid => pid * B
  mask := fun pid j => pid * B + j.val < ncols
  scratch := [(⟨"S"⟩, fun pid => pid * B)]

/-- The fused kernel's IO signature: the **same interface by construction**
(structure update of `swigluUnfusedIO` — buffers, windows, and mask shared
verbatim), with the fused kernel plugged in and **no scratch** (it stages
nothing through memory). -/
def swigluFusedIO (ncols B : Nat) : MaskedKernelIO₂ :=
  { swigluUnfusedIO ncols B with
    kernel := swiglu_fused ⟨"X"⟩ ⟨"Y"⟩ ⟨"OUT"⟩ ncols B
    scratch := [] }

/-- **The headline**: fused SwiGLU is equivalent to the unfused two-step
pipeline on their shared masked IO signature, for **every** rounding model —
see the module docstring for the full contract `≡[R]` unfolds to (both
kernels terminate from any state, active `OUT` lanes agree, each frames
outside `OUT` ∪ its own scratch). Proof: `MaskedKernelIO₂.Equiv.intro`
assembles the region-model equivalence — termination is unconditional, the
`OUT`-agreement leg is the region-level refinement theorem's memory
agreement (idempotence of rounding), the frames are the masked scatter frame
lemmas — with the flat-memory bridge side conditions (`FlattenOk` +
`TraceSafeR` per kernel). -/
specification swiglu_equiv (R : RoundingModel) (ncols B : Nat) :
    swigluFusedIO ncols B ≡[R] swigluUnfusedIO ncols B := by
  refine MaskedKernelIO₂.Equiv.intro _ _ ?_ ?_ ?_ ?_ ?_
  · -- FlattenOk, fused
    exact swiglu_fused_flattenOk ⟨"X"⟩ ⟨"Y"⟩ ⟨"OUT"⟩ ncols B
  · -- FlattenOk, unfused
    exact swiglu_unfused_flattenOk ⟨"X"⟩ ⟨"Y"⟩ ⟨"S"⟩ ⟨"OUT"⟩ ncols B
  · -- TraceSafeR, fused (its accesses need no scratch bounds)
    intro bounds t h1 h2 h3 _
    simp only [swigluFusedIO, swigluUnfusedIO] at h1 h2 h3 ⊢
    exact swiglu_fused_traceSafeR ⟨"X"⟩ ⟨"Y"⟩ ⟨"OUT"⟩ ncols B t R bounds
      h1 h2 h3
  · -- TraceSafeR, unfused (S bounds from the scratch-window hypothesis)
    intro bounds t h1 h2 h3 hsc
    simp only [swigluFusedIO, swigluUnfusedIO] at h1 h2 h3 hsc ⊢
    exact swiglu_unfused_traceSafeR ⟨"X"⟩ ⟨"Y"⟩ ⟨"S"⟩ ⟨"OUT"⟩ ncols B t R
      bounds h1 h2 h3
      (fun j hj => hsc (⟨"S"⟩, fun pid => pid * B) (by simp) j hj)
  · -- the region-model equivalence, from ANY state s₀ (no input hypotheses)
    intro s₀
    simp only [swigluFusedIO, swigluUnfusedIO]
    -- termination of both sides
    obtain ⟨s1, hexec1⟩ :=
      swiglu_fused_execR_isSome ⟨"X"⟩ ⟨"Y"⟩ ⟨"OUT"⟩ ncols B s₀ R
    obtain ⟨sA, hexecA⟩ := silu_step_execR_isSome ⟨"X"⟩ ⟨"S"⟩ ncols B s₀ R
    obtain ⟨s2, hexecB⟩ :=
      mul_step_execR_isSome ⟨"Y"⟩ ⟨"S"⟩ ⟨"OUT"⟩ ncols B sA R
    have hexec2 : execR R
        ((swiglu_unfused ⟨"X"⟩ ⟨"Y"⟩ ⟨"S"⟩ ⟨"OUT"⟩ ncols B).toAlgKernel) s₀
        = some s2 := by
      rw [exec_swiglu_unfusedR, hexecA]
      exact hexecB
    have hpids : sA.pids = s₀.pids :=
      silu_step_execR_pids ⟨"X"⟩ ⟨"S"⟩ ncols B s₀ R sA hexecA
    -- the inputs both kernels actually see: whatever s₀ holds (no hypotheses)
    have hxs : InputLoadedAt s₀ ⟨"X"⟩ B
        (fun j => s₀.readMem ⟨"X"⟩ (s₀.pid * B + j.val)) := fun _ => rfl
    have hys : InputLoadedAt s₀ ⟨"Y"⟩ B
        (fun j => s₀.readMem ⟨"Y"⟩ (s₀.pid * B + j.val)) := fun _ => rfl
    -- the region-level refinement, unpacked at this execution pair by the
    -- library `ComputeRefine.Refines.out`: memories agree outside S
    have hmem12 : ∀ r, r ∉ [(⟨"S"⟩ : RegionName)] →
        ∀ o, s1.mem r o = s2.mem r o :=
      (swiglu_fused_refines_unfused ⟨"X"⟩ ⟨"Y"⟩ ⟨"S"⟩ ⟨"OUT"⟩ ncols B s₀
          _ _ R hxs hys (by decide)).out
        (by simp [swiglu_fused, ComputeExpr.toAlgorithm?])
        (by simp [swiglu_unfused])
        hexec1 hexec2
    refine ⟨s1, s2, hexec1, hexec2, ?_, ?_, ?_⟩
    · -- active OUT lanes agree (OUT ∉ [S])
      intro j hj
      have hmem := hmem12 ⟨"OUT"⟩ (by decide) (s₀.pid * B + j.val)
      unfold BlockState.readMem
      rw [hmem]
    · -- fused frame: writes only the active OUT window (no scratch)
      intro r o hcond _hscr
      by_cases hrOUT : r = ⟨"OUT"⟩
      · subst hrOUT
        rcases hcond with hne | hno
        · exact absurd rfl hne
        · exact swiglu_fused_preserves_unhitR ⟨"X"⟩ ⟨"Y"⟩ ⟨"OUT"⟩ ncols B s₀
            R s1 o (fun i hi heq => hno i hi heq.symm) hexec1
      · exact swiglu_fused_preservesR ⟨"X"⟩ ⟨"Y"⟩ ⟨"OUT"⟩ ncols B s₀ R s1 r
          hrOUT hexec1 o
    · -- unfused frame: writes only active OUT ∪ active S (its scratch)
      intro r o hcond hscr
      have hA_mem : sA.mem r o = s₀.mem r o := by
        by_cases hrS : r = ⟨"S"⟩
        · subst hrS
          exact silu_step_preserves_unhitR ⟨"X"⟩ ⟨"S"⟩ ncols B s₀ R sA o
            (fun i hi heq =>
              hscr (⟨"S"⟩, fun pid => pid * B) (by simp) rfl i hi heq.symm)
            hexecA
        · exact silu_step_preservesR ⟨"X"⟩ ⟨"S"⟩ ncols B s₀ R sA r hrS
            hexecA o
      have hB_mem : s2.mem r o = sA.mem r o := by
        by_cases hrOUT : r = ⟨"OUT"⟩
        · subst hrOUT
          rcases hcond with hne | hno
          · exact absurd rfl hne
          · refine mul_step_preserves_unhitR ⟨"Y"⟩ ⟨"S"⟩ ⟨"OUT"⟩ ncols B sA
              R s2 o (fun i hi => ?_) hexecB
            simp only [programLaneOffset, BlockState.pid_eq, hpids]
            intro heq
            exact hno i (by rwa [hpids] at hi) heq.symm
        · exact mul_step_preservesR ⟨"Y"⟩ ⟨"S"⟩ ⟨"OUT"⟩ ncols B sA R s2 r
            hrOUT hexecB o
      rw [hB_mem, hA_mem]

end Swiglu.spec

/-! ## Trust audit (compile-time gate)

These commands re-audit the public result every time the file is elaborated —
if either gate fails (a smuggled axiom / `sorry`, or a foreign constant in the
trusted statement) the file stops compiling. See
`VeriTile.Meta.StatementAudit`. -/

-- (1) No `sorry`, no smuggled axiom — in the region-level core's and the
-- public headline's transitive proofs.
#axiomsClean swiglu_fused_refines_unfused
#axiomsClean swiglu_equiv

-- (2) The headline's statement surface is the two masked IO signatures plus
-- the audit-once kernel-equivalence combinator — NO spec (there are none),
-- no other project constant. If a spec-like definition ever creeps into the
-- statement, this fails.
#stmtSurfaceSubset swiglu_equiv ⊆
  [swigluFusedIO, swigluUnfusedIO, VeriTile.Triton.MaskedKernelIO₂.Equiv,
   RoundingModel]

-- (There is no `#specNonCircular` gate: the file defines no specs at all, so a
-- self-referential spec is impossible by construction.)

end VeriTile.Bench.Examples.SwigluRounding
