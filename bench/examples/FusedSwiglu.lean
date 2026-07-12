import VeriTile.Triton
import VeriTile.Examples.Common
import VeriTile.Meta.StatementAudit

/-!
# SwiGLU fused vs unfused — rounding-invariance pilot (#447 Phase C)

Self-contained showcase, read top to bottom: **kernels** first (what we
built), the **supporting lemmas** in the middle (`private` plumbing — the
grind), the **theorem** last (one public headline `swiglu_fused_refines_unfused`
plus its `private` derivation steps), then a compile-time **trust audit**. The
three real sections below are `Swiglu.kernels`, `Swiglu.lemmas`,
`Swiglu.theorems`.

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
fixed-`R` unpacker used below (`ComputeKernel.ExecCorrectR.out`) is
kernel-agnostic and lives in the library (`VeriTile.Triton.Float.Refine`).

## The public result (bottom of file)

The single public headline is **`swiglu_fused_refines_unfused`** — a
kernel-vs-kernel refinement on `ComputeRefine.Refines`: for every rounding
model `R`, from the same state the fused kernel and the unfused pipeline
perform the same writes outside the scratch tensor `S`. Its statement
mentions only the kernels, the loaded-input contract, the rounding-model
surface, and the library refinement surface — **no spec** (the
`#stmtSurfaceSubset` gate below enforces this). Everything else — the per-kernel
realizations, frame lemmas and the pipeline split — is `private` scaffolding.
There are **no spec definitions**: each kernel's expected per-lane output is
written inline as a `castTo` closed form, so a self-referential spec is
impossible by construction (and there is no `#specNonCircular` gate to run).
-/

namespace VeriTile.Bench.Examples.SwigluRounding

open VeriTile.Triton
open VeriTile.Examples (InputLoadedAt outWritesTo floatCell InputCellsLoadedAt programLaneOffset)

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
by the launch bridge below). -/
def swiglu_unfused (X Y S OUT : RegionName) (ncols BLOCK_N : Nat) : ComputeKernel :=
  ComputeKernel.seq [X, Y] [OUT]
    [silu_step X S ncols BLOCK_N, mul_step S Y OUT ncols BLOCK_N]

end Swiglu.kernels

/- Variables shared by every lemma and the theorem below — declared **once**, at
namespace scope, so both `Swiglu.lemmas` and `Swiglu.theorems` inherit them
(`end <section>` only clears variables declared *inside* that section). Each
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
(region holds `xs` at `s.pid * N + i`); output claims use its write-side
counterpart `VeriTile.Examples.outWritesTo`. `programLaneOffset` is that same
address, named for the proofs. -/

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

/-! ## Frame lemmas (the writes each kernel does NOT perform) -/

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

end Swiglu.lemmas

/-! ## The headline theorems -/
section Swiglu.theorems

/-- **fused refines unfused** (`ComputeRefine.Refines`, scratch `[S]`): for
every rounding model `R`, running the fused kernel and the unfused pipeline
from the same state performs THE SAME WRITES — the final memories agree at
every cell outside the scratch tensor `S`. Idempotence (`round ∘ round =
round`, a defining field of `RoundingModel`) is what makes the two kernels'
matched rounding sites coincide; no hypothesis is needed. -/
specification swiglu_fused_refines_unfused
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

/-! ## Trust audit (compile-time gate)

These commands re-audit the public result every time the file is elaborated —
if either gate fails (a smuggled axiom / `sorry`, or a foreign constant in the
trusted statement) the file stops compiling. See
`VeriTile.Meta.StatementAudit`. -/

-- (1) No `sorry`, no smuggled axiom, in the public theorem's transitive proof.
#axiomsClean swiglu_fused_refines_unfused

-- (2) The headline is a *kernel-vs-kernel* refinement: its statement may mention
-- ONLY the kernels, the loaded-input contract, the rounding-model surface, and
-- the library refinement surface — NO spec (there are none). If a spec-like
-- definition ever creeps into the statement, this fails.
#stmtSurfaceSubset swiglu_fused_refines_unfused ⊆
  [swiglu_fused, swiglu_unfused, InputLoadedAt, ComputeRefine.Refines,
   RoundingModel, BlockState, RegionName]

-- (There is no `#specNonCircular` gate: the file defines no specs at all, so a
-- self-referential spec is impossible by construction.)

end Swiglu.theorems

end VeriTile.Bench.Examples.SwigluRounding
