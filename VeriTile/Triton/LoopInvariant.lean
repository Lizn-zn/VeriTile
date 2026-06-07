/-
VeriTile.Triton.LoopInvariant

The forLoop induction lemma family. Master lemma `forLoop_inv` (spec §4.1)
plus ergonomics corollaries `forLoop_readout_scalar` (§4.2) and
`forLoop_readout_tile` (§4.3). See documents/ForLoopInvDesign.md
for the full interface design.
-/

import VeriTile.Triton.Core
import VeriTile.Triton.Semantics

namespace VeriTile.Triton

/-- Auxiliary form of `forLoop_inv` quantified over the starting index `i` and
    state `s`. The master `forLoop_inv` is the `i = 0` instance. -/
theorem forLoopAux_inv
    {idx : RegName} {n : Nat} {body : List Stmt}
    {P : Nat → BlockState → Prop}
    (h_step :
      ∀ i s, i < n → P i s →
        ∃ s',
          stepStmts body (s.setReg idx .nat [] (Tile.scalar i)) = some s' ∧
          P (i + 1) s') :
    ∀ i, i ≤ n → ∀ s, P i s →
      ∃ s_final,
        stepForLoopAux idx i n body s = some s_final ∧ P n s_final := by
  -- Induction on `n - i`. Equivalent to `Nat.le_induction` from `i = n`
  -- downward, but easier to phrase here.
  intro i hi s hPs
  -- Pull out the measure once.
  have key : ∀ k i, n - i = k → i ≤ n → ∀ s, P i s →
      ∃ s_final, stepForLoopAux idx i n body s = some s_final ∧ P n s_final := by
    intro k
    induction k with
    | zero =>
        -- n - i = 0 means i ≥ n; combined with i ≤ n we get i = n.
        intro i heq hi_le s hP
        have hi_eq : i = n := by omega
        subst hi_eq
        refine ⟨s, ?_, hP⟩
        exact stepForLoopAux.step_eq_self s
    | succ k ih =>
        -- n - i = k+1 ⇒ i < n.
        intro i heq hi_le s hP
        have hi_lt : i < n := by omega
        obtain ⟨s', h_body, hP'⟩ := h_step i s hi_lt hP
        have heq' : n - (i + 1) = k := by omega
        have hi'_le : i + 1 ≤ n := hi_lt
        obtain ⟨s_final, h_aux, hP_n⟩ := ih (i + 1) heq' hi'_le s' hP'
        refine ⟨s_final, ?_, hP_n⟩
        rw [stepForLoopAux.step_lt hi_lt]
        rw [h_body]
        simpa using h_aux
  exact key (n - i) i rfl hi s hPs

/-- **Master loop-induction lemma.** For any `forLoop` of body length `n`
    over register `idx`, given an entry invariant `P 0 s_init` and a step
    obligation showing each iteration preserves `P` (and does not error),
    the final state satisfies `P n`.

    Spec: `documents/ForLoopInvDesign.md` §4.1 (Form 1). -/
theorem forLoop_inv
    {idx : RegName} {n : Nat} {body : List Stmt}
    {P : Nat → BlockState → Prop} {s_init : BlockState}
    (h_init : P 0 s_init)
    (h_step :
      ∀ i s, i < n → P i s →
        ∃ s',
          stepStmts body (s.setReg idx .nat [] (Tile.scalar i)) = some s' ∧
          P (i + 1) s') :
    ∃ s_final,
      stepStmt (.forLoop idx n body) s_init = some s_final ∧
      P n s_final := by
  obtain ⟨s_final, h_aux, hP⟩ :=
    forLoopAux_inv (P := P) h_step 0 (Nat.zero_le _) s_init h_init
  refine ⟨s_final, ?_, hP⟩
  -- stepStmt (forLoop ...) = stepForLoopAux ... 0 n body s_init by definition.
  simpa [stepForLoopAux.forLoop_unfold] using h_aux

/-- **Scalar-register readout corollary.** Combines `forLoop_inv` with a
    proof that some output register holds a target scalar value when `P n`
    holds; the conclusion is the standard "kernel correctness reads
    register `outReg` and finds `f n`" form used throughout Tier 2 / 3-A. -/
theorem forLoop_readout_scalar
    {idx outReg : RegName} {n : Nat} {body : List Stmt}
    {P : Nat → BlockState → Prop} {s_init : BlockState}
    {f : Nat → ℝ}
    (h_init : P 0 s_init)
    (h_step :
      ∀ i s, i < n → P i s →
        ∃ s',
          stepStmts body (s.setReg idx .nat [] (Tile.scalar i)) = some s' ∧
          P (i + 1) s')
    (h_readout :
      ∀ s, P n s → s.regs .real [] outReg = some (Tile.scalar (f n))) :
    ∃ s_final,
      stepStmt (.forLoop idx n body) s_init = some s_final ∧
      s_final.regs .real [] outReg = some (Tile.scalar (f n)) := by
  obtain ⟨s_final, h_eq, hP⟩ := forLoop_inv h_init h_step
  exact ⟨s_final, h_eq, h_readout _ hP⟩

/-- **Tile-register readout corollary.** Same as `forLoop_readout_scalar` but
    for an output register containing a real vector tile; used by FA-1
    forward (`O` register) and any kernel writing a tile-valued accumulator.

    The readout is in `WithBot ℝ` to match `TileCarrier .real`. Callers whose
    `f` is `ℝ`-valued can compose with `some` at the call site. -/
theorem forLoop_readout_tile
    {idx outReg : RegName} {n : Nat} {body : List Stmt}
    {P : Nat → BlockState → Prop} {s_init : BlockState}
    {len : Nat} {f : Nat → Fin len → WithBot ℝ}
    (h_init : P 0 s_init)
    (h_step :
      ∀ i s, i < n → P i s →
        ∃ s',
          stepStmts body (s.setReg idx .nat [] (Tile.scalar i)) = some s' ∧
          P (i + 1) s')
    (h_readout :
      ∀ s, P n s → s.regs .real [len] outReg = some (Tile.vec (f n))) :
    ∃ s_final,
      stepStmt (.forLoop idx n body) s_init = some s_final ∧
      s_final.regs .real [len] outReg = some (Tile.vec (f n)) := by
  obtain ⟨s_final, h_eq, hP⟩ := forLoop_inv h_init h_step
  exact ⟨s_final, h_eq, h_readout _ hP⟩

/-- Auxiliary invariant principle for `forRange`.

The loop counter advances by an arbitrary positive `step`, so the final counter
is not definitionally `stop`; it is the first reached counter with
`stop ≤ final`. The invariant is therefore indexed by the current counter and
the conclusion exposes that final counter existentially. -/
theorem forRangeAux_inv
    {idx : RegName} {stop step : Nat} {body : List Stmt}
    {P : Nat → BlockState → Prop}
    (hstep : step ≠ 0)
    (h_step :
      ∀ i s, i < stop → P i s →
        ∃ s',
          stepStmts body (s.setReg idx .nat [] (Tile.scalar i)) = some s' ∧
          P (i + step) s') :
    ∀ cur s, P cur s →
      ∃ final s_final,
        stepForRangeAux idx cur stop step body s = some s_final ∧
        stop ≤ final ∧ P final s_final := by
  have hstep_pos : 0 < step := Nat.pos_of_ne_zero hstep
  have key :
      ∀ k cur s, stop - cur = k → P cur s →
        ∃ final s_final,
          stepForRangeAux idx cur stop step body s = some s_final ∧
          stop ≤ final ∧ P final s_final := by
    intro k
    induction k using Nat.strong_induction_on with
    | h k ih =>
      intro cur s hk hP
      by_cases hdone : stop ≤ cur
      · refine ⟨cur, s, ?_, hdone, hP⟩
        exact stepForRangeAux.step_ge hstep hdone
      · have hlt : cur < stop := Nat.lt_of_not_ge hdone
        obtain ⟨s', h_body, hP'⟩ := h_step cur s hlt hP
        have hmeasure : stop - (cur + step) < k := by omega
        obtain ⟨final, s_final, h_aux, hfinal, hPfinal⟩ :=
          ih (stop - (cur + step)) hmeasure (cur + step) s' rfl hP'
        refine ⟨final, s_final, ?_, hfinal, hPfinal⟩
        rw [stepForRangeAux.step_lt hstep hlt]
        rw [h_body]
        exact h_aux
  intro cur s hP
  exact key (stop - cur) cur s rfl hP

/-- Master invariant principle for static `forRange`. -/
theorem forRange_inv
    {idx : RegName} {start stop step : Nat} {body : List Stmt}
    {P : Nat → BlockState → Prop} {s_init : BlockState}
    (hstep : step ≠ 0)
    (h_init : P start s_init)
    (h_step :
      ∀ i s, i < stop → P i s →
        ∃ s',
          stepStmts body (s.setReg idx .nat [] (Tile.scalar i)) = some s' ∧
          P (i + step) s') :
    ∃ final s_final,
      stepStmt (.forRange idx start stop step body) s_init = some s_final ∧
      stop ≤ final ∧ P final s_final := by
  obtain ⟨final, s_final, h_aux, hfinal, hP⟩ :=
    forRangeAux_inv hstep h_step start s_init h_init
  refine ⟨final, s_final, ?_, hfinal, hP⟩
  simpa [stepForRangeAux.forRange_unfold] using h_aux

/-- **Master invariant principle for `forRangeDyn`.**

Like `forRange_inv`, but the loop bounds come from runtime `Op .nat []`
expressions evaluated against the entry state. The caller supplies `evalOp`
evidence resolving `start`/`stop`/`step` to concrete `Nat`s, plus the entry
invariant and per-iteration step obligation. The final counter is exposed
existentially (it is the first counter with `stop ≤ final`). -/
theorem forRangeDyn_inv
    {idx : RegName} {startOp stopOp stepOp : Op .nat []}
    {start stop step : Nat} {body : List Stmt}
    {P : Nat → BlockState → Prop} {s_init : BlockState}
    (hStart : evalOp startOp s_init = some (Tile.scalar start))
    (hStop : evalOp stopOp s_init = some (Tile.scalar stop))
    (hStepOp : evalOp stepOp s_init = some (Tile.scalar step))
    (hstep : step ≠ 0)
    (h_init : P start s_init)
    (h_step :
      ∀ i s, i < stop → P i s →
        ∃ s',
          stepStmts body (s.setReg idx .nat [] (Tile.scalar i)) = some s' ∧
          P (i + step) s') :
    ∃ final s_final,
      stepStmt (.forRangeDyn idx startOp stopOp stepOp body) s_init = some s_final ∧
      stop ≤ final ∧ P final s_final := by
  obtain ⟨final, s_final, h_aux, hfinal, hP⟩ :=
    forRangeAux_inv hstep h_step start s_init h_init
  refine ⟨final, s_final, ?_, hfinal, hP⟩
  rw [stepForRangeAux.forRangeDyn_unfold, hStart, hStop, hStepOp]
  simp only [Option.bind_some]
  show stepForRangeAux idx start stop step body s_init = some s_final
  exact h_aux

/-- Scalar-register readout corollary for static `forRange`. -/
theorem forRange_readout_scalar
    {idx outReg : RegName} {start stop step : Nat} {body : List Stmt}
    {P : Nat → BlockState → Prop} {s_init : BlockState}
    {f : Nat → ℝ}
    (hstep : step ≠ 0)
    (h_init : P start s_init)
    (h_step :
      ∀ i s, i < stop → P i s →
        ∃ s',
          stepStmts body (s.setReg idx .nat [] (Tile.scalar i)) = some s' ∧
          P (i + step) s')
    (h_readout :
      ∀ final s, stop ≤ final → P final s →
        s.regs .real [] outReg = some (Tile.scalar (f final))) :
    ∃ final s_final,
      stepStmt (.forRange idx start stop step body) s_init = some s_final ∧
      stop ≤ final ∧
      s_final.regs .real [] outReg = some (Tile.scalar (f final)) := by
  obtain ⟨final, s_final, h_eq, hfinal, hP⟩ :=
    forRange_inv hstep h_init h_step
  exact ⟨final, s_final, h_eq, hfinal, h_readout final s_final hfinal hP⟩

/-- Tile-register readout corollary for static `forRange`. -/
theorem forRange_readout_tile
    {idx outReg : RegName} {start stop step : Nat} {body : List Stmt}
    {P : Nat → BlockState → Prop} {s_init : BlockState}
    {len : Nat} {f : Nat → Fin len → WithBot ℝ}
    (hstep : step ≠ 0)
    (h_init : P start s_init)
    (h_step :
      ∀ i s, i < stop → P i s →
        ∃ s',
          stepStmts body (s.setReg idx .nat [] (Tile.scalar i)) = some s' ∧
          P (i + step) s')
    (h_readout :
      ∀ final s, stop ≤ final → P final s →
        s.regs .real [len] outReg = some (Tile.vec (f final))) :
    ∃ final s_final,
      stepStmt (.forRange idx start stop step body) s_init = some s_final ∧
      stop ≤ final ∧
      s_final.regs .real [len] outReg = some (Tile.vec (f final)) := by
  obtain ⟨final, s_final, h_eq, hfinal, hP⟩ :=
    forRange_inv hstep h_init h_step
  exact ⟨final, s_final, h_eq, hfinal, h_readout final s_final hfinal hP⟩

/-- Single-iteration corollary for `forRange`. When `start < stop ≤ start + step`,
the range `[start, stop)` contains exactly one step-aligned index (`start`),
so the loop executes the body once and terminates.

This special case avoids constructing an inductive `P` for kernels that the
Python test only ever drives through a single chunk (e.g. `var_len_copy` with
`length ≤ BLOCK_SIZE`, or single-block rmsnorm/layernorm regimes). -/
theorem forRange_single_step
    {idx : RegName} {start stop step : Nat} {body : List Stmt}
    {s_init s_body : BlockState}
    (hstep : step ≠ 0)
    (hlt : start < stop)
    (hle : stop ≤ start + step)
    (hBody :
      stepStmts body (s_init.setReg idx .nat [] (Tile.scalar start))
        = some s_body) :
    stepStmt (.forRange idx start stop step body) s_init = some s_body := by
  rw [stepForRangeAux.forRange_unfold, stepForRangeAux.step_lt hstep hlt,
      hBody]
  simp [Option.bind, stepForRangeAux.step_ge hstep hle]

/-- **Master invariant principle for dynamic `forRangeDyn`.**

The mirror of `forRange_inv` for loops whose bounds come from runtime register
values. The caller supplies the resolved `start`/`stop`/`step` Nats together
with the corresponding `evalOp` evidence; the loop then runs the same
`stepForRangeAux` machinery as the static `forRange`, so the rest of the
interface (positive step, entry invariant, per-iteration step obligation,
existential final counter) is identical.

This unblocks any kernel whose iteration bound is `cdiv(K, BLOCK_K)` or another
runtime-computed value (e.g. GEMM K-loops, chunk-cumsum chunk loops). -/
theorem forRangeDyn_inv
    {idx : RegName} {startOp stopOp stepOp : Op .nat []}
    {start stop step : Nat} {body : List Stmt}
    {P : Nat → BlockState → Prop} {s_init : BlockState}
    (hStart : evalOp startOp s_init = some (Tile.scalar start))
    (hStop : evalOp stopOp s_init = some (Tile.scalar stop))
    (hStepOp : evalOp stepOp s_init = some (Tile.scalar step))
    (hstep : step ≠ 0)
    (h_init : P start s_init)
    (h_step :
      ∀ i s, i < stop → P i s →
        ∃ s',
          stepStmts body (s.setReg idx .nat [] (Tile.scalar i)) = some s' ∧
          P (i + step) s') :
    ∃ final s_final,
      stepStmt (.forRangeDyn idx startOp stopOp stepOp body) s_init = some s_final ∧
      stop ≤ final ∧ P final s_final := by
  obtain ⟨final, s_final, h_aux, hfinal, hP⟩ :=
    forRangeAux_inv hstep h_step start s_init h_init
  refine ⟨final, s_final, ?_, hfinal, hP⟩
  rw [stepForRangeAux.forRangeDyn_unfold, hStart, hStop, hStepOp]
  simpa [Option.bind] using h_aux

/-- Single-iteration variant for `forRangeDyn` after the dynamic bounds have
been resolved. Use when start/stop/step come from runtime register values
(e.g. `length` loaded from a metadata buffer). The caller must supply the
resolved Nat values and the corresponding evalOp evidence. -/
theorem forRangeDyn_single_step
    {idx : RegName} {startOp stopOp stepOp : Op .nat []}
    {start stop step : Nat} {body : List Stmt}
    {s_init s_body : BlockState}
    (hStart : evalOp startOp s_init = some (Tile.scalar start))
    (hStop : evalOp stopOp s_init = some (Tile.scalar stop))
    (hStepOp : evalOp stepOp s_init = some (Tile.scalar step))
    (hstep : step ≠ 0)
    (hlt : start < stop)
    (hle : stop ≤ start + step)
    (hBody :
      stepStmts body (s_init.setReg idx .nat [] (Tile.scalar start))
        = some s_body) :
    stepStmt (.forRangeDyn idx startOp stopOp stepOp body) s_init = some s_body := by
  rw [stepForRangeAux.forRangeDyn_unfold, hStart, hStop, hStepOp]
  simp only [Option.bind]
  show stepForRangeAux idx start stop step body s_init = some s_body
  rw [stepForRangeAux.step_lt hstep hlt, hBody]
  simp [Option.bind, stepForRangeAux.step_ge hstep hle]

/-- **Master invariant principle for dynamic `forRangeDyn`.**

The mirror of `forRange_inv` for loops whose bounds come from runtime register
values. The caller supplies the resolved `start`/`stop`/`step` Nats together
with the corresponding `evalOp` evidence; the loop then runs the same
`stepForRangeAux` machinery as the static `forRange`, so the rest of the
interface (positive step, entry invariant, per-iteration step obligation,
existential final counter) is identical.

This unblocks any kernel whose iteration bound is `cdiv(K, BLOCK_K)` or another
runtime-computed value (e.g. GEMM K-loops). -/
theorem forRangeDyn_inv
    {idx : RegName} {startOp stopOp stepOp : Op .nat []}
    {start stop step : Nat} {body : List Stmt}
    {P : Nat → BlockState → Prop} {s_init : BlockState}
    (hStart : evalOp startOp s_init = some (Tile.scalar start))
    (hStop : evalOp stopOp s_init = some (Tile.scalar stop))
    (hStepOp : evalOp stepOp s_init = some (Tile.scalar step))
    (hstep : step ≠ 0)
    (h_init : P start s_init)
    (h_step :
      ∀ i s, i < stop → P i s →
        ∃ s',
          stepStmts body (s.setReg idx .nat [] (Tile.scalar i)) = some s' ∧
          P (i + step) s') :
    ∃ final s_final,
      stepStmt (.forRangeDyn idx startOp stopOp stepOp body) s_init = some s_final ∧
      stop ≤ final ∧ P final s_final := by
  obtain ⟨final, s_final, h_aux, hfinal, hP⟩ :=
    forRangeAux_inv hstep h_step start s_init h_init
  refine ⟨final, s_final, ?_, hfinal, hP⟩
  rw [stepForRangeAux.forRangeDyn_unfold, hStart, hStop, hStepOp]
  simpa [Option.bind] using h_aux

end VeriTile.Triton
