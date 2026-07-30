import VeriTile.Triton

/-!
# `rmsnorm_fused` — strict per-kernel correctness

`rms_norm_fwd_fused` is a fused RMSNorm forward: each program `row` normalizes
one row of `X` by its root-mean-square, scales by per-column weights `W`, and
writes `Y[row] = (x / sqrt(mean(x²) + eps)) * w`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body, for one program (one row). The host launch
(`rms_norm_fwd_fused[(M,)](...)`, the grid over rows `M`, the host-side
`BLOCK_SIZE` choice, scheduling, and how the runtime composes per-row writes
into one buffer) is the *trusted boundary*, not a proof obligation here. Because
`row = tl.program_id(0)` is universally quantified (via `s.pid`), the per-program
statement covers every row of the grid.

## Proof architecture

```
rms_norm_fwd_fused_output_summary             ← TOP THEOREM
  ├─ (toAlgorithm? = Except.ok _)             surface lowers to the algorithm layer
  └─ rms_norm_fwd_fused_compute_correct       ← ComputeCorrect over the masked store
       └─ rms_norm_fwd_fused_correct          ← algorithm-layer readback per lane
            └─ rmsnormCarrierSpec_eq_rmsnormSpec
                 └─ rmsVarCarrier_eq_rmsMeanSq (uses VeriTile.Triton.Math.RMSNorm /
                                                TiledL2Norm reduction lemmas)
```

The row spec is `TiledRMSNorm.rmsAffine` from `VeriTile.Triton.Math.RMSNorm`
(reusing `TiledRMSNorm.rmsMeanSq` / `TiledL2Norm` reduction lemmas rather than
inlining the reduction math).

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); the `.to(tl.float32)`
casts reduce to identity at the algorithm layer (post-erasure all dtypes unify
to `ℝ`). The reduction `tl.sum(_var) / N` sums over the *padded* `BLOCK_SIZE`
block, but out-of-range lanes are masked to `0` (load `other=0.0` plus the
explicit `tl.where`), so the sum equals the logical row length `N`. The
reciprocal std is `rstd = 1 / sqrt(meanSq + eps)`; the affine step is
`x_hat * w`. The Python wrapper picks `BLOCK_SIZE ≥ N` (raising otherwise), so
correctness is stated under the `0 < N ≤ BLOCK_SIZE` precondition, where both
`range(0, N, BLOCK_SIZE)` loops execute exactly the `off = 0` iteration (single
block).
-/

namespace VeriTile.Bench.TritonBenchG.RmsnormFused

open VeriTile.Triton

set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

/-- Faithful `forRange` transcription of `rmsnorm_fused.py`'s
`rms_norm_fwd_fused`.

The Python wrapper chooses `BLOCK_SIZE >= N` and raises otherwise, so the
correctness theorem below proves the full loop-shaped kernel under that
runtime precondition. Under the precondition both `range(0, N, BLOCK_SIZE)`
loops execute exactly the `off = 0` iteration.

Allowed mechanical Lean-syntax-only changes:
- Python `N` / `BLOCK_SIZE: tl.constexpr` -> Lean `Nat` parameters. -/
def rms_norm_fwd_fused
    (X Y W : RegionName) (stride N BLOCK_SIZE : Nat) (eps : ℝ) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  Y += row * $(stride)
  X += row * $(stride)
  _var = tl.zeros([$(BLOCK_SIZE)], dtype=tl.float32)
  for off in range(0, $(N), $(BLOCK_SIZE)) {
    cols = off + tl.arange(0, $(BLOCK_SIZE))
    x = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
    x = tl.where(cols < $(N), x, 0.0)
    _var += x * x
  }
  var = tl.sum(_var, axis=0) / $(N)
  rstd = 1 / tl.sqrt(var + $(eps))
  for off in range(0, $(N), $(BLOCK_SIZE)) {
    cols = off + tl.arange(0, $(BLOCK_SIZE))
    mask = cols < $(N)
    w = tl.load(W + cols, mask=mask)
    x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
    x_hat = x * rstd
    y = x_hat * w
    tl.store(Y + cols, y, mask=mask)
  }
}

def xOffset (s : BlockState) (stride : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pid * stride + i.val

def yOffset (s : BlockState) (stride : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pid * stride + i.val

noncomputable def rmsInputTile
    (s : BlockState) (X : RegionName) (stride N BLOCK_SIZE : Nat) :
    Tile .real [BLOCK_SIZE] :=
  Tile.maskedRowTile
    (fun i : Fin BLOCK_SIZE =>
      Tile.maskedRowLoad
        (fun k : Fin BLOCK_SIZE => some (s.readMem X (xOffset s stride k)))
        (fun k : Fin BLOCK_SIZE => k.val < N)
        (some (0.0 : ℝ) : WithBot ℝ)
        i)
    (fun i : Fin BLOCK_SIZE => i.val < N)
    (some (0.0 : ℝ) : WithBot ℝ)

noncomputable def rmsLoad
    (s : BlockState) (X : RegionName) (stride N BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) : ℝ :=
  Tile.maskedRowLoad
    (fun k : Fin BLOCK_SIZE => s.readMem X (xOffset s stride k))
    (fun k : Fin BLOCK_SIZE => k.val < N)
    0
    i

noncomputable def rmsWeight
    (s : BlockState) (W : RegionName) (i : Fin BLOCK_SIZE) : ℝ :=
  s.readMem W i.val

noncomputable def rmsVarCarrier
    (s : BlockState) (X : RegionName) (stride N BLOCK_SIZE : Nat) :
    WithBot ℝ :=
  Option.map₂ (fun a n => a / n)
    ((Tile.reduceSum (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false
      (Tile.bop (NumericDType.mul .real) (Broadcast.consSame Broadcast.nil)
        (rmsInputTile s X stride N BLOCK_SIZE)
        (rmsInputTile s X stride N BLOCK_SIZE))).data PUnit.unit)
    ((Tile.scalar (dtype := .real) (some (N : ℝ) : WithBot ℝ)).data PUnit.unit)

theorem rmsVarCarrier_eq_rmsMeanSq
    (s : BlockState) (X : RegionName) (stride N BLOCK_SIZE : Nat) :
    rmsVarCarrier s X stride N BLOCK_SIZE =
      some (TiledRMSNorm.rmsMeanSq (rmsLoad s X stride N BLOCK_SIZE) N) := by
  simp [rmsVarCarrier, rmsInputTile, Tile.reduceSum, Tile.reduceSumDrop,
    TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
    Tile.bop, NumericDType.mul]
  have hsum := TiledL2Norm.reduceSum_masked_sq_eq_some_sum
    (fun k : Fin BLOCK_SIZE => s.readMem X (xOffset s stride k))
    (fun k : Fin BLOCK_SIZE => k.val < N)
  refine (congrArg
    (fun a : WithBot ℝ => Option.map (fun x : ℝ => x / (N : ℝ)) a)
    hsum).trans ?_
  simp [TiledRMSNorm.rmsMeanSq, TiledL2Norm.l2NormSqSum, rmsLoad,
    Tile.maskedRowLoad]

noncomputable def rmsInvCarrier
    (s : BlockState) (X : RegionName) (stride N BLOCK_SIZE : Nat)
    (eps : ℝ) : WithBot ℝ :=
  Option.map (fun b => b⁻¹)
    (WithBot.realSqrt
      (Option.map (fun a => a + eps)
        (rmsVarCarrier s X stride N BLOCK_SIZE)))

noncomputable def rmsnormCarrierSpec
    (s : BlockState) (X W : RegionName)
    (stride N BLOCK_SIZE : Nat) (eps : ℝ) (i : Fin BLOCK_SIZE) : ℝ :=
  WithBot.unbotD 0
    (Option.map₂ (fun scaled w => scaled * w)
      (Option.map₂ (fun x inv => x * inv)
        (some (s.readMem X (xOffset s stride i)))
        (rmsInvCarrier s X stride N BLOCK_SIZE eps))
      (some (s.readMem W i.val)))

noncomputable def rmsnormSpec
    (s : BlockState) (X W : RegionName)
    (stride N BLOCK_SIZE : Nat) (eps : ℝ) (i : Fin BLOCK_SIZE) : ℝ :=
  TiledRMSNorm.rmsAffine
    (rmsLoad s X stride N BLOCK_SIZE)
    (rmsWeight s W)
    N eps i

theorem rmsnormCarrierSpec_eq_rmsnormSpec
    (s : BlockState) (X W : RegionName)
    (stride N BLOCK_SIZE : Nat) (eps : ℝ) (i : Fin BLOCK_SIZE)
    (hi : i.val < N) :
    rmsnormCarrierSpec s X W stride N BLOCK_SIZE eps i =
      rmsnormSpec s X W stride N BLOCK_SIZE eps i := by
  unfold rmsnormCarrierSpec rmsnormSpec rmsInvCarrier
  rw [rmsVarCarrier_eq_rmsMeanSq]
  simp [TiledRMSNorm.rmsAffine, TiledRMSNorm.rmsRstd,
    rmsLoad, rmsWeight, hi]

/-- Algorithm-layer correctness for the fused RMSNorm kernel under the Python
wrapper's `N <= BLOCK_SIZE` launch precondition. -/
theorem rms_norm_fwd_fused_correct
    (X Y W : RegionName) (stride N BLOCK_SIZE : Nat) (eps : ℝ)
    (s s' : BlockState)
    (hNpos : 0 < N) (hNle : N ≤ BLOCK_SIZE)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => yOffset s stride i))
    (hExec : exec (rms_norm_fwd_fused X Y W stride N BLOCK_SIZE eps) s =
        some s') :
    ∀ i : Fin BLOCK_SIZE,
      s'.readMem Y (yOffset s stride i) =
        if i.val < N then
          rmsnormSpec s X W stride N BLOCK_SIZE eps i
        else s.readMem Y (yOffset s stride i) := by
  intro i
  by_cases hB : 0 < BLOCK_SIZE
  · have hStep : BLOCK_SIZE ≠ 0 := Nat.ne_of_gt hB
    simp [exec, rms_norm_fwd_fused, stepStmts, stepStmt, evalOp, evalOp.eq_def,
          ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
          stepForRangeAux.step_lt, stepForRangeAux.step_ge,
          Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop, Tile.reduceSum,
          Tile.reduceSumDrop, Tile.select, TileShape.axisDim, TileShape.eraseAxis,
          TileShape.insertAxisIndex, NumericDType.add, NumericDType.mul,
          NumericDType.div, ComparableDType.lt, hNpos, hNle,
          Nat.not_lt.mpr hNle, hStep] at hExec
    subst s'
    simp only [yOffset]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
          (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
    by_cases hi : i.val < N
    · rw [← rmsnormCarrierSpec_eq_rmsnormSpec s X W stride N BLOCK_SIZE eps i hi]
      simp only [hi, ↓reduceIte]
      simp [hi, rmsnormCarrierSpec, rmsInvCarrier, rmsVarCarrier, rmsInputTile,
            xOffset, Tile.reduceSum, Tile.reduceSumDrop, Tile.select,
            TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
            WithBot.realSqrt, NumericDType.mul, FloatDType.cast]
      rfl
    · simp [hi]
  · exact False.elim (hB (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Compute-facing correctness for the fused RMSNorm kernel under the Python
wrapper's `N <= BLOCK_SIZE` launch precondition. -/
theorem rms_norm_fwd_fused_compute_correct
    (X Y W : RegionName) (stride N BLOCK_SIZE : Nat) (eps : ℝ)
    (s : BlockState)
    (hNpos : 0 < N) (hNle : N ≤ BLOCK_SIZE)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => yOffset s stride i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := rms_norm_fwd_fused X Y W stride N BLOCK_SIZE eps)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => i.val < N)
        (fun i => (Y, yOffset s stride i)))
      (expected := fun i => rmsnormSpec s X W stride N BLOCK_SIZE eps i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [rms_norm_fwd_fused, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := rms_norm_fwd_fused_correct X Y W stride N BLOCK_SIZE eps
    s s' hNpos hNle hOutInj hExec i
  simpa [hActive] using h

/-- Per-kernel output summary for `rms_norm_fwd_fused`: the DSL surface lowers to
the algorithm layer, and the masked store to `Y` is compute-correct — every
active lane (`i.val < N`) holds the RMSNorm spec `rmsnormSpec`, out-of-bounds
lanes are preserved. Stated under the `0 < N ≤ BLOCK_SIZE` single-block launch
precondition chosen by the Python wrapper. -/
specification rms_norm_fwd_fused_output_summary
    (X Y W : RegionName) (stride N BLOCK_SIZE : Nat) (eps : ℝ)
    (s : BlockState)
    (hNpos : 0 < N) (hNle : N ≤ BLOCK_SIZE)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => yOffset s stride i)) :
    (∃ alg, (rms_norm_fwd_fused X Y W stride N BLOCK_SIZE eps).toAlgorithm? =
        Except.ok alg) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel := rms_norm_fwd_fused X Y W stride N BLOCK_SIZE eps)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => i.val < N)
        (fun i => (Y, yOffset s stride i)))
      (expected := fun i => rmsnormSpec s X W stride N BLOCK_SIZE eps i) := by
  refine ⟨?_, ?_⟩
  · simp only [rms_norm_fwd_fused, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
    exact ⟨_, rfl⟩
  · exact rms_norm_fwd_fused_compute_correct X Y W stride N BLOCK_SIZE eps
      s hNpos hNle hOutInj

/-! ## The `⊨[R]` streaming headline (wave-5 S3 per-step emit genre)

Everything below is purely additive; the exact surface above is untouched.
This is a consumer of the per-step emit skin `StreamEmitMasked2DKernelIO₂`
(streaming genre, style S3): the store sits **inside** the second
`for off in range(0, N, BLOCK_SIZE)` pass, so the output is a per-step
`BLOCK_SIZE`-lane window family rather than one terminal tile, and the
kernel's spec `f t j` is the genre's *two-pass* shape — the step-`t` tile
combined with a fold over the entire stream.

The launch regime is the exact headline's: the Python wrapper picks
`BLOCK_SIZE ≥ N` (raising otherwise), so under `0 < N ≤ BLOCK_SIZE` both
`forRange` clauses execute exactly the `off = 0` iteration and the stream
has a single step (`T := 1`).

Structure of the `execR R` story: this kernel has **zero rounding events**.
Every load and store is at `.real`, and the `.to(tl.float32)` casts are
erased outright by the compute-to-algorithm lowering (no `Op.castFloat`
survives — see `fused_body_decomp`), so both passes step identically under
`stepStmtsR R` and `stepStmts` (`fused_castFree`) and the exact
`rms_norm_fwd_fused_correct` readback transports to `execR R` verbatim. The
skin's readback contract at the default `outDType := .real` grid carries
`R.round .real`, the identity by the model's defining `round_real` — the
∀-`R` face is the exact streaming contract via the `RoundingModel` `.real`
identity fields, not a `.triv` special case. -/

section IOFace

open scoped VeriTile.Triton.StreamEmitMasked2DKernelIO₂

set_option linter.unusedVariables false

/-! ### The lowered body, decomposed

The four segments of the lowered kernel body, named so the `forRange`
clauses can be lifted to `execR R` one body at a time. `fused_body_decomp`
is `rfl`: this *is* the algorithm the DSL surface lowers to. Note that the
two `.to(tl.float32)` casts leave no trace — `ComputeDType.fp32.eraseDType`
is `TileDType.real`, so the lowered body is literally cast-free. -/

/-- The prologue: `row`, the two row-base pointer bumps, and the `_var`
accumulator seed. -/
private def fusedPre (X Y : RegionName) (stride B : Nat) : List Stmt :=
  [ Stmt.assign .nat [] "row" (Op.programId 0),
    Stmt.assign .ptr [] "Y"
      (Op.ptrAdd Broadcast.nil (Op.ptrBase Y)
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "row") (Op.constNat stride))),
    Stmt.assign .ptr [] "X"
      (Op.ptrAdd Broadcast.nil (Op.ptrBase X)
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "row") (Op.constNat stride))),
    Stmt.assign .real [B] "_var" (Op.full [B] (Op.const 0)) ]

/-- Pass 1's body: masked `x` load, the `tl.where` re-zeroing, and the
`_var += x * x` fold. Register-only apart from the masked load. -/
private def fusedVarBody (N B : Nat) : List Stmt :=
  [ Stmt.assign .nat [B] "cols"
      (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "off") (Op.arange B)),
    Stmt.assign .real [B] "x"
      (Op.load .real
        (MemAccess.ptr
          (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "X") (Op.ref .nat [B] "cols")))
        (MaskOpt.maskOther
          (Op.lt .nat Broadcast.scalarR (Op.ref .nat [B] "cols") (Op.constNat N))
          ((Op.const 0.0).broadcast [B]))),
    Stmt.assign .real [B] "x"
      ((Op.lt .nat Broadcast.scalarR (Op.ref .nat [B] "cols") (Op.constNat N)).where
        (Op.ref .real [B] "x") ((Op.const 0.0).broadcast [B])),
    Stmt.assign .real [B] "_var"
      (Op.add .real (Broadcast.consSame Broadcast.nil) (Op.ref .real [B] "_var")
        (Op.mul .real (Broadcast.consSame Broadcast.nil) (Op.ref .real [B] "x")
          (Op.ref .real [B] "x"))) ]

/-- The reduce / reciprocal-sqrt tail between the two passes. -/
private def fusedPost (N B : Nat) (eps : ℝ) : List Stmt :=
  [ Stmt.assign .real [] "var"
      (Op.div .real Broadcast.nil
        (Op.reduceSum (shape := [B]) ⟨0, by simp⟩ Bool.false (Op.ref .real [B] "_var"))
        (Op.const (N : ℝ))),
    Stmt.assign .real [] "rstd"
      (Op.div .real Broadcast.nil (Op.const 1)
        (Op.sqrt (Op.add .real Broadcast.nil (Op.ref .real [] "var") (Op.const eps)))) ]

/-- Pass 2's body — the **emit** body: two masked loads, the affine scaling,
and the in-loop masked `.real` store into `Y`. -/
private def fusedWbBody (W : RegionName) (N B : Nat) : List Stmt :=
  [ Stmt.assign .nat [B] "cols"
      (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "off") (Op.arange B)),
    Stmt.assign .bool [B] "mask"
      (Op.lt .nat Broadcast.scalarR (Op.ref .nat [B] "cols") (Op.constNat N)),
    Stmt.assign .real [B] "w"
      (Op.load .real (MemAccess.region W (Op.ref .nat [B] "cols"))
        (MaskOpt.mask (Op.ref .bool [B] "mask"))),
    Stmt.assign .real [B] "x"
      (Op.load .real
        (MemAccess.ptr
          (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "X") (Op.ref .nat [B] "cols")))
        (MaskOpt.maskOther (Op.ref .bool [B] "mask")
          ((Op.const 0.0).broadcast [B]))),
    Stmt.assign .real [B] "x_hat"
      (Op.mul .real Broadcast.scalarR (Op.ref .real [B] "x") (Op.ref .real [] "rstd")),
    Stmt.assign .real [B] "y"
      (Op.mul .real (Broadcast.consSame Broadcast.nil) (Op.ref .real [B] "x_hat")
        (Op.ref .real [B] "w")),
    Stmt.store .real [B]
      (MemAccess.ptr
        (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "Y") (Op.ref .nat [B] "cols")))
      (Op.ref .real [B] "y") (MaskOpt.mask (Op.ref .bool [B] "mask")) ]

/-- The lowered body is `prologue ++ pass 1 ++ tail ++ pass 2`, definitionally. -/
private theorem fused_body_decomp (X Y W : RegionName) (stride N B : Nat) (eps : ℝ) :
    (rms_norm_fwd_fused X Y W stride N B eps).toAlgKernel.body
      = fusedPre X Y stride B
        ++ [Stmt.forRange "off" 0 N B (fusedVarBody N B)]
        ++ fusedPost N B eps
        ++ [Stmt.forRange "off" 0 N B (fusedWbBody W N B)] := rfl

/-! ### Cast-free collapses and the covered fragment -/

/-- A statement list whose statements each step identically under
`stepStmtR R` and `stepStmt` steps identically as a list. -/
private theorem stepStmtsR_congr (R : RoundingModel) :
    ∀ (l : List Stmt),
      (∀ st ∈ l, ∀ t : BlockState, stepStmtR R st t = stepStmt st t) →
      ∀ t : BlockState, stepStmtsR R l t = stepStmts l t
  | [], _, _ => by simp only [stepStmtsR, stepStmts]
  | st :: rest, h, t => by
      simp only [stepStmtsR, stepStmts, h st List.mem_cons_self t]
      cases stepStmt st t with
      | none => rfl
      | some t' =>
          exact stepStmtsR_congr R rest
            (fun a ha => h a (List.mem_cons_of_mem st ha)) t'

/-- Pass 1's body is cast-free: `evalOpR R` is `evalOp` on it. -/
private theorem fusedVarBody_castFree (R : RoundingModel) (N B : Nat)
    (t : BlockState) :
    stepStmtsR R (fusedVarBody N B) t = stepStmts (fusedVarBody N B) t := by
  simp only [fusedVarBody, stepStmtsR, stepStmts, stepStmtR, stepStmt,
    evalOpR.eq_def, evalOp.eq_def]
  rfl

/-- Pass 2's body is cast-free **including its in-loop masked `.real`
store**: `stepStmtR` delegates a `.real`-typed store to the exact
`writeMemTyped` (`writeMemTypedR R .real` is definitionally the exact
write), so the whole emitting loop steps identically under `stepStmtsR R`. -/
private theorem fusedWbBody_castFree (R : RoundingModel) (W : RegionName)
    (N B : Nat) (t : BlockState) :
    stepStmtsR R (fusedWbBody W N B) t = stepStmts (fusedWbBody W N B) t := by
  simp only [fusedWbBody, stepStmtsR, stepStmts, stepStmtR, stepStmt,
    evalOpR.eq_def, evalOp.eq_def, BlockState.writeMemTypedR]
  rfl

/-- The whole lowered body is cast-free: the four register-only prologue /
tail assigns collapse pointwise, and the two `forRange` clauses lift their
cast-free bodies through `stepForRangeAuxR_castFree`. -/
private theorem fused_castFree (R : RoundingModel) (X Y W : RegionName)
    (stride N B : Nat) (eps : ℝ) (t : BlockState) :
    stepStmtsR R (rms_norm_fwd_fused X Y W stride N B eps).toAlgKernel.body t
      = stepStmts (rms_norm_fwd_fused X Y W stride N B eps).toAlgKernel.body t := by
  revert t
  rw [fused_body_decomp]
  refine stepStmtsR_congr R _ ?_
  intro st hst
  simp only [fusedPre, fusedPost, List.append_assoc, List.singleton_append,
    List.cons_append, List.nil_append, List.mem_cons, List.not_mem_nil,
    or_false] at hst
  rcases hst with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
  · intro t; simp only [stepStmtR, stepStmt, evalOpR.eq_def, evalOp.eq_def]
  · intro t; simp only [stepStmtR, stepStmt, evalOpR.eq_def, evalOp.eq_def]
  · intro t; simp only [stepStmtR, stepStmt, evalOpR.eq_def, evalOp.eq_def]
  · intro t; simp only [stepStmtR, stepStmt, evalOpR.eq_def, evalOp.eq_def]
  · intro t
    rw [stepStmtR_forRange,
      stepForRangeAuxR_castFree R _ (fusedVarBody_castFree R N B) "off",
      ← stepForRangeAux.forRange_unfold]
  · intro t; simp only [stepStmtR, stepStmt, evalOpR.eq_def, evalOp.eq_def]
  · intro t; simp only [stepStmtR, stepStmt, evalOpR.eq_def, evalOp.eq_def]
  · intro t
    rw [stepStmtR_forRange,
      stepForRangeAuxR_castFree R _ (fusedWbBody_castFree R W N B) "off",
      ← stepForRangeAux.forRange_unfold]

/-- `execR R` on the fused kernel **is** the exact `exec`. -/
private theorem fused_execR_eq (R : RoundingModel) (X Y W : RegionName)
    (stride N B : Nat) (eps : ℝ) (t : BlockState) :
    execR R (rms_norm_fwd_fused X Y W stride N B eps).toAlgKernel t
      = exec (rms_norm_fwd_fused X Y W stride N B eps).toAlgKernel t :=
  fused_castFree R X Y W stride N B eps t

/-- The two-pass surface sits inside the flat-memory bridge's covered
fragment. -/
theorem rms_norm_fwd_fused_flattenOk (X Y W : RegionName)
    (stride N B : Nat) (eps : ℝ) :
    ((rms_norm_fwd_fused X Y W stride N B eps).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  rw [fused_body_decomp]
  simp [fusedPre, fusedVarBody, fusedPost, fusedWbBody, StmtList.FlattenOk,
    Stmt.FlattenOk, Op.FlattenOk.eq_def]

/-! ### IO signature -/

/-- **Streaming IO signature** of `rms_norm_fwd_fused` on the two-stream
per-step emit skin (S3: in-loop store), in the Python wrapper's single-block
launch regime `0 < N ≤ BLOCK_SIZE`: both `range(0, N, BLOCK_SIZE)` passes run
exactly the `off = 0` iteration, so the stream has `T := 1` step of
`BLOCK_SIZE` lanes. Step `t` reads the `BLOCK_SIZE`-lane `x` tile (`read1`,
both passes read the same addresses) and the `w` tile (`read2`, pass 2 only);
step `t` of pass 2 stores the `BLOCK_SIZE`-lane output window (`write`) at
the **`.real`** grid — the kernel's store is the untyped
`tl.store(Y + cols, y, mask=mask)`, which lowers to
`Stmt.store TileDType.real` (see `fusedWbBody`), so the per-step stores carry
no quantization event. The windows transcribe the kernel's pointer arithmetic
verbatim (`cols = off + tl.arange(0, BLOCK_SIZE)`, i.e.
`t·BLOCK_SIZE + j`, against the row-bumped `X`/`Y` pointers):

* `read1` step `t`, lane `j`: `pid₀·stride + (t·BLOCK_SIZE + j)` — the
  kernel's `X + cols` after `X += row * stride`.
* `read2` step `t`, lane `j`: `t·BLOCK_SIZE + j` — the kernel's `W + cols`
  (`W` is not row-bumped).
* `write` step `t`, lane `j`: `pid₀·stride + (t·BLOCK_SIZE + j)` — the
  kernel's `Y + cols` after `Y += row * stride`.

All three masks are the kernel's single `cols < N`. The second pid is inert
(the launch grid is 1-D). -/
def rmsnormFusedKernelIO (X Y W : RegionName) (stride N BLOCK_SIZE : Nat)
    (eps : ℝ) : StreamEmitMasked2DKernelIO₂ where
  kernel := rms_norm_fwd_fused X Y W stride N BLOCK_SIZE eps
  inp1 := X
  inp2 := W
  out := Y
  T := 1
  B1 := BLOCK_SIZE
  B2 := BLOCK_SIZE
  C := BLOCK_SIZE
  outDType := .real
  read1 := fun p₀ _ t j => p₀ * stride + (t.val * BLOCK_SIZE + j.val)
  read2 := fun _ _ t j => t.val * BLOCK_SIZE + j.val
  write := fun p₀ _ t j => p₀ * stride + (t.val * BLOCK_SIZE + j.val)
  mask1 := fun _ _ t j => t.val * BLOCK_SIZE + j.val < N
  mask2 := fun _ _ t j => t.val * BLOCK_SIZE + j.val < N
  writeMask := fun _ _ t j => t.val * BLOCK_SIZE + j.val < N

/-! ### The stream-level spec -/

/-- The guarded stream-level sum of squares: the pass-1 fold `_var += x * x`
over the whole curried `x` stream, guarded by the kernel's window
(`t·BLOCK_SIZE + e < N`) — the skin's contract only pins `xs` on masked
lanes, so the spec must not read unmasked lanes. -/
noncomputable def rmsFusedStreamSumSq (N B : Nat) (xs : Fin 1 → Fin B → ℝ) : ℝ :=
  ∑ u : Fin 1, ∑ e : Fin B, if u.val * B + e.val < N then xs u e ^ 2 else 0

/-- The stream-level RMS-norm spec (the genre's two-pass shape): output
window `(t, j)` holds the step-`t` `x` value times
`rsqrt(Σ x²/N + eps)` — the fold over the *entire* stream — times the
step-`t` `w` value. Algebraically `TiledRMSNorm.rmsAffine` with the
`Fin BLOCK_SIZE` guarded sum re-read as the stream double sum. -/
noncomputable def rmsFusedStreamSpec (N B : Nat) (eps : ℝ)
    (xs ws : Fin 1 → Fin B → ℝ) (t : Fin 1) (j : Fin B) : ℝ :=
  xs t j * (Real.sqrt (rmsFusedStreamSumSq N B xs / (N : ℝ) + eps))⁻¹ * ws t j

/-! ### The stream-lane spec bridge -/

/-- Under the stream pin, the guarded stream double sum **is** the exact
stack's `∑ i, rmsLoad i * rmsLoad i` (the single step collapses the outer
sum, and the `other=0.0` default matches the guard's `else 0`). -/
private theorem rmsFusedStreamSumSq_eq_l2 (X : RegionName) (s₀ : BlockState)
    (stride N B : Nat) (xs : Fin 1 → Fin B → ℝ)
    (hx : ∀ (t : Fin 1) (e : Fin B), t.val * B + e.val < N →
      s₀.readMem X (s₀.pids 0 * stride + (t.val * B + e.val)) = xs t e) :
    rmsFusedStreamSumSq N B xs
      = TiledL2Norm.l2NormSqSum (rmsLoad s₀ X stride N B) := by
  unfold rmsFusedStreamSumSq TiledL2Norm.l2NormSqSum
  rw [Fin.sum_univ_one]
  refine Finset.sum_congr rfl fun e _ => ?_
  have hz : (0 : Fin 1).val * B + e.val = e.val := by simp
  by_cases h : e.val < N
  · have hxe := hx 0 e (by rw [hz]; exact h)
    rw [hz] at hxe
    rw [if_pos (by rw [hz]; exact h), ← hxe]
    simp only [rmsLoad, Tile.maskedRowLoad, if_pos h, xOffset, BlockState.pid]
    ring
  · rw [if_neg (by rw [hz]; exact h)]
    simp only [rmsLoad, Tile.maskedRowLoad, if_neg h]
    ring

/-- Per-lane spec bridge: at a masked window `(t, j)` the stream spec **is**
the exact stack's `rmsnormSpec` at lane `j` (rstd spelling: `1/√v = (√v)⁻¹`). -/
private theorem rmsFusedStreamSpec_eq_rmsnormSpec (X W : RegionName)
    (s₀ : BlockState) (stride N B : Nat) (eps : ℝ)
    (xs ws : Fin 1 → Fin B → ℝ)
    (hx : ∀ (t : Fin 1) (e : Fin B), t.val * B + e.val < N →
      s₀.readMem X (s₀.pids 0 * stride + (t.val * B + e.val)) = xs t e)
    (hw : ∀ (t : Fin 1) (e : Fin B), t.val * B + e.val < N →
      s₀.readMem W (t.val * B + e.val) = ws t e)
    (t : Fin 1) (j : Fin B) (hj : t.val * B + j.val < N) :
    rmsFusedStreamSpec N B eps xs ws t j = rmsnormSpec s₀ X W stride N B eps j := by
  have ht : t = 0 := Fin.ext (by omega)
  subst ht
  have hz : (0 : Fin 1).val * B + j.val = j.val := by simp
  have hjN : j.val < N := by rw [hz] at hj; exact hj
  have hxj : rmsLoad s₀ X stride N B j = xs 0 j := by
    have hxe := hx 0 j hj
    rw [hz] at hxe
    simp only [rmsLoad, Tile.maskedRowLoad, if_pos hjN, xOffset, BlockState.pid]
    exact hxe
  have hwj : rmsWeight s₀ W j = ws 0 j := by
    have hwe := hw 0 j hj
    rw [hz] at hwe
    simp only [rmsWeight]
    exact hwe
  unfold rmsFusedStreamSpec rmsnormSpec TiledRMSNorm.rmsAffine
    TiledRMSNorm.rmsRstd TiledRMSNorm.rmsMeanSq
  rw [hxj, hwj, rmsFusedStreamSumSq_eq_l2 X s₀ stride N B xs hx, one_div]

/-! ### The `TraceSafeR` walk -/

set_option maxHeartbeats 4000000 in
/-- **The `TraceSafeR` walk for the whole two-pass kernel.** Under
`0 < N ≤ BLOCK_SIZE` each `forRange` unrolls to its single `off = 0`
iteration, so the safety obligation reduces to exactly the three window
bound groups: the masked `x` load (both passes), the masked `w` load, and
the masked in-loop store. -/
private theorem rms_norm_fwd_fused_traceSafeR (R : RoundingModel)
    (bounds : RegionBounds) (X Y W : RegionName) (stride N B : Nat) (eps : ℝ)
    (hNpos : 0 < N) (hNle : N ≤ B) (s : BlockState)
    (hbx : ∀ j : Fin B, j.val < N → s.pids 0 * stride + j.val < bounds X)
    (hbw : ∀ j : Fin B, j.val < N → j.val < bounds W)
    (hbo : ∀ j : Fin B, j.val < N → s.pids 0 * stride + j.val < bounds Y) :
    ((rms_norm_fwd_fused X Y W stride N B eps).toAlgKernel).TraceSafeR R bounds s := by
  have hStep : B ≠ 0 := by omega
  have hvar := stepForRangeAuxR_castFree R _ (fusedVarBody_castFree R N B) "off"
  have hwb := stepForRangeAuxR_castFree R _ (fusedWbBody_castFree R W N B) "off"
  simp only [fusedVarBody, fusedWbBody] at hvar hwb
  unfold Kernel.TraceSafeR
  rw [fused_body_decomp]
  simp [fusedPre, fusedVarBody, fusedPost, fusedWbBody,
    Stmt.TraceSafeListR, Stmt.TraceSafeR, Stmt.forRangeTraceSafeR,
    Op.SafeAtR.eq_def, MemAccess.SafeAtR, MaskOpt.SafeAtR, MaskOpt.ActiveR,
    MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR,
    stepStmtR, stepStmtsR, evalOpR.eq_def, hvar, hwb,
    stepForRangeAux.step_lt, stepForRangeAux.step_ge,
    stepStmts, stepStmt, evalOp.eq_def,
    Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop, Tile.reduceSum,
    Tile.reduceSumDrop, Tile.select, TileShape.axisDim, TileShape.eraseAxis,
    TileShape.insertAxisIndex, NumericDType.add, NumericDType.mul,
    NumericDType.div, ComparableDType.lt, hNpos, hNle,
    Nat.not_lt.mpr hNle, hStep]
  exact ⟨hbx, hbw, hbx, hbo⟩

/-! ### The rounded Hoare triple (`hrun`) -/

/-- A masked exact `writeMem` scatter `foldl` leaves every cell outside its
active window untouched (the cell-level frame of the emitting store). -/
private theorem foldl_writeMem_prop_preserve_cell {α : Type}
    {region : RegionName} (ofn : α → Nat) (vfn : α → ℝ)
    (P : α → Prop) [DecidablePred P] (r : RegionName) (oo : Nat)
    (l : List α) (s : BlockState)
    (hnot : ∀ k ∈ l, P k → ¬(r = region ∧ oo = ofn k)) :
    (l.foldl (fun acc k =>
        if P k then acc.writeMem region (ofn k) (vfn k) else acc) s).mem r oo
      = s.mem r oo := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
      rw [List.foldl_cons]
      by_cases hm : P hd
      · simp only [hm, if_true]
        rw [ih _ (fun k hk hmk => hnot k (List.mem_cons_of_mem hd hk) hmk),
          BlockState.writeMem_mem]
        exact if_neg (hnot hd List.mem_cons_self hm)
      · simp only [hm, if_false]
        exact ih _ (fun k hk hmk => hnot k (List.mem_cons_of_mem hd hk) hmk)

/-- Termination of the fused kernel in the single-block regime. -/
private theorem rms_norm_fwd_fused_terminates (X Y W : RegionName)
    (stride N BLOCK_SIZE : Nat) (eps : ℝ) (s : BlockState)
    (hNpos : 0 < N) (hNle : N ≤ BLOCK_SIZE) :
    ∃ s', exec (rms_norm_fwd_fused X Y W stride N BLOCK_SIZE eps) s = some s' := by
  have hB : BLOCK_SIZE ≠ 0 := by omega
  simp [exec, rms_norm_fwd_fused, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
        stepForRangeAux.step_lt, stepForRangeAux.step_ge,
        Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop, Tile.reduceSum,
        Tile.reduceSumDrop, Tile.select, TileShape.axisDim, TileShape.eraseAxis,
        TileShape.insertAxisIndex, NumericDType.add, NumericDType.mul,
        NumericDType.div, ComparableDType.lt, hNpos, hNle,
        Nat.not_lt.mpr hNle, hB]

/-- **Cell-level frame of the fused kernel**: the only memory event is the
single masked in-loop store, so every cell outside the active output window
is untouched. -/
private theorem rms_norm_fwd_fused_frame (X Y W : RegionName)
    (stride N BLOCK_SIZE : Nat) (eps : ℝ) (s s' : BlockState)
    (hNpos : 0 < N) (hNle : N ≤ BLOCK_SIZE)
    (hExec : exec (rms_norm_fwd_fused X Y W stride N BLOCK_SIZE eps) s = some s')
    (r : RegionName) (oo : Nat)
    (hcond : r ≠ Y ∨ ∀ i : Fin BLOCK_SIZE, i.val < N → oo ≠ yOffset s stride i) :
    s'.mem r oo = s.mem r oo := by
  have hStep : BLOCK_SIZE ≠ 0 := by omega
  simp [exec, rms_norm_fwd_fused, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
        stepForRangeAux.step_lt, stepForRangeAux.step_ge,
        Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop, Tile.reduceSum,
        Tile.reduceSumDrop, Tile.select, TileShape.axisDim, TileShape.eraseAxis,
        TileShape.insertAxisIndex, NumericDType.add, NumericDType.mul,
        NumericDType.div, ComparableDType.lt, hNpos, hNle,
        Nat.not_lt.mpr hNle, hStep] at hExec
  subst s'
  refine Eq.trans (foldl_writeMem_prop_preserve_cell _ _ _ r oo _ _ ?_) rfl
  intro k _ hk hbad
  rcases hcond with hne | hno
  · exact hne hbad.1
  · exact hno ⟨k.1.val, k.1.isLt⟩ hk (by simpa [yOffset] using hbad.2)

/-! ### The headline -/

set_option maxHeartbeats 4000000 in
/-- **The `⊨[R]` streaming headline (wave-5 S3 per-step emit genre).** For
every rounding model `R`, the faithful `rms_norm_fwd_fused` surface
implements, on its `StreamEmitMasked2DKernelIO₂` signature, the **ideal ℝ
two-pass RMS-norm** over the streamed tiles: emitted window `(t, j)` holds
`x[t,j] · (√(Σ_guarded x²/N + eps))⁻¹ · w[t,j]`, where the guarded double sum
folds the *entire* `x` stream — the spec `f` is exact real arithmetic. The
full Hoare triple is the skin's: for every disjoint flat allocation of
`[X, W, Y]`, every pid pair, and every launch state with both masked input
streams pinned, `execR R` terminates, every write-active emitted cell reads
back through `readMemAs .real` as the ideal value, and every flat cell
outside the active output window is untouched.

Rounding story: the kernel has **zero rounding events**. Both masked loads,
both passes' arithmetic and the in-loop store are at `.real`, and the two
`.to(tl.float32)` casts are erased outright by the compute-to-algorithm
lowering (`fused_body_decomp` is `rfl` onto a body with no `Op.castFloat`).
The skin's boundary quantization therefore degenerates: the readback
contract's `R.round .real` is the identity by the model's defining
`round_real`, and the `.real` in-loop stores are exact under `execR R` — the
∀-`R` face holds via the `RoundingModel` `.real` identity fields, not as a
`.triv` special case.

Layer map: `fused_castFree` collapses `execR R` onto the exact `exec`
statement by statement (the two `forRange` clauses lift their cast-free
bodies through `stepForRangeAuxR_castFree`), so the proven exact readback
`rms_norm_fwd_fused_correct` is reused unchanged; the `⊨[R]` face adds the
`TraceSafeR` walk (`rms_norm_fwd_fused_traceSafeR`), termination
(`rms_norm_fwd_fused_terminates`), the per-cell memory frame
(`rms_norm_fwd_fused_frame`), and the stream-lane spec bridge
(`rmsFusedStreamSpec_eq_rmsnormSpec`).

Both hypotheses are the *exact* headline `rms_norm_fwd_fused_output_summary`'s
own launch precondition, with the same provenance:

* `hNpos : 0 < N`, `hNle : N ≤ BLOCK_SIZE` — the Python wrapper picks
  `BLOCK_SIZE ≥ N` and raises otherwise, so every real launch is in this
  single-block regime; there both `range(0, N, BLOCK_SIZE)` passes execute
  exactly the `off = 0` iteration, which is what makes the stream one step
  wide (`T := 1`) and the closed form available.

The exact headline's third side condition `hOutInj` is **not** carried here:
the write window `pid₀·stride + i` is injective in `i` outright, and this
face discharges it.

Relation to the exact surface: the exact headline
`rms_norm_fwd_fused_output_summary` (`Realizes_without_Rounding`) above is
retained unchanged; this `⊨[R]` face restates the same RMS-norm content on
the streaming emit skin, for every `R` at once (at the `.real` grid the two
faces carry the same exact cell). Both faces are kept per the
rounding-as-default doctrine. -/
specification rms_norm_fwd_fused_io_correctness (R : RoundingModel)
    (X Y W : RegionName) (stride N BLOCK_SIZE : Nat) (eps : ℝ)
    (hNpos : 0 < N) (hNle : N ≤ BLOCK_SIZE) :
    rmsnormFusedKernelIO X Y W stride N BLOCK_SIZE eps ⊨[R]
      fun _ _ xs ws t j => rmsFusedStreamSpec N BLOCK_SIZE eps xs ws t j := by
  refine StreamEmitMasked2DKernelIO₂.ImplementsR.intro _ ?_ ?_ ?_
  · exact rms_norm_fwd_fused_flattenOk X Y W stride N BLOCK_SIZE eps
  · -- safety walk
    intro bounds s xs ws _hx _hw hbr1 hbr2 hbw
    simp only [rmsnormFusedKernelIO] at hbr1 hbr2 hbw ⊢
    refine rms_norm_fwd_fused_traceSafeR R bounds X Y W stride N BLOCK_SIZE eps
      hNpos hNle s (fun j hj => ?_) (fun j hj => ?_) (fun j hj => ?_)
    · have h := hbr1 0 j (by simpa using hj)
      simpa using h
    · have h := hbr2 0 j (by simpa using hj)
      simpa using h
    · have h := hbw 0 j (by simpa using hj)
      simpa using h
  · -- the rounded Hoare triple
    intro s₀ xs ws _hundef hx hw
    simp only [rmsnormFusedKernelIO] at hx hw ⊢
    obtain ⟨sfin, hexec⟩ :=
      rms_norm_fwd_fused_terminates X Y W stride N BLOCK_SIZE eps s₀ hNpos hNle
    have hInj : Function.Injective
        (fun i : Fin BLOCK_SIZE => yOffset s₀ stride i) := by
      intro a b hab
      simp only [yOffset] at hab
      exact Fin.ext (Nat.add_left_cancel hab)
    refine ⟨sfin, ?_, ?_, ?_⟩
    · rw [fused_execR_eq]
      exact hexec
    · intro t j hj
      have ht : t = 0 := Fin.ext (by omega)
      subst ht
      have hz : (0 : Fin 1).val * BLOCK_SIZE + j.val = j.val := by simp
      have hjN : j.val < N := by rw [hz] at hj; exact hj
      have hval := rms_norm_fwd_fused_correct X Y W stride N BLOCK_SIZE eps
        s₀ sfin hNpos hNle hInj hexec j
      rw [if_pos hjN] at hval
      rw [hz, BlockState.readMemAs_real]
      have haddr : s₀.pids 0 * stride + j.val = yOffset s₀ stride j := rfl
      rw [haddr, hval,
        rmsFusedStreamSpec_eq_rmsnormSpec X W s₀ stride N BLOCK_SIZE eps xs ws
          hx hw 0 j hj]
      simp [FloatDType.ofReal]
    · intro r oo hcond
      refine rms_norm_fwd_fused_frame X Y W stride N BLOCK_SIZE eps s₀ sfin
        hNpos hNle hexec r oo ?_
      rcases hcond with hne | hno
      · exact Or.inl hne
      · refine Or.inr fun i hi => ?_
        have h := hno 0 i (by simpa using hi)
        simpa [yOffset] using h

end IOFace

end VeriTile.Bench.TritonBenchG.RmsnormFused
