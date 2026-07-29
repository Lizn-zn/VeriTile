import VeriTile.Triton

/-!
# `rmsnorm_fused_llama` — strict per-kernel correctness

`_rms_norm_fwd_fused` is the Llama-style fused RMSNorm forward: each program
`row` normalizes one row of `X` by its root-mean-square, scales by per-column
weights `W`, casts the result to float16, and writes
`Y[row] = fp16((x / sqrt(mean(x²) + eps)) * w)`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body, for one program (one row). The host launch
(`_rms_norm_fwd_fused[(M,)](...)`, the grid over rows `M`, the host-fixed
`BLOCK_SIZE = 16384`, scheduling, and how the runtime composes per-row writes
into one buffer) is the *trusted boundary*, not a proof obligation here. Because
`row = tl.program_id(0)` is universally quantified (via `s.pid`), the per-program
statement covers every row of the grid.

## Proof architecture

```
rms_norm_fwd_fused_llama_output_summary       ← TOP THEOREM
  ├─ (toAlgorithm? = Except.ok _)             surface lowers to the algorithm layer
  └─ rms_norm_fwd_fused_llama_compute_correct ← ComputeCorrect over the masked fp16 store
       └─ rms_norm_fwd_fused_llama_correct    ← algorithm-layer readback per lane
            └─ scatter_memcell_fp16_prop_masked_nd  (fp16 masked-scatter readback)
                 └─ foldl_writeMemTyped_fp16_preserve_masked
```

The RMSNorm row math (`rmsInputTile`, `rmsVarCarrier`, `rmsInvCarrier`,
`rmsnormCarrierSpec` / `rmsnormSpec`) is defined inline in this file rather than
reusing `VeriTile.Triton.Math.RMSNorm`.

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); the `.to(tl.float32)`
input/weight casts reduce to identity at the algorithm layer. The **store cast
to float16 is modeled explicitly**: the spec writes the `MemCell.of .fp16`
obtained by `FloatDType.real.cast FloatDType.fp16`, so the fp16 rounding of the
final value is part of the proved statement (not erased). The reduction
`tl.sum(_var) / N` sums over the *padded* `BLOCK_SIZE` block, but out-of-range
lanes are masked to `0` (load `other=0.0`), so the sum equals the logical row
length `N`. The reciprocal std is `rstd = 1 / sqrt(meanSq + eps)`; the affine
step is `x_hat * w`. Correctness is stated under the `0 < N ≤ BLOCK_SIZE`
single-block precondition (Python fixes `BLOCK_SIZE = 16384` after checking
`N ≤ BLOCK_SIZE`), where both `range(0, N, BLOCK_SIZE)` loops execute exactly the
`off = 0` iteration. `@triton.autotune` is not modeled.
-/

namespace VeriTile.Bench.TritonBenchG.RmsnormFusedLlama

open VeriTile.Triton

set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

/-- Faithful `forRange` transcription of `rmsnorm_fused_llama.py`'s
`_rms_norm_fwd_fused`.

The Python wrapper fixes `BLOCK_SIZE = 16384` after checking `N <= BLOCK_SIZE`,
so the correctness theorem below proves the loop-shaped kernel under that
runtime precondition. Under the precondition both `range(0, N, BLOCK_SIZE)`
loops execute exactly the `off = 0` iteration.

Allowed mechanical Lean-syntax-only changes:
- Python `N` / `BLOCK_SIZE: tl.constexpr` -> Lean `Nat` parameters. -/
def rms_norm_fwd_fused_llama
    (X Y W : RegionName) (stride N BLOCK_SIZE : Nat) (eps : ℝ) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  Y += row * $(stride)
  X += row * $(stride)
  _var = tl.zeros([$(BLOCK_SIZE)], dtype=tl.float32)
  for off in range(0, $(N), $(BLOCK_SIZE)) {
    cols = off + tl.arange(0, $(BLOCK_SIZE))
    x = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
    _var += x * x
  }
  var = tl.sum(_var, axis=0) / $(N)
  rstd = 1 / tl.sqrt(var + $(eps))
  for off in range(0, $(N), $(BLOCK_SIZE)) {
    cols = off + tl.arange(0, $(BLOCK_SIZE))
    mask = cols < $(N)
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
    x_hat = x * rstd
    y = x_hat * w
    tl.store(Y + cols, (y).to(tl.float16), mask=mask)
  }
}

def xOffset (s : BlockState) (stride : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pid * stride + i.val

def yOffset (s : BlockState) (stride : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pid * stride + i.val

noncomputable def rmsInputTile
    (s : BlockState) (X : RegionName) (stride N BLOCK_SIZE : Nat) :
    Tile .real [BLOCK_SIZE] :=
  { data := fun idx =>
      if idx.1.val < N then
        some (s.readMem X (xOffset s stride idx.1))
      else some (0.0 : ℝ) }

noncomputable def rmsVarCarrier
    (s : BlockState) (X : RegionName) (stride N BLOCK_SIZE : Nat) :
    WithBot ℝ :=
  Option.map₂ (fun a n => a / n)
    ((Tile.reduceSum (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false
      (Tile.bop (NumericDType.mul .real) (Broadcast.consSame Broadcast.nil)
        (rmsInputTile s X stride N BLOCK_SIZE)
        (rmsInputTile s X stride N BLOCK_SIZE))).data PUnit.unit)
    ((Tile.scalar (dtype := .real) (some (N : ℝ) : WithBot ℝ)).data PUnit.unit)

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
  rmsnormCarrierSpec s X W stride N BLOCK_SIZE eps i

/-- Algorithm-layer correctness for the Llama RMSNorm kernel under the Python
wrapper's `N <= BLOCK_SIZE` launch precondition. -/
theorem rms_norm_fwd_fused_llama_correct
    (X Y W : RegionName) (stride N BLOCK_SIZE : Nat) (eps : ℝ)
    (s s' : BlockState)
    (hNpos : 0 < N) (hNle : N ≤ BLOCK_SIZE)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => yOffset s stride i))
    (hExec : exec (rms_norm_fwd_fused_llama X Y W stride N BLOCK_SIZE eps)
        s = some s') :
    ∀ i : Fin BLOCK_SIZE,
      s'.mem Y (yOffset s stride i) =
        if i.val < N then
          MemCell.of .fp16
            (FloatDType.real.cast FloatDType.fp16
              (some (rmsnormSpec s X W stride N BLOCK_SIZE eps i)))
        else s.mem Y (yOffset s stride i) := by
  intro i
  have h_inj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE] => s.pids 0 * stride + idx.1.val) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [yOffset] using h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  by_cases hB : 0 < BLOCK_SIZE
  · have hStep : BLOCK_SIZE ≠ 0 := Nat.ne_of_gt hB
    simp [exec, rms_norm_fwd_fused_llama, stepStmts, stepStmt, evalOp, evalOp.eq_def,
          ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
          stepForRangeAux.step_lt, stepForRangeAux.step_ge,
          Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop, Tile.reduceSum,
          Tile.reduceSumDrop, Tile.select, TileShape.axisDim, TileShape.eraseAxis,
          TileShape.insertAxisIndex, NumericDType.add, NumericDType.mul,
          NumericDType.div, ComparableDType.lt, hNpos, hNle,
          Nat.not_lt.mpr hNle, hStep] at hExec
    subst s'
    simp only [yOffset]
    rw [scatter_memcell_fp16_prop_masked_nd _ _ _ _ h_inj (i, PUnit.unit)]
    by_cases hi : i.val < N
    · simp only [hi, ↓reduceIte]
      simp [hi, rmsnormSpec, rmsnormCarrierSpec, rmsInvCarrier, rmsVarCarrier,
            rmsInputTile, xOffset, Tile.reduceSum, Tile.reduceSumDrop, Tile.select,
            TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
            WithBot.realSqrt, NumericDType.mul, FloatDType.cast]
      rfl
    · simp [hi]
  · exact False.elim (hB (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Compute-facing correctness for the Llama RMSNorm kernel under the Python
wrapper's `N <= BLOCK_SIZE` launch precondition. -/
theorem rms_norm_fwd_fused_llama_compute_correct
    (X Y W : RegionName) (stride N BLOCK_SIZE : Nat) (eps : ℝ)
    (s : BlockState)
    (hNpos : 0 < N) (hNle : N ≤ BLOCK_SIZE)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => yOffset s stride i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := rms_norm_fwd_fused_llama X Y W stride N BLOCK_SIZE eps)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => i.val < N)
        (fun i => (Y, yOffset s stride i)))
      (expected := fun i =>
        MemCell.of .fp16
          (FloatDType.real.cast FloatDType.fp16
            (some (rmsnormSpec s X W stride N BLOCK_SIZE eps i)))) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [rms_norm_fwd_fused_llama, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := rms_norm_fwd_fused_llama_correct X Y W stride N BLOCK_SIZE eps
    s s' hNpos hNle hOutInj hExec i
  simpa [hActive] using h

/-- Per-kernel output summary for `_rms_norm_fwd_fused` (Llama): the DSL surface
lowers to the algorithm layer, and the masked fp16 store to `Y` is
compute-correct — every active lane (`i.val < N`) holds the fp16-cast RMSNorm
spec, out-of-bounds lanes are preserved. Stated under the `0 < N ≤ BLOCK_SIZE`
single-block launch precondition chosen by the Python wrapper. -/
specification rms_norm_fwd_fused_llama_output_summary
    (X Y W : RegionName) (stride N BLOCK_SIZE : Nat) (eps : ℝ)
    (s : BlockState)
    (hNpos : 0 < N) (hNle : N ≤ BLOCK_SIZE)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => yOffset s stride i)) :
    (∃ alg, (rms_norm_fwd_fused_llama X Y W stride N BLOCK_SIZE eps).toAlgorithm? =
        Except.ok alg) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel := rms_norm_fwd_fused_llama X Y W stride N BLOCK_SIZE eps)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => i.val < N)
        (fun i => (Y, yOffset s stride i)))
      (expected := fun i =>
        MemCell.of .fp16
          (FloatDType.real.cast FloatDType.fp16
            (some (rmsnormSpec s X W stride N BLOCK_SIZE eps i)))) := by
  refine ⟨?_, ?_⟩
  · simp only [rms_norm_fwd_fused_llama, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
    exact ⟨_, rfl⟩
  · exact rms_norm_fwd_fused_llama_compute_correct X Y W stride N BLOCK_SIZE eps
      s hNpos hNle hOutInj

/-! ## The `⊨[R]` streaming headline (wave-5 S3 per-step emit genre)

Everything below is purely additive; the exact surface above is untouched.
This is a consumer of the per-step emit skin `StreamEmitMasked2DKernelIO₂`
(streaming genre, style S3): the store sits **inside** the second
`for off in range(0, N, BLOCK_SIZE)` pass, so the output is a per-step
`BLOCK_SIZE`-lane window family rather than one terminal tile, and the
kernel's spec `f t j` is the genre's *two-pass* shape — the step-`t` tile
combined with a fold over the entire stream.

The launch regime is the exact headline's: the Python wrapper fixes
`BLOCK_SIZE = 16384` after checking `N ≤ BLOCK_SIZE`, so under
`0 < N ≤ BLOCK_SIZE` both `forRange` clauses execute exactly the `off = 0`
iteration and the stream has a single step (`T := 1`).

Structure of the `execR R` story: unlike its `.real`-storing sibling this
kernel **does** round, exactly once per emitted cell. The `.to(tl.float32)`
casts are erased outright by the compute-to-algorithm lowering, but the
store's `(y).to(tl.float16)` survives as a genuine
`Op.castFloat FloatDType.real FloatDType.fp16` under a
`Stmt.store TileDType.fp16` (see `llamaWbBody`), so under `execR R` the
lane value crosses **two** rounding sites at the same grid: `evalOpR`'s
`R.cast .real .fp16`, then the typed store's `R.storeValue .fp16`. The
model's defining `round_idem` collapses that composition to a single
`R.round .fp16` (`RoundingModel.storeValue_cast`, packaged here as
`llama_storeValue_cast_fp16`), which is precisely the skin's contract at
`outDType := .fp16`: the emitted cell reads back through `readMemAs .fp16`
as `FloatDType.fp16.ofReal (R.round .fp16 (f … t j))`. **No hypothesis on
`R` is used** — the face is genuinely ∀-`R`. -/

section IOFace

open scoped VeriTile.Triton.StreamEmitMasked2DKernelIO₂

set_option linter.unusedVariables false

/-! ### The lowered body, decomposed

The four segments of the lowered kernel body. `llama_body_decomp` is `rfl`:
this *is* the algorithm the DSL surface lowers to. The two
`.to(tl.float32)` casts leave no trace (`ComputeDType.fp32.eraseDType` is
`TileDType.real`); the store's `.to(tl.float16)` does — it is the kernel's
single rounding event. -/

/-- The prologue: `row`, the two row-base pointer bumps, and the `_var`
accumulator seed. -/
private def llamaPre (X Y : RegionName) (stride B : Nat) : List Stmt :=
  [ Stmt.assign .nat [] "row" (Op.programId 0),
    Stmt.assign .ptr [] "Y"
      (Op.ptrAdd Broadcast.nil (Op.ptrBase Y)
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "row") (Op.constNat stride))),
    Stmt.assign .ptr [] "X"
      (Op.ptrAdd Broadcast.nil (Op.ptrBase X)
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "row") (Op.constNat stride))),
    Stmt.assign .real [B] "_var" (Op.full [B] (Op.const 0)) ]

/-- Pass 1's body: masked `x` load and the `_var += x * x` fold. -/
private def llamaVarBody (N B : Nat) : List Stmt :=
  [ Stmt.assign .nat [B] "cols"
      (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "off") (Op.arange B)),
    Stmt.assign .real [B] "x"
      (Op.load .real
        (MemAccess.ptr
          (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "X") (Op.ref .nat [B] "cols")))
        (MaskOpt.maskOther
          (Op.lt .nat Broadcast.scalarR (Op.ref .nat [B] "cols") (Op.constNat N))
          ((Op.const 0.0).broadcast [B]))),
    Stmt.assign .real [B] "_var"
      (Op.add .real (Broadcast.consSame Broadcast.nil) (Op.ref .real [B] "_var")
        (Op.mul .real (Broadcast.consSame Broadcast.nil) (Op.ref .real [B] "x")
          (Op.ref .real [B] "x"))) ]

/-- The reduce / reciprocal-sqrt tail between the two passes. -/
private def llamaPost (N B : Nat) (eps : ℝ) : List Stmt :=
  [ Stmt.assign .real [] "var"
      (Op.div .real Broadcast.nil
        (Op.reduceSum (shape := [B]) ⟨0, by simp⟩ Bool.false (Op.ref .real [B] "_var"))
        (Op.const (N : ℝ))),
    Stmt.assign .real [] "rstd"
      (Op.div .real Broadcast.nil (Op.const 1)
        (Op.sqrt (Op.add .real Broadcast.nil (Op.ref .real [] "var") (Op.const eps)))) ]

/-- Pass 2's body — the **emit** body: two masked loads, the affine scaling,
and the in-loop masked **fp16** store into `Y` (the `Op.castFloat` under a
`.fp16`-typed store is the kernel's only rounding event). -/
private def llamaWbBody (W : RegionName) (N B : Nat) : List Stmt :=
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
    Stmt.store .fp16 [B]
      (MemAccess.ptr
        (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "Y") (Op.ref .nat [B] "cols")))
      (Op.castFloat FloatDType.real FloatDType.fp16 (Op.ref .real [B] "y"))
      (MaskOpt.mask (Op.ref .bool [B] "mask")) ]

/-- The lowered body is `prologue ++ pass 1 ++ tail ++ pass 2`, definitionally. -/
private theorem llama_body_decomp (X Y W : RegionName) (stride N B : Nat) (eps : ℝ) :
    (rms_norm_fwd_fused_llama X Y W stride N B eps).toAlgKernel.body
      = llamaPre X Y stride B
        ++ [Stmt.forRange "off" 0 N B (llamaVarBody N B)]
        ++ llamaPost N B eps
        ++ [Stmt.forRange "off" 0 N B (llamaWbBody W N B)] := rfl

/-! ### `execR` loop unrolling and the single-round collapse -/

/-- `stepForRangeAuxR` in-range unrolling — the `R` mirror of
`stepForRangeAux.step_lt` (this kernel's emitting pass is *not* cast-free,
so the loop is unrolled under `execR R` directly rather than transported to
the exact stepper). -/
private theorem stepForRangeAuxR_step_lt (R : RoundingModel) {idx : RegName}
    {cur stop step : Nat} {body : List Stmt} {s : BlockState}
    (hstep : step ≠ 0) (h : cur < stop) :
    stepForRangeAuxR R idx cur stop step body s
      = (stepStmtsR R body (s.setReg idx .nat [] (Tile.scalar cur))).bind
          (stepForRangeAuxR R idx (cur + step) stop step body) := by
  conv_lhs => unfold stepForRangeAuxR
  simp [hstep, h]
  cases hbody : stepStmtsR R body (s.setReg idx .nat [] (Tile.scalar cur)) <;> rfl

/-- `stepForRangeAuxR` exhaustion — the `R` mirror of
`stepForRangeAux.step_ge`. -/
private theorem stepForRangeAuxR_step_ge (R : RoundingModel) {idx : RegName}
    {cur stop step : Nat} {body : List Stmt} {s : BlockState}
    (hstep : step ≠ 0) (h : stop ≤ cur) :
    stepForRangeAuxR R idx cur stop step body s = some s := by
  unfold stepForRangeAuxR
  simp [hstep, Nat.not_lt.mpr h]

/-- **The two-events-one-round collapse.** `evalOpR`'s
`R.cast .real .fp16` lands the lane value on the fp16 grid and the
`.fp16`-typed store re-quantizes at the same grid; by the model's defining
`round_idem` the composition is a *single* `R.round .fp16` of the ideal
value (`RoundingModel.storeValue_cast`), extended here to the undefined
carrier, where both sides are `R.round .fp16 0` by the `⊥ ↦ 0` finite
fallback. -/
private theorem llama_storeValue_cast_fp16 (R : RoundingModel) (v : WithBot ℝ) :
    R.storeValue .fp16 (R.cast .real .fp16 v)
      = R.round .fp16 (WithBot.unbotD 0 v) := by
  cases v with
  | bot =>
      show R.storeValue .fp16 (R.cast .real .fp16 (⊥ : WithBot ℝ))
        = R.round .fp16 0
      simp [RoundingModel.storeValue, RoundingModel.cast, FloatDType.storeValue]
  | coe x => exact R.storeValue_cast .fp16 x

/-- The two-pass surface sits inside the flat-memory bridge's covered
fragment. -/
theorem rms_norm_fwd_fused_llama_flattenOk (X Y W : RegionName)
    (stride N B : Nat) (eps : ℝ) :
    ((rms_norm_fwd_fused_llama X Y W stride N B eps).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  rw [llama_body_decomp]
  simp [llamaPre, llamaVarBody, llamaPost, llamaWbBody, StmtList.FlattenOk,
    Stmt.FlattenOk, Op.FlattenOk.eq_def]

/-! ### IO signature -/

/-- **Streaming IO signature** of `_rms_norm_fwd_fused` (Llama) on the
two-stream per-step emit skin (S3: in-loop store), in the Python wrapper's
single-block launch regime `0 < N ≤ BLOCK_SIZE`: both
`range(0, N, BLOCK_SIZE)` passes run exactly the `off = 0` iteration, so the
stream has `T := 1` step of `BLOCK_SIZE` lanes. Step `t` reads the
`BLOCK_SIZE`-lane `x` tile (`read1`, both passes read the same addresses)
and the `w` tile (`read2`, pass 2 only); step `t` of pass 2 stores the
`BLOCK_SIZE`-lane output window (`write`) at the **`.fp16`** grid — the
kernel's store is `tl.store(Y + cols, (y).to(tl.float16), mask=mask)`, which
lowers to `Stmt.store TileDType.fp16` with an
`Op.castFloat FloatDType.real FloatDType.fp16` value (see `llamaWbBody`), so
every emitted cell carries exactly one quantization event. The windows
transcribe the kernel's pointer arithmetic verbatim
(`cols = off + tl.arange(0, BLOCK_SIZE)`, i.e. `t·BLOCK_SIZE + j`, against
the row-bumped `X`/`Y` pointers):

* `read1` step `t`, lane `j`: `pid₀·stride + (t·BLOCK_SIZE + j)` — the
  kernel's `X + cols` after `X += row * stride`.
* `read2` step `t`, lane `j`: `t·BLOCK_SIZE + j` — the kernel's `W + cols`
  (`W` is not row-bumped).
* `write` step `t`, lane `j`: `pid₀·stride + (t·BLOCK_SIZE + j)` — the
  kernel's `Y + cols` after `Y += row * stride`.

All three masks are the kernel's single `cols < N`. The second pid is inert
(the launch grid is 1-D). -/
def rmsnormFusedLlamaKernelIO (X Y W : RegionName) (stride N BLOCK_SIZE : Nat)
    (eps : ℝ) : StreamEmitMasked2DKernelIO₂ where
  kernel := rms_norm_fwd_fused_llama X Y W stride N BLOCK_SIZE eps
  inp1 := X
  inp2 := W
  out := Y
  T := 1
  B1 := BLOCK_SIZE
  B2 := BLOCK_SIZE
  C := BLOCK_SIZE
  outDType := .fp16
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
noncomputable def rmsLlamaStreamSumSq (N B : Nat) (xs : Fin 1 → Fin B → ℝ) : ℝ :=
  ∑ u : Fin 1, ∑ e : Fin B, if u.val * B + e.val < N then xs u e ^ 2 else 0

/-- The stream-level RMS-norm spec (the genre's two-pass shape): output
window `(t, j)` holds the step-`t` `x` value times `rsqrt(Σ x²/N + eps)` —
the fold over the *entire* stream — times the step-`t` `w` value. This is
the **ideal ℝ** value; the fp16 quantization lives in the skin's readback
contract, not in `f`. -/
noncomputable def rmsLlamaStreamSpec (N B : Nat) (eps : ℝ)
    (xs ws : Fin 1 → Fin B → ℝ) (t : Fin 1) (j : Fin B) : ℝ :=
  xs t j * (Real.sqrt (rmsLlamaStreamSumSq N B xs / (N : ℝ) + eps))⁻¹ * ws t j

/-! ### The carrier-to-ℝ bridge for the exact spec

The exact stack states `rmsnormSpec` at the `WithBot ℝ` carrier layer
(`rmsVarCarrier` / `rmsInvCarrier`). These two lemmas evaluate that carrier
into ordinary real arithmetic, which is what the stream spec speaks. -/

/-- The kernel's padded block reduction of masked squares is defined and
equals the guarded real sum (`other=0.0` lanes contribute `0`). -/
private theorem llama_masked_sq_sum (B N : Nat) (load : Fin B → ℝ) :
    @Finset.sum (Fin B) (WithBot ℝ) _ Finset.univ
        (fun k => Option.map₂ (fun x1 x2 : ℝ => x1 * x2)
          (if k.val < N then (some (load k) : WithBot ℝ) else (some (0.0 : ℝ) : WithBot ℝ))
          (if k.val < N then (some (load k) : WithBot ℝ) else (some (0.0 : ℝ) : WithBot ℝ)))
      = some (∑ k : Fin B, if k.val < N then load k ^ 2 else 0) := by
  have hcongr : ∀ k : Fin B,
      Option.map₂ (fun x1 x2 : ℝ => x1 * x2)
        (if k.val < N then (some (load k) : WithBot ℝ) else (some (0.0 : ℝ) : WithBot ℝ))
        (if k.val < N then (some (load k) : WithBot ℝ) else (some (0.0 : ℝ) : WithBot ℝ))
      = (((if k.val < N then load k ^ 2 else 0 : ℝ)) : WithBot ℝ) := by
    intro k
    by_cases h : k.val < N
    · simp only [if_pos h]
      rw [show Option.map₂ (fun x1 x2 : ℝ => x1 * x2) (some (load k)) (some (load k))
            = some (load k * load k) from rfl, ← sq]
      rfl
    · simp only [if_neg h]
      have h0 : (0.0 : ℝ) * 0.0 = 0 := by norm_num
      rw [show Option.map₂ (fun x1 x2 : ℝ => x1 * x2) (some (0.0 : ℝ)) (some (0.0 : ℝ))
            = some ((0.0 : ℝ) * 0.0) from rfl, h0]
      rfl
  calc @Finset.sum (Fin B) (WithBot ℝ) _ Finset.univ _
      = ∑ k : Fin B, (((if k.val < N then load k ^ 2 else 0 : ℝ)) : WithBot ℝ) :=
        Finset.sum_congr rfl (fun k _ => hcongr k)
    _ = _ := (WithBot.coe_sum Finset.univ _).symm

/-- `rmsVarCarrier` is the defined real mean square over the guarded row. -/
private theorem llama_varCarrier_eq (s : BlockState) (X : RegionName)
    (stride N B : Nat) :
    rmsVarCarrier s X stride N B
      = some ((∑ k : Fin B,
          if k.val < N then (s.readMem X (xOffset s stride k)) ^ 2 else 0) / (N : ℝ)) := by
  unfold rmsVarCarrier
  simp [rmsInputTile, Tile.reduceSum, Tile.reduceSumDrop, TileShape.axisDim,
    TileShape.eraseAxis, TileShape.insertAxisIndex, Tile.bop, NumericDType.mul,
    WithBot.realMul]
  refine (congrArg (fun a : WithBot ℝ => Option.map (fun x : ℝ => x / (N : ℝ)) a)
    (llama_masked_sq_sum B N (fun k => s.readMem X (xOffset s stride k)))).trans ?_
  rfl

/-- The exact spec in ordinary real arithmetic. -/
private theorem llama_spec_eq (s : BlockState) (X W : RegionName)
    (stride N B : Nat) (eps : ℝ) (i : Fin B) :
    rmsnormSpec s X W stride N B eps i
      = s.readMem X (xOffset s stride i)
        * (Real.sqrt ((∑ k : Fin B,
            if k.val < N then (s.readMem X (xOffset s stride k)) ^ 2 else 0) / (N : ℝ)
              + eps))⁻¹
        * s.readMem W i.val := by
  unfold rmsnormSpec rmsnormCarrierSpec rmsInvCarrier
  rw [llama_varCarrier_eq]
  rfl

/-! ### The stream-lane spec bridge -/

/-- Under the stream pin, the guarded stream double sum **is** the exact
stack's guarded row sum of squares (the single step collapses the outer
sum). -/
private theorem rmsLlamaStreamSumSq_eq (X : RegionName) (s₀ : BlockState)
    (stride N B : Nat) (xs : Fin 1 → Fin B → ℝ)
    (hx : ∀ (t : Fin 1) (e : Fin B), t.val * B + e.val < N →
      s₀.readMem X (s₀.pids 0 * stride + (t.val * B + e.val)) = xs t e) :
    rmsLlamaStreamSumSq N B xs
      = ∑ k : Fin B, if k.val < N then (s₀.readMem X (xOffset s₀ stride k)) ^ 2 else 0 := by
  unfold rmsLlamaStreamSumSq
  rw [Fin.sum_univ_one]
  refine Finset.sum_congr rfl fun e _ => ?_
  have hz : (0 : Fin 1).val * B + e.val = e.val := by simp
  by_cases h : e.val < N
  · have hxe := hx 0 e (by rw [hz]; exact h)
    rw [hz] at hxe
    rw [if_pos (by rw [hz]; exact h), if_pos h, ← hxe]
    rfl
  · rw [if_neg (by rw [hz]; exact h), if_neg h]

/-- Per-lane spec bridge: at a masked window `(t, j)` the stream spec **is**
the exact stack's `rmsnormSpec` at lane `j`. -/
private theorem rmsLlamaStreamSpec_eq_rmsnormSpec (X W : RegionName)
    (s₀ : BlockState) (stride N B : Nat) (eps : ℝ)
    (xs ws : Fin 1 → Fin B → ℝ)
    (hx : ∀ (t : Fin 1) (e : Fin B), t.val * B + e.val < N →
      s₀.readMem X (s₀.pids 0 * stride + (t.val * B + e.val)) = xs t e)
    (hw : ∀ (t : Fin 1) (e : Fin B), t.val * B + e.val < N →
      s₀.readMem W (t.val * B + e.val) = ws t e)
    (t : Fin 1) (j : Fin B) (hj : t.val * B + j.val < N) :
    rmsLlamaStreamSpec N B eps xs ws t j
      = rmsnormSpec s₀ X W stride N B eps j := by
  have ht : t = 0 := Fin.ext (by omega)
  subst ht
  have hz : (0 : Fin 1).val * B + j.val = j.val := by simp
  have hjN : j.val < N := by rw [hz] at hj; exact hj
  have hxj : s₀.readMem X (xOffset s₀ stride j) = xs 0 j := by
    have hxe := hx 0 j hj
    rw [hz] at hxe
    exact hxe
  have hwj : s₀.readMem W j.val = ws 0 j := by
    have hwe := hw 0 j hj
    rw [hz] at hwe
    exact hwe
  unfold rmsLlamaStreamSpec
  rw [llama_spec_eq, hxj, hwj,
    rmsLlamaStreamSumSq_eq X s₀ stride N B xs hx]

/-! ### The `TraceSafeR` walk -/

set_option maxHeartbeats 4000000 in
/-- **The `TraceSafeR` walk for the whole two-pass kernel.** Under
`0 < N ≤ BLOCK_SIZE` each `forRange` unrolls to its single `off = 0`
iteration, so the safety obligation reduces to exactly the three window
bound groups: the masked `x` load (both passes), the masked `w` load, and
the masked in-loop fp16 store. -/
private theorem rms_norm_fwd_fused_llama_traceSafeR (R : RoundingModel)
    (bounds : RegionBounds) (X Y W : RegionName) (stride N B : Nat) (eps : ℝ)
    (hNpos : 0 < N) (hNle : N ≤ B) (s : BlockState)
    (hbx : ∀ j : Fin B, j.val < N → s.pids 0 * stride + j.val < bounds X)
    (hbw : ∀ j : Fin B, j.val < N → j.val < bounds W)
    (hbo : ∀ j : Fin B, j.val < N → s.pids 0 * stride + j.val < bounds Y) :
    ((rms_norm_fwd_fused_llama X Y W stride N B eps).toAlgKernel).TraceSafeR
      R bounds s := by
  have hStep : B ≠ 0 := by omega
  unfold Kernel.TraceSafeR
  rw [llama_body_decomp]
  simp [llamaPre, llamaVarBody, llamaPost, llamaWbBody,
    Stmt.TraceSafeListR, Stmt.TraceSafeR, Stmt.forRangeTraceSafeR,
    Op.SafeAtR.eq_def, MemAccess.SafeAtR, MaskOpt.SafeAtR, MaskOpt.ActiveR,
    MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR,
    stepStmtR, stepStmtsR, evalOpR.eq_def,
    stepForRangeAuxR_step_lt, stepForRangeAuxR_step_ge,
    BlockState.writeMemTypedR_fp16,
    Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop, Tile.reduceSum,
    Tile.reduceSumDrop, Tile.select, TileShape.axisDim, TileShape.eraseAxis,
    TileShape.insertAxisIndex, NumericDType.add, NumericDType.mul,
    NumericDType.div, ComparableDType.lt, hNpos, hNle,
    Nat.not_lt.mpr hNle, hStep]
  exact ⟨hbx, hbw, hbx, hbo⟩

/-! ### The rounded Hoare triple (`hrun`) -/

/-- Termination of the Llama kernel under `execR R`, in the single-block
regime. -/
private theorem rms_norm_fwd_fused_llama_terminatesR (R : RoundingModel)
    (X Y W : RegionName) (stride N B : Nat) (eps : ℝ) (s : BlockState)
    (hNpos : 0 < N) (hNle : N ≤ B) :
    ∃ s', execR R (rms_norm_fwd_fused_llama X Y W stride N B eps).toAlgKernel s
      = some s' := by
  have hB : B ≠ 0 := by omega
  simp [execR, rms_norm_fwd_fused_llama, stepStmtsR, stepStmtR, evalOpR.eq_def,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
        stepForRangeAuxR_step_lt, stepForRangeAuxR_step_ge,
        BlockState.writeMemTypedR_fp16,
        Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop, Tile.reduceSum,
        Tile.reduceSumDrop, Tile.select, TileShape.axisDim, TileShape.eraseAxis,
        TileShape.insertAxisIndex, NumericDType.add, NumericDType.mul,
        NumericDType.div, ComparableDType.lt, hNpos, hNle,
        Nat.not_lt.mpr hNle, hB]

set_option maxHeartbeats 4000000 in
/-- **The rounded readback.** Under `execR R` every write-active lane's cell
holds the ideal real spec value quantized **once** at the fp16 grid: the
`castFloat` and the typed store round at the same grid and collapse through
`llama_storeValue_cast_fp16`. -/
private theorem rms_norm_fwd_fused_llama_readbackR (R : RoundingModel)
    (X Y W : RegionName) (stride N B : Nat) (eps : ℝ) (s s' : BlockState)
    (hNpos : 0 < N) (hNle : N ≤ B)
    (hOutInj : Function.Injective (fun i : Fin B => yOffset s stride i))
    (hExec : execR R (rms_norm_fwd_fused_llama X Y W stride N B eps).toAlgKernel s
      = some s')
    (i : Fin B) (hi : i.val < N) :
    s'.mem Y (yOffset s stride i)
      = MemCell.of FloatDType.fp16.toTileDType
          (FloatDType.fp16.ofReal
            (R.round .fp16 (rmsnormSpec s X W stride N B eps i))) := by
  have hB : B ≠ 0 := by omega
  have h_inj : Function.Injective
      (fun idx : TileIndex [B] => s.pids 0 * stride + idx.1.val) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [yOffset] using h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  simp [execR, rms_norm_fwd_fused_llama, stepStmtsR, stepStmtR, evalOpR.eq_def,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
        stepForRangeAuxR_step_lt, stepForRangeAuxR_step_ge,
        BlockState.writeMemTypedR_fp16,
        Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop, Tile.reduceSum,
        Tile.reduceSumDrop, Tile.select, TileShape.axisDim, TileShape.eraseAxis,
        TileShape.insertAxisIndex, NumericDType.add, NumericDType.mul,
        NumericDType.div, ComparableDType.lt, hNpos, hNle,
        Nat.not_lt.mpr hNle, hB] at hExec
  subst s'
  simp only [yOffset]
  rw [BlockState.scatter_memcell_R_prop_masked_nd R FloatDType.fp16
        (region := Y) _ _ _ _ h_inj (i, PUnit.unit)]
  simp only [hi, ↓reduceIte]
  rw [llama_storeValue_cast_fp16]
  simp [hi, rmsnormSpec, rmsnormCarrierSpec, rmsInvCarrier, rmsVarCarrier,
        rmsInputTile, xOffset, Tile.reduceSum, Tile.reduceSumDrop, Tile.select,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        WithBot.realSqrt, NumericDType.mul]
  rfl

set_option maxHeartbeats 4000000 in
/-- **Cell-level frame.** The only memory event is the single masked in-loop
fp16 store, so every cell outside the active output window is untouched. -/
private theorem rms_norm_fwd_fused_llama_frameR (R : RoundingModel)
    (X Y W : RegionName) (stride N B : Nat) (eps : ℝ) (s s' : BlockState)
    (hNpos : 0 < N) (hNle : N ≤ B)
    (hExec : execR R (rms_norm_fwd_fused_llama X Y W stride N B eps).toAlgKernel s
      = some s')
    (r : RegionName) (oo : Nat)
    (hcond : r ≠ Y ∨ ∀ i : Fin B, i.val < N → oo ≠ yOffset s stride i) :
    s'.mem r oo = s.mem r oo := by
  have hB : B ≠ 0 := by omega
  simp [execR, rms_norm_fwd_fused_llama, stepStmtsR, stepStmtR, evalOpR.eq_def,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
        stepForRangeAuxR_step_lt, stepForRangeAuxR_step_ge,
        BlockState.writeMemTypedR_fp16,
        Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop, Tile.reduceSum,
        Tile.reduceSumDrop, Tile.select, TileShape.axisDim, TileShape.eraseAxis,
        TileShape.insertAxisIndex, NumericDType.add, NumericDType.mul,
        NumericDType.div, ComparableDType.lt, hNpos, hNle,
        Nat.not_lt.mpr hNle, hB] at hExec
  subst s'
  by_cases hr : r = Y
  · subst hr
    rcases hcond with hne | hno
    · exact absurd rfl hne
    · refine Eq.trans (BlockState.foldl_writeMemAsR_preserve_masked_prop R
        FloatDType.fp16 _ _ _ oo _ ?_ _) rfl
      intro k _ hk
      exact fun hbad => hno ⟨k.1.val, k.1.isLt⟩ hk (by simpa [yOffset] using hbad.symm)
  · exact Eq.trans (BlockState.foldl_writeMemAsR_preserve_other_region R
      FloatDType.fp16 _ _ _ r hr oo _ _) rfl

/-! ### The headline -/

set_option maxHeartbeats 4000000 in
/-- **The `⊨[R]` streaming headline (wave-5 S3 per-step emit genre).** For
every rounding model `R`, the faithful `_rms_norm_fwd_fused` (Llama) surface
implements, on its `StreamEmitMasked2DKernelIO₂` signature, the **ideal ℝ
two-pass RMS-norm** over the streamed tiles, quantized **once** at the fp16
grid: emitted window `(t, j)` reads back through `readMemAs .fp16` as
`FloatDType.fp16.ofReal (R.round .fp16 (x[t,j] · (√(Σ_guarded x²/N + eps))⁻¹ ·
w[t,j]))`, where the guarded double sum folds the *entire* `x` stream. The
full Hoare triple is the skin's: for every disjoint flat allocation of
`[X, W, Y]`, every pid pair, and every launch state with both masked input
streams pinned, `execR R` terminates, every write-active emitted cell reads
back as above, and every flat cell outside the active output window is
untouched.

Rounding story — the honest part of this port. The kernel's store is
`tl.store(Y + cols, (y).to(tl.float16), mask=mask)`, which lowers to
`Stmt.store TileDType.fp16` carrying an
`Op.castFloat FloatDType.real FloatDType.fp16` (verified by
`llama_body_decomp`, which is `rfl`), so `outDType := .real` would be
*false* here. Under `execR R` the lane value crosses **two** rounding sites
at the same grid — `evalOpR`'s `R.cast .real .fp16` and the typed store's
`R.storeValue .fp16` — and the model's *defining* `round_idem` field
collapses them to a single `R.round .fp16`
(`RoundingModel.storeValue_cast`, packaged as
`llama_storeValue_cast_fp16`). That is exactly one rounding event per
emitted cell, which is what the skin's `outDType := .fp16` contract asserts.
**No hypothesis on `R` is used**: there is no `R.round .fp16 = id` pin, and
the face holds for every rounding model, including genuinely lossy ones.
The two `.to(tl.float32)` casts, by contrast, are erased outright by the
compute-to-algorithm lowering and are not rounding events in this model.

Layer map: the exact `exec`-level stack above cannot be reused verbatim
(pass 2 is *not* cast-free — it is the rounding pass), so the `execR R` run
is redone directly: the loops are unrolled through the `R` mirrors
`stepForRangeAuxR_step_lt` / `_step_ge` (single iteration each, by the
launch regime), termination is `rms_norm_fwd_fused_llama_terminatesR`, the
rounded per-cell readback is `rms_norm_fwd_fused_llama_readbackR` (masked
`writeMemAsR` scatter readback + the single-round collapse), the per-cell
frame is `rms_norm_fwd_fused_llama_frameR`, safety is
`rms_norm_fwd_fused_llama_traceSafeR`, and the exact stack's carrier-level
`rmsnormSpec` is evaluated into real arithmetic by `llama_spec_eq` before
the stream-lane bridge `rmsLlamaStreamSpec_eq_rmsnormSpec`.

Both hypotheses are the *exact* headline
`rms_norm_fwd_fused_llama_output_summary`'s own launch precondition, with
the same provenance:

* `hNpos : 0 < N`, `hNle : N ≤ BLOCK_SIZE` — the Python wrapper fixes
  `BLOCK_SIZE = 16384` after checking `N ≤ BLOCK_SIZE`, so every real launch
  is in this single-block regime; there both `range(0, N, BLOCK_SIZE)`
  passes execute exactly the `off = 0` iteration, which is what makes the
  stream one step wide (`T := 1`) and the closed form available.

The exact headline's third side condition `hOutInj` is **not** carried here:
the write window `pid₀·stride + i` is injective in `i` outright, and this
face discharges it.

Relation to the exact surface: the exact headline
`rms_norm_fwd_fused_llama_output_summary` (`Realizes_without_Rounding`,
whose fp16 store is modeled by the *exact* `FloatDType.cast .real .fp16`)
above is retained unchanged; this `⊨[R]` face restates the same RMS-norm
content on the streaming emit skin with the cast replaced by the abstract
rounding model, for every `R` at once — at `R := .triv` the two faces carry
the same cell. Both faces are kept per the rounding-as-default doctrine. -/
specification rms_norm_fwd_fused_llama_io_correctness (R : RoundingModel)
    (X Y W : RegionName) (stride N BLOCK_SIZE : Nat) (eps : ℝ)
    (hNpos : 0 < N) (hNle : N ≤ BLOCK_SIZE) :
    rmsnormFusedLlamaKernelIO X Y W stride N BLOCK_SIZE eps ⊨[R]
      fun _ _ xs ws t j => rmsLlamaStreamSpec N BLOCK_SIZE eps xs ws t j := by
  refine StreamEmitMasked2DKernelIO₂.ImplementsR.intro _ ?_ ?_ ?_
  · exact rms_norm_fwd_fused_llama_flattenOk X Y W stride N BLOCK_SIZE eps
  · -- safety walk
    intro bounds s xs ws _hx _hw hbr1 hbr2 hbw
    simp only [rmsnormFusedLlamaKernelIO] at hbr1 hbr2 hbw ⊢
    refine rms_norm_fwd_fused_llama_traceSafeR R bounds X Y W stride N
      BLOCK_SIZE eps hNpos hNle s (fun j hj => ?_) (fun j hj => ?_) (fun j hj => ?_)
    · have h := hbr1 0 j (by simpa using hj)
      simpa using h
    · have h := hbr2 0 j (by simpa using hj)
      simpa using h
    · have h := hbw 0 j (by simpa using hj)
      simpa using h
  · -- the rounded Hoare triple
    intro s₀ xs ws _hundef hx hw
    simp only [rmsnormFusedLlamaKernelIO] at hx hw ⊢
    obtain ⟨sfin, hexec⟩ := rms_norm_fwd_fused_llama_terminatesR R X Y W stride N
      BLOCK_SIZE eps s₀ hNpos hNle
    have hInj : Function.Injective
        (fun i : Fin BLOCK_SIZE => yOffset s₀ stride i) := by
      intro a b hab
      simp only [yOffset] at hab
      exact Fin.ext (Nat.add_left_cancel hab)
    refine ⟨sfin, hexec, ?_, ?_⟩
    · intro t j hj
      have ht : t = 0 := Fin.ext (by omega)
      subst ht
      have hz : (0 : Fin 1).val * BLOCK_SIZE + j.val = j.val := by simp
      have hjN : j.val < N := by rw [hz] at hj; exact hj
      have hcell := rms_norm_fwd_fused_llama_readbackR R X Y W stride N BLOCK_SIZE
        eps s₀ sfin hNpos hNle hInj hexec j hjN
      rw [hz]
      have haddr : s₀.pids 0 * stride + j.val = yOffset s₀ stride j := rfl
      rw [haddr, BlockState.readMemAs_fp16_of_cell hcell,
        rmsLlamaStreamSpec_eq_rmsnormSpec X W s₀ stride N BLOCK_SIZE eps xs ws
          hx hw 0 j hj]
    · intro r oo hcond
      refine rms_norm_fwd_fused_llama_frameR R X Y W stride N BLOCK_SIZE eps s₀
        sfin hNpos hNle hexec r oo ?_
      rcases hcond with hne | hno
      · exact Or.inl hne
      · refine Or.inr fun i hi => ?_
        have h := hno 0 i (by simpa using hi)
        simpa [yOffset] using h

end IOFace

end VeriTile.Bench.TritonBenchG.RmsnormFusedLlama
