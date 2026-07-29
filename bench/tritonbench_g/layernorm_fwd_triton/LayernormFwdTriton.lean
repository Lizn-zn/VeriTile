import VeriTile.Triton

/-!
# `layernorm_fwd_triton` — strict per-kernel correctness

`_layer_norm_fwd_kernel` is a tutorial-style LayerNorm forward: each program
`(Seq, H)` normalizes one row of `X` by its mean and variance, scales by
per-head weights `W`, and writes `Y = ((x - mean) * rstd) * w` where
`rstd = 1 / sqrt(var + eps)`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body, for one program. The host launch (`_layer_norm_fwd_kernel[grid](...)` with
`grid = (X.shape[0], X.shape[1])`, the host-side `BLOCK_SIZE = 128` choice,
scheduling, and how the runtime composes per-program writes into one buffer) is
the *trusted boundary*, not a proof obligation here. Because the two program ids
`Seq = tl.program_id(0)` and `H = tl.program_id(1)` are universally quantified
(via `s.pids 0` / `s.pids 1`), the per-program statement covers every program of
the 2D grid.

## Proof architecture

```
layernorm_fwd_triton_compute_fullN_correct    ← TOP THEOREM (general N, multi-block)
  └─ layernorm_fwd_triton_fullN_correct        ← algorithm-layer readback per column
       ├─ layernorm_fwd_triton_staged_fullN_correct_from_preloop
       ├─ layernormMeanForRange_context_*       ← mean-loop forRange invariant
       ├─ layernormVarForRange_context_*        ← var-loop forRange invariant
       └─ layernormOutForRange_fullN_of_init    ← output-loop forRange invariant
            └─ layernormOutLoopBody_step_output_invariant
layernorm_fwd_triton_compute_correct          ← one-block slice (0 < N ≤ BLOCK_SIZE)
  └─ layernorm_fwd_triton_correct
```

`layernorm_fwd_triton_compute_fullN_correct` is the strongest result: it covers
arbitrary `N` (mean / var / output loops each tiled over `BLOCK_SIZE` and closed
by `forRange` loop invariants). The one-block `*_compute_correct` is the
single-iteration specialization.

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); the `.to(tl.float32)`
input/weight casts and the store cast `(y).to(X.dtype.element_ty)` reduce to
identity at the algorithm layer (post-erasure all dtypes unify to `ℝ`). Both
reduction loops (`_mean`, `_var`) sum over the *padded* `BLOCK_SIZE` blocks, but
out-of-range lanes are masked to `0` (load `other=0.0`, plus the explicit
`tl.where` for the centered values), so each sum equals the logical row length
`N`. The mean is `(∑ x) / N`, the variance `(∑ (x - mean)²) / N`, and
`rstd = 1 / sqrt(var + eps)`; the affine step is `(x - mean) * rstd * w`. The
fullN theorem requires only `0 < BLOCK_SIZE` plus output/input region
disjointness (`X ≠ Y`, `W ≠ Y`); the one-block slice instead assumes
`0 < N`, `N ≤ BLOCK_SIZE`, and output-offset injectivity (`hOutInj`), and does
*not* need `0 < BLOCK_SIZE`/`X ≠ Y`/`W ≠ Y`. The kernel's unused `stride_*_hd`
strides are carried as unused Lean parameters.
-/

namespace VeriTile.Bench.TritonBenchG.LayernormFwdTriton

open VeriTile.Triton

set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

/-- Documented transcription of `layernorm_fwd_triton.py`'s
`_layer_norm_fwd_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `N` / `BLOCK_SIZE: tl.constexpr` → Lean `Nat` parameters.
- The Python `stride_x_hd`, `stride_y_hd`, and `stride_w_hd` parameters are kept
  as unused Lean parameters because the source kernel body does not use them. -/
def layernorm_fwd_triton
    (X W Y : RegionName)
    (stride_x_N stride_x_hn _stride_x_hd
      stride_y_N stride_y_hn _stride_y_hd
      stride_w_hn _stride_w_hd : Nat)
    (N BLOCK_SIZE : Nat) (eps : ℝ) :
  ComputeKernel := triton {
  Seq = tl.program_id(0)
  H = tl.program_id(1)
  X += Seq * $(stride_x_N) + H * $(stride_x_hn)
  Y += Seq * $(stride_y_N) + H * $(stride_y_hn)
  W += H * $(stride_w_hn)
  _mean = tl.zeros([$(BLOCK_SIZE)], dtype=tl.float32)
  for off in range(0, $(N), $(BLOCK_SIZE)) {
    cols = off + tl.arange(0, $(BLOCK_SIZE))
    a = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
    _mean += a
  }
  mean = tl.sum(_mean, axis=0) / $(N)
  _var = tl.zeros([$(BLOCK_SIZE)], dtype=tl.float32)
  for off in range(0, $(N), $(BLOCK_SIZE)) {
    cols = off + tl.arange(0, $(BLOCK_SIZE))
    x = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
    x = tl.where(cols < $(N), x - mean, 0.0)
    _var += x * x
  }
  var = tl.sum(_var, axis=0) / $(N)
  rstd = 1 / tl.sqrt(var + $(eps))
  for off in range(0, $(N), $(BLOCK_SIZE)) {
    cols = off + tl.arange(0, $(BLOCK_SIZE))
    mask = cols < $(N)
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
    x_hat = (x - mean) * rstd
    y = x_hat * w
    tl.store(Y + cols, (y).to(X.dtype.element_ty), mask=mask)
  }
}

def xOffset
    (s : BlockState) (stride_x_N stride_x_hn : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pids 0 * stride_x_N + s.pids 1 * stride_x_hn + i.val

def yOffset
    (s : BlockState) (stride_y_N stride_y_hn : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pids 0 * stride_y_N + s.pids 1 * stride_y_hn + i.val

def wOffset (s : BlockState) (stride_w_hn : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pids 1 * stride_w_hn + i.val

noncomputable def layernormInputTile
    (s : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE : Nat) :
    Tile .real [BLOCK_SIZE] :=
  { data := fun idx =>
      if idx.1.val < N then
        some (s.readMem X (xOffset s stride_x_N stride_x_hn idx.1))
      else some (0.0 : ℝ) }

noncomputable def layernormMeanCarrier
    (s : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE : Nat) :
    WithBot ℝ :=
  Option.map (fun a => a / (N : ℝ))
    ((Tile.reduceSum (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false
      (layernormInputTile s X stride_x_N stride_x_hn N BLOCK_SIZE)).data
        PUnit.unit)

noncomputable def layernormCenteredTile
    (s : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE : Nat) :
    Tile .real [BLOCK_SIZE] :=
  { data := fun idx =>
      if idx.1.val < N then
        Option.map₂ (fun x mean => x - mean)
          (if idx.1.val < N then
            some (s.readMem X (xOffset s stride_x_N stride_x_hn idx.1))
          else some (0.0 : ℝ))
          (layernormMeanCarrier s X stride_x_N stride_x_hn N BLOCK_SIZE)
      else some (0.0 : ℝ) }

noncomputable def layernormVarCarrier
    (s : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE : Nat) :
    WithBot ℝ :=
  Option.map (fun a => a / (N : ℝ))
    ((Tile.reduceSum (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false
      (Tile.bop (NumericDType.mul .real) (Broadcast.consSame Broadcast.nil)
        (layernormCenteredTile s X stride_x_N stride_x_hn N BLOCK_SIZE)
        (layernormCenteredTile s X stride_x_N stride_x_hn N BLOCK_SIZE))).data
        PUnit.unit)

noncomputable def layernormInvVarCarrier
    (s : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE : Nat) (eps : ℝ) :
    WithBot ℝ :=
  Option.map (fun b => b⁻¹)
    (WithBot.realSqrt
      (Option.map (fun a => a + eps)
        (layernormVarCarrier s X stride_x_N stride_x_hn N BLOCK_SIZE)))

noncomputable def layernormYSpec
    (s : BlockState) (X W : RegionName)
    (stride_x_N stride_x_hn stride_w_hn N BLOCK_SIZE : Nat) (eps : ℝ)
    (i : Fin BLOCK_SIZE) : ℝ :=
  WithBot.unbotD 0
    (Option.map₂ (fun scaled w => scaled * w)
      (Option.map₂ (fun centered inv => centered * inv)
        (Option.map₂ (fun x mean => x - mean)
          (some (s.readMem X (xOffset s stride_x_N stride_x_hn i)))
          (layernormMeanCarrier s X stride_x_N stride_x_hn N BLOCK_SIZE))
        (layernormInvVarCarrier s X stride_x_N stride_x_hn N BLOCK_SIZE eps))
      (some (s.readMem W (wOffset s stride_w_hn i))))

def xColOffset
    (s : BlockState) (stride_x_N stride_x_hn : Nat) (col : Nat) : Nat :=
  s.pids 0 * stride_x_N + s.pids 1 * stride_x_hn + col

def yColOffset
    (s : BlockState) (stride_y_N stride_y_hn : Nat) (col : Nat) : Nat :=
  s.pids 0 * stride_y_N + s.pids 1 * stride_y_hn + col

def wColOffset (s : BlockState) (stride_w_hn : Nat) (col : Nat) : Nat :=
  s.pids 1 * stride_w_hn + col

/-- Full-N mean used by the Python `for off in range(0, N, BLOCK_SIZE)` path. -/
noncomputable def layernormMeanFullNSpec
    (s : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N _BLOCK_SIZE : Nat) : ℝ :=
  (∑ j : Fin N, s.readMem X (xColOffset s stride_x_N stride_x_hn j.val)) /
    (N : ℝ)

/-- Full-N variance after subtracting the full-N mean. -/
noncomputable def layernormVarFullNSpec
    (s : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE : Nat) : ℝ :=
  (∑ j : Fin N,
      (s.readMem X (xColOffset s stride_x_N stride_x_hn j.val) -
        layernormMeanFullNSpec s X stride_x_N stride_x_hn N BLOCK_SIZE)^2) /
    (N : ℝ)

noncomputable def layernormRstdFullNSpec
    (s : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE : Nat) (eps : ℝ) : ℝ :=
  (Real.sqrt
    (layernormVarFullNSpec s X stride_x_N stride_x_hn N BLOCK_SIZE + eps))⁻¹

/-- Full-N output spec for every Python-observable output column. -/
noncomputable def layernormYFullNSpec
    (s : BlockState) (X W : RegionName)
    (stride_x_N stride_x_hn stride_w_hn N BLOCK_SIZE : Nat) (eps : ℝ)
    (i : Fin N) : ℝ :=
  ((s.readMem X (xColOffset s stride_x_N stride_x_hn i.val) -
      layernormMeanFullNSpec s X stride_x_N stride_x_hn N BLOCK_SIZE) *
    layernormRstdFullNSpec s X stride_x_N stride_x_hn N BLOCK_SIZE eps) *
    s.readMem W (wColOffset s stride_w_hn i.val)

/-- Statements before the first reduction loop. -/
def layernormMeanPreLoop
    (X W Y : RegionName)
    (stride_x_N stride_x_hn stride_y_N stride_y_hn stride_w_hn
      BLOCK_SIZE : Nat) :
    List Stmt :=
  [ .assign .nat [] "Seq" (.programId 0)
  , .assign .nat [] "H" (.programId 1)
  , .assign .ptr [] "X"
      (.ptrAdd Broadcast.nil (.ptrBase X)
        (.add NumericDType.nat Broadcast.nil
          (.mul NumericDType.nat Broadcast.nil
            (.ref .nat [] "Seq") (.constNat stride_x_N))
          (.mul NumericDType.nat Broadcast.nil
            (.ref .nat [] "H") (.constNat stride_x_hn))))
  , .assign .ptr [] "Y"
      (.ptrAdd Broadcast.nil (.ptrBase Y)
        (.add NumericDType.nat Broadcast.nil
          (.mul NumericDType.nat Broadcast.nil
            (.ref .nat [] "Seq") (.constNat stride_y_N))
          (.mul NumericDType.nat Broadcast.nil
            (.ref .nat [] "H") (.constNat stride_y_hn))))
  , .assign .ptr [] "W"
      (.ptrAdd Broadcast.nil (.ptrBase W)
        (.mul NumericDType.nat Broadcast.nil
          (.ref .nat [] "H") (.constNat stride_w_hn)))
  , .assign .real [BLOCK_SIZE] "_mean"
      (.full [BLOCK_SIZE] (.const 0))
  ]

def layernormMeanLoopBody
    (N BLOCK_SIZE : Nat) : List Stmt :=
  [ .assign .nat [BLOCK_SIZE] "cols"
      (.add NumericDType.nat Broadcast.scalarL
        (.ref .nat [] "off")
        (.arange BLOCK_SIZE))
  , .assign .real [BLOCK_SIZE] "a"
      (.load .real
        (.ptr
          (.ptrAdd Broadcast.scalarL
            (.ref .ptr [] "X")
            (.ref .nat [BLOCK_SIZE] "cols")))
        (.maskOther
          (.lt ComparableDType.nat Broadcast.scalarR
            (.ref .nat [BLOCK_SIZE] "cols") (.constNat N))
          ((Op.const 0.0).broadcast [BLOCK_SIZE])))
  , .assign .real [BLOCK_SIZE] "_mean"
      (.add NumericDType.real Broadcast.nil.consSame
        (.ref .real [BLOCK_SIZE] "_mean")
        (.ref .real [BLOCK_SIZE] "a"))
  ]

def layernormMeanPostLoop (N BLOCK_SIZE : Nat) : List Stmt :=
  [ .assign .real [] "mean"
      (.div NumericDType.real Broadcast.nil
        (.reduceSum (⟨0, by simp⟩ : Fin [BLOCK_SIZE].length) Bool.false
          (.ref .real [BLOCK_SIZE] "_mean"))
        (.const (N : ℝ)))
  , .assign .real [BLOCK_SIZE] "_var"
      (.full [BLOCK_SIZE] (.const 0))
  ]

def layernormVarLoopBody
    (N BLOCK_SIZE : Nat) : List Stmt :=
  [ .assign .nat [BLOCK_SIZE] "cols"
      (.add NumericDType.nat Broadcast.scalarL
        (.ref .nat [] "off")
        (.arange BLOCK_SIZE))
  , .assign .real [BLOCK_SIZE] "x"
      (.load .real
        (.ptr
          (.ptrAdd Broadcast.scalarL
            (.ref .ptr [] "X")
            (.ref .nat [BLOCK_SIZE] "cols")))
        (.maskOther
          (.lt ComparableDType.nat Broadcast.scalarR
            (.ref .nat [BLOCK_SIZE] "cols") (.constNat N))
          ((Op.const 0.0).broadcast [BLOCK_SIZE])))
  , .assign .real [BLOCK_SIZE] "x"
      (.where
        (.lt ComparableDType.nat Broadcast.scalarR
          (.ref .nat [BLOCK_SIZE] "cols") (.constNat N))
        (.sub NumericDType.real Broadcast.scalarR
          (.ref .real [BLOCK_SIZE] "x")
          (.ref .real [] "mean"))
        ((Op.const 0.0).broadcast [BLOCK_SIZE]))
  , .assign .real [BLOCK_SIZE] "_var"
      (.add NumericDType.real Broadcast.nil.consSame
        (.ref .real [BLOCK_SIZE] "_var")
        (.mul NumericDType.real Broadcast.nil.consSame
          (.ref .real [BLOCK_SIZE] "x")
          (.ref .real [BLOCK_SIZE] "x")))
  ]

def layernormVarPostLoop (N BLOCK_SIZE : Nat) (eps : ℝ) : List Stmt :=
  [ .assign .real [] "var"
      (.div NumericDType.real Broadcast.nil
        (.reduceSum (⟨0, by simp⟩ : Fin [BLOCK_SIZE].length) Bool.false
          (.ref .real [BLOCK_SIZE] "_var"))
        (.const (N : ℝ)))
  , .assign .real [] "rstd"
      (.div NumericDType.real Broadcast.nil
        (.const 1)
        (.sqrt
          (.add NumericDType.real Broadcast.nil
            (.ref .real [] "var")
            (.const eps))))
  ]

def layernormOutLoopBody
    (N BLOCK_SIZE : Nat) :
    List Stmt :=
  [ .assign .nat [BLOCK_SIZE] "cols"
      (.add NumericDType.nat Broadcast.scalarL
        (.ref .nat [] "off")
        (.arange BLOCK_SIZE))
  , .assign .bool [BLOCK_SIZE] "mask"
      (.lt ComparableDType.nat Broadcast.scalarR
        (.ref .nat [BLOCK_SIZE] "cols") (.constNat N))
  , .assign .real [BLOCK_SIZE] "w"
      (.load .real
        (.ptr
          (.ptrAdd Broadcast.scalarL
            (.ref .ptr [] "W")
            (.ref .nat [BLOCK_SIZE] "cols")))
        (.mask (.ref .bool [BLOCK_SIZE] "mask")))
  , .assign .real [BLOCK_SIZE] "x"
      (.load .real
        (.ptr
          (.ptrAdd Broadcast.scalarL
            (.ref .ptr [] "X")
            (.ref .nat [BLOCK_SIZE] "cols")))
        (.maskOther
          (.ref .bool [BLOCK_SIZE] "mask")
          ((Op.const 0.0).broadcast [BLOCK_SIZE])))
  , .assign .real [BLOCK_SIZE] "x_hat"
      (.mul NumericDType.real Broadcast.scalarR
        (.sub NumericDType.real Broadcast.scalarR
          (.ref .real [BLOCK_SIZE] "x")
          (.ref .real [] "mean"))
        (.ref .real [] "rstd"))
  , .assign .real [BLOCK_SIZE] "y"
      (.mul NumericDType.real Broadcast.nil.consSame
        (.ref .real [BLOCK_SIZE] "x_hat")
        (.ref .real [BLOCK_SIZE] "w"))
  , .store .real [BLOCK_SIZE]
      (.ptr
        (.ptrAdd Broadcast.scalarL
          (.ref .ptr [] "Y")
          (.ref .nat [BLOCK_SIZE] "cols")))
      (.ref .real [BLOCK_SIZE] "y")
      (.mask (.ref .bool [BLOCK_SIZE] "mask"))
  ]

theorem layernorm_fwd_triton_toAlg_body
    (X W Y : RegionName)
    (stride_x_N stride_x_hn stride_x_hd
      stride_y_N stride_y_hn stride_y_hd stride_w_hn stride_w_hd
      N BLOCK_SIZE : Nat)
    (eps : ℝ) :
    (layernorm_fwd_triton X W Y
      stride_x_N stride_x_hn stride_x_hd
      stride_y_N stride_y_hn stride_y_hd stride_w_hn stride_w_hd
      N BLOCK_SIZE eps).toAlgKernel.body =
      layernormMeanPreLoop X W Y stride_x_N stride_x_hn stride_y_N stride_y_hn
        stride_w_hn BLOCK_SIZE ++
      [ .forRange "off" 0 N BLOCK_SIZE
          (layernormMeanLoopBody N BLOCK_SIZE) ] ++
      layernormMeanPostLoop N BLOCK_SIZE ++
      [ .forRange "off" 0 N BLOCK_SIZE
          (layernormVarLoopBody N BLOCK_SIZE) ] ++
      layernormVarPostLoop N BLOCK_SIZE eps ++
      [ .forRange "off" 0 N BLOCK_SIZE
          (layernormOutLoopBody N BLOCK_SIZE) ] := by
  simp [layernorm_fwd_triton, layernormMeanPreLoop, layernormMeanLoopBody,
    layernormMeanPostLoop, layernormVarLoopBody, layernormVarPostLoop,
    layernormOutLoopBody, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType]

/-- Per-lane mean-loop partial sum after columns `0..off`. -/
noncomputable def layernormMeanLanePrefix
    (s : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE off : Nat)
    (j : Fin BLOCK_SIZE) : ℝ :=
  ((Finset.range off).filter fun col => col < N ∧ col % BLOCK_SIZE = j.val).sum
    fun col => s.readMem X (xColOffset s stride_x_N stride_x_hn col)

noncomputable def layernormMeanAccumulatorSpec
    (s : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE off : Nat) :
    Tile .real [BLOCK_SIZE] :=
  { data := fun idx =>
      some (layernormMeanLanePrefix s X stride_x_N stride_x_hn
        N BLOCK_SIZE off idx.1) }

theorem layernormMeanLanePrefix_final_eq_class_sum
    (s : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE off : Nat)
    (j : Fin BLOCK_SIZE) (hFinal : N ≤ off) :
    layernormMeanLanePrefix s X stride_x_N stride_x_hn N BLOCK_SIZE off j =
      ((Finset.range N).filter fun col => col % BLOCK_SIZE = j.val).sum
        fun col => s.readMem X (xColOffset s stride_x_N stride_x_hn col) := by
  classical
  unfold layernormMeanLanePrefix
  apply Finset.sum_congr
  · ext col
    simp only [Finset.mem_filter, Finset.mem_range, and_assoc]
    constructor
    · intro h
      exact ⟨h.2.1, h.2.2⟩
    · intro h
      exact ⟨Nat.lt_of_lt_of_le h.1 hFinal, h.1, h.2⟩
  · intro col _; rfl

theorem layernormMeanFullNCarrier_partition_by_lane
    (s : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE : Nat)
    (hB : 0 < BLOCK_SIZE) :
    (∑ j : Fin N, s.readMem X (xColOffset s stride_x_N stride_x_hn j.val)) =
      ∑ j : Fin BLOCK_SIZE,
        ((Finset.range N).filter fun col => col % BLOCK_SIZE = j.val).sum
          fun col => s.readMem X (xColOffset s stride_x_N stride_x_hn col) := by
  classical
  rw [Finset.sum_fin_eq_sum_range]
  rw [← Finset.sum_biUnion]
  · apply Finset.sum_congr
    · ext col
      simp only [Finset.mem_biUnion, Finset.mem_univ, true_and,
        Finset.mem_filter, Finset.mem_range]
      constructor
      · intro hcol
        exact ⟨⟨col % BLOCK_SIZE, Nat.mod_lt col hB⟩, hcol, rfl⟩
      · rintro ⟨j, hcol, _hmod⟩
        exact hcol
    · intro col hmem
      have hcol : col < N := by
        simp only [Finset.mem_biUnion, Finset.mem_univ, true_and,
          Finset.mem_filter, Finset.mem_range] at hmem
        rcases hmem with ⟨j, hcol, _hmod⟩
        exact hcol
      simp [hcol]
  · intro i _ j _ hij
    apply Finset.disjoint_left.mpr
    intro col hi hj
    simp only [Finset.mem_filter, Finset.mem_range] at hi hj
    have hval : i.val = j.val := by
      rw [← hi.2, ← hj.2]
    exact hij (Fin.ext hval)

theorem layernormMeanAccumulatorSpec_final_sum_eq_fullN
    (s : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE off : Nat)
    (hB : 0 < BLOCK_SIZE) (hFinal : N ≤ off) :
    (∑ j : Fin BLOCK_SIZE,
      WithBot.unbotD 0
        ((layernormMeanAccumulatorSpec s X stride_x_N stride_x_hn
          N BLOCK_SIZE off).data (j, PUnit.unit))) =
      ∑ j : Fin N, s.readMem X (xColOffset s stride_x_N stride_x_hn j.val) := by
  classical
  rw [layernormMeanFullNCarrier_partition_by_lane s X stride_x_N stride_x_hn
    N BLOCK_SIZE hB]
  apply Finset.sum_congr rfl
  intro j _
  simp [layernormMeanAccumulatorSpec,
    layernormMeanLanePrefix_final_eq_class_sum s X stride_x_N stride_x_hn
      N BLOCK_SIZE off j hFinal]

theorem layernormMeanAccumulatorSpec_reduceSum
    (s : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE off : Nat) :
    (Tile.reduceSum (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false
      (layernormMeanAccumulatorSpec s X stride_x_N stride_x_hn
        N BLOCK_SIZE off)).data PUnit.unit =
      some
        (∑ j : Fin BLOCK_SIZE,
          WithBot.unbotD 0
            ((layernormMeanAccumulatorSpec s X stride_x_N stride_x_hn
              N BLOCK_SIZE off).data (j, PUnit.unit))) := by
  simp [Tile.reduceSum, Tile.reduceSumDrop, TileShape.axisDim,
    TileShape.eraseAxis, TileShape.insertAxisIndex, layernormMeanAccumulatorSpec]
  rfl

noncomputable def layernormMeanChunkSpec
    (s : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE off : Nat) :
    Tile .real [BLOCK_SIZE] :=
  { data := fun idx =>
      some
        (if off + idx.1.val < N then
          s.readMem X (xColOffset s stride_x_N stride_x_hn (off + idx.1.val))
        else
          0) }

@[simp] theorem layernormMeanLanePrefix_zero
    (s : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE : Nat)
    (j : Fin BLOCK_SIZE) :
    layernormMeanLanePrefix s X stride_x_N stride_x_hn
        N BLOCK_SIZE 0 j = 0 := by
  simp [layernormMeanLanePrefix]

theorem layernormMeanAccumulatorSpec_zero
    (s : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE : Nat) :
    layernormMeanAccumulatorSpec s X stride_x_N stride_x_hn
        N BLOCK_SIZE 0 =
      { data := fun _ : TileIndex [BLOCK_SIZE] => some 0 } := by
  ext idx
  simp [layernormMeanAccumulatorSpec]

theorem layernormLoopOffset_mod_step
    (off BLOCK_SIZE : Nat) (hOff : off % BLOCK_SIZE = 0) :
    (off + BLOCK_SIZE) % BLOCK_SIZE = 0 := by
  rw [Nat.add_mod, hOff]
  simp

theorem layernormChunkLane_mod
    (off BLOCK_SIZE : Nat) (j : Fin BLOCK_SIZE)
    (hOff : off % BLOCK_SIZE = 0) :
    (off + j.val) % BLOCK_SIZE = j.val := by
  rw [Nat.add_mod, hOff]
  simp [Nat.mod_eq_of_lt j.isLt]

theorem layernormChunkLane_not_mem_current
    (N off BLOCK_SIZE : Nat) (j : Fin BLOCK_SIZE) :
    off + j.val ∉ (Finset.range off).filter
      (fun col => col < N ∧ col % BLOCK_SIZE = j.val) := by
  simp

theorem layernormMeanLanePrefix_step
    (s : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE off : Nat)
    (j : Fin BLOCK_SIZE) (hOff : off % BLOCK_SIZE = 0) :
    layernormMeanLanePrefix s X stride_x_N stride_x_hn
        N BLOCK_SIZE (off + BLOCK_SIZE) j =
      layernormMeanLanePrefix s X stride_x_N stride_x_hn
          N BLOCK_SIZE off j +
        if off + j.val < N then
          s.readMem X (xColOffset s stride_x_N stride_x_hn (off + j.val))
        else
          0 := by
  classical
  let pred : Nat → Prop := fun col => col < N ∧ col % BLOCK_SIZE = j.val
  let f : Nat → ℝ := fun col =>
    s.readMem X (xColOffset s stride_x_N stride_x_hn col)
  have hunique :
      ∀ col, off ≤ col → col < off + BLOCK_SIZE → col % BLOCK_SIZE = j.val →
        col = off + j.val := by
    intro col hle hlt hmod
    obtain ⟨k, rfl⟩ := Nat.exists_eq_add_of_le hle
    have hklt : k < BLOCK_SIZE := by omega
    have hmodk : (off + k) % BLOCK_SIZE = k := by
      rw [Nat.add_mod, hOff, Nat.mod_eq_of_lt hklt]
      simpa using Nat.mod_eq_of_lt hklt
    rw [hmodk] at hmod
    omega
  by_cases hjN : off + j.val < N
  · have hset :
        (Finset.range (off + BLOCK_SIZE)).filter pred =
          insert (off + j.val) ((Finset.range off).filter pred) := by
      ext col
      simp only [Finset.mem_filter, Finset.mem_range, Finset.mem_insert]
      constructor
      · intro h
        by_cases hltOff : col < off
        · exact Or.inr ⟨hltOff, h.2⟩
        · exact Or.inl (hunique col (Nat.le_of_not_gt hltOff) h.1 h.2.2)
      · intro h
        rcases h with h | h
        · subst h
          exact ⟨by omega, hjN, layernormChunkLane_mod off BLOCK_SIZE j hOff⟩
        · exact ⟨by omega, h.2⟩
    unfold layernormMeanLanePrefix
    simp [hjN]
    change ((Finset.range (off + BLOCK_SIZE)).filter pred).sum f =
      ((Finset.range off).filter pred).sum f + f (off + j.val)
    rw [hset]
    rw [Finset.sum_insert]
    · ring
    · exact layernormChunkLane_not_mem_current N off BLOCK_SIZE j
  · have hset :
        (Finset.range (off + BLOCK_SIZE)).filter pred =
          (Finset.range off).filter pred := by
      ext col
      simp only [Finset.mem_filter, Finset.mem_range]
      constructor
      · intro h
        by_cases hltOff : col < off
        · exact ⟨hltOff, h.2⟩
        · have hcol : col = off + j.val :=
            hunique col (Nat.le_of_not_gt hltOff) h.1 h.2.2
          omega
      · intro h
        exact ⟨by omega, h.2⟩
    unfold layernormMeanLanePrefix
    simp [hjN]
    change ((Finset.range (off + BLOCK_SIZE)).filter pred).sum f =
      ((Finset.range off).filter pred).sum f
    rw [hset]

theorem layernormMeanAccumulatorSpec_step
    (s : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE off : Nat)
    (hOff : off % BLOCK_SIZE = 0) :
    layernormMeanAccumulatorSpec s X stride_x_N stride_x_hn
        N BLOCK_SIZE (off + BLOCK_SIZE) =
      { data := fun idx : TileIndex [BLOCK_SIZE] =>
          some
            (WithBot.unbotD 0
                ((layernormMeanAccumulatorSpec s X stride_x_N stride_x_hn
                  N BLOCK_SIZE off).data idx) +
              WithBot.unbotD 0
                ((layernormMeanChunkSpec s X stride_x_N stride_x_hn
                  N BLOCK_SIZE off).data idx)) } := by
  ext idx
  by_cases hcol : off + idx.1.val < N
  · simp [layernormMeanAccumulatorSpec, layernormMeanChunkSpec, hcol,
      layernormMeanLanePrefix_step s X stride_x_N stride_x_hn
        N BLOCK_SIZE off idx.1 hOff]
  · simp [layernormMeanAccumulatorSpec, layernormMeanChunkSpec, hcol,
      layernormMeanLanePrefix_step s X stride_x_N stride_x_hn
        N BLOCK_SIZE off idx.1 hOff]

theorem layernormMeanPreLoop_step_regs
    (s st : BlockState) (X W Y : RegionName)
    (stride_x_N stride_x_hn stride_y_N stride_y_hn stride_w_hn BLOCK_SIZE : Nat)
    (hStep :
      stepStmts
        (layernormMeanPreLoop X W Y stride_x_N stride_x_hn stride_y_N
          stride_y_hn stride_w_hn BLOCK_SIZE) s = some st) :
    st.regs .real [BLOCK_SIZE] "_mean" =
        some { data := fun _ : TileIndex [BLOCK_SIZE] => some 0 } ∧
      st.regs .ptr [] "X" =
        some (Tile.scalar (Region.cast X,
          s.pids 0 * stride_x_N + s.pids 1 * stride_x_hn)) ∧
      st.regs .ptr [] "Y" =
        some (Tile.scalar (Region.cast Y,
          s.pids 0 * stride_y_N + s.pids 1 * stride_y_hn)) ∧
      st.regs .ptr [] "W" =
        some (Tile.scalar (Region.cast W, s.pids 1 * stride_w_hn)) ∧
      st.regs .nat [] "Seq" = some (Tile.scalar (s.pids 0)) ∧
      st.regs .nat [] "H" = some (Tile.scalar (s.pids 1)) ∧
      (∀ R offset, st.readMem R offset = s.readMem R offset) := by
  unfold layernormMeanPreLoop at hStep
  simp [stepStmts, stepStmt, evalOp, evalOp.eq_def, Tile.bop, Tile.ptrAdd,
    NumericDType.add, NumericDType.mul, Option.bind] at hStep
  subst st
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp [BlockState.setReg]
  · simp [BlockState.setReg]
  · simp [BlockState.setReg]
  · simp [BlockState.setReg]
  · simp [BlockState.setReg]
  · simp [BlockState.setReg]
  · intro R offset
    rfl

theorem layernormMeanLoopBody_step_accumulator_update
    (s0 st st' : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE off : Nat)
    (hAcc :
      st.regs .real [BLOCK_SIZE] "_mean" =
        some (layernormMeanAccumulatorSpec s0 X stride_x_N stride_x_hn
          N BLOCK_SIZE off))
    (hXPtr :
      st.regs .ptr [] "X" =
        some (Tile.scalar (Region.cast X,
          s0.pids 0 * stride_x_N + s0.pids 1 * stride_x_hn)))
    (hRead : ∀ offset, st.readMem X offset = s0.readMem X offset)
    (hStep :
      stepStmts (layernormMeanLoopBody N BLOCK_SIZE)
        (st.setReg "off" .nat [] (Tile.scalar off)) = some st') :
    st'.regs .real [BLOCK_SIZE] "_mean" =
      some
        { data := fun idx : TileIndex [BLOCK_SIZE] =>
            some
              (WithBot.unbotD 0
                  ((layernormMeanAccumulatorSpec s0 X stride_x_N stride_x_hn
                    N BLOCK_SIZE off).data idx) +
                WithBot.unbotD 0
                  ((layernormMeanChunkSpec s0 X stride_x_N stride_x_hn
                    N BLOCK_SIZE off).data idx)) } := by
  unfold layernormMeanLoopBody at hStep
  simp [stepStmts, stepStmt, evalOp, evalOp.eq_def, hAcc, hXPtr, Tile.bop, Tile.ptrAdd,
    NumericDType.add, ComparableDType.lt, Option.bind, hRead, xColOffset] at hStep
  subst st'
  simp [BlockState.setReg]
  ext idx
  by_cases hcol : off + idx.1.val < N
  · simp [layernormMeanAccumulatorSpec, layernormMeanChunkSpec, hcol,
      xColOffset]
  · simp [layernormMeanAccumulatorSpec, layernormMeanChunkSpec, hcol,
      xColOffset]
    constructor
    · intro h
      linarith
    · intro h
      linarith

theorem layernormMeanLoopBody_step_preserves_context
    (s0 st st' : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE off : Nat)
    (hAcc :
      st.regs .real [BLOCK_SIZE] "_mean" =
        some (layernormMeanAccumulatorSpec s0 X stride_x_N stride_x_hn
          N BLOCK_SIZE off))
    (hXPtr :
      st.regs .ptr [] "X" =
        some (Tile.scalar (Region.cast X,
          s0.pids 0 * stride_x_N + s0.pids 1 * stride_x_hn)))
    (hStep :
      stepStmts (layernormMeanLoopBody N BLOCK_SIZE)
        (st.setReg "off" .nat [] (Tile.scalar off)) = some st') :
    st'.regs .ptr [] "X" =
        some (Tile.scalar (Region.cast X,
          s0.pids 0 * stride_x_N + s0.pids 1 * stride_x_hn)) ∧
      (∀ R offset, st'.readMem R offset = st.readMem R offset) := by
  unfold layernormMeanLoopBody at hStep
  simp [stepStmts, stepStmt, evalOp, evalOp.eq_def, hAcc, hXPtr, Tile.bop, Tile.ptrAdd,
    NumericDType.add, ComparableDType.lt, Option.bind] at hStep
  subst st'
  refine ⟨?_, ?_⟩
  · simp [BlockState.setReg, hXPtr]
  · intro R offset
    rfl

def layernormMeanLoopInvariant
    (s0 : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE off : Nat)
    (st : BlockState) : Prop :=
  off % BLOCK_SIZE = 0 ∧
    st.regs .real [BLOCK_SIZE] "_mean" =
      some (layernormMeanAccumulatorSpec s0 X stride_x_N stride_x_hn
        N BLOCK_SIZE off)

def layernormMeanLoopContextInvariant
    (s0 : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE off : Nat)
    (st : BlockState) : Prop :=
  layernormMeanLoopInvariant s0 X stride_x_N stride_x_hn N BLOCK_SIZE off st ∧
    st.regs .ptr [] "X" =
      some (Tile.scalar (Region.cast X,
        s0.pids 0 * stride_x_N + s0.pids 1 * stride_x_hn)) ∧
    (∀ offset, st.readMem X offset = s0.readMem X offset)

theorem layernormMeanLoopInvariant_init_of_zero_reg
    (s0 st : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE : Nat)
    (hReg :
      st.regs .real [BLOCK_SIZE] "_mean" =
        some { data := fun _ : TileIndex [BLOCK_SIZE] => some 0 }) :
    layernormMeanLoopInvariant s0 X stride_x_N stride_x_hn N BLOCK_SIZE 0 st := by
  refine ⟨?_, ?_⟩
  · simp
  · simpa [layernormMeanAccumulatorSpec_zero] using hReg

theorem layernormMeanLoopContextInvariant_init_of_preloop
    (s0 st : BlockState) (X W Y : RegionName)
    (stride_x_N stride_x_hn stride_y_N stride_y_hn stride_w_hn N BLOCK_SIZE : Nat)
    (hStep :
      stepStmts
        (layernormMeanPreLoop X W Y stride_x_N stride_x_hn stride_y_N
          stride_y_hn stride_w_hn BLOCK_SIZE) s0 = some st) :
    layernormMeanLoopContextInvariant s0 X stride_x_N stride_x_hn
      N BLOCK_SIZE 0 st := by
  rcases layernormMeanPreLoop_step_regs s0 st X W Y stride_x_N stride_x_hn
      stride_y_N stride_y_hn stride_w_hn BLOCK_SIZE hStep with
    ⟨hZero, hXPtr, _hYPtr, _hWPtr, _hSeq, _hH, hRead⟩
  exact ⟨layernormMeanLoopInvariant_init_of_zero_reg s0 st X stride_x_N
      stride_x_hn N BLOCK_SIZE hZero, hXPtr, hRead X⟩

theorem layernormMeanLoopInvariant_step_of_accumulator_update
    (s0 st' : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE off : Nat)
    (hOff : off % BLOCK_SIZE = 0)
    (hUpdate :
      st'.regs .real [BLOCK_SIZE] "_mean" =
        some
          { data := fun idx : TileIndex [BLOCK_SIZE] =>
              some
                (WithBot.unbotD 0
                    ((layernormMeanAccumulatorSpec s0 X stride_x_N stride_x_hn
                      N BLOCK_SIZE off).data idx) +
                  WithBot.unbotD 0
                    ((layernormMeanChunkSpec s0 X stride_x_N stride_x_hn
                      N BLOCK_SIZE off).data idx)) }) :
    layernormMeanLoopInvariant s0 X stride_x_N stride_x_hn
      N BLOCK_SIZE (off + BLOCK_SIZE) st' := by
  refine ⟨?_, ?_⟩
  · exact layernormLoopOffset_mod_step off BLOCK_SIZE hOff
  · simpa [layernormMeanAccumulatorSpec_step s0 X stride_x_N stride_x_hn
      N BLOCK_SIZE off hOff] using hUpdate

theorem layernormMeanLoopContextInvariant_step_of_body
    (s0 st st' : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE off : Nat)
    (hCtx : layernormMeanLoopContextInvariant s0 X stride_x_N stride_x_hn
      N BLOCK_SIZE off st)
    (hStep :
      stepStmts (layernormMeanLoopBody N BLOCK_SIZE)
        (st.setReg "off" .nat [] (Tile.scalar off)) = some st') :
    layernormMeanLoopContextInvariant s0 X stride_x_N stride_x_hn
      N BLOCK_SIZE (off + BLOCK_SIZE) st' := by
  rcases hCtx with ⟨hInv, hXPtr, hRead⟩
  have hUpdate :=
    layernormMeanLoopBody_step_accumulator_update s0 st st' X stride_x_N
      stride_x_hn N BLOCK_SIZE off hInv.2 hXPtr hRead hStep
  rcases layernormMeanLoopBody_step_preserves_context s0 st st' X stride_x_N
      stride_x_hn N BLOCK_SIZE off hInv.2 hXPtr hStep with
    ⟨hXPtr', hReadStep⟩
  refine ⟨?_, hXPtr', ?_⟩
  · exact layernormMeanLoopInvariant_step_of_accumulator_update s0 st' X
      stride_x_N stride_x_hn N BLOCK_SIZE off hInv.1 hUpdate
  · intro offset
    rw [hReadStep X offset, hRead offset]

theorem layernormMeanLoopContextInvariant_body_step_exists
    (s0 st : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE off : Nat)
    (hCtx : layernormMeanLoopContextInvariant s0 X stride_x_N stride_x_hn
      N BLOCK_SIZE off st) :
    ∃ st',
      stepStmts (layernormMeanLoopBody N BLOCK_SIZE)
        (st.setReg "off" .nat [] (Tile.scalar off)) = some st' ∧
      layernormMeanLoopContextInvariant s0 X stride_x_N stride_x_hn
        N BLOCK_SIZE (off + BLOCK_SIZE) st' := by
  rcases hCtx with ⟨hInv, hXPtr, hRead⟩
  cases hStep :
      stepStmts (layernormMeanLoopBody N BLOCK_SIZE)
        (st.setReg "off" .nat [] (Tile.scalar off)) with
  | none =>
      unfold layernormMeanLoopBody at hStep
      simp [stepStmts, stepStmt, evalOp, evalOp.eq_def, hInv.2, hXPtr, Tile.bop, Tile.ptrAdd,
        NumericDType.add, ComparableDType.lt, Option.bind] at hStep
  | some st' =>
      refine ⟨st', rfl, ?_⟩
      exact layernormMeanLoopContextInvariant_step_of_body s0 st st' X
        stride_x_N stride_x_hn N BLOCK_SIZE off ⟨hInv, hXPtr, hRead⟩ hStep

theorem layernormMeanForRange_context_of_preloop
    (s0 stPre stMean : BlockState) (X W Y : RegionName)
    (stride_x_N stride_x_hn stride_y_N stride_y_hn stride_w_hn N BLOCK_SIZE : Nat)
    (hStepNe : BLOCK_SIZE ≠ 0)
    (hPre :
      stepStmts
        (layernormMeanPreLoop X W Y stride_x_N stride_x_hn stride_y_N
          stride_y_hn stride_w_hn BLOCK_SIZE) s0 = some stPre)
    (hLoop :
      stepStmt (.forRange "off" 0 N BLOCK_SIZE
        (layernormMeanLoopBody N BLOCK_SIZE)) stPre = some stMean) :
    ∃ final,
      N ≤ final ∧
        layernormMeanLoopContextInvariant s0 X stride_x_N stride_x_hn
          N BLOCK_SIZE final stMean := by
  have hInit :
      layernormMeanLoopContextInvariant s0 X stride_x_N stride_x_hn
        N BLOCK_SIZE 0 stPre :=
    layernormMeanLoopContextInvariant_init_of_preloop s0 stPre X W Y
      stride_x_N stride_x_hn stride_y_N stride_y_hn stride_w_hn N
      BLOCK_SIZE hPre
  obtain ⟨final, stFinal, hFor, hFinal, hCtx⟩ :=
    forRange_inv
      (idx := "off") (start := 0) (stop := N) (step := BLOCK_SIZE)
      (body := layernormMeanLoopBody N BLOCK_SIZE)
      (P := layernormMeanLoopContextInvariant s0 X stride_x_N stride_x_hn
        N BLOCK_SIZE)
      (s_init := stPre)
      hStepNe hInit
      (by
        intro off st _hlt hCtx
        exact layernormMeanLoopContextInvariant_body_step_exists s0 st X
          stride_x_N stride_x_hn N BLOCK_SIZE off hCtx)
  have hEq : stFinal = stMean := by
    rw [hLoop] at hFor
    injection hFor with h
    exact h.symm
  subst hEq
  exact ⟨final, hFinal, hCtx⟩

theorem layernormMeanLoopInvariant_reduceSum_to_fullN
    (s0 st : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE off : Nat)
    (hInv : layernormMeanLoopContextInvariant s0 X stride_x_N stride_x_hn
      N BLOCK_SIZE off st)
    (hB : 0 < BLOCK_SIZE) (hFinal : N ≤ off) :
    ∃ acc : Tile .real [BLOCK_SIZE],
      st.regs .real [BLOCK_SIZE] "_mean" = some acc ∧
        WithBot.unbotD 0
          ((Tile.reduceSum (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false
            acc).data PUnit.unit) / (N : ℝ) =
          layernormMeanFullNSpec s0 X stride_x_N stride_x_hn N BLOCK_SIZE := by
  refine ⟨layernormMeanAccumulatorSpec s0 X stride_x_N stride_x_hn
      N BLOCK_SIZE off, hInv.1.2, ?_⟩
  rw [layernormMeanAccumulatorSpec_reduceSum]
  simp [layernormMeanFullNSpec,
    layernormMeanAccumulatorSpec_final_sum_eq_fullN s0 X stride_x_N
      stride_x_hn N BLOCK_SIZE off hB hFinal]

/-- Per-lane variance-loop partial sum after columns `0..off`, centered at the
full-N mean computed by the first loop. -/
noncomputable def layernormVarLanePrefix
    (s : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE off : Nat)
    (j : Fin BLOCK_SIZE) : ℝ :=
  ((Finset.range off).filter fun col => col < N ∧ col % BLOCK_SIZE = j.val).sum
    fun col =>
      (s.readMem X (xColOffset s stride_x_N stride_x_hn col) -
        layernormMeanFullNSpec s X stride_x_N stride_x_hn N BLOCK_SIZE)^2

noncomputable def layernormVarAccumulatorSpec
    (s : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE off : Nat) :
    Tile .real [BLOCK_SIZE] :=
  { data := fun idx =>
      some (layernormVarLanePrefix s X stride_x_N stride_x_hn
        N BLOCK_SIZE off idx.1) }

theorem layernormVarLanePrefix_final_eq_class_sum
    (s : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE off : Nat)
    (j : Fin BLOCK_SIZE) (hFinal : N ≤ off) :
    layernormVarLanePrefix s X stride_x_N stride_x_hn N BLOCK_SIZE off j =
      ((Finset.range N).filter fun col => col % BLOCK_SIZE = j.val).sum
        fun col =>
          (s.readMem X (xColOffset s stride_x_N stride_x_hn col) -
            layernormMeanFullNSpec s X stride_x_N stride_x_hn N BLOCK_SIZE)^2 := by
  classical
  unfold layernormVarLanePrefix
  apply Finset.sum_congr
  · ext col
    simp only [Finset.mem_filter, Finset.mem_range, and_assoc]
    constructor
    · intro h
      exact ⟨h.2.1, h.2.2⟩
    · intro h
      exact ⟨Nat.lt_of_lt_of_le h.1 hFinal, h.1, h.2⟩
  · intro col _; rfl

theorem layernormVarFullNCarrier_partition_by_lane
    (s : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE : Nat)
    (hB : 0 < BLOCK_SIZE) :
    (∑ j : Fin N,
      (s.readMem X (xColOffset s stride_x_N stride_x_hn j.val) -
        layernormMeanFullNSpec s X stride_x_N stride_x_hn N BLOCK_SIZE)^2) =
      ∑ j : Fin BLOCK_SIZE,
        ((Finset.range N).filter fun col => col % BLOCK_SIZE = j.val).sum
          fun col =>
            (s.readMem X (xColOffset s stride_x_N stride_x_hn col) -
              layernormMeanFullNSpec s X stride_x_N stride_x_hn N BLOCK_SIZE)^2 := by
  classical
  rw [Finset.sum_fin_eq_sum_range]
  rw [← Finset.sum_biUnion]
  · apply Finset.sum_congr
    · ext col
      simp only [Finset.mem_biUnion, Finset.mem_univ, true_and,
        Finset.mem_filter, Finset.mem_range]
      constructor
      · intro hcol
        exact ⟨⟨col % BLOCK_SIZE, Nat.mod_lt col hB⟩, hcol, rfl⟩
      · rintro ⟨j, hcol, _hmod⟩
        exact hcol
    · intro col hmem
      have hcol : col < N := by
        simp only [Finset.mem_biUnion, Finset.mem_univ, true_and,
          Finset.mem_filter, Finset.mem_range] at hmem
        rcases hmem with ⟨j, hcol, _hmod⟩
        exact hcol
      simp [hcol]
  · intro i _ j _ hij
    apply Finset.disjoint_left.mpr
    intro col hi hj
    simp only [Finset.mem_filter, Finset.mem_range] at hi hj
    have hval : i.val = j.val := by
      rw [← hi.2, ← hj.2]
    exact hij (Fin.ext hval)

theorem layernormVarAccumulatorSpec_final_sum_eq_fullN
    (s : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE off : Nat)
    (hB : 0 < BLOCK_SIZE) (hFinal : N ≤ off) :
    (∑ j : Fin BLOCK_SIZE,
      WithBot.unbotD 0
        ((layernormVarAccumulatorSpec s X stride_x_N stride_x_hn
          N BLOCK_SIZE off).data (j, PUnit.unit))) =
      ∑ j : Fin N,
        (s.readMem X (xColOffset s stride_x_N stride_x_hn j.val) -
          layernormMeanFullNSpec s X stride_x_N stride_x_hn N BLOCK_SIZE)^2 := by
  classical
  rw [layernormVarFullNCarrier_partition_by_lane s X stride_x_N stride_x_hn
    N BLOCK_SIZE hB]
  apply Finset.sum_congr rfl
  intro j _
  simp [layernormVarAccumulatorSpec,
    layernormVarLanePrefix_final_eq_class_sum s X stride_x_N stride_x_hn
      N BLOCK_SIZE off j hFinal]

theorem layernormVarAccumulatorSpec_reduceSum
    (s : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE off : Nat) :
    (Tile.reduceSum (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false
      (layernormVarAccumulatorSpec s X stride_x_N stride_x_hn
        N BLOCK_SIZE off)).data PUnit.unit =
      some
        (∑ j : Fin BLOCK_SIZE,
          WithBot.unbotD 0
            ((layernormVarAccumulatorSpec s X stride_x_N stride_x_hn
              N BLOCK_SIZE off).data (j, PUnit.unit))) := by
  simp [Tile.reduceSum, Tile.reduceSumDrop, TileShape.axisDim,
    TileShape.eraseAxis, TileShape.insertAxisIndex, layernormVarAccumulatorSpec]
  rfl

noncomputable def layernormVarChunkSpec
    (s : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE off : Nat) :
    Tile .real [BLOCK_SIZE] :=
  { data := fun idx =>
      some
        (if off + idx.1.val < N then
          (s.readMem X (xColOffset s stride_x_N stride_x_hn (off + idx.1.val)) -
            layernormMeanFullNSpec s X stride_x_N stride_x_hn N BLOCK_SIZE)^2
        else
          0) }

@[simp] theorem layernormVarLanePrefix_zero
    (s : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE : Nat)
    (j : Fin BLOCK_SIZE) :
    layernormVarLanePrefix s X stride_x_N stride_x_hn
        N BLOCK_SIZE 0 j = 0 := by
  simp [layernormVarLanePrefix]

theorem layernormVarAccumulatorSpec_zero
    (s : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE : Nat) :
    layernormVarAccumulatorSpec s X stride_x_N stride_x_hn
        N BLOCK_SIZE 0 =
      { data := fun _ : TileIndex [BLOCK_SIZE] => some 0 } := by
  ext idx
  simp [layernormVarAccumulatorSpec]

theorem layernormVarLanePrefix_step
    (s : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE off : Nat)
    (j : Fin BLOCK_SIZE) (hOff : off % BLOCK_SIZE = 0) :
    layernormVarLanePrefix s X stride_x_N stride_x_hn
        N BLOCK_SIZE (off + BLOCK_SIZE) j =
      layernormVarLanePrefix s X stride_x_N stride_x_hn
          N BLOCK_SIZE off j +
        if off + j.val < N then
          (s.readMem X (xColOffset s stride_x_N stride_x_hn (off + j.val)) -
            layernormMeanFullNSpec s X stride_x_N stride_x_hn N BLOCK_SIZE)^2
        else
          0 := by
  classical
  let pred : Nat → Prop := fun col => col < N ∧ col % BLOCK_SIZE = j.val
  let f : Nat → ℝ := fun col =>
    (s.readMem X (xColOffset s stride_x_N stride_x_hn col) -
      layernormMeanFullNSpec s X stride_x_N stride_x_hn N BLOCK_SIZE)^2
  have hunique :
      ∀ col, off ≤ col → col < off + BLOCK_SIZE → col % BLOCK_SIZE = j.val →
        col = off + j.val := by
    intro col hle hlt hmod
    obtain ⟨k, rfl⟩ := Nat.exists_eq_add_of_le hle
    have hklt : k < BLOCK_SIZE := by omega
    have hmodk : (off + k) % BLOCK_SIZE = k := by
      rw [Nat.add_mod, hOff, Nat.mod_eq_of_lt hklt]
      simpa using Nat.mod_eq_of_lt hklt
    rw [hmodk] at hmod
    omega
  by_cases hjN : off + j.val < N
  · have hset :
        (Finset.range (off + BLOCK_SIZE)).filter pred =
          insert (off + j.val) ((Finset.range off).filter pred) := by
      ext col
      simp only [Finset.mem_filter, Finset.mem_range, Finset.mem_insert]
      constructor
      · intro h
        by_cases hltOff : col < off
        · exact Or.inr ⟨hltOff, h.2⟩
        · exact Or.inl (hunique col (Nat.le_of_not_gt hltOff) h.1 h.2.2)
      · intro h
        rcases h with h | h
        · subst h
          exact ⟨by omega, hjN, layernormChunkLane_mod off BLOCK_SIZE j hOff⟩
        · exact ⟨by omega, h.2⟩
    unfold layernormVarLanePrefix
    simp [hjN]
    change ((Finset.range (off + BLOCK_SIZE)).filter pred).sum f =
      ((Finset.range off).filter pred).sum f + f (off + j.val)
    rw [hset]
    rw [Finset.sum_insert]
    · ring
    · exact layernormChunkLane_not_mem_current N off BLOCK_SIZE j
  · have hset :
        (Finset.range (off + BLOCK_SIZE)).filter pred =
          (Finset.range off).filter pred := by
      ext col
      simp only [Finset.mem_filter, Finset.mem_range]
      constructor
      · intro h
        by_cases hltOff : col < off
        · exact ⟨hltOff, h.2⟩
        · have hcol : col = off + j.val :=
            hunique col (Nat.le_of_not_gt hltOff) h.1 h.2.2
          omega
      · intro h
        exact ⟨by omega, h.2⟩
    unfold layernormVarLanePrefix
    simp [hjN]
    change ((Finset.range (off + BLOCK_SIZE)).filter pred).sum f =
      ((Finset.range off).filter pred).sum f
    rw [hset]

theorem layernormVarAccumulatorSpec_step
    (s : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE off : Nat)
    (hOff : off % BLOCK_SIZE = 0) :
    layernormVarAccumulatorSpec s X stride_x_N stride_x_hn
        N BLOCK_SIZE (off + BLOCK_SIZE) =
      { data := fun idx : TileIndex [BLOCK_SIZE] =>
          some
            (WithBot.unbotD 0
                ((layernormVarAccumulatorSpec s X stride_x_N stride_x_hn
                  N BLOCK_SIZE off).data idx) +
              WithBot.unbotD 0
                ((layernormVarChunkSpec s X stride_x_N stride_x_hn
                  N BLOCK_SIZE off).data idx)) } := by
  ext idx
  by_cases hcol : off + idx.1.val < N
  · simp [layernormVarAccumulatorSpec, layernormVarChunkSpec, hcol,
      layernormVarLanePrefix_step s X stride_x_N stride_x_hn
        N BLOCK_SIZE off idx.1 hOff]
  · simp [layernormVarAccumulatorSpec, layernormVarChunkSpec, hcol,
      layernormVarLanePrefix_step s X stride_x_N stride_x_hn
        N BLOCK_SIZE off idx.1 hOff]

def layernormVarLoopInvariant
    (s0 : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE off : Nat)
    (st : BlockState) : Prop :=
  off % BLOCK_SIZE = 0 ∧
    st.regs .real [BLOCK_SIZE] "_var" =
      some (layernormVarAccumulatorSpec s0 X stride_x_N stride_x_hn
        N BLOCK_SIZE off)

def layernormVarLoopContextInvariant
    (s0 : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE off : Nat)
    (st : BlockState) : Prop :=
  layernormVarLoopInvariant s0 X stride_x_N stride_x_hn N BLOCK_SIZE off st ∧
    st.regs .ptr [] "X" =
      some (Tile.scalar (Region.cast X,
        s0.pids 0 * stride_x_N + s0.pids 1 * stride_x_hn)) ∧
    st.regs .real [] "mean" =
      some (Tile.scalar (layernormMeanFullNSpec s0 X stride_x_N stride_x_hn
        N BLOCK_SIZE)) ∧
    (∀ offset, st.readMem X offset = s0.readMem X offset)

theorem layernormVarLoopInvariant_init_of_zero_reg
    (s0 st : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE : Nat)
    (hReg :
      st.regs .real [BLOCK_SIZE] "_var" =
        some { data := fun _ : TileIndex [BLOCK_SIZE] => some 0 }) :
    layernormVarLoopInvariant s0 X stride_x_N stride_x_hn N BLOCK_SIZE 0 st := by
  refine ⟨?_, ?_⟩
  · simp
  · simpa [layernormVarAccumulatorSpec_zero] using hReg

theorem layernormMeanPostLoop_step_to_var_init
    (s0 stMean stVar : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE final : Nat)
    (hMeanCtx : layernormMeanLoopContextInvariant s0 X stride_x_N stride_x_hn
      N BLOCK_SIZE final stMean)
    (hFinal : N ≤ final)
    (hBlockPos : 0 < BLOCK_SIZE)
    (hStep : stepStmts (layernormMeanPostLoop N BLOCK_SIZE) stMean = some stVar) :
    layernormVarLoopContextInvariant s0 X stride_x_N stride_x_hn
      N BLOCK_SIZE 0 stVar := by
  rcases layernormMeanLoopInvariant_reduceSum_to_fullN s0 stMean X
      stride_x_N stride_x_hn N BLOCK_SIZE final hMeanCtx hBlockPos hFinal with
    ⟨acc, hMeanReg, hReduce⟩
  have hAccEq :
      acc =
        layernormMeanAccumulatorSpec s0 X stride_x_N stride_x_hn
          N BLOCK_SIZE final := by
    rw [hMeanCtx.1.2] at hMeanReg
    injection hMeanReg with h
    exact h.symm
  subst acc
  unfold layernormMeanPostLoop at hStep
  simp [stepStmts, stepStmt, evalOp, evalOp.eq_def, hMeanReg, Tile.bop, Tile.reduceSum,
    Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis,
    TileShape.insertAxisIndex, NumericDType.div, Option.bind] at hStep
  subst stVar
  refine ⟨?_, ?_, ?_, ?_⟩
  · exact layernormVarLoopInvariant_init_of_zero_reg s0 _ X stride_x_N
      stride_x_hn N BLOCK_SIZE (by simp [BlockState.setReg])
  · simp [BlockState.setReg, hMeanCtx.2.1]
  · simp [BlockState.setReg]
    have hSum :
        (∑ i : Fin BLOCK_SIZE,
          layernormMeanLanePrefix s0 X stride_x_N stride_x_hn
            N BLOCK_SIZE final i) =
          ∑ j : Fin N, s0.readMem X (xColOffset s0 stride_x_N stride_x_hn j.val) := by
      simpa [layernormMeanAccumulatorSpec] using
        layernormMeanAccumulatorSpec_final_sum_eq_fullN s0 X stride_x_N
          stride_x_hn N BLOCK_SIZE final hBlockPos hFinal
    simpa [layernormMeanAccumulatorSpec, layernormMeanFullNSpec] using
      congrArg
        (fun x : ℝ =>
          Tile.scalar (dtype := .real) ((x / (N : ℝ) : ℝ) : WithBot ℝ))
        hSum
  · intro offset
    change stMean.readMem X offset = s0.readMem X offset
    exact hMeanCtx.2.2 offset

theorem layernormVarLoopBody_step_accumulator_update
    (s0 st st' : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE off : Nat)
    (hAcc :
      st.regs .real [BLOCK_SIZE] "_var" =
        some (layernormVarAccumulatorSpec s0 X stride_x_N stride_x_hn
          N BLOCK_SIZE off))
    (hXPtr :
      st.regs .ptr [] "X" =
        some (Tile.scalar (Region.cast X,
          s0.pids 0 * stride_x_N + s0.pids 1 * stride_x_hn)))
    (hMean :
      st.regs .real [] "mean" =
        some (Tile.scalar (layernormMeanFullNSpec s0 X stride_x_N stride_x_hn
          N BLOCK_SIZE)))
    (hRead : ∀ offset, st.readMem X offset = s0.readMem X offset)
    (hStep :
      stepStmts (layernormVarLoopBody N BLOCK_SIZE)
        (st.setReg "off" .nat [] (Tile.scalar off)) = some st') :
    st'.regs .real [BLOCK_SIZE] "_var" =
      some
        { data := fun idx : TileIndex [BLOCK_SIZE] =>
            some
              (WithBot.unbotD 0
                  ((layernormVarAccumulatorSpec s0 X stride_x_N stride_x_hn
                    N BLOCK_SIZE off).data idx) +
                WithBot.unbotD 0
                  ((layernormVarChunkSpec s0 X stride_x_N stride_x_hn
                    N BLOCK_SIZE off).data idx)) } := by
  unfold layernormVarLoopBody at hStep
  simp [stepStmts, stepStmt, evalOp, evalOp.eq_def, hAcc, hXPtr, hMean, Tile.bop, Tile.ptrAdd,
    NumericDType.add, NumericDType.sub, NumericDType.mul, ComparableDType.lt,
    Option.bind, hRead, xColOffset] at hStep
  subst st'
  simp [BlockState.setReg]
  ext idx
  by_cases hcol : off + idx.1.val < N
  · simp [layernormVarAccumulatorSpec, layernormVarChunkSpec, hcol,
      xColOffset, sq]
  · simp [layernormVarAccumulatorSpec, layernormVarChunkSpec, hcol,
      xColOffset]
    constructor
    · intro h
      linarith
    · intro h
      linarith

theorem layernormVarLoopBody_step_preserves_context
    (s0 st st' : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE off : Nat)
    (hAcc :
      st.regs .real [BLOCK_SIZE] "_var" =
        some (layernormVarAccumulatorSpec s0 X stride_x_N stride_x_hn
          N BLOCK_SIZE off))
    (hXPtr :
      st.regs .ptr [] "X" =
        some (Tile.scalar (Region.cast X,
          s0.pids 0 * stride_x_N + s0.pids 1 * stride_x_hn)))
    (hMean :
      st.regs .real [] "mean" =
        some (Tile.scalar (layernormMeanFullNSpec s0 X stride_x_N stride_x_hn
          N BLOCK_SIZE)))
    (hStep :
      stepStmts (layernormVarLoopBody N BLOCK_SIZE)
        (st.setReg "off" .nat [] (Tile.scalar off)) = some st') :
    st'.regs .ptr [] "X" =
        some (Tile.scalar (Region.cast X,
          s0.pids 0 * stride_x_N + s0.pids 1 * stride_x_hn)) ∧
      st'.regs .real [] "mean" =
        some (Tile.scalar (layernormMeanFullNSpec s0 X stride_x_N stride_x_hn
          N BLOCK_SIZE)) ∧
      (∀ R offset, st'.readMem R offset = st.readMem R offset) := by
  unfold layernormVarLoopBody at hStep
  simp [stepStmts, stepStmt, evalOp, evalOp.eq_def, hAcc, hXPtr, hMean, Tile.bop, Tile.ptrAdd,
    NumericDType.add, NumericDType.sub, NumericDType.mul, ComparableDType.lt,
    Option.bind] at hStep
  subst st'
  refine ⟨?_, ?_, ?_⟩
  · simp [BlockState.setReg, hXPtr]
  · simp [BlockState.setReg, hMean]
  · intro R offset
    rfl

theorem layernormVarLoopInvariant_step_of_accumulator_update
    (s0 st' : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE off : Nat)
    (hOff : off % BLOCK_SIZE = 0)
    (hUpdate :
      st'.regs .real [BLOCK_SIZE] "_var" =
        some
          { data := fun idx : TileIndex [BLOCK_SIZE] =>
              some
                (WithBot.unbotD 0
                    ((layernormVarAccumulatorSpec s0 X stride_x_N stride_x_hn
                      N BLOCK_SIZE off).data idx) +
                  WithBot.unbotD 0
                    ((layernormVarChunkSpec s0 X stride_x_N stride_x_hn
                      N BLOCK_SIZE off).data idx)) }) :
    layernormVarLoopInvariant s0 X stride_x_N stride_x_hn
      N BLOCK_SIZE (off + BLOCK_SIZE) st' := by
  refine ⟨?_, ?_⟩
  · exact layernormLoopOffset_mod_step off BLOCK_SIZE hOff
  · simpa [layernormVarAccumulatorSpec_step s0 X stride_x_N stride_x_hn
      N BLOCK_SIZE off hOff] using hUpdate

theorem layernormVarLoopContextInvariant_step_of_body
    (s0 st st' : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE off : Nat)
    (hCtx : layernormVarLoopContextInvariant s0 X stride_x_N stride_x_hn
      N BLOCK_SIZE off st)
    (hStep :
      stepStmts (layernormVarLoopBody N BLOCK_SIZE)
        (st.setReg "off" .nat [] (Tile.scalar off)) = some st') :
    layernormVarLoopContextInvariant s0 X stride_x_N stride_x_hn
      N BLOCK_SIZE (off + BLOCK_SIZE) st' := by
  rcases hCtx with ⟨hInv, hXPtr, hMean, hRead⟩
  have hUpdate :=
    layernormVarLoopBody_step_accumulator_update s0 st st' X stride_x_N
      stride_x_hn N BLOCK_SIZE off hInv.2 hXPtr hMean hRead hStep
  rcases layernormVarLoopBody_step_preserves_context s0 st st' X stride_x_N
      stride_x_hn N BLOCK_SIZE off hInv.2 hXPtr hMean hStep with
    ⟨hXPtr', hMean', hReadStep⟩
  refine ⟨?_, hXPtr', hMean', ?_⟩
  · exact layernormVarLoopInvariant_step_of_accumulator_update s0 st' X
      stride_x_N stride_x_hn N BLOCK_SIZE off hInv.1 hUpdate
  · intro offset
    rw [hReadStep X offset, hRead offset]

theorem layernormVarLoopContextInvariant_body_step_exists
    (s0 st : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE off : Nat)
    (hCtx : layernormVarLoopContextInvariant s0 X stride_x_N stride_x_hn
      N BLOCK_SIZE off st) :
    ∃ st',
      stepStmts (layernormVarLoopBody N BLOCK_SIZE)
        (st.setReg "off" .nat [] (Tile.scalar off)) = some st' ∧
      layernormVarLoopContextInvariant s0 X stride_x_N stride_x_hn
        N BLOCK_SIZE (off + BLOCK_SIZE) st' := by
  rcases hCtx with ⟨hInv, hXPtr, hMean, hRead⟩
  cases hStep :
      stepStmts (layernormVarLoopBody N BLOCK_SIZE)
        (st.setReg "off" .nat [] (Tile.scalar off)) with
  | none =>
      unfold layernormVarLoopBody at hStep
      simp [stepStmts, stepStmt, evalOp, evalOp.eq_def, hInv.2, hXPtr, hMean, Tile.bop,
        Tile.ptrAdd, NumericDType.add, NumericDType.sub, NumericDType.mul,
        ComparableDType.lt, Option.bind] at hStep
  | some st' =>
      refine ⟨st', rfl, ?_⟩
      exact layernormVarLoopContextInvariant_step_of_body s0 st st' X
        stride_x_N stride_x_hn N BLOCK_SIZE off ⟨hInv, hXPtr, hMean, hRead⟩
        hStep

theorem layernormVarForRange_context_of_init
    (s0 stInit stVar : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE : Nat)
    (hStepNe : BLOCK_SIZE ≠ 0)
    (hInit :
      layernormVarLoopContextInvariant s0 X stride_x_N stride_x_hn
        N BLOCK_SIZE 0 stInit)
    (hLoop :
      stepStmt (.forRange "off" 0 N BLOCK_SIZE
        (layernormVarLoopBody N BLOCK_SIZE)) stInit = some stVar) :
    ∃ final,
      N ≤ final ∧
        layernormVarLoopContextInvariant s0 X stride_x_N stride_x_hn
          N BLOCK_SIZE final stVar := by
  obtain ⟨final, stFinal, hFor, hFinal, hCtx⟩ :=
    forRange_inv
      (idx := "off") (start := 0) (stop := N) (step := BLOCK_SIZE)
      (body := layernormVarLoopBody N BLOCK_SIZE)
      (P := layernormVarLoopContextInvariant s0 X stride_x_N stride_x_hn
        N BLOCK_SIZE)
      (s_init := stInit)
      hStepNe hInit
      (by
        intro off st _hlt hCtx
        exact layernormVarLoopContextInvariant_body_step_exists s0 st X
          stride_x_N stride_x_hn N BLOCK_SIZE off hCtx)
  have hEq : stFinal = stVar := by
    rw [hLoop] at hFor
    injection hFor with h
    exact h.symm
  subst hEq
  exact ⟨final, hFinal, hCtx⟩

theorem layernormVarLoopInvariant_reduceSum_to_fullN
    (s0 st : BlockState) (X : RegionName)
    (stride_x_N stride_x_hn N BLOCK_SIZE off : Nat)
    (hInv : layernormVarLoopContextInvariant s0 X stride_x_N stride_x_hn
      N BLOCK_SIZE off st)
    (hB : 0 < BLOCK_SIZE) (hFinal : N ≤ off) :
    ∃ acc : Tile .real [BLOCK_SIZE],
      st.regs .real [BLOCK_SIZE] "_var" = some acc ∧
        WithBot.unbotD 0
          ((Tile.reduceSum (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false
            acc).data PUnit.unit) / (N : ℝ) =
          layernormVarFullNSpec s0 X stride_x_N stride_x_hn N BLOCK_SIZE := by
  refine ⟨layernormVarAccumulatorSpec s0 X stride_x_N stride_x_hn
      N BLOCK_SIZE off, hInv.1.2, ?_⟩
  rw [layernormVarAccumulatorSpec_reduceSum]
  simp [layernormVarFullNSpec,
    layernormVarAccumulatorSpec_final_sum_eq_fullN s0 X stride_x_N
      stride_x_hn N BLOCK_SIZE off hB hFinal]

theorem yColOffset_injective
    (s : BlockState) (stride_y_N stride_y_hn N : Nat) :
    Function.Injective
      (fun i : Fin N => yColOffset s stride_y_N stride_y_hn i.val) := by
  intro a b h
  apply Fin.ext
  unfold yColOffset at h
  exact Nat.add_left_cancel h

def layernormOutLoopInvariant
    (s0 : BlockState) (X W Y : RegionName)
    (stride_x_N stride_x_hn stride_y_N stride_y_hn stride_w_hn
      N BLOCK_SIZE : Nat)
    (eps : ℝ) (off : Nat) (st : BlockState) : Prop :=
  ∀ i : Fin N,
    i.val < off →
      st.readMem Y (yColOffset s0 stride_y_N stride_y_hn i.val) =
        layernormYFullNSpec s0 X W stride_x_N stride_x_hn stride_w_hn
          N BLOCK_SIZE eps i

theorem layernormOutLoopInvariant_zero
    (s0 st : BlockState) (X W Y : RegionName)
    (stride_x_N stride_x_hn stride_y_N stride_y_hn stride_w_hn
      N BLOCK_SIZE : Nat)
    (eps : ℝ) :
    layernormOutLoopInvariant s0 X W Y stride_x_N stride_x_hn
      stride_y_N stride_y_hn stride_w_hn N BLOCK_SIZE eps 0 st := by
  intro i hlt
  omega

def layernormOutLoopContextInvariant
    (s0 : BlockState) (X W Y : RegionName)
    (stride_x_N stride_x_hn stride_y_N stride_y_hn stride_w_hn
      N BLOCK_SIZE off : Nat)
    (eps : ℝ) (st : BlockState) : Prop :=
  layernormOutLoopInvariant s0 X W Y stride_x_N stride_x_hn
      stride_y_N stride_y_hn stride_w_hn N BLOCK_SIZE eps off st ∧
    st.regs .ptr [] "X" =
      some (Tile.scalar (Region.cast X,
        s0.pids 0 * stride_x_N + s0.pids 1 * stride_x_hn)) ∧
    st.regs .ptr [] "Y" =
      some (Tile.scalar (Region.cast Y,
        s0.pids 0 * stride_y_N + s0.pids 1 * stride_y_hn)) ∧
    st.regs .ptr [] "W" =
      some (Tile.scalar (Region.cast W, s0.pids 1 * stride_w_hn)) ∧
    st.regs .real [] "mean" =
      some (Tile.scalar (layernormMeanFullNSpec s0 X stride_x_N stride_x_hn
        N BLOCK_SIZE)) ∧
    st.regs .real [] "rstd" =
      some (Tile.scalar (layernormRstdFullNSpec s0 X stride_x_N stride_x_hn
        N BLOCK_SIZE eps)) ∧
    (∀ offset, st.readMem X offset = s0.readMem X offset) ∧
    (∀ offset, st.readMem W offset = s0.readMem W offset)

theorem layernormVarPostLoop_step_to_out_init
    (s0 stVar stOutInit : BlockState) (X W Y : RegionName)
    (stride_x_N stride_x_hn stride_y_N stride_y_hn stride_w_hn
      N BLOCK_SIZE final : Nat)
    (eps : ℝ)
    (hVarCtx : layernormVarLoopContextInvariant s0 X stride_x_N stride_x_hn
      N BLOCK_SIZE final stVar)
    (hFinal : N ≤ final)
    (hBlockPos : 0 < BLOCK_SIZE)
    (hYPtr :
      stVar.regs .ptr [] "Y" =
        some (Tile.scalar (Region.cast Y,
          s0.pids 0 * stride_y_N + s0.pids 1 * stride_y_hn)))
    (hWPtr :
      stVar.regs .ptr [] "W" =
        some (Tile.scalar (Region.cast W, s0.pids 1 * stride_w_hn)))
    (hReadW : ∀ offset, stVar.readMem W offset = s0.readMem W offset)
    (hStep : stepStmts (layernormVarPostLoop N BLOCK_SIZE eps) stVar = some stOutInit) :
    layernormOutLoopContextInvariant s0 X W Y stride_x_N stride_x_hn
      stride_y_N stride_y_hn stride_w_hn N BLOCK_SIZE 0 eps stOutInit := by
  rcases layernormVarLoopInvariant_reduceSum_to_fullN s0 stVar X
      stride_x_N stride_x_hn N BLOCK_SIZE final hVarCtx hBlockPos hFinal with
    ⟨acc, hVarReg, _hReduce⟩
  have hAccEq :
      acc =
        layernormVarAccumulatorSpec s0 X stride_x_N stride_x_hn
          N BLOCK_SIZE final := by
    rw [hVarCtx.1.2] at hVarReg
    injection hVarReg with h
    exact h.symm
  subst acc
  unfold layernormVarPostLoop at hStep
  simp [stepStmts, stepStmt, evalOp, evalOp.eq_def, hVarReg, Tile.bop, Tile.uop,
    Tile.reduceSum, Tile.reduceSumDrop, TileShape.axisDim,
    TileShape.eraseAxis, TileShape.insertAxisIndex, NumericDType.add,
    NumericDType.div, Option.bind, WithBot.realSqrt] at hStep
  subst stOutInit
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · exact layernormOutLoopInvariant_zero s0 _ X W Y stride_x_N stride_x_hn
      stride_y_N stride_y_hn stride_w_hn N BLOCK_SIZE eps
  · simp [BlockState.setReg, hVarCtx.2.1]
  · simp [BlockState.setReg, hYPtr]
  · simp [BlockState.setReg, hWPtr]
  · simp [BlockState.setReg, hVarCtx.2.2.1]
  · have hSum :
        (∑ i : Fin BLOCK_SIZE,
          layernormVarLanePrefix s0 X stride_x_N stride_x_hn
            N BLOCK_SIZE final i) =
          ∑ j : Fin N,
            (s0.readMem X (xColOffset s0 stride_x_N stride_x_hn j.val) -
              layernormMeanFullNSpec s0 X stride_x_N stride_x_hn N BLOCK_SIZE)^2 := by
      simpa [layernormVarAccumulatorSpec] using
        layernormVarAccumulatorSpec_final_sum_eq_fullN s0 X stride_x_N
          stride_x_hn N BLOCK_SIZE final hBlockPos hFinal
    simpa [BlockState.setReg, layernormVarAccumulatorSpec,
      layernormRstdFullNSpec, layernormVarFullNSpec, div_eq_mul_inv] using
      congrArg
        (fun x : ℝ =>
          Tile.scalar (dtype := .real)
            (((Real.sqrt (x * (N : ℝ)⁻¹ + eps))⁻¹ : ℝ) : WithBot ℝ))
        hSum
  · intro offset
    change stVar.readMem X offset = s0.readMem X offset
    exact hVarCtx.2.2.2 offset
  · intro offset
    change stVar.readMem W offset = s0.readMem W offset
    exact hReadW offset

theorem layernormOutLoopBody_step_write_current
    (s0 st st' : BlockState) (X W Y : RegionName)
    (stride_x_N stride_x_hn stride_y_N stride_y_hn stride_w_hn
      N BLOCK_SIZE off : Nat)
    (eps : ℝ) (i : Fin BLOCK_SIZE)
    (hActive : off + i.val < N)
    (hXPtr :
      st.regs .ptr [] "X" =
        some (Tile.scalar (Region.cast X,
          s0.pids 0 * stride_x_N + s0.pids 1 * stride_x_hn)))
    (hYPtr :
      st.regs .ptr [] "Y" =
        some (Tile.scalar (Region.cast Y,
          s0.pids 0 * stride_y_N + s0.pids 1 * stride_y_hn)))
    (hWPtr :
      st.regs .ptr [] "W" =
        some (Tile.scalar (Region.cast W, s0.pids 1 * stride_w_hn)))
    (hMean :
      st.regs .real [] "mean" =
        some (Tile.scalar (layernormMeanFullNSpec s0 X stride_x_N stride_x_hn
          N BLOCK_SIZE)))
    (hRstd :
      st.regs .real [] "rstd" =
        some (Tile.scalar (layernormRstdFullNSpec s0 X stride_x_N stride_x_hn
          N BLOCK_SIZE eps)))
    (hReadX : ∀ offset, st.readMem X offset = s0.readMem X offset)
    (hReadW : ∀ offset, st.readMem W offset = s0.readMem W offset)
    (hStep :
      stepStmts (layernormOutLoopBody N BLOCK_SIZE)
        (st.setReg "off" .nat [] (Tile.scalar off)) = some st') :
    st'.readMem Y (yColOffset s0 stride_y_N stride_y_hn (off + i.val)) =
      layernormYFullNSpec s0 X W stride_x_N stride_x_hn stride_w_hn
        N BLOCK_SIZE eps ⟨off + i.val, hActive⟩ := by
  unfold layernormOutLoopBody at hStep
  simp [stepStmts, stepStmt, evalOp, evalOp.eq_def, hXPtr, hYPtr, hWPtr, hMean, hRstd,
    Tile.bop, Tile.ptrAdd, NumericDType.add, NumericDType.sub, NumericDType.mul,
    ComparableDType.lt, Option.bind, hReadX, hReadW, xColOffset, yColOffset,
    wColOffset] at hStep
  subst st'
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE] =>
        s0.pids 0 * stride_y_N + s0.pids 1 * stride_y_hn + (off + idx.1.val)) := by
    intro a b h
    have hbase :
        off + a.1.val = off + b.1.val := by
      exact Nat.add_left_cancel h
    have hval : a.1.val = b.1.val := Nat.add_left_cancel hbase
    cases a with
    | mk ah atail =>
      cases b with
      | mk bh bt =>
        cases atail
        cases bt
        simp only at hval
        cases Fin.ext hval
        rfl
  simp only [yColOffset]
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hRawInj
    ((i, PUnit.unit) : TileIndex [BLOCK_SIZE])]
  simp [hActive, layernormYFullNSpec, xColOffset, wColOffset]

theorem layernormOutLoopBody_step_preserves_regs
    (s0 st st' : BlockState) (X W Y : RegionName)
    (stride_x_N stride_x_hn stride_y_N stride_y_hn stride_w_hn
      N BLOCK_SIZE off : Nat)
    (eps : ℝ)
    (hXPtr :
      st.regs .ptr [] "X" =
        some (Tile.scalar (Region.cast X,
          s0.pids 0 * stride_x_N + s0.pids 1 * stride_x_hn)))
    (hYPtr :
      st.regs .ptr [] "Y" =
        some (Tile.scalar (Region.cast Y,
          s0.pids 0 * stride_y_N + s0.pids 1 * stride_y_hn)))
    (hWPtr :
      st.regs .ptr [] "W" =
        some (Tile.scalar (Region.cast W, s0.pids 1 * stride_w_hn)))
    (hMean :
      st.regs .real [] "mean" =
        some (Tile.scalar (layernormMeanFullNSpec s0 X stride_x_N stride_x_hn
          N BLOCK_SIZE)))
    (hRstd :
      st.regs .real [] "rstd" =
        some (Tile.scalar (layernormRstdFullNSpec s0 X stride_x_N stride_x_hn
          N BLOCK_SIZE eps)))
    (hReadX : ∀ offset, st.readMem X offset = s0.readMem X offset)
    (hReadW : ∀ offset, st.readMem W offset = s0.readMem W offset)
    (hStep :
      stepStmts (layernormOutLoopBody N BLOCK_SIZE)
        (st.setReg "off" .nat [] (Tile.scalar off)) = some st') :
    st'.regs .ptr [] "X" =
        some (Tile.scalar (Region.cast X,
          s0.pids 0 * stride_x_N + s0.pids 1 * stride_x_hn)) ∧
      st'.regs .ptr [] "Y" =
        some (Tile.scalar (Region.cast Y,
          s0.pids 0 * stride_y_N + s0.pids 1 * stride_y_hn)) ∧
      st'.regs .ptr [] "W" =
        some (Tile.scalar (Region.cast W, s0.pids 1 * stride_w_hn)) ∧
      st'.regs .real [] "mean" =
        some (Tile.scalar (layernormMeanFullNSpec s0 X stride_x_N stride_x_hn
          N BLOCK_SIZE)) ∧
      st'.regs .real [] "rstd" =
        some (Tile.scalar (layernormRstdFullNSpec s0 X stride_x_N stride_x_hn
          N BLOCK_SIZE eps)) := by
  unfold layernormOutLoopBody at hStep
  simp [stepStmts, stepStmt, evalOp, evalOp.eq_def, hXPtr, hYPtr, hWPtr, hMean, hRstd,
    Tile.bop, Tile.ptrAdd, NumericDType.add, NumericDType.sub, NumericDType.mul,
    ComparableDType.lt, Option.bind, hReadX, hReadW, xColOffset, yColOffset,
    wColOffset] at hStep
  subst st'
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · simp [hXPtr]
  · simp [hYPtr]
  · simp [hWPtr]
  · simp [hMean]
  · simp [hRstd]

theorem layernormOutLoopBody_step_preserves_reads
    (s0 st st' : BlockState) (X W Y : RegionName)
    (stride_x_N stride_x_hn stride_y_N stride_y_hn stride_w_hn
      N BLOCK_SIZE off : Nat)
    (eps : ℝ)
    (hXPtr :
      st.regs .ptr [] "X" =
        some (Tile.scalar (Region.cast X,
          s0.pids 0 * stride_x_N + s0.pids 1 * stride_x_hn)))
    (hYPtr :
      st.regs .ptr [] "Y" =
        some (Tile.scalar (Region.cast Y,
          s0.pids 0 * stride_y_N + s0.pids 1 * stride_y_hn)))
    (hWPtr :
      st.regs .ptr [] "W" =
        some (Tile.scalar (Region.cast W, s0.pids 1 * stride_w_hn)))
    (hMean :
      st.regs .real [] "mean" =
        some (Tile.scalar (layernormMeanFullNSpec s0 X stride_x_N stride_x_hn
          N BLOCK_SIZE)))
    (hRstd :
      st.regs .real [] "rstd" =
        some (Tile.scalar (layernormRstdFullNSpec s0 X stride_x_N stride_x_hn
          N BLOCK_SIZE eps)))
    (hReadX : ∀ offset, st.readMem X offset = s0.readMem X offset)
    (hReadW : ∀ offset, st.readMem W offset = s0.readMem W offset)
    (hXYNe : X ≠ Y)
    (hWYNe : W ≠ Y)
    (hStep :
      stepStmts (layernormOutLoopBody N BLOCK_SIZE)
        (st.setReg "off" .nat [] (Tile.scalar off)) = some st') :
    (∀ offset, st'.readMem X offset = s0.readMem X offset) ∧
      (∀ offset, st'.readMem W offset = s0.readMem W offset) := by
  unfold layernormOutLoopBody at hStep
  simp [stepStmts, stepStmt, evalOp, evalOp.eq_def, hXPtr, hYPtr, hWPtr, hMean, hRstd,
    Tile.bop, Tile.ptrAdd, NumericDType.add, NumericDType.sub, NumericDType.mul,
    ComparableDType.lt, Option.bind, hReadX, hReadW, xColOffset, yColOffset,
    wColOffset] at hStep
  subst st'
  constructor
  · intro offset
    rw [BlockState.foldl_writeMem_const_region_prop_masked_readMem_other
      (region := Y)
      (P := fun lane : TileIndex [BLOCK_SIZE] => off + lane.1.val < N)
      (R := X) (off := offset) (hRR := hXYNe)]
    exact hReadX offset
  · intro offset
    rw [BlockState.foldl_writeMem_const_region_prop_masked_readMem_other
      (region := Y)
      (P := fun lane : TileIndex [BLOCK_SIZE] => off + lane.1.val < N)
      (R := W) (off := offset) (hRR := hWYNe)]
    exact hReadW offset

theorem layernormOutLoopBody_step_preserves_old_output
    (s0 st st' : BlockState) (X W Y : RegionName)
    (stride_x_N stride_x_hn stride_y_N stride_y_hn stride_w_hn
      N BLOCK_SIZE off : Nat)
    (eps : ℝ) (col : Fin N)
    (hOld : col.val < off)
    (hXPtr :
      st.regs .ptr [] "X" =
        some (Tile.scalar (Region.cast X,
          s0.pids 0 * stride_x_N + s0.pids 1 * stride_x_hn)))
    (hYPtr :
      st.regs .ptr [] "Y" =
        some (Tile.scalar (Region.cast Y,
          s0.pids 0 * stride_y_N + s0.pids 1 * stride_y_hn)))
    (hWPtr :
      st.regs .ptr [] "W" =
        some (Tile.scalar (Region.cast W, s0.pids 1 * stride_w_hn)))
    (hMean :
      st.regs .real [] "mean" =
        some (Tile.scalar (layernormMeanFullNSpec s0 X stride_x_N stride_x_hn
          N BLOCK_SIZE)))
    (hRstd :
      st.regs .real [] "rstd" =
        some (Tile.scalar (layernormRstdFullNSpec s0 X stride_x_N stride_x_hn
          N BLOCK_SIZE eps)))
    (hReadX : ∀ offset, st.readMem X offset = s0.readMem X offset)
    (hReadW : ∀ offset, st.readMem W offset = s0.readMem W offset)
    (hStep :
      stepStmts (layernormOutLoopBody N BLOCK_SIZE)
        (st.setReg "off" .nat [] (Tile.scalar off)) = some st') :
    st'.readMem Y (yColOffset s0 stride_y_N stride_y_hn col.val) =
      st.readMem Y (yColOffset s0 stride_y_N stride_y_hn col.val) := by
  unfold layernormOutLoopBody at hStep
  simp [stepStmts, stepStmt, evalOp, evalOp.eq_def, hXPtr, hYPtr, hWPtr, hMean, hRstd,
    Tile.bop, Tile.ptrAdd, NumericDType.add, NumericDType.sub, NumericDType.mul,
    ComparableDType.lt, Option.bind, hReadX, hReadW, xColOffset, yColOffset,
    wColOffset] at hStep
  subst st'
  simp only [yColOffset]
  rw [BlockState.scatter_prop_masked_preserves_other_offset
    (region := Y)
    (P := fun lane : TileIndex [BLOCK_SIZE] => off + lane.1.val < N)
    (off := s0.pids 0 * stride_y_N + s0.pids 1 * stride_y_hn + col.val)]
  · rfl
  · intro lane hActive hEq
    have hFinEq :
        (⟨off + lane.1.val, hActive⟩ : Fin N) = col := by
      apply yColOffset_injective s0 stride_y_N stride_y_hn N
      simpa [yColOffset] using hEq
    have hVal : off + lane.1.val = col.val := congrArg Fin.val hFinEq
    omega

theorem layernormOutLoopBody_step_output_invariant
    (s0 st st' : BlockState) (X W Y : RegionName)
    (stride_x_N stride_x_hn stride_y_N stride_y_hn stride_w_hn
      N BLOCK_SIZE off : Nat)
    (eps : ℝ)
    (hInv : layernormOutLoopInvariant s0 X W Y stride_x_N stride_x_hn
      stride_y_N stride_y_hn stride_w_hn N BLOCK_SIZE eps off st)
    (hXPtr :
      st.regs .ptr [] "X" =
        some (Tile.scalar (Region.cast X,
          s0.pids 0 * stride_x_N + s0.pids 1 * stride_x_hn)))
    (hYPtr :
      st.regs .ptr [] "Y" =
        some (Tile.scalar (Region.cast Y,
          s0.pids 0 * stride_y_N + s0.pids 1 * stride_y_hn)))
    (hWPtr :
      st.regs .ptr [] "W" =
        some (Tile.scalar (Region.cast W, s0.pids 1 * stride_w_hn)))
    (hMean :
      st.regs .real [] "mean" =
        some (Tile.scalar (layernormMeanFullNSpec s0 X stride_x_N stride_x_hn
          N BLOCK_SIZE)))
    (hRstd :
      st.regs .real [] "rstd" =
        some (Tile.scalar (layernormRstdFullNSpec s0 X stride_x_N stride_x_hn
          N BLOCK_SIZE eps)))
    (hReadX : ∀ offset, st.readMem X offset = s0.readMem X offset)
    (hReadW : ∀ offset, st.readMem W offset = s0.readMem W offset)
    (hStep :
      stepStmts (layernormOutLoopBody N BLOCK_SIZE)
        (st.setReg "off" .nat [] (Tile.scalar off)) = some st') :
    layernormOutLoopInvariant s0 X W Y stride_x_N stride_x_hn
      stride_y_N stride_y_hn stride_w_hn N BLOCK_SIZE eps
      (off + BLOCK_SIZE) st' := by
  intro col hWritten
  by_cases hOld : col.val < off
  · rw [layernormOutLoopBody_step_preserves_old_output s0 st st' X W Y
      stride_x_N stride_x_hn stride_y_N stride_y_hn stride_w_hn
      N BLOCK_SIZE off eps col hOld hXPtr hYPtr hWPtr hMean hRstd hReadX hReadW
      hStep]
    exact hInv col hOld
  · have hOffLe : off ≤ col.val := Nat.le_of_not_gt hOld
    let lane : Fin BLOCK_SIZE := ⟨col.val - off, by omega⟩
    have hLaneActive : off + lane.val < N := by
      have hcol : off + lane.val = col.val := by
        simp [lane]
        omega
      simp [hcol, col.isLt]
    have hcolEq : off + lane.val = col.val := by
      simp [lane]
      omega
    have hRead := layernormOutLoopBody_step_write_current s0 st st' X W Y
      stride_x_N stride_x_hn stride_y_N stride_y_hn stride_w_hn
      N BLOCK_SIZE off eps lane hLaneActive hXPtr hYPtr hWPtr hMean hRstd
      hReadX hReadW hStep
    simpa [hcolEq] using hRead

theorem layernormOutLoopContextInvariant_step_of_body
    (s0 st st' : BlockState) (X W Y : RegionName)
    (stride_x_N stride_x_hn stride_y_N stride_y_hn stride_w_hn
      N BLOCK_SIZE off : Nat)
    (eps : ℝ)
    (hCtx : layernormOutLoopContextInvariant s0 X W Y stride_x_N stride_x_hn
      stride_y_N stride_y_hn stride_w_hn N BLOCK_SIZE off eps st)
    (hXYNe : X ≠ Y)
    (hWYNe : W ≠ Y)
    (hStep :
      stepStmts (layernormOutLoopBody N BLOCK_SIZE)
        (st.setReg "off" .nat [] (Tile.scalar off)) = some st') :
    layernormOutLoopContextInvariant s0 X W Y stride_x_N stride_x_hn
      stride_y_N stride_y_hn stride_w_hn N BLOCK_SIZE (off + BLOCK_SIZE) eps st' := by
  rcases hCtx with
    ⟨hOutInv, hXPtr, hYPtr, hWPtr, hMean, hRstd, hReadX, hReadW⟩
  have hOutInv' :=
    layernormOutLoopBody_step_output_invariant s0 st st' X W Y
      stride_x_N stride_x_hn stride_y_N stride_y_hn stride_w_hn
      N BLOCK_SIZE off eps hOutInv hXPtr hYPtr hWPtr hMean hRstd hReadX hReadW
      hStep
  rcases layernormOutLoopBody_step_preserves_regs s0 st st' X W Y
      stride_x_N stride_x_hn stride_y_N stride_y_hn stride_w_hn
      N BLOCK_SIZE off eps hXPtr hYPtr hWPtr hMean hRstd hReadX hReadW hStep with
    ⟨hXPtr', hYPtr', hWPtr', hMean', hRstd'⟩
  rcases layernormOutLoopBody_step_preserves_reads s0 st st' X W Y
      stride_x_N stride_x_hn stride_y_N stride_y_hn stride_w_hn
      N BLOCK_SIZE off eps hXPtr hYPtr hWPtr hMean hRstd hReadX hReadW
      hXYNe hWYNe hStep with
    ⟨hReadX', hReadW'⟩
  exact ⟨hOutInv', hXPtr', hYPtr', hWPtr', hMean', hRstd', hReadX', hReadW'⟩

theorem layernormOutLoopContextInvariant_body_step_exists
    (s0 st : BlockState) (X W Y : RegionName)
    (stride_x_N stride_x_hn stride_y_N stride_y_hn stride_w_hn
      N BLOCK_SIZE off : Nat)
    (eps : ℝ)
    (hCtx : layernormOutLoopContextInvariant s0 X W Y stride_x_N stride_x_hn
      stride_y_N stride_y_hn stride_w_hn N BLOCK_SIZE off eps st)
    (hXYNe : X ≠ Y)
    (hWYNe : W ≠ Y) :
    ∃ st',
      stepStmts (layernormOutLoopBody N BLOCK_SIZE)
        (st.setReg "off" .nat [] (Tile.scalar off)) = some st' ∧
      layernormOutLoopContextInvariant s0 X W Y stride_x_N stride_x_hn
        stride_y_N stride_y_hn stride_w_hn N BLOCK_SIZE (off + BLOCK_SIZE)
        eps st' := by
  rcases hCtx with
    ⟨hOutInv, hXPtr, hYPtr, hWPtr, hMean, hRstd, hReadX, hReadW⟩
  cases hStep :
      stepStmts (layernormOutLoopBody N BLOCK_SIZE)
        (st.setReg "off" .nat [] (Tile.scalar off)) with
  | none =>
      unfold layernormOutLoopBody at hStep
      simp [stepStmts, stepStmt, evalOp, evalOp.eq_def, hXPtr, hYPtr, hWPtr, hMean, hRstd,
        Tile.bop, Tile.ptrAdd, NumericDType.add, NumericDType.sub,
        NumericDType.mul, ComparableDType.lt, Option.bind, hReadX, hReadW,
        xColOffset, yColOffset, wColOffset] at hStep
  | some st' =>
      refine ⟨st', rfl, ?_⟩
      exact layernormOutLoopContextInvariant_step_of_body s0 st st' X W Y
        stride_x_N stride_x_hn stride_y_N stride_y_hn stride_w_hn
        N BLOCK_SIZE off eps
        ⟨hOutInv, hXPtr, hYPtr, hWPtr, hMean, hRstd, hReadX, hReadW⟩
        hXYNe hWYNe hStep

theorem layernormOutForRange_context
    (s0 stInit stOut : BlockState) (X W Y : RegionName)
    (stride_x_N stride_x_hn stride_y_N stride_y_hn stride_w_hn
      N BLOCK_SIZE : Nat)
    (eps : ℝ)
    (hStepNe : BLOCK_SIZE ≠ 0)
    (hInit : layernormOutLoopContextInvariant s0 X W Y stride_x_N stride_x_hn
      stride_y_N stride_y_hn stride_w_hn N BLOCK_SIZE 0 eps stInit)
    (hXYNe : X ≠ Y)
    (hWYNe : W ≠ Y)
    (hLoop :
      stepStmt (.forRange "off" 0 N BLOCK_SIZE
        (layernormOutLoopBody N BLOCK_SIZE)) stInit = some stOut) :
    ∃ final,
      N ≤ final ∧
        layernormOutLoopContextInvariant s0 X W Y stride_x_N stride_x_hn
          stride_y_N stride_y_hn stride_w_hn N BLOCK_SIZE final eps stOut := by
  obtain ⟨final, stFinal, hFor, hFinal, hCtx⟩ :=
    forRange_inv
      (idx := "off") (start := 0) (stop := N) (step := BLOCK_SIZE)
      (body := layernormOutLoopBody N BLOCK_SIZE)
      (P := fun off st =>
        layernormOutLoopContextInvariant s0 X W Y stride_x_N stride_x_hn
          stride_y_N stride_y_hn stride_w_hn N BLOCK_SIZE off eps st)
      (s_init := stInit)
      hStepNe hInit
      (by
        intro off st _hlt hCtx
        exact layernormOutLoopContextInvariant_body_step_exists s0 st X W Y
          stride_x_N stride_x_hn stride_y_N stride_y_hn stride_w_hn
          N BLOCK_SIZE off eps hCtx hXYNe hWYNe)
  have hEq : stFinal = stOut := by
    rw [hLoop] at hFor
    injection hFor with h
    exact h.symm
  subst hEq
  exact ⟨final, hFinal, hCtx⟩

theorem layernormOutForRange_fullN_of_init
    (s0 stInit stOut : BlockState) (X W Y : RegionName)
    (stride_x_N stride_x_hn stride_y_N stride_y_hn stride_w_hn
      N BLOCK_SIZE : Nat)
    (eps : ℝ)
    (hStepNe : BLOCK_SIZE ≠ 0)
    (hInit : layernormOutLoopContextInvariant s0 X W Y stride_x_N stride_x_hn
      stride_y_N stride_y_hn stride_w_hn N BLOCK_SIZE 0 eps stInit)
    (hXYNe : X ≠ Y)
    (hWYNe : W ≠ Y)
    (hLoop :
      stepStmt (.forRange "off" 0 N BLOCK_SIZE
        (layernormOutLoopBody N BLOCK_SIZE)) stInit = some stOut) :
    ∀ i : Fin N,
      stOut.readMem Y (yColOffset s0 stride_y_N stride_y_hn i.val) =
        layernormYFullNSpec s0 X W stride_x_N stride_x_hn stride_w_hn
          N BLOCK_SIZE eps i := by
  obtain ⟨final, hFinal, hCtx⟩ :=
    layernormOutForRange_context s0 stInit stOut X W Y stride_x_N stride_x_hn
      stride_y_N stride_y_hn stride_w_hn N BLOCK_SIZE eps hStepNe hInit
      hXYNe hWYNe hLoop
  intro i
  exact hCtx.1 i (Nat.lt_of_lt_of_le i.isLt hFinal)

theorem layernorm_fwd_triton_staged_fullN_correct
    (s0 stPre stMean stVarInit stVar stOutInit stOut : BlockState)
    (X W Y : RegionName)
    (stride_x_N stride_x_hn stride_y_N stride_y_hn stride_w_hn
      N BLOCK_SIZE : Nat)
    (eps : ℝ)
    (hBlockPos : 0 < BLOCK_SIZE)
    (hXYNe : X ≠ Y)
    (hWYNe : W ≠ Y)
    (hYPtrVar :
      stVar.regs .ptr [] "Y" =
        some (Tile.scalar (Region.cast Y,
          s0.pids 0 * stride_y_N + s0.pids 1 * stride_y_hn)))
    (hWPtrVar :
      stVar.regs .ptr [] "W" =
        some (Tile.scalar (Region.cast W, s0.pids 1 * stride_w_hn)))
    (hReadWVar : ∀ offset, stVar.readMem W offset = s0.readMem W offset)
    (hPre :
      stepStmts
        (layernormMeanPreLoop X W Y stride_x_N stride_x_hn stride_y_N
          stride_y_hn stride_w_hn BLOCK_SIZE) s0 = some stPre)
    (hMeanLoop :
      stepStmt (.forRange "off" 0 N BLOCK_SIZE
        (layernormMeanLoopBody N BLOCK_SIZE)) stPre = some stMean)
    (hMeanPost :
      stepStmts (layernormMeanPostLoop N BLOCK_SIZE) stMean = some stVarInit)
    (hVarLoop :
      stepStmt (.forRange "off" 0 N BLOCK_SIZE
        (layernormVarLoopBody N BLOCK_SIZE)) stVarInit = some stVar)
    (hVarPost :
      stepStmts (layernormVarPostLoop N BLOCK_SIZE eps) stVar = some stOutInit)
    (hOutLoop :
      stepStmt (.forRange "off" 0 N BLOCK_SIZE
        (layernormOutLoopBody N BLOCK_SIZE)) stOutInit = some stOut) :
    ∀ i : Fin N,
      stOut.readMem Y (yColOffset s0 stride_y_N stride_y_hn i.val) =
        layernormYFullNSpec s0 X W stride_x_N stride_x_hn stride_w_hn
          N BLOCK_SIZE eps i := by
  have hStepNe : BLOCK_SIZE ≠ 0 := Nat.ne_of_gt hBlockPos
  obtain ⟨meanFinal, hMeanFinal, hMeanCtx⟩ :=
    layernormMeanForRange_context_of_preloop s0 stPre stMean X W Y
      stride_x_N stride_x_hn stride_y_N stride_y_hn stride_w_hn N BLOCK_SIZE
      hStepNe hPre hMeanLoop
  have hVarInit :
      layernormVarLoopContextInvariant s0 X stride_x_N stride_x_hn
        N BLOCK_SIZE 0 stVarInit :=
    layernormMeanPostLoop_step_to_var_init s0 stMean stVarInit X
      stride_x_N stride_x_hn N BLOCK_SIZE meanFinal hMeanCtx hMeanFinal
      hBlockPos hMeanPost
  obtain ⟨varFinal, hVarFinal, hVarCtx⟩ :=
    layernormVarForRange_context_of_init s0 stVarInit stVar X
      stride_x_N stride_x_hn N BLOCK_SIZE hStepNe hVarInit hVarLoop
  have hOutInit :
      layernormOutLoopContextInvariant s0 X W Y stride_x_N stride_x_hn
        stride_y_N stride_y_hn stride_w_hn N BLOCK_SIZE 0 eps stOutInit :=
    layernormVarPostLoop_step_to_out_init s0 stVar stOutInit X W Y
      stride_x_N stride_x_hn stride_y_N stride_y_hn stride_w_hn
      N BLOCK_SIZE varFinal eps hVarCtx hVarFinal hBlockPos hYPtrVar hWPtrVar
      hReadWVar hVarPost
  exact layernormOutForRange_fullN_of_init s0 stOutInit stOut X W Y
    stride_x_N stride_x_hn stride_y_N stride_y_hn stride_w_hn N BLOCK_SIZE eps
    hStepNe hOutInit hXYNe hWYNe hOutLoop

theorem layernormMeanLoopBody_step_preserves_ptrs_read
    (s0 st st' : BlockState) (X W Y R : RegionName)
    (stride_x_N stride_x_hn stride_y_N stride_y_hn stride_w_hn
      N BLOCK_SIZE off : Nat)
    (hCtx : layernormMeanLoopContextInvariant s0 X stride_x_N stride_x_hn
      N BLOCK_SIZE off st)
    (hYPtr :
      st.regs .ptr [] "Y" =
        some (Tile.scalar (Region.cast Y,
          s0.pids 0 * stride_y_N + s0.pids 1 * stride_y_hn)))
    (hWPtr :
      st.regs .ptr [] "W" =
        some (Tile.scalar (Region.cast W, s0.pids 1 * stride_w_hn)))
    (hStep :
      stepStmts (layernormMeanLoopBody N BLOCK_SIZE)
        (st.setReg "off" .nat [] (Tile.scalar off)) = some st') :
    st'.regs .ptr [] "Y" =
        some (Tile.scalar (Region.cast Y,
          s0.pids 0 * stride_y_N + s0.pids 1 * stride_y_hn)) ∧
      st'.regs .ptr [] "W" =
        some (Tile.scalar (Region.cast W, s0.pids 1 * stride_w_hn)) ∧
      (∀ offset, st'.readMem R offset = st.readMem R offset) := by
  rcases hCtx with ⟨hInv, hXPtr, _hReadX⟩
  unfold layernormMeanLoopBody at hStep
  simp [stepStmts, stepStmt, evalOp, evalOp.eq_def, hInv.2, hXPtr, Tile.bop, Tile.ptrAdd,
    NumericDType.add, ComparableDType.lt, Option.bind] at hStep
  subst st'
  refine ⟨?_, ?_, ?_⟩
  · simp [BlockState.setReg, hYPtr]
  · simp [BlockState.setReg, hWPtr]
  · intro offset
    rfl

theorem layernormMeanForRange_context_ptrs_read_of_preloop
    (s0 stPre stMean : BlockState) (X W Y R : RegionName)
    (stride_x_N stride_x_hn stride_y_N stride_y_hn stride_w_hn N BLOCK_SIZE : Nat)
    (hStepNe : BLOCK_SIZE ≠ 0)
    (hPre :
      stepStmts
        (layernormMeanPreLoop X W Y stride_x_N stride_x_hn stride_y_N
          stride_y_hn stride_w_hn BLOCK_SIZE) s0 = some stPre)
    (hLoop :
      stepStmt (.forRange "off" 0 N BLOCK_SIZE
        (layernormMeanLoopBody N BLOCK_SIZE)) stPre = some stMean) :
    ∃ final,
      N ≤ final ∧
        layernormMeanLoopContextInvariant s0 X stride_x_N stride_x_hn
          N BLOCK_SIZE final stMean ∧
        stMean.regs .ptr [] "Y" =
          some (Tile.scalar (Region.cast Y,
            s0.pids 0 * stride_y_N + s0.pids 1 * stride_y_hn)) ∧
        stMean.regs .ptr [] "W" =
          some (Tile.scalar (Region.cast W, s0.pids 1 * stride_w_hn)) ∧
        (∀ offset, stMean.readMem R offset = s0.readMem R offset) := by
  have hInitMean :
      layernormMeanLoopContextInvariant s0 X stride_x_N stride_x_hn
        N BLOCK_SIZE 0 stPre :=
    layernormMeanLoopContextInvariant_init_of_preloop s0 stPre X W Y
      stride_x_N stride_x_hn stride_y_N stride_y_hn stride_w_hn N
      BLOCK_SIZE hPre
  rcases layernormMeanPreLoop_step_regs s0 stPre X W Y stride_x_N stride_x_hn
      stride_y_N stride_y_hn stride_w_hn BLOCK_SIZE hPre with
    ⟨_hZero, _hXPtr, hYPtrInit, hWPtrInit, _hSeq, _hH, hReadInit⟩
  obtain ⟨final, stFinal, hFor, hFinal, hCtx⟩ :=
    forRange_inv
      (idx := "off") (start := 0) (stop := N) (step := BLOCK_SIZE)
      (body := layernormMeanLoopBody N BLOCK_SIZE)
      (P := fun off st =>
        layernormMeanLoopContextInvariant s0 X stride_x_N stride_x_hn
          N BLOCK_SIZE off st ∧
        st.regs .ptr [] "Y" =
          some (Tile.scalar (Region.cast Y,
            s0.pids 0 * stride_y_N + s0.pids 1 * stride_y_hn)) ∧
        st.regs .ptr [] "W" =
          some (Tile.scalar (Region.cast W, s0.pids 1 * stride_w_hn)) ∧
        (∀ offset, st.readMem R offset = s0.readMem R offset))
      (s_init := stPre)
      hStepNe ⟨hInitMean, hYPtrInit, hWPtrInit, hReadInit R⟩
      (by
        intro off st _hlt hCtx
        rcases hCtx with ⟨hMeanCtx, hYPtr, hWPtr, hRead⟩
        obtain ⟨st', hStep, hMeanCtx'⟩ :=
          layernormMeanLoopContextInvariant_body_step_exists s0 st X
            stride_x_N stride_x_hn N BLOCK_SIZE off hMeanCtx
        rcases layernormMeanLoopBody_step_preserves_ptrs_read s0 st st' X W Y R
            stride_x_N stride_x_hn stride_y_N stride_y_hn stride_w_hn
            N BLOCK_SIZE off hMeanCtx hYPtr hWPtr hStep with
          ⟨hYPtr', hWPtr', hReadStep⟩
        refine ⟨st', hStep, hMeanCtx', hYPtr', hWPtr', ?_⟩
        intro offset
        rw [hReadStep offset, hRead offset])
  have hEq : stFinal = stMean := by
    rw [hLoop] at hFor
    injection hFor with h
    exact h.symm
  subst hEq
  exact ⟨final, hFinal, hCtx.1, hCtx.2.1, hCtx.2.2.1, hCtx.2.2.2⟩

theorem layernormMeanPostLoop_step_preserves_ptrs_read
    (stMean stVarInit : BlockState) (W Y R : RegionName)
    (s0 : BlockState)
    (stride_y_N stride_y_hn stride_w_hn N BLOCK_SIZE : Nat)
    (acc : Tile .real [BLOCK_SIZE])
    (hMeanReg : stMean.regs .real [BLOCK_SIZE] "_mean" = some acc)
    (hYPtr :
      stMean.regs .ptr [] "Y" =
        some (Tile.scalar (Region.cast Y,
          s0.pids 0 * stride_y_N + s0.pids 1 * stride_y_hn)))
    (hWPtr :
      stMean.regs .ptr [] "W" =
        some (Tile.scalar (Region.cast W, s0.pids 1 * stride_w_hn)))
    (hStep :
      stepStmts (layernormMeanPostLoop N BLOCK_SIZE) stMean = some stVarInit) :
    stVarInit.regs .ptr [] "Y" =
        some (Tile.scalar (Region.cast Y,
          s0.pids 0 * stride_y_N + s0.pids 1 * stride_y_hn)) ∧
      stVarInit.regs .ptr [] "W" =
        some (Tile.scalar (Region.cast W, s0.pids 1 * stride_w_hn)) ∧
      (∀ offset, stVarInit.readMem R offset = stMean.readMem R offset) := by
  unfold layernormMeanPostLoop at hStep
  simp [stepStmts, stepStmt, evalOp, evalOp.eq_def, hMeanReg, Tile.bop, Tile.reduceSum,
    Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis,
    TileShape.insertAxisIndex, NumericDType.div, Option.bind] at hStep
  subst stVarInit
  refine ⟨?_, ?_, ?_⟩
  · simp [BlockState.setReg, hYPtr]
  · simp [BlockState.setReg, hWPtr]
  · intro offset
    rfl

theorem layernormVarLoopBody_step_preserves_ptrs_read
    (s0 st st' : BlockState) (X W Y R : RegionName)
    (stride_x_N stride_x_hn stride_y_N stride_y_hn stride_w_hn
      N BLOCK_SIZE off : Nat)
    (hCtx : layernormVarLoopContextInvariant s0 X stride_x_N stride_x_hn
      N BLOCK_SIZE off st)
    (hYPtr :
      st.regs .ptr [] "Y" =
        some (Tile.scalar (Region.cast Y,
          s0.pids 0 * stride_y_N + s0.pids 1 * stride_y_hn)))
    (hWPtr :
      st.regs .ptr [] "W" =
        some (Tile.scalar (Region.cast W, s0.pids 1 * stride_w_hn)))
    (hStep :
      stepStmts (layernormVarLoopBody N BLOCK_SIZE)
        (st.setReg "off" .nat [] (Tile.scalar off)) = some st') :
    st'.regs .ptr [] "Y" =
        some (Tile.scalar (Region.cast Y,
          s0.pids 0 * stride_y_N + s0.pids 1 * stride_y_hn)) ∧
      st'.regs .ptr [] "W" =
        some (Tile.scalar (Region.cast W, s0.pids 1 * stride_w_hn)) ∧
      (∀ offset, st'.readMem R offset = st.readMem R offset) := by
  rcases hCtx with ⟨hInv, hXPtr, hMean, _hReadX⟩
  unfold layernormVarLoopBody at hStep
  simp [stepStmts, stepStmt, evalOp, evalOp.eq_def, hInv.2, hXPtr, hMean, Tile.bop,
    Tile.ptrAdd, NumericDType.add, NumericDType.sub, NumericDType.mul,
    ComparableDType.lt, Option.bind] at hStep
  subst st'
  refine ⟨?_, ?_, ?_⟩
  · simp [BlockState.setReg, hYPtr]
  · simp [BlockState.setReg, hWPtr]
  · intro offset
    rfl

theorem layernormVarForRange_context_ptrs_read_of_init
    (s0 stInit stVar : BlockState) (X W Y R : RegionName)
    (stride_x_N stride_x_hn stride_y_N stride_y_hn stride_w_hn N BLOCK_SIZE : Nat)
    (hStepNe : BLOCK_SIZE ≠ 0)
    (hInit :
      layernormVarLoopContextInvariant s0 X stride_x_N stride_x_hn
        N BLOCK_SIZE 0 stInit)
    (hYPtrInit :
      stInit.regs .ptr [] "Y" =
        some (Tile.scalar (Region.cast Y,
          s0.pids 0 * stride_y_N + s0.pids 1 * stride_y_hn)))
    (hWPtrInit :
      stInit.regs .ptr [] "W" =
        some (Tile.scalar (Region.cast W, s0.pids 1 * stride_w_hn)))
    (hReadInit : ∀ offset, stInit.readMem R offset = s0.readMem R offset)
    (hLoop :
      stepStmt (.forRange "off" 0 N BLOCK_SIZE
        (layernormVarLoopBody N BLOCK_SIZE)) stInit = some stVar) :
    ∃ final,
      N ≤ final ∧
        layernormVarLoopContextInvariant s0 X stride_x_N stride_x_hn
          N BLOCK_SIZE final stVar ∧
        stVar.regs .ptr [] "Y" =
          some (Tile.scalar (Region.cast Y,
            s0.pids 0 * stride_y_N + s0.pids 1 * stride_y_hn)) ∧
        stVar.regs .ptr [] "W" =
          some (Tile.scalar (Region.cast W, s0.pids 1 * stride_w_hn)) ∧
        (∀ offset, stVar.readMem R offset = s0.readMem R offset) := by
  obtain ⟨final, stFinal, hFor, hFinal, hCtx⟩ :=
    forRange_inv
      (idx := "off") (start := 0) (stop := N) (step := BLOCK_SIZE)
      (body := layernormVarLoopBody N BLOCK_SIZE)
      (P := fun off st =>
        layernormVarLoopContextInvariant s0 X stride_x_N stride_x_hn
          N BLOCK_SIZE off st ∧
        st.regs .ptr [] "Y" =
          some (Tile.scalar (Region.cast Y,
            s0.pids 0 * stride_y_N + s0.pids 1 * stride_y_hn)) ∧
        st.regs .ptr [] "W" =
          some (Tile.scalar (Region.cast W, s0.pids 1 * stride_w_hn)) ∧
        (∀ offset, st.readMem R offset = s0.readMem R offset))
      (s_init := stInit)
      hStepNe ⟨hInit, hYPtrInit, hWPtrInit, hReadInit⟩
      (by
        intro off st _hlt hCtx
        rcases hCtx with ⟨hVarCtx, hYPtr, hWPtr, hRead⟩
        obtain ⟨st', hStep, hVarCtx'⟩ :=
          layernormVarLoopContextInvariant_body_step_exists s0 st X
            stride_x_N stride_x_hn N BLOCK_SIZE off hVarCtx
        rcases layernormVarLoopBody_step_preserves_ptrs_read s0 st st' X W Y R
            stride_x_N stride_x_hn stride_y_N stride_y_hn stride_w_hn
            N BLOCK_SIZE off hVarCtx hYPtr hWPtr hStep with
          ⟨hYPtr', hWPtr', hReadStep⟩
        refine ⟨st', hStep, hVarCtx', hYPtr', hWPtr', ?_⟩
        intro offset
        rw [hReadStep offset, hRead offset])
  have hEq : stFinal = stVar := by
    rw [hLoop] at hFor
    injection hFor with h
    exact h.symm
  subst hEq
  exact ⟨final, hFinal, hCtx.1, hCtx.2.1, hCtx.2.2.1, hCtx.2.2.2⟩

theorem layernorm_fwd_triton_staged_fullN_correct_from_preloop
    (s0 stPre stMean stVarInit stVar stOutInit stOut : BlockState)
    (X W Y : RegionName)
    (stride_x_N stride_x_hn stride_y_N stride_y_hn stride_w_hn
      N BLOCK_SIZE : Nat)
    (eps : ℝ)
    (hBlockPos : 0 < BLOCK_SIZE)
    (hXYNe : X ≠ Y)
    (hWYNe : W ≠ Y)
    (hPre :
      stepStmts
        (layernormMeanPreLoop X W Y stride_x_N stride_x_hn stride_y_N
          stride_y_hn stride_w_hn BLOCK_SIZE) s0 = some stPre)
    (hMeanLoop :
      stepStmt (.forRange "off" 0 N BLOCK_SIZE
        (layernormMeanLoopBody N BLOCK_SIZE)) stPre = some stMean)
    (hMeanPost :
      stepStmts (layernormMeanPostLoop N BLOCK_SIZE) stMean = some stVarInit)
    (hVarLoop :
      stepStmt (.forRange "off" 0 N BLOCK_SIZE
        (layernormVarLoopBody N BLOCK_SIZE)) stVarInit = some stVar)
    (hVarPost :
      stepStmts (layernormVarPostLoop N BLOCK_SIZE eps) stVar = some stOutInit)
    (hOutLoop :
      stepStmt (.forRange "off" 0 N BLOCK_SIZE
        (layernormOutLoopBody N BLOCK_SIZE)) stOutInit = some stOut) :
    ∀ i : Fin N,
      stOut.readMem Y (yColOffset s0 stride_y_N stride_y_hn i.val) =
        layernormYFullNSpec s0 X W stride_x_N stride_x_hn stride_w_hn
          N BLOCK_SIZE eps i := by
  have hStepNe : BLOCK_SIZE ≠ 0 := Nat.ne_of_gt hBlockPos
  obtain ⟨meanFinal, _hMeanFinal, hMeanCtx, hYPtrMean, hWPtrMean,
      hReadWMean⟩ :=
    layernormMeanForRange_context_ptrs_read_of_preloop s0 stPre stMean
      X W Y W stride_x_N stride_x_hn stride_y_N stride_y_hn stride_w_hn
      N BLOCK_SIZE hStepNe hPre hMeanLoop
  have hVarInit :
      layernormVarLoopContextInvariant s0 X stride_x_N stride_x_hn
        N BLOCK_SIZE 0 stVarInit :=
    layernormMeanPostLoop_step_to_var_init s0 stMean stVarInit X
      stride_x_N stride_x_hn N BLOCK_SIZE meanFinal hMeanCtx _hMeanFinal
      hBlockPos hMeanPost
  rcases layernormMeanPostLoop_step_preserves_ptrs_read stMean stVarInit W Y W
      s0 stride_y_N stride_y_hn stride_w_hn N BLOCK_SIZE
      (layernormMeanAccumulatorSpec s0 X stride_x_N stride_x_hn
        N BLOCK_SIZE meanFinal) hMeanCtx.1.2 hYPtrMean hWPtrMean hMeanPost with
    ⟨hYPtrVarInit, hWPtrVarInit, hReadWVarInitStep⟩
  have hReadWVarInit : ∀ offset, stVarInit.readMem W offset = s0.readMem W offset := by
    intro offset
    rw [hReadWVarInitStep offset, hReadWMean offset]
  obtain ⟨_varFinal, _hVarFinal, _hVarCtx, hYPtrVar, hWPtrVar, hReadWVar⟩ :=
    layernormVarForRange_context_ptrs_read_of_init s0 stVarInit stVar X W Y W
      stride_x_N stride_x_hn stride_y_N stride_y_hn stride_w_hn N BLOCK_SIZE
      hStepNe hVarInit hYPtrVarInit hWPtrVarInit hReadWVarInit hVarLoop
  exact layernorm_fwd_triton_staged_fullN_correct s0 stPre stMean stVarInit
    stVar stOutInit stOut X W Y stride_x_N stride_x_hn stride_y_N stride_y_hn
    stride_w_hn N BLOCK_SIZE eps hBlockPos hXYNe hWYNe hYPtrVar hWPtrVar
    hReadWVar hPre hMeanLoop hMeanPost hVarLoop hVarPost hOutLoop

theorem layernorm_fwd_triton_fullN_correct
    (X W Y : RegionName)
    (stride_x_N stride_x_hn stride_x_hd
      stride_y_N stride_y_hn stride_y_hd stride_w_hn stride_w_hd
      N BLOCK_SIZE : Nat)
    (eps : ℝ) (s s' : BlockState)
    (hBlockPos : 0 < BLOCK_SIZE)
    (hXYNe : X ≠ Y)
    (hWYNe : W ≠ Y)
    (hExec : exec (layernorm_fwd_triton X W Y
        stride_x_N stride_x_hn stride_x_hd
        stride_y_N stride_y_hn stride_y_hd stride_w_hn stride_w_hd
        N BLOCK_SIZE eps) s = some s') :
    ∀ i : Fin N,
      s'.readMem Y (yColOffset s stride_y_N stride_y_hn i.val) =
        layernormYFullNSpec s X W stride_x_N stride_x_hn stride_w_hn
          N BLOCK_SIZE eps i := by
  unfold exec at hExec
  rw [layernorm_fwd_triton_toAlg_body] at hExec
  rw [stepStmts.append_some_iff] at hExec
  rcases hExec with ⟨stOutInit, hBeforeOut, hOutLoopList⟩
  rw [stepStmts.append_some_iff] at hBeforeOut
  rcases hBeforeOut with ⟨stVar, hBeforeVarPost, hVarPost⟩
  rw [stepStmts.append_some_iff] at hBeforeVarPost
  rcases hBeforeVarPost with ⟨stVarInit, hBeforeVarLoop, hVarLoopList⟩
  rw [stepStmts.append_some_iff] at hBeforeVarLoop
  rcases hBeforeVarLoop with ⟨stMean, hBeforeMeanPost, hMeanPost⟩
  rw [stepStmts.append_some_iff] at hBeforeMeanPost
  rcases hBeforeMeanPost with ⟨stPre, hPre, hMeanLoopList⟩
  have hMeanLoop :
      stepStmt (.forRange "off" 0 N BLOCK_SIZE
        (layernormMeanLoopBody N BLOCK_SIZE)) stPre = some stMean := by
    unfold stepStmts at hMeanLoopList
    unfold stepStmt
    cases hAux :
        stepForRangeAux "off" 0 N BLOCK_SIZE
          (layernormMeanLoopBody N BLOCK_SIZE) stPre <;>
      simp [hAux] at hMeanLoopList ⊢
    exact hMeanLoopList
  have hVarLoop :
      stepStmt (.forRange "off" 0 N BLOCK_SIZE
        (layernormVarLoopBody N BLOCK_SIZE)) stVarInit = some stVar := by
    unfold stepStmts at hVarLoopList
    unfold stepStmt
    cases hAux :
        stepForRangeAux "off" 0 N BLOCK_SIZE
          (layernormVarLoopBody N BLOCK_SIZE) stVarInit <;>
      simp [hAux] at hVarLoopList ⊢
    exact hVarLoopList
  have hOutLoop :
      stepStmt (.forRange "off" 0 N BLOCK_SIZE
        (layernormOutLoopBody N BLOCK_SIZE)) stOutInit = some s' := by
    unfold stepStmts at hOutLoopList
    unfold stepStmt
    cases hAux :
        stepForRangeAux "off" 0 N BLOCK_SIZE
          (layernormOutLoopBody N BLOCK_SIZE) stOutInit <;>
      simp [hAux] at hOutLoopList ⊢
    exact hOutLoopList
  exact layernorm_fwd_triton_staged_fullN_correct_from_preloop s stPre stMean
    stVarInit stVar stOutInit s' X W Y stride_x_N stride_x_hn stride_y_N
    stride_y_hn stride_w_hn N BLOCK_SIZE eps hBlockPos hXYNe hWYNe hPre
    hMeanLoop hMeanPost hVarLoop hVarPost hOutLoop

/-- Compute-facing correctness for the full-N layernorm forward kernel. -/
theorem layernorm_fwd_triton_compute_fullN_correct
    (X W Y : RegionName)
    (stride_x_N stride_x_hn stride_x_hd
      stride_y_N stride_y_hn stride_y_hd stride_w_hn stride_w_hd
      N BLOCK_SIZE : Nat)
    (eps : ℝ) (s : BlockState)
    (hBlockPos : 0 < BLOCK_SIZE)
    (hXYNe : X ≠ Y)
    (hWYNe : W ≠ Y) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := layernorm_fwd_triton X W Y
        stride_x_N stride_x_hn stride_x_hd
        stride_y_N stride_y_hn stride_y_hd stride_w_hn stride_w_hd
        N BLOCK_SIZE eps)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin N => True)
        (fun i => (Y, yColOffset s stride_y_N stride_y_hn i.val)))
      (expected := fun i =>
        layernormYFullNSpec s X W stride_x_N stride_x_hn stride_w_hn
          N BLOCK_SIZE eps i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [layernorm_fwd_triton, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i _hActive
  exact layernorm_fwd_triton_fullN_correct X W Y
    stride_x_N stride_x_hn stride_x_hd
    stride_y_N stride_y_hn stride_y_hd stride_w_hn stride_w_hd
    N BLOCK_SIZE eps s s' hBlockPos hXYNe hWYNe hExec i

/-- Algorithm-layer correctness for the one-block layernorm forward slice. -/
theorem layernorm_fwd_triton_correct
    (X W Y : RegionName)
    (stride_x_N stride_x_hn stride_x_hd
      stride_y_N stride_y_hn stride_y_hd stride_w_hn stride_w_hd
      N BLOCK_SIZE : Nat)
    (eps : ℝ) (s s' : BlockState)
    (hNpos : 0 < N) (hNle : N ≤ BLOCK_SIZE)
    (_hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => yOffset s stride_y_N stride_y_hn i))
    (hExec : exec (layernorm_fwd_triton X W Y
        stride_x_N stride_x_hn stride_x_hd
        stride_y_N stride_y_hn stride_y_hd stride_w_hn stride_w_hd
        N BLOCK_SIZE eps) s = some s') :
    ∀ i : Fin BLOCK_SIZE,
      s'.readMem Y (yOffset s stride_y_N stride_y_hn i) =
        if i.val < N then
          layernormYSpec s X W stride_x_N stride_x_hn stride_w_hn
            N BLOCK_SIZE eps i
        else s.readMem Y (yOffset s stride_y_N stride_y_hn i) := by
  intro i
  by_cases hB : 0 < BLOCK_SIZE
  · have hStep : BLOCK_SIZE ≠ 0 := Nat.ne_of_gt hB
    simp [layernorm_fwd_triton, ComputeKernel.toAlgKernel,
          ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?,
          ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
          ComputeOp.toAlgorithm?, Except.bind, Bind.bind, pure, Pure.pure] at hExec
    simp only [exec, stepStmts, stepStmt, evalOp, evalOp.eq_def,
          stepForRangeAux.step_lt, stepForRangeAux.step_ge,
          Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop, Tile.reduceSum,
          Tile.reduceSumDrop, Tile.select, TileShape.axisDim, TileShape.eraseAxis,
          TileShape.insertAxisIndex, NumericDType.add, NumericDType.sub,
          NumericDType.mul, NumericDType.div, ComparableDType.lt, hNpos, hNle,
          Nat.not_lt.mpr hNle, hStep] at hExec
    simp [stepForRangeAux.step_lt hStep hNpos,
          stepForRangeAux.step_ge hStep hNle] at hExec
    simp only [stepStmts, stepStmt, evalOp, evalOp.eq_def, Tile.bop, Tile.cop, Tile.ptrAdd,
          Tile.uop, Tile.reduceSum, Tile.reduceSumDrop, Tile.select,
          TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
          NumericDType.add, NumericDType.sub, NumericDType.mul,
          NumericDType.div, ComparableDType.lt, hNpos, hNle,
          Nat.not_lt.mpr hNle, hStep] at hExec
    simp [BlockState.setReg] at hExec
    subst s'
    simp only [yOffset]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
          (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
    by_cases hi : i.val < N
    · simp only [hi, ↓reduceIte]
      simp [hi, layernormYSpec, layernormInvVarCarrier, layernormVarCarrier,
            layernormCenteredTile, layernormMeanCarrier, layernormInputTile,
            xOffset, wOffset, Tile.reduceSum, Tile.reduceSumDrop, Tile.select,
            TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
            WithBot.realSqrt, NumericDType.add, NumericDType.sub,
            NumericDType.mul, NumericDType.div, FloatDType.cast]
      rfl
    · simp [hi, BlockState.readMem]
  · exact False.elim (hB (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))
/-- Compute-facing correctness for the one-block layernorm forward slice. -/
theorem layernorm_fwd_triton_compute_correct
    (X W Y : RegionName)
    (stride_x_N stride_x_hn stride_x_hd
      stride_y_N stride_y_hn stride_y_hd stride_w_hn stride_w_hd
      N BLOCK_SIZE : Nat)
    (eps : ℝ) (s : BlockState)
    (hNpos : 0 < N) (hNle : N ≤ BLOCK_SIZE)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => yOffset s stride_y_N stride_y_hn i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := layernorm_fwd_triton X W Y
        stride_x_N stride_x_hn stride_x_hd
        stride_y_N stride_y_hn stride_y_hd stride_w_hn stride_w_hd
        N BLOCK_SIZE eps)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => i.val < N)
        (fun i => (Y, yOffset s stride_y_N stride_y_hn i)))
      (expected := fun i =>
        layernormYSpec s X W stride_x_N stride_x_hn stride_w_hn
          N BLOCK_SIZE eps i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [layernorm_fwd_triton, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := layernorm_fwd_triton_correct X W Y
    stride_x_N stride_x_hn stride_x_hd
    stride_y_N stride_y_hn stride_y_hd stride_w_hn stride_w_hd
    N BLOCK_SIZE eps s s' hNpos hNle hOutInj hExec i
  simpa [hActive] using h

/-- Per-kernel output summary for `_layer_norm_fwd_kernel`: the DSL surface
lowers to the algorithm layer, and the masked store to `Y` is compute-correct
for arbitrary `N` — every output column holds the full-`N` LayerNorm spec
`layernormYFullNSpec`. Built on the multi-block `*_compute_fullN_correct`
result; requires only `0 < BLOCK_SIZE` and output/input disjointness. -/
specification layernorm_fwd_triton_output_summary
    (X W Y : RegionName)
    (stride_x_N stride_x_hn stride_x_hd
      stride_y_N stride_y_hn stride_y_hd stride_w_hn stride_w_hd
      N BLOCK_SIZE : Nat)
    (eps : ℝ) (s : BlockState)
    (hBlockPos : 0 < BLOCK_SIZE)
    (hXYNe : X ≠ Y)
    (hWYNe : W ≠ Y) :
    (∃ alg, (layernorm_fwd_triton X W Y
        stride_x_N stride_x_hn stride_x_hd
        stride_y_N stride_y_hn stride_y_hd stride_w_hn stride_w_hd
        N BLOCK_SIZE eps).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel := layernorm_fwd_triton X W Y
        stride_x_N stride_x_hn stride_x_hd
        stride_y_N stride_y_hn stride_y_hd stride_w_hn stride_w_hd
        N BLOCK_SIZE eps)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin N => True)
        (fun i => (Y, yColOffset s stride_y_N stride_y_hn i.val)))
      (expected := fun i =>
        layernormYFullNSpec s X W stride_x_N stride_x_hn stride_w_hn
          N BLOCK_SIZE eps i) := by
  refine ⟨?_, ?_⟩
  · simp only [layernorm_fwd_triton, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
    exact ⟨_, rfl⟩
  · exact layernorm_fwd_triton_compute_fullN_correct X W Y
      stride_x_N stride_x_hn stride_x_hd
      stride_y_N stride_y_hn stride_y_hd stride_w_hn stride_w_hd
      N BLOCK_SIZE eps s hBlockPos hXYNe hWYNe


/-! ## The `⊨[R]` streaming headline (wave-5 S3 per-step emit genre)

Everything below is purely additive; the exact surface above is untouched.
This is a consumer of the per-step emit skin `StreamEmitMasked2DKernelIO₂`
(streaming genre, style S3): the store sits **inside** the third pass, so
the output is a per-step `BLOCK_SIZE`-lane window family rather than one
terminal tile, and the kernel's spec `f t j` is the genre's *multi-pass*
shape — the step-`t` tiles combined with two folds over the entire stream
(mean, then variance about that mean).

Structure of the `execR R` story: this kernel has **zero rounding events**.
Every load and store is at `.real` — the Python `.to(tl.float32)` input
casts and the store cast `(y).to(X.dtype.element_ty)` are already erased by
the lowering (see `layernormOutLoopBody`: the emitted statement is a plain
`Stmt.store .real`), so `outDType := .real` is the honest grid and there is
no boundary quantization event. All three passes therefore collapse
verbatim onto the exact stepper (`stepForRangeAuxR_castFree`), and the
whole proven mean / var / out invariant stack above is reused unchanged;
the `⊨[R]` face adds only the `TraceSafeR` walk, the per-cell memory frame,
and the stream-lane spec bridge. The skin's readback contract at the
default `outDType := .real` grid carries `R.round .real`, the identity by
`round_real` — the ∀-`R` face is the exact streaming contract via the
model's `.real` identity fields, not a `.triv` special case. -/

section IOFace

open scoped VeriTile.Triton.StreamEmitMasked2DKernelIO₂

set_option maxHeartbeats 4000000
set_option linter.unusedVariables false

/-! ### Stream geometry: trip count and windows -/

/-- Trip count of all three `for off in range(0, N, BLOCK_SIZE)` passes:
`⌈N / BLOCK_SIZE⌉`. -/
def lnNumSteps (N B : Nat) : Nat := (N + B - 1) / B

private theorem lnNumSteps_mul_ge (N B : Nat) (hB : 0 < B) :
    N ≤ lnNumSteps N B * B := by
  rcases Nat.eq_zero_or_pos N with rfl | hN
  · exact Nat.zero_le _
  · unfold lnNumSteps
    have heq : N + B - 1 = (N - 1) + B := by omega
    rw [heq, Nat.add_div_right _ hB]
    have h2 : (N - 1) % B + 1 ≤ B := Nat.mod_lt _ hB
    calc N = (N - 1) + 1 := by omega
      _ = (N - 1) / B * B + ((N - 1) % B + 1) := by
          rw [← Nat.add_assoc, Nat.div_add_mod']
      _ ≤ (N - 1) / B * B + B := Nat.add_le_add_left h2 _
      _ = ((N - 1) / B + 1) * B := (Nat.succ_mul _ _).symm

private theorem lnStep_lt_numSteps (N B i : Nat) (hB : 0 < B) (hi : i < N) :
    i / B < lnNumSteps N B := by
  have h2 : i / B * B < lnNumSteps N B * B :=
    Nat.lt_of_le_of_lt (Nat.div_mul_le_self i B)
      (Nat.lt_of_lt_of_le hi (lnNumSteps_mul_ge N B hB))
  exact Nat.lt_of_mul_lt_mul_right h2

/-! ### Re-blocking arithmetic: `k ↔ (k / B, k % B)` -/

private theorem ln_sum_blocks_lanes (B c : Nat) (hB : 0 < B) (H : Nat → ℝ) :
    (∑ b : Fin c, ∑ j : Fin B, H (b.val * B + j.val))
      = ∑ k : Fin (c * B), H k.val := by
  rw [← Fintype.sum_prod_type']
  apply Finset.sum_nbij' (i := fun p : Fin c × Fin B => (⟨p.1.val * B + p.2.val, by
        have h1 := p.1.isLt; have h2 := p.2.isLt
        calc p.1.val * B + p.2.val < p.1.val * B + B := by omega
          _ = (p.1.val + 1) * B := by ring
          _ ≤ c * B := Nat.mul_le_mul_right _ (by omega)⟩ : Fin (c*B)))
    (j := fun k : Fin (c*B) => (⟨k.val / B,
        (Nat.div_lt_iff_lt_mul hB).mpr (by have := k.isLt; omega)⟩,
      ⟨k.val % B, Nat.mod_lt _ hB⟩))
  · intro p _; simp
  · intro k _; simp
  · intro p _
    apply Prod.ext
    · apply Fin.ext; show (p.1.val * B + p.2.val) / B = p.1.val
      rw [Nat.mul_comm p.1.val B, Nat.mul_add_div hB, Nat.div_eq_of_lt p.2.isLt,
        Nat.add_zero]
    · apply Fin.ext; show (p.1.val * B + p.2.val) % B = p.2.val
      rw [Nat.mul_comm p.1.val B, Nat.mul_add_mod, Nat.mod_eq_of_lt p.2.isLt]
  · intro k _; apply Fin.ext; show (k.val / B) * B + k.val % B = k.val
    rw [Nat.mul_comm (k.val / B) B]; exact Nat.div_add_mod k.val B
  · intro p _; rfl

private theorem ln_sum_fin_extend (N M : Nat) (hNM : N ≤ M) (f : Nat → ℝ) :
    (∑ k : Fin M, (if k.val < N then f k.val else 0)) = ∑ k : Fin N, f k.val := by
  rw [Fin.sum_univ_eq_sum_range (fun k => if k < N then f k else 0) M,
      Fin.sum_univ_eq_sum_range (fun k => f k) N]
  rw [← Finset.sum_subset (s₁ := Finset.range N) (s₂ := Finset.range M)
        (fun x hx => Finset.mem_range.mpr
          (lt_of_lt_of_le (Finset.mem_range.mp hx) hNM))
        (by
          intro k hk hknotN
          simp only [Finset.mem_range] at hk hknotN
          simp [Nat.not_lt.mp hknotN])]
  apply Finset.sum_congr rfl
  intro k hk; simp only [Finset.mem_range] at hk; simp [hk]

/-- The guarded block×lane double sum **is** the flat `Σ_{k<N}` sum. -/
private theorem ln_reblock (B c N : Nat) (hB : 0 < B) (hge : N ≤ c * B)
    (f : Nat → ℝ) :
    (∑ j : Fin B, ∑ b : Fin c,
        (if (b.val * B + j.val) < N then f (b.val * B + j.val) else 0))
      = ∑ k : Fin N, f k.val := by
  rw [Finset.sum_comm]
  rw [ln_sum_blocks_lanes B c hB (fun m => if m < N then f m else 0)]
  exact ln_sum_fin_extend N (c*B) hge f

/-! ### IO signature -/

/-- **Streaming IO signature** of `layernorm_fwd_triton` on the two-stream
per-step emit skin (S3: in-loop store). Step `t` of any of the three passes
(at `off = t·BLOCK_SIZE`) addresses the `BLOCK_SIZE`-lane column window
`cols = t·BLOCK_SIZE + j`; the mean and variance passes read the `X` window
(`read1`), the third pass re-reads that same `X` window plus the `W` window
(`read2`) and stores the `BLOCK_SIZE`-lane output window (`write`) at the
**`.real`** grid (`outDType` default — `layernormOutLoopBody`'s emitted
statement is a plain `Stmt.store .real`, the Python
`(y).to(X.dtype.element_ty)` cast having been erased by the lowering, so the
per-step stores carry no quantization event). The windows transcribe the
kernel's pointer arithmetic verbatim (`X += Seq*stride_x_N + H*stride_x_hn`,
`W += H*stride_w_hn`, `Y += Seq*stride_y_N + H*stride_y_hn`, all indexed by
`cols`):

* `read1` step `t`, lane `j`:
  `pid₀·stride_x_N + pid₁·stride_x_hn + (t·BLOCK_SIZE + j)` — the file's
  `xColOffset`.
* `read2` step `t`, lane `j`: `pid₁·stride_w_hn + (t·BLOCK_SIZE + j)` — the
  file's `wColOffset`.
* `write` step `t`, lane `j`:
  `pid₀·stride_y_N + pid₁·stride_y_hn + (t·BLOCK_SIZE + j)` — the file's
  `yColOffset`.

All three masks are the kernel's single `cols < N`. -/
def layernormKernelIO (X W Y : RegionName)
    (stride_x_N stride_x_hn stride_x_hd
      stride_y_N stride_y_hn stride_y_hd
      stride_w_hn stride_w_hd N BLOCK_SIZE : Nat) (eps : ℝ) :
    StreamEmitMasked2DKernelIO₂ where
  kernel := layernorm_fwd_triton X W Y stride_x_N stride_x_hn stride_x_hd
    stride_y_N stride_y_hn stride_y_hd stride_w_hn stride_w_hd N BLOCK_SIZE eps
  inp1 := X
  inp2 := W
  out := Y
  T := lnNumSteps N BLOCK_SIZE
  B1 := BLOCK_SIZE
  B2 := BLOCK_SIZE
  C := BLOCK_SIZE
  read1 := fun p₀ p₁ t j =>
    p₀ * stride_x_N + p₁ * stride_x_hn + (t.val * BLOCK_SIZE + j.val)
  read2 := fun _ p₁ t j => p₁ * stride_w_hn + (t.val * BLOCK_SIZE + j.val)
  write := fun p₀ p₁ t j =>
    p₀ * stride_y_N + p₁ * stride_y_hn + (t.val * BLOCK_SIZE + j.val)
  mask1 := fun _ _ t j => t.val * BLOCK_SIZE + j.val < N
  mask2 := fun _ _ t j => t.val * BLOCK_SIZE + j.val < N
  writeMask := fun _ _ t j => t.val * BLOCK_SIZE + j.val < N

/-! ### The stream-level spec -/

/-- The guarded stream-level sum of the `X` stream: the mean pass's
`_mean += a` fold over the whole curried stream, guarded by the kernel's
window (`t·B + e < N`) — the contract only pins `xs` on masked lanes, so
the spec must not read unmasked lanes. -/
noncomputable def lnStreamSum (N B : Nat)
    (xs : Fin (lnNumSteps N B) → Fin B → ℝ) : ℝ :=
  ∑ u : Fin (lnNumSteps N B), ∑ e : Fin B,
    if u.val * B + e.val < N then xs u e else 0

/-- The stream-level mean: `tl.sum(_mean, axis=0) / N`. -/
noncomputable def lnStreamMean (N B : Nat)
    (xs : Fin (lnNumSteps N B) → Fin B → ℝ) : ℝ :=
  lnStreamSum N B xs / (N : ℝ)

/-- The stream-level variance: the second pass's `_var += x*x` fold over the
whole stream, centred at `lnStreamMean` and guarded by the same window (the
kernel's `tl.where(cols < N, x - mean, 0.0)`), divided by `N`. -/
noncomputable def lnStreamVar (N B : Nat)
    (xs : Fin (lnNumSteps N B) → Fin B → ℝ) : ℝ :=
  (∑ u : Fin (lnNumSteps N B), ∑ e : Fin B,
      if u.val * B + e.val < N then (xs u e - lnStreamMean N B xs) ^ 2 else 0)
    / (N : ℝ)

/-- The stream-level LayerNorm spec (the genre's three-pass shape): emitted
window `(t, j)` holds `((x[t,j] - mean) · rstd) · w[t,j]`, where `mean` and
`rstd = 1/√(var + eps)` are folds over the *entire* `x` stream.
Algebraically `layernormYFullNSpec` with the `Fin N` sums re-blocked to the
guarded stream double sums. -/
noncomputable def lnStreamSpec (N B : Nat) (eps : ℝ)
    (xs ws : Fin (lnNumSteps N B) → Fin B → ℝ)
    (t : Fin (lnNumSteps N B)) (j : Fin B) : ℝ :=
  ((xs t j - lnStreamMean N B xs) *
      (Real.sqrt (lnStreamVar N B xs + eps))⁻¹) * ws t j

/-! ### The stream-lane spec bridge -/

/-- Under the stream pin, the guarded stream double sum **is** the exact
stack's `Σ_{k<N} x[k]` (re-blocking `k ↔ (k/B, k%B)` via `ln_reblock`). -/
private theorem lnStreamSum_eq (X : RegionName) (s₀ : BlockState)
    (stride_x_N stride_x_hn N B : Nat) (hB : 0 < B)
    (xs : Fin (lnNumSteps N B) → Fin B → ℝ)
    (hx : ∀ (t : Fin (lnNumSteps N B)) (e : Fin B), t.val * B + e.val < N →
      s₀.readMem X (s₀.pids 0 * stride_x_N + s₀.pids 1 * stride_x_hn +
          (t.val * B + e.val)) = xs t e) :
    lnStreamSum N B xs
      = ∑ k : Fin N,
          s₀.readMem X (xColOffset s₀ stride_x_N stride_x_hn k.val) := by
  unfold lnStreamSum
  rw [Finset.sum_comm,
    ← ln_reblock B (lnNumSteps N B) N hB (lnNumSteps_mul_ge N B hB)
      (fun k => s₀.readMem X (xColOffset s₀ stride_x_N stride_x_hn k))]
  refine Finset.sum_congr rfl fun e _ => Finset.sum_congr rfl fun t _ => ?_
  by_cases h : t.val * B + e.val < N
  · rw [if_pos h, if_pos h, ← hx t e h]
    rfl
  · rw [if_neg h, if_neg h]

private theorem lnStreamMean_eq (X : RegionName) (s₀ : BlockState)
    (stride_x_N stride_x_hn N B : Nat) (hB : 0 < B)
    (xs : Fin (lnNumSteps N B) → Fin B → ℝ)
    (hx : ∀ (t : Fin (lnNumSteps N B)) (e : Fin B), t.val * B + e.val < N →
      s₀.readMem X (s₀.pids 0 * stride_x_N + s₀.pids 1 * stride_x_hn +
          (t.val * B + e.val)) = xs t e) :
    lnStreamMean N B xs
      = layernormMeanFullNSpec s₀ X stride_x_N stride_x_hn N B := by
  unfold lnStreamMean layernormMeanFullNSpec
  rw [lnStreamSum_eq X s₀ stride_x_N stride_x_hn N B hB xs hx]

private theorem lnStreamVar_eq (X : RegionName) (s₀ : BlockState)
    (stride_x_N stride_x_hn N B : Nat) (hB : 0 < B)
    (xs : Fin (lnNumSteps N B) → Fin B → ℝ)
    (hx : ∀ (t : Fin (lnNumSteps N B)) (e : Fin B), t.val * B + e.val < N →
      s₀.readMem X (s₀.pids 0 * stride_x_N + s₀.pids 1 * stride_x_hn +
          (t.val * B + e.val)) = xs t e) :
    lnStreamVar N B xs
      = layernormVarFullNSpec s₀ X stride_x_N stride_x_hn N B := by
  unfold lnStreamVar layernormVarFullNSpec
  rw [lnStreamMean_eq X s₀ stride_x_N stride_x_hn N B hB xs hx]
  congr 1
  rw [Finset.sum_comm,
    ← ln_reblock B (lnNumSteps N B) N hB (lnNumSteps_mul_ge N B hB)
      (fun k => (s₀.readMem X (xColOffset s₀ stride_x_N stride_x_hn k) -
        layernormMeanFullNSpec s₀ X stride_x_N stride_x_hn N B) ^ 2)]
  refine Finset.sum_congr rfl fun e _ => Finset.sum_congr rfl fun t _ => ?_
  by_cases h : t.val * B + e.val < N
  · rw [if_pos h, if_pos h, ← hx t e h]
    rfl
  · rw [if_neg h, if_neg h]

/-- Per-lane spec bridge: at a masked window `(t, j)` the stream spec **is**
the exact stack's `layernormYFullNSpec` at global lane `t·B + j`. -/
private theorem lnStreamSpec_eq_layernormYFullNSpec (X W : RegionName)
    (s₀ : BlockState) (stride_x_N stride_x_hn stride_w_hn N B : Nat) (eps : ℝ)
    (hB : 0 < B) (xs ws : Fin (lnNumSteps N B) → Fin B → ℝ)
    (hx : ∀ (t : Fin (lnNumSteps N B)) (e : Fin B), t.val * B + e.val < N →
      s₀.readMem X (s₀.pids 0 * stride_x_N + s₀.pids 1 * stride_x_hn +
          (t.val * B + e.val)) = xs t e)
    (hw : ∀ (t : Fin (lnNumSteps N B)) (e : Fin B), t.val * B + e.val < N →
      s₀.readMem W (s₀.pids 1 * stride_w_hn + (t.val * B + e.val)) = ws t e)
    (t : Fin (lnNumSteps N B)) (j : Fin B) (hj : t.val * B + j.val < N) :
    lnStreamSpec N B eps xs ws t j
      = layernormYFullNSpec s₀ X W stride_x_N stride_x_hn stride_w_hn N B eps
          ⟨t.val * B + j.val, hj⟩ := by
  unfold lnStreamSpec layernormYFullNSpec layernormRstdFullNSpec
  rw [lnStreamMean_eq X s₀ stride_x_N stride_x_hn N B hB xs hx,
    lnStreamVar_eq X s₀ stride_x_N stride_x_hn N B hB xs hx,
    ← hx t j hj, ← hw t j hj]
  rfl

/-! ### Cast-free collapses and the covered fragment -/

/-- The erased `.to(tl.float32)` is a `.real → .real` cast: exact under
every `R` — `R.roundW .real` is the identity by the model's defining
`round_real`. Kept in the collapse simp sets for uniformity with the
family; this kernel's lowering leaves no `castFloat` at all. -/
private theorem Rcast_real_real (R : RoundingModel) :
    R.cast .real .real = FloatDType.cast .real .real := by
  funext v
  simp [RoundingModel.cast, FloatDType.cast]

/-- The prologue is cast-free: it steps identically under `stepStmtsR R`. -/
private theorem lnPreLoop_castFree (R : RoundingModel) (X W Y : RegionName)
    (stride_x_N stride_x_hn stride_y_N stride_y_hn stride_w_hn BLOCK_SIZE : Nat)
    (t : BlockState) :
    stepStmtsR R (layernormMeanPreLoop X W Y stride_x_N stride_x_hn stride_y_N
        stride_y_hn stride_w_hn BLOCK_SIZE) t
      = stepStmts (layernormMeanPreLoop X W Y stride_x_N stride_x_hn stride_y_N
        stride_y_hn stride_w_hn BLOCK_SIZE) t := by
  simp only [layernormMeanPreLoop, stepStmtsR, stepStmts, stepStmtR, stepStmt,
    evalOpR.eq_def, evalOp.eq_def, Rcast_real_real]
  rfl

/-- The mean-pass body is cast-free (masked `.real` load, a real add). -/
private theorem lnMeanBody_castFree (R : RoundingModel) (N BLOCK_SIZE : Nat)
    (t : BlockState) :
    stepStmtsR R (layernormMeanLoopBody N BLOCK_SIZE) t
      = stepStmts (layernormMeanLoopBody N BLOCK_SIZE) t := by
  simp only [layernormMeanLoopBody, stepStmtsR, stepStmts, stepStmtR, stepStmt,
    evalOpR.eq_def, evalOp.eq_def, Rcast_real_real]
  rfl

/-- The mean reduce / `_var` init segment is cast-free. -/
private theorem lnMeanPost_castFree (R : RoundingModel) (N BLOCK_SIZE : Nat)
    (t : BlockState) :
    stepStmtsR R (layernormMeanPostLoop N BLOCK_SIZE) t
      = stepStmts (layernormMeanPostLoop N BLOCK_SIZE) t := by
  simp only [layernormMeanPostLoop, stepStmtsR, stepStmts, stepStmtR, stepStmt,
    evalOpR.eq_def, evalOp.eq_def, Rcast_real_real]
  rfl

/-- The variance-pass body is cast-free. -/
private theorem lnVarBody_castFree (R : RoundingModel) (N BLOCK_SIZE : Nat)
    (t : BlockState) :
    stepStmtsR R (layernormVarLoopBody N BLOCK_SIZE) t
      = stepStmts (layernormVarLoopBody N BLOCK_SIZE) t := by
  simp only [layernormVarLoopBody, stepStmtsR, stepStmts, stepStmtR, stepStmt,
    evalOpR.eq_def, evalOp.eq_def, Rcast_real_real]
  rfl

/-- The variance reduce / `rstd` segment is cast-free. -/
private theorem lnVarPost_castFree (R : RoundingModel) (N BLOCK_SIZE : Nat)
    (eps : ℝ) (t : BlockState) :
    stepStmtsR R (layernormVarPostLoop N BLOCK_SIZE eps) t
      = stepStmts (layernormVarPostLoop N BLOCK_SIZE eps) t := by
  simp only [layernormVarPostLoop, stepStmtsR, stepStmts, stepStmtR, stepStmt,
    evalOpR.eq_def, evalOp.eq_def, Rcast_real_real]
  rfl

/-- The output-pass body is cast-free **including its in-loop masked `.real`
store**: `stepStmtR` delegates a `.real`-typed store to the exact
`writeMemTyped` (`writeMemTypedR R .real` is definitionally the exact
write), so the whole storing loop steps identically under `stepStmtsR R` and
the exact output-loop invariant stack transports to `execR`. -/
private theorem lnOutBody_castFree (R : RoundingModel) (N BLOCK_SIZE : Nat)
    (t : BlockState) :
    stepStmtsR R (layernormOutLoopBody N BLOCK_SIZE) t
      = stepStmts (layernormOutLoopBody N BLOCK_SIZE) t := by
  simp only [layernormOutLoopBody, stepStmtsR, stepStmts, stepStmtR, stepStmt,
    evalOpR.eq_def, evalOp.eq_def, Rcast_real_real,
    BlockState.writeMemTypedR]
  rfl

/-- The full three-pass surface sits inside the flat-memory bridge's covered
fragment (`FlattenOk`; the three `forRange` clauses recurse into the
cast-free bodies). -/
theorem layernorm_fwd_triton_flattenOk (X W Y : RegionName)
    (stride_x_N stride_x_hn stride_x_hd
      stride_y_N stride_y_hn stride_y_hd stride_w_hn stride_w_hd
      N BLOCK_SIZE : Nat) (eps : ℝ) :
    ((layernorm_fwd_triton X W Y stride_x_N stride_x_hn stride_x_hd
      stride_y_N stride_y_hn stride_y_hd stride_w_hn stride_w_hd
      N BLOCK_SIZE eps).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  rw [layernorm_fwd_triton_toAlg_body]
  simp [layernormMeanPreLoop, layernormMeanLoopBody, layernormMeanPostLoop,
    layernormVarLoopBody, layernormVarPostLoop, layernormOutLoopBody,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]


/-! ### Per-op evaluation and its `R`-independence

The three passes' index / address / mask ops are nat-, ptr- and bool-typed:
no float arithmetic, hence `evalOpR R` is `evalOp` on the nose. -/

private theorem ln_colsR_eq (R : RoundingModel) (B : Nat) (s : BlockState) :
    evalOpR R (Op.add NumericDType.nat Broadcast.scalarL
        (Op.ref .nat [] "off") (Op.arange B)) s
      = evalOp (Op.add NumericDType.nat Broadcast.scalarL
        (Op.ref .nat [] "off") (Op.arange B)) s := by
  simp only [evalOpR.eq_def, evalOp.eq_def]

private theorem ln_addrR_eq (R : RoundingModel) (P : RegName) (B : Nat)
    (s : BlockState) :
    evalOpR R (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] P)
        (Op.ref .nat [B] "cols")) s
      = evalOp (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] P)
        (Op.ref .nat [B] "cols")) s := by
  simp only [evalOpR.eq_def, evalOp.eq_def]

private theorem ln_maskR_eq (R : RoundingModel) (N B : Nat) (s : BlockState) :
    evalOpR R (Op.lt ComparableDType.nat Broadcast.scalarR
        (Op.ref .nat [B] "cols") (Op.constNat N)) s
      = evalOp (Op.lt ComparableDType.nat Broadcast.scalarR
        (Op.ref .nat [B] "cols") (Op.constNat N)) s := by
  simp only [evalOpR.eq_def, evalOp.eq_def]

/-- Per-lane value of the `cols = off + tl.arange(0, BLOCK_SIZE)` op. -/
private theorem ln_cols_eval (B : Nat) (s : BlockState) (off : Nat)
    (hoff : s.regs .nat [] "off" = some (Tile.scalar off)) :
    evalOp (Op.add NumericDType.nat Broadcast.scalarL
        (Op.ref .nat [] "off") (Op.arange B)) s
      = some (Tile.vec (fun j : Fin B => off + j.val)) := by
  rw [evalOp_add, evalOp_ref, hoff, evalOp_arange]
  apply congrArg some
  ext j
  simp only [Tile.bop_data, Tile.vec, Tile.scalar, Broadcast.rightIndex_scalarL,
    NumericDType.add]

/-- Per-lane value of a masked access's address op `PTR + cols`. -/
private theorem ln_addr_eval (P : RegName) (B : Nat) (Rg : RegionName)
    (base : Nat) (s : BlockState) (off : Nat)
    (hp : s.regs .ptr [] P = some (Tile.scalar (Region.cast Rg, base)))
    (hc : s.regs .nat [B] "cols"
      = some (Tile.vec (fun j : Fin B => off + j.val))) :
    evalOp (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] P)
        (Op.ref .nat [B] "cols")) s
      = some (Tile.vec (fun j : Fin B =>
          (Region.cast Rg, base + (off + j.val)))) := by
  rw [evalOp_ptrAdd, evalOp_ref, hp, evalOp_ref, hc]
  apply congrArg some
  ext j
  · simp only [Tile.ptrAdd_data, Tile.vec, Tile.scalar,
      Broadcast.leftIndex_scalarL]
  · simp only [Tile.ptrAdd_data, Tile.vec, Tile.scalar,
      Broadcast.leftIndex_scalarL, Broadcast.rightIndex_scalarL]

/-- Per-lane value of the kernel's single mask `cols < N`. -/
private theorem ln_mask_eval (N B : Nat) (s : BlockState) (off : Nat)
    (hc : s.regs .nat [B] "cols"
      = some (Tile.vec (fun j : Fin B => off + j.val))) :
    evalOp (Op.lt ComparableDType.nat Broadcast.scalarR
        (Op.ref .nat [B] "cols") (Op.constNat N)) s
      = some (Tile.vec (fun j : Fin B => decide (off + j.val < N))) := by
  rw [evalOp_lt, evalOp_ref, hc, evalOp_constNat]
  apply congrArg some
  ext j
  simp only [Tile.cop_data, Tile.vec, Tile.scalar, Broadcast.leftIndex_scalarR,
    ComparableDType.lt]
  rfl

/-! ### The per-body `TraceSafeListR` walks -/

set_option maxHeartbeats 4000000 in
/-- Per-iteration `TraceSafeListR` for the **mean pass** body: the `cols`
assign and the `_mean` accumulate are register-only; the masked `X` load's
**active** lanes are exactly the skin's `mask1` window at step `off / B`, in
bounds by the `read1` window bound (instantiated at the raw counter
`off`). -/
private theorem ln_meanBodySafeR (R : RoundingModel) (bounds : RegionBounds)
    (Xr : RegionName) (stride_x_N stride_x_hn N B : Nat)
    (s0 st : BlockState) (off : Nat)
    (hXPtr : st.regs .ptr [] "X"
      = some (Tile.scalar (Region.cast Xr,
          s0.pids 0 * stride_x_N + s0.pids 1 * stride_x_hn)))
    (hbx : ∀ j : Fin B, off + j.val < N →
      s0.pids 0 * stride_x_N + s0.pids 1 * stride_x_hn + (off + j.val)
        < bounds Xr) :
    Stmt.TraceSafeListR R bounds (layernormMeanLoopBody N B)
      (st.setReg "off" .nat [] (Tile.scalar off)) := by
  unfold layernormMeanLoopBody
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR]) (fun t1 ht1 => ?_)
  rw [stepStmtR_assign_eq_some
    ((ln_colsR_eq R B _).trans
      (ln_cols_eval B _ off (BlockState.setReg_same _ _ _ _ _)))] at ht1
  obtain rfl := Option.some.inj ht1
  set q1 := (st.setReg "off" .nat [] (Tile.scalar off)).setReg "cols" .nat [B]
    (Tile.vec (fun j : Fin B => off + j.val)) with hq1
  have hc1 : q1.regs .nat [B] "cols"
      = some (Tile.vec (fun j : Fin B => off + j.val)) := by
    rw [hq1]; exact BlockState.setReg_same _ _ _ _ _
  have hp1 : q1.regs .ptr [] "X"
      = some (Tile.scalar (Region.cast Xr,
          s0.pids 0 * stride_x_N + s0.pids 1 * stride_x_hn)) := by
    rw [hq1]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("X" : RegName) ≠ "cols" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("X" : RegName) ≠ "off" by decide)]
    exact hXPtr
  refine Stmt.TraceSafeListR.cons_intro ?_ (fun t2 ht2 => ?_)
  · simp only [Stmt.TraceSafeR, Op.SafeAtR, MaskOpt.ActiveR,
      MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR,
      and_true, true_and, and_self]
    intro ptrs hptrs idx hactive
    rw [ln_addrR_eq, ln_addr_eval "X" B Xr _ q1 off hp1 hc1] at hptrs
    obtain rfl := Option.some.inj hptrs
    obtain ⟨masks, hm, hmi⟩ := hactive
    rw [ln_maskR_eq, ln_mask_eval N B q1 off hc1] at hm
    obtain rfl := Option.some.inj hm
    have hlt : off + idx.1.val < N := by simpa [Tile.vec] using hmi
    simpa [Region.cast_id, Tile.vec] using hbx idx.1 hlt
  · obtain ⟨v2, -, rfl⟩ := stepStmtR_assign_inv ht2
    exact Stmt.TraceSafeListR.cons_intro
      (by simp [Stmt.TraceSafeR, Op.SafeAtR])
      (fun _ _ => Stmt.TraceSafeListR.nil_intro)

set_option maxHeartbeats 4000000 in
/-- Per-iteration `TraceSafeListR` for the **variance pass** body: same walk
as the mean pass (the centring `tl.where` and the `_var` accumulate are
register-only). -/
private theorem ln_varBodySafeR (R : RoundingModel) (bounds : RegionBounds)
    (Xr : RegionName) (stride_x_N stride_x_hn N B : Nat)
    (s0 st : BlockState) (off : Nat)
    (hXPtr : st.regs .ptr [] "X"
      = some (Tile.scalar (Region.cast Xr,
          s0.pids 0 * stride_x_N + s0.pids 1 * stride_x_hn)))
    (hbx : ∀ j : Fin B, off + j.val < N →
      s0.pids 0 * stride_x_N + s0.pids 1 * stride_x_hn + (off + j.val)
        < bounds Xr) :
    Stmt.TraceSafeListR R bounds (layernormVarLoopBody N B)
      (st.setReg "off" .nat [] (Tile.scalar off)) := by
  unfold layernormVarLoopBody
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR]) (fun t1 ht1 => ?_)
  rw [stepStmtR_assign_eq_some
    ((ln_colsR_eq R B _).trans
      (ln_cols_eval B _ off (BlockState.setReg_same _ _ _ _ _)))] at ht1
  obtain rfl := Option.some.inj ht1
  set q1 := (st.setReg "off" .nat [] (Tile.scalar off)).setReg "cols" .nat [B]
    (Tile.vec (fun j : Fin B => off + j.val)) with hq1
  have hc1 : q1.regs .nat [B] "cols"
      = some (Tile.vec (fun j : Fin B => off + j.val)) := by
    rw [hq1]; exact BlockState.setReg_same _ _ _ _ _
  have hp1 : q1.regs .ptr [] "X"
      = some (Tile.scalar (Region.cast Xr,
          s0.pids 0 * stride_x_N + s0.pids 1 * stride_x_hn)) := by
    rw [hq1]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("X" : RegName) ≠ "cols" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("X" : RegName) ≠ "off" by decide)]
    exact hXPtr
  refine Stmt.TraceSafeListR.cons_intro ?_ (fun t2 ht2 => ?_)
  · simp only [Stmt.TraceSafeR, Op.SafeAtR, MaskOpt.ActiveR,
      MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR,
      and_true, true_and, and_self]
    intro ptrs hptrs idx hactive
    rw [ln_addrR_eq, ln_addr_eval "X" B Xr _ q1 off hp1 hc1] at hptrs
    obtain rfl := Option.some.inj hptrs
    obtain ⟨masks, hm, hmi⟩ := hactive
    rw [ln_maskR_eq, ln_mask_eval N B q1 off hc1] at hm
    obtain rfl := Option.some.inj hm
    have hlt : off + idx.1.val < N := by simpa [Tile.vec] using hmi
    simpa [Region.cast_id, Tile.vec] using hbx idx.1 hlt
  · obtain ⟨v2, -, rfl⟩ := stepStmtR_assign_inv ht2
    refine Stmt.TraceSafeListR.cons_intro
      (by simp [Stmt.TraceSafeR, Op.SafeAtR]) (fun t3 ht3 => ?_)
    obtain ⟨v3, -, rfl⟩ := stepStmtR_assign_inv ht3
    exact Stmt.TraceSafeListR.cons_intro
      (by simp [Stmt.TraceSafeR, Op.SafeAtR])
      (fun _ _ => Stmt.TraceSafeListR.nil_intro)


set_option maxHeartbeats 8000000 in
/-- Per-iteration `TraceSafeListR` for the **output pass** body: the `cols` /
`mask` / `x_hat` / `y` assigns are register-only; the masked `W` and `X`
loads' and the masked store's **active** lanes are the skin's `mask2` /
`mask1` / `writeMask` windows at step `off / B`, in bounds by the
corresponding window bounds (instantiated at the raw counter `off`). -/
private theorem ln_outBodySafeR (R : RoundingModel) (bounds : RegionBounds)
    (Xr Wr Yr : RegionName)
    (stride_x_N stride_x_hn stride_y_N stride_y_hn stride_w_hn N B : Nat)
    (s0 st : BlockState) (off : Nat)
    (hXPtr : st.regs .ptr [] "X"
      = some (Tile.scalar (Region.cast Xr,
          s0.pids 0 * stride_x_N + s0.pids 1 * stride_x_hn)))
    (hYPtr : st.regs .ptr [] "Y"
      = some (Tile.scalar (Region.cast Yr,
          s0.pids 0 * stride_y_N + s0.pids 1 * stride_y_hn)))
    (hWPtr : st.regs .ptr [] "W"
      = some (Tile.scalar (Region.cast Wr, s0.pids 1 * stride_w_hn)))
    (hbx : ∀ j : Fin B, off + j.val < N →
      s0.pids 0 * stride_x_N + s0.pids 1 * stride_x_hn + (off + j.val)
        < bounds Xr)
    (hbw : ∀ j : Fin B, off + j.val < N →
      s0.pids 1 * stride_w_hn + (off + j.val) < bounds Wr)
    (hbo : ∀ j : Fin B, off + j.val < N →
      s0.pids 0 * stride_y_N + s0.pids 1 * stride_y_hn + (off + j.val)
        < bounds Yr) :
    Stmt.TraceSafeListR R bounds (layernormOutLoopBody N B)
      (st.setReg "off" .nat [] (Tile.scalar off)) := by
  unfold layernormOutLoopBody
  -- cols
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR]) (fun t1 ht1 => ?_)
  rw [stepStmtR_assign_eq_some
    ((ln_colsR_eq R B _).trans
      (ln_cols_eval B _ off (BlockState.setReg_same _ _ _ _ _)))] at ht1
  obtain rfl := Option.some.inj ht1
  set q1 := (st.setReg "off" .nat [] (Tile.scalar off)).setReg "cols" .nat [B]
    (Tile.vec (fun j : Fin B => off + j.val)) with hq1
  have hc1 : q1.regs .nat [B] "cols"
      = some (Tile.vec (fun j : Fin B => off + j.val)) := by
    rw [hq1]; exact BlockState.setReg_same _ _ _ _ _
  have hx1 : q1.regs .ptr [] "X"
      = some (Tile.scalar (Region.cast Xr,
          s0.pids 0 * stride_x_N + s0.pids 1 * stride_x_hn)) := by
    rw [hq1]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("X" : RegName) ≠ "cols" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("X" : RegName) ≠ "off" by decide)]
    exact hXPtr
  have hy1 : q1.regs .ptr [] "Y"
      = some (Tile.scalar (Region.cast Yr,
          s0.pids 0 * stride_y_N + s0.pids 1 * stride_y_hn)) := by
    rw [hq1]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("Y" : RegName) ≠ "cols" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("Y" : RegName) ≠ "off" by decide)]
    exact hYPtr
  have hw1 : q1.regs .ptr [] "W"
      = some (Tile.scalar (Region.cast Wr, s0.pids 1 * stride_w_hn)) := by
    rw [hq1]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("W" : RegName) ≠ "cols" by decide),
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("W" : RegName) ≠ "off" by decide)]
    exact hWPtr
  -- mask
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR]) (fun t2 ht2 => ?_)
  rw [stepStmtR_assign_eq_some
    ((ln_maskR_eq R N B _).trans (ln_mask_eval N B q1 off hc1))] at ht2
  obtain rfl := Option.some.inj ht2
  set q2 := q1.setReg "mask" .bool [B]
    (Tile.vec (fun j : Fin B => decide (off + j.val < N))) with hq2
  have hm2 : q2.regs .bool [B] "mask"
      = some (Tile.vec (fun j : Fin B => decide (off + j.val < N))) := by
    rw [hq2]; exact BlockState.setReg_same _ _ _ _ _
  have hc2 : q2.regs .nat [B] "cols"
      = some (Tile.vec (fun j : Fin B => off + j.val)) := by
    rw [hq2]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
      (show ("cols" : RegName) ≠ "mask" by decide)]
    exact hc1
  have hx2 : q2.regs .ptr [] "X"
      = some (Tile.scalar (Region.cast Xr,
          s0.pids 0 * stride_x_N + s0.pids 1 * stride_x_hn)) := by
    rw [hq2]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
      (show ("X" : RegName) ≠ "mask" by decide)]
    exact hx1
  have hy2 : q2.regs .ptr [] "Y"
      = some (Tile.scalar (Region.cast Yr,
          s0.pids 0 * stride_y_N + s0.pids 1 * stride_y_hn)) := by
    rw [hq2]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
      (show ("Y" : RegName) ≠ "mask" by decide)]
    exact hy1
  have hw2 : q2.regs .ptr [] "W"
      = some (Tile.scalar (Region.cast Wr, s0.pids 1 * stride_w_hn)) := by
    rw [hq2]
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
      (show ("W" : RegName) ≠ "mask" by decide)]
    exact hw1
  -- the masked W load
  refine Stmt.TraceSafeListR.cons_intro ?_ (fun t3 ht3 => ?_)
  · simp only [Stmt.TraceSafeR, Op.SafeAtR, MaskOpt.ActiveR,
      MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR,
      and_true, true_and, and_self]
    intro ptrs hptrs idx hactive
    rw [ln_addrR_eq, ln_addr_eval "W" B Wr _ q2 off hw2 hc2] at hptrs
    obtain rfl := Option.some.inj hptrs
    obtain ⟨masks, hm, hmi⟩ := hactive
    rw [evalOpR_ref, hm2] at hm
    obtain rfl := Option.some.inj hm
    have hlt : off + idx.1.val < N := by simpa [Tile.vec] using hmi
    simpa [Region.cast_id, Tile.vec] using hbw idx.1 hlt
  · obtain ⟨v3, -, rfl⟩ := stepStmtR_assign_inv ht3
    set q3 := q2.setReg "w" .real [B] v3 with hq3
    have hm3 : q3.regs .bool [B] "mask"
        = some (Tile.vec (fun j : Fin B => decide (off + j.val < N))) := by
      rw [hq3]
      simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("mask" : RegName) ≠ "w" by decide)]
      exact hm2
    have hc3 : q3.regs .nat [B] "cols"
        = some (Tile.vec (fun j : Fin B => off + j.val)) := by
      rw [hq3]
      simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("cols" : RegName) ≠ "w" by decide)]
      exact hc2
    have hx3 : q3.regs .ptr [] "X"
        = some (Tile.scalar (Region.cast Xr,
            s0.pids 0 * stride_x_N + s0.pids 1 * stride_x_hn)) := by
      rw [hq3]
      simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("X" : RegName) ≠ "w" by decide)]
      exact hx2
    have hy3 : q3.regs .ptr [] "Y"
        = some (Tile.scalar (Region.cast Yr,
            s0.pids 0 * stride_y_N + s0.pids 1 * stride_y_hn)) := by
      rw [hq3]
      simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
        (show ("Y" : RegName) ≠ "w" by decide)]
      exact hy2
    -- the masked X load
    refine Stmt.TraceSafeListR.cons_intro ?_ (fun t4 ht4 => ?_)
    · simp only [Stmt.TraceSafeR, Op.SafeAtR, MaskOpt.ActiveR,
        MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR,
        and_true, true_and, and_self]
      intro ptrs hptrs idx hactive
      rw [ln_addrR_eq, ln_addr_eval "X" B Xr _ q3 off hx3 hc3] at hptrs
      obtain rfl := Option.some.inj hptrs
      obtain ⟨masks, hm, hmi⟩ := hactive
      rw [evalOpR_ref, hm3] at hm
      obtain rfl := Option.some.inj hm
      have hlt : off + idx.1.val < N := by simpa [Tile.vec] using hmi
      simpa [Region.cast_id, Tile.vec] using hbx idx.1 hlt
    · obtain ⟨v4, -, rfl⟩ := stepStmtR_assign_inv ht4
      set q4 := q3.setReg "x" .real [B] v4 with hq4
      -- x_hat (register-only)
      refine Stmt.TraceSafeListR.cons_intro
        (by simp [Stmt.TraceSafeR, Op.SafeAtR]) (fun t5 ht5 => ?_)
      obtain ⟨v5, -, rfl⟩ := stepStmtR_assign_inv ht5
      set q5 := q4.setReg "x_hat" .real [B] v5 with hq5
      -- y (register-only)
      refine Stmt.TraceSafeListR.cons_intro
        (by simp [Stmt.TraceSafeR, Op.SafeAtR]) (fun t6 ht6 => ?_)
      obtain ⟨v6, -, rfl⟩ := stepStmtR_assign_inv ht6
      set q6 := q5.setReg "y" .real [B] v6 with hq6
      have hm6 : q6.regs .bool [B] "mask"
          = some (Tile.vec (fun j : Fin B => decide (off + j.val < N))) := by
        rw [hq6, hq5, hq4]
        simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
            (show ("mask" : RegName) ≠ "y" by decide),
          BlockState.setReg_ne_name _ _ _ _ _ _ _ _
            (show ("mask" : RegName) ≠ "x_hat" by decide),
          BlockState.setReg_ne_name _ _ _ _ _ _ _ _
            (show ("mask" : RegName) ≠ "x" by decide)]
        exact hm3
      have hc6 : q6.regs .nat [B] "cols"
          = some (Tile.vec (fun j : Fin B => off + j.val)) := by
        rw [hq6, hq5, hq4]
        simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
            (show ("cols" : RegName) ≠ "y" by decide),
          BlockState.setReg_ne_name _ _ _ _ _ _ _ _
            (show ("cols" : RegName) ≠ "x_hat" by decide),
          BlockState.setReg_ne_name _ _ _ _ _ _ _ _
            (show ("cols" : RegName) ≠ "x" by decide)]
        exact hc3
      have hy6 : q6.regs .ptr [] "Y"
          = some (Tile.scalar (Region.cast Yr,
              s0.pids 0 * stride_y_N + s0.pids 1 * stride_y_hn)) := by
        rw [hq6, hq5, hq4]
        simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _
            (show ("Y" : RegName) ≠ "y" by decide),
          BlockState.setReg_ne_name _ _ _ _ _ _ _ _
            (show ("Y" : RegName) ≠ "x_hat" by decide),
          BlockState.setReg_ne_name _ _ _ _ _ _ _ _
            (show ("Y" : RegName) ≠ "x" by decide)]
        exact hy3
      -- the masked store
      refine Stmt.TraceSafeListR.cons_intro ?_
        (fun _ _ => Stmt.TraceSafeListR.nil_intro)
      simp only [Stmt.TraceSafeR, MemAccess.SafeAtR, MaskOpt.SafeAtR,
        Op.SafeAtR, MaskOpt.ActiveR, MemAccess.ActiveAddressSafeR,
        memAccessActiveAddressSafeR, and_true, true_and, and_self]
      intro ptrs hptrs idx hactive
      rw [ln_addrR_eq, ln_addr_eval "Y" B Yr _ q6 off hy6 hc6] at hptrs
      obtain rfl := Option.some.inj hptrs
      obtain ⟨masks, hm, hmi⟩ := hactive
      rw [evalOpR_ref, hm6] at hm
      obtain rfl := Option.some.inj hm
      have hlt : off + idx.1.val < N := by simpa [Tile.vec] using hmi
      simpa [Region.cast_id, Tile.vec] using hbo idx.1 hlt


set_option maxHeartbeats 8000000 in
/-- **The `TraceSafeR` walk for the whole three-pass kernel** — driven by
`Stmt.forRangeTraceSafeR_inv` over the exact stack's mean / variance /
output context invariants (each carried alongside the loop counter's
divisibility by the stride), with the counter advancing by the loops'
stride `BLOCK_SIZE`. The three bound groups are the skin's
`read1` / `read2` / `write` windows; the divisibility conjunct converts the
raw counter into the step index `off / B < ⌈N/B⌉` the windows are phrased
over, and `hXYNe` / `hWYNe` feed the output-pass step (its readback clause
needs the store not to clobber the streamed inputs). -/
private theorem layernorm_traceSafeR (R : RoundingModel) (bounds : RegionBounds)
    (X W Y : RegionName)
    (stride_x_N stride_x_hn stride_x_hd
      stride_y_N stride_y_hn stride_y_hd stride_w_hn stride_w_hd
      N BLOCK_SIZE : Nat) (eps : ℝ)
    (hBlockPos : 0 < BLOCK_SIZE) (hXYNe : X ≠ Y) (hWYNe : W ≠ Y)
    (s : BlockState)
    (hbx : ∀ (t : Fin (lnNumSteps N BLOCK_SIZE)) (j : Fin BLOCK_SIZE),
      t.val * BLOCK_SIZE + j.val < N →
      s.pids 0 * stride_x_N + s.pids 1 * stride_x_hn +
        (t.val * BLOCK_SIZE + j.val) < bounds X)
    (hbw : ∀ (t : Fin (lnNumSteps N BLOCK_SIZE)) (j : Fin BLOCK_SIZE),
      t.val * BLOCK_SIZE + j.val < N →
      s.pids 1 * stride_w_hn + (t.val * BLOCK_SIZE + j.val) < bounds W)
    (hbo : ∀ (t : Fin (lnNumSteps N BLOCK_SIZE)) (j : Fin BLOCK_SIZE),
      t.val * BLOCK_SIZE + j.val < N →
      s.pids 0 * stride_y_N + s.pids 1 * stride_y_hn +
        (t.val * BLOCK_SIZE + j.val) < bounds Y) :
    ((layernorm_fwd_triton X W Y stride_x_N stride_x_hn stride_x_hd
      stride_y_N stride_y_hn stride_y_hd stride_w_hn stride_w_hd
      N BLOCK_SIZE eps).toAlgKernel).TraceSafeR R bounds s := by
  have hStepNe : BLOCK_SIZE ≠ 0 := Nat.ne_of_gt hBlockPos
  -- window instantiators at a raw in-range counter
  have hstepT : ∀ i, BLOCK_SIZE ∣ i → ∀ j : Fin BLOCK_SIZE, i + j.val < N →
      ∃ t : Fin (lnNumSteps N BLOCK_SIZE),
        t.val * BLOCK_SIZE + j.val = i + j.val := by
    intro i hiB j hij
    have hiN : i < N := Nat.lt_of_le_of_lt (Nat.le_add_right _ _) hij
    refine ⟨⟨i / BLOCK_SIZE, lnStep_lt_numSteps N BLOCK_SIZE i hBlockPos hiN⟩, ?_⟩
    simp [Nat.div_mul_cancel hiB]
  have hbx' : ∀ i, BLOCK_SIZE ∣ i → ∀ j : Fin BLOCK_SIZE, i + j.val < N →
      s.pids 0 * stride_x_N + s.pids 1 * stride_x_hn + (i + j.val) < bounds X := by
    intro i hiB j hij
    obtain ⟨t, ht⟩ := hstepT i hiB j hij
    have h := hbx t j (by rw [ht]; exact hij)
    rwa [ht] at h
  have hbw' : ∀ i, BLOCK_SIZE ∣ i → ∀ j : Fin BLOCK_SIZE, i + j.val < N →
      s.pids 1 * stride_w_hn + (i + j.val) < bounds W := by
    intro i hiB j hij
    obtain ⟨t, ht⟩ := hstepT i hiB j hij
    have h := hbw t j (by rw [ht]; exact hij)
    rwa [ht] at h
  have hbo' : ∀ i, BLOCK_SIZE ∣ i → ∀ j : Fin BLOCK_SIZE, i + j.val < N →
      s.pids 0 * stride_y_N + s.pids 1 * stride_y_hn + (i + j.val) < bounds Y := by
    intro i hiB j hij
    obtain ⟨t, ht⟩ := hstepT i hiB j hij
    have h := hbo t j (by rw [ht]; exact hij)
    rwa [ht] at h
  unfold Kernel.TraceSafeR
  rw [layernorm_fwd_triton_toAlg_body]
  simp only [List.append_assoc, List.singleton_append]
  refine Stmt.TraceSafeListR.append_intro _ _ ?_ ?_
  · -- prologue: register-only assigns, safe at every state
    refine Stmt.TraceSafeListR.of_forall _ _ ?_
    intro stmt hst s'
    simp only [layernormMeanPreLoop, List.mem_cons, List.not_mem_nil,
      or_false] at hst
    rcases hst with rfl | rfl | rfl | rfl | rfl | rfl <;>
      simp [Stmt.TraceSafeR, Op.SafeAtR]
  · intro s1 hs1
    rw [lnPreLoop_castFree] at hs1
    refine Stmt.TraceSafeListR.cons_intro ?_ ?_
    · -- mean pass
      simp only [Stmt.TraceSafeR]
      refine Stmt.forRangeTraceSafeR_inv R bounds "off" N BLOCK_SIZE
        (layernormMeanLoopBody N BLOCK_SIZE)
        (fun off st => BLOCK_SIZE ∣ off ∧
          layernormMeanLoopContextInvariant s X stride_x_N stride_x_hn
            N BLOCK_SIZE off st) ?_ 0 s1
        ⟨⟨0, rfl⟩, layernormMeanLoopContextInvariant_init_of_preloop s s1 X W Y
          stride_x_N stride_x_hn stride_y_N stride_y_hn stride_w_hn N
          BLOCK_SIZE hs1⟩
      intro i stt hi hP
      refine ⟨ln_meanBodySafeR R bounds X stride_x_N stride_x_hn N BLOCK_SIZE
          s stt i hP.2.2.1 (fun j hj => hbx' i hP.1 j hj), ?_⟩
      obtain ⟨st', hstep, hP'⟩ :=
        layernormMeanLoopContextInvariant_body_step_exists s stt X stride_x_N
          stride_x_hn N BLOCK_SIZE i hP.2
      exact ⟨st', by rw [lnMeanBody_castFree]; exact hstep,
        Nat.dvd_add hP.1 dvd_rfl, hP'⟩
    · intro s2 hs2
      rw [stepStmtR_forRange,
        stepForRangeAuxR_castFree R _ (lnMeanBody_castFree R N BLOCK_SIZE) "off",
        ← stepForRangeAux.forRange_unfold] at hs2
      obtain ⟨meanFinal, hMeanFinal, hMeanCtx, hYPtrMean, hWPtrMean,
          hReadWMean⟩ :=
        layernormMeanForRange_context_ptrs_read_of_preloop s s1 s2 X W Y W
          stride_x_N stride_x_hn stride_y_N stride_y_hn stride_w_hn N
          BLOCK_SIZE hStepNe hs1 hs2
      refine Stmt.TraceSafeListR.append_intro _ _ ?_ ?_
      · refine Stmt.TraceSafeListR.of_forall _ _ ?_
        intro stmt hst s'
        simp only [layernormMeanPostLoop, List.mem_cons, List.not_mem_nil,
          or_false] at hst
        rcases hst with rfl | rfl <;> simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]
      · intro s3 hs3
        rw [lnMeanPost_castFree] at hs3
        have hVarInit :
            layernormVarLoopContextInvariant s X stride_x_N stride_x_hn
              N BLOCK_SIZE 0 s3 :=
          layernormMeanPostLoop_step_to_var_init s s2 s3 X stride_x_N
            stride_x_hn N BLOCK_SIZE meanFinal hMeanCtx hMeanFinal hBlockPos hs3
        obtain ⟨hYPtrVarInit, hWPtrVarInit, hReadWStep⟩ :=
          layernormMeanPostLoop_step_preserves_ptrs_read s2 s3 W Y W s
            stride_y_N stride_y_hn stride_w_hn N BLOCK_SIZE
            (layernormMeanAccumulatorSpec s X stride_x_N stride_x_hn
              N BLOCK_SIZE meanFinal) hMeanCtx.1.2 hYPtrMean hWPtrMean hs3
        have hReadWVarInit :
            ∀ offset, s3.readMem W offset = s.readMem W offset := by
          intro offset
          rw [hReadWStep offset, hReadWMean offset]
        refine Stmt.TraceSafeListR.cons_intro ?_ ?_
        · -- variance pass
          simp only [Stmt.TraceSafeR]
          refine Stmt.forRangeTraceSafeR_inv R bounds "off" N BLOCK_SIZE
            (layernormVarLoopBody N BLOCK_SIZE)
            (fun off st => BLOCK_SIZE ∣ off ∧
              layernormVarLoopContextInvariant s X stride_x_N stride_x_hn
                N BLOCK_SIZE off st) ?_ 0 s3 ⟨⟨0, rfl⟩, hVarInit⟩
          intro i stt hi hP
          refine ⟨ln_varBodySafeR R bounds X stride_x_N stride_x_hn N BLOCK_SIZE
              s stt i hP.2.2.1 (fun j hj => hbx' i hP.1 j hj), ?_⟩
          obtain ⟨st', hstep, hP'⟩ :=
            layernormVarLoopContextInvariant_body_step_exists s stt X stride_x_N
              stride_x_hn N BLOCK_SIZE i hP.2
          exact ⟨st', by rw [lnVarBody_castFree]; exact hstep,
            Nat.dvd_add hP.1 dvd_rfl, hP'⟩
        · intro s4 hs4
          rw [stepStmtR_forRange,
            stepForRangeAuxR_castFree R _ (lnVarBody_castFree R N BLOCK_SIZE)
              "off",
            ← stepForRangeAux.forRange_unfold] at hs4
          obtain ⟨varFinal, hVarFinal, hVarCtx, hYPtrVar, hWPtrVar,
              hReadWVar⟩ :=
            layernormVarForRange_context_ptrs_read_of_init s s3 s4 X W Y W
              stride_x_N stride_x_hn stride_y_N stride_y_hn stride_w_hn N
              BLOCK_SIZE hStepNe hVarInit hYPtrVarInit hWPtrVarInit
              hReadWVarInit hs4
          refine Stmt.TraceSafeListR.append_intro _ _ ?_ ?_
          · refine Stmt.TraceSafeListR.of_forall _ _ ?_
            intro stmt hst s'
            simp only [layernormVarPostLoop, List.mem_cons, List.not_mem_nil,
              or_false] at hst
            rcases hst with rfl | rfl <;> simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]
          · intro s5 hs5
            rw [lnVarPost_castFree] at hs5
            have hOutInit :
                layernormOutLoopContextInvariant s X W Y stride_x_N stride_x_hn
                  stride_y_N stride_y_hn stride_w_hn N BLOCK_SIZE 0 eps s5 :=
              layernormVarPostLoop_step_to_out_init s s4 s5 X W Y stride_x_N
                stride_x_hn stride_y_N stride_y_hn stride_w_hn N BLOCK_SIZE
                varFinal eps hVarCtx hVarFinal hBlockPos hYPtrVar hWPtrVar
                hReadWVar hs5
            refine Stmt.TraceSafeListR.cons_intro ?_
              (fun _ _ => Stmt.TraceSafeListR.nil_intro)
            -- output pass
            simp only [Stmt.TraceSafeR]
            refine Stmt.forRangeTraceSafeR_inv R bounds "off" N BLOCK_SIZE
              (layernormOutLoopBody N BLOCK_SIZE)
              (fun off st => BLOCK_SIZE ∣ off ∧
                layernormOutLoopContextInvariant s X W Y stride_x_N stride_x_hn
                  stride_y_N stride_y_hn stride_w_hn N BLOCK_SIZE off eps st)
              ?_ 0 s5 ⟨⟨0, rfl⟩, hOutInit⟩
            intro i stt hi hP
            obtain ⟨hOutInv, hXPtr, hYPtr, hWPtr, hMean, hRstd, hReadX,
              hReadW⟩ := hP.2
            refine ⟨ln_outBodySafeR R bounds X W Y stride_x_N stride_x_hn
                stride_y_N stride_y_hn stride_w_hn N BLOCK_SIZE s stt i
                hXPtr hYPtr hWPtr (fun j hj => hbx' i hP.1 j hj)
                (fun j hj => hbw' i hP.1 j hj)
                (fun j hj => hbo' i hP.1 j hj), ?_⟩
            obtain ⟨st', hstep, hP'⟩ :=
              layernormOutLoopContextInvariant_body_step_exists s stt X W Y
                stride_x_N stride_x_hn stride_y_N stride_y_hn stride_w_hn N
                BLOCK_SIZE i eps hP.2 hXYNe hWYNe
            exact ⟨st', by rw [lnOutBody_castFree]; exact hstep,
              Nat.dvd_add hP.1 dvd_rfl, hP'⟩


/-! ### Cell-level memory frames

The exact stack proves per-lane `readMem` values; the `⊨[R]` Hoare triple
additionally needs a per-**cell** frame, so the five register-only segments
get a generic assigns-don't-touch-memory lemma and the storing output-pass
body gets a cell-level frame twin of its readback step. -/

/-- A run of `assign` statements never touches memory. -/
private theorem ln_stepStmts_assigns_mem :
    ∀ (l : List Stmt),
      (∀ stmt ∈ l, ∃ dt sh nm, ∃ e : Op dt sh, stmt = Stmt.assign dt sh nm e) →
      ∀ {s s' : BlockState}, stepStmts l s = some s' → s'.mem = s.mem
  | [], _, s, s', h => by
      rw [stepStmts.nil] at h
      obtain rfl := Option.some.inj h
      rfl
  | stmt :: rest, hall, s, s', h => by
      obtain ⟨dt, sh, nm, e, rfl⟩ := hall _ List.mem_cons_self
      cases hv : evalOp e s with
      | none => simp [stepStmts, stepStmt, hv] at h
      | some v =>
          rw [stepStmts.cons_some (stepStmt_assign_eq_some hv)] at h
          rw [ln_stepStmts_assigns_mem rest
            (fun st' hst' => hall st' (List.mem_cons_of_mem _ hst')) h]
          rfl

/-- Cell-level frame of a `Prop`-masked exact `writeMem` scatter `foldl`:
every cell not hit by an active lane is untouched (the cell-level sibling of
the shared `scatter_prop_masked_preserves_other_offset`). -/
private theorem ln_foldl_writeMem_prop_preserve_cell {α : Type}
    {region : RegionName} (ofn : α → Nat) (vfn : α → ℝ) (P : α → Prop)
    [DecidablePred P] (r : RegionName) (oo : Nat) (l : List α) (s : BlockState)
    (hnot : ∀ k ∈ l, P k → ¬(r = region ∧ oo = ofn k)) :
    (l.foldl (fun acc k =>
        if P k then acc.writeMem region (ofn k) (vfn k) else acc) s).mem r oo
      = s.mem r oo := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
      rw [List.foldl_cons]
      by_cases hm : P hd
      · rw [if_pos hm,
          ih _ (fun k hk hmk => hnot k (List.mem_cons_of_mem hd hk) hmk),
          BlockState.writeMem_mem]
        exact if_neg (hnot hd List.mem_cons_self hm)
      · rw [if_neg hm]
        exact ih _ fun k hk hmk => hnot k (List.mem_cons_of_mem hd hk) hmk

set_option maxHeartbeats 8000000 in
/-- **Cell-level frame of one output-pass iteration** (the `mem` twin of
`layernormOutLoopBody_step_preserves_old_output`, same walk): from the
output-loop context pins, one storing body iteration leaves every cell off
the `{(Y, yColOffset · k) : k < N}` write window untouched — the masked
scatter store only hits active lanes `off + j < N`, whose offsets are
`yColOffset · (off + j)`. -/
private theorem ln_outBody_step_frame
    (s0 st st' : BlockState) (X W Y : RegionName)
    (stride_x_N stride_x_hn stride_y_N stride_y_hn stride_w_hn
      N BLOCK_SIZE off : Nat)
    (eps : ℝ)
    (hXPtr :
      st.regs .ptr [] "X" =
        some (Tile.scalar (Region.cast X,
          s0.pids 0 * stride_x_N + s0.pids 1 * stride_x_hn)))
    (hYPtr :
      st.regs .ptr [] "Y" =
        some (Tile.scalar (Region.cast Y,
          s0.pids 0 * stride_y_N + s0.pids 1 * stride_y_hn)))
    (hWPtr :
      st.regs .ptr [] "W" =
        some (Tile.scalar (Region.cast W, s0.pids 1 * stride_w_hn)))
    (hMean :
      st.regs .real [] "mean" =
        some (Tile.scalar (layernormMeanFullNSpec s0 X stride_x_N stride_x_hn
          N BLOCK_SIZE)))
    (hRstd :
      st.regs .real [] "rstd" =
        some (Tile.scalar (layernormRstdFullNSpec s0 X stride_x_N stride_x_hn
          N BLOCK_SIZE eps)))
    (hReadX : ∀ offset, st.readMem X offset = s0.readMem X offset)
    (hReadW : ∀ offset, st.readMem W offset = s0.readMem W offset)
    (hStep :
      stepStmts (layernormOutLoopBody N BLOCK_SIZE)
        (st.setReg "off" .nat [] (Tile.scalar off)) = some st')
    (r : RegionName) (oo : Nat)
    (hcond : r ≠ Y ∨
      ∀ i : Fin N, oo ≠ yColOffset s0 stride_y_N stride_y_hn i.val) :
    st'.mem r oo = st.mem r oo := by
  unfold layernormOutLoopBody at hStep
  simp [stepStmts, stepStmt, evalOp, evalOp.eq_def, hXPtr, hYPtr, hWPtr, hMean,
    hRstd, Tile.bop, Tile.ptrAdd, NumericDType.add, NumericDType.sub,
    NumericDType.mul, ComparableDType.lt, Option.bind, hReadX, hReadW,
    xColOffset, yColOffset, wColOffset] at hStep
  subst st'
  refine Eq.trans (ln_foldl_writeMem_prop_preserve_cell _ _ _ r oo _ _ ?_) ?_
  · intro lane _ hact hbad
    rcases hcond with hne | hno
    · exact hne hbad.1
    · exact hno ⟨off + lane.1.val, hact⟩ (by simpa [yColOffset] using hbad.2)
  · rfl


/-! ### The rounded Hoare triple (`hrun`) -/

set_option maxHeartbeats 8000000 in
/-- Termination, per-lane values and the per-cell frame of the whole
three-pass kernel under `execR R`, from an **arbitrary** launch state: the
exact mean / variance / output invariant stack runs verbatim (all three
passes are cast-free, so `execR R` collapses onto the exact stepper),
extended with the per-segment memory frames. -/
private theorem layernorm_runR (R : RoundingModel) (X W Y : RegionName)
    (stride_x_N stride_x_hn stride_x_hd
      stride_y_N stride_y_hn stride_y_hd stride_w_hn stride_w_hd
      N BLOCK_SIZE : Nat) (eps : ℝ)
    (hBlockPos : 0 < BLOCK_SIZE) (hXYNe : X ≠ Y) (hWYNe : W ≠ Y)
    (s₀ : BlockState) :
    ∃ sfin,
      execR R (layernorm_fwd_triton X W Y stride_x_N stride_x_hn stride_x_hd
        stride_y_N stride_y_hn stride_y_hd stride_w_hn stride_w_hd
        N BLOCK_SIZE eps).toAlgKernel s₀ = some sfin
      ∧ (∀ i : Fin N,
          sfin.readMem Y (yColOffset s₀ stride_y_N stride_y_hn i.val)
            = layernormYFullNSpec s₀ X W stride_x_N stride_x_hn stride_w_hn
                N BLOCK_SIZE eps i)
      ∧ (∀ r oo, (r ≠ Y ∨
            ∀ i : Fin N, oo ≠ yColOffset s₀ stride_y_N stride_y_hn i.val) →
          sfin.mem r oo = s₀.mem r oo) := by
  have hStepNe : BLOCK_SIZE ≠ 0 := Nat.ne_of_gt hBlockPos
  -- prologue
  obtain ⟨sPre, hPre⟩ :
      ∃ sPre, stepStmts (layernormMeanPreLoop X W Y stride_x_N stride_x_hn
        stride_y_N stride_y_hn stride_w_hn BLOCK_SIZE) s₀ = some sPre := by
    cases h : stepStmts (layernormMeanPreLoop X W Y stride_x_N stride_x_hn
        stride_y_N stride_y_hn stride_w_hn BLOCK_SIZE) s₀ with
    | none =>
        unfold layernormMeanPreLoop at h
        simp [stepStmts, stepStmt, evalOp, evalOp.eq_def, Tile.bop, Tile.ptrAdd,
          NumericDType.add, NumericDType.mul, Option.bind] at h
    | some sPre => exact ⟨sPre, rfl⟩
  have hPreMem : sPre.mem = s₀.mem :=
    ln_stepStmts_assigns_mem _
      (by
        intro stmt hst
        simp only [layernormMeanPreLoop, List.mem_cons, List.not_mem_nil,
          or_false] at hst
        rcases hst with rfl | rfl | rfl | rfl | rfl | rfl <;>
          exact ⟨_, _, _, _, rfl⟩)
      hPre
  -- mean pass, with the memory frame carried alongside the context invariant
  obtain ⟨meanFinal, sMean, hMeanLoop, hMeanFinal, hMeanCtx, hMeanMem⟩ :=
    forRange_inv (idx := "off") (start := 0) (stop := N) (step := BLOCK_SIZE)
      (body := layernormMeanLoopBody N BLOCK_SIZE)
      (P := fun off st =>
        layernormMeanLoopContextInvariant s₀ X stride_x_N stride_x_hn
          N BLOCK_SIZE off st ∧ st.mem = s₀.mem)
      hStepNe
      ⟨layernormMeanLoopContextInvariant_init_of_preloop s₀ sPre X W Y
        stride_x_N stride_x_hn stride_y_N stride_y_hn stride_w_hn N BLOCK_SIZE
        hPre, hPreMem⟩
      (fun i stt hlt hP => by
        obtain ⟨st', hstep, hinv'⟩ :=
          layernormMeanLoopContextInvariant_body_step_exists s₀ stt X
            stride_x_N stride_x_hn N BLOCK_SIZE i hP.1
        refine ⟨st', hstep, hinv', ?_⟩
        have hmm : st'.mem
            = (stt.setReg "off" .nat [] (Tile.scalar i)).mem :=
          ln_stepStmts_assigns_mem (layernormMeanLoopBody N BLOCK_SIZE)
            (by
              intro stmt hst
              simp only [layernormMeanLoopBody, List.mem_cons,
                List.not_mem_nil, or_false] at hst
              rcases hst with rfl | rfl | rfl <;> exact ⟨_, _, _, _, rfl⟩)
            hstep
        rw [hmm]
        exact hP.2)
  obtain ⟨_meanFinal', _hMeanFinal', _hMeanCtx', hYPtrMean, hWPtrMean,
      hReadWMean⟩ :=
    layernormMeanForRange_context_ptrs_read_of_preloop s₀ sPre sMean X W Y W
      stride_x_N stride_x_hn stride_y_N stride_y_hn stride_w_hn N BLOCK_SIZE
      hStepNe hPre hMeanLoop
  -- mean reduce / `_var` init
  obtain ⟨sVarInit, hMeanPost⟩ :
      ∃ sv, stepStmts (layernormMeanPostLoop N BLOCK_SIZE) sMean = some sv := by
    cases h : stepStmts (layernormMeanPostLoop N BLOCK_SIZE) sMean with
    | none =>
        unfold layernormMeanPostLoop at h
        simp [stepStmts, stepStmt, evalOp, evalOp.eq_def, hMeanCtx.1.2, Tile.bop,
          Tile.reduceSum, Tile.reduceSumDrop, TileShape.axisDim,
          TileShape.eraseAxis, TileShape.insertAxisIndex, NumericDType.div,
          Option.bind] at h
    | some sv => exact ⟨sv, rfl⟩
  have hVarInitMem : sVarInit.mem = s₀.mem := by
    rw [ln_stepStmts_assigns_mem (layernormMeanPostLoop N BLOCK_SIZE)
      (by
        intro stmt hst
        simp only [layernormMeanPostLoop, List.mem_cons, List.not_mem_nil,
          or_false] at hst
        rcases hst with rfl | rfl <;> exact ⟨_, _, _, _, rfl⟩)
      hMeanPost]
    exact hMeanMem
  have hVarInit :
      layernormVarLoopContextInvariant s₀ X stride_x_N stride_x_hn
        N BLOCK_SIZE 0 sVarInit :=
    layernormMeanPostLoop_step_to_var_init s₀ sMean sVarInit X stride_x_N
      stride_x_hn N BLOCK_SIZE meanFinal hMeanCtx hMeanFinal hBlockPos hMeanPost
  obtain ⟨hYPtrVarInit, hWPtrVarInit, hReadWStep⟩ :=
    layernormMeanPostLoop_step_preserves_ptrs_read sMean sVarInit W Y W s₀
      stride_y_N stride_y_hn stride_w_hn N BLOCK_SIZE
      (layernormMeanAccumulatorSpec s₀ X stride_x_N stride_x_hn N BLOCK_SIZE
        meanFinal) hMeanCtx.1.2 hYPtrMean hWPtrMean hMeanPost
  have hReadWVarInit : ∀ offset, sVarInit.readMem W offset = s₀.readMem W offset := by
    intro offset
    rw [hReadWStep offset, hReadWMean offset]
  -- variance pass
  obtain ⟨varFinal, sVar, hVarLoop, hVarFinal, hVarCtx, hVarMem⟩ :=
    forRange_inv (idx := "off") (start := 0) (stop := N) (step := BLOCK_SIZE)
      (body := layernormVarLoopBody N BLOCK_SIZE)
      (P := fun off st =>
        layernormVarLoopContextInvariant s₀ X stride_x_N stride_x_hn
          N BLOCK_SIZE off st ∧ st.mem = s₀.mem)
      hStepNe ⟨hVarInit, hVarInitMem⟩
      (fun i stt hlt hP => by
        obtain ⟨st', hstep, hinv'⟩ :=
          layernormVarLoopContextInvariant_body_step_exists s₀ stt X
            stride_x_N stride_x_hn N BLOCK_SIZE i hP.1
        refine ⟨st', hstep, hinv', ?_⟩
        have hmm : st'.mem
            = (stt.setReg "off" .nat [] (Tile.scalar i)).mem :=
          ln_stepStmts_assigns_mem (layernormVarLoopBody N BLOCK_SIZE)
            (by
              intro stmt hst
              simp only [layernormVarLoopBody, List.mem_cons,
                List.not_mem_nil, or_false] at hst
              rcases hst with rfl | rfl | rfl | rfl <;> exact ⟨_, _, _, _, rfl⟩)
            hstep
        rw [hmm]
        exact hP.2)
  obtain ⟨_varFinal', _hVarFinal', _hVarCtx', hYPtrVar, hWPtrVar, hReadWVar⟩ :=
    layernormVarForRange_context_ptrs_read_of_init s₀ sVarInit sVar X W Y W
      stride_x_N stride_x_hn stride_y_N stride_y_hn stride_w_hn N BLOCK_SIZE
      hStepNe hVarInit hYPtrVarInit hWPtrVarInit hReadWVarInit hVarLoop
  -- variance reduce / `rstd`
  obtain ⟨sOutInit, hVarPost⟩ :
      ∃ so, stepStmts (layernormVarPostLoop N BLOCK_SIZE eps) sVar = some so := by
    cases h : stepStmts (layernormVarPostLoop N BLOCK_SIZE eps) sVar with
    | none =>
        unfold layernormVarPostLoop at h
        simp [stepStmts, stepStmt, evalOp, evalOp.eq_def, hVarCtx.1.2, Tile.bop,
          Tile.uop, Tile.reduceSum, Tile.reduceSumDrop, TileShape.axisDim,
          TileShape.eraseAxis, TileShape.insertAxisIndex, NumericDType.add,
          NumericDType.div, Option.bind, WithBot.realSqrt] at h
    | some so => exact ⟨so, rfl⟩
  have hOutInitMem : sOutInit.mem = s₀.mem := by
    rw [ln_stepStmts_assigns_mem (layernormVarPostLoop N BLOCK_SIZE eps)
      (by
        intro stmt hst
        simp only [layernormVarPostLoop, List.mem_cons, List.not_mem_nil,
          or_false] at hst
        rcases hst with rfl | rfl <;> exact ⟨_, _, _, _, rfl⟩)
      hVarPost]
    exact hVarMem
  have hOutInit :
      layernormOutLoopContextInvariant s₀ X W Y stride_x_N stride_x_hn
        stride_y_N stride_y_hn stride_w_hn N BLOCK_SIZE 0 eps sOutInit :=
    layernormVarPostLoop_step_to_out_init s₀ sVar sOutInit X W Y stride_x_N
      stride_x_hn stride_y_N stride_y_hn stride_w_hn N BLOCK_SIZE varFinal eps
      hVarCtx hVarFinal hBlockPos hYPtrVar hWPtrVar hReadWVar hVarPost
  -- output pass, with the conditional cell frame carried alongside
  obtain ⟨outFinal, sOut, hOutLoop, hOutFinal, _hOutCtx, hOutFrame⟩ :=
    forRange_inv (idx := "off") (start := 0) (stop := N) (step := BLOCK_SIZE)
      (body := layernormOutLoopBody N BLOCK_SIZE)
      (P := fun off st =>
        layernormOutLoopContextInvariant s₀ X W Y stride_x_N stride_x_hn
          stride_y_N stride_y_hn stride_w_hn N BLOCK_SIZE off eps st ∧
        ∀ r oo, (r ≠ Y ∨
            ∀ i : Fin N, oo ≠ yColOffset s₀ stride_y_N stride_y_hn i.val) →
          st.mem r oo = s₀.mem r oo)
      hStepNe ⟨hOutInit, fun r oo _ => by rw [hOutInitMem]⟩
      (fun i stt hlt hQ => by
        obtain ⟨st', hstep, hinv'⟩ :=
          layernormOutLoopContextInvariant_body_step_exists s₀ stt X W Y
            stride_x_N stride_x_hn stride_y_N stride_y_hn stride_w_hn N
            BLOCK_SIZE i eps hQ.1 hXYNe hWYNe
        refine ⟨st', hstep, hinv', ?_⟩
        intro r oo hcond
        obtain ⟨hOutInv, hXPtr, hYPtr, hWPtr, hMean, hRstd, hReadX, hReadW⟩ :=
          hQ.1
        rw [ln_outBody_step_frame s₀ stt st' X W Y stride_x_N stride_x_hn
          stride_y_N stride_y_hn stride_w_hn N BLOCK_SIZE i eps hXPtr hYPtr
          hWPtr hMean hRstd hReadX hReadW hstep r oo hcond]
        exact hQ.2 r oo hcond)
  -- assemble the `execR` run through the cast-free collapses
  have hMeanLoopR : stepStmtR R (Stmt.forRange "off" 0 N BLOCK_SIZE
      (layernormMeanLoopBody N BLOCK_SIZE)) sPre = some sMean := by
    rw [stepStmtR_forRange,
      stepForRangeAuxR_castFree R _ (lnMeanBody_castFree R N BLOCK_SIZE) "off",
      ← stepForRangeAux.forRange_unfold]
    exact hMeanLoop
  have hVarLoopR : stepStmtR R (Stmt.forRange "off" 0 N BLOCK_SIZE
      (layernormVarLoopBody N BLOCK_SIZE)) sVarInit = some sVar := by
    rw [stepStmtR_forRange,
      stepForRangeAuxR_castFree R _ (lnVarBody_castFree R N BLOCK_SIZE) "off",
      ← stepForRangeAux.forRange_unfold]
    exact hVarLoop
  have hOutLoopR : stepStmtR R (Stmt.forRange "off" 0 N BLOCK_SIZE
      (layernormOutLoopBody N BLOCK_SIZE)) sOutInit = some sOut := by
    rw [stepStmtR_forRange,
      stepForRangeAuxR_castFree R _ (lnOutBody_castFree R N BLOCK_SIZE) "off",
      ← stepForRangeAux.forRange_unfold]
    exact hOutLoop
  refine ⟨sOut, ?_, ?_, hOutFrame⟩
  · unfold execR
    rw [layernorm_fwd_triton_toAlg_body]
    simp only [List.append_assoc, List.singleton_append, List.cons_append,
      List.nil_append]
    rw [stepStmtsR_append R (layernormMeanPreLoop X W Y stride_x_N stride_x_hn
        stride_y_N stride_y_hn stride_w_hn BLOCK_SIZE) _ s₀,
      lnPreLoop_castFree, hPre, Option.bind_some,
      stepStmtsR_cons_some hMeanLoopR,
      stepStmtsR_append R (layernormMeanPostLoop N BLOCK_SIZE) _ sMean,
      lnMeanPost_castFree, hMeanPost, Option.bind_some,
      stepStmtsR_cons_some hVarLoopR,
      stepStmtsR_append R (layernormVarPostLoop N BLOCK_SIZE eps) _ sVar,
      lnVarPost_castFree, hVarPost, Option.bind_some,
      stepStmtsR_cons_some hOutLoopR, stepStmtsR_nil]
  · exact layernorm_fwd_triton_staged_fullN_correct_from_preloop s₀ sPre sMean
      sVarInit sVar sOutInit sOut X W Y stride_x_N stride_x_hn stride_y_N
      stride_y_hn stride_w_hn N BLOCK_SIZE eps hBlockPos hXYNe hWYNe hPre
      hMeanLoop hMeanPost hVarLoop hVarPost hOutLoop


/-! ### The headline -/

set_option maxHeartbeats 4000000 in
/-- **The `⊨[R]` streaming headline (wave-5 S3 per-step emit genre).** For
every rounding model `R`, the faithful `layernorm_fwd_triton` surface
implements, on its `StreamEmitMasked2DKernelIO₂` signature, the **ideal ℝ
three-pass LayerNorm** over the streamed tiles: emitted window `(t, j)`
holds `((x[t,j] − mean) · (√(var + eps))⁻¹) · w[t,j]`, where
`mean = (Σ_guarded x)/N` and `var = (Σ_guarded (x − mean)²)/N` are guarded
double sums folding the *entire* `x` stream — the spec `f` is exact real
arithmetic. The kernel has **zero rounding events**: every load, all three
passes' arithmetic and the per-step stores are at `.real`, and the Python
`.to(tl.float32)` input casts and the store cast `(y).to(X.dtype.element_ty)`
are erased by the lowering (`layernormOutLoopBody`'s emitted statement is a
plain `Stmt.store .real`, so the `outDType := .real` default is the honest
grid, not a modelling shortcut). The skin's boundary quantization therefore
degenerates: the readback contract's `R.round .real` is the identity by the
model's defining `round_real`, and the `.real` in-loop stores are exact
under `execR R` — the ∀-`R` face holds via the `RoundingModel` `.real`
identity fields, not as a `.triv` special case.

Layer map: all three passes are cast-free (`lnMeanBody_castFree`,
`lnVarBody_castFree`, `lnOutBody_castFree` and the three register-only
segments), so under `execR R` they collapse verbatim onto the exact stepper
and the proven mean / variance / output invariant stack above
(`layernormMeanLoopContextInvariant`, `layernormVarLoopContextInvariant`,
`layernormOutLoopContextInvariant`, closed by
`layernorm_fwd_triton_staged_fullN_correct_from_preloop`) is reused
unchanged; the `⊨[R]` face adds only the `TraceSafeR` walk
(`layernorm_traceSafeR`), the per-cell memory frame
(`ln_outBody_step_frame`, the `mem` twin of
`layernormOutLoopBody_step_preserves_old_output`), and the stream-lane spec
bridge (`ln_reblock` re-blocking `k ↔ (k/B, k%B)`).

All three hypotheses are truth-forced — they are exactly the exact
headline `layernorm_fwd_triton_output_summary`'s side conditions:

* `hBlockPos : 0 < BLOCK_SIZE` — all three loops step by `BLOCK_SIZE`
  (`range(0, N, BLOCK_SIZE)`); at `BLOCK_SIZE = 0` and `N > 0` no `forRange`
  advances, `execR` cannot terminate the way the invariant stack requires,
  and the step index `off / BLOCK_SIZE` is meaningless. It holds for every
  real launch.
* `hXYNe : X ≠ Y`, `hWYNe : W ≠ Y` — the output pass stores into `Y`
  **between** its re-reads of `X` and `W` (each iteration reloads both from
  the *same* base pointers); if `Y` aliased either input, later blocks would
  re-read already-overwritten values and the closed form would be false.

No further hypothesis is needed. In particular the emit window is
injective in the global lane with **no** side condition: `write` is
`pid₀·stride_y_N + pid₁·stride_y_hn + k`, whose dependence on `k` is a bare
`+ k` (the file's `yColOffset_injective`), so a nonzero column stride is not
required here — unlike the strided-output members of this genre.

Relation to the exact surface: the exact headline
`layernorm_fwd_triton_output_summary` (`Realizes_without_Rounding`) above is
retained unchanged; this `⊨[R]` face restates the same LayerNorm content on
the streaming emit skin, for every `R` at once (at the `.real` grid the two
faces carry the same exact cell). Both faces are kept per the
rounding-as-default doctrine. -/
specification layernorm_fwd_triton_io_correctness (R : RoundingModel)
    (X W Y : RegionName)
    (stride_x_N stride_x_hn stride_x_hd
      stride_y_N stride_y_hn stride_y_hd stride_w_hn stride_w_hd
      N BLOCK_SIZE : Nat) (eps : ℝ)
    (hBlockPos : 0 < BLOCK_SIZE) (hXYNe : X ≠ Y) (hWYNe : W ≠ Y) :
    layernormKernelIO X W Y stride_x_N stride_x_hn stride_x_hd
      stride_y_N stride_y_hn stride_y_hd stride_w_hn stride_w_hd
      N BLOCK_SIZE eps ⊨[R]
      fun _ _ xs ws t j => lnStreamSpec N BLOCK_SIZE eps xs ws t j := by
  refine StreamEmitMasked2DKernelIO₂.ImplementsR.intro _ ?_ ?_ ?_
  · exact layernorm_fwd_triton_flattenOk X W Y stride_x_N stride_x_hn
      stride_x_hd stride_y_N stride_y_hn stride_y_hd stride_w_hn stride_w_hd
      N BLOCK_SIZE eps
  · -- safety walk
    intro bounds s xs ws _hx _hw hbr1 hbr2 hbw
    simp only [layernormKernelIO] at hbr1 hbr2 hbw ⊢
    exact layernorm_traceSafeR R bounds X W Y stride_x_N stride_x_hn
      stride_x_hd stride_y_N stride_y_hn stride_y_hd stride_w_hn stride_w_hd
      N BLOCK_SIZE eps hBlockPos hXYNe hWYNe s hbr1 hbr2 hbw
  · -- the rounded Hoare triple
    intro s₀ xs ws _hundef hx hw
    simp only [layernormKernelIO] at hx hw ⊢
    obtain ⟨sfin, hexec, hval, hframe⟩ :=
      layernorm_runR R X W Y stride_x_N stride_x_hn stride_x_hd stride_y_N
        stride_y_hn stride_y_hd stride_w_hn stride_w_hd N BLOCK_SIZE eps
        hBlockPos hXYNe hWYNe s₀
    refine ⟨sfin, hexec, ?_, ?_⟩
    · intro t j hj
      have hk : t.val * BLOCK_SIZE + j.val < N := hj
      have hval' : sfin.readMem Y (s₀.pids 0 * stride_y_N +
            s₀.pids 1 * stride_y_hn + (t.val * BLOCK_SIZE + j.val))
          = layernormYFullNSpec s₀ X W stride_x_N stride_x_hn stride_w_hn
              N BLOCK_SIZE eps ⟨t.val * BLOCK_SIZE + j.val, hk⟩ :=
        hval ⟨t.val * BLOCK_SIZE + j.val, hk⟩
      rw [BlockState.readMemAs_real, hval',
        ← lnStreamSpec_eq_layernormYFullNSpec X W s₀ stride_x_N stride_x_hn
          stride_w_hn N BLOCK_SIZE eps hBlockPos xs ws hx hw t j hj]
      simp [FloatDType.ofReal]
    · intro r oo hcond
      refine hframe r oo ?_
      rcases hcond with hne | hno
      · exact Or.inl hne
      · refine Or.inr fun i => ?_
        have hdm : i.val / BLOCK_SIZE * BLOCK_SIZE + i.val % BLOCK_SIZE
            = i.val := by
          rw [Nat.mul_comm]
          exact Nat.div_add_mod i.val BLOCK_SIZE
        have h := hno
          ⟨i.val / BLOCK_SIZE,
            lnStep_lt_numSteps N BLOCK_SIZE i.val hBlockPos i.isLt⟩
          ⟨i.val % BLOCK_SIZE, Nat.mod_lt _ hBlockPos⟩ (by simp [hdm, i.isLt])
        simpa [hdm, yColOffset] using h

end IOFace

end VeriTile.Bench.TritonBenchG.LayernormFwdTriton
