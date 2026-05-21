import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL
import VeriTile.Triton.LoopInvariant

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

noncomputable def layernormOutChunkValue
    (s : BlockState) (X W : RegionName)
    (stride_x_N stride_x_hn stride_w_hn N BLOCK_SIZE : Nat)
    (eps : ℝ) (off : Nat) (lane : TileIndex [BLOCK_SIZE]) : ℝ :=
  if h : off + lane.1.val < N then
    layernormYFullNSpec s X W stride_x_N stride_x_hn stride_w_hn
      N BLOCK_SIZE eps ⟨off + lane.1.val, h⟩
  else
    0

noncomputable def layernormOutChunkStoreState
    (s0 st : BlockState) (X W Y : RegionName)
    (stride_x_N stride_x_hn stride_y_N stride_y_hn stride_w_hn
      N BLOCK_SIZE : Nat)
    (eps : ℝ) (off : Nat) : BlockState :=
  (TileShape.allIndices [BLOCK_SIZE]).foldl
    (fun acc lane =>
      if off + lane.1.val < N then
        acc.writeMem Y
          (yColOffset s0 stride_y_N stride_y_hn (off + lane.1.val))
          (layernormOutChunkValue s0 X W stride_x_N stride_x_hn stride_w_hn
            N BLOCK_SIZE eps off lane)
      else
        acc)
    st

theorem yColOffset_injective
    (s : BlockState) (stride_y_N stride_y_hn N : Nat) :
    Function.Injective
      (fun i : Fin N => yColOffset s stride_y_N stride_y_hn i.val) := by
  intro a b h
  apply Fin.ext
  unfold yColOffset at h
  exact Nat.add_left_cancel h

theorem layernormOutChunkStore_read_current
    (s0 st : BlockState) (X W Y : RegionName)
    (stride_x_N stride_x_hn stride_y_N stride_y_hn stride_w_hn
      N BLOCK_SIZE : Nat)
    (eps : ℝ) (off : Nat) (i : Fin BLOCK_SIZE)
    (hActive : off + i.val < N)
    (hNoCollision :
      ∀ lane : TileIndex [BLOCK_SIZE],
        off + lane.1.val < N →
          yColOffset s0 stride_y_N stride_y_hn (off + lane.1.val) =
            yColOffset s0 stride_y_N stride_y_hn (off + i.val) →
          lane = ((i, PUnit.unit) : TileIndex [BLOCK_SIZE])) :
    (layernormOutChunkStoreState s0 st X W Y stride_x_N stride_x_hn
        stride_y_N stride_y_hn stride_w_hn N BLOCK_SIZE eps off).readMem
      Y (yColOffset s0 stride_y_N stride_y_hn (off + i.val)) =
      layernormYFullNSpec s0 X W stride_x_N stride_x_hn stride_w_hn
        N BLOCK_SIZE eps ⟨off + i.val, hActive⟩ := by
  unfold layernormOutChunkStoreState
  rw [BlockState.scatter_readback_prop_masked_nd_of_true
    (region := Y)
    (s := st)
    (offsetFn := fun lane : TileIndex [BLOCK_SIZE] =>
      yColOffset s0 stride_y_N stride_y_hn (off + lane.1.val))
    (valueFn := fun lane : TileIndex [BLOCK_SIZE] =>
      layernormOutChunkValue s0 X W stride_x_N stride_x_hn stride_w_hn
        N BLOCK_SIZE eps off lane)
    (P := fun lane : TileIndex [BLOCK_SIZE] => off + lane.1.val < N)
    ((i, PUnit.unit) : TileIndex [BLOCK_SIZE]) hActive hNoCollision]
  simp [layernormOutChunkValue, hActive]

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

theorem layernormOutChunkStore_preserve_old
    (s0 st : BlockState) (X W Y : RegionName)
    (stride_x_N stride_x_hn stride_y_N stride_y_hn stride_w_hn
      N BLOCK_SIZE : Nat)
    (eps : ℝ) (off : Nat) (i : Fin N)
    (hOld : i.val < off) :
    (layernormOutChunkStoreState s0 st X W Y stride_x_N stride_x_hn
        stride_y_N stride_y_hn stride_w_hn N BLOCK_SIZE eps off).readMem
      Y (yColOffset s0 stride_y_N stride_y_hn i.val) =
      st.readMem Y (yColOffset s0 stride_y_N stride_y_hn i.val) := by
  unfold layernormOutChunkStoreState
  rw [BlockState.scatter_prop_masked_preserves_other_offset]
  intro lane hActive hEq
  have hFinEq :
      (⟨off + lane.1.val, hActive⟩ : Fin N) = i := by
    apply yColOffset_injective s0 stride_y_N stride_y_hn N
    simpa using hEq
  have hVal : off + lane.1.val = i.val := congrArg Fin.val hFinEq
  omega

theorem layernormOutChunkStore_noCollision
    (s0 : BlockState)
    (stride_y_N stride_y_hn N BLOCK_SIZE off : Nat)
    (i : Fin BLOCK_SIZE)
    (hActive : off + i.val < N) :
    ∀ lane : TileIndex [BLOCK_SIZE],
      off + lane.1.val < N →
        yColOffset s0 stride_y_N stride_y_hn (off + lane.1.val) =
          yColOffset s0 stride_y_N stride_y_hn (off + i.val) →
        lane = ((i, PUnit.unit) : TileIndex [BLOCK_SIZE]) := by
  intro lane hLane hEq
  have hFinEq :
      (⟨off + lane.1.val, hLane⟩ : Fin N) =
        ⟨off + i.val, hActive⟩ := by
    apply yColOffset_injective s0 stride_y_N stride_y_hn N
    simpa using hEq
  have hLaneVal : lane.1 = i := by
    apply Fin.ext
    have hVal : off + lane.1.val = off + i.val := congrArg Fin.val hFinEq
    omega
  cases lane with
  | mk laneHead laneTail =>
      cases laneTail
      cases hLaneVal
      rfl

theorem layernormOutLoopInvariant_step_of_chunk_store
    (s0 st : BlockState) (X W Y : RegionName)
    (stride_x_N stride_x_hn stride_y_N stride_y_hn stride_w_hn
      N BLOCK_SIZE off : Nat)
    (eps : ℝ)
    (hInv : layernormOutLoopInvariant s0 X W Y stride_x_N stride_x_hn
      stride_y_N stride_y_hn stride_w_hn N BLOCK_SIZE eps off st) :
    layernormOutLoopInvariant s0 X W Y stride_x_N stride_x_hn
      stride_y_N stride_y_hn stride_w_hn N BLOCK_SIZE eps
      (off + BLOCK_SIZE)
      (layernormOutChunkStoreState s0 st X W Y stride_x_N stride_x_hn
        stride_y_N stride_y_hn stride_w_hn N BLOCK_SIZE eps off) := by
  intro col hWritten
  by_cases hOld : col.val < off
  · rw [layernormOutChunkStore_preserve_old s0 st X W Y stride_x_N stride_x_hn
      stride_y_N stride_y_hn stride_w_hn N BLOCK_SIZE eps off col hOld]
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
    have hRead := layernormOutChunkStore_read_current s0 st X W Y
      stride_x_N stride_x_hn stride_y_N stride_y_hn stride_w_hn
      N BLOCK_SIZE eps off lane hLaneActive
      (layernormOutChunkStore_noCollision s0 stride_y_N stride_y_hn
        N BLOCK_SIZE off lane hLaneActive)
    simpa [hcolEq] using hRead

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
    ComputeCorrect.Realizes
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
    ComputeCorrect.Realizes
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
end VeriTile.Bench.TritonBenchG.LayernormFwdTriton
