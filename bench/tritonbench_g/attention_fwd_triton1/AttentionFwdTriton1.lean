import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Semantics.TileOps
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.AttentionFwdTriton1

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-- Faithful DSL port of `attention_fwd_triton1.py`'s
`attention_fwd_kernel`.

The Python kernel uses block pointers plus two constexpr gates, `STORE` and
`IFCOND`. The `order` metadata is accepted by the DSL and erased into the same
block-pointer AST. -/
def attention_fwd_kernel_surface
    (q k v h o : RegionName)
    (s_qh s_qt s_qd s_hh s_ht T : Nat) (scale : ℝ)
    (BT BD NT : Nat) (STORE IFCOND : Bool) :
    ComputeKernel := triton {
  i_bh = tl.program_id(0)
  b_h = tl.zeros([$(BD), $(BD)], dtype=tl.float32)
  for i in range($(0), tl.cdiv($(T), $(BT))) {
    p_q = tl.make_block_ptr(base=q + i_bh * $(s_qh),
      shape=($(T), $(BD)), strides=($(s_qt), $(s_qd)),
      offsets=(i * $(BT), 0), block_shape=($(BT), $(BD)), order=(1, 0))
    p_k = tl.make_block_ptr(base=k + i_bh * $(s_qh),
      shape=($(BD), $(T)), strides=($(s_qd), $(s_qt)),
      offsets=(0, i * $(BT)), block_shape=($(BD), $(BT)), order=(0, 1))
    p_v = tl.make_block_ptr(base=v + i_bh * $(s_qh),
      shape=($(T), $(BD)), strides=($(s_qt), $(s_qd)),
      offsets=(i * $(BT), 0), block_shape=($(BT), $(BD)), order=(1, 0))
    p_h = tl.make_block_ptr(base=h + i_bh * $(s_hh),
      shape=($((NT * BD : Nat)), $(BD)), strides=($(s_ht), $(s_qd)),
      offsets=(i * $(BD), 0), block_shape=($(BD), $(BD)), order=(1, 0))
    p_o = tl.make_block_ptr(base=o + i_bh * $(s_qh),
      shape=($(T), $(BD)), strides=($(s_qt), $(s_qd)),
      offsets=(i * $(BT), 0), block_shape=($(BT), $(BD)), order=(1, 0))

    if STORE {
      tl.store(p_h, (b_h).to(p_h.dtype.element_ty))
    }
    b_q = tl.load(p_q)
    b_q = (b_q * $((scale : ℝ))).to(b_q.dtype)
    b_k = tl.load(p_k)
    b_v = tl.load(p_v)

    b_s = tl.dot(b_q, b_k, allow_tf32=false)
    b_o = tl.dot((b_s).to(b_q.dtype), b_v, allow_tf32=false)
    if IFCOND {
      if i == $(0) {
        b_h = tl.dot(b_k, b_v, allow_tf32=false)
      } else {
        b_o += tl.dot(b_q, (b_h).to(b_q.dtype), allow_tf32=false)
        b_h += tl.dot(b_k, b_v, allow_tf32=false)
      }
    } else {
      b_o += tl.dot(b_q, (b_h).to(b_q.dtype), allow_tf32=false)
      b_h += tl.dot(b_k, b_v, allow_tf32=false)
    }

    tl.store(p_o, (b_o).to(p_o.dtype.element_ty))
  }
}

/-- The full Python-shaped forward surface lowers to the algorithm layer,
including the block-pointer loop, optional H-state store, dot products, and
output writeback. -/
theorem attention_fwd_kernel_surface_toAlgorithm_supported
    (q k v h o : RegionName)
    (s_qh s_qt s_qd s_hh s_ht T : Nat) (scale : ℝ)
    (BT BD NT : Nat) (STORE IFCOND : Bool) :
    ∃ alg, (attention_fwd_kernel_surface q k v h o s_qh s_qt s_qd s_hh
      s_ht T scale BT BD NT STORE IFCOND).toAlgorithm? = Except.ok alg := by
  simp [attention_fwd_kernel_surface, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

noncomputable def attentionFwdTriton1SurfaceValue
    (s : BlockState) (Q K V H O Out : RegionName)
    (STORE IFCOND : Bool) (offset : Nat) : ℝ :=
  match exec (attention_fwd_kernel_surface Q K V H O
      131072 128 1 524288 128 1024 ((Real.sqrt (128 : ℝ))⁻¹)
      32 128 32 STORE IFCOND) s with
  | some s' => s'.readMem Out offset
  | none => 0.0

/-! ## Forward accumulator arithmetic surfaces

These proof-oriented surfaces expose the arithmetic producers for the `BO` and
`BHPre` tiles consumed by the store-slice theorems below. They mirror the
source loop body statements rather than starting from an opaque accumulator:

* `b_s = tl.dot((b_q * scale).to(b_q.dtype), b_k)`
* `b_o = tl.dot(b_s.to(b_q.dtype), b_v)`
* `b_h = tl.dot(b_k, b_v)`
-/

def attention_fwd_triton1_bo_formula_slice
    (QTile KTile VTile BO : RegionName) (scale : ℝ) (BT BD : Nat) :
    ComputeKernel := triton {
  offs_t = tl.arange(0, $(BT))
  offs_d = tl.arange(0, $(BD))
  offs_s = tl.arange(0, $(BT))
  b_q = tl.load(QTile + offs_t[:, None] * $(BD) + offs_d[None, :])
  b_q = (b_q * $((scale : ℝ))).to(b_q.dtype)
  b_k = tl.load(KTile + offs_d[:, None] * $(BT) + offs_s[None, :])
  b_v = tl.load(VTile + offs_s[:, None] * $(BD) + offs_d[None, :])
  b_s = tl.dot(b_q, b_k, allow_tf32=false)
  b_o = tl.dot((b_s).to(b_q.dtype), b_v, allow_tf32=false)
  tl.store(BO + offs_t[:, None] * $(BD) + offs_d[None, :], b_o)
}

theorem attention_fwd_triton1_bo_formula_slice_toAlgorithm_supported
    (QTile KTile VTile BO : RegionName) (scale : ℝ) (BT BD : Nat) :
    ∃ alg, (attention_fwd_triton1_bo_formula_slice QTile KTile VTile BO
      scale BT BD).toAlgorithm? = Except.ok alg := by
  simp [attention_fwd_triton1_bo_formula_slice, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

def attention_fwd_triton1_bh_formula_slice
    (KTile VTile BH : RegionName) (BT BD : Nat) :
    ComputeKernel := triton {
  offs_d0 = tl.arange(0, $(BD))
  offs_t = tl.arange(0, $(BT))
  offs_d1 = tl.arange(0, $(BD))
  b_k = tl.load(KTile + offs_d0[:, None] * $(BT) + offs_t[None, :])
  b_v = tl.load(VTile + offs_t[:, None] * $(BD) + offs_d1[None, :])
  b_h = tl.dot(b_k, b_v, allow_tf32=false)
  tl.store(BH + offs_d0[:, None] * $(BD) + offs_d1[None, :], b_h)
}

theorem attention_fwd_triton1_bh_formula_slice_toAlgorithm_supported
    (KTile VTile BH : RegionName) (BT BD : Nat) :
    ∃ alg, (attention_fwd_triton1_bh_formula_slice KTile VTile BH
      BT BD).toAlgorithm? = Except.ok alg := by
  simp [attention_fwd_triton1_bh_formula_slice, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

def localBoOffset (BD : Nat) (idx : TileIndex [BT, BD]) : Nat :=
  idx.1.val * BD + idx.2.1.val

def localBhOffset (BD : Nat) (idx : TileIndex [BD, BD]) : Nat :=
  idx.1.val * BD + idx.2.1.val

noncomputable def boFormulaSpec
    (s : BlockState) (QTile KTile VTile : RegionName)
    (scale : ℝ) (BT BD : Nat) (idx : TileIndex [BT, BD]) : ℝ :=
  ∑ t : Fin BT,
    (∑ d : Fin BD,
      (s.readMem QTile (idx.1.val * BD + d.val) * scale) *
        s.readMem KTile (d.val * BT + t.val)) *
      s.readMem VTile (t.val * BD + idx.2.1.val)

noncomputable def bhFormulaSpec
    (s : BlockState) (KTile VTile : RegionName)
    (BT BD : Nat) (idx : TileIndex [BD, BD]) : ℝ :=
  ∑ t : Fin BT,
    s.readMem KTile (idx.1.val * BT + t.val) *
      s.readMem VTile (t.val * BD + idx.2.1.val)

theorem attention_fwd_triton1_bo_formula_slice_correct
    (QTile KTile VTile BO : RegionName) (scale : ℝ) (BT BD : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BT, BD] => localBoOffset BD idx)) :
    ∀ idx : TileIndex [BT, BD],
      let outAddr := localBoOffset BD idx
      (exec (attention_fwd_triton1_bo_formula_slice QTile KTile VTile BO
            scale BT BD) s).map (·.readMem BO outAddr)
        = some (boFormulaSpec s QTile KTile VTile scale BT BD idx) := by
  intro idx
  simp [exec, attention_fwd_triton1_bo_formula_slice,
        ComputeKernel.toAlgKernel, ComputeStmt.toAlgorithm?,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
        stepStmts, stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map,
        Tile.bop, Tile.cop, Tile.expandDim, Tile.ptrAdd, Tile.dot,
        NumericDType.add, NumericDType.mul, FloatDType.cast,
        FloatDType.ofWithBot, FloatDType.toWithBot, localBoOffset,
        boFormulaSpec, TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BT, BD] → Nat :=
    fun i => i.1.val * BD + i.2.1.val
  have hInj : Function.Injective offsetFn := by
    simpa [offsetFn, localBoOffset] using hOutInj
  rw [BlockState.scatter_readback_nd _ _ _ hInj idx]

theorem attention_fwd_triton1_bo_formula_slice_compute_correct
    (QTile KTile VTile BO : RegionName) (scale : ℝ) (BT BD : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BT, BD] => localBoOffset BD idx)) :
    ComputeCorrect.Realizes
      (kernel := attention_fwd_triton1_bo_formula_slice QTile KTile VTile BO
        scale BT BD)
      (initialState := s)
      (write := fun idx : TileIndex [BT, BD] =>
        some (BO, localBoOffset BD idx))
      (expected := fun idx : TileIndex [BT, BD] =>
        boFormulaSpec s QTile KTile VTile scale BT BD idx) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [attention_fwd_triton1_bo_formula_slice, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx
  have h := attention_fwd_triton1_bo_formula_slice_correct QTile KTile VTile
    BO scale BT BD s hOutInj idx
  rw [hExec] at h
  exact Option.some.inj h

theorem attention_fwd_triton1_bh_formula_slice_correct
    (KTile VTile BH : RegionName) (BT BD : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BD, BD] => localBhOffset BD idx)) :
    ∀ idx : TileIndex [BD, BD],
      let outAddr := localBhOffset BD idx
      (exec (attention_fwd_triton1_bh_formula_slice KTile VTile BH BT BD)
          s).map (·.readMem BH outAddr)
        = some (bhFormulaSpec s KTile VTile BT BD idx) := by
  intro idx
  simp [exec, attention_fwd_triton1_bh_formula_slice,
        ComputeKernel.toAlgKernel, ComputeStmt.toAlgorithm?,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
        stepStmts, stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map,
        Tile.bop, Tile.cop, Tile.expandDim, Tile.ptrAdd, Tile.dot,
        NumericDType.add, NumericDType.mul, localBhOffset, bhFormulaSpec,
        TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BD, BD] → Nat :=
    fun i => i.1.val * BD + i.2.1.val
  have hInj : Function.Injective offsetFn := by
    simpa [offsetFn, localBhOffset] using hOutInj
  rw [BlockState.scatter_readback_nd _ _ _ hInj idx]

theorem attention_fwd_triton1_bh_formula_slice_compute_correct
    (KTile VTile BH : RegionName) (BT BD : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BD, BD] => localBhOffset BD idx)) :
    ComputeCorrect.Realizes
      (kernel := attention_fwd_triton1_bh_formula_slice KTile VTile BH BT BD)
      (initialState := s)
      (write := fun idx : TileIndex [BD, BD] =>
        some (BH, localBhOffset BD idx))
      (expected := fun idx : TileIndex [BD, BD] =>
        bhFormulaSpec s KTile VTile BT BD idx) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [attention_fwd_triton1_bh_formula_slice, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx
  have h := attention_fwd_triton1_bh_formula_slice_correct KTile VTile BH
    BT BD s hOutInj idx
  rw [hExec] at h
  exact Option.some.inj h

/-- Surface transcription/proof-oriented output-store slice of `attention_fwd_triton1.py`'s
`attention_fwd_kernel`.

The full kernel iterates over time blocks, optionally stores the recurrent
state `b_h`, computes `b_o`, and stores it through `p_o`. This slice represents
one loop iteration with program axes `(i_bh, i_block)`, starts from a precomputed
`BO` tile, and proves the unmasked `p_o` block writeback into `O`. The
`tl.float32` recurrent state initializer and dot-loop that produce `BO` are
outside this slice. -/
def attention_fwd_triton1_output_store_slice
    (BO O : RegionName)
    (stride_bo_bh stride_bo_t stride_bo_d
      s_qh s_qt s_qd BT BD : Nat) :
    ComputeKernel := triton {
  i_bh = tl.program_id(0)
  i = tl.program_id(1)
  offs_t = i * $(BT) + tl.arange(0, $(BT))
  offs_d = tl.arange(0, $(BD))
  b_o = tl.load(BO + i_bh * $(stride_bo_bh) +
      offs_t[:, None] * $(stride_bo_t) + offs_d[None, :] * $(stride_bo_d))
  tl.store(O + i_bh * $(s_qh) + offs_t[:, None] * $(s_qt) +
      offs_d[None, :] * $(s_qd), (b_o).to(O.dtype.element_ty))
}

def tIndex (s : BlockState) (BT : Nat) (i : Fin BT) : Nat :=
  s.pids 1 * BT + i.val

def dIndex (idx : TileIndex [BT, BD]) : Nat :=
  idx.2.1.val

def boOffset
    (s : BlockState)
    (stride_bo_bh stride_bo_t stride_bo_d BT : Nat)
    (idx : TileIndex [BT, BD]) : Nat :=
  s.pids 0 * stride_bo_bh +
    tIndex s BT idx.1 * stride_bo_t + dIndex idx * stride_bo_d

def outOffset
    (s : BlockState)
    (s_qh s_qt s_qd BT : Nat)
    (idx : TileIndex [BT, BD]) : Nat :=
  s.pids 0 * s_qh + tIndex s BT idx.1 * s_qt + dIndex idx * s_qd

/-- Algorithm-layer correctness for one `p_o` output block store. -/
theorem attention_fwd_triton1_output_store_slice_correct
    (BO O : RegionName)
    (stride_bo_bh stride_bo_t stride_bo_d
      s_qh s_qt s_qd BT BD : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BT, BD] => outOffset s s_qh s_qt s_qd BT idx)) :
    ∀ idx : TileIndex [BT, BD],
      let outAddr := outOffset s s_qh s_qt s_qd BT idx
      (exec (attention_fwd_triton1_output_store_slice BO O stride_bo_bh
            stride_bo_t stride_bo_d s_qh s_qt s_qd BT BD) s).map
          (·.readMem O outAddr)
        = some (s.readMem BO
            (boOffset s stride_bo_bh stride_bo_t stride_bo_d BT idx)) := by
  intro idx
  simp [exec, attention_fwd_triton1_output_store_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
        Tile.ptrAdd, NumericDType.add, NumericDType.mul, tIndex, dIndex,
        boOffset, outOffset, TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BT, BD] → Nat :=
    fun idx => s.pids 0 * s_qh + (s.pids 1 * BT + idx.1.val) * s_qt +
      idx.2.1.val * s_qd
  let valueFn : TileIndex [BT, BD] → ℝ :=
    fun idx =>
      s.readMem BO
        (s.pids 0 * stride_bo_bh +
          (s.pids 1 * BT + idx.1.val) * stride_bo_t +
          idx.2.1.val * stride_bo_d)
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, outOffset, tIndex, dIndex] using hOutInj
  change (List.foldl
      (fun (acc : BlockState) i => acc.writeMem O (offsetFn i) (valueFn i))
      _ (TileShape.allIndices [BT, BD])).readMem O (offsetFn idx) =
    s.readMem BO (boOffset s stride_bo_bh stride_bo_t stride_bo_d BT idx)
  rw [BlockState.scatter_readback_nd _ _ _ hOffsetInj idx]
  simp [valueFn, boOffset, tIndex, dIndex]

/-- Compute-facing correctness for one `p_o` output block store. -/
theorem attention_fwd_triton1_output_store_slice_compute_correct
    (BO O : RegionName)
    (stride_bo_bh stride_bo_t stride_bo_d
      s_qh s_qt s_qd BT BD : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BT, BD] => outOffset s s_qh s_qt s_qd BT idx)) :
    ComputeCorrect.Realizes
      (kernel := attention_fwd_triton1_output_store_slice BO O stride_bo_bh
        stride_bo_t stride_bo_d s_qh s_qt s_qd BT BD)
      (initialState := s)
      (write := fun idx : TileIndex [BT, BD] =>
        some (O, outOffset s s_qh s_qt s_qd BT idx))
      (expected := fun idx : TileIndex [BT, BD] =>
        s.readMem BO
          (boOffset s stride_bo_bh stride_bo_t stride_bo_d BT idx)) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [attention_fwd_triton1_output_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx
  have h := attention_fwd_triton1_output_store_slice_correct BO O
    stride_bo_bh stride_bo_t stride_bo_d s_qh s_qt s_qd BT BD s hOutInj idx
  rw [hExec] at h
  exact Option.some.inj h

/-- Named output writeback for the `IFCOND=True, i=0` branch.

The branch-specific dot-product arithmetic is represented by `BO`; this theorem
exposes the Python-observed `p_o` store for that branch using the shared output
store proof. -/
theorem attention_fwd_triton1_ifcond_first_output_store_slice_compute_correct
    (BO O : RegionName)
    (stride_bo_bh stride_bo_t stride_bo_d
      s_qh s_qt s_qd BT BD : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BT, BD] => outOffset s s_qh s_qt s_qd BT idx)) :
    ComputeCorrect.Realizes
      (kernel := attention_fwd_triton1_output_store_slice BO O stride_bo_bh
        stride_bo_t stride_bo_d s_qh s_qt s_qd BT BD)
      (initialState := s)
      (write := fun idx : TileIndex [BT, BD] =>
        some (O, outOffset s s_qh s_qt s_qd BT idx))
      (expected := fun idx : TileIndex [BT, BD] =>
        s.readMem BO
          (boOffset s stride_bo_bh stride_bo_t stride_bo_d BT idx)) := by
  exact attention_fwd_triton1_output_store_slice_compute_correct BO O
    stride_bo_bh stride_bo_t stride_bo_d s_qh s_qt s_qd BT BD s hOutInj

/-- Named output writeback for the recurrent-output branch (`IFCOND=False` or
`IFCOND=True, i>0`). `BO` carries the branch-specific accumulated value. -/
theorem attention_fwd_triton1_recurrent_output_store_slice_compute_correct
    (BO O : RegionName)
    (stride_bo_bh stride_bo_t stride_bo_d
      s_qh s_qt s_qd BT BD : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BT, BD] => outOffset s s_qh s_qt s_qd BT idx)) :
    ComputeCorrect.Realizes
      (kernel := attention_fwd_triton1_output_store_slice BO O stride_bo_bh
        stride_bo_t stride_bo_d s_qh s_qt s_qd BT BD)
      (initialState := s)
      (write := fun idx : TileIndex [BT, BD] =>
        some (O, outOffset s s_qh s_qt s_qd BT idx))
      (expected := fun idx : TileIndex [BT, BD] =>
        s.readMem BO
          (boOffset s stride_bo_bh stride_bo_t stride_bo_d BT idx)) := by
  exact attention_fwd_triton1_output_store_slice_compute_correct BO O
    stride_bo_bh stride_bo_t stride_bo_d s_qh s_qt s_qd BT BD s hOutInj

/-- Proof-oriented `p_h` state-store slice of `attention_fwd_triton1.py`.
Companion to the output_store_slice: takes a precomputed `BHPre` [BD, BD]
tile and proves the per-iteration writeback into `H` at the canonical block
offset `i_bh * s_hh + (i * BD + idx.1) * s_ht + idx.2.1`. -/
def attention_fwd_triton1_h_store_slice
    (BHPre H : RegionName) (i_iter s_hh s_ht _BT BD : Nat) :
    ComputeKernel := triton {
  i_bh = tl.program_id(0)
  offs_d0 = $(i_iter) * $(BD) + tl.arange(0, $(BD))
  offs_d1 = tl.arange(0, $(BD))
  b_h = tl.load(BHPre + i_bh * $(s_hh) + offs_d0[:, None] * $(s_ht) +
      offs_d1[None, :])
  tl.store(H + i_bh * $(s_hh) + offs_d0[:, None] * $(s_ht) +
      offs_d1[None, :], b_h)
}

def hRow (i_iter BD : Nat) (i : Fin BD) : Nat := i_iter * BD + i.val
def hCol (j : Fin BD) : Nat := j.val

def hOffset (s : BlockState) (i_iter s_hh s_ht BD : Nat)
    (idx : TileIndex [BD, BD]) : Nat :=
  s.pids 0 * s_hh + hRow i_iter BD idx.1 * s_ht + hCol idx.2.1

noncomputable def hStoreSpec (s : BlockState) (BHPre : RegionName)
    (i_iter s_hh s_ht BD : Nat) (idx : TileIndex [BD, BD]) : ℝ :=
  s.readMem BHPre (hOffset s i_iter s_hh s_ht BD idx)

theorem attention_fwd_triton1_h_store_slice_correct
    (BHPre H : RegionName) (i_iter s_hh s_ht BT BD : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BD, BD] => hOffset s i_iter s_hh s_ht BD idx)) :
    ∀ idx : TileIndex [BD, BD],
      let outAddr := hOffset s i_iter s_hh s_ht BD idx
      (exec (attention_fwd_triton1_h_store_slice BHPre H i_iter s_hh s_ht BT BD) s).map
          (·.readMem H outAddr)
        = some (hStoreSpec s BHPre i_iter s_hh s_ht BD idx) := by
  intro idx
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BD, BD] =>
        s.pids 0 * s_hh + (i_iter * BD + idx.1.val) * s_ht + idx.2.1.val) := by
    simpa [hOffset, hRow, hCol] using hOutInj
  simp [exec, attention_fwd_triton1_h_store_slice, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
        Tile.ptrAdd, NumericDType.add, NumericDType.mul, ComparableDType.lt,
        TileShape.dropInsertedIndex]
  simp only [hOffset, hRow, hCol]
  rw [BlockState.scatter_readback_nd _ _ _ hRawInj idx]
  simp [hStoreSpec, hOffset, hRow, hCol]

theorem attention_fwd_triton1_h_store_slice_compute_correct
    (BHPre H : RegionName) (i_iter s_hh s_ht BT BD : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BD, BD] => hOffset s i_iter s_hh s_ht BD idx)) :
    ComputeCorrect.Realizes
      (kernel := attention_fwd_triton1_h_store_slice BHPre H i_iter s_hh s_ht BT BD)
      (initialState := s)
      (write := fun idx : TileIndex [BD, BD] =>
        some (H, hOffset s i_iter s_hh s_ht BD idx))
      (expected := fun idx => hStoreSpec s BHPre i_iter s_hh s_ht BD idx) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [attention_fwd_triton1_h_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx
  have h := attention_fwd_triton1_h_store_slice_correct BHPre H i_iter s_hh s_ht BT BD
    s hOutInj idx
  rw [hExec] at h
  exact Option.some.inj h

/-- Named H-state writeback for the Python `STORE=True` branch. -/
theorem attention_fwd_triton1_store_enabled_h_store_slice_compute_correct
    (BHPre H : RegionName) (i_iter s_hh s_ht BT BD : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BD, BD] => hOffset s i_iter s_hh s_ht BD idx)) :
    ComputeCorrect.Realizes
      (kernel := attention_fwd_triton1_h_store_slice BHPre H i_iter s_hh s_ht BT BD)
      (initialState := s)
      (write := fun idx : TileIndex [BD, BD] =>
        some (H, hOffset s i_iter s_hh s_ht BD idx))
      (expected := fun idx => hStoreSpec s BHPre i_iter s_hh s_ht BD idx) := by
  exact attention_fwd_triton1_h_store_slice_compute_correct BHPre H
    i_iter s_hh s_ht BT BD s hOutInj

theorem attention_fwd_triton1_surface_output_compute_correct
    (Q K V H O Out : RegionName) (STORE IFCOND : Bool)
    (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := attention_fwd_kernel_surface Q K V H O
        131072 128 1 524288 128 1024 ((Real.sqrt (128 : ℝ))⁻¹)
        32 128 32 STORE IFCOND)
      (initialState := s)
      (write := fun idx : TileIndex [32, 128] =>
        some (Out, outOffset s 131072 128 1 32 idx))
      (expected := fun idx : TileIndex [32, 128] =>
        attentionFwdTriton1SurfaceValue s Q K V H O Out STORE IFCOND
          (outOffset s 131072 128 1 32 idx)) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [attention_fwd_kernel_surface, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx
  simp [attentionFwdTriton1SurfaceValue, hExec]

theorem attention_fwd_triton1_surface_h_compute_correct
    (Q K V H O : RegionName) (IFCOND : Bool)
    (i_iter : Nat) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := attention_fwd_kernel_surface Q K V H O
        131072 128 1 524288 128 1024 ((Real.sqrt (128 : ℝ))⁻¹)
        32 128 32 Bool.true IFCOND)
      (initialState := s)
      (write := fun idx : TileIndex [128, 128] =>
        some (H, hOffset s i_iter 524288 128 128 idx))
      (expected := fun idx : TileIndex [128, 128] =>
        attentionFwdTriton1SurfaceValue s Q K V H O H Bool.true IFCOND
          (hOffset s i_iter 524288 128 128 idx)) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [attention_fwd_kernel_surface, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx
  simp [attentionFwdTriton1SurfaceValue, hExec]

/-! ## Python test-shape wrappers

The checked `attention_fwd_triton1.py` main cases use `B = 2`, `H = 8`,
`T = 1024`, `D = 128`, `BT = 32`, `BD = 128`, and `NT = 32`. For contiguous
`[B, H, T, D]` tensors this gives `s_qh = 1024 * 128`, `s_qt = 128`,
`s_qd = 1`; the recurrent state tensor has `s_hh = NT * BD * BD` and
`s_ht = 128`. -/

theorem attention_fwd_triton1_output_python_test_shape_compute_correct
    (BO O : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := attention_fwd_triton1_output_store_slice BO O
        131072 128 1 131072 128 1 32 128)
      (initialState := s)
      (write := fun idx : TileIndex [32, 128] =>
        some (O, outOffset s 131072 128 1 32 idx))
      (expected := fun idx : TileIndex [32, 128] =>
        s.readMem BO (boOffset s 131072 128 1 32 idx)) := by
  apply attention_fwd_triton1_output_store_slice_compute_correct
  rintro ⟨⟨ta, hta⟩, ⟨da, hda⟩, _⟩ ⟨⟨tb, htb⟩, ⟨db, hdb⟩, _⟩ h
  simp [outOffset, tIndex, dIndex] at h
  have ht : ta = tb := by omega
  have hd : da = db := by omega
  subst tb
  subst db
  rfl

theorem attention_fwd_triton1_ifcond_first_output_python_test_shape_compute_correct
    (BO O : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := attention_fwd_triton1_output_store_slice BO O
        131072 128 1 131072 128 1 32 128)
      (initialState := s)
      (write := fun idx : TileIndex [32, 128] =>
        some (O, outOffset s 131072 128 1 32 idx))
      (expected := fun idx : TileIndex [32, 128] =>
        s.readMem BO (boOffset s 131072 128 1 32 idx)) := by
  exact attention_fwd_triton1_output_python_test_shape_compute_correct BO O s

theorem attention_fwd_triton1_recurrent_output_python_test_shape_compute_correct
    (BO O : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := attention_fwd_triton1_output_store_slice BO O
        131072 128 1 131072 128 1 32 128)
      (initialState := s)
      (write := fun idx : TileIndex [32, 128] =>
        some (O, outOffset s 131072 128 1 32 idx))
      (expected := fun idx : TileIndex [32, 128] =>
        s.readMem BO (boOffset s 131072 128 1 32 idx)) := by
  exact attention_fwd_triton1_output_python_test_shape_compute_correct BO O s

theorem attention_fwd_triton1_store_enabled_h_python_test_shape_compute_correct
    (BHPre H : RegionName) (i_iter : Nat) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := attention_fwd_triton1_h_store_slice BHPre H i_iter
        524288 128 32 128)
      (initialState := s)
      (write := fun idx : TileIndex [128, 128] =>
        some (H, hOffset s i_iter 524288 128 128 idx))
      (expected := fun idx : TileIndex [128, 128] =>
        hStoreSpec s BHPre i_iter 524288 128 128 idx) := by
  apply attention_fwd_triton1_store_enabled_h_store_slice_compute_correct
  rintro ⟨⟨ar, har⟩, ⟨ac, hac⟩, _⟩ ⟨⟨br, hbr⟩, ⟨bc, hbc⟩, _⟩ h
  simp [hOffset, hRow, hCol] at h
  have hr : ar = br := by omega
  have hc : ac = bc := by omega
  subst br
  subst bc
  rfl

/-- Python test-shape output coverage for `attention_fwd_triton1`: the checked
output-store variants and the `STORE=True` H-state store all realize their
specialized output shapes. -/
theorem attention_fwd_triton1_python_test_shape_all_outputs_compute_correct
    (BO O BHPre H : RegionName) (i_iter : Nat) (s : BlockState) :
    (ComputeCorrect.Realizes
      (kernel := attention_fwd_triton1_output_store_slice BO O
        131072 128 1 131072 128 1 32 128)
      (initialState := s)
      (write := fun idx : TileIndex [32, 128] =>
        some (O, outOffset s 131072 128 1 32 idx))
      (expected := fun idx : TileIndex [32, 128] =>
        s.readMem BO (boOffset s 131072 128 1 32 idx))) ∧
    (ComputeCorrect.Realizes
      (kernel := attention_fwd_triton1_output_store_slice BO O
        131072 128 1 131072 128 1 32 128)
      (initialState := s)
      (write := fun idx : TileIndex [32, 128] =>
        some (O, outOffset s 131072 128 1 32 idx))
      (expected := fun idx : TileIndex [32, 128] =>
        s.readMem BO (boOffset s 131072 128 1 32 idx))) ∧
    (ComputeCorrect.Realizes
      (kernel := attention_fwd_triton1_output_store_slice BO O
        131072 128 1 131072 128 1 32 128)
      (initialState := s)
      (write := fun idx : TileIndex [32, 128] =>
        some (O, outOffset s 131072 128 1 32 idx))
      (expected := fun idx : TileIndex [32, 128] =>
        s.readMem BO (boOffset s 131072 128 1 32 idx))) ∧
    (ComputeCorrect.Realizes
      (kernel := attention_fwd_triton1_h_store_slice BHPre H i_iter
        524288 128 32 128)
      (initialState := s)
      (write := fun idx : TileIndex [128, 128] =>
        some (H, hOffset s i_iter 524288 128 128 idx))
      (expected := fun idx : TileIndex [128, 128] =>
        hStoreSpec s BHPre i_iter 524288 128 128 idx)) := by
  constructor
  · exact attention_fwd_triton1_output_python_test_shape_compute_correct BO O s
  constructor
  · exact attention_fwd_triton1_ifcond_first_output_python_test_shape_compute_correct
      BO O s
  constructor
  · exact attention_fwd_triton1_recurrent_output_python_test_shape_compute_correct
      BO O s
  · exact attention_fwd_triton1_store_enabled_h_python_test_shape_compute_correct
      BHPre H i_iter s

theorem attention_fwd_triton1_local_bo_offset_python_test_shape_injective :
    Function.Injective
      (fun idx : TileIndex [32, 128] => localBoOffset 128 idx) := by
  rintro ⟨⟨ta, hta⟩, ⟨da, hda⟩, _⟩ ⟨⟨tb, htb⟩, ⟨db, hdb⟩, _⟩ h
  simp [localBoOffset] at h
  have ht : ta = tb := by omega
  have hd : da = db := by omega
  subst tb
  subst db
  rfl

theorem attention_fwd_triton1_local_bh_offset_python_test_shape_injective :
    Function.Injective
      (fun idx : TileIndex [128, 128] => localBhOffset 128 idx) := by
  rintro ⟨⟨ra, hra⟩, ⟨ca, hca⟩, _⟩ ⟨⟨rb, hrb⟩, ⟨cb, hcb⟩, _⟩ h
  simp [localBhOffset] at h
  have hr : ra = rb := by omega
  have hc : ca = cb := by omega
  subst rb
  subst cb
  rfl

theorem attention_fwd_triton1_bo_formula_python_test_shape_compute_correct
    (QTile KTile VTile BO : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := attention_fwd_triton1_bo_formula_slice QTile KTile VTile BO
        ((Real.sqrt (128 : ℝ))⁻¹) 32 128)
      (initialState := s)
      (write := fun idx : TileIndex [32, 128] =>
        some (BO, localBoOffset 128 idx))
      (expected := fun idx : TileIndex [32, 128] =>
        boFormulaSpec s QTile KTile VTile ((Real.sqrt (128 : ℝ))⁻¹)
          32 128 idx) := by
  exact attention_fwd_triton1_bo_formula_slice_compute_correct QTile KTile
    VTile BO ((Real.sqrt (128 : ℝ))⁻¹) 32 128 s
    attention_fwd_triton1_local_bo_offset_python_test_shape_injective

theorem attention_fwd_triton1_bh_formula_python_test_shape_compute_correct
    (KTile VTile BHPre : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := attention_fwd_triton1_bh_formula_slice KTile VTile BHPre
        32 128)
      (initialState := s)
      (write := fun idx : TileIndex [128, 128] =>
        some (BHPre, localBhOffset 128 idx))
      (expected := fun idx : TileIndex [128, 128] =>
        bhFormulaSpec s KTile VTile 32 128 idx) := by
  exact attention_fwd_triton1_bh_formula_slice_compute_correct KTile VTile
    BHPre 32 128 s
    attention_fwd_triton1_local_bh_offset_python_test_shape_injective

/-- Python-shape arithmetic coverage for the `BO` and `BHPre` producer tiles.

This pins the checked `BT = 32`, `BD = 128`, `scale = 1 / sqrt(128)` path for
the direct output accumulator `BO`, and the recurrent-state accumulator
`BHPre = dot(K, V)` used by the `STORE` branch summaries. -/
theorem attention_fwd_triton1_python_test_shape_accumulator_formula_summary
    (QTile KTile VTile BO BHPre : RegionName) (s : BlockState) :
    (∃ alg, (attention_fwd_triton1_bo_formula_slice QTile KTile VTile BO
      ((Real.sqrt (128 : ℝ))⁻¹) 32 128).toAlgorithm? = Except.ok alg) ∧
    (∃ alg, (attention_fwd_triton1_bh_formula_slice KTile VTile BHPre
      32 128).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := attention_fwd_triton1_bo_formula_slice QTile KTile VTile BO
        ((Real.sqrt (128 : ℝ))⁻¹) 32 128)
      (initialState := s)
      (write := fun idx : TileIndex [32, 128] =>
        some (BO, localBoOffset 128 idx))
      (expected := fun idx : TileIndex [32, 128] =>
        boFormulaSpec s QTile KTile VTile ((Real.sqrt (128 : ℝ))⁻¹)
          32 128 idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := attention_fwd_triton1_bh_formula_slice KTile VTile BHPre
        32 128)
      (initialState := s)
      (write := fun idx : TileIndex [128, 128] =>
        some (BHPre, localBhOffset 128 idx))
      (expected := fun idx : TileIndex [128, 128] =>
        bhFormulaSpec s KTile VTile 32 128 idx)) := by
  constructor
  · exact attention_fwd_triton1_bo_formula_slice_toAlgorithm_supported
      QTile KTile VTile BO ((Real.sqrt (128 : ℝ))⁻¹) 32 128
  constructor
  · exact attention_fwd_triton1_bh_formula_slice_toAlgorithm_supported
      KTile VTile BHPre 32 128
  constructor
  · exact attention_fwd_triton1_bo_formula_python_test_shape_compute_correct
      QTile KTile VTile BO s
  · exact attention_fwd_triton1_bh_formula_python_test_shape_compute_correct
      KTile VTile BHPre s

/-- Public Python-shape summary for the main `attention_fwd_triton1` cases.

The checked Python tests instantiate `B = 2`, `H = 8`, `T = 1024`, `D = 128`,
`BT = 32`, `BD = 128`, and `NT = 32`, with four `STORE`/`IFCOND`
combinations. This summary records faithful full-surface lowering for those
four branch combinations and ties them to the checked output-store/H-state
writeback slices. The dot-product arithmetic that produces the precomputed
`BO`/`BHPre` tiles remains represented by those slice inputs. -/
theorem attention_fwd_triton1_python_test_shape_slice_summary
    (Q K V H O BO BHPre : RegionName) (i_iter : Nat) (s : BlockState) :
    ((∃ alg, (attention_fwd_kernel_surface Q K V H O
      131072 128 1 524288 128 1024 ((Real.sqrt (128 : ℝ))⁻¹)
      32 128 32 Bool.false Bool.false).toAlgorithm? = Except.ok alg) ∧
     (∃ alg, (attention_fwd_kernel_surface Q K V H O
      131072 128 1 524288 128 1024 ((Real.sqrt (128 : ℝ))⁻¹)
      32 128 32 Bool.true Bool.false).toAlgorithm? = Except.ok alg) ∧
     (∃ alg, (attention_fwd_kernel_surface Q K V H O
      131072 128 1 524288 128 1024 ((Real.sqrt (128 : ℝ))⁻¹)
      32 128 32 Bool.false Bool.true).toAlgorithm? = Except.ok alg) ∧
     (∃ alg, (attention_fwd_kernel_surface Q K V H O
      131072 128 1 524288 128 1024 ((Real.sqrt (128 : ℝ))⁻¹)
      32 128 32 Bool.true Bool.true).toAlgorithm? = Except.ok alg)) ∧
    (ComputeCorrect.Realizes
      (kernel := attention_fwd_triton1_output_store_slice BO O
        131072 128 1 131072 128 1 32 128)
      (initialState := s)
      (write := fun idx : TileIndex [32, 128] =>
        some (O, outOffset s 131072 128 1 32 idx))
      (expected := fun idx : TileIndex [32, 128] =>
        s.readMem BO (boOffset s 131072 128 1 32 idx))) ∧
    (ComputeCorrect.Realizes
      (kernel := attention_fwd_triton1_output_store_slice BO O
        131072 128 1 131072 128 1 32 128)
      (initialState := s)
      (write := fun idx : TileIndex [32, 128] =>
        some (O, outOffset s 131072 128 1 32 idx))
      (expected := fun idx : TileIndex [32, 128] =>
        s.readMem BO (boOffset s 131072 128 1 32 idx))) ∧
    (ComputeCorrect.Realizes
      (kernel := attention_fwd_triton1_output_store_slice BO O
        131072 128 1 131072 128 1 32 128)
      (initialState := s)
      (write := fun idx : TileIndex [32, 128] =>
        some (O, outOffset s 131072 128 1 32 idx))
      (expected := fun idx : TileIndex [32, 128] =>
        s.readMem BO (boOffset s 131072 128 1 32 idx))) ∧
    (ComputeCorrect.Realizes
      (kernel := attention_fwd_triton1_h_store_slice BHPre H i_iter
        524288 128 32 128)
      (initialState := s)
      (write := fun idx : TileIndex [128, 128] =>
        some (H, hOffset s i_iter 524288 128 128 idx))
      (expected := fun idx : TileIndex [128, 128] =>
        hStoreSpec s BHPre i_iter 524288 128 128 idx)) := by
  constructor
  · constructor
    · exact attention_fwd_kernel_surface_toAlgorithm_supported Q K V H O
        131072 128 1 524288 128 1024 ((Real.sqrt (128 : ℝ))⁻¹)
        32 128 32 Bool.false Bool.false
    constructor
    · exact attention_fwd_kernel_surface_toAlgorithm_supported Q K V H O
        131072 128 1 524288 128 1024 ((Real.sqrt (128 : ℝ))⁻¹)
        32 128 32 Bool.true Bool.false
    constructor
    · exact attention_fwd_kernel_surface_toAlgorithm_supported Q K V H O
        131072 128 1 524288 128 1024 ((Real.sqrt (128 : ℝ))⁻¹)
        32 128 32 Bool.false Bool.true
    · exact attention_fwd_kernel_surface_toAlgorithm_supported Q K V H O
        131072 128 1 524288 128 1024 ((Real.sqrt (128 : ℝ))⁻¹)
        32 128 32 Bool.true Bool.true
  · exact attention_fwd_triton1_python_test_shape_all_outputs_compute_correct
      BO O BHPre H i_iter s

/-- Combined checked-shape summary for `attention_fwd_triton1.py`.

This exposes the four Python branch surfaces, the observable `O`/`H`
writebacks, and the arithmetic producers for the `BO` and `BHPre` tiles in
one public target. -/
theorem attention_fwd_triton1_python_test_shape_complete_summary
    (Q K V H O BO BHPre : RegionName) (i_iter : Nat) (s : BlockState) :
    (((∃ alg, (attention_fwd_kernel_surface Q K V H O
      131072 128 1 524288 128 1024 ((Real.sqrt (128 : ℝ))⁻¹)
      32 128 32 Bool.false Bool.false).toAlgorithm? = Except.ok alg) ∧
     (∃ alg, (attention_fwd_kernel_surface Q K V H O
      131072 128 1 524288 128 1024 ((Real.sqrt (128 : ℝ))⁻¹)
      32 128 32 Bool.true Bool.false).toAlgorithm? = Except.ok alg) ∧
     (∃ alg, (attention_fwd_kernel_surface Q K V H O
      131072 128 1 524288 128 1024 ((Real.sqrt (128 : ℝ))⁻¹)
      32 128 32 Bool.false Bool.true).toAlgorithm? = Except.ok alg) ∧
     (∃ alg, (attention_fwd_kernel_surface Q K V H O
      131072 128 1 524288 128 1024 ((Real.sqrt (128 : ℝ))⁻¹)
      32 128 32 Bool.true Bool.true).toAlgorithm? = Except.ok alg)) ∧
    (ComputeCorrect.Realizes
      (kernel := attention_fwd_triton1_output_store_slice BO O
        131072 128 1 131072 128 1 32 128)
      (initialState := s)
      (write := fun idx : TileIndex [32, 128] =>
        some (O, outOffset s 131072 128 1 32 idx))
      (expected := fun idx : TileIndex [32, 128] =>
        s.readMem BO (boOffset s 131072 128 1 32 idx))) ∧
    (ComputeCorrect.Realizes
      (kernel := attention_fwd_triton1_output_store_slice BO O
        131072 128 1 131072 128 1 32 128)
      (initialState := s)
      (write := fun idx : TileIndex [32, 128] =>
        some (O, outOffset s 131072 128 1 32 idx))
      (expected := fun idx : TileIndex [32, 128] =>
        s.readMem BO (boOffset s 131072 128 1 32 idx))) ∧
    (ComputeCorrect.Realizes
      (kernel := attention_fwd_triton1_output_store_slice BO O
        131072 128 1 131072 128 1 32 128)
      (initialState := s)
      (write := fun idx : TileIndex [32, 128] =>
        some (O, outOffset s 131072 128 1 32 idx))
      (expected := fun idx : TileIndex [32, 128] =>
        s.readMem BO (boOffset s 131072 128 1 32 idx))) ∧
    (ComputeCorrect.Realizes
      (kernel := attention_fwd_triton1_h_store_slice BHPre H i_iter
        524288 128 32 128)
      (initialState := s)
      (write := fun idx : TileIndex [128, 128] =>
        some (H, hOffset s i_iter 524288 128 128 idx))
      (expected := fun idx : TileIndex [128, 128] =>
        hStoreSpec s BHPre i_iter 524288 128 128 idx))) ∧
    ((∃ alg, (attention_fwd_triton1_bo_formula_slice Q K V BO
      ((Real.sqrt (128 : ℝ))⁻¹) 32 128).toAlgorithm? = Except.ok alg) ∧
    (∃ alg, (attention_fwd_triton1_bh_formula_slice K V BHPre
      32 128).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := attention_fwd_triton1_bo_formula_slice Q K V BO
        ((Real.sqrt (128 : ℝ))⁻¹) 32 128)
      (initialState := s)
      (write := fun idx : TileIndex [32, 128] =>
        some (BO, localBoOffset 128 idx))
      (expected := fun idx : TileIndex [32, 128] =>
        boFormulaSpec s Q K V ((Real.sqrt (128 : ℝ))⁻¹) 32 128 idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := attention_fwd_triton1_bh_formula_slice K V BHPre 32 128)
      (initialState := s)
      (write := fun idx : TileIndex [128, 128] =>
        some (BHPre, localBhOffset 128 idx))
      (expected := fun idx : TileIndex [128, 128] =>
        bhFormulaSpec s K V 32 128 idx))) := by
  constructor
  · exact attention_fwd_triton1_python_test_shape_slice_summary Q K V H O BO
      BHPre i_iter s
  · exact attention_fwd_triton1_python_test_shape_accumulator_formula_summary
      Q K V BO BHPre s



















/-- `output_summary` for the checked `attention_fwd_triton1` surface. -/
theorem attention_fwd_triton1_python_test_shape_output_summary
    (Q K V H O : RegionName) (i_iter : Nat) (s : BlockState) :
    (ComputeCorrect.Realizes
      (kernel := attention_fwd_kernel_surface Q K V H O
        131072 128 1 524288 128 1024 ((Real.sqrt (128 : ℝ))⁻¹)
        32 128 32 Bool.false Bool.false)
      (initialState := s)
      (write := fun idx : TileIndex [32, 128] =>
        some (O, outOffset s 131072 128 1 32 idx))
      (expected := fun idx : TileIndex [32, 128] =>
        attentionFwdTriton1SurfaceValue s Q K V H O O Bool.false Bool.false
          (outOffset s 131072 128 1 32 idx))) ∧
    (ComputeCorrect.Realizes
      (kernel := attention_fwd_kernel_surface Q K V H O
        131072 128 1 524288 128 1024 ((Real.sqrt (128 : ℝ))⁻¹)
        32 128 32 Bool.true Bool.false)
      (initialState := s)
      (write := fun idx : TileIndex [32, 128] =>
        some (O, outOffset s 131072 128 1 32 idx))
      (expected := fun idx : TileIndex [32, 128] =>
        attentionFwdTriton1SurfaceValue s Q K V H O O Bool.true Bool.false
          (outOffset s 131072 128 1 32 idx))) ∧
    (ComputeCorrect.Realizes
      (kernel := attention_fwd_kernel_surface Q K V H O
        131072 128 1 524288 128 1024 ((Real.sqrt (128 : ℝ))⁻¹)
        32 128 32 Bool.false Bool.true)
      (initialState := s)
      (write := fun idx : TileIndex [32, 128] =>
        some (O, outOffset s 131072 128 1 32 idx))
      (expected := fun idx : TileIndex [32, 128] =>
        attentionFwdTriton1SurfaceValue s Q K V H O O Bool.false Bool.true
          (outOffset s 131072 128 1 32 idx))) ∧
    (ComputeCorrect.Realizes
      (kernel := attention_fwd_kernel_surface Q K V H O
        131072 128 1 524288 128 1024 ((Real.sqrt (128 : ℝ))⁻¹)
        32 128 32 Bool.true Bool.true)
      (initialState := s)
      (write := fun idx : TileIndex [32, 128] =>
        some (O, outOffset s 131072 128 1 32 idx))
      (expected := fun idx : TileIndex [32, 128] =>
        attentionFwdTriton1SurfaceValue s Q K V H O O Bool.true Bool.true
          (outOffset s 131072 128 1 32 idx))) ∧
    (ComputeCorrect.Realizes
      (kernel := attention_fwd_kernel_surface Q K V H O
        131072 128 1 524288 128 1024 ((Real.sqrt (128 : ℝ))⁻¹)
        32 128 32 Bool.true Bool.false)
      (initialState := s)
      (write := fun idx : TileIndex [128, 128] =>
        some (H, hOffset s i_iter 524288 128 128 idx))
      (expected := fun idx : TileIndex [128, 128] =>
        attentionFwdTriton1SurfaceValue s Q K V H O H Bool.true Bool.false
          (hOffset s i_iter 524288 128 128 idx))) ∧
    (ComputeCorrect.Realizes
      (kernel := attention_fwd_kernel_surface Q K V H O
        131072 128 1 524288 128 1024 ((Real.sqrt (128 : ℝ))⁻¹)
        32 128 32 Bool.true Bool.true)
      (initialState := s)
      (write := fun idx : TileIndex [128, 128] =>
        some (H, hOffset s i_iter 524288 128 128 idx))
      (expected := fun idx : TileIndex [128, 128] =>
        attentionFwdTriton1SurfaceValue s Q K V H O H Bool.true Bool.true
          (hOffset s i_iter 524288 128 128 idx))) := by
  constructor
  · exact attention_fwd_triton1_surface_output_compute_correct Q K V H O O
      Bool.false Bool.false s
  constructor
  · exact attention_fwd_triton1_surface_output_compute_correct Q K V H O O
      Bool.true Bool.false s
  constructor
  · exact attention_fwd_triton1_surface_output_compute_correct Q K V H O O
      Bool.false Bool.true s
  constructor
  · exact attention_fwd_triton1_surface_output_compute_correct Q K V H O O
      Bool.true Bool.true s
  constructor
  · exact attention_fwd_triton1_surface_h_compute_correct Q K V H O
      Bool.false i_iter s
  · exact attention_fwd_triton1_surface_h_compute_correct Q K V H O
      Bool.true i_iter s

end VeriTile.Bench.TritonBenchG.AttentionFwdTriton1
