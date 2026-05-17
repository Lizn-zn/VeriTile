import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
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
        evalOp, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
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

/-- Proof-oriented `p_h` state-store slice of `attention_fwd_triton1.py`.
Companion to the output_store_slice: takes a precomputed `BHPre` [BD, BD]
tile and proves the per-iteration writeback into `H` at the canonical block
offset `i_bh * s_hh + (i * BD + idx.1) * s_ht + idx.2.1`. -/
def attention_fwd_triton1_h_store_slice
    (BHPre H : RegionName) (i_iter s_hh s_ht BT BD : Nat) :
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
  simp [exec, attention_fwd_triton1_h_store_slice, stepStmts, stepStmt, evalOp,
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

end VeriTile.Bench.TritonBenchG.AttentionFwdTriton1
