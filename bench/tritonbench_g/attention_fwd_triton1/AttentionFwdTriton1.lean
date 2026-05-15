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

end VeriTile.Bench.TritonBenchG.AttentionFwdTriton1
