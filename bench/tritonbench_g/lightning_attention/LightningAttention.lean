import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Semantics.TileOps
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.LightningAttention

open VeriTile.Triton

set_option linter.unusedSimpArgs false
set_option linter.unusedVariables false

/-- Faithful transcription of `lightning_attention.py`'s `_fwd_kernel`.

This covers the full forward recurrent tile loop. -/
def lightning_attention_forward_surface
    (Q K V Out : RegionName)
    (_b h n d e BLOCK NUM_BLOCK BLOCK_MODEL : Nat) :
    ComputeKernel := triton {
  off_bh = tl.program_id(0)
  off_bh % $(h)
  off_e = tl.program_id(1)
  qk_offset = off_bh * $(n) * $(d)
  v_offset = off_bh * $(n) * $(e)
  o_offset = off_bh * $(n) * $(e)
  e_offset = off_e * $(BLOCK_MODEL)
  Q_block_ptr = Q + qk_offset + tl.arange(0, $(d))[None, :]
  K_trans_block_ptr = K + qk_offset + tl.arange(0, $(d))[:, None]
  V_block_ptr = V + v_offset + e_offset + tl.arange(0, $(BLOCK_MODEL))[None, :]
  O_block_ptr = Out + o_offset + e_offset + tl.arange(0, $(BLOCK_MODEL))[None, :]
  off_block = tl.arange(0, $(BLOCK))
  index = off_block[:, None] - off_block[None, :]
  kv = tl.zeros([$(d), $(BLOCK_MODEL)], dtype=tl.float32)
  for i in range($(0), $(NUM_BLOCK), $(1)) {
    q = tl.load(Q_block_ptr + off_block[:, None] * $(d),
      mask=off_block[:, None] < $(n), other=0.0).to(tl.float32)
    k_trans = tl.load(K_trans_block_ptr + off_block[None, :] * $(d),
      mask=off_block[None, :] < $(n), other=0.0).to(tl.float32)
    v = tl.load(V_block_ptr + off_block[:, None] * $(e),
      mask=off_block[:, None] < $(n), other=0.0).to(tl.float32)
    qk = tl.dot(q, k_trans)
    qk = tl.where(index >= 0, qk, 0)
    o_intra = tl.dot(qk, v)
    o_inter = tl.dot(q, kv)
    o = o_intra + o_inter
    tl.store(O_block_ptr + off_block[:, None] * $(e),
      (o).to(O_block_ptr.dtype.element_ty), mask=off_block[:, None] < $(n))
    kv += tl.dot(k_trans, v)
    off_block += $(BLOCK)
  }
}

/-- The full Python-shaped forward recurrent attention surface lowers to the
algorithm layer, including the causal `tl.where`, recurrent `kv` update, and
masked output store. -/
theorem lightning_attention_forward_surface_toAlgorithm_supported
    (Q K V Out : RegionName)
    (_b h n d e BLOCK NUM_BLOCK BLOCK_MODEL : Nat) :
    ∃ alg, (lightning_attention_forward_surface Q K V Out _b h n d e BLOCK
      NUM_BLOCK BLOCK_MODEL).toAlgorithm? = Except.ok alg := by
  simp [lightning_attention_forward_surface, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-- Faithful transcription of `lightning_attention.py`'s `_bwd_intra_kernel`.

This records the intra-block backward path: diagonal causal masks, `DQ`/`DK`/
`DV` stores, and the Python test's block layout. -/
def lightning_attention_bwd_intra_surface
    (Q K V DO DQ DK DV : RegionName)
    (_b h n d e BLOCK NUM_BLOCK CBLOCK NUM_CBLOCK : Nat) :
    ComputeKernel := triton {
  off_bh = tl.program_id(0)
  off_block = tl.program_id(1)
  off_bh % $(h)
  qk_offset = off_bh * $(n) * $(d)
  v_offset = off_bh * $(n) * $(e)
  o_offset = off_bh * $(n) * $(e)
  block_offset = off_block * $(BLOCK) + tl.arange(0, $(BLOCK))
  Q_trans_block_ptr =
    Q + qk_offset + block_offset[None, :] * $(d) + tl.arange(0, $(d))[:, None]
  K_block_ptr =
    K + qk_offset + block_offset[:, None] * $(d) + tl.arange(0, $(d))[None, :]
  V_trans_block_ptr =
    V + v_offset + block_offset[None, :] * $(e) + tl.arange(0, $(e))[:, None]
  DQ_block_ptr =
    DQ + qk_offset + block_offset[:, None] * $(d) + tl.arange(0, $(d))[None, :]
  DK_trans_block_ptr =
    DK + qk_offset + block_offset[None, :] * $(d) + tl.arange(0, $(d))[:, None]
  DV_block_ptr =
    DV + v_offset + block_offset[:, None] * $(e) + tl.arange(0, $(e))[None, :]
  DO_block_ptr =
    DO + o_offset + block_offset[:, None] * $(e) + tl.arange(0, $(e))[None, :]
  array = tl.arange(0, $(BLOCK))
  index = array[:, None] - array[None, :]
  k = tl.load(K_block_ptr, mask=block_offset[:, None] < $(n),
    other=0.0).to(tl.float32)
  v_trans = tl.load(V_trans_block_ptr, mask=block_offset[None, :] < $(n),
    other=0.0).to(tl.float32)
  b_do = tl.load(DO_block_ptr, mask=block_offset[:, None] < $(n),
    other=0.0).to(tl.float32)
  q_trans = tl.load(Q_trans_block_ptr, mask=block_offset[None, :] < $(n),
    other=0.0).to(tl.float32)
  dqk = tl.dot(b_do, v_trans)
  dqk = tl.where(index >= 0, dqk, 0)
  dq_intra = tl.dot(dqk, k)
  dk_intra_trans = tl.dot(q_trans, dqk)
  qk_trans = tl.dot(k, q_trans)
  qk_trans = tl.where(index <= 0, qk_trans, 0)
  dv_intra = tl.dot(qk_trans, b_do)
  dq = dq_intra
  dk_trans = dk_intra_trans
  dv = dv_intra
  tl.store(DQ_block_ptr, (dq).to(DQ_block_ptr.dtype.element_ty),
    mask=block_offset[:, None] < $(n))
  tl.store(DK_trans_block_ptr, (dk_trans).to(DK_trans_block_ptr.dtype.element_ty),
    mask=block_offset[None, :] < $(n))
  tl.store(DV_block_ptr, (dv).to(DV_block_ptr.dtype.element_ty),
    mask=block_offset[:, None] < $(n))
}

/-- The full intra-block backward surface lowers through the algorithm layer. -/
theorem lightning_attention_bwd_intra_surface_toAlgorithm_supported
    (Q K V DO DQ DK DV : RegionName)
    (_b h n d e BLOCK NUM_BLOCK CBLOCK NUM_CBLOCK : Nat) :
    ∃ alg, (lightning_attention_bwd_intra_surface Q K V DO DQ DK DV
      _b h n d e BLOCK NUM_BLOCK CBLOCK NUM_CBLOCK).toAlgorithm? =
        Except.ok alg := by
  simp [lightning_attention_bwd_intra_surface, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-- Faithful transcription of `lightning_attention.py`'s `_bwd_inter_kernel`.

The surface preserves both Python loop nests: the forward scan that accumulates
and writes inter-block `DQ`, and the reverse scan that accumulates and writes
inter-block `DK`/`DV`. -/
def lightning_attention_bwd_inter_surface
    (Q K V DO DQ DK DV : RegionName)
    (_b h n d e BLOCK NUM_BLOCK CBLOCK NUM_CBLOCK : Nat) :
    ComputeKernel := triton {
  off_bh = tl.program_id(0)
  off_bh % $(h)
  qk_offset = off_bh * $(n) * $(d)
  v_offset = off_bh * $(n) * $(e)
  o_offset = off_bh * $(n) * $(e)
  DQ_block_ptr =
    DQ + qk_offset + tl.arange(0, $(CBLOCK))[:, None] * $(d) +
      tl.arange(0, $(d))[None, :]
  K_block_ptr =
    K + qk_offset + tl.arange(0, $(CBLOCK))[:, None] * $(d) +
      tl.arange(0, $(d))[None, :]
  V_trans_block_ptr =
    V + v_offset + tl.arange(0, $(CBLOCK))[None, :] * $(e) +
      tl.arange(0, $(e))[:, None]
  DO_block_ptr =
    DO + o_offset + tl.arange(0, $(CBLOCK))[:, None] * $(e) +
      tl.arange(0, $(e))[None, :]
  off_block1 = tl.arange(0, $(CBLOCK))
  off_block2 = tl.arange(0, $(CBLOCK))
  kv_trans = tl.zeros([$(e), $(d)], dtype=tl.float32)
  for i in range($(0), $(NUM_BLOCK), $(1)) {
    for j in range($(0), $(NUM_CBLOCK), $(1)) {
      if i > 0 {
        b_do = tl.load(DO_block_ptr, mask=off_block1[:, None] < $(n),
          other=0.0).to(tl.float32)
        dq_inter = tl.dot(b_do, kv_trans)
        dq = dq_inter + tl.load(DQ_block_ptr,
          mask=off_block1[:, None] < $(n), other=0.0)
        tl.store(DQ_block_ptr, (dq).to(DQ_block_ptr.dtype.element_ty),
          mask=off_block1[:, None] < $(n))
      }
      DQ_block_ptr += $(CBLOCK) * $(d)
      DO_block_ptr += $(CBLOCK) * $(e)
      off_block1 += $(CBLOCK)
    }
    kv_trans_current = tl.zeros([$(e), $(d)], dtype=tl.float32)
    for j in range($(0), $(NUM_CBLOCK), $(1)) {
      v_trans = tl.load(V_trans_block_ptr, mask=off_block2[None, :] < $(n),
        other=0.0).to(tl.float32)
      k = tl.load(K_block_ptr, mask=off_block2[:, None] < $(n),
        other=0.0).to(tl.float32)
      kv_trans_current += tl.dot(v_trans, k)
      K_block_ptr += $(CBLOCK) * $(d)
      V_trans_block_ptr += $(CBLOCK) * $(e)
      off_block2 += $(CBLOCK)
    }
    kv_trans += kv_trans_current
  }
  m = $(NUM_BLOCK) * $(BLOCK)
  off_block1 = m + tl.arange(0, $(CBLOCK))
  off_block2 = m + tl.arange(0, $(CBLOCK))
  Q_trans_block_ptr =
    Q + qk_offset + m * $(d) + tl.arange(0, $(CBLOCK))[None, :] * $(d) +
      tl.arange(0, $(d))[:, None]
  K_block_ptr =
    K + qk_offset + m * $(d) + tl.arange(0, $(CBLOCK))[:, None] * $(d) +
      tl.arange(0, $(d))[None, :]
  V_trans_block_ptr =
    V + v_offset + m * $(e) + tl.arange(0, $(CBLOCK))[None, :] * $(e) +
      tl.arange(0, $(e))[:, None]
  DK_trans_block_ptr =
    DK + qk_offset + m * $(d) + tl.arange(0, $(CBLOCK))[None, :] * $(d) +
      tl.arange(0, $(d))[:, None]
  DV_block_ptr =
    DV + v_offset + m * $(e) + tl.arange(0, $(CBLOCK))[:, None] * $(e) +
      tl.arange(0, $(e))[None, :]
  DO_block_ptr =
    DO + o_offset + m * $(e) + tl.arange(0, $(CBLOCK))[:, None] * $(e) +
      tl.arange(0, $(e))[None, :]
  dkv = tl.zeros([$(d), $(e)], dtype=tl.float32)
  for i in range($(NUM_BLOCK) - $(1), -$(1), -$(1)) {
    for j in range($(NUM_CBLOCK) - $(1), -$(1), -$(1)) {
      K_block_ptr -= $(CBLOCK) * $(d)
      V_trans_block_ptr -= $(CBLOCK) * $(e)
      DK_trans_block_ptr -= $(CBLOCK) * $(d)
      DV_block_ptr -= $(CBLOCK) * $(e)
      off_block1 -= $(CBLOCK)
      if i < $(NUM_BLOCK) - $(1) {
        k = tl.load(K_block_ptr, mask=off_block1[:, None] < $(n),
          other=0.0).to(tl.float32)
        v_trans = tl.load(V_trans_block_ptr, mask=off_block1[None, :] < $(n),
          other=0.0).to(tl.float32)
        dk_inter_trans = tl.dot(dkv, v_trans)
        dv_inter = tl.dot(k, dkv)
        dk_trans = dk_inter_trans + tl.load(DK_trans_block_ptr,
          mask=off_block1[None, :] < $(n), other=0.0)
        dv = dv_inter + tl.load(DV_block_ptr,
          mask=off_block1[:, None] < $(n), other=0.0)
        tl.store(DK_trans_block_ptr, (dk_trans).to(DK_trans_block_ptr.dtype.element_ty),
          mask=off_block1[None, :] < $(n))
        tl.store(DV_block_ptr, (dv).to(DV_block_ptr.dtype.element_ty),
          mask=off_block1[:, None] < $(n))
      }
    }
    dkv_current = tl.zeros([$(d), $(e)], dtype=tl.float32)
    for j in range($(NUM_CBLOCK) - $(1), -$(1), -$(1)) {
      DO_block_ptr -= $(CBLOCK) * $(e)
      Q_trans_block_ptr -= $(CBLOCK) * $(d)
      off_block2 -= $(CBLOCK)
      b_do = tl.load(DO_block_ptr, mask=off_block2[:, None] < $(n),
        other=0.0).to(tl.float32)
      q_trans = tl.load(Q_trans_block_ptr, mask=off_block2[None, :] < $(n),
        other=0.0).to(tl.float32)
      dkv_current += tl.dot(q_trans, b_do)
    }
    dkv += dkv_current
  }
}

/-- The inter-block backward surface lowers with the Python forward and reverse
loop nests preserved. -/
theorem lightning_attention_bwd_inter_surface_toAlgorithm_supported
    (Q K V DO DQ DK DV : RegionName)
    (_b h n d e BLOCK NUM_BLOCK CBLOCK NUM_CBLOCK : Nat) :
    ∃ alg, (lightning_attention_bwd_inter_surface Q K V DO DQ DK DV
      _b h n d e BLOCK NUM_BLOCK CBLOCK NUM_CBLOCK).toAlgorithm? =
        Except.ok alg := by
  simp [lightning_attention_bwd_inter_surface, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-- Surface transcription/proof-oriented forward output-store slice of
`lightning_attention.py`'s `_fwd_kernel`.

The full kernel computes a recurrent attention tile `o`. This slice starts from
a precomputed `OAcc` tile and proves the masked writeback into `Out`. The full forward recurrent `kv` loop is represented above by
`lightning_attention_forward_surface`; the backward kernels' negative-step loops
remain separate modeling work. -/
def lightning_attention_forward_store_slice
    (OAcc Out : RegionName) (n e BLOCK BLOCK_MODEL : Nat) :
    ComputeKernel := triton {
  off_bh = tl.program_id(0)
  off_e = tl.program_id(1)
  off_block = tl.program_id(2) * $(BLOCK) + tl.arange(0, $(BLOCK))
  offs_e = tl.arange(0, $(BLOCK_MODEL))
  mask = off_block[:, None] < $(n)
  o = tl.load(OAcc + off_bh * $(n) * $(e) +
      off_block[:, None] * $(e) + off_e * $(BLOCK_MODEL) + offs_e[None, :],
      mask=mask, other=0.0)
  tl.store(Out + off_bh * $(n) * $(e) +
      off_block[:, None] * $(e) + off_e * $(BLOCK_MODEL) + offs_e[None, :],
      o, mask=mask)
}

def rowIndex (s : BlockState) (BLOCK : Nat) (i : Fin BLOCK) : Nat :=
  s.pids 2 * BLOCK + i.val

def colIndex (idx : TileIndex [BLOCK, BLOCK_MODEL]) : Nat :=
  idx.2.1.val

def active (s : BlockState) (n BLOCK : Nat) (idx : TileIndex [BLOCK, BLOCK_MODEL]) :
    Prop :=
  rowIndex s BLOCK idx.1 < n

instance activeDecidable (s : BlockState) (n BLOCK BLOCK_MODEL : Nat)
    (idx : TileIndex [BLOCK, BLOCK_MODEL]) :
    Decidable (active s n BLOCK idx) := by
  unfold active
  infer_instance

def tileOffset (s : BlockState) (n e BLOCK BLOCK_MODEL : Nat)
    (idx : TileIndex [BLOCK, BLOCK_MODEL]) : Nat :=
  s.pids 0 * n * e + rowIndex s BLOCK idx.1 * e +
    s.pids 1 * BLOCK_MODEL + colIndex idx

noncomputable def storeValue (s : BlockState) (OAcc : RegionName)
    (n e BLOCK BLOCK_MODEL : Nat) (idx : TileIndex [BLOCK, BLOCK_MODEL]) : ℝ :=
  WithBot.unbotD 0
    (if active s n BLOCK idx then
      some (s.readMem OAcc (tileOffset s n e BLOCK BLOCK_MODEL idx))
    else some (0.0 : ℝ))

theorem lightning_attention_forward_store_slice_correct
    (OAcc Out : RegionName) (n e BLOCK BLOCK_MODEL : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK, BLOCK_MODEL] =>
        tileOffset s n e BLOCK BLOCK_MODEL idx)) :
    ∀ idx : TileIndex [BLOCK, BLOCK_MODEL],
      let outAddr := tileOffset s n e BLOCK BLOCK_MODEL idx
      (exec (lightning_attention_forward_store_slice OAcc Out n e BLOCK BLOCK_MODEL) s).map
          (·.readMem Out outAddr)
        = some (if active s n BLOCK idx then
            storeValue s OAcc n e BLOCK BLOCK_MODEL idx
          else s.readMem Out outAddr) := by
  intro idx
  simp [exec, lightning_attention_forward_store_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.remap,
        Tile.expandDim, Tile.ptrAdd,
        NumericDType.add, NumericDType.mul, ComparableDType.lt,
        rowIndex, colIndex, active, tileOffset, TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BLOCK, BLOCK_MODEL] → Nat :=
    fun idx =>
      s.pids 0 * n * e + (s.pids 2 * BLOCK + idx.1.val) * e +
        s.pids 1 * BLOCK_MODEL + idx.2.1.val
  let valueFn : TileIndex [BLOCK, BLOCK_MODEL] → ℝ :=
    fun idx =>
      WithBot.unbotD 0
        (if s.pids 2 * BLOCK + idx.1.val < n then
          some (s.readMem OAcc
            (s.pids 0 * n * e + (s.pids 2 * BLOCK + idx.1.val) * e +
              s.pids 1 * BLOCK_MODEL + idx.2.1.val))
        else some (0.0 : ℝ))
  let P : TileIndex [BLOCK, BLOCK_MODEL] → Prop :=
    fun idx => s.pids 2 * BLOCK + idx.1.val < n
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, tileOffset, rowIndex, colIndex] using hOutInj
  change (List.foldl
      (fun (acc : BlockState) i =>
        if P i then acc.writeMem Out (offsetFn i) (valueFn i) else acc)
      _ (TileShape.allIndices [BLOCK, BLOCK_MODEL])).readMem Out
        (offsetFn idx) =
    if P idx then storeValue s OAcc n e BLOCK BLOCK_MODEL idx
    else s.readMem Out (offsetFn idx)
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj idx]
  by_cases hActive : s.pids 2 * BLOCK + idx.1.val < n
  · simp [offsetFn, valueFn, P, storeValue, active, tileOffset, rowIndex,
      colIndex, hActive]
  · simp [offsetFn, valueFn, P, storeValue, active, tileOffset, rowIndex,
      colIndex, hActive]

theorem lightning_attention_forward_store_slice_compute_correct
    (OAcc Out : RegionName) (n e BLOCK BLOCK_MODEL : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK, BLOCK_MODEL] =>
        tileOffset s n e BLOCK BLOCK_MODEL idx)) :
    ComputeCorrect.Realizes
      (kernel := lightning_attention_forward_store_slice OAcc Out n e BLOCK BLOCK_MODEL)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK, BLOCK_MODEL] => active s n BLOCK idx)
        (fun idx : TileIndex [BLOCK, BLOCK_MODEL] =>
          (Out, tileOffset s n e BLOCK BLOCK_MODEL idx)))
      (expected := fun idx : TileIndex [BLOCK, BLOCK_MODEL] =>
        storeValue s OAcc n e BLOCK BLOCK_MODEL idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [lightning_attention_forward_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := lightning_attention_forward_store_slice_correct OAcc Out n e
    BLOCK BLOCK_MODEL s hOutInj idx
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-- Forward output store slice that includes the Python arithmetic
`o = o_intra + o_inter` before the masked writeback. -/
def lightning_attention_forward_sum_store_slice
    (OIntra OInter Out : RegionName) (n e BLOCK BLOCK_MODEL : Nat) :
    ComputeKernel := triton {
  off_bh = tl.program_id(0)
  off_e = tl.program_id(1)
  off_block = tl.program_id(2) * $(BLOCK) + tl.arange(0, $(BLOCK))
  offs_e = tl.arange(0, $(BLOCK_MODEL))
  mask = off_block[:, None] < $(n)
  o_intra = tl.load(OIntra + off_bh * $(n) * $(e) +
      off_block[:, None] * $(e) + off_e * $(BLOCK_MODEL) + offs_e[None, :],
      mask=mask, other=0.0)
  o_inter = tl.load(OInter + off_bh * $(n) * $(e) +
      off_block[:, None] * $(e) + off_e * $(BLOCK_MODEL) + offs_e[None, :],
      mask=mask, other=0.0)
  o = o_intra + o_inter
  tl.store(Out + off_bh * $(n) * $(e) +
      off_block[:, None] * $(e) + off_e * $(BLOCK_MODEL) + offs_e[None, :],
      o, mask=mask)
}

noncomputable def sumStoreValue
    (s : BlockState) (OIntra OInter : RegionName)
    (n e BLOCK BLOCK_MODEL : Nat) (idx : TileIndex [BLOCK, BLOCK_MODEL]) : ℝ :=
  WithBot.unbotD 0
    (if active s n BLOCK idx then
      some (s.readMem OIntra (tileOffset s n e BLOCK BLOCK_MODEL idx) +
        s.readMem OInter (tileOffset s n e BLOCK BLOCK_MODEL idx))
    else some (0.0 : ℝ))

theorem lightning_attention_forward_sum_store_slice_correct
    (OIntra OInter Out : RegionName) (n e BLOCK BLOCK_MODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK, BLOCK_MODEL] =>
        tileOffset s n e BLOCK BLOCK_MODEL idx)) :
    ∀ idx : TileIndex [BLOCK, BLOCK_MODEL],
      let outAddr := tileOffset s n e BLOCK BLOCK_MODEL idx
      (exec (lightning_attention_forward_sum_store_slice OIntra OInter Out
          n e BLOCK BLOCK_MODEL) s).map (·.readMem Out outAddr)
        = some (if active s n BLOCK idx then
            sumStoreValue s OIntra OInter n e BLOCK BLOCK_MODEL idx
          else s.readMem Out outAddr) := by
  intro idx
  simp [exec, lightning_attention_forward_sum_store_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.remap,
        Tile.expandDim, Tile.ptrAdd, NumericDType.add, NumericDType.mul,
        ComparableDType.lt, rowIndex, colIndex, active, tileOffset,
        TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BLOCK, BLOCK_MODEL] → Nat :=
    fun idx =>
      s.pids 0 * n * e + (s.pids 2 * BLOCK + idx.1.val) * e +
        s.pids 1 * BLOCK_MODEL + idx.2.1.val
  let valueFn : TileIndex [BLOCK, BLOCK_MODEL] → ℝ :=
    fun idx =>
      WithBot.unbotD 0
        (Option.map₂ (fun a b => a + b)
          (if s.pids 2 * BLOCK + idx.1.val < n then
            some (s.readMem OIntra (offsetFn idx))
          else some (0.0 : ℝ))
          (if s.pids 2 * BLOCK + idx.1.val < n then
            some (s.readMem OInter (offsetFn idx))
          else some (0.0 : ℝ)))
  let P : TileIndex [BLOCK, BLOCK_MODEL] → Prop :=
    fun idx => s.pids 2 * BLOCK + idx.1.val < n
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, tileOffset, rowIndex, colIndex] using hOutInj
  change (List.foldl
      (fun (acc : BlockState) i =>
        if P i then acc.writeMem Out (offsetFn i) (valueFn i) else acc)
      _ (TileShape.allIndices [BLOCK, BLOCK_MODEL])).readMem Out
        (offsetFn idx) =
    if P idx then sumStoreValue s OIntra OInter n e BLOCK BLOCK_MODEL idx
    else s.readMem Out (offsetFn idx)
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj idx]
  by_cases hActive : s.pids 2 * BLOCK + idx.1.val < n
  · simp [offsetFn, valueFn, P, sumStoreValue, active, tileOffset, rowIndex,
      colIndex, hActive]
  · simp [offsetFn, valueFn, P, sumStoreValue, active, tileOffset, rowIndex,
      colIndex, hActive]

theorem lightning_attention_forward_sum_store_slice_compute_correct
    (OIntra OInter Out : RegionName) (n e BLOCK BLOCK_MODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK, BLOCK_MODEL] =>
        tileOffset s n e BLOCK BLOCK_MODEL idx)) :
    ComputeCorrect.Realizes
      (kernel := lightning_attention_forward_sum_store_slice OIntra OInter Out
        n e BLOCK BLOCK_MODEL)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK, BLOCK_MODEL] => active s n BLOCK idx)
        (fun idx : TileIndex [BLOCK, BLOCK_MODEL] =>
          (Out, tileOffset s n e BLOCK BLOCK_MODEL idx)))
      (expected := fun idx : TileIndex [BLOCK, BLOCK_MODEL] =>
        sumStoreValue s OIntra OInter n e BLOCK BLOCK_MODEL idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [lightning_attention_forward_sum_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := lightning_attention_forward_sum_store_slice_correct OIntra OInter
    Out n e BLOCK BLOCK_MODEL s hOutInj idx
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-- Proof-oriented backward gradient tile-store slice of
`lightning_attention.py`.

The Python backward path writes `DQ`, `DK`, and `DV` from intra- and inter-block
accumulators. This generic slice fixes one block row and proves the masked
writeback from a precomputed gradient tile into any of those output regions. -/
def lightning_attention_bwd_grad_store_slice
    (GradPre Out : RegionName) (n width BLOCK WIDTH : Nat) :
    ComputeKernel := triton {
  off_bh = tl.program_id(0)
  off_block = tl.program_id(1) * $(BLOCK) + tl.arange(0, $(BLOCK))
  offs_w = tl.arange(0, $(WIDTH))
  mask = (off_block[:, None] < $(n)) & (offs_w[None, :] < $(WIDTH))
  grad = tl.load(GradPre + off_bh * $(n) * $(width) +
      off_block[:, None] * $(width) + offs_w[None, :],
      mask=mask, other=0.0)
  tl.store(Out + off_bh * $(n) * $(width) +
      off_block[:, None] * $(width) + offs_w[None, :],
      (grad).to(Out.dtype.element_ty), mask=mask)
}

def gradRowIndex (s : BlockState) (BLOCK : Nat) (i : Fin BLOCK) : Nat :=
  s.pids 1 * BLOCK + i.val

def gradColIndex (idx : TileIndex [BLOCK, WIDTH]) : Nat :=
  idx.2.1.val

def activeGrad (s : BlockState) (n BLOCK : Nat)
    (idx : TileIndex [BLOCK, WIDTH]) : Prop :=
  gradRowIndex s BLOCK idx.1 < n

instance activeGradDecidable (s : BlockState) (n BLOCK WIDTH : Nat)
    (idx : TileIndex [BLOCK, WIDTH]) :
    Decidable (activeGrad s n BLOCK idx) := by
  unfold activeGrad
  infer_instance

def gradTileOffset (s : BlockState) (n width BLOCK WIDTH : Nat)
    (idx : TileIndex [BLOCK, WIDTH]) : Nat :=
  s.pids 0 * n * width + gradRowIndex s BLOCK idx.1 * width + gradColIndex idx

noncomputable def gradStoreValue (s : BlockState) (GradPre : RegionName)
    (n width BLOCK WIDTH : Nat) (idx : TileIndex [BLOCK, WIDTH]) : ℝ :=
  WithBot.unbotD 0
    (if activeGrad s n BLOCK idx then
      some (s.readMem GradPre (gradTileOffset s n width BLOCK WIDTH idx))
    else some (0.0 : ℝ))

theorem lightning_attention_bwd_grad_store_slice_correct
    (GradPre Out : RegionName) (n width BLOCK WIDTH : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK, WIDTH] =>
        gradTileOffset s n width BLOCK WIDTH idx)) :
    ∀ idx : TileIndex [BLOCK, WIDTH],
      let outAddr := gradTileOffset s n width BLOCK WIDTH idx
      (exec (lightning_attention_bwd_grad_store_slice GradPre Out n width BLOCK WIDTH)
          s).map (·.readMem Out outAddr)
        = some (if activeGrad s n BLOCK idx then
            gradStoreValue s GradPre n width BLOCK WIDTH idx
          else s.readMem Out outAddr) := by
  intro idx
  simp [exec, lightning_attention_bwd_grad_store_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
        Tile.ptrAdd, NumericDType.add, NumericDType.mul, ComparableDType.lt,
        gradRowIndex, gradColIndex, activeGrad, gradTileOffset,
        TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BLOCK, WIDTH] → Nat :=
    fun idx =>
      s.pids 0 * n * width + (s.pids 1 * BLOCK + idx.1.val) * width +
        idx.2.1.val
  let valueFn : TileIndex [BLOCK, WIDTH] → ℝ :=
    fun idx =>
      WithBot.unbotD 0
        (if s.pids 1 * BLOCK + idx.1.val < n then
          some (s.readMem GradPre
            (s.pids 0 * n * width + (s.pids 1 * BLOCK + idx.1.val) * width +
              idx.2.1.val))
        else some (0.0 : ℝ))
  let P : TileIndex [BLOCK, WIDTH] → Prop :=
    fun idx => s.pids 1 * BLOCK + idx.1.val < n
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, gradTileOffset, gradRowIndex, gradColIndex] using hOutInj
  change (List.foldl
      (fun (acc : BlockState) i =>
        if P i then acc.writeMem Out (offsetFn i) (valueFn i) else acc)
      _ (TileShape.allIndices [BLOCK, WIDTH])).readMem Out
        (offsetFn idx) =
    if P idx then gradStoreValue s GradPre n width BLOCK WIDTH idx
    else s.readMem Out (offsetFn idx)
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj idx]
  by_cases hActive : s.pids 1 * BLOCK + idx.1.val < n
  · rfl
  · rfl

theorem lightning_attention_bwd_grad_store_slice_compute_correct
    (GradPre Out : RegionName) (n width BLOCK WIDTH : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK, WIDTH] =>
        gradTileOffset s n width BLOCK WIDTH idx)) :
    ComputeCorrect.Realizes
      (kernel := lightning_attention_bwd_grad_store_slice GradPre Out n width BLOCK WIDTH)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK, WIDTH] => activeGrad s n BLOCK idx)
        (fun idx : TileIndex [BLOCK, WIDTH] =>
          (Out, gradTileOffset s n width BLOCK WIDTH idx)))
      (expected := fun idx : TileIndex [BLOCK, WIDTH] =>
        gradStoreValue s GradPre n width BLOCK WIDTH idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [lightning_attention_bwd_grad_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := lightning_attention_bwd_grad_store_slice_correct GradPre Out n width
    BLOCK WIDTH s hOutInj idx
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-- Backward inter-kernel DQ accumulation slice.

This captures the Python `_bwd_inter_kernel` writeback
`dq = dq_inter + tl.load(DQ_block_ptr, ...)` before storing back into `DQ`.
The surrounding loop and `dq_inter = tl.dot(do, kv_trans)` producer remain
separate proof obligations, but this is stronger than a precomputed-gradient
store because it includes the Python-observable in-place DQ accumulation. -/
def lightning_attention_bwd_dq_accum_store_slice
    (DQInter DQ : RegionName) (n width BLOCK WIDTH : Nat) :
    ComputeKernel := triton {
  off_bh = tl.program_id(0)
  off_block = tl.program_id(1) * $(BLOCK) + tl.arange(0, $(BLOCK))
  offs_w = tl.arange(0, $(WIDTH))
  mask = (off_block[:, None] < $(n)) & (offs_w[None, :] < $(WIDTH))
  dq_inter = tl.load(DQInter + off_bh * $(n) * $(width) +
      off_block[:, None] * $(width) + offs_w[None, :],
      mask=mask, other=0.0)
  dq_prev = tl.load(DQ + off_bh * $(n) * $(width) +
      off_block[:, None] * $(width) + offs_w[None, :],
      mask=mask, other=0.0)
  dq = dq_inter + dq_prev
  tl.store(DQ + off_bh * $(n) * $(width) +
      off_block[:, None] * $(width) + offs_w[None, :],
      (dq).to(DQ.dtype.element_ty), mask=mask)
}

noncomputable def dqAccumStoreValue
    (s : BlockState) (DQInter DQ : RegionName)
    (n width BLOCK WIDTH : Nat) (idx : TileIndex [BLOCK, WIDTH]) : ℝ :=
  WithBot.unbotD 0
    (if activeGrad s n BLOCK idx then
      some (s.readMem DQInter (gradTileOffset s n width BLOCK WIDTH idx) +
        s.readMem DQ (gradTileOffset s n width BLOCK WIDTH idx))
    else some (0.0 : ℝ))

theorem lightning_attention_bwd_dq_accum_store_slice_correct
    (DQInter DQ : RegionName) (n width BLOCK WIDTH : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK, WIDTH] =>
        gradTileOffset s n width BLOCK WIDTH idx)) :
    ∀ idx : TileIndex [BLOCK, WIDTH],
      let outAddr := gradTileOffset s n width BLOCK WIDTH idx
      (exec (lightning_attention_bwd_dq_accum_store_slice DQInter DQ
          n width BLOCK WIDTH) s).map (·.readMem DQ outAddr)
        = some (if activeGrad s n BLOCK idx then
            dqAccumStoreValue s DQInter DQ n width BLOCK WIDTH idx
          else s.readMem DQ outAddr) := by
  intro idx
  simp [exec, lightning_attention_bwd_dq_accum_store_slice, stepStmts,
        stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop,
        Tile.expandDim, Tile.ptrAdd, NumericDType.add, NumericDType.mul,
        ComparableDType.lt, FloatDType.cast, FloatDType.ofWithBot,
        FloatDType.toWithBot, gradRowIndex, gradColIndex, activeGrad,
        gradTileOffset, TileShape.dropInsertedIndex, ComputeExpr.toAlgorithm?,
        ComputeOp.toAlgorithm?]
  let offsetFn : TileIndex [BLOCK, WIDTH] → Nat :=
    fun idx =>
      s.pids 0 * n * width + (s.pids 1 * BLOCK + idx.1.val) * width +
        idx.2.1.val
  let valueFn : TileIndex [BLOCK, WIDTH] → ℝ :=
    fun idx =>
      WithBot.unbotD 0
        (Option.map₂ (fun a b => a + b)
          (if s.pids 1 * BLOCK + idx.1.val < n then
            some (s.readMem DQInter (offsetFn idx))
          else some (0.0 : ℝ))
          (if s.pids 1 * BLOCK + idx.1.val < n then
            some (s.readMem DQ (offsetFn idx))
          else some (0.0 : ℝ)))
  let P : TileIndex [BLOCK, WIDTH] → Prop :=
    fun idx => s.pids 1 * BLOCK + idx.1.val < n
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, gradTileOffset, gradRowIndex, gradColIndex] using hOutInj
  change (List.foldl
      (fun (acc : BlockState) i =>
        if P i then acc.writeMem DQ (offsetFn i) (valueFn i) else acc)
      _ (TileShape.allIndices [BLOCK, WIDTH])).readMem DQ
        (offsetFn idx) =
    if P idx then dqAccumStoreValue s DQInter DQ n width BLOCK WIDTH idx
    else s.readMem DQ (offsetFn idx)
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj idx]
  by_cases hActive : s.pids 1 * BLOCK + idx.1.val < n
  · simp [offsetFn, valueFn, P, dqAccumStoreValue, activeGrad, gradTileOffset,
      gradRowIndex, gradColIndex, hActive, NumericDType.add]
  · simp [offsetFn, valueFn, P, dqAccumStoreValue, activeGrad, gradTileOffset,
      gradRowIndex, gradColIndex, hActive]

theorem lightning_attention_bwd_dq_accum_store_slice_compute_correct
    (DQInter DQ : RegionName) (n width BLOCK WIDTH : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK, WIDTH] =>
        gradTileOffset s n width BLOCK WIDTH idx)) :
    ComputeCorrect.Realizes
      (kernel := lightning_attention_bwd_dq_accum_store_slice DQInter DQ
        n width BLOCK WIDTH)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK, WIDTH] => activeGrad s n BLOCK idx)
        (fun idx : TileIndex [BLOCK, WIDTH] =>
          (DQ, gradTileOffset s n width BLOCK WIDTH idx)))
      (expected := fun idx : TileIndex [BLOCK, WIDTH] =>
        dqAccumStoreValue s DQInter DQ n width BLOCK WIDTH idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [lightning_attention_bwd_dq_accum_store_slice, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := lightning_attention_bwd_dq_accum_store_slice_correct DQInter DQ
    n width BLOCK WIDTH s hOutInj idx
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-- Backward inter-kernel DQ producer slice:
`dq_inter = tl.dot(do, kv_trans)`. This is the arithmetic producer consumed by
the DQ accumulation slice above. -/
def lightning_attention_bwd_dq_inter_dot_slice
    (DO KVTrans DQInter : RegionName) (BLOCK E WIDTH : Nat) :
    ComputeKernel := triton {
  offs_b = tl.arange(0, $(BLOCK))
  offs_e = tl.arange(0, $(E))
  offs_w = tl.arange(0, $(WIDTH))
  b_do = tl.load(DO + offs_b[:, None] * $(E) + offs_e[None, :])
  kv_trans = tl.load(KVTrans + offs_e[:, None] * $(WIDTH) + offs_w[None, :])
  dq_inter = tl.dot(b_do, kv_trans)
  tl.store(DQInter + offs_b[:, None] * $(WIDTH) + offs_w[None, :], dq_inter)
}

def dqInterOffset (WIDTH : Nat) (idx : TileIndex [BLOCK, WIDTH]) : Nat :=
  idx.1.val * WIDTH + idx.2.1.val

noncomputable def dqInterDotSpec
    (s : BlockState) (DO KVTrans : RegionName) (BLOCK E WIDTH : Nat)
    (idx : TileIndex [BLOCK, WIDTH]) : ℝ :=
  ∑ e : Fin E,
    s.readMem DO (idx.1.val * E + e.val) *
      s.readMem KVTrans (e.val * WIDTH + idx.2.1.val)

theorem lightning_attention_bwd_dq_inter_dot_slice_correct
    (DO KVTrans DQInter : RegionName) (BLOCK E WIDTH : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK, WIDTH] => dqInterOffset WIDTH idx)) :
    ∀ idx : TileIndex [BLOCK, WIDTH],
      let outAddr := dqInterOffset WIDTH idx
      (exec (lightning_attention_bwd_dq_inter_dot_slice DO KVTrans DQInter
          BLOCK E WIDTH) s).map (·.readMem DQInter outAddr)
        = some (dqInterDotSpec s DO KVTrans BLOCK E WIDTH idx) := by
  intro idx
  simp [exec, lightning_attention_bwd_dq_inter_dot_slice,
        ComputeKernel.toAlgKernel, ComputeStmt.toAlgorithm?,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop,
        Tile.expandDim, Tile.ptrAdd, Tile.dot, NumericDType.add,
        NumericDType.mul, dqInterOffset, dqInterDotSpec,
        TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BLOCK, WIDTH] → Nat :=
    fun idx => idx.1.val * WIDTH + idx.2.1.val
  have hInj : Function.Injective offsetFn := by
    simpa [offsetFn, dqInterOffset] using hOutInj
  rw [BlockState.scatter_readback_nd _ _ _ hInj idx]

theorem lightning_attention_bwd_dq_inter_dot_slice_compute_correct
    (DO KVTrans DQInter : RegionName) (BLOCK E WIDTH : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK, WIDTH] => dqInterOffset WIDTH idx)) :
    ComputeCorrect.Realizes
      (kernel := lightning_attention_bwd_dq_inter_dot_slice DO KVTrans DQInter
        BLOCK E WIDTH)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK, WIDTH] =>
        some (DQInter, dqInterOffset WIDTH idx))
      (expected := fun idx : TileIndex [BLOCK, WIDTH] =>
        dqInterDotSpec s DO KVTrans BLOCK E WIDTH idx) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [lightning_attention_bwd_dq_inter_dot_slice, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx
  have h := lightning_attention_bwd_dq_inter_dot_slice_correct DO KVTrans
    DQInter BLOCK E WIDTH s hOutInj idx
  rw [hExec] at h
  exact Option.some.inj h

/-- Named DQ writeback correctness for the Python backward path. -/
theorem lightning_attention_bwd_dq_store_slice_compute_correct
    (DQPre DQ : RegionName) (n width BLOCK WIDTH : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK, WIDTH] =>
        gradTileOffset s n width BLOCK WIDTH idx)) :
    ComputeCorrect.Realizes
      (kernel := lightning_attention_bwd_grad_store_slice DQPre DQ n width BLOCK WIDTH)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK, WIDTH] => activeGrad s n BLOCK idx)
        (fun idx : TileIndex [BLOCK, WIDTH] =>
          (DQ, gradTileOffset s n width BLOCK WIDTH idx)))
      (expected := fun idx : TileIndex [BLOCK, WIDTH] =>
        gradStoreValue s DQPre n width BLOCK WIDTH idx) := by
  exact lightning_attention_bwd_grad_store_slice_compute_correct DQPre DQ
    n width BLOCK WIDTH s hOutInj

/-- Named DK writeback correctness for the Python backward path. -/
theorem lightning_attention_bwd_dk_store_slice_compute_correct
    (DKPre DK : RegionName) (n width BLOCK WIDTH : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK, WIDTH] =>
        gradTileOffset s n width BLOCK WIDTH idx)) :
    ComputeCorrect.Realizes
      (kernel := lightning_attention_bwd_grad_store_slice DKPre DK n width BLOCK WIDTH)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK, WIDTH] => activeGrad s n BLOCK idx)
        (fun idx : TileIndex [BLOCK, WIDTH] =>
          (DK, gradTileOffset s n width BLOCK WIDTH idx)))
      (expected := fun idx : TileIndex [BLOCK, WIDTH] =>
        gradStoreValue s DKPre n width BLOCK WIDTH idx) := by
  exact lightning_attention_bwd_grad_store_slice_compute_correct DKPre DK
    n width BLOCK WIDTH s hOutInj

/-- Named DV writeback correctness for the Python backward path. -/
theorem lightning_attention_bwd_dv_store_slice_compute_correct
    (DVPre DV : RegionName) (n width BLOCK WIDTH : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK, WIDTH] =>
        gradTileOffset s n width BLOCK WIDTH idx)) :
    ComputeCorrect.Realizes
      (kernel := lightning_attention_bwd_grad_store_slice DVPre DV n width BLOCK WIDTH)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK, WIDTH] => activeGrad s n BLOCK idx)
        (fun idx : TileIndex [BLOCK, WIDTH] =>
          (DV, gradTileOffset s n width BLOCK WIDTH idx)))
      (expected := fun idx : TileIndex [BLOCK, WIDTH] =>
        gradStoreValue s DVPre n width BLOCK WIDTH idx) := by
  exact lightning_attention_bwd_grad_store_slice_compute_correct DVPre DV
    n width BLOCK WIDTH s hOutInj

/-! ## Python test-shape wrappers

The checked Python test uses `b = 2`, `h = 8`, `n = 128`, `d = 64`, and
`e = 128`. The forward launcher fixes `BLOCK = 64`, `NUM_BLOCK = 2`, and
`BLOCK_MODEL = 32`, so each forward output tile has shape `[64, 32]`. The
backward launcher uses `BLOCK = 64`; DQ/DK row tiles have width `64`, and DV
row tiles have width `128`. -/

theorem lightning_attention_forward_python_test_shape_offset_injective
    (s : BlockState) :
    Function.Injective
      (fun idx : TileIndex [64, 32] =>
        tileOffset s 128 128 64 32 idx) := by
  intro a b h
  rcases a with ⟨ra, ca, hta⟩
  rcases b with ⟨rb, cb, htb⟩
  rcases ca with ⟨ca, hca⟩
  rcases cb with ⟨cb, hcb⟩
  simp [tileOffset, rowIndex, colIndex] at h
  have hr : ra = rb := by omega
  have hc : ca = cb := by omega
  subst rb
  subst cb
  rfl

theorem lightning_attention_bwd_qk_python_test_shape_offset_injective
    (s : BlockState) :
    Function.Injective
      (fun idx : TileIndex [64, 64] =>
        gradTileOffset s 128 64 64 64 idx) := by
  intro a b h
  rcases a with ⟨ra, ca, hta⟩
  rcases b with ⟨rb, cb, htb⟩
  rcases ca with ⟨ca, hca⟩
  rcases cb with ⟨cb, hcb⟩
  simp [gradTileOffset, gradRowIndex, gradColIndex] at h
  have hr : ra = rb := by omega
  have hc : ca = cb := by omega
  subst rb
  subst cb
  rfl

theorem lightning_attention_bwd_v_python_test_shape_offset_injective
    (s : BlockState) :
    Function.Injective
      (fun idx : TileIndex [64, 128] =>
        gradTileOffset s 128 128 64 128 idx) := by
  intro a b h
  rcases a with ⟨ra, ca, hta⟩
  rcases b with ⟨rb, cb, htb⟩
  rcases ca with ⟨ca, hca⟩
  rcases cb with ⟨cb, hcb⟩
  simp [gradTileOffset, gradRowIndex, gradColIndex] at h
  have hr : ra = rb := by omega
  have hc : ca = cb := by omega
  subst rb
  subst cb
  rfl

theorem lightning_attention_forward_store_python_test_shape_compute_correct
    (OAcc Out : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := lightning_attention_forward_store_slice OAcc Out 128 128 64 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [64, 32] => active s 128 64 idx)
        (fun idx : TileIndex [64, 32] =>
          (Out, tileOffset s 128 128 64 32 idx)))
      (expected := fun idx : TileIndex [64, 32] =>
        storeValue s OAcc 128 128 64 32 idx) := by
  exact lightning_attention_forward_store_slice_compute_correct OAcc Out
    128 128 64 32 s
    (lightning_attention_forward_python_test_shape_offset_injective s)

theorem lightning_attention_forward_sum_store_python_test_shape_compute_correct
    (OIntra OInter Out : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := lightning_attention_forward_sum_store_slice OIntra OInter Out
        128 128 64 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [64, 32] => active s 128 64 idx)
        (fun idx : TileIndex [64, 32] =>
          (Out, tileOffset s 128 128 64 32 idx)))
      (expected := fun idx : TileIndex [64, 32] =>
        sumStoreValue s OIntra OInter 128 128 64 32 idx) := by
  exact lightning_attention_forward_sum_store_slice_compute_correct OIntra
    OInter Out 128 128 64 32 s
    (lightning_attention_forward_python_test_shape_offset_injective s)

theorem lightning_attention_bwd_dq_store_python_test_shape_compute_correct
    (DQPre DQ : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := lightning_attention_bwd_grad_store_slice DQPre DQ 128 64 64 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [64, 64] => activeGrad s 128 64 idx)
        (fun idx : TileIndex [64, 64] =>
          (DQ, gradTileOffset s 128 64 64 64 idx)))
      (expected := fun idx : TileIndex [64, 64] =>
        gradStoreValue s DQPre 128 64 64 64 idx) := by
  exact lightning_attention_bwd_dq_store_slice_compute_correct DQPre DQ
    128 64 64 64 s
    (lightning_attention_bwd_qk_python_test_shape_offset_injective s)

theorem lightning_attention_bwd_dq_accum_python_test_shape_compute_correct
    (DQInter DQ : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := lightning_attention_bwd_dq_accum_store_slice DQInter DQ
        128 64 64 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [64, 64] => activeGrad s 128 64 idx)
        (fun idx : TileIndex [64, 64] =>
          (DQ, gradTileOffset s 128 64 64 64 idx)))
      (expected := fun idx : TileIndex [64, 64] =>
        dqAccumStoreValue s DQInter DQ 128 64 64 64 idx) := by
  exact lightning_attention_bwd_dq_accum_store_slice_compute_correct DQInter DQ
    128 64 64 64 s
    (lightning_attention_bwd_qk_python_test_shape_offset_injective s)

theorem lightning_attention_bwd_dq_inter_offset_python_test_shape_injective :
    Function.Injective
      (fun idx : TileIndex [64, 64] => dqInterOffset 64 idx) := by
  rintro ⟨⟨ra, hra⟩, ⟨ca, hca⟩, _⟩ ⟨⟨rb, hrb⟩, ⟨cb, hcb⟩, _⟩ h
  simp [dqInterOffset] at h
  have hr : ra = rb := by omega
  have hc : ca = cb := by omega
  subst rb
  subst cb
  rfl

theorem lightning_attention_bwd_dq_inter_dot_python_test_shape_compute_correct
    (DO KVTrans DQInter : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := lightning_attention_bwd_dq_inter_dot_slice DO KVTrans DQInter
        64 128 64)
      (initialState := s)
      (write := fun idx : TileIndex [64, 64] =>
        some (DQInter, dqInterOffset 64 idx))
      (expected := fun idx : TileIndex [64, 64] =>
        dqInterDotSpec s DO KVTrans 64 128 64 idx) := by
  exact lightning_attention_bwd_dq_inter_dot_slice_compute_correct DO KVTrans
    DQInter 64 128 64 s
    lightning_attention_bwd_dq_inter_offset_python_test_shape_injective

theorem lightning_attention_bwd_dk_store_python_test_shape_compute_correct
    (DKPre DK : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := lightning_attention_bwd_grad_store_slice DKPre DK 128 64 64 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [64, 64] => activeGrad s 128 64 idx)
        (fun idx : TileIndex [64, 64] =>
          (DK, gradTileOffset s 128 64 64 64 idx)))
      (expected := fun idx : TileIndex [64, 64] =>
        gradStoreValue s DKPre 128 64 64 64 idx) := by
  exact lightning_attention_bwd_dk_store_slice_compute_correct DKPre DK
    128 64 64 64 s
    (lightning_attention_bwd_qk_python_test_shape_offset_injective s)

theorem lightning_attention_bwd_dv_store_python_test_shape_compute_correct
    (DVPre DV : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := lightning_attention_bwd_grad_store_slice DVPre DV 128 128 64 128)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [64, 128] => activeGrad s 128 64 idx)
        (fun idx : TileIndex [64, 128] =>
          (DV, gradTileOffset s 128 128 64 128 idx)))
      (expected := fun idx : TileIndex [64, 128] =>
        gradStoreValue s DVPre 128 128 64 128 idx) := by
  exact lightning_attention_bwd_dv_store_slice_compute_correct DVPre DV
    128 128 64 128 s
    (lightning_attention_bwd_v_python_test_shape_offset_injective s)

/-- Python forward test-shape output coverage for Lightning Attention: both the
direct precomputed output writeback and the summed intra/inter writeback realize
the masked `Out` store shape used by the checked launcher. -/
theorem lightning_attention_forward_python_test_shape_all_outputs_compute_correct
    (OAcc OIntra OInter Out : RegionName) (s : BlockState) :
    (ComputeCorrect.Realizes
      (kernel := lightning_attention_forward_store_slice OAcc Out 128 128 64 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [64, 32] => active s 128 64 idx)
        (fun idx : TileIndex [64, 32] =>
          (Out, tileOffset s 128 128 64 32 idx)))
      (expected := fun idx : TileIndex [64, 32] =>
        storeValue s OAcc 128 128 64 32 idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := lightning_attention_forward_sum_store_slice OIntra OInter Out
        128 128 64 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [64, 32] => active s 128 64 idx)
        (fun idx : TileIndex [64, 32] =>
          (Out, tileOffset s 128 128 64 32 idx)))
      (expected := fun idx : TileIndex [64, 32] =>
        sumStoreValue s OIntra OInter 128 128 64 32 idx)) := by
  constructor
  · exact lightning_attention_forward_store_python_test_shape_compute_correct
      OAcc Out s
  · exact lightning_attention_forward_sum_store_python_test_shape_compute_correct
      OIntra OInter Out s

/-- Python backward test-shape output coverage for Lightning Attention: the DQ
inter-kernel accumulation and the final `DQ`/`DK`/`DV` gradient writebacks all
realize the masked store shapes used by the checked launcher. -/
theorem lightning_attention_bwd_python_test_shape_all_outputs_compute_correct
    (DQInter DQPre DKPre DVPre DQ DK DV : RegionName) (s : BlockState) :
    (ComputeCorrect.Realizes
      (kernel := lightning_attention_bwd_dq_accum_store_slice DQInter DQ
        128 64 64 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [64, 64] => activeGrad s 128 64 idx)
        (fun idx : TileIndex [64, 64] =>
          (DQ, gradTileOffset s 128 64 64 64 idx)))
      (expected := fun idx : TileIndex [64, 64] =>
        dqAccumStoreValue s DQInter DQ 128 64 64 64 idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := lightning_attention_bwd_grad_store_slice DQPre DQ
        128 64 64 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [64, 64] => activeGrad s 128 64 idx)
        (fun idx : TileIndex [64, 64] =>
          (DQ, gradTileOffset s 128 64 64 64 idx)))
      (expected := fun idx : TileIndex [64, 64] =>
        gradStoreValue s DQPre 128 64 64 64 idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := lightning_attention_bwd_grad_store_slice DKPre DK
        128 64 64 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [64, 64] => activeGrad s 128 64 idx)
        (fun idx : TileIndex [64, 64] =>
          (DK, gradTileOffset s 128 64 64 64 idx)))
      (expected := fun idx : TileIndex [64, 64] =>
        gradStoreValue s DKPre 128 64 64 64 idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := lightning_attention_bwd_grad_store_slice DVPre DV
        128 128 64 128)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [64, 128] => activeGrad s 128 64 idx)
        (fun idx : TileIndex [64, 128] =>
          (DV, gradTileOffset s 128 128 64 128 idx)))
      (expected := fun idx : TileIndex [64, 128] =>
        gradStoreValue s DVPre 128 128 64 128 idx)) := by
  constructor
  · exact lightning_attention_bwd_dq_accum_python_test_shape_compute_correct
      DQInter DQ s
  constructor
  · exact lightning_attention_bwd_dq_store_python_test_shape_compute_correct
      DQPre DQ s
  constructor
  · exact lightning_attention_bwd_dk_store_python_test_shape_compute_correct
      DKPre DK s
  · exact lightning_attention_bwd_dv_store_python_test_shape_compute_correct
      DVPre DV s

/-- Checked-shape producer for the DQ inter-kernel accumulator used before the
in-place DQ accumulation writeback. -/
theorem lightning_attention_bwd_dq_inter_python_test_shape_formula_summary
    (DO KVTrans DQInter : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := lightning_attention_bwd_dq_inter_dot_slice DO KVTrans DQInter
        64 128 64)
      (initialState := s)
      (write := fun idx : TileIndex [64, 64] =>
        some (DQInter, dqInterOffset 64 idx))
      (expected := fun idx : TileIndex [64, 64] =>
        dqInterDotSpec s DO KVTrans 64 128 64 idx) := by
  exact lightning_attention_bwd_dq_inter_dot_python_test_shape_compute_correct
    DO KVTrans DQInter s

/-- Python test-shape lowering proposition for all Lightning Attention kernels
launched by `LightningAttention2NoDecay`: one forward kernel plus the intra- and
inter-block backward kernels. -/
abbrev lightning_attention_python_test_shape_surface_prop
    (Q K V Out DO DQ DK DV : RegionName) : Prop :=
    (∃ alg, (lightning_attention_forward_surface Q K V Out 2 8 128 64 128
      64 2 32).toAlgorithm? = Except.ok alg) ∧
    (∃ alg, (lightning_attention_bwd_intra_surface Q K V DO DQ DK DV
      2 8 128 64 128 64 2 32 2).toAlgorithm? = Except.ok alg) ∧
    (∃ alg, (lightning_attention_bwd_inter_surface Q K V DO DQ DK DV
      2 8 128 64 128 64 2 32 2).toAlgorithm? = Except.ok alg)

/-- Python test-shape output proposition for the checked result tuple:
forward `Out`, plus backward `DQ`, `DK`, and `DV`. -/
abbrev lightning_attention_python_test_shape_outputs_prop
    (OAcc OIntra OInter DQInter DQPre DKPre DVPre Out DQ DK DV : RegionName)
    (s : BlockState) : Prop :=
    (ComputeCorrect.Realizes
      (kernel := lightning_attention_forward_store_slice OAcc Out 128 128 64 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [64, 32] => active s 128 64 idx)
        (fun idx : TileIndex [64, 32] =>
          (Out, tileOffset s 128 128 64 32 idx)))
      (expected := fun idx : TileIndex [64, 32] =>
        storeValue s OAcc 128 128 64 32 idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := lightning_attention_forward_sum_store_slice OIntra OInter Out
        128 128 64 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [64, 32] => active s 128 64 idx)
        (fun idx : TileIndex [64, 32] =>
          (Out, tileOffset s 128 128 64 32 idx)))
      (expected := fun idx : TileIndex [64, 32] =>
        sumStoreValue s OIntra OInter 128 128 64 32 idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := lightning_attention_bwd_dq_accum_store_slice DQInter DQ
        128 64 64 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [64, 64] => activeGrad s 128 64 idx)
        (fun idx : TileIndex [64, 64] =>
          (DQ, gradTileOffset s 128 64 64 64 idx)))
      (expected := fun idx : TileIndex [64, 64] =>
        dqAccumStoreValue s DQInter DQ 128 64 64 64 idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := lightning_attention_bwd_grad_store_slice DQPre DQ
        128 64 64 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [64, 64] => activeGrad s 128 64 idx)
        (fun idx : TileIndex [64, 64] =>
          (DQ, gradTileOffset s 128 64 64 64 idx)))
      (expected := fun idx : TileIndex [64, 64] =>
        gradStoreValue s DQPre 128 64 64 64 idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := lightning_attention_bwd_grad_store_slice DKPre DK
        128 64 64 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [64, 64] => activeGrad s 128 64 idx)
        (fun idx : TileIndex [64, 64] =>
          (DK, gradTileOffset s 128 64 64 64 idx)))
      (expected := fun idx : TileIndex [64, 64] =>
        gradStoreValue s DKPre 128 64 64 64 idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := lightning_attention_bwd_grad_store_slice DVPre DV
        128 128 64 128)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [64, 128] => activeGrad s 128 64 idx)
        (fun idx : TileIndex [64, 128] =>
          (DV, gradTileOffset s 128 128 64 128 idx)))
      (expected := fun idx : TileIndex [64, 128] =>
        gradStoreValue s DVPre 128 128 64 128 idx))

/-- End-to-end checked-shape summary for `test_lightning_attention2_no_decay`.

This ties the exact Python-launched forward/backward DSL surfaces to the
observable result tuple `Out`, `DQ`, `DK`, and `DV`. The arithmetic loop bodies
are preserved as surfaces; the compute-correct part remains at the existing
masked output-slice granularity. -/
theorem lightning_attention_python_test_shape_output_summary
    (Q K V Out DO DQ DK DV OAcc OIntra OInter DQInter DQPre DKPre DVPre :
      RegionName)
    (s : BlockState) :
    lightning_attention_python_test_shape_surface_prop Q K V Out DO DQ DK DV ∧
    lightning_attention_python_test_shape_outputs_prop OAcc OIntra OInter
      DQInter DQPre DKPre DVPre Out DQ DK DV s := by
  constructor
  · constructor
    · exact lightning_attention_forward_surface_toAlgorithm_supported Q K V Out
        2 8 128 64 128 64 2 32
    constructor
    · exact lightning_attention_bwd_intra_surface_toAlgorithm_supported Q K V
        DO DQ DK DV 2 8 128 64 128 64 2 32 2
    · exact lightning_attention_bwd_inter_surface_toAlgorithm_supported Q K V
        DO DQ DK DV 2 8 128 64 128 64 2 32 2
  · constructor
    · exact (lightning_attention_forward_python_test_shape_all_outputs_compute_correct
        OAcc OIntra OInter Out s).1
    constructor
    · exact (lightning_attention_forward_python_test_shape_all_outputs_compute_correct
        OAcc OIntra OInter Out s).2
    exact lightning_attention_bwd_python_test_shape_all_outputs_compute_correct
      DQInter DQPre DKPre DVPre DQ DK DV s

/-- Combined checked-shape summary for `test_lightning_attention2_no_decay`.

This exposes the launched forward/backward surfaces, observable `Out`/`DQ`/`DK`/`DV`
writebacks, and the `DQInter = dot(DO, KVTrans)` producer in one public target. -/
theorem lightning_attention_python_test_shape_complete_summary
    (Q K V Out DO DQ DK DV OAcc OIntra OInter DQInter DQPre DKPre DVPre
      KVTrans : RegionName)
    (s : BlockState) :
    (lightning_attention_python_test_shape_surface_prop Q K V Out DO DQ DK DV ∧
      lightning_attention_python_test_shape_outputs_prop OAcc OIntra OInter
        DQInter DQPre DKPre DVPre Out DQ DK DV s) ∧
    ComputeCorrect.Realizes
      (kernel := lightning_attention_bwd_dq_inter_dot_slice DO KVTrans DQInter
        64 128 64)
      (initialState := s)
      (write := fun idx : TileIndex [64, 64] =>
        some (DQInter, dqInterOffset 64 idx))
      (expected := fun idx : TileIndex [64, 64] =>
        dqInterDotSpec s DO KVTrans 64 128 64 idx) := by
  constructor
  · exact lightning_attention_python_test_shape_output_summary Q K V Out DO DQ
      DK DV OAcc OIntra OInter DQInter DQPre DKPre DVPre s
  · exact lightning_attention_bwd_dq_inter_python_test_shape_formula_summary DO
      KVTrans DQInter s

end VeriTile.Bench.TritonBenchG.LightningAttention
