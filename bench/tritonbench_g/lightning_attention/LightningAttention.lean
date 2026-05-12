import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.LightningAttention

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-- Surface transcription of `lightning_attention.py`'s `_fwd_kernel`.

This covers the full forward recurrent tile loop. The Python line `off_bh % h`
is a no-op expression statement and is intentionally not represented because
the DSL has no bare expression statement form. -/
def lightning_attention_forward_surface
    (Q K V Out : RegionName)
    (_b _h n d e BLOCK NUM_BLOCK BLOCK_MODEL : Nat) :
    ComputeKernel := triton {
  off_bh = tl.program_id(axis=0)
  off_e = tl.program_id(axis=1)
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

/-- Surface transcription/proof-oriented forward output-store slice of
`lightning_attention.py`'s `_fwd_kernel`.

The full kernel computes a recurrent attention tile `o`. This slice starts from
a precomputed `OAcc` tile and proves the masked writeback into `Out`. The full forward recurrent `kv` loop is represented above by
`lightning_attention_forward_surface`; the backward kernels' negative-step loops
remain separate modeling work. -/
def lightning_attention_forward_store_slice
    (OAcc Out : RegionName) (n e BLOCK BLOCK_MODEL : Nat) :
    ComputeKernel := triton {
  off_bh = tl.program_id(axis=0)
  off_e = tl.program_id(axis=1)
  off_block = tl.program_id(axis=2) * $(BLOCK) + tl.arange(0, $(BLOCK))
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
        evalOp, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.remap,
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
  · rfl
  · rfl

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

end VeriTile.Bench.TritonBenchG.LightningAttention
