import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
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
        evalOp, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.remap,
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
        evalOp, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
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

end VeriTile.Bench.TritonBenchG.LightningAttention
