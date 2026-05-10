import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.DestindexCopyKv2

open VeriTile.Triton

/-- Faithful transcription of `destindex_copy_kv2.py`'s
`_fwd_kernel_destindex_copy_kv`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_DMODEL: tl.constexpr` / `BLOCK_HEAD: tl.constexpr` -> Lean
  `Nat` parameters.
- The Python head-only mask is made explicitly two-dimensional with the
  tautological `offs_d < BLOCK_DMODEL` conjunct so the current DSL does not
  need to infer that broadcast. -/
def fwd_kernel_destindex_copy_kv
    (K Dest_loc Out : RegionName)
    (stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
      head_num BLOCK_DMODEL BLOCK_HEAD : Nat) :
    ComputeKernel := triton {
  cur_index = tl.program_id(0)
  offs_h = tl.arange(0, $(BLOCK_HEAD))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  dest_index = tl.load(Dest_loc + cur_index)
  k_ptrs = K + cur_index * $(stride_k_bs) +
    $(stride_k_h) * offs_h[:, None] + $(stride_k_d) * offs_d[None, :]
  o_ptrs = Out + dest_index * $(stride_o_bs) +
    $(stride_o_h) * offs_h[:, None] + $(stride_o_d) * offs_d[None, :]
  k = tl.load(k_ptrs,
    mask=(offs_h[:, None] < $(head_num)) and (offs_d[None, :] < $(BLOCK_DMODEL)),
    other=0.0)
  tl.store(o_ptrs, k,
    mask=(offs_h[:, None] < $(head_num)) and (offs_d[None, :] < $(BLOCK_DMODEL)))
}

def headIndex (idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL]) : Nat :=
  idx.1.val

def dimIndex (idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL]) : Nat :=
  idx.2.1.val

def sourceAddr
    (s : BlockState) (stride_k_bs stride_k_h stride_k_d : Nat)
    (idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL]) : Nat :=
  s.pid * stride_k_bs + stride_k_h * headIndex idx + stride_k_d * dimIndex idx

def destBase (s : BlockState) (Dest_loc : RegionName) : Nat :=
  s.readMemValue .nat Dest_loc s.pid

def outAddr
    (s : BlockState) (Dest_loc : RegionName)
    (stride_o_bs stride_o_h stride_o_d : Nat)
    (idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL]) : Nat :=
  destBase s Dest_loc * stride_o_bs + stride_o_h * headIndex idx + stride_o_d * dimIndex idx

def active (head_num BLOCK_DMODEL : Nat) (idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL]) : Prop :=
  headIndex idx < head_num ∧ dimIndex idx < BLOCK_DMODEL

instance activeDecidable (head_num BLOCK_DMODEL : Nat)
    (idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL]) :
    Decidable (active head_num BLOCK_DMODEL idx) := by
  unfold active
  infer_instance

/-- Algorithm-layer cellwise correctness for `_fwd_kernel_destindex_copy_kv`. -/
theorem fwd_kernel_destindex_copy_kv_correct
    (K Dest_loc Out : RegionName)
    (stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
      head_num BLOCK_DMODEL BLOCK_HEAD : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL] =>
        outAddr s Dest_loc stride_o_bs stride_o_h stride_o_d idx)) :
    ∀ idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL],
      (exec (fwd_kernel_destindex_copy_kv K Dest_loc Out
          stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
          head_num BLOCK_DMODEL BLOCK_HEAD) s).map
          (fun s' => s'.readMem Out
            (outAddr s Dest_loc stride_o_bs stride_o_h stride_o_d idx))
        = some (if active head_num BLOCK_DMODEL idx then
            s.readMem K (sourceAddr s stride_k_bs stride_k_h stride_k_d idx)
          else
            s.readMem Out (outAddr s Dest_loc stride_o_bs stride_o_h stride_o_d idx)) := by
  intro idx
  simp [exec, fwd_kernel_destindex_copy_kv, stepStmts, stepStmt, evalOp,
        Tile.bop, Tile.cop, Tile.ptrAdd, Tile.expandDim,
        NumericDType.add, NumericDType.mul, ComparableDType.lt,
        BlockState.readMemValue, Option.bind, Option.map,
        TileShape.insertAxis, TileShape.dropInsertedIndex]
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL] =>
        (match s.readMemTyped TileDType.nat Dest_loc (s.pids 0) with
          | some value => value
          | none => BlockState.defaultCarrier TileDType.nat) * stride_o_bs +
          stride_o_h * idx.1.val + stride_o_d * idx.2.1.val) := by
    simpa [outAddr, destBase, headIndex, dimIndex, BlockState.pid_eq,
      BlockState.readMemValue] using hOutInj
  simp [active, outAddr, sourceAddr, destBase, headIndex, dimIndex,
        BlockState.pid_eq, BlockState.readMemValue]
  by_cases hHead : idx.1.val < head_num
  · have hDim : idx.2.1.val < BLOCK_DMODEL := idx.2.1.isLt
    simpa [hHead, hDim] using
      (BlockState.scatter_readback_prop_masked_nd
        (region := Out)
        (shape := [BLOCK_HEAD, BLOCK_DMODEL])
        (offsetFn := fun i : TileIndex [BLOCK_HEAD, BLOCK_DMODEL] =>
          (match s.readMemTyped TileDType.nat Dest_loc (s.pids 0) with
            | some value => value
            | none => BlockState.defaultCarrier TileDType.nat) * stride_o_bs +
            stride_o_h * i.1.val + stride_o_d * i.2.1.val)
        (valueFn := fun i : TileIndex [BLOCK_HEAD, BLOCK_DMODEL] =>
          WithBot.unbotD 0
            (if i.1.val < head_num ∧ i.2.1.val < BLOCK_DMODEL then
              some (s.readMem K
                (s.pids 0 * stride_k_bs + stride_k_h * i.1.val +
                  stride_k_d * i.2.1.val))
            else some 0.0))
        (P := fun i : TileIndex [BLOCK_HEAD, BLOCK_DMODEL] =>
          i.1.val < head_num ∧ i.2.1.val < BLOCK_DMODEL)
        _ hRawInj idx)
  · simpa [hHead] using
      (BlockState.scatter_readback_prop_masked_nd
        (region := Out)
        (shape := [BLOCK_HEAD, BLOCK_DMODEL])
        (offsetFn := fun i : TileIndex [BLOCK_HEAD, BLOCK_DMODEL] =>
          (match s.readMemTyped TileDType.nat Dest_loc (s.pids 0) with
            | some value => value
            | none => BlockState.defaultCarrier TileDType.nat) * stride_o_bs +
            stride_o_h * i.1.val + stride_o_d * i.2.1.val)
        (valueFn := fun i : TileIndex [BLOCK_HEAD, BLOCK_DMODEL] =>
          WithBot.unbotD 0
            (if i.1.val < head_num ∧ i.2.1.val < BLOCK_DMODEL then
              some (s.readMem K
                (s.pids 0 * stride_k_bs + stride_k_h * i.1.val +
                  stride_k_d * i.2.1.val))
            else some 0.0))
        (P := fun i : TileIndex [BLOCK_HEAD, BLOCK_DMODEL] =>
          i.1.val < head_num ∧ i.2.1.val < BLOCK_DMODEL)
        _ hRawInj idx)

/-- Executed-state form of `fwd_kernel_destindex_copy_kv_correct`. -/
theorem fwd_kernel_destindex_copy_kv_correct_of_exec
    (K Dest_loc Out : RegionName)
    (stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
      head_num BLOCK_DMODEL BLOCK_HEAD : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL] =>
        outAddr s Dest_loc stride_o_bs stride_o_h stride_o_d idx))
    (s' : BlockState)
    (hExec : exec (fwd_kernel_destindex_copy_kv K Dest_loc Out
        stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
        head_num BLOCK_DMODEL BLOCK_HEAD) s = some s') :
    ∀ idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL],
      s'.readMem Out (outAddr s Dest_loc stride_o_bs stride_o_h stride_o_d idx)
        = if active head_num BLOCK_DMODEL idx then
            s.readMem K (sourceAddr s stride_k_bs stride_k_h stride_k_d idx)
          else
            s.readMem Out (outAddr s Dest_loc stride_o_bs stride_o_h stride_o_d idx) := by
  intro idx
  have h := fwd_kernel_destindex_copy_kv_correct K Dest_loc Out
    stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
    head_num BLOCK_DMODEL BLOCK_HEAD s hOutInj idx
  rw [hExec] at h
  simpa using h

/-- Compute-facing correctness for `_fwd_kernel_destindex_copy_kv`. -/
theorem fwd_kernel_destindex_copy_kv_compute_correct
    (K Dest_loc Out : RegionName)
    (stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
      head_num BLOCK_DMODEL BLOCK_HEAD : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL] =>
        outAddr s Dest_loc stride_o_bs stride_o_h stride_o_d idx)) :
    ComputeCorrect.Realizes
      (kernel := fwd_kernel_destindex_copy_kv K Dest_loc Out
        stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
        head_num BLOCK_DMODEL BLOCK_HEAD)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL] => active head_num BLOCK_DMODEL idx)
        (fun idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL] =>
          (Out, outAddr s Dest_loc stride_o_bs stride_o_h stride_o_d idx)))
      (expected := fun idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL] =>
        s.readMem K (sourceAddr s stride_k_bs stride_k_h stride_k_d idx)) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [fwd_kernel_destindex_copy_kv]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := fwd_kernel_destindex_copy_kv_correct_of_exec K Dest_loc Out
    stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
    head_num BLOCK_DMODEL BLOCK_HEAD s hOutInj s' hExec idx
  simpa [hActive] using h

end VeriTile.Bench.TritonBenchG.DestindexCopyKv2
