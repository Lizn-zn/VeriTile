import VeriTile.Triton
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.DestindexCopyKv2.RealReference

open VeriTile.Triton
set_option maxHeartbeats 8000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def writeAddress (row : Int) (bs hs ds h d : Nat) : Nat :=
  (row*(bs:Int) + (hs*h:Nat) + (ds*d:Nat)).toNat

/-- Faithful transcription of `destindex_copy_kv2.py`'s
`_fwd_kernel_destindex_copy_kv`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_DMODEL: tl.constexpr` / `BLOCK_HEAD: tl.constexpr` -> Lean
  `Nat` parameters. -/
def fwd_kernel_destindex_copy_kv
    (K : RegionName) (Dest_loc : Region .int) (Out : RegionName)
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
  k = tl.load(k_ptrs, mask=offs_h[:, None] < $(head_num), other=0.0)
  tl.store(o_ptrs, k, mask=offs_h[:, None] < $(head_num))
}

def headIndex (idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL]) : Nat :=
  idx.1.val

def dimIndex (idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL]) : Nat :=
  idx.2.1.val

def sourceAddr
    (s : BlockState) (stride_k_bs stride_k_h stride_k_d : Nat)
    (idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL]) : Nat :=
  s.pid * stride_k_bs + stride_k_h * headIndex idx + stride_k_d * dimIndex idx

def destBase (s : BlockState) (Dest_loc : RegionName) : Int :=
  s.readMemValue .int Dest_loc s.pid

def outAddr
    (s : BlockState) (Dest_loc : RegionName)
    (stride_o_bs stride_o_h stride_o_d : Nat)
    (idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL]) : Nat :=
  writeAddress (destBase s Dest_loc) stride_o_bs stride_o_h stride_o_d (headIndex idx) (dimIndex idx)

def active (head_num : Nat) (idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL]) : Prop :=
  headIndex idx < head_num

instance activeDecidable (head_num : Nat)
    (idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL]) :
    Decidable (active head_num idx) := by
  unfold active
  infer_instance

@[simp] theorem headMask_remap_data
    (head_num BLOCK_HEAD BLOCK_DMODEL : Nat)
    (idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL]) :
    (@Tile.remap TileDType.bool [BLOCK_HEAD, 1] [BLOCK_HEAD, BLOCK_DMODEL]
        Broadcast.nil.consL.consSame.leftIndex
        ({ data := fun i : TileIndex [BLOCK_HEAD, 1] =>
          decide (i.1.val < head_num) } : Tile TileDType.bool [BLOCK_HEAD, 1])).data idx =
      decide (idx.1.val < head_num) := rfl

/-- Algorithm-layer cellwise correctness for `_fwd_kernel_destindex_copy_kv`. -/
theorem fwd_kernel_destindex_copy_kv_correct
    (K : RegionName) (Dest_loc : Region .int) (Out : RegionName)
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
        = some (if active head_num idx then
            s.readMem K (sourceAddr s stride_k_bs stride_k_h stride_k_d idx)
          else
            s.readMem Out (outAddr s Dest_loc stride_o_bs stride_o_h stride_o_d idx)) := by
  intro idx
  simp [exec, fwd_kernel_destindex_copy_kv, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Tile.bop, Tile.cop, Tile.ptrAdd, Tile.expandDim,
        NumericDType.add, NumericDType.mul, Tile.uop, writeAddress, ComparableDType.lt,
        BlockState.readMemValue, Option.bind, Option.map,
        TileShape.insertAxis, TileShape.dropInsertedIndex]
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL] =>
        writeAddress (match s.readMemTyped TileDType.int Dest_loc (s.pids 0) with
          | some value => value
          | none => BlockState.defaultCarrier TileDType.int) stride_o_bs stride_o_h stride_o_d idx.1.val idx.2.1.val) := by
    simpa [outAddr, destBase, headIndex, dimIndex, BlockState.pid_eq,
      BlockState.readMemValue] using hOutInj
  simp [active, outAddr, sourceAddr, destBase, headIndex, dimIndex,
        BlockState.pid_eq, BlockState.readMemValue]
  by_cases hHead : idx.1.val < head_num
  · simpa [hHead] using
      (BlockState.scatter_readback_prop_masked_nd
        (region := Out)
        (shape := [BLOCK_HEAD, BLOCK_DMODEL])
        (offsetFn := fun i : TileIndex [BLOCK_HEAD, BLOCK_DMODEL] =>
          writeAddress (match s.readMemTyped TileDType.int Dest_loc (s.pids 0) with
            | some value => value
            | none => BlockState.defaultCarrier TileDType.int) stride_o_bs stride_o_h stride_o_d i.1.val i.2.1.val)
        (valueFn := fun i : TileIndex [BLOCK_HEAD, BLOCK_DMODEL] =>
          WithBot.unbotD 0
            (if i.1.val < head_num then
              some (s.readMem K
                (s.pids 0 * stride_k_bs + stride_k_h * i.1.val +
                  stride_k_d * i.2.1.val))
            else some 0.0))
        (P := fun i : TileIndex [BLOCK_HEAD, BLOCK_DMODEL] =>
          i.1.val < head_num)
        _ hRawInj idx)
  · simpa [hHead] using
      (BlockState.scatter_readback_prop_masked_nd
        (region := Out)
        (shape := [BLOCK_HEAD, BLOCK_DMODEL])
        (offsetFn := fun i : TileIndex [BLOCK_HEAD, BLOCK_DMODEL] =>
          writeAddress (match s.readMemTyped TileDType.int Dest_loc (s.pids 0) with
            | some value => value
            | none => BlockState.defaultCarrier TileDType.int) stride_o_bs stride_o_h stride_o_d i.1.val i.2.1.val)
        (valueFn := fun i : TileIndex [BLOCK_HEAD, BLOCK_DMODEL] =>
          WithBot.unbotD 0
            (if i.1.val < head_num then
              some (s.readMem K
                (s.pids 0 * stride_k_bs + stride_k_h * i.1.val +
                  stride_k_d * i.2.1.val))
            else some 0.0))
        (P := fun i : TileIndex [BLOCK_HEAD, BLOCK_DMODEL] =>
          i.1.val < head_num)
        _ hRawInj idx)

/-- Executed-state form of `fwd_kernel_destindex_copy_kv_correct`. -/
theorem fwd_kernel_destindex_copy_kv_correct_of_exec
    (K : RegionName) (Dest_loc : Region .int) (Out : RegionName)
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
        = if active head_num idx then
            s.readMem K (sourceAddr s stride_k_bs stride_k_h stride_k_d idx)
          else
            s.readMem Out (outAddr s Dest_loc stride_o_bs stride_o_h stride_o_d idx) := by
  intro idx
  have h := fwd_kernel_destindex_copy_kv_correct K Dest_loc Out
    stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
    head_num BLOCK_DMODEL BLOCK_HEAD s hOutInj idx
  rw [hExec] at h
  simpa using h

/-- Frame half: every memory cell the masked scatter does not actively hit —
every cell of every region other than `Out`, and the inactive cells of the
dest-indexed window itself — is preserved by the run. -/
theorem fwd_kernel_destindex_copy_kv_frame
    (K : RegionName) (Dest_loc : Region .int) (Out : RegionName)
    (stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
      head_num BLOCK_DMODEL BLOCK_HEAD : Nat)
    (s s1 : BlockState)
    (hExec : exec ((fwd_kernel_destindex_copy_kv K Dest_loc Out
        stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
        head_num BLOCK_DMODEL BLOCK_HEAD).toAlgKernel) s = some s1)
    (r : RegionName) (o : Nat)
    (hmiss : ∀ idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL],
      idx.1.val < head_num →
      ¬(Out = r ∧ writeAddress (s.readMemValue .int Dest_loc (s.pids 0)) stride_o_bs stride_o_h stride_o_d idx.1.val idx.2.1.val = o)) :
    s1.mem r o = s.mem r o := by
  simp [exec, fwd_kernel_destindex_copy_kv, ComputeKernel.toAlgKernel, stepStmts,
        stepStmt, evalOp.eq_def, ComputeExpr.toAlgorithm?,
        Tile.bop, Tile.cop, Tile.ptrAdd, Tile.expandDim,
        NumericDType.add, NumericDType.mul, Tile.uop, writeAddress, ComparableDType.lt,
        BlockState.setReg, Option.bind, Option.map,
        TileShape.insertAxis, TileShape.dropInsertedIndex] at hExec
  subst hExec
  refine Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_) rfl
  intro k _ hmk hc
  exact hmiss k hmk
    (by simpa [BlockState.readMemValue, BlockState.readMemAs] using hc)

/-- Termination: the kernel executes to completion from any state
(straight-line body: one scalar `.int` load, pointer arithmetic, one masked
load, one masked scatter). -/
theorem fwd_kernel_destindex_copy_kv_exec_isSome
    (K : RegionName) (Dest_loc : Region .int) (Out : RegionName)
    (stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
      head_num BLOCK_DMODEL BLOCK_HEAD : Nat)
    (s : BlockState) :
    ∃ s1, exec ((fwd_kernel_destindex_copy_kv K Dest_loc Out
        stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
        head_num BLOCK_DMODEL BLOCK_HEAD).toAlgKernel) s = some s1 := by
  simp [exec, fwd_kernel_destindex_copy_kv, ComputeKernel.toAlgKernel, stepStmts,
        stepStmt, evalOp.eq_def, ComputeExpr.toAlgorithm?,
        Tile.bop, Tile.cop, Tile.ptrAdd, Tile.expandDim,
        NumericDType.add, NumericDType.mul, Tile.uop, writeAddress, ComparableDType.lt,
        BlockState.setReg, Option.bind, Option.map,
        TileShape.insertAxis, TileShape.dropInsertedIndex]

/-- Per-execution safety walk: one computational unfold walks the whole body —
the unmasked scalar `.int` load of `Dest_loc[pid]`, the two pointer-tile
computations, the masked `K` load, and the masked `Out` scatter — reducing the
three memory accesses to the cell/lane-wise bounds hypotheses. -/
theorem fwd_kernel_destindex_copy_kv_traceSafe
    (K : RegionName) (Dest_loc : Region .int) (Out : RegionName)
    (stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
      head_num BLOCK_DMODEL BLOCK_HEAD : Nat)
    (bounds : RegionBounds) (s : BlockState)
    (hd : s.pids 0 < bounds Dest_loc)
    (hk : ∀ idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL],
      idx.1.val < head_num →
      s.pids 0 * stride_k_bs + stride_k_h * idx.1.val + stride_k_d * idx.2.1.val
        < bounds K)
    (ho : ∀ idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL],
      idx.1.val < head_num →
      writeAddress (s.readMemValue .int Dest_loc (s.pids 0)) stride_o_bs stride_o_h stride_o_d idx.1.val idx.2.1.val < bounds Out) :
    Kernel.TraceSafe bounds
      ((fwd_kernel_destindex_copy_kv K Dest_loc Out
        stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
        head_num BLOCK_DMODEL BLOCK_HEAD).toAlgKernel) s := by
  unfold Kernel.TraceSafe
  simp [fwd_kernel_destindex_copy_kv, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
    MaskOpt.SafeAt, MemAccess.SafeAt, stepStmt, evalOp.eq_def,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    Op.PointerAddressesSafeOn, Op.MemorySafe,
    MaskOpt.Active, BlockState.setReg,
    Tile.bop, Tile.cop, Tile.ptrAdd, Tile.expandDim,
    Option.bind, Option.map, TileShape.insertAxis, TileShape.dropInsertedIndex,
    NumericDType.add, NumericDType.mul, Tile.uop, writeAddress, ComparableDType.lt]
  refine ⟨hd, fun a b h1 => hk (a, b, PUnit.unit) h1, fun a b h1 => ?_⟩
  exact (by
    simpa [BlockState.readMemValue, BlockState.readMemAs] using
      ho (a, b, PUnit.unit) h1)

/-- The kernel sits inside the flat-memory bridge's covered fragment (a scalar
`.nat` load, pointer arithmetic, a masked load and a masked scatter store). -/
theorem fwd_kernel_destindex_copy_kv_flattenOk
    (K : RegionName) (Dest_loc : Region .int) (Out : RegionName)
    (stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
      head_num BLOCK_DMODEL BLOCK_HEAD : Nat) :
    ((fwd_kernel_destindex_copy_kv K Dest_loc Out
        stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
        head_num BLOCK_DMODEL BLOCK_HEAD).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [fwd_kernel_destindex_copy_kv, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, StmtList.FlattenOk, Stmt.FlattenOk,
    Op.FlattenOk.eq_def]


#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.DestindexCopyKv2.RealReference
