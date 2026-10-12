import VeriTile.Triton
import VeriTile.Meta.StatementAudit

/-! Complete signed-index real reference and direct execution lemmas.
This is an internal source model; source projection is checked separately. -/
namespace VeriTile.Bench.TritonBenchG.DestindexCopy.RealReference

open VeriTile.Triton

set_option linter.unusedSimpArgs false
set_option maxHeartbeats 8000000
set_option maxRecDepth 20000

def writeAddress (row : Int) (rowStride columnStride column : Nat) : Nat :=
  (row * (rowStride : Int) + (columnStride * column : Nat)).toNat

/-- Faithful transcription of `destindex_copy.py`'s
`_fwd_kernel_destindex_copy_kv`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_DMODEL_*: tl.constexpr` -> Lean `Nat` parameters.
- The unused Python head-count and head-stride arguments are retained at the
  theorem boundary, matching the original kernel signature. -/
def fwd_kernel_destindex_copy_kv
    (KV_nope KV_rope : RegionName) (Dest_loc : Region .int) (O_nope O_rope : RegionName)
    (stride_kv_nope_bs _stride_kv_nope_h stride_kv_nope_d
      stride_kv_rope_bs _stride_kv_rope_h stride_kv_rope_d
      stride_o_nope_bs _stride_o_nope_h stride_o_nope_d
      stride_o_rope_bs _stride_o_rope_h stride_o_rope_d
      _kv_nope_head_num _kv_rope_head_num
      BLOCK_DMODEL_NOPE BLOCK_DMODEL_ROPE : Nat) :
    ComputeKernel := triton {
  cur_index = tl.program_id(0)
  offs_d_nope = tl.arange(0, $(BLOCK_DMODEL_NOPE))
  offs_d_rope = tl.arange(0, $(BLOCK_DMODEL_ROPE))
  dest_index = tl.load(Dest_loc + cur_index)

  kv_nope_ptrs = KV_nope + cur_index * $(stride_kv_nope_bs) +
    $(stride_kv_nope_d) * offs_d_nope[None, :]
  kv_rope_ptrs = KV_rope + cur_index * $(stride_kv_rope_bs) +
    $(stride_kv_rope_d) * offs_d_rope[None, :]

  o_nope_ptrs = O_nope + dest_index * $(stride_o_nope_bs) +
    $(stride_o_nope_d) * offs_d_nope[None, :]
  o_rope_ptrs = O_rope + dest_index * $(stride_o_rope_bs) +
    $(stride_o_rope_d) * offs_d_rope[None, :]

  kv_nope = tl.load(kv_nope_ptrs)
  kv_rope = tl.load(kv_rope_ptrs)

  tl.store(o_nope_ptrs, kv_nope)
  tl.store(o_rope_ptrs, kv_rope)
}

/-- The memory-equivalent state immediately before the two stores execute. -/
def preStoreState
    (KV_nope KV_rope : RegionName) (Dest_loc : Region .int) (O_nope O_rope : RegionName)
    (stride_kv_nope_bs stride_kv_nope_d
      stride_kv_rope_bs stride_kv_rope_d
      stride_o_nope_bs stride_o_nope_d
      stride_o_rope_bs stride_o_rope_d
      BLOCK_DMODEL_NOPE BLOCK_DMODEL_ROPE : Nat)
    (s : BlockState) : BlockState :=
  let s1 := s.setReg "cur_index" TileDType.nat [] (Tile.scalar (s.pids 0))
  let s2 := s1.setReg "offs_d_nope" TileDType.nat [BLOCK_DMODEL_NOPE] (Tile.vec fun i => i.val)
  let s3 := s2.setReg "offs_d_rope" TileDType.nat [BLOCK_DMODEL_ROPE] (Tile.vec fun i => i.val)
  let s4 := s3.setReg "dest_index" TileDType.int []
    (Tile.scalar
      (match s.readMemTyped TileDType.int Dest_loc (s.pids 0) with
      | some value => value
      | none => BlockState.defaultCarrier TileDType.int))
  let s5 := s4.setReg "kv_nope_ptrs" TileDType.ptr [1, BLOCK_DMODEL_NOPE]
    { data := fun i =>
      (KV_nope, s.pids 0 * stride_kv_nope_bs + stride_kv_nope_d * i.2.1.val) }
  let s6 := s5.setReg "kv_rope_ptrs" TileDType.ptr [1, BLOCK_DMODEL_ROPE]
    { data := fun i =>
      (KV_rope, s.pids 0 * stride_kv_rope_bs + stride_kv_rope_d * i.2.1.val) }
  let s7 := s6.setReg "o_nope_ptrs" TileDType.ptr [1, BLOCK_DMODEL_NOPE]
    { data := fun i =>
      (O_nope,
        writeAddress (match s.readMemTyped TileDType.int Dest_loc (s.pids 0) with
        | some value => value
        | none => BlockState.defaultCarrier TileDType.int) stride_o_nope_bs stride_o_nope_d i.2.1.val) }
  let s8 := s7.setReg "o_rope_ptrs" TileDType.ptr [1, BLOCK_DMODEL_ROPE]
    { data := fun i =>
      (O_rope,
        writeAddress (match s.readMemTyped TileDType.int Dest_loc (s.pids 0) with
        | some value => value
        | none => BlockState.defaultCarrier TileDType.int) stride_o_rope_bs stride_o_rope_d i.2.1.val) }
  let s9 := s8.setReg "kv_nope" TileDType.real [1, BLOCK_DMODEL_NOPE]
    { data := fun i =>
      some (s.readMem KV_nope
        (s.pids 0 * stride_kv_nope_bs + stride_kv_nope_d * i.2.1.val)) }
  s9.setReg "kv_rope" TileDType.real [1, BLOCK_DMODEL_ROPE]
    { data := fun i =>
      some (s.readMem KV_rope
        (s.pids 0 * stride_kv_rope_bs + stride_kv_rope_d * i.2.1.val)) }

def dimNope (idx : TileIndex [1, BLOCK_DMODEL_NOPE]) : Nat :=
  idx.2.1.val

def dimRope (idx : TileIndex [1, BLOCK_DMODEL_ROPE]) : Nat :=
  idx.2.1.val

def destBase (s : BlockState) (Dest_loc : RegionName) : Int :=
  s.readMemValue .int Dest_loc s.pid

def sourceNopeAddr
    (s : BlockState) (stride_kv_nope_bs stride_kv_nope_d : Nat)
    (idx : TileIndex [1, BLOCK_DMODEL_NOPE]) : Nat :=
  s.pid * stride_kv_nope_bs + stride_kv_nope_d * dimNope idx

def sourceRopeAddr
    (s : BlockState) (stride_kv_rope_bs stride_kv_rope_d : Nat)
    (idx : TileIndex [1, BLOCK_DMODEL_ROPE]) : Nat :=
  s.pid * stride_kv_rope_bs + stride_kv_rope_d * dimRope idx

def outNopeAddr
    (s : BlockState) (Dest_loc : RegionName)
    (stride_o_nope_bs stride_o_nope_d : Nat)
    (idx : TileIndex [1, BLOCK_DMODEL_NOPE]) : Nat :=
  writeAddress (destBase s Dest_loc) stride_o_nope_bs stride_o_nope_d (dimNope idx)

def outRopeAddr
    (s : BlockState) (Dest_loc : RegionName)
    (stride_o_rope_bs stride_o_rope_d : Nat)
    (idx : TileIndex [1, BLOCK_DMODEL_ROPE]) : Nat :=
  writeAddress (destBase s Dest_loc) stride_o_rope_bs stride_o_rope_d (dimRope idx)

/-- Executed-state readback for the `O_rope` store. -/
theorem fwd_kernel_destindex_copy_kv_rope_correct_of_exec
    (KV_nope KV_rope : RegionName) (Dest_loc : Region .int) (O_nope O_rope : RegionName)
    (stride_kv_nope_bs stride_kv_nope_h stride_kv_nope_d
      stride_kv_rope_bs stride_kv_rope_h stride_kv_rope_d
      stride_o_nope_bs stride_o_nope_h stride_o_nope_d
      stride_o_rope_bs stride_o_rope_h stride_o_rope_d
      kv_nope_head_num kv_rope_head_num
      BLOCK_DMODEL_NOPE BLOCK_DMODEL_ROPE : Nat)
    (s s' : BlockState)
    (hOutRopeInj : Function.Injective
      (fun idx : TileIndex [1, BLOCK_DMODEL_ROPE] =>
        outRopeAddr s Dest_loc stride_o_rope_bs stride_o_rope_d idx))
    (hExec : exec (fwd_kernel_destindex_copy_kv KV_nope KV_rope Dest_loc O_nope O_rope
        stride_kv_nope_bs stride_kv_nope_h stride_kv_nope_d
        stride_kv_rope_bs stride_kv_rope_h stride_kv_rope_d
        stride_o_nope_bs stride_o_nope_h stride_o_nope_d
        stride_o_rope_bs stride_o_rope_h stride_o_rope_d
        kv_nope_head_num kv_rope_head_num
        BLOCK_DMODEL_NOPE BLOCK_DMODEL_ROPE) s = some s') :
    ∀ idx : TileIndex [1, BLOCK_DMODEL_ROPE],
      s'.readMem O_rope (outRopeAddr s Dest_loc stride_o_rope_bs stride_o_rope_d idx)
        = s.readMem KV_rope (sourceRopeAddr s stride_kv_rope_bs stride_kv_rope_d idx) := by
  intro idx
  have hRawInj : Function.Injective
      (fun idx : TileIndex [1, BLOCK_DMODEL_ROPE] =>
        writeAddress (match s.readMemTyped TileDType.int Dest_loc (s.pids 0) with
          | some value => value
          | none => BlockState.defaultCarrier TileDType.int) stride_o_rope_bs stride_o_rope_d idx.2.1.val) := by
    simpa [outRopeAddr, destBase, dimRope, BlockState.pid_eq,
      BlockState.readMemValue] using hOutRopeInj
  have hRead :
      (exec (fwd_kernel_destindex_copy_kv KV_nope KV_rope Dest_loc O_nope O_rope
          stride_kv_nope_bs stride_kv_nope_h stride_kv_nope_d
          stride_kv_rope_bs stride_kv_rope_h stride_kv_rope_d
          stride_o_nope_bs stride_o_nope_h stride_o_nope_d
          stride_o_rope_bs stride_o_rope_h stride_o_rope_d
          kv_nope_head_num kv_rope_head_num
          BLOCK_DMODEL_NOPE BLOCK_DMODEL_ROPE) s).map
          (fun s' => s'.readMem O_rope
            (outRopeAddr s Dest_loc stride_o_rope_bs stride_o_rope_d idx))
        = some (s.readMem KV_rope
            (sourceRopeAddr s stride_kv_rope_bs stride_kv_rope_d idx)) := by
    simp [exec, fwd_kernel_destindex_copy_kv, stepStmts, stepStmt, evalOp.eq_def,
          Tile.bop, Tile.ptrAdd, Tile.expandDim, NumericDType.add, NumericDType.mul, Tile.uop, writeAddress,
          BlockState.readMemValue, Option.bind, Option.map,
          TileShape.insertAxis, TileShape.dropInsertedIndex]
    simp [outRopeAddr, sourceRopeAddr, destBase, dimRope,
          BlockState.pid_eq, BlockState.readMemValue]
    simpa using
      (BlockState.scatter_readback_nd
        (region := O_rope)
        (shape := [1, BLOCK_DMODEL_ROPE])
        (s := List.foldl
          (fun acc k => acc.writeMem O_nope
            (writeAddress (match s.readMemTyped TileDType.int Dest_loc (s.pids 0) with
              | some value => value
              | none => BlockState.defaultCarrier TileDType.int) stride_o_nope_bs stride_o_nope_d k.2.1.val)
            (s.readMem KV_nope
              (s.pids 0 * stride_kv_nope_bs + stride_kv_nope_d * k.2.1.val)))
          (preStoreState KV_nope KV_rope Dest_loc O_nope O_rope
            stride_kv_nope_bs stride_kv_nope_d
            stride_kv_rope_bs stride_kv_rope_d
            stride_o_nope_bs stride_o_nope_d
            stride_o_rope_bs stride_o_rope_d
            BLOCK_DMODEL_NOPE BLOCK_DMODEL_ROPE s)
          (TileShape.allIndices [1, BLOCK_DMODEL_NOPE]))
        (offsetFn := fun i : TileIndex [1, BLOCK_DMODEL_ROPE] =>
          writeAddress (match s.readMemTyped TileDType.int Dest_loc (s.pids 0) with
            | some value => value
            | none => BlockState.defaultCarrier TileDType.int) stride_o_rope_bs stride_o_rope_d i.2.1.val)
        (valueFn := fun i : TileIndex [1, BLOCK_DMODEL_ROPE] =>
          s.readMem KV_rope
            (s.pids 0 * stride_kv_rope_bs + stride_kv_rope_d * i.2.1.val))
        hRawInj idx)
  simpa [hExec] using hRead

/-- Executed-state readback for the `O_nope` store. -/
theorem fwd_kernel_destindex_copy_kv_nope_correct_of_exec
    (KV_nope KV_rope : RegionName) (Dest_loc : Region .int) (O_nope O_rope : RegionName)
    (stride_kv_nope_bs stride_kv_nope_h stride_kv_nope_d
      stride_kv_rope_bs stride_kv_rope_h stride_kv_rope_d
      stride_o_nope_bs stride_o_nope_h stride_o_nope_d
      stride_o_rope_bs stride_o_rope_h stride_o_rope_d
      kv_nope_head_num kv_rope_head_num
      BLOCK_DMODEL_NOPE BLOCK_DMODEL_ROPE : Nat)
    (s s' : BlockState)
    (hRegion : O_nope ≠ O_rope)
    (hOutNopeInj : Function.Injective
      (fun idx : TileIndex [1, BLOCK_DMODEL_NOPE] =>
        outNopeAddr s Dest_loc stride_o_nope_bs stride_o_nope_d idx))
    (hExec : exec (fwd_kernel_destindex_copy_kv KV_nope KV_rope Dest_loc O_nope O_rope
        stride_kv_nope_bs stride_kv_nope_h stride_kv_nope_d
        stride_kv_rope_bs stride_kv_rope_h stride_kv_rope_d
        stride_o_nope_bs stride_o_nope_h stride_o_nope_d
        stride_o_rope_bs stride_o_rope_h stride_o_rope_d
        kv_nope_head_num kv_rope_head_num
        BLOCK_DMODEL_NOPE BLOCK_DMODEL_ROPE) s = some s') :
    ∀ idx : TileIndex [1, BLOCK_DMODEL_NOPE],
      s'.readMem O_nope (outNopeAddr s Dest_loc stride_o_nope_bs stride_o_nope_d idx)
        = s.readMem KV_nope (sourceNopeAddr s stride_kv_nope_bs stride_kv_nope_d idx) := by
  intro idx
  have hRawInj : Function.Injective
      (fun idx : TileIndex [1, BLOCK_DMODEL_NOPE] =>
        writeAddress (match s.readMemTyped TileDType.int Dest_loc (s.pids 0) with
          | some value => value
          | none => BlockState.defaultCarrier TileDType.int) stride_o_nope_bs stride_o_nope_d idx.2.1.val) := by
    simpa [outNopeAddr, destBase, dimNope, BlockState.pid_eq,
      BlockState.readMemValue] using hOutNopeInj
  have hRead :
      (exec (fwd_kernel_destindex_copy_kv KV_nope KV_rope Dest_loc O_nope O_rope
          stride_kv_nope_bs stride_kv_nope_h stride_kv_nope_d
          stride_kv_rope_bs stride_kv_rope_h stride_kv_rope_d
          stride_o_nope_bs stride_o_nope_h stride_o_nope_d
          stride_o_rope_bs stride_o_rope_h stride_o_rope_d
          kv_nope_head_num kv_rope_head_num
          BLOCK_DMODEL_NOPE BLOCK_DMODEL_ROPE) s).map
          (fun s' => s'.readMem O_nope
            (outNopeAddr s Dest_loc stride_o_nope_bs stride_o_nope_d idx))
        = some (s.readMem KV_nope
            (sourceNopeAddr s stride_kv_nope_bs stride_kv_nope_d idx)) := by
    simp [exec, fwd_kernel_destindex_copy_kv, stepStmts, stepStmt, evalOp.eq_def,
          Tile.bop, Tile.ptrAdd, Tile.expandDim, NumericDType.add, NumericDType.mul, Tile.uop, writeAddress,
          BlockState.readMemValue, Option.bind, Option.map,
          TileShape.insertAxis, TileShape.dropInsertedIndex]
    simp [outNopeAddr, sourceNopeAddr, destBase, dimNope,
          BlockState.pid_eq, BlockState.readMemValue]
    change (List.foldl
          (fun acc i => acc.writeMem O_rope
            (writeAddress (match s.readMemTyped TileDType.int Dest_loc (s.pids 0) with
              | some value => value
              | none => BlockState.defaultCarrier TileDType.int) stride_o_rope_bs stride_o_rope_d i.2.1.val)
            (s.readMem KV_rope
              (s.pids 0 * stride_kv_rope_bs + stride_kv_rope_d * i.2.1.val)))
          (List.foldl
            (fun acc i => acc.writeMem O_nope
              (writeAddress (match s.readMemTyped TileDType.int Dest_loc (s.pids 0) with
                | some value => value
                | none => BlockState.defaultCarrier TileDType.int) stride_o_nope_bs stride_o_nope_d i.2.1.val)
              (s.readMem KV_nope
                (s.pids 0 * stride_kv_nope_bs + stride_kv_nope_d * i.2.1.val)))
            (preStoreState KV_nope KV_rope Dest_loc O_nope O_rope
              stride_kv_nope_bs stride_kv_nope_d
              stride_kv_rope_bs stride_kv_rope_d
              stride_o_nope_bs stride_o_nope_d
              stride_o_rope_bs stride_o_rope_d
              BLOCK_DMODEL_NOPE BLOCK_DMODEL_ROPE s)
            (TileShape.allIndices [1, BLOCK_DMODEL_NOPE]))
          (TileShape.allIndices [1, BLOCK_DMODEL_ROPE])).readMem O_nope
          (writeAddress (match s.readMemTyped TileDType.int Dest_loc (s.pids 0) with
            | some value => value
            | none => BlockState.defaultCarrier TileDType.int) stride_o_nope_bs stride_o_nope_d idx.2.1.val)
        = s.readMem KV_nope
            (s.pids 0 * stride_kv_nope_bs + stride_kv_nope_d * idx.2.1.val)
    rw [BlockState.scatter_preserves_other_region
      (region := O_rope)
      (offsetFn := fun i : TileIndex [1, BLOCK_DMODEL_ROPE] =>
        writeAddress (match s.readMemTyped TileDType.int Dest_loc (s.pids 0) with
          | some value => value
          | none => BlockState.defaultCarrier TileDType.int) stride_o_rope_bs stride_o_rope_d i.2.1.val)
      (valueFn := fun i : TileIndex [1, BLOCK_DMODEL_ROPE] =>
        s.readMem KV_rope
          (s.pids 0 * stride_kv_rope_bs + stride_kv_rope_d * i.2.1.val))
      (R := O_nope) hRegion
      (off := writeAddress (match s.readMemTyped TileDType.int Dest_loc (s.pids 0) with
        | some value => value
        | none => BlockState.defaultCarrier TileDType.int) stride_o_nope_bs stride_o_nope_d idx.2.1.val)
      (l := TileShape.allIndices [1, BLOCK_DMODEL_ROPE])
      (s := List.foldl
        (fun acc i => acc.writeMem O_nope
          (writeAddress (match s.readMemTyped TileDType.int Dest_loc (s.pids 0) with
            | some value => value
            | none => BlockState.defaultCarrier TileDType.int) stride_o_nope_bs stride_o_nope_d i.2.1.val)
          (s.readMem KV_nope
            (s.pids 0 * stride_kv_nope_bs + stride_kv_nope_d * i.2.1.val)))
        (preStoreState KV_nope KV_rope Dest_loc O_nope O_rope
          stride_kv_nope_bs stride_kv_nope_d
          stride_kv_rope_bs stride_kv_rope_d
          stride_o_nope_bs stride_o_nope_d
          stride_o_rope_bs stride_o_rope_d
          BLOCK_DMODEL_NOPE BLOCK_DMODEL_ROPE s)
        (TileShape.allIndices [1, BLOCK_DMODEL_NOPE]))]
    simpa using
      (BlockState.scatter_readback_nd
        (region := O_nope)
        (shape := [1, BLOCK_DMODEL_NOPE])
        (s := preStoreState KV_nope KV_rope Dest_loc O_nope O_rope
          stride_kv_nope_bs stride_kv_nope_d
          stride_kv_rope_bs stride_kv_rope_d
          stride_o_nope_bs stride_o_nope_d
          stride_o_rope_bs stride_o_rope_d
          BLOCK_DMODEL_NOPE BLOCK_DMODEL_ROPE s)
        (offsetFn := fun i : TileIndex [1, BLOCK_DMODEL_NOPE] =>
          writeAddress (match s.readMemTyped TileDType.int Dest_loc (s.pids 0) with
            | some value => value
            | none => BlockState.defaultCarrier TileDType.int) stride_o_nope_bs stride_o_nope_d i.2.1.val)
        (valueFn := fun i : TileIndex [1, BLOCK_DMODEL_NOPE] =>
          s.readMem KV_nope
            (s.pids 0 * stride_kv_nope_bs + stride_kv_nope_d * i.2.1.val))
        hRawInj idx)
  simpa [hExec] using hRead

/-! ## The `⊨` metadata triple and headline

The kernel's data tiles are the 1-row shapes `[1, BLOCK_DMODEL_*]`; the skin's
lane `j : Fin BLOCK_DMODEL_*` is identified with tile cell `(0, j)`, so the
executed readbacks above are instantiated at `((0 : Fin 1), j, PUnit.unit)`. -/

/-- An unmasked `writeMem` scatter-store `foldl` leaves every memory cell it
does not hit unchanged (cell-level frame for one store). -/
private theorem foldl_writeMem_preserve_cell {α : Type} {region : RegionName}
    (offsetFn : α → Nat) (valueFn : α → ℝ)
    (r : RegionName) (o : Nat) (l : List α) (s : BlockState)
    (hnot : ∀ k ∈ l, ¬(region = r ∧ offsetFn k = o)) :
    (l.foldl (fun acc k => acc.writeMem region (offsetFn k) (valueFn k)) s).mem r o
      = s.mem r o := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
      rw [List.foldl_cons, ih _ (fun k hk => hnot k (List.mem_cons_of_mem hd hk)),
        BlockState.writeMem_mem]
      exact if_neg (fun hc => hnot hd List.mem_cons_self ⟨hc.1.symm, hc.2.symm⟩)

/-- Termination: the kernel executes to completion from any state (straight-line
body: one scalar `.int` load, pointer arithmetic, two unmasked loads, two
unmasked scatters). -/
theorem fwd_kernel_destindex_copy_kv_exec_isSome
    (KV_nope KV_rope : RegionName) (Dest_loc : Region .int) (O_nope O_rope : RegionName)
    (stride_kv_nope_bs stride_kv_nope_h stride_kv_nope_d
      stride_kv_rope_bs stride_kv_rope_h stride_kv_rope_d
      stride_o_nope_bs stride_o_nope_h stride_o_nope_d
      stride_o_rope_bs stride_o_rope_h stride_o_rope_d
      kv_nope_head_num kv_rope_head_num
      BLOCK_DMODEL_NOPE BLOCK_DMODEL_ROPE : Nat)
    (s : BlockState) :
    ∃ s1, exec ((fwd_kernel_destindex_copy_kv KV_nope KV_rope Dest_loc O_nope O_rope
        stride_kv_nope_bs stride_kv_nope_h stride_kv_nope_d
        stride_kv_rope_bs stride_kv_rope_h stride_kv_rope_d
        stride_o_nope_bs stride_o_nope_h stride_o_nope_d
        stride_o_rope_bs stride_o_rope_h stride_o_rope_d
        kv_nope_head_num kv_rope_head_num
        BLOCK_DMODEL_NOPE BLOCK_DMODEL_ROPE).toAlgKernel) s = some s1 := by
  simp [exec, fwd_kernel_destindex_copy_kv, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, stepStmts, stepStmt, evalOp.eq_def,
    Tile.bop, Tile.ptrAdd, Tile.expandDim, NumericDType.add, NumericDType.mul, Tile.uop, writeAddress,
    BlockState.setReg, Option.bind, Option.map,
    TileShape.insertAxis, TileShape.dropInsertedIndex]

/-- Frame half: every memory cell the two unmasked scatters do not hit — every
cell of every region other than `O_nope` / `O_rope`, and the un-written cells of
the two dest-indexed windows — is preserved by the run. -/
theorem fwd_kernel_destindex_copy_kv_frame
    (KV_nope KV_rope : RegionName) (Dest_loc : Region .int) (O_nope O_rope : RegionName)
    (stride_kv_nope_bs stride_kv_nope_h stride_kv_nope_d
      stride_kv_rope_bs stride_kv_rope_h stride_kv_rope_d
      stride_o_nope_bs stride_o_nope_h stride_o_nope_d
      stride_o_rope_bs stride_o_rope_h stride_o_rope_d
      kv_nope_head_num kv_rope_head_num
      BLOCK_DMODEL_NOPE BLOCK_DMODEL_ROPE : Nat)
    (s s1 : BlockState)
    (hExec : exec ((fwd_kernel_destindex_copy_kv KV_nope KV_rope Dest_loc O_nope O_rope
        stride_kv_nope_bs stride_kv_nope_h stride_kv_nope_d
        stride_kv_rope_bs stride_kv_rope_h stride_kv_rope_d
        stride_o_nope_bs stride_o_nope_h stride_o_nope_d
        stride_o_rope_bs stride_o_rope_h stride_o_rope_d
        kv_nope_head_num kv_rope_head_num
        BLOCK_DMODEL_NOPE BLOCK_DMODEL_ROPE).toAlgKernel) s = some s1)
    (r : RegionName) (o : Nat)
    (hmissNope : ∀ idx : TileIndex [1, BLOCK_DMODEL_NOPE],
      ¬(O_nope = r ∧ writeAddress (s.readMemValue .int Dest_loc (s.pids 0)) stride_o_nope_bs stride_o_nope_d idx.2.1.val = o))
    (hmissRope : ∀ idx : TileIndex [1, BLOCK_DMODEL_ROPE],
      ¬(O_rope = r ∧ writeAddress (s.readMemValue .int Dest_loc (s.pids 0)) stride_o_rope_bs stride_o_rope_d idx.2.1.val = o)) :
    s1.mem r o = s.mem r o := by
  simp [exec, fwd_kernel_destindex_copy_kv, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, stepStmts, stepStmt, evalOp.eq_def,
    Tile.bop, Tile.ptrAdd, Tile.expandDim, NumericDType.add, NumericDType.mul, Tile.uop, writeAddress,
    BlockState.setReg, Option.bind, Option.map,
    TileShape.insertAxis, TileShape.dropInsertedIndex] at hExec
  subst hExec
  refine Eq.trans (foldl_writeMem_preserve_cell _ _ r o _ _ ?_)
    (Eq.trans (foldl_writeMem_preserve_cell _ _ r o _ _ ?_) rfl)
  · rintro k _ ⟨hreg, hoff⟩
    exact hmissRope k
      ⟨hreg, by simpa [BlockState.readMemValue, BlockState.readMemAs] using hoff⟩
  · rintro k _ ⟨hreg, hoff⟩
    exact hmissNope k
      ⟨hreg, by simpa [BlockState.readMemValue, BlockState.readMemAs] using hoff⟩

/-- Per-execution safety walk: one computational unfold walks the whole body —
the unmasked scalar `.int` load of `Dest_loc[pid]`, the two pointer-tile
computations, the two unmasked loads and the two unmasked scatters — reducing
the five memory accesses to the cell/lane-wise bounds hypotheses. -/
theorem fwd_kernel_destindex_copy_kv_traceSafe
    (KV_nope KV_rope : RegionName) (Dest_loc : Region .int) (O_nope O_rope : RegionName)
    (stride_kv_nope_bs stride_kv_nope_h stride_kv_nope_d
      stride_kv_rope_bs stride_kv_rope_h stride_kv_rope_d
      stride_o_nope_bs stride_o_nope_h stride_o_nope_d
      stride_o_rope_bs stride_o_rope_h stride_o_rope_d
      kv_nope_head_num kv_rope_head_num
      BLOCK_DMODEL_NOPE BLOCK_DMODEL_ROPE : Nat)
    (bounds : RegionBounds) (s : BlockState)
    (hd : s.pids 0 < bounds Dest_loc)
    (hkNope : ∀ j : Fin BLOCK_DMODEL_NOPE,
      s.pids 0 * stride_kv_nope_bs + stride_kv_nope_d * j.val < bounds KV_nope)
    (hkRope : ∀ j : Fin BLOCK_DMODEL_ROPE,
      s.pids 0 * stride_kv_rope_bs + stride_kv_rope_d * j.val < bounds KV_rope)
    (hoNope : ∀ j : Fin BLOCK_DMODEL_NOPE,
      writeAddress (s.readMemValue .int Dest_loc (s.pids 0)) stride_o_nope_bs stride_o_nope_d j.val < bounds O_nope)
    (hoRope : ∀ j : Fin BLOCK_DMODEL_ROPE,
      writeAddress (s.readMemValue .int Dest_loc (s.pids 0)) stride_o_rope_bs stride_o_rope_d j.val < bounds O_rope) :
    Kernel.TraceSafe bounds
      ((fwd_kernel_destindex_copy_kv KV_nope KV_rope Dest_loc O_nope O_rope
        stride_kv_nope_bs stride_kv_nope_h stride_kv_nope_d
        stride_kv_rope_bs stride_kv_rope_h stride_kv_rope_d
        stride_o_nope_bs stride_o_nope_h stride_o_nope_d
        stride_o_rope_bs stride_o_rope_h stride_o_rope_d
        kv_nope_head_num kv_rope_head_num
        BLOCK_DMODEL_NOPE BLOCK_DMODEL_ROPE).toAlgKernel) s := by
  unfold Kernel.TraceSafe
  simp [fwd_kernel_destindex_copy_kv, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
    MaskOpt.SafeAt, MemAccess.SafeAt, stepStmt, evalOp.eq_def,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    Op.PointerAddressesSafeOn, Op.MemorySafe,
    MaskOpt.Active, BlockState.setReg,
    Tile.bop, Tile.ptrAdd, Tile.expandDim,
    Option.bind, Option.map, TileShape.insertAxis, TileShape.dropInsertedIndex,
    NumericDType.add, NumericDType.mul]
  refine ⟨hd, hkNope, hkRope, fun a => ?_, fun a => ?_⟩
  · simpa [BlockState.readMemValue, BlockState.readMemAs] using hoNope a
  · simpa [BlockState.readMemValue, BlockState.readMemAs] using hoRope a

/-- The kernel sits inside the flat-memory bridge's covered fragment (a scalar
`.nat` load, pointer arithmetic, two unmasked loads and two unmasked scatter
stores). -/
theorem fwd_kernel_destindex_copy_kv_flattenOk
    (KV_nope KV_rope : RegionName) (Dest_loc : Region .int) (O_nope O_rope : RegionName)
    (stride_kv_nope_bs stride_kv_nope_h stride_kv_nope_d
      stride_kv_rope_bs stride_kv_rope_h stride_kv_rope_d
      stride_o_nope_bs stride_o_nope_h stride_o_nope_d
      stride_o_rope_bs stride_o_rope_h stride_o_rope_d
      kv_nope_head_num kv_rope_head_num
      BLOCK_DMODEL_NOPE BLOCK_DMODEL_ROPE : Nat) :
    ((fwd_kernel_destindex_copy_kv KV_nope KV_rope Dest_loc O_nope O_rope
        stride_kv_nope_bs stride_kv_nope_h stride_kv_nope_d
        stride_kv_rope_bs stride_kv_rope_h stride_kv_rope_d
        stride_o_nope_bs stride_o_nope_h stride_o_nope_d
        stride_o_rope_bs stride_o_rope_h stride_o_rope_d
        kv_nope_head_num kv_rope_head_num
        BLOCK_DMODEL_NOPE BLOCK_DMODEL_ROPE).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [fwd_kernel_destindex_copy_kv, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, StmtList.FlattenOk, Stmt.FlattenOk,
    Op.FlattenOk.eq_def]


#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.DestindexCopy.RealReference
