import VeriTile.Triton

/-!
# `destindex_copy` — strict per-kernel correctness

`_fwd_kernel_destindex_copy_kv` is a dual KV-cache scatter (MLA nope/rope
split): program `cur_index` reads `Dest_loc[cur_index]` for a destination row,
loads the per-program `nope` and `rope` source rows from `KV_nope` / `KV_rope`,
and stores them into `O_nope` / `O_rope` at the dest-indexed row. There is no
load/store mask — every `[1, BLOCK_DMODEL_*]` lane is copied.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`_fwd_kernel_destindex_copy_kv[grid](...)`, the grid size
`(seq_len,)`, `BLOCK_DMODEL_*` rounding to powers of two, and how the runtime
composes the per-program scatters into the two output buffers) is the *trusted
boundary*, not a proof obligation here. Because `cur_index = pid` is universally
quantified, the per-program statement covers every program of the grid.

## Proof architecture

```
fwd_kernel_destindex_copy_kv_output_summary           ← TOP THEOREM
  ├─ (toAlgorithm? = Except.ok _)                      surface lowers to the algorithm layer
  ├─ fwd_kernel_destindex_copy_kv_nope_compute_correct ← ComputeCorrect for the O_nope scatter
  │    └─ fwd_kernel_destindex_copy_kv_nope_correct_of_exec
  └─ fwd_kernel_destindex_copy_kv_rope_compute_correct ← ComputeCorrect for the O_rope scatter
       └─ fwd_kernel_destindex_copy_kv_rope_correct_of_exec
```

`preStoreState` materializes the register/memory state just before the two
stores, so each store's readback is proved against the other store's writes.

## Modeling boundary

Arithmetic/values are over `ℝ` (not bit-accurate IEEE float); the `float16`
dtype cast is erased (post-erasure all dtypes unify to `ℝ`). The destination
row index is data-dependent: `dest_index = Dest_loc[cur_index]`, read from the
nat region `Dest_loc`. The `nope` readback uses `O_nope ≠ O_rope` (the two
output buffers are distinct, so the `rope` store does not clobber `O_nope`).
Both stores carry an injectivity side condition (`hOutNopeInj` / `hOutRopeInj`)
that each program's output offset map is injective over its tile. The kernel's
unused head-count / head-stride arguments are kept at the signature boundary.
-/

namespace VeriTile.Bench.TritonBenchG.DestindexCopy

open VeriTile.Triton

/-- Faithful transcription of `destindex_copy.py`'s
`_fwd_kernel_destindex_copy_kv`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_DMODEL_*: tl.constexpr` -> Lean `Nat` parameters.
- The unused Python head-count and head-stride arguments are retained at the
  theorem boundary, matching the original kernel signature. -/
def fwd_kernel_destindex_copy_kv
    (KV_nope KV_rope : RegionName) (Dest_loc : Region .nat) (O_nope O_rope : RegionName)
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
    (KV_nope KV_rope : RegionName) (Dest_loc : Region .nat) (O_nope O_rope : RegionName)
    (stride_kv_nope_bs stride_kv_nope_d
      stride_kv_rope_bs stride_kv_rope_d
      stride_o_nope_bs stride_o_nope_d
      stride_o_rope_bs stride_o_rope_d
      BLOCK_DMODEL_NOPE BLOCK_DMODEL_ROPE : Nat)
    (s : BlockState) : BlockState :=
  let s1 := s.setReg "cur_index" TileDType.nat [] (Tile.scalar (s.pids 0))
  let s2 := s1.setReg "offs_d_nope" TileDType.nat [BLOCK_DMODEL_NOPE] (Tile.vec fun i => i.val)
  let s3 := s2.setReg "offs_d_rope" TileDType.nat [BLOCK_DMODEL_ROPE] (Tile.vec fun i => i.val)
  let s4 := s3.setReg "dest_index" TileDType.nat []
    (Tile.scalar
      (match s.readMemTyped TileDType.nat Dest_loc (s.pids 0) with
      | some value => value
      | none => BlockState.defaultCarrier TileDType.nat))
  let s5 := s4.setReg "kv_nope_ptrs" TileDType.ptr [1, BLOCK_DMODEL_NOPE]
    { data := fun i =>
      (KV_nope, s.pids 0 * stride_kv_nope_bs + stride_kv_nope_d * i.2.1.val) }
  let s6 := s5.setReg "kv_rope_ptrs" TileDType.ptr [1, BLOCK_DMODEL_ROPE]
    { data := fun i =>
      (KV_rope, s.pids 0 * stride_kv_rope_bs + stride_kv_rope_d * i.2.1.val) }
  let s7 := s6.setReg "o_nope_ptrs" TileDType.ptr [1, BLOCK_DMODEL_NOPE]
    { data := fun i =>
      (O_nope,
        (match s.readMemTyped TileDType.nat Dest_loc (s.pids 0) with
        | some value => value
        | none => BlockState.defaultCarrier TileDType.nat) * stride_o_nope_bs +
        stride_o_nope_d * i.2.1.val) }
  let s8 := s7.setReg "o_rope_ptrs" TileDType.ptr [1, BLOCK_DMODEL_ROPE]
    { data := fun i =>
      (O_rope,
        (match s.readMemTyped TileDType.nat Dest_loc (s.pids 0) with
        | some value => value
        | none => BlockState.defaultCarrier TileDType.nat) * stride_o_rope_bs +
        stride_o_rope_d * i.2.1.val) }
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

def destBase (s : BlockState) (Dest_loc : RegionName) : Nat :=
  s.readMemValue .nat Dest_loc s.pid

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
  destBase s Dest_loc * stride_o_nope_bs + stride_o_nope_d * dimNope idx

def outRopeAddr
    (s : BlockState) (Dest_loc : RegionName)
    (stride_o_rope_bs stride_o_rope_d : Nat)
    (idx : TileIndex [1, BLOCK_DMODEL_ROPE]) : Nat :=
  destBase s Dest_loc * stride_o_rope_bs + stride_o_rope_d * dimRope idx

/-- Executed-state readback for the `O_rope` store. -/
theorem fwd_kernel_destindex_copy_kv_rope_correct_of_exec
    (KV_nope KV_rope : RegionName) (Dest_loc : Region .nat) (O_nope O_rope : RegionName)
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
        (match s.readMemTyped TileDType.nat Dest_loc (s.pids 0) with
          | some value => value
          | none => BlockState.defaultCarrier TileDType.nat) * stride_o_rope_bs +
          stride_o_rope_d * idx.2.1.val) := by
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
    simp [exec, fwd_kernel_destindex_copy_kv, stepStmts, stepStmt, evalOp, evalOp.eq_def,
          Tile.bop, Tile.ptrAdd, Tile.expandDim, NumericDType.add, NumericDType.mul,
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
            ((match s.readMemTyped TileDType.nat Dest_loc (s.pids 0) with
              | some value => value
              | none => BlockState.defaultCarrier TileDType.nat) * stride_o_nope_bs +
              stride_o_nope_d * k.2.1.val)
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
          (match s.readMemTyped TileDType.nat Dest_loc (s.pids 0) with
            | some value => value
            | none => BlockState.defaultCarrier TileDType.nat) * stride_o_rope_bs +
            stride_o_rope_d * i.2.1.val)
        (valueFn := fun i : TileIndex [1, BLOCK_DMODEL_ROPE] =>
          s.readMem KV_rope
            (s.pids 0 * stride_kv_rope_bs + stride_kv_rope_d * i.2.1.val))
        hRawInj idx)
  simpa [hExec] using hRead

/-- Executed-state readback for the `O_nope` store. -/
theorem fwd_kernel_destindex_copy_kv_nope_correct_of_exec
    (KV_nope KV_rope : RegionName) (Dest_loc : Region .nat) (O_nope O_rope : RegionName)
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
        (match s.readMemTyped TileDType.nat Dest_loc (s.pids 0) with
          | some value => value
          | none => BlockState.defaultCarrier TileDType.nat) * stride_o_nope_bs +
          stride_o_nope_d * idx.2.1.val) := by
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
    simp [exec, fwd_kernel_destindex_copy_kv, stepStmts, stepStmt, evalOp, evalOp.eq_def,
          Tile.bop, Tile.ptrAdd, Tile.expandDim, NumericDType.add, NumericDType.mul,
          BlockState.readMemValue, Option.bind, Option.map,
          TileShape.insertAxis, TileShape.dropInsertedIndex]
    simp [outNopeAddr, sourceNopeAddr, destBase, dimNope,
          BlockState.pid_eq, BlockState.readMemValue]
    change (List.foldl
          (fun acc i => acc.writeMem O_rope
            ((match s.readMemTyped TileDType.nat Dest_loc (s.pids 0) with
              | some value => value
              | none => BlockState.defaultCarrier TileDType.nat) * stride_o_rope_bs +
              stride_o_rope_d * i.2.1.val)
            (s.readMem KV_rope
              (s.pids 0 * stride_kv_rope_bs + stride_kv_rope_d * i.2.1.val)))
          (List.foldl
            (fun acc i => acc.writeMem O_nope
              ((match s.readMemTyped TileDType.nat Dest_loc (s.pids 0) with
                | some value => value
                | none => BlockState.defaultCarrier TileDType.nat) * stride_o_nope_bs +
                stride_o_nope_d * i.2.1.val)
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
          ((match s.readMemTyped TileDType.nat Dest_loc (s.pids 0) with
            | some value => value
            | none => BlockState.defaultCarrier TileDType.nat) * stride_o_nope_bs +
            stride_o_nope_d * idx.2.1.val)
        = s.readMem KV_nope
            (s.pids 0 * stride_kv_nope_bs + stride_kv_nope_d * idx.2.1.val)
    rw [BlockState.scatter_preserves_other_region
      (region := O_rope)
      (offsetFn := fun i : TileIndex [1, BLOCK_DMODEL_ROPE] =>
        (match s.readMemTyped TileDType.nat Dest_loc (s.pids 0) with
          | some value => value
          | none => BlockState.defaultCarrier TileDType.nat) * stride_o_rope_bs +
          stride_o_rope_d * i.2.1.val)
      (valueFn := fun i : TileIndex [1, BLOCK_DMODEL_ROPE] =>
        s.readMem KV_rope
          (s.pids 0 * stride_kv_rope_bs + stride_kv_rope_d * i.2.1.val))
      (R := O_nope) hRegion
      (off := (match s.readMemTyped TileDType.nat Dest_loc (s.pids 0) with
        | some value => value
        | none => BlockState.defaultCarrier TileDType.nat) * stride_o_nope_bs +
        stride_o_nope_d * idx.2.1.val)
      (l := TileShape.allIndices [1, BLOCK_DMODEL_ROPE])
      (s := List.foldl
        (fun acc i => acc.writeMem O_nope
          ((match s.readMemTyped TileDType.nat Dest_loc (s.pids 0) with
            | some value => value
            | none => BlockState.defaultCarrier TileDType.nat) * stride_o_nope_bs +
            stride_o_nope_d * i.2.1.val)
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
          (match s.readMemTyped TileDType.nat Dest_loc (s.pids 0) with
            | some value => value
            | none => BlockState.defaultCarrier TileDType.nat) * stride_o_nope_bs +
            stride_o_nope_d * i.2.1.val)
        (valueFn := fun i : TileIndex [1, BLOCK_DMODEL_NOPE] =>
          s.readMem KV_nope
            (s.pids 0 * stride_kv_nope_bs + stride_kv_nope_d * i.2.1.val))
        hRawInj idx)
  simpa [hExec] using hRead

/-- Compute-facing correctness for the `O_nope` output. -/
theorem fwd_kernel_destindex_copy_kv_nope_compute_correct
    (KV_nope KV_rope : RegionName) (Dest_loc : Region .nat) (O_nope O_rope : RegionName)
    (stride_kv_nope_bs stride_kv_nope_h stride_kv_nope_d
      stride_kv_rope_bs stride_kv_rope_h stride_kv_rope_d
      stride_o_nope_bs stride_o_nope_h stride_o_nope_d
      stride_o_rope_bs stride_o_rope_h stride_o_rope_d
      kv_nope_head_num kv_rope_head_num
      BLOCK_DMODEL_NOPE BLOCK_DMODEL_ROPE : Nat)
    (s : BlockState)
    (hRegion : O_nope ≠ O_rope)
    (hOutNopeInj : Function.Injective
      (fun idx : TileIndex [1, BLOCK_DMODEL_NOPE] =>
        outNopeAddr s Dest_loc stride_o_nope_bs stride_o_nope_d idx)) :
    ComputeCorrect.Realizes
      (kernel := fwd_kernel_destindex_copy_kv KV_nope KV_rope Dest_loc O_nope O_rope
        stride_kv_nope_bs stride_kv_nope_h stride_kv_nope_d
        stride_kv_rope_bs stride_kv_rope_h stride_kv_rope_d
        stride_o_nope_bs stride_o_nope_h stride_o_nope_d
        stride_o_rope_bs stride_o_rope_h stride_o_rope_d
        kv_nope_head_num kv_rope_head_num
        BLOCK_DMODEL_NOPE BLOCK_DMODEL_ROPE)
      (initialState := s)
      (write := fun idx : TileIndex [1, BLOCK_DMODEL_NOPE] =>
        some (O_nope, outNopeAddr s Dest_loc stride_o_nope_bs stride_o_nope_d idx))
      (expected := fun idx : TileIndex [1, BLOCK_DMODEL_NOPE] =>
        s.readMem KV_nope (sourceNopeAddr s stride_kv_nope_bs stride_kv_nope_d idx)) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [fwd_kernel_destindex_copy_kv]
  intro s0 s' hExec hs0
  subst s0
  intro idx
  exact fwd_kernel_destindex_copy_kv_nope_correct_of_exec
    KV_nope KV_rope Dest_loc O_nope O_rope
    stride_kv_nope_bs stride_kv_nope_h stride_kv_nope_d
    stride_kv_rope_bs stride_kv_rope_h stride_kv_rope_d
    stride_o_nope_bs stride_o_nope_h stride_o_nope_d
    stride_o_rope_bs stride_o_rope_h stride_o_rope_d
    kv_nope_head_num kv_rope_head_num
    BLOCK_DMODEL_NOPE BLOCK_DMODEL_ROPE s s' hRegion hOutNopeInj hExec idx

/-- Compute-facing correctness for the `O_rope` output. -/
theorem fwd_kernel_destindex_copy_kv_rope_compute_correct
    (KV_nope KV_rope : RegionName) (Dest_loc : Region .nat) (O_nope O_rope : RegionName)
    (stride_kv_nope_bs stride_kv_nope_h stride_kv_nope_d
      stride_kv_rope_bs stride_kv_rope_h stride_kv_rope_d
      stride_o_nope_bs stride_o_nope_h stride_o_nope_d
      stride_o_rope_bs stride_o_rope_h stride_o_rope_d
      kv_nope_head_num kv_rope_head_num
      BLOCK_DMODEL_NOPE BLOCK_DMODEL_ROPE : Nat)
    (s : BlockState)
    (hOutRopeInj : Function.Injective
      (fun idx : TileIndex [1, BLOCK_DMODEL_ROPE] =>
        outRopeAddr s Dest_loc stride_o_rope_bs stride_o_rope_d idx)) :
    ComputeCorrect.Realizes
      (kernel := fwd_kernel_destindex_copy_kv KV_nope KV_rope Dest_loc O_nope O_rope
        stride_kv_nope_bs stride_kv_nope_h stride_kv_nope_d
        stride_kv_rope_bs stride_kv_rope_h stride_kv_rope_d
        stride_o_nope_bs stride_o_nope_h stride_o_nope_d
        stride_o_rope_bs stride_o_rope_h stride_o_rope_d
        kv_nope_head_num kv_rope_head_num
        BLOCK_DMODEL_NOPE BLOCK_DMODEL_ROPE)
      (initialState := s)
      (write := fun idx : TileIndex [1, BLOCK_DMODEL_ROPE] =>
        some (O_rope, outRopeAddr s Dest_loc stride_o_rope_bs stride_o_rope_d idx))
      (expected := fun idx : TileIndex [1, BLOCK_DMODEL_ROPE] =>
        s.readMem KV_rope (sourceRopeAddr s stride_kv_rope_bs stride_kv_rope_d idx)) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [fwd_kernel_destindex_copy_kv]
  intro s0 s' hExec hs0
  subst s0
  intro idx
  exact fwd_kernel_destindex_copy_kv_rope_correct_of_exec
    KV_nope KV_rope Dest_loc O_nope O_rope
    stride_kv_nope_bs stride_kv_nope_h stride_kv_nope_d
    stride_kv_rope_bs stride_kv_rope_h stride_kv_rope_d
    stride_o_nope_bs stride_o_nope_h stride_o_nope_d
    stride_o_rope_bs stride_o_rope_h stride_o_rope_d
    kv_nope_head_num kv_rope_head_num
    BLOCK_DMODEL_NOPE BLOCK_DMODEL_ROPE s s' hOutRopeInj hExec idx

/-- Per-kernel output summary for `_fwd_kernel_destindex_copy_kv`: the DSL
surface lowers to the algorithm layer, and both dest-indexed scatters are
compute-correct — every `O_nope` cell holds the matching `KV_nope` cell and
every `O_rope` cell holds the matching `KV_rope` cell. -/
theorem fwd_kernel_destindex_copy_kv_output_summary
    (KV_nope KV_rope : RegionName) (Dest_loc : Region .nat) (O_nope O_rope : RegionName)
    (stride_kv_nope_bs stride_kv_nope_h stride_kv_nope_d
      stride_kv_rope_bs stride_kv_rope_h stride_kv_rope_d
      stride_o_nope_bs stride_o_nope_h stride_o_nope_d
      stride_o_rope_bs stride_o_rope_h stride_o_rope_d
      kv_nope_head_num kv_rope_head_num
      BLOCK_DMODEL_NOPE BLOCK_DMODEL_ROPE : Nat)
    (s : BlockState)
    (hRegion : O_nope ≠ O_rope)
    (hOutNopeInj : Function.Injective
      (fun idx : TileIndex [1, BLOCK_DMODEL_NOPE] =>
        outNopeAddr s Dest_loc stride_o_nope_bs stride_o_nope_d idx))
    (hOutRopeInj : Function.Injective
      (fun idx : TileIndex [1, BLOCK_DMODEL_ROPE] =>
        outRopeAddr s Dest_loc stride_o_rope_bs stride_o_rope_d idx)) :
    (∃ alg, (fwd_kernel_destindex_copy_kv KV_nope KV_rope Dest_loc O_nope O_rope
        stride_kv_nope_bs stride_kv_nope_h stride_kv_nope_d
        stride_kv_rope_bs stride_kv_rope_h stride_kv_rope_d
        stride_o_nope_bs stride_o_nope_h stride_o_nope_d
        stride_o_rope_bs stride_o_rope_h stride_o_rope_d
        kv_nope_head_num kv_rope_head_num
        BLOCK_DMODEL_NOPE BLOCK_DMODEL_ROPE).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := fwd_kernel_destindex_copy_kv KV_nope KV_rope Dest_loc O_nope O_rope
        stride_kv_nope_bs stride_kv_nope_h stride_kv_nope_d
        stride_kv_rope_bs stride_kv_rope_h stride_kv_rope_d
        stride_o_nope_bs stride_o_nope_h stride_o_nope_d
        stride_o_rope_bs stride_o_rope_h stride_o_rope_d
        kv_nope_head_num kv_rope_head_num
        BLOCK_DMODEL_NOPE BLOCK_DMODEL_ROPE)
      (initialState := s)
      (write := fun idx : TileIndex [1, BLOCK_DMODEL_NOPE] =>
        some (O_nope, outNopeAddr s Dest_loc stride_o_nope_bs stride_o_nope_d idx))
      (expected := fun idx : TileIndex [1, BLOCK_DMODEL_NOPE] =>
        s.readMem KV_nope (sourceNopeAddr s stride_kv_nope_bs stride_kv_nope_d idx)) ∧
    ComputeCorrect.Realizes
      (kernel := fwd_kernel_destindex_copy_kv KV_nope KV_rope Dest_loc O_nope O_rope
        stride_kv_nope_bs stride_kv_nope_h stride_kv_nope_d
        stride_kv_rope_bs stride_kv_rope_h stride_kv_rope_d
        stride_o_nope_bs stride_o_nope_h stride_o_nope_d
        stride_o_rope_bs stride_o_rope_h stride_o_rope_d
        kv_nope_head_num kv_rope_head_num
        BLOCK_DMODEL_NOPE BLOCK_DMODEL_ROPE)
      (initialState := s)
      (write := fun idx : TileIndex [1, BLOCK_DMODEL_ROPE] =>
        some (O_rope, outRopeAddr s Dest_loc stride_o_rope_bs stride_o_rope_d idx))
      (expected := fun idx : TileIndex [1, BLOCK_DMODEL_ROPE] =>
        s.readMem KV_rope (sourceRopeAddr s stride_kv_rope_bs stride_kv_rope_d idx)) := by
  refine ⟨⟨_, rfl⟩, ?_, ?_⟩
  · exact fwd_kernel_destindex_copy_kv_nope_compute_correct
      KV_nope KV_rope Dest_loc O_nope O_rope
      stride_kv_nope_bs stride_kv_nope_h stride_kv_nope_d
      stride_kv_rope_bs stride_kv_rope_h stride_kv_rope_d
      stride_o_nope_bs stride_o_nope_h stride_o_nope_d
      stride_o_rope_bs stride_o_rope_h stride_o_rope_d
      kv_nope_head_num kv_rope_head_num
      BLOCK_DMODEL_NOPE BLOCK_DMODEL_ROPE s hRegion hOutNopeInj
  · exact fwd_kernel_destindex_copy_kv_rope_compute_correct
      KV_nope KV_rope Dest_loc O_nope O_rope
      stride_kv_nope_bs stride_kv_nope_h stride_kv_nope_d
      stride_kv_rope_bs stride_kv_rope_h stride_kv_rope_d
      stride_o_nope_bs stride_o_nope_h stride_o_nope_d
      stride_o_rope_bs stride_o_rope_h stride_o_rope_d
      kv_nope_head_num kv_rope_head_num
      BLOCK_DMODEL_NOPE BLOCK_DMODEL_ROPE s hOutRopeInj

end VeriTile.Bench.TritonBenchG.DestindexCopy
