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
destindex_copy_correctness                       ← TOP SPECIFICATION
  · destindexCopyIO ⊨ (xs, ys): the plain metadata masked copy triple
    (MetaMasked2DKernelIO₂ₓ₂ — one `.nat` slot `Dest_loc[pid₀] = m₁` driving
     both dest-indexed write windows; two different-width UNMASKED data tiles)
  ├─ fwd_kernel_destindex_copy_kv_flattenOk       bridge fragment membership
  ├─ fwd_kernel_destindex_copy_kv_traceSafe       per-execution safety walk
  └─ fwd_kernel_destindex_copy_kv_region_run      region-model metadata triple
       ├─ fwd_kernel_destindex_copy_kv_exec_isSome         termination
       ├─ fwd_kernel_destindex_copy_kv_nope_correct_of_exec  executed `O_nope` readback
       ├─ fwd_kernel_destindex_copy_kv_rope_correct_of_exec  executed `O_rope` readback
       └─ fwd_kernel_destindex_copy_kv_frame               two-store cell frame
```

`preStoreState` materializes the register/memory state just before the two
stores, so each store's readback is proved against the other store's writes.

The skin's two data channels are flat `Fin BLOCK_DMODEL_NOPE` /
`Fin BLOCK_DMODEL_ROPE` lanes; this kernel's tiles are the 1-row 2-D shapes
`[1, BLOCK_DMODEL_*]` (the `[None, :]` broadcast), so lane `j` is tile cell
`(0, j)` — a genuinely one-dimensional row, so no `Lane2D` flattening is
needed, only the trivial `Fin B → TileIndex [1, B]` embedding.

## Modeling boundary

Arithmetic/values are over `ℝ` (not bit-accurate IEEE float); the `float16`
dtype cast is erased (post-erasure all dtypes unify to `ℝ`). The destination
row index is data-dependent: `dest_index = Dest_loc[cur_index]`, read from the
nat region `Dest_loc`; the `⊨` headline pins it as the metadata slot binder
`m₁`. The headline carries three honest side conditions. `hNopeInj` /
`hRopeInj`: the tile part of each output address map,
`stride_o_*_d · dim`, is injective. These are **required for truth** — an
unmasked scatter whose lanes collide has a last-writer-wins outcome, so the
per-lane readback claim is simply false without it. Both are
`Dest_loc`-independent (the dest row only shifts every address by the constant
`m₁ · stride_o_*_bs`), and the natural layout `stride_o_*_d = 1` satisfies them.
`hRegion : O_nope ≠ O_rope` is required so the `O_nope` readback survives the
later `O_rope` store. The kernel's unused head-count / head-stride arguments are
kept at the signature boundary.
-/

namespace VeriTile.Bench.TritonBenchG.DestindexCopy

open VeriTile.Triton

set_option linter.unusedSimpArgs false

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
    simp [exec, fwd_kernel_destindex_copy_kv, stepStmts, stepStmt, evalOp.eq_def,
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
    simp [exec, fwd_kernel_destindex_copy_kv, stepStmts, stepStmt, evalOp.eq_def,
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

/-- The tile part of a 1-row output address map is injective as soon as its
column stride is injective over the lanes. The dest-row base only shifts every
address by the constant `base · strideBs`, so it drops out. -/
private theorem outAddr_inj_of_lane {B : Nat} (base strideBs strideD : Nat)
    (hLane : Function.Injective (fun j : Fin B => strideD * j.val)) :
    Function.Injective
      (fun idx : TileIndex [1, B] => base * strideBs + strideD * idx.2.1.val) := by
  intro a b hab
  obtain ⟨a1, a2, a3⟩ := a
  obtain ⟨b1, b2, b3⟩ := b
  have hcol : a2 = b2 := hLane (Nat.add_left_cancel hab)
  subst hcol
  rw [Subsingleton.elim a1 b1, Subsingleton.elim a3 b3]

/-- Termination: the kernel executes to completion from any state (straight-line
body: one scalar `.nat` load, pointer arithmetic, two unmasked loads, two
unmasked scatters). -/
private theorem fwd_kernel_destindex_copy_kv_exec_isSome
    (KV_nope KV_rope : RegionName) (Dest_loc : Region .nat) (O_nope O_rope : RegionName)
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
    Tile.bop, Tile.ptrAdd, Tile.expandDim, NumericDType.add, NumericDType.mul,
    BlockState.setReg, Option.bind, Option.map,
    TileShape.insertAxis, TileShape.dropInsertedIndex]

/-- Frame half: every memory cell the two unmasked scatters do not hit — every
cell of every region other than `O_nope` / `O_rope`, and the un-written cells of
the two dest-indexed windows — is preserved by the run. -/
private theorem fwd_kernel_destindex_copy_kv_frame
    (KV_nope KV_rope : RegionName) (Dest_loc : Region .nat) (O_nope O_rope : RegionName)
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
      ¬(O_nope = r ∧ s.readMemValue .nat Dest_loc (s.pids 0) * stride_o_nope_bs +
          stride_o_nope_d * idx.2.1.val = o))
    (hmissRope : ∀ idx : TileIndex [1, BLOCK_DMODEL_ROPE],
      ¬(O_rope = r ∧ s.readMemValue .nat Dest_loc (s.pids 0) * stride_o_rope_bs +
          stride_o_rope_d * idx.2.1.val = o)) :
    s1.mem r o = s.mem r o := by
  simp [exec, fwd_kernel_destindex_copy_kv, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, stepStmts, stepStmt, evalOp.eq_def,
    Tile.bop, Tile.ptrAdd, Tile.expandDim, NumericDType.add, NumericDType.mul,
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
the unmasked scalar `.nat` load of `Dest_loc[pid]`, the two pointer-tile
computations, the two unmasked loads and the two unmasked scatters — reducing
the five memory accesses to the cell/lane-wise bounds hypotheses. -/
theorem fwd_kernel_destindex_copy_kv_traceSafe
    (KV_nope KV_rope : RegionName) (Dest_loc : Region .nat) (O_nope O_rope : RegionName)
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
      s.readMemValue .nat Dest_loc (s.pids 0) * stride_o_nope_bs +
        stride_o_nope_d * j.val < bounds O_nope)
    (hoRope : ∀ j : Fin BLOCK_DMODEL_ROPE,
      s.readMemValue .nat Dest_loc (s.pids 0) * stride_o_rope_bs +
        stride_o_rope_d * j.val < bounds O_rope) :
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
    (KV_nope KV_rope : RegionName) (Dest_loc : Region .nat) (O_nope O_rope : RegionName)
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

/-- **The region-model metadata triple** — termination, both dest-indexed output
value legs, and frame off the two scatter windows, from any launch state pinning
the loaded dest row to `m₁` and the source lanes to `xs1`/`xs2`. This is the
`hrun` obligation of the `⊨` headline. `O_nope ≠ O_rope` lets the `O_nope`
readback see through the later `O_rope` store. -/
theorem fwd_kernel_destindex_copy_kv_region_run
    (KV_nope KV_rope : RegionName) (Dest_loc : Region .nat) (O_nope O_rope : RegionName)
    (stride_kv_nope_bs stride_kv_nope_h stride_kv_nope_d
      stride_kv_rope_bs stride_kv_rope_h stride_kv_rope_d
      stride_o_nope_bs stride_o_nope_h stride_o_nope_d
      stride_o_rope_bs stride_o_rope_h stride_o_rope_d
      kv_nope_head_num kv_rope_head_num
      BLOCK_DMODEL_NOPE BLOCK_DMODEL_ROPE : Nat)
    (hRegion : O_nope ≠ O_rope)
    (hNopeInj : Function.Injective
      (fun j : Fin BLOCK_DMODEL_NOPE => stride_o_nope_d * j.val))
    (hRopeInj : Function.Injective
      (fun j : Fin BLOCK_DMODEL_ROPE => stride_o_rope_d * j.val))
    (s₀ : BlockState) (m₁ : Nat)
    (xs1 : Fin BLOCK_DMODEL_NOPE → ℝ) (xs2 : Fin BLOCK_DMODEL_ROPE → ℝ)
    (hm₁ : s₀.readMemValue .nat Dest_loc (s₀.pids 0) = m₁)
    (hx1 : ∀ j : Fin BLOCK_DMODEL_NOPE,
      s₀.readMem KV_nope (s₀.pids 0 * stride_kv_nope_bs + stride_kv_nope_d * j.val) = xs1 j)
    (hx2 : ∀ j : Fin BLOCK_DMODEL_ROPE,
      s₀.readMem KV_rope (s₀.pids 0 * stride_kv_rope_bs + stride_kv_rope_d * j.val) = xs2 j) :
    ∃ s1, exec ((fwd_kernel_destindex_copy_kv KV_nope KV_rope Dest_loc O_nope O_rope
          stride_kv_nope_bs stride_kv_nope_h stride_kv_nope_d
          stride_kv_rope_bs stride_kv_rope_h stride_kv_rope_d
          stride_o_nope_bs stride_o_nope_h stride_o_nope_d
          stride_o_rope_bs stride_o_rope_h stride_o_rope_d
          kv_nope_head_num kv_rope_head_num
          BLOCK_DMODEL_NOPE BLOCK_DMODEL_ROPE).toAlgKernel) s₀ = some s1
      ∧ (∀ j : Fin BLOCK_DMODEL_NOPE,
          s1.readMem O_nope (m₁ * stride_o_nope_bs + stride_o_nope_d * j.val) = xs1 j)
      ∧ (∀ j : Fin BLOCK_DMODEL_ROPE,
          s1.readMem O_rope (m₁ * stride_o_rope_bs + stride_o_rope_d * j.val) = xs2 j)
      ∧ (∀ r o,
          (r ≠ O_nope ∨ ∀ j : Fin BLOCK_DMODEL_NOPE,
            o ≠ m₁ * stride_o_nope_bs + stride_o_nope_d * j.val) →
          (r ≠ O_rope ∨ ∀ j : Fin BLOCK_DMODEL_ROPE,
            o ≠ m₁ * stride_o_rope_bs + stride_o_rope_d * j.val) →
          s1.mem r o = s₀.mem r o) := by
  have hpid : s₀.pid = s₀.pids 0 := BlockState.pid_eq s₀
  have hbase : destBase s₀ Dest_loc = m₁ := by simpa [destBase, hpid] using hm₁
  have hOutNopeInj : Function.Injective
      (fun idx : TileIndex [1, BLOCK_DMODEL_NOPE] =>
        outNopeAddr s₀ Dest_loc stride_o_nope_bs stride_o_nope_d idx) := by
    have h := outAddr_inj_of_lane (destBase s₀ Dest_loc) stride_o_nope_bs stride_o_nope_d hNopeInj
    simpa [outNopeAddr, dimNope] using h
  have hOutRopeInj : Function.Injective
      (fun idx : TileIndex [1, BLOCK_DMODEL_ROPE] =>
        outRopeAddr s₀ Dest_loc stride_o_rope_bs stride_o_rope_d idx) := by
    have h := outAddr_inj_of_lane (destBase s₀ Dest_loc) stride_o_rope_bs stride_o_rope_d hRopeInj
    simpa [outRopeAddr, dimRope] using h
  obtain ⟨s1, hs1⟩ := fwd_kernel_destindex_copy_kv_exec_isSome
    KV_nope KV_rope Dest_loc O_nope O_rope
    stride_kv_nope_bs stride_kv_nope_h stride_kv_nope_d
    stride_kv_rope_bs stride_kv_rope_h stride_kv_rope_d
    stride_o_nope_bs stride_o_nope_h stride_o_nope_d
    stride_o_rope_bs stride_o_rope_h stride_o_rope_d
    kv_nope_head_num kv_rope_head_num
    BLOCK_DMODEL_NOPE BLOCK_DMODEL_ROPE s₀
  have hs1' : exec (fwd_kernel_destindex_copy_kv KV_nope KV_rope Dest_loc O_nope O_rope
      stride_kv_nope_bs stride_kv_nope_h stride_kv_nope_d
      stride_kv_rope_bs stride_kv_rope_h stride_kv_rope_d
      stride_o_nope_bs stride_o_nope_h stride_o_nope_d
      stride_o_rope_bs stride_o_rope_h stride_o_rope_d
      kv_nope_head_num kv_rope_head_num
      BLOCK_DMODEL_NOPE BLOCK_DMODEL_ROPE) s₀ = some s1 := by
    simpa [ComputeKernel.toAlgKernel, fwd_kernel_destindex_copy_kv,
      ComputeExpr.toAlgorithm?] using hs1
  refine ⟨s1, hs1, ?_, ?_, ?_⟩
  · intro j
    have h := fwd_kernel_destindex_copy_kv_nope_correct_of_exec
      KV_nope KV_rope Dest_loc O_nope O_rope
      stride_kv_nope_bs stride_kv_nope_h stride_kv_nope_d
      stride_kv_rope_bs stride_kv_rope_h stride_kv_rope_d
      stride_o_nope_bs stride_o_nope_h stride_o_nope_d
      stride_o_rope_bs stride_o_rope_h stride_o_rope_d
      kv_nope_head_num kv_rope_head_num
      BLOCK_DMODEL_NOPE BLOCK_DMODEL_ROPE s₀ s1 hRegion hOutNopeInj hs1'
      ((0 : Fin 1), j, PUnit.unit)
    simp only [outNopeAddr, sourceNopeAddr, dimNope, hbase, hpid] at h
    rw [h]; exact hx1 j
  · intro j
    have h := fwd_kernel_destindex_copy_kv_rope_correct_of_exec
      KV_nope KV_rope Dest_loc O_nope O_rope
      stride_kv_nope_bs stride_kv_nope_h stride_kv_nope_d
      stride_kv_rope_bs stride_kv_rope_h stride_kv_rope_d
      stride_o_nope_bs stride_o_nope_h stride_o_nope_d
      stride_o_rope_bs stride_o_rope_h stride_o_rope_d
      kv_nope_head_num kv_rope_head_num
      BLOCK_DMODEL_NOPE BLOCK_DMODEL_ROPE s₀ s1 hOutRopeInj hs1'
      ((0 : Fin 1), j, PUnit.unit)
    simp only [outRopeAddr, sourceRopeAddr, dimRope, hbase, hpid] at h
    rw [h]; exact hx2 j
  · intro r o hc1 hc2
    refine fwd_kernel_destindex_copy_kv_frame
      KV_nope KV_rope Dest_loc O_nope O_rope
      stride_kv_nope_bs stride_kv_nope_h stride_kv_nope_d
      stride_kv_rope_bs stride_kv_rope_h stride_kv_rope_d
      stride_o_nope_bs stride_o_nope_h stride_o_nope_d
      stride_o_rope_bs stride_o_rope_h stride_o_rope_d
      kv_nope_head_num kv_rope_head_num
      BLOCK_DMODEL_NOPE BLOCK_DMODEL_ROPE s₀ s1 hs1 r o (fun idx => ?_) (fun idx => ?_)
    · rintro ⟨hreg, hoff⟩
      rcases hc1 with hne | hno
      · exact hne hreg.symm
      · exact hno idx.2.1 (by rw [← hm₁]; exact hoff.symm)
    · rintro ⟨hreg, hoff⟩
      rcases hc2 with hne | hno
      · exact hne hreg.symm
      · exact hno idx.2.1 (by rw [← hm₁]; exact hoff.symm)

/-- `_fwd_kernel_destindex_copy_kv`'s metadata-genre **IO signature** — the whole
kernel-specific audit surface of the `⊨` headline (`MetaMasked2DKernelIO₂ₓ₂`,
the plain metadata genre with one `.nat` slot and two independent data tiles):

* `mbuf1 = Dest_loc`, `mwin1 = pid₀` — the `.nat` metadata slot: program
  `cur_index = pid₀` reads cell `pid₀` of `Dest_loc`, yielding the destination
  row `m₁`;
* `in1 = KV_nope → out1 = O_nope`, `in2 = KV_rope → out2 = O_rope` — the two
  data channels, `B1 = BLOCK_DMODEL_NOPE` / `B2 = BLOCK_DMODEL_ROPE` lanes;
* `read1`/`read2` — lane `j` reads
  `pid₀·stride_kv_*_bs + stride_kv_*_d·j` (the program's own source row);
* `write1`/`write2` — lane `j` writes `m₁·stride_o_*_bs + stride_o_*_d·j`: the
  address **eats the loaded slot value**, which is the whole point of the
  scatter;
* masks default to always-True (the copy loads and stores are unmasked).

The slot cell, windows, and masks are declared, not parsed from the kernel; the
headline **proves** the kernel's actual slot load, addressing and masking match
them. Buffer sizes are not signature content: the headline quantifies over every
allocation whose extents cover the declared cells. -/
def destindexCopyIO
    (KV_nope KV_rope : RegionName) (Dest_loc : Region .nat) (O_nope O_rope : RegionName)
    (stride_kv_nope_bs stride_kv_nope_h stride_kv_nope_d
      stride_kv_rope_bs stride_kv_rope_h stride_kv_rope_d
      stride_o_nope_bs stride_o_nope_h stride_o_nope_d
      stride_o_rope_bs stride_o_rope_h stride_o_rope_d
      kv_nope_head_num kv_rope_head_num
      BLOCK_DMODEL_NOPE BLOCK_DMODEL_ROPE : Nat) :
    MetaMasked2DKernelIO₂ₓ₂ where
  kernel := fwd_kernel_destindex_copy_kv KV_nope KV_rope Dest_loc O_nope O_rope
    stride_kv_nope_bs stride_kv_nope_h stride_kv_nope_d
    stride_kv_rope_bs stride_kv_rope_h stride_kv_rope_d
    stride_o_nope_bs stride_o_nope_h stride_o_nope_d
    stride_o_rope_bs stride_o_rope_h stride_o_rope_d
    kv_nope_head_num kv_rope_head_num
    BLOCK_DMODEL_NOPE BLOCK_DMODEL_ROPE
  mbuf1 := Dest_loc
  in1 := KV_nope
  in2 := KV_rope
  out1 := O_nope
  out2 := O_rope
  B1 := BLOCK_DMODEL_NOPE
  B2 := BLOCK_DMODEL_ROPE
  mwin1 := fun pid₀ _ => pid₀
  read1 := fun pid₀ _ _ j => pid₀ * stride_kv_nope_bs + stride_kv_nope_d * j.val
  read2 := fun pid₀ _ _ j => pid₀ * stride_kv_rope_bs + stride_kv_rope_d * j.val
  write1 := fun _ _ m₁ j => m₁ * stride_o_nope_bs + stride_o_nope_d * j.val
  write2 := fun _ _ m₁ j => m₁ * stride_o_rope_bs + stride_o_rope_d * j.val

open scoped VeriTile.Triton.MetaMasked2DKernelIO₂ₓ₂ in
/-- **The headline**: the dual dest-indexed KV scatter implements the pure copy
`(xs, ys)` on its metadata IO signature — for every disjoint flat placement of
the buffers, every program id whose declared cells/lanes are in bounds, and
every launch state pinning the loaded destination row to `m₁` and the two source
tiles to `xs`/`ys`, the translated pointer kernel terminates, every lane of the
**`m₁`-indexed** `O_nope` / `O_rope` rows holds `xs j` / `ys j`, and every other
memory cell is unchanged.

Side conditions (required for truth, not convenience): `hNopeInj` / `hRopeInj`,
the tile part of each output address map is injective — without them the
unmasked scatter is last-writer-wins and the per-lane readback is false; both
are `Dest_loc`-independent (the loaded row only shifts every address by the
constant `m₁ · stride_o_*_bs`). `hRegion : O_nope ≠ O_rope`, so the `O_nope`
readback survives the later `O_rope` store.

Proof: `MetaMasked2DKernelIO₂ₓ₂.Implements.intro` assembles the region-model
metadata triple with the flat-memory bridge side conditions. -/
specification destindex_copy_correctness
    (KV_nope KV_rope : RegionName) (Dest_loc : Region .nat) (O_nope O_rope : RegionName)
    (stride_kv_nope_bs stride_kv_nope_h stride_kv_nope_d
      stride_kv_rope_bs stride_kv_rope_h stride_kv_rope_d
      stride_o_nope_bs stride_o_nope_h stride_o_nope_d
      stride_o_rope_bs stride_o_rope_h stride_o_rope_d
      kv_nope_head_num kv_rope_head_num
      BLOCK_DMODEL_NOPE BLOCK_DMODEL_ROPE : Nat)
    (hRegion : O_nope ≠ O_rope)
    (hNopeInj : Function.Injective
      (fun j : Fin BLOCK_DMODEL_NOPE => stride_o_nope_d * j.val))
    (hRopeInj : Function.Injective
      (fun j : Fin BLOCK_DMODEL_ROPE => stride_o_rope_d * j.val)) :
    destindexCopyIO KV_nope KV_rope Dest_loc O_nope O_rope
        stride_kv_nope_bs stride_kv_nope_h stride_kv_nope_d
        stride_kv_rope_bs stride_kv_rope_h stride_kv_rope_d
        stride_o_nope_bs stride_o_nope_h stride_o_nope_d
        stride_o_rope_bs stride_o_rope_h stride_o_rope_d
        kv_nope_head_num kv_rope_head_num
        BLOCK_DMODEL_NOPE BLOCK_DMODEL_ROPE
      ⊨ fun _ _ _ xs ys => (xs, ys) := by
  refine MetaMasked2DKernelIO₂ₓ₂.Implements.intro _ ?_ ?_ ?_
  · exact fwd_kernel_destindex_copy_kv_flattenOk KV_nope KV_rope Dest_loc O_nope O_rope
      stride_kv_nope_bs stride_kv_nope_h stride_kv_nope_d
      stride_kv_rope_bs stride_kv_rope_h stride_kv_rope_d
      stride_o_nope_bs stride_o_nope_h stride_o_nope_d
      stride_o_rope_bs stride_o_rope_h stride_o_rope_d
      kv_nope_head_num kv_rope_head_num
      BLOCK_DMODEL_NOPE BLOCK_DMODEL_ROPE
  · intro bounds s m₁ hm₁ hb1 hbr1 hbr2 hbw1 hbw2
    simp only [destindexCopyIO] at hm₁ hb1 hbr1 hbr2 hbw1 hbw2
    refine fwd_kernel_destindex_copy_kv_traceSafe KV_nope KV_rope Dest_loc O_nope O_rope
      stride_kv_nope_bs stride_kv_nope_h stride_kv_nope_d
      stride_kv_rope_bs stride_kv_rope_h stride_kv_rope_d
      stride_o_nope_bs stride_o_nope_h stride_o_nope_d
      stride_o_rope_bs stride_o_rope_h stride_o_rope_d
      kv_nope_head_num kv_rope_head_num
      BLOCK_DMODEL_NOPE BLOCK_DMODEL_ROPE bounds s hb1
      (fun j => hbr1 j trivial) (fun j => hbr2 j trivial)
      (fun j => ?_) (fun j => ?_)
    · rw [hm₁]; exact hbw1 j trivial
    · rw [hm₁]; exact hbw2 j trivial
  · intro s₀ m₁ xs1 xs2 hm₁ hx1 hx2
    simp only [destindexCopyIO] at hm₁ hx1 hx2
    obtain ⟨s1, hexec, hv1, hv2, hframe⟩ :=
      fwd_kernel_destindex_copy_kv_region_run KV_nope KV_rope Dest_loc O_nope O_rope
        stride_kv_nope_bs stride_kv_nope_h stride_kv_nope_d
        stride_kv_rope_bs stride_kv_rope_h stride_kv_rope_d
        stride_o_nope_bs stride_o_nope_h stride_o_nope_d
        stride_o_rope_bs stride_o_rope_h stride_o_rope_d
        kv_nope_head_num kv_rope_head_num
        BLOCK_DMODEL_NOPE BLOCK_DMODEL_ROPE hRegion hNopeInj hRopeInj s₀ m₁ xs1 xs2 hm₁
        (fun j => hx1 j trivial) (fun j => hx2 j trivial)
    refine ⟨s1, hexec, fun j _ => hv1 j, fun j _ => hv2 j, ?_⟩
    intro r o hc1 hc2
    refine hframe r o ?_ ?_
    · rcases hc1 with h | h
      · exact Or.inl h
      · exact Or.inr (fun j => h j trivial)
    · rcases hc2 with h | h
      · exact Or.inl h
      · exact Or.inr (fun j => h j trivial)

end VeriTile.Bench.TritonBenchG.DestindexCopy
