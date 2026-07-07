import VeriTile.Core
import VeriTile.Semantics
import VeriTile.Float
import VeriTile.Frontend.Triton.DSL

/-!
# `destindex_copy_kv2` — strict per-kernel correctness

`_fwd_kernel_destindex_copy_kv` is a KV-cache scatter: program `cur_index`
reads `Dest_loc[cur_index]` to obtain a destination row, loads the
`[head, dim]` tile of source `K` for that program, and stores it into `Out`
at the dest-indexed row. Unlike the kv1 variant the store mask is only
`offs_h < head_num` (the dim axis is always in-bounds because `BLOCK_DMODEL`
equals `head_dim`).

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`_fwd_kernel_destindex_copy_kv[grid](...)`, the grid size
`(seq_len,)`, `BLOCK_HEAD` rounding to a power of two, `BLOCK_DMODEL = head_dim`,
and how the runtime composes the per-program scatters into one `Out` buffer) is
the *trusted boundary*, not a proof obligation here. Because `cur_index = pid`
is universally quantified, the per-program statement covers every program of
the grid.

## Proof architecture

```
fwd_kernel_destindex_copy_kv_output_summary       ← TOP THEOREM
  ├─ (toAlgorithm? = Except.ok _)                 surface lowers to the algorithm layer
  └─ fwd_kernel_destindex_copy_kv_compute_correct ← ComputeCorrect over the masked scatter
       └─ fwd_kernel_destindex_copy_kv_correct_of_exec   executed-state readback per cell
            └─ fwd_kernel_destindex_copy_kv_correct      algorithm-layer readback per cell
```

The spec is a pure copy: every active (`head < head_num`) `[head, dim]` cell of
`Out` at the dest-indexed row holds the matching cell of `K`; inactive head
rows are preserved.

## Modeling boundary

Arithmetic/values are over `ℝ` (not bit-accurate IEEE float); the `float16`
dtype cast is erased (post-erasure all dtypes unify to `ℝ`). The destination
row index is data-dependent: `dest_index = Dest_loc[cur_index]`, read from the
nat region `Dest_loc`; the proofs carry an `hOutInj` side condition that the
output offset map is injective over the tile (no two cells of one program
alias), which the host guarantees via distinct dest rows / strides.
-/

namespace VeriTile.Bench.TritonBenchG.DestindexCopyKv2

open VeriTile

/-- Faithful transcription of `destindex_copy_kv2.py`'s
`_fwd_kernel_destindex_copy_kv`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_DMODEL: tl.constexpr` / `BLOCK_HEAD: tl.constexpr` -> Lean
  `Nat` parameters. -/
def fwd_kernel_destindex_copy_kv
    (K : RegionName) (Dest_loc : Region .nat) (Out : RegionName)
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

def destBase (s : BlockState) (Dest_loc : RegionName) : Nat :=
  s.readMemValue .nat Dest_loc s.pid

def outAddr
    (s : BlockState) (Dest_loc : RegionName)
    (stride_o_bs stride_o_h stride_o_d : Nat)
    (idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL]) : Nat :=
  destBase s Dest_loc * stride_o_bs + stride_o_h * headIndex idx + stride_o_d * dimIndex idx

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
    (K : RegionName) (Dest_loc : Region .nat) (Out : RegionName)
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
          (match s.readMemTyped TileDType.nat Dest_loc (s.pids 0) with
            | some value => value
            | none => BlockState.defaultCarrier TileDType.nat) * stride_o_bs +
            stride_o_h * i.1.val + stride_o_d * i.2.1.val)
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
    (K : RegionName) (Dest_loc : Region .nat) (Out : RegionName)
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

/-- Compute-facing correctness for `_fwd_kernel_destindex_copy_kv`. -/
theorem fwd_kernel_destindex_copy_kv_compute_correct
    (K : RegionName) (Dest_loc : Region .nat) (Out : RegionName)
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
        (fun idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL] => active head_num idx)
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

/-- Per-kernel output summary for `_fwd_kernel_destindex_copy_kv`: the DSL
surface lowers to the algorithm layer, and the dest-indexed masked scatter to
`Out` is compute-correct — every active (`head < head_num`) cell holds the
matching cell of `K`, inactive head rows are preserved. -/
theorem fwd_kernel_destindex_copy_kv_output_summary
    (K : RegionName) (Dest_loc : Region .nat) (Out : RegionName)
    (stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
      head_num BLOCK_DMODEL BLOCK_HEAD : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL] =>
        outAddr s Dest_loc stride_o_bs stride_o_h stride_o_d idx)) :
    (∃ alg, (fwd_kernel_destindex_copy_kv K Dest_loc Out
        stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
        head_num BLOCK_DMODEL BLOCK_HEAD).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := fwd_kernel_destindex_copy_kv K Dest_loc Out
        stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
        head_num BLOCK_DMODEL BLOCK_HEAD)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL] => active head_num idx)
        (fun idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL] =>
          (Out, outAddr s Dest_loc stride_o_bs stride_o_h stride_o_d idx)))
      (expected := fun idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL] =>
        s.readMem K (sourceAddr s stride_k_bs stride_k_h stride_k_d idx)) := by
  refine ⟨⟨_, rfl⟩, ?_⟩
  exact fwd_kernel_destindex_copy_kv_compute_correct K Dest_loc Out
    stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
    head_num BLOCK_DMODEL BLOCK_HEAD s hOutInj

end VeriTile.Bench.TritonBenchG.DestindexCopyKv2
