import VeriTile.Triton

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
fwd_kernel_destindex_copy_kv_correctness      ← TOP SPECIFICATION
  · destindexCopyKvIO ⊨ the pure copy `xs`: the metadata-genre masked triple
    (MetaMasked2DKernelIO₁ — the `.nat` slot is `Dest_loc[pid₀]`, whose loaded
     value drives the write window)
  ├─ fwd_kernel_destindex_copy_kv_flattenOk       bridge fragment membership
  ├─ fwd_kernel_destindex_copy_kv_traceSafe       per-execution safety walk
  └─ fwd_kernel_destindex_copy_kv_region_run      region-model metadata triple
       ├─ fwd_kernel_destindex_copy_kv_exec_isSome     termination
       ├─ fwd_kernel_destindex_copy_kv_correct_of_exec executed-state readback
       │    └─ fwd_kernel_destindex_copy_kv_correct    algorithm-layer readback
       └─ fwd_kernel_destindex_copy_kv_frame           masked-scatter cell frame
```

The spec is a pure copy: every active (`head < head_num`) `[head, dim]` cell of
`Out` at the dest-indexed row holds the matching cell of `K`.

The skin's data channel is a flat `Fin (BLOCK_HEAD * BLOCK_DMODEL)` lane
index; `laneIndex` / `laneOf` are the row-major flattening between it and the
kernel's genuine 2-D `[BLOCK_HEAD, BLOCK_DMODEL]` tile index, so lane `j` is
tile cell `(j / BLOCK_DMODEL, j % BLOCK_DMODEL)`.

## Modeling boundary

Arithmetic/values are over `ℝ` (not bit-accurate IEEE float); the `float16`
dtype cast is erased (post-erasure all dtypes unify to `ℝ`). The destination
row index is data-dependent: `dest_index = Dest_loc[cur_index]`, read from the
nat region `Dest_loc`; the `⊨` headline pins it as the metadata slot binder
`m₁`. The headline carries one honest side condition, `hInj`: the tile part of
the output address map, `stride_o_h · head + stride_o_d · dim`, is injective.
This is **required for truth** — a masked scatter whose lanes collide has a
last-writer-wins outcome, so the per-lane readback claim is false without it.
It is `Dest_loc`-independent (the dest row only shifts every address by the
constant `dest_index · stride_o_bs`), and the natural layout
`stride_o_h = head_dim`, `stride_o_d = 1` satisfies it.
-/

namespace VeriTile.Bench.TritonBenchG.DestindexCopyKv2

open VeriTile.Triton

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

/-! ## Flattening the 2-D tile into the skin's lane index

The `⊨` family's data channels are flat `Fin B` lanes; this kernel's tile is
genuinely two-dimensional, so lane `j` is identified with tile cell
`(j / BLOCK_DMODEL, j % BLOCK_DMODEL)` (row-major). `laneOf` is the inverse. -/

/-- Tile cell of lane `j` — row-major decode of the flat lane index. -/
def laneIndex (BLOCK_HEAD BLOCK_DMODEL : Nat)
    (j : Fin (BLOCK_HEAD * BLOCK_DMODEL)) : TileIndex [BLOCK_HEAD, BLOCK_DMODEL] :=
  have hD : 0 < BLOCK_DMODEL :=
    Nat.pos_of_ne_zero fun h => absurd j.isLt (by simp [h])
  (⟨j.val / BLOCK_DMODEL,
      Nat.div_lt_of_lt_mul (by simp only [Nat.mul_comm BLOCK_DMODEL BLOCK_HEAD]; exact j.isLt)⟩,
    ⟨j.val % BLOCK_DMODEL, Nat.mod_lt _ hD⟩, PUnit.unit)

/-- Flat lane index of a tile cell — row-major encode, inverse to
`laneIndex`. -/
def laneOf (BLOCK_HEAD BLOCK_DMODEL : Nat)
    (idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL]) :
    Fin (BLOCK_HEAD * BLOCK_DMODEL) :=
  ⟨idx.1.val * BLOCK_DMODEL + idx.2.1.val, by
    have h1 : idx.1.val + 1 ≤ BLOCK_HEAD := idx.1.isLt
    have h2 : idx.2.1.val < BLOCK_DMODEL := idx.2.1.isLt
    calc idx.1.val * BLOCK_DMODEL + idx.2.1.val
        < idx.1.val * BLOCK_DMODEL + BLOCK_DMODEL := Nat.add_lt_add_left h2 _
      _ = (idx.1.val + 1) * BLOCK_DMODEL := by ring
      _ ≤ BLOCK_HEAD * BLOCK_DMODEL := Nat.mul_le_mul_right BLOCK_DMODEL h1⟩

@[simp] theorem laneOf_div (BLOCK_HEAD BLOCK_DMODEL : Nat)
    (idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL]) :
    (laneOf BLOCK_HEAD BLOCK_DMODEL idx).val / BLOCK_DMODEL = idx.1.val := by
  show (idx.1.val * BLOCK_DMODEL + idx.2.1.val) / BLOCK_DMODEL = idx.1.val
  have h2 : idx.2.1.val < BLOCK_DMODEL := idx.2.1.isLt
  have hD : 0 < BLOCK_DMODEL := Nat.lt_of_le_of_lt (Nat.zero_le _) h2
  rw [Nat.mul_comm, Nat.mul_add_div hD, Nat.div_eq_of_lt h2, Nat.add_zero]

@[simp] theorem laneOf_mod (BLOCK_HEAD BLOCK_DMODEL : Nat)
    (idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL]) :
    (laneOf BLOCK_HEAD BLOCK_DMODEL idx).val % BLOCK_DMODEL = idx.2.1.val := by
  show (idx.1.val * BLOCK_DMODEL + idx.2.1.val) % BLOCK_DMODEL = idx.2.1.val
  have h2 : idx.2.1.val < BLOCK_DMODEL := idx.2.1.isLt
  rw [Nat.mul_comm, Nat.mul_add_mod, Nat.mod_eq_of_lt h2]

/-- A masked scatter-store `foldl` leaves every memory cell it does not
actively hit unchanged (cell-level frame for the masked store). -/
private theorem foldl_store_preserve_cell {α : Type} {region : RegionName}
    (offsetFn : α → Nat) (valueFn : α → ℝ) (P : α → Prop) [DecidablePred P]
    (r : RegionName) (o : Nat) (l : List α) (s : BlockState)
    (hnot : ∀ k ∈ l, P k → ¬(region = r ∧ offsetFn k = o)) :
    (l.foldl (fun acc k =>
        if P k then acc.writeMem region (offsetFn k) (valueFn k) else acc)
      s).mem r o = s.mem r o := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
      rw [List.foldl_cons]
      by_cases hP : P hd
      · rw [if_pos hP,
          ih _ (fun k hk => hnot k (List.mem_cons_of_mem hd hk)),
          BlockState.writeMem_mem]
        exact if_neg (fun hc =>
          hnot hd List.mem_cons_self hP ⟨hc.1.symm, hc.2.symm⟩)
      · rw [if_neg hP]
        exact ih _ (fun k hk => hnot k (List.mem_cons_of_mem hd hk))

/-- Frame half: every memory cell the masked scatter does not actively hit —
every cell of every region other than `Out`, and the inactive cells of the
dest-indexed window itself — is preserved by the run. -/
private theorem fwd_kernel_destindex_copy_kv_frame
    (K : RegionName) (Dest_loc : Region .nat) (Out : RegionName)
    (stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
      head_num BLOCK_DMODEL BLOCK_HEAD : Nat)
    (s s1 : BlockState)
    (hExec : exec ((fwd_kernel_destindex_copy_kv K Dest_loc Out
        stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
        head_num BLOCK_DMODEL BLOCK_HEAD).toAlgKernel) s = some s1)
    (r : RegionName) (o : Nat)
    (hmiss : ∀ idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL],
      idx.1.val < head_num →
      ¬(Out = r ∧ s.readMemValue .nat Dest_loc (s.pids 0) * stride_o_bs +
          stride_o_h * idx.1.val + stride_o_d * idx.2.1.val = o)) :
    s1.mem r o = s.mem r o := by
  simp [exec, fwd_kernel_destindex_copy_kv, ComputeKernel.toAlgKernel, stepStmts,
        stepStmt, evalOp.eq_def, ComputeExpr.toAlgorithm?,
        Tile.bop, Tile.cop, Tile.ptrAdd, Tile.expandDim,
        NumericDType.add, NumericDType.mul, ComparableDType.lt,
        BlockState.setReg, Option.bind, Option.map,
        TileShape.insertAxis, TileShape.dropInsertedIndex] at hExec
  subst hExec
  refine Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_) rfl
  intro k _ hmk hc
  exact hmiss k hmk
    (by simpa [BlockState.readMemValue, BlockState.readMemAs] using hc)

/-- Termination: the kernel executes to completion from any state
(straight-line body: one scalar `.nat` load, pointer arithmetic, one masked
load, one masked scatter). -/
private theorem fwd_kernel_destindex_copy_kv_exec_isSome
    (K : RegionName) (Dest_loc : Region .nat) (Out : RegionName)
    (stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
      head_num BLOCK_DMODEL BLOCK_HEAD : Nat)
    (s : BlockState) :
    ∃ s1, exec ((fwd_kernel_destindex_copy_kv K Dest_loc Out
        stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
        head_num BLOCK_DMODEL BLOCK_HEAD).toAlgKernel) s = some s1 := by
  simp [exec, fwd_kernel_destindex_copy_kv, ComputeKernel.toAlgKernel, stepStmts,
        stepStmt, evalOp.eq_def, ComputeExpr.toAlgorithm?,
        Tile.bop, Tile.cop, Tile.ptrAdd, Tile.expandDim,
        NumericDType.add, NumericDType.mul, ComparableDType.lt,
        BlockState.setReg, Option.bind, Option.map,
        TileShape.insertAxis, TileShape.dropInsertedIndex]

/-- **The region-model metadata triple** — termination, active-lane output
values, and frame off the active scatter cells, from any launch state pinning
the loaded dest row to `m₁` and the active `K` lanes to `xs`. This is the
`hrun` obligation of the `⊨` headline. -/
theorem fwd_kernel_destindex_copy_kv_region_run
    (K : RegionName) (Dest_loc : Region .nat) (Out : RegionName)
    (stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
      head_num BLOCK_DMODEL BLOCK_HEAD : Nat)
    (hInj : Function.Injective
      (fun idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL] =>
        stride_o_h * headIndex idx + stride_o_d * dimIndex idx))
    (s₀ : BlockState) (m₁ : Nat) (xs : Fin (BLOCK_HEAD * BLOCK_DMODEL) → ℝ)
    (hm : s₀.readMemValue .nat Dest_loc (s₀.pids 0) = m₁)
    (hx : ∀ j : Fin (BLOCK_HEAD * BLOCK_DMODEL),
      j.val / BLOCK_DMODEL < head_num →
      s₀.readMem K (s₀.pids 0 * stride_k_bs +
        stride_k_h * (j.val / BLOCK_DMODEL) +
        stride_k_d * (j.val % BLOCK_DMODEL)) = xs j) :
    ∃ s1, exec ((fwd_kernel_destindex_copy_kv K Dest_loc Out
          stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
          head_num BLOCK_DMODEL BLOCK_HEAD).toAlgKernel) s₀ = some s1
      ∧ (∀ j : Fin (BLOCK_HEAD * BLOCK_DMODEL),
          j.val / BLOCK_DMODEL < head_num →
          s1.readMem Out (m₁ * stride_o_bs +
            stride_o_h * (j.val / BLOCK_DMODEL) +
            stride_o_d * (j.val % BLOCK_DMODEL)) = xs j)
      ∧ (∀ r o,
          (r ≠ Out ∨ ∀ j : Fin (BLOCK_HEAD * BLOCK_DMODEL),
            j.val / BLOCK_DMODEL < head_num →
            o ≠ m₁ * stride_o_bs + stride_o_h * (j.val / BLOCK_DMODEL) +
              stride_o_d * (j.val % BLOCK_DMODEL)) →
          s1.mem r o = s₀.mem r o) := by
  have hpid : s₀.pid = s₀.pids 0 := BlockState.pid_eq s₀
  have hbase : destBase s₀ Dest_loc = m₁ := by
    simpa [destBase, hpid] using hm
  have hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL] =>
        outAddr s₀ Dest_loc stride_o_bs stride_o_h stride_o_d idx) := by
    intro a b hab
    refine hInj ?_
    simp only [outAddr] at hab
    rw [Nat.add_assoc, Nat.add_assoc] at hab
    exact Nat.add_left_cancel hab
  obtain ⟨s1, hs1⟩ := fwd_kernel_destindex_copy_kv_exec_isSome K Dest_loc Out
    stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
    head_num BLOCK_DMODEL BLOCK_HEAD s₀
  have hs1' : exec (fwd_kernel_destindex_copy_kv K Dest_loc Out
      stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
      head_num BLOCK_DMODEL BLOCK_HEAD) s₀ = some s1 := by
    simpa [ComputeKernel.toAlgKernel, fwd_kernel_destindex_copy_kv,
      ComputeExpr.toAlgorithm?] using hs1
  refine ⟨s1, hs1, fun j hj => ?_, fun r o hcond => ?_⟩
  · have h := fwd_kernel_destindex_copy_kv_correct_of_exec K Dest_loc Out
      stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
      head_num BLOCK_DMODEL BLOCK_HEAD s₀ hOutInj s1 hs1'
      (laneIndex BLOCK_HEAD BLOCK_DMODEL j)
    have hact : active head_num (laneIndex BLOCK_HEAD BLOCK_DMODEL j) := hj
    rw [if_pos hact] at h
    simp only [outAddr, sourceAddr, headIndex, dimIndex, laneIndex, hbase, hpid] at h ⊢
    rw [h, hx j hj]
  · refine fwd_kernel_destindex_copy_kv_frame K Dest_loc Out
      stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
      head_num BLOCK_DMODEL BLOCK_HEAD s₀ s1 hs1 r o
      (fun idx h1 ⟨hr, ho⟩ => ?_)
    rcases hcond with hne | hno
    · exact hne hr.symm
    · refine hno (laneOf BLOCK_HEAD BLOCK_DMODEL idx) ?_ ?_
      · simpa using h1
      · simp only [laneOf_div, laneOf_mod]
        rw [← ho, hm]

/-- Per-execution safety walk: one computational unfold walks the whole body —
the unmasked scalar `.nat` load of `Dest_loc[pid]`, the two pointer-tile
computations, the masked `K` load, and the masked `Out` scatter — reducing the
three memory accesses to the cell/lane-wise bounds hypotheses. -/
theorem fwd_kernel_destindex_copy_kv_traceSafe
    (K : RegionName) (Dest_loc : Region .nat) (Out : RegionName)
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
      s.readMemValue .nat Dest_loc (s.pids 0) * stride_o_bs +
        stride_o_h * idx.1.val + stride_o_d * idx.2.1.val < bounds Out) :
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
    NumericDType.add, NumericDType.mul, ComparableDType.lt]
  refine ⟨hd, fun a b h1 => hk (a, b, PUnit.unit) h1, fun a b h1 => ?_⟩
  exact (by
    simpa [BlockState.readMemValue, BlockState.readMemAs] using
      ho (a, b, PUnit.unit) h1)

/-- The kernel sits inside the flat-memory bridge's covered fragment (a scalar
`.nat` load, pointer arithmetic, a masked load and a masked scatter store). -/
theorem fwd_kernel_destindex_copy_kv_flattenOk
    (K : RegionName) (Dest_loc : Region .nat) (Out : RegionName)
    (stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
      head_num BLOCK_DMODEL BLOCK_HEAD : Nat) :
    ((fwd_kernel_destindex_copy_kv K Dest_loc Out
        stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
        head_num BLOCK_DMODEL BLOCK_HEAD).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [fwd_kernel_destindex_copy_kv, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, StmtList.FlattenOk, Stmt.FlattenOk,
    Op.FlattenOk.eq_def]

/-- `_fwd_kernel_destindex_copy_kv`'s metadata-genre **IO signature** — the
whole kernel-specific audit surface of the `⊨` headline
(`MetaMasked2DKernelIO₁`, the genre where a value *loaded from memory* drives
the windows):

* `mbuf1 = mbuf2 = Dest_loc`, `mwin1 = mwin2 = pid₀` — the `.nat` metadata
  slot: program `cur_index = pid₀` reads cell `pid₀` of `Dest_loc`, yielding
  the destination row `m₁`. This kernel has **one** slot; the genre carries
  two, so the second is wired to the same cell (hence `m₂ = m₁`) and is
  ignored by every window;
* `inp = K`, `out = Out`;
* `B = BLOCK_HEAD * BLOCK_DMODEL` — the flattened `[head, dim]` tile: lane `j`
  is cell `(j / BLOCK_DMODEL, j % BLOCK_DMODEL)`;
* `read` — lane `j` reads `pid₀·stride_k_bs + stride_k_h·head + stride_k_d·dim`
  (the program's own source row);
* `write` — lane `j` writes `m₁·stride_o_bs + stride_o_h·head +
  stride_o_d·dim`: the address **eats the loaded slot value**, which is the
  whole point of the scatter;
* `mask` (and the defaulted `writeMask`) — the active lanes
  `head < head_num` (this variant masks only the head axis).

The slot cell, windows, and masks are declared, not parsed from the kernel;
the headline **proves** the kernel's actual slot load, addressing and masking
match them. Buffer sizes are not signature content: the headline quantifies
over every allocation whose extents cover the declared cells. -/
def destindexCopyKvIO
    (K : RegionName) (Dest_loc : Region .nat) (Out : RegionName)
    (stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
      head_num BLOCK_DMODEL BLOCK_HEAD : Nat) :
    MetaMasked2DKernelIO₁ where
  kernel := fwd_kernel_destindex_copy_kv K Dest_loc Out
    stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
    head_num BLOCK_DMODEL BLOCK_HEAD
  mbuf1 := Dest_loc
  mbuf2 := Dest_loc
  inp := K
  out := Out
  B := BLOCK_HEAD * BLOCK_DMODEL
  mwin1 := fun pid₀ _ => pid₀
  mwin2 := fun pid₀ _ => pid₀
  read := fun pid₀ _ _ _ j =>
    pid₀ * stride_k_bs + stride_k_h * (j.val / BLOCK_DMODEL) +
      stride_k_d * (j.val % BLOCK_DMODEL)
  write := fun _ _ m₁ _ j =>
    m₁ * stride_o_bs + stride_o_h * (j.val / BLOCK_DMODEL) +
      stride_o_d * (j.val % BLOCK_DMODEL)
  mask := fun _ _ _ _ j => j.val / BLOCK_DMODEL < head_num

open scoped VeriTile.Triton.MetaMasked2DKernelIO₁ in
/-- **The headline**: the dest-indexed KV scatter implements the pure copy `xs`
on its metadata IO signature — for every disjoint flat placement of the
buffers, every program id whose declared cells/lanes are in bounds, and every
launch state pinning the loaded destination row to `m₁` and the active source
lanes to `xs`, the translated pointer kernel terminates, every active lane
(`head < head_num`) of the **`m₁`-indexed** output row holds
`xs j`, and every other memory cell is unchanged.

Side condition `hInj` (required for truth, not convenience): the tile part of
the output address map is injective, i.e. no two cells of one program's tile
scatter to the same cell. Without it the masked scatter is last-writer-wins
and the per-lane readback is false. It does not mention `Dest_loc`: the loaded
row only shifts every address by the constant `m₁ · stride_o_bs`.

Proof: `MetaMasked2DKernelIO₁.Implements.intro` assembles the region-model
metadata triple with the flat-memory bridge side conditions. -/
specification fwd_kernel_destindex_copy_kv_correctness
    (K : RegionName) (Dest_loc : Region .nat) (Out : RegionName)
    (stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
      head_num BLOCK_DMODEL BLOCK_HEAD : Nat)
    (hInj : Function.Injective
      (fun idx : TileIndex [BLOCK_HEAD, BLOCK_DMODEL] =>
        stride_o_h * headIndex idx + stride_o_d * dimIndex idx)) :
    destindexCopyKvIO K Dest_loc Out
        stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
        head_num BLOCK_DMODEL BLOCK_HEAD
      ⊨ fun _ _ _ _ xs => xs := by
  refine MetaMasked2DKernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact fwd_kernel_destindex_copy_kv_flattenOk K Dest_loc Out
      stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
      head_num BLOCK_DMODEL BLOCK_HEAD
  · intro bounds s m₁ _m₂ hm₁ _hm₂ hb1 _hb2 hbr hbw
    simp only [destindexCopyKvIO] at hm₁ hb1 hbr hbw
    refine fwd_kernel_destindex_copy_kv_traceSafe K Dest_loc Out
      stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
      head_num BLOCK_DMODEL BLOCK_HEAD bounds s hb1
      (fun idx h1 => ?_) (fun idx h1 => ?_)
    · simpa using hbr (laneOf BLOCK_HEAD BLOCK_DMODEL idx) (by simpa using h1)
    · rw [hm₁]
      simpa using hbw (laneOf BLOCK_HEAD BLOCK_DMODEL idx) (by simpa using h1)
  · intro s₀ m₁ _m₂ xs hm₁ _hm₂ hx
    exact fwd_kernel_destindex_copy_kv_region_run K Dest_loc Out
      stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
      head_num BLOCK_DMODEL BLOCK_HEAD hInj s₀ m₁ xs hm₁ hx

end VeriTile.Bench.TritonBenchG.DestindexCopyKv2
