import VeriTile.Triton

/-!
# `relu_strided_buffer` — strict per-kernel correctness

`relu_strided_buffer.py`'s `relu_forward_kernel_rank_1` is the FlagGems
pointwise-codegen elementwise ReLU over a rank-1 strided buffer: each program
covers one or more `tile_size0`-wide tiles of the flat task space `[0, s0)`,
loads the input tile through a `tl.make_block_ptr` view with
`boundary_check`, applies the `relu_forward` helper (`tl.where(x > 0, x, 0)`),
and stores the tile through the output block-pointer view. The
`one_tile_per_cta: tl.constexpr` flag selects between a monolithic
one-tile-per-program body and a grid-stride loop
(`for j in range(0, tiles_per_cta)` with `num_ctas = tl.num_programs(0)`).

## Scope

This file verifies **the Triton kernel itself** — the per-program
`@triton.jit` body. The host launch (`relu_forward_wrapper_rank_1`, the
`heuristics_for_tile_size` / `heuristics_for_num_warps` choices, the
`num_ctas = min(65536, num_tiles)` grid, and how the runtime composes
per-program writes into one buffer) is the *trusted boundary*, not a proof
obligation here. Program ids and the launch-grid size are universally
quantified (`s.pids 0` / `s.numPids 0`), so each per-program statement covers
every program of its grid.

## Proof architecture

```
relu_strided_buffer_output_summary_general                    ← TOP THEOREM
  ├─ relu_one_tile_surface_toAlgorithm_supported     one-tile surface lowers
  ├─ relu_grid_stride_surface_toAlgorithm_supported  grid-stride surface lowers
  ├─ relu_one_tile_compute_correct              one_tile_per_cta=true branch
  │    └─ relu_one_tile_exec_correct            per-lane exec readback
  └─ relu_grid_stride_compute_correct           one_tile_per_cta=false branch
       └─ relu_grid_stride_exec_correct         per-(j, lane) exec readback
            └─ gs_loop_readback                 grid-stride loop invariant
                 └─ (one iteration = perTile_steps;
                     cross-iteration disjointness = gsTask_ne_of_ne)

shared per-tile engine (both branches, any state whose `tile_id` holds `T`):
  perTile_steps          tile_id0/offset0 bookkeeping → block-ptr load →
    ├─ makeBlockPtr_1d_eval                inlined relu_forward → block-ptr
    ├─ load_blockPtr_1d_checked_eval       store; plus untouched-cell /
    ├─ store_blockPtr_1d_checked_step      other-region / register
    └─ storeValue_where_relu               preservation facts
```

The spec is the reusable `TiledActivation.relu` oracle (`max 0 x`) applied to
the input cell this lane covers — a closed form over **input** memory, never a
read-back of the kernel's own output.

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float). The
`.to(in0_ptr.type.element_ty)` / `.to(out0_bptr.type.element_ty)` dtype
round-trip casts erase to the identity at the algorithm layer. Layout is
exactly the kernel's block-pointer contract: task index `t = tile_id·tile_size0
+ lane` reads `in0_ptr + t·in0_stride0` and writes `out0_ptr +
t·out0_stride0`, active iff `t < s0` (the `boundary_check=(0,)` guard on both
the load and the store). Honest side-conditions: a positive output stride
(`0 < out0_stride0`, which makes the store footprint injective — the wrapper
always passes a torch stride ≥ 1); the grid-stride branch additionally needs
`in0_ptr ≠ out0_ptr` (iteration `j+1` loads after iteration `j` stores) and
`0 < s.numPids 0` (a launched grid has at least one program; the semantics'
default is `1`).

## Translation-surface blocker

Translation-surface blocker: the `one_tile_per_cta` constexpr branch is split
into two Lean surfaces (`relu_forward_kernel_rank_1_one_tile_surface` for
`one_tile_per_cta = true`, `relu_forward_kernel_rank_1_grid_stride_surface`
for `one_tile_per_cta = false`), the `relu_forward` helper `@triton.jit`
(`return tl.where(x > 0, x, 0)`) is inlined at its single call site, and the
rank-1 stride-order constexprs (`in0_stride_order0 = out0_stride_order0 = 0`,
their only rank-1 value) are instantiated in the `boundary_check=(...)` /
`order=(...)` slots. The textual py↔lean scans in
`bench/audit_tritonbench_g.sh` exempt this port on this marker (registered in
`proof_blockers.md`).
-/

namespace VeriTile.Bench.TritonBenchG.ReluStridedBuffer

open VeriTile.Triton

set_option linter.unusedSimpArgs false
set_option linter.unusedVariables false

/-! **★ Main theorem:** `relu_strided_buffer_output_summary_general`
(dimension-general `s0`, `in0_stride0`, `out0_stride0`, `tile_size0`,
`tiles_per_cta`); the Python benchmark shapes are instantiations. -/

/-! # ══════════ CORRECT — genuine / dimension-general (review this) ══════════ -/

section Correct

/-! ## Surfaces -/

/-- Faithful transcription of `relu_strided_buffer.py`'s
`relu_forward_kernel_rank_1`, specialized to the `one_tile_per_cta = true`
(monolithic) branch: one `tile_size0`-wide tile per program, block-pointer
load/store with `boundary_check` on axis 0, `relu_forward` inlined as
`tl.where(in0 > 0, in0, 0)`. -/
def relu_forward_kernel_rank_1_one_tile_surface
    (in0_ptr out0_ptr : RegionName)
    (in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta tile_size0 : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0)
  num_tiles0 = tl.cdiv($(s0), $(tile_size0))
  tile_id = pid
  tile_id0 = tile_id
  offset0 = tile_id0 * $(tile_size0)
  in0_bptr = tl.make_block_ptr(base=in0_ptr, shape=($(s0)), strides=($(in0_stride0)),
    offsets=(offset0), block_shape=($(tile_size0)), order=(0))
  in0 = (tl.load(in0_bptr, boundary_check=([0] : List Nat))).to(in0_ptr.type.element_ty)
  out0 = tl.where(in0 > 0, in0, 0)
  out0_bptr = tl.make_block_ptr(base=out0_ptr, shape=($(s0)), strides=($(out0_stride0)),
    offsets=(offset0), block_shape=($(tile_size0)), order=(0))
  tl.store(out0_bptr, (out0).to(out0_bptr.type.element_ty), boundary_check=([0] : List Nat))
}

/-- Faithful transcription of `relu_forward_kernel_rank_1`, specialized to the
`one_tile_per_cta = false` (grid-stride-loop) branch: program `pid` covers
tiles `pid + j·num_ctas` for `j < tiles_per_cta`, with
`num_ctas = tl.num_programs(0)`. -/
def relu_forward_kernel_rank_1_grid_stride_surface
    (in0_ptr out0_ptr : RegionName)
    (in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta tile_size0 : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0)
  num_tiles0 = tl.cdiv($(s0), $(tile_size0))
  num_ctas = tl.num_programs(0)
  for j in range($(0), $(tiles_per_cta)) {
    tile_id = pid + j * num_ctas
    tile_id0 = tile_id
    offset0 = tile_id0 * $(tile_size0)
    in0_bptr = tl.make_block_ptr(base=in0_ptr, shape=($(s0)), strides=($(in0_stride0)),
      offsets=(offset0), block_shape=($(tile_size0)), order=(0))
    in0 = (tl.load(in0_bptr, boundary_check=([0] : List Nat))).to(in0_ptr.type.element_ty)
    out0 = tl.where(in0 > 0, in0, 0)
    out0_bptr = tl.make_block_ptr(base=out0_ptr, shape=($(s0)), strides=($(out0_stride0)),
      offsets=(offset0), block_shape=($(tile_size0)), order=(0))
    tl.store(out0_bptr, (out0).to(out0_bptr.type.element_ty), boundary_check=([0] : List Nat))
  }
}

/-- The one-tile surface lowers to the algorithm layer. -/
theorem relu_one_tile_surface_toAlgorithm_supported
    (in0_ptr out0_ptr : RegionName)
    (in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta tile_size0 : Nat) :
    ∃ alg, (relu_forward_kernel_rank_1_one_tile_surface in0_ptr out0_ptr
      in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta tile_size0).toAlgorithm?
        = Except.ok alg := by
  simp [relu_forward_kernel_rank_1_one_tile_surface,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-- The grid-stride surface (loop and all) lowers to the algorithm layer. -/
theorem relu_grid_stride_surface_toAlgorithm_supported
    (in0_ptr out0_ptr : RegionName)
    (in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta tile_size0 : Nat) :
    ∃ alg, (relu_forward_kernel_rank_1_grid_stride_surface in0_ptr out0_ptr
      in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta tile_size0).toAlgorithm?
        = Except.ok alg := by
  simp [relu_forward_kernel_rank_1_grid_stride_surface,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-! ## Closed-form spec

The genuine elementwise specification: task `t` of the flat task space holds
`relu(in0[t·in0_stride0]) = max 0 (in0[t·in0_stride0])` at
`out0[t·out0_stride0]`. A closed form over **input** memory only. -/

/-- Flat task index covered by lane `i` of tile `tile_id0`. -/
def taskIndex (tile_id0 tile_size0 : Nat) (i : Fin tile_size0) : Nat :=
  tile_id0 * tile_size0 + i.val

/-- Genuine spec value of task `t`: `relu` of the input cell `t·in0_stride0`. -/
noncomputable def reluSpec (s : BlockState) (in0_ptr : RegionName)
    (in0_stride0 t : Nat) : ℝ :=
  TiledActivation.relu (s.readMem in0_ptr (t * in0_stride0))

/-- Output-footprint injectivity for one tile: with a positive output stride,
distinct lanes of a tile hit distinct `out0` cells. -/
theorem oneTileOutAddr_injective (tile_id0 : Nat) {out0_stride0 tile_size0 : Nat}
    (hStride : 0 < out0_stride0) :
    Function.Injective (fun i : Fin tile_size0 =>
      taskIndex tile_id0 tile_size0 i * out0_stride0) := by
  intro a b h
  simp only [taskIndex] at h
  exact Fin.ext (Nat.add_left_cancel (Nat.eq_of_mul_eq_mul_right hStride h))

/-! ## Shared statement lists (algorithm layer)

Stepping `exec` through `simp` on the erased `toAlgKernel` term is too
expensive; instead the two surfaces' bodies are named as explicit `Stmt`
lists (checked by `rfl` against the macro output) and stepped lemma-by-lemma,
following the `matmul_tma` precedent. -/

/-- The per-tile tail shared by both branches (everything after `tile_id` is
resolved): `tile_id0` / `offset0` bookkeeping, the input block-pointer +
boundary-checked load, the inlined `relu_forward` `tl.where`, and the output
block-pointer + boundary-checked store. -/
def perTileStmts (in0_ptr out0_ptr : RegionName)
    (in0_stride0 out0_stride0 s0 tile_size0 : Nat) : List Stmt :=
  [ Stmt.assign .nat [] "tile_id0" (Op.ref .nat [] "tile_id"),
    Stmt.assign .nat [] "offset0"
      (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "tile_id0")
        (Op.constNat tile_size0)),
    Stmt.assign .blockPtr [tile_size0] "in0_bptr"
      (Op.makeBlockPtrDynOffsets in0_ptr (Op.constNat 0) [s0] [tile_size0]
        [in0_stride0] [Op.ref .nat [] "offset0"]),
    Stmt.assign .real [tile_size0] "in0"
      (Op.load .real
        (MemAccess.blockPtr (Op.ref .blockPtr [tile_size0] "in0_bptr") [0])
        MaskOpt.none),
    Stmt.assign .real [tile_size0] "out0"
      ((Op.gt ComparableDType.real Broadcast.scalarR
          (Op.ref .real [tile_size0] "in0") (Op.const 0)).where
        (Op.ref .real [tile_size0] "in0") ((Op.const 0).broadcast [tile_size0])),
    Stmt.assign .blockPtr [tile_size0] "out0_bptr"
      (Op.makeBlockPtrDynOffsets out0_ptr (Op.constNat 0) [s0] [tile_size0]
        [out0_stride0] [Op.ref .nat [] "offset0"]),
    Stmt.store .real [tile_size0]
      (MemAccess.blockPtr (Op.ref .blockPtr [tile_size0] "out0_bptr") [0])
      (Op.ref .real [tile_size0] "out0") MaskOpt.none ]

/-- `num_tiles0 = tl.cdiv(s0, tile_size0)` at the algorithm layer. -/
def numTiles0Stmt (s0 tile_size0 : Nat) : Stmt :=
  Stmt.assign .nat [] "num_tiles0"
    (Op.div NumericDType.nat Broadcast.nil
      (Op.sub NumericDType.nat Broadcast.nil
        (Op.add NumericDType.nat Broadcast.nil (Op.constNat s0)
          (Op.constNat tile_size0))
        (Op.constNat 1))
      (Op.constNat tile_size0))

/-- Algorithm-layer body of the one-tile surface. -/
theorem oneTile_body_eq (in0_ptr out0_ptr : RegionName)
    (in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta tile_size0 : Nat) :
    (relu_forward_kernel_rank_1_one_tile_surface in0_ptr out0_ptr
      in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta tile_size0).toAlgKernel.body
    = Stmt.assign .nat [] "pid" (Op.programId 0)
      :: numTiles0Stmt s0 tile_size0
      :: Stmt.assign .nat [] "tile_id" (Op.ref .nat [] "pid")
      :: perTileStmts in0_ptr out0_ptr in0_stride0 out0_stride0 s0 tile_size0 := rfl

/-- Algorithm-layer body of the grid-stride surface. -/
theorem gridStride_body_eq (in0_ptr out0_ptr : RegionName)
    (in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta tile_size0 : Nat) :
    (relu_forward_kernel_rank_1_grid_stride_surface in0_ptr out0_ptr
      in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta tile_size0).toAlgKernel.body
    = [ Stmt.assign .nat [] "pid" (Op.programId 0),
        numTiles0Stmt s0 tile_size0,
        Stmt.assign .nat [] "num_ctas" (Op.numPrograms 0),
        Stmt.forRangeDyn "j" (Op.constNat 0) (Op.constNat tiles_per_cta)
          (Op.constNat 1)
          (Stmt.assign .nat [] "tile_id"
              (Op.add NumericDType.nat Broadcast.nil (Op.ref .nat [] "pid")
                (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "j")
                  (Op.ref .nat [] "num_ctas")))
            :: perTileStmts in0_ptr out0_ptr in0_stride0 out0_stride0 s0 tile_size0) ] := rfl

/-! ## Stepping helpers — 1D boundary-checked block pointers -/

/-- 1D `tl.make_block_ptr` whose offset comes from a `nat` register: evaluates
to the constant `BlockPtr` tile at the register's scalar value. -/
theorem makeBlockPtr_1d_eval (R : RegionName) (len stride BT : Nat)
    (offName : RegName) (t : BlockState) (V : Tile .nat []) (off : Nat)
    (hoff : t.regs .nat [] offName = some V)
    (hV : V.data PUnit.unit = off) :
    evalOp (Op.makeBlockPtrDynOffsets R (Op.constNat 0) [len] [BT] [stride]
        [Op.ref .nat [] offName]) t
      = some (⟨fun _ : TileIndex [BT] =>
          { region := R, baseOffset := 0, parentShape := [len],
            blockShape := [BT], strides := [stride], offsets := [off] }⟩
          : Tile .blockPtr [BT]) := by
  rw [makeBlockPtr2_eval]
  simp only [evalOp_constNat, evalOp_ref, hoff, List.mapM_cons, List.mapM_nil,
    Option.bind_some, Option.pure_def, Option.bind_none, Option.bind_eq_bind,
    hV]
  rfl

/-- Boundary-checked (`boundary_check=(0,)`) 1D block-pointer real load:
in-bounds lanes (`off + i < len`) read `readMem` at `(off + i)·stride`;
out-of-bounds lanes pad with the load default `0`. -/
theorem load_blockPtr_1d_checked_eval (R : RegionName) (len stride BT off : Nat)
    (bpName : RegName) (t : BlockState)
    (hbp : t.regs .blockPtr [BT] bpName = some (⟨fun _ : TileIndex [BT] =>
        { region := R, baseOffset := 0, parentShape := [len],
          blockShape := [BT], strides := [stride], offsets := [off] }⟩ :
        Tile .blockPtr [BT])) :
    evalOp (Op.load .real
        (MemAccess.blockPtr (Op.ref .blockPtr [BT] bpName) [0]) MaskOpt.none) t
      = some (⟨fun idx : TileIndex [BT] =>
          if off + idx.1.val < len then
            some (t.readMem R ((off + idx.1.val) * stride))
          else some 0⟩ : Tile .real [BT]) := by
  simp only [evalOp, evalOp_ref, hbp, Option.bind_some, Option.bind]
  refine congrArg some ?_
  congr 1
  funext idx
  obtain ⟨i1, u⟩ := idx
  simp only [TileShape.indexToList, BlockPtr.inBounds_1d, BlockPtr.address_1d,
    Nat.zero_add, BlockState.readMemValue_real, decide_eq_true_eq]
  by_cases h : off + i1.val < len
  · simp [h]
  · simp [h, BlockState.defaultCarrier]

/-- Boundary-checked (`boundary_check=(0,)`) 1D block-pointer real store: the
resulting state is the masked scatter that writes `(vt i).unbotD 0` at
`(off + i)·stride` exactly on the in-bounds lanes. -/
theorem store_blockPtr_1d_checked_step (R : RegionName) (len stride BT off : Nat)
    (bpName vName : RegName) (vt : Tile .real [BT]) (t : BlockState)
    (hbp : t.regs .blockPtr [BT] bpName = some (⟨fun _ : TileIndex [BT] =>
        { region := R, baseOffset := 0, parentShape := [len],
          blockShape := [BT], strides := [stride], offsets := [off] }⟩ :
        Tile .blockPtr [BT]))
    (hv : t.regs .real [BT] vName = some vt) :
    stepStmt (Stmt.store .real [BT]
        (MemAccess.blockPtr (Op.ref .blockPtr [BT] bpName) [0])
        (Op.ref .real [BT] vName) MaskOpt.none) t
      = some ((TileShape.allIndices [BT]).foldl
          (fun acc idx =>
            if off + idx.1.val < len then
              acc.writeMem R ((off + idx.1.val) * stride) ((vt.data idx).unbotD 0)
            else acc) t) := by
  simp only [stepStmt, evalOp_ref, hv, hbp, Option.bind_some, Option.bind, bind]
  refine congrArg some ?_
  apply List.foldl_ext
  intro acc idx _
  obtain ⟨i1, u⟩ := idx
  simp only [TileShape.indexToList, BlockPtr.inBounds_1d, BlockPtr.address_1d,
    Nat.zero_add, Bool.true_and, BlockState.writeMemTyped_real,
    FloatDType.real_storeValue, decide_eq_true_eq]

/-- `(Op.const c).broadcast shape` evaluates to the constant tile. -/
private theorem evalOp_broadcast_const (c : ℝ) (shape : TileShape)
    (s : BlockState) :
    evalOp ((Op.const c).broadcast shape) s
      = some (⟨fun _ => some c⟩ : Tile .real shape) := by
  simp only [evalOp, evalOp_const, Option.bind_some]
  rfl

/-- The inlined `relu_forward` store value: `tl.where(x > 0, x, 0)` followed by
the store's `⊥ ↦ 0` fallback is exactly `relu x = max 0 x` on a loaded lane. -/
private theorem storeValue_where_relu (x : ℝ) :
    (WithBot.unbotD 0
        (if ComparableDType.real.gt (some x) (some 0) then (some x : WithBot ℝ)
         else some 0))
      = TiledActivation.relu x := by
  by_cases h : (0 : ℝ) < x
  · rw [if_pos]
    · simp [TiledActivation.relu, max_eq_right h.le]
    · simp only [ComparableDType.real_gt_eq_true, gt_iff_lt]
      exact WithBot.coe_lt_coe.mpr h
  · rw [if_neg]
    · simp [TiledActivation.relu, max_eq_left (not_lt.mp h)]
    · simp only [ComparableDType.real_gt_eq_true, gt_iff_lt]
      intro hc
      exact h (WithBot.coe_lt_coe.mp hc)

/-- Lane-level output-footprint injectivity over `TileIndex [tile_size0]`. -/
private theorem outAddr_tileIndex_injective (T : Nat)
    {out0_stride0 tile_size0 : Nat} (hStride : 0 < out0_stride0) :
    Function.Injective (fun idx : TileIndex [tile_size0] =>
      (T * tile_size0 + idx.1.val) * out0_stride0) := by
  rintro ⟨a, u⟩ ⟨b, u'⟩ h
  simp only at h
  have hab : a = b :=
    Fin.ext (Nat.add_left_cancel (Nat.eq_of_mul_eq_mul_right hStride h))
  cases hab; cases u; cases u'; rfl

/-! ## One tile of work — shared by both branches

`perTile_steps` executes the seven shared statements (`tile_id0` bookkeeping →
block-ptr load → `relu_forward` → block-ptr store) from any state whose
`tile_id` register holds `T`, and characterizes the result: active lanes of
tile `T` receive the genuine ReLU value, everything else — other `out0` cells,
other regions, and all registers except the six per-tile scratch registers —
is untouched. -/

set_option maxHeartbeats 1000000 in
theorem perTile_steps
    (in0_ptr out0_ptr : RegionName)
    (in0_stride0 out0_stride0 s0 tile_size0 : Nat)
    (hStride : 0 < out0_stride0)
    (T : Nat) (t : BlockState) (V : Tile .nat [])
    (hT : t.regs .nat [] "tile_id" = some V)
    (hV : V.data PUnit.unit = T) :
    ∃ t', stepStmts (perTileStmts in0_ptr out0_ptr in0_stride0 out0_stride0
        s0 tile_size0) t = some t'
      ∧ (∀ i : Fin tile_size0, taskIndex T tile_size0 i < s0 →
          t'.readMem out0_ptr (taskIndex T tile_size0 i * out0_stride0)
            = reluSpec t in0_ptr in0_stride0 (taskIndex T tile_size0 i))
      ∧ (∀ o : Nat, (∀ i : Fin tile_size0, taskIndex T tile_size0 i < s0 →
            o ≠ taskIndex T tile_size0 i * out0_stride0) →
          t'.readMem out0_ptr o = t.readMem out0_ptr o)
      ∧ (∀ (R : RegionName) (o : Nat), R ≠ out0_ptr →
          t'.readMem R o = t.readMem R o)
      ∧ (∀ (dtype : TileDType) (shape : TileShape) (name : RegName),
          name ≠ "tile_id0" → name ≠ "offset0" → name ≠ "in0_bptr" →
          name ≠ "in0" → name ≠ "out0" → name ≠ "out0_bptr" →
          t'.regs dtype shape name = t.regs dtype shape name) := by
  -- the intermediate register values (raw evaluation results)
  set offV : Tile .nat [] :=
    Tile.bop (NumericDType.mul .nat) Broadcast.nil V (Tile.scalar tile_size0)
    with hoffV
  set inBP : Tile .blockPtr [tile_size0] := ⟨fun _ =>
    { region := in0_ptr, baseOffset := 0, parentShape := [s0],
      blockShape := [tile_size0], strides := [in0_stride0],
      offsets := [T * tile_size0] }⟩ with hinBP
  set outBP : Tile .blockPtr [tile_size0] := ⟨fun _ =>
    { region := out0_ptr, baseOffset := 0, parentShape := [s0],
      blockShape := [tile_size0], strides := [out0_stride0],
      offsets := [T * tile_size0] }⟩ with houtBP
  set inT : Tile .real [tile_size0] := ⟨fun idx =>
    if T * tile_size0 + idx.1.val < s0 then
      some (t.readMem in0_ptr ((T * tile_size0 + idx.1.val) * in0_stride0))
    else some 0⟩ with hinT
  set outT : Tile .real [tile_size0] :=
    Tile.select
      (Tile.cop (ComparableDType.gt .real) Broadcast.scalarR inT
        (Tile.scalar (some 0)))
      inT (⟨fun _ => some 0⟩ : Tile .real [tile_size0]) with houtT
  -- the six register-writing statements
  set t1 := t.setReg "tile_id0" .nat [] V with ht1
  set t2 := t1.setReg "offset0" .nat [] offV with ht2
  set t3 := t2.setReg "in0_bptr" .blockPtr [tile_size0] inBP with ht3
  set t4 := t3.setReg "in0" .real [tile_size0] inT with ht4
  set t5 := t4.setReg "out0" .real [tile_size0] outT with ht5
  set t6 := t5.setReg "out0_bptr" .blockPtr [tile_size0] outBP with ht6
  have hoffData : offV.data PUnit.unit = T * tile_size0 := by
    rw [hoffV, Tile.bop_data]
    show NumericDType.mul .nat (V.data PUnit.unit) tile_size0 = T * tile_size0
    rw [hV]
    rfl
  have hstep : stepStmts (perTileStmts in0_ptr out0_ptr in0_stride0
      out0_stride0 s0 tile_size0) t
      = some ((TileShape.allIndices [tile_size0]).foldl
          (fun acc idx =>
            if T * tile_size0 + idx.1.val < s0 then
              acc.writeMem out0_ptr ((T * tile_size0 + idx.1.val) * out0_stride0)
                ((outT.data idx).unbotD 0)
            else acc) t6) := by
    simp only [perTileStmts]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (show evalOp (Op.ref .nat [] "tile_id") t = some V by simp [hT]))]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (show evalOp (Op.mul NumericDType.nat Broadcast.nil
          (Op.ref .nat [] "tile_id0") (Op.constNat tile_size0)) t1
        = some offV by simp [ht1, hoffV]))]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (makeBlockPtr_1d_eval in0_ptr s0 in0_stride0 tile_size0 "offset0" t2
        offV (T * tile_size0) (by simp [ht2]) hoffData))]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (show evalOp (Op.load .real (MemAccess.blockPtr
          (Op.ref .blockPtr [tile_size0] "in0_bptr") [0]) MaskOpt.none) t3
        = some inT from
        load_blockPtr_1d_checked_eval in0_ptr s0 in0_stride0 tile_size0
          (T * tile_size0) "in0_bptr" t3 (by simp [ht3, hinBP])))]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (show evalOp ((Op.gt ComparableDType.real Broadcast.scalarR
            (Op.ref .real [tile_size0] "in0") (Op.const 0)).where
          (Op.ref .real [tile_size0] "in0")
          ((Op.const 0).broadcast [tile_size0])) t4
        = some outT by
        simp only [evalOp_where, evalOp_gt, evalOp_ref, evalOp_const,
          evalOp_broadcast_const, Option.bind_some,
          show t4.regs .real [tile_size0] "in0" = some inT by simp [ht4],
          houtT]
        rfl))]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (makeBlockPtr_1d_eval out0_ptr s0 out0_stride0 tile_size0 "offset0" t5
        offV (T * tile_size0) (by simp [ht5, ht4, ht3, ht2]) hoffData))]
    rw [stepStmts.cons_some (store_blockPtr_1d_checked_step out0_ptr s0
      out0_stride0 tile_size0 (T * tile_size0) "out0_bptr" "out0" outT t6
      (by simp [ht6, houtBP]) (by simp [ht6, ht5]))]
    exact stepStmts.nil
  refine ⟨_, hstep, ?_, ?_, ?_, ?_⟩
  · -- active lanes: genuine ReLU value
    intro i hi
    simp only [taskIndex] at hi ⊢
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
      (outAddr_tileIndex_injective T hStride) (i, PUnit.unit)]
    rw [if_pos hi]
    simp only [houtT, hinT, Tile.select_data, Tile.cop_data,
      Broadcast.leftIndex_scalarR, Broadcast.rightIndex_scalarR,
      Tile.scalar, if_pos hi, reluSpec]
    exact storeValue_where_relu _
  · -- untouched `out0` cells
    intro o ho
    have hpres := BlockState.scatter_prop_masked_preserves_other_offset
      (region := out0_ptr)
      (fun idx : TileIndex [tile_size0] =>
        (T * tile_size0 + idx.1.val) * out0_stride0)
      (fun idx : TileIndex [tile_size0] => (outT.data idx).unbotD 0)
      (fun idx : TileIndex [tile_size0] => T * tile_size0 + idx.1.val < s0)
      o (fun idx hidx => by
        have := ho idx.1 (by simpa [taskIndex] using hidx)
        simpa [taskIndex] using Ne.symm this)
      (TileShape.allIndices [tile_size0]) t6
    rw [hpres]
    simp [ht6, ht5, ht4, ht3, ht2, ht1]
  · -- other regions untouched
    intro R o hR
    have hpres := BlockState.scatter_prop_masked_preserves_other_region
      (region := out0_ptr)
      (fun idx : TileIndex [tile_size0] =>
        (T * tile_size0 + idx.1.val) * out0_stride0)
      (fun idx : TileIndex [tile_size0] => (outT.data idx).unbotD 0)
      (fun idx : TileIndex [tile_size0] => T * tile_size0 + idx.1.val < s0)
      R hR o (TileShape.allIndices [tile_size0]) t6
    rw [hpres]
    simp [ht6, ht5, ht4, ht3, ht2, ht1]
  · -- registers other than the six per-tile scratch registers untouched
    intro dtype shape name h1 h2 h3 h4 h5 h6
    rw [BlockState.foldl_writeMem_prop_masked_regs]
    simp [ht6, ht5, ht4, ht3, ht2, ht1, h1, h2, h3, h4, h5, h6]

/-- Raw evaluated `tl.cdiv(s0, tile_size0)` register value (assigned to
`num_tiles0`, never read back by either branch). -/
noncomputable def numTiles0Val (s0 tile_size0 : Nat) : Tile .nat [] :=
  Tile.bop (NumericDType.div .nat) Broadcast.nil
    (Tile.bop (NumericDType.sub .nat) Broadcast.nil
      (Tile.bop (NumericDType.add .nat) Broadcast.nil (Tile.scalar s0)
        (Tile.scalar tile_size0))
      (Tile.scalar 1))
    (Tile.scalar tile_size0)

private theorem numTiles0_step (s0 tile_size0 : Nat) (t : BlockState) :
    stepStmt (numTiles0Stmt s0 tile_size0) t
      = some (t.setReg "num_tiles0" .nat [] (numTiles0Val s0 tile_size0)) := by
  simp only [numTiles0Stmt]
  exact stepStmt_assign_eq_some (by
    simp only [evalOp_div, evalOp_sub, evalOp_add, evalOp_constNat,
      Option.bind_some, numTiles0Val]
    rfl)

/-! ## `one_tile_per_cta = true` branch -/

/-- Per-lane exec readback for the one-tile branch: active lanes
(`pid·tile_size0 + i < s0`) hold the genuine ReLU value; out-of-bounds lanes
are preserved (the store's `boundary_check` masks them). -/
theorem relu_one_tile_exec_correct
    (in0_ptr out0_ptr : RegionName)
    (in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta tile_size0 : Nat)
    (s : BlockState)
    (hStride : 0 < out0_stride0) :
    ∀ i : Fin tile_size0,
      (exec (relu_forward_kernel_rank_1_one_tile_surface in0_ptr out0_ptr
          in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta tile_size0) s).map
          (·.readMem out0_ptr (taskIndex (s.pids 0) tile_size0 i * out0_stride0))
        = some (if taskIndex (s.pids 0) tile_size0 i < s0
            then reluSpec s in0_ptr in0_stride0 (taskIndex (s.pids 0) tile_size0 i)
            else s.readMem out0_ptr
              (taskIndex (s.pids 0) tile_size0 i * out0_stride0)) := by
  intro i
  set sp3 := ((s.setReg "pid" .nat [] (Tile.scalar (s.pids 0))).setReg
      "num_tiles0" .nat [] (numTiles0Val s0 tile_size0)).setReg
      "tile_id" .nat [] (Tile.scalar (s.pids 0)) with hsp3
  obtain ⟨t', hstep, hact, hpresOut, _, _⟩ :=
    perTile_steps in0_ptr out0_ptr in0_stride0 out0_stride0 s0 tile_size0
      hStride (s.pids 0) sp3 (Tile.scalar (s.pids 0)) (by simp [hsp3]) rfl
  rw [show exec (relu_forward_kernel_rank_1_one_tile_surface in0_ptr out0_ptr
        in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta
        tile_size0).toAlgKernel s
      = stepStmts ((relu_forward_kernel_rank_1_one_tile_surface in0_ptr out0_ptr
        in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta
        tile_size0).toAlgKernel).body s from rfl,
    oneTile_body_eq]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 0 s))]
  rw [stepStmts.cons_some (numTiles0_step s0 tile_size0 _)]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ref .nat [] "pid")
        ((s.setReg "pid" .nat [] (Tile.scalar (s.pids 0))).setReg
          "num_tiles0" .nat [] (numTiles0Val s0 tile_size0))
      = some (Tile.scalar (s.pids 0)) by simp))]
  rw [show ((s.setReg "pid" .nat [] (Tile.scalar (s.pids 0))).setReg
        "num_tiles0" .nat [] (numTiles0Val s0 tile_size0)).setReg
        "tile_id" .nat [] (Tile.scalar (s.pids 0)) = sp3 from rfl]
  rw [hstep, Option.map_some]
  by_cases hlt : taskIndex (s.pids 0) tile_size0 i < s0
  · rw [if_pos hlt, hact i hlt]
    rfl
  · rw [if_neg hlt, hpresOut _ (fun i' hi' heq => hlt (by
      have h1 := Nat.eq_of_mul_eq_mul_right hStride heq
      omega))]
    rfl

/-- Compute-facing correctness for the `one_tile_per_cta = true` branch. -/
theorem relu_one_tile_compute_correct
    (in0_ptr out0_ptr : RegionName)
    (in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta tile_size0 : Nat)
    (s : BlockState) (hStride : 0 < out0_stride0) :
    ComputeCorrect.Realizes
      (kernel := relu_forward_kernel_rank_1_one_tile_surface in0_ptr out0_ptr
        in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta tile_size0)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin tile_size0 => taskIndex (s.pids 0) tile_size0 i < s0)
        (fun i => (out0_ptr, taskIndex (s.pids 0) tile_size0 i * out0_stride0)))
      (expected := fun i =>
        reluSpec s in0_ptr in0_stride0 (taskIndex (s.pids 0) tile_size0 i)) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [relu_forward_kernel_rank_1_one_tile_surface,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0' s' hExec hs0
  subst s0'
  intro i hActive
  have h := relu_one_tile_exec_correct in0_ptr out0_ptr in0_stride0 out0_stride0
    s0 num_tasks tiles_per_cta tile_size0 s hStride i
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-! ## `one_tile_per_cta = false` (grid-stride) branch -/

/-- Distinct grid-stride iterations of one program cover disjoint task
indices (`tile_id = pid + j·num_ctas` are distinct for distinct `j` when the
grid has at least one program). -/
private theorem gsTask_ne_of_ne {tile_size0 : Nat} (P C : Nat) (hC : 0 < C)
    {j j' : Nat} (hne : j ≠ j') (i i' : Fin tile_size0) :
    taskIndex (P + j * C) tile_size0 i ≠ taskIndex (P + j' * C) tile_size0 i' := by
  have key : ∀ a b : Nat, a < b → ∀ x y : Fin tile_size0,
      taskIndex (P + a * C) tile_size0 x < taskIndex (P + b * C) tile_size0 y := by
    intro a b hab x y
    have h1 : a * C + 1 ≤ b * C := by
      have h := Nat.mul_le_mul_right C (Nat.succ_le_of_lt hab)
      calc a * C + 1 ≤ a * C + C := by omega
        _ = (a + 1) * C := by ring
        _ ≤ b * C := h
    have h2 : (P + a * C + 1) * tile_size0 ≤ (P + b * C) * tile_size0 :=
      Nat.mul_le_mul_right _ (by omega)
    have hx := x.isLt
    calc taskIndex (P + a * C) tile_size0 x
        = (P + a * C) * tile_size0 + x.val := rfl
      _ < (P + a * C + 1) * tile_size0 := by
          have hexp : (P + a * C + 1) * tile_size0
              = (P + a * C) * tile_size0 + tile_size0 := by ring
          omega
      _ ≤ (P + b * C) * tile_size0 := h2
      _ ≤ taskIndex (P + b * C) tile_size0 y := Nat.le_add_right _ _
  rcases Nat.lt_or_ge j j' with h | h
  · exact Nat.ne_of_lt (key j j' h i i')
  · exact (Nat.ne_of_lt (key j' j
      (Nat.lt_of_le_of_ne h (fun e => hne e.symm)) i' i)).symm

set_option maxHeartbeats 1000000 in
/-- **Grid-stride loop invariant.** Running the loop from iteration `c` (with
`n` iterations remaining) on any state whose `pid` / `num_ctas` registers hold
`P` / `C`: every active lane of every remaining tile receives the genuine ReLU
value, all other `out0` cells and the whole `in0` region are untouched. -/
theorem gs_loop_readback
    (in0_ptr out0_ptr : RegionName)
    (in0_stride0 out0_stride0 s0 tile_size0 : Nat)
    (hStride : 0 < out0_stride0) (hDisj : in0_ptr ≠ out0_ptr)
    (P C : Nat) (hC : 0 < C) (tiles_per_cta : Nat) :
    ∀ (n c : Nat), c + n = tiles_per_cta →
    ∀ (t : BlockState),
      t.regs .nat [] "pid" = some (Tile.scalar P) →
      t.regs .nat [] "num_ctas" = some (Tile.scalar C) →
      ∀ s', stepForRangeAux "j" c tiles_per_cta 1
          (Stmt.assign .nat [] "tile_id"
              (Op.add NumericDType.nat Broadcast.nil (Op.ref .nat [] "pid")
                (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "j")
                  (Op.ref .nat [] "num_ctas")))
            :: perTileStmts in0_ptr out0_ptr in0_stride0 out0_stride0 s0 tile_size0)
          t = some s' →
        (∀ (j : Nat) (i : Fin tile_size0), c ≤ j → j < tiles_per_cta →
          taskIndex (P + j * C) tile_size0 i < s0 →
          s'.readMem out0_ptr (taskIndex (P + j * C) tile_size0 i * out0_stride0)
            = reluSpec t in0_ptr in0_stride0 (taskIndex (P + j * C) tile_size0 i))
        ∧ (∀ o : Nat, (∀ (j : Nat) (i : Fin tile_size0), c ≤ j →
              j < tiles_per_cta → taskIndex (P + j * C) tile_size0 i < s0 →
              o ≠ taskIndex (P + j * C) tile_size0 i * out0_stride0) →
            s'.readMem out0_ptr o = t.readMem out0_ptr o)
        ∧ (∀ o : Nat, s'.readMem in0_ptr o = t.readMem in0_ptr o) := by
  intro n
  induction n with
  | zero =>
      intro c hc t hpid hctas s' hs'
      rw [stepForRangeAux.step_ge one_ne_zero (by omega)] at hs'
      cases hs'
      refine ⟨?_, ?_, ?_⟩
      · intro j i hcj hjlt
        exact absurd hjlt (by omega)
      · intro o _
        rfl
      · intro o
        rfl
  | succ n ih =>
      intro c hc t hpid hctas s' hs'
      have hlt : c < tiles_per_cta := by omega
      rw [stepForRangeAux.step_lt one_ne_zero hlt] at hs'
      set tJ := t.setReg "j" .nat [] (Tile.scalar c) with htJ
      set addV : Tile .nat [] := Tile.bop (NumericDType.add .nat) Broadcast.nil
        (Tile.scalar P) (Tile.bop (NumericDType.mul .nat) Broadcast.nil
          (Tile.scalar c) (Tile.scalar C)) with haddV
      set tA := tJ.setReg "tile_id" .nat [] addV with htA
      obtain ⟨t', hstepBody, hact, hpresOut, hpresReg, hregs⟩ :=
        perTile_steps in0_ptr out0_ptr in0_stride0 out0_stride0 s0 tile_size0
          hStride (P + c * C) tA addV (by simp [htA]) (by rw [haddV]; rfl)
      have hbody : stepStmts
          (Stmt.assign .nat [] "tile_id"
              (Op.add NumericDType.nat Broadcast.nil (Op.ref .nat [] "pid")
                (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "j")
                  (Op.ref .nat [] "num_ctas")))
            :: perTileStmts in0_ptr out0_ptr in0_stride0 out0_stride0 s0
              tile_size0)
          tJ = some t' := by
        rw [stepStmts.cons_some (stepStmt_assign_eq_some
          (show evalOp (Op.add NumericDType.nat Broadcast.nil
              (Op.ref .nat [] "pid")
              (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "j")
                (Op.ref .nat [] "num_ctas"))) tJ = some addV by
            simp only [evalOp_add, evalOp_mul, evalOp_ref,
              show tJ.regs .nat [] "pid" = some (Tile.scalar P) by
                simp [htJ, hpid],
              show tJ.regs .nat [] "j" = some (Tile.scalar c) by simp [htJ],
              show tJ.regs .nat [] "num_ctas" = some (Tile.scalar C) by
                simp [htJ, hctas],
              Option.bind_some, haddV]
            rfl))]
        exact hstepBody
      rw [hbody, Option.bind_some] at hs'
      have hpid' : t'.regs .nat [] "pid" = some (Tile.scalar P) := by
        rw [hregs _ _ _ (by decide) (by decide) (by decide) (by decide)
          (by decide) (by decide)]
        simp [htA, htJ, hpid]
      have hctas' : t'.regs .nat [] "num_ctas" = some (Tile.scalar C) := by
        rw [hregs _ _ _ (by decide) (by decide) (by decide) (by decide)
          (by decide) (by decide)]
        simp [htA, htJ, hctas]
      obtain ⟨ih1, ih2, ih3⟩ := ih (c + 1) (by omega) t' hpid' hctas' s' hs'
      have hmemIn : ∀ o : Nat, t'.readMem in0_ptr o = t.readMem in0_ptr o := by
        intro o
        rw [hpresReg in0_ptr o hDisj]
        rfl
      refine ⟨?_, ?_, ?_⟩
      · intro j i hcj hjlt hactive
        rcases Nat.eq_or_lt_of_le hcj with rfl | hgt
        · have hstable := ih2
            (taskIndex (P + c * C) tile_size0 i * out0_stride0)
            (fun j' i' hcj' _ _ heq =>
              absurd (Nat.eq_of_mul_eq_mul_right hStride heq)
                (gsTask_ne_of_ne P C hC (show c ≠ j' by omega) i i'))
          rw [hstable, hact i hactive]
          rfl
        · rw [ih1 j i (by omega) hjlt hactive]
          simp only [reluSpec]
          rw [hmemIn]
      · intro o ho
        rw [ih2 o (fun j' i' hcj' hjlt' hact' =>
          ho j' i' (by omega) hjlt' hact')]
        rw [hpresOut o (fun i' hact' => ho c i' (Nat.le_refl c) hlt hact')]
        rfl
      · intro o
        rw [ih3 o]
        exact hmemIn o

private theorem stepStmts_singleton (st : Stmt) (s : BlockState) :
    stepStmts [st] s = stepStmt st s := by
  cases h : stepStmt st s with
  | none =>
      unfold stepStmts
      rw [h]
  | some s1 =>
      rw [stepStmts.cons_some h, stepStmts.nil]

set_option maxHeartbeats 1000000 in
/-- Per-(iteration, lane) exec readback for the grid-stride branch: after the
full grid-stride loop, every active lane of every tile `pid + j·num_ctas`
(`j < tiles_per_cta`) holds the genuine ReLU value. -/
theorem relu_grid_stride_exec_correct
    (in0_ptr out0_ptr : RegionName)
    (in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta tile_size0 : Nat)
    (s : BlockState) (hStride : 0 < out0_stride0) (hDisj : in0_ptr ≠ out0_ptr)
    (hGrid : 0 < s.numPids 0)
    (s' : BlockState)
    (hExec : exec (relu_forward_kernel_rank_1_grid_stride_surface in0_ptr
        out0_ptr in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta
        tile_size0) s = some s') :
    ∀ (j : Nat) (i : Fin tile_size0), j < tiles_per_cta →
      taskIndex (s.pids 0 + j * s.numPids 0) tile_size0 i < s0 →
      s'.readMem out0_ptr
          (taskIndex (s.pids 0 + j * s.numPids 0) tile_size0 i * out0_stride0)
        = reluSpec s in0_ptr in0_stride0
            (taskIndex (s.pids 0 + j * s.numPids 0) tile_size0 i) := by
  rw [show exec (relu_forward_kernel_rank_1_grid_stride_surface in0_ptr
        out0_ptr in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta
        tile_size0).toAlgKernel s
      = stepStmts ((relu_forward_kernel_rank_1_grid_stride_surface in0_ptr
        out0_ptr in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta
        tile_size0).toAlgKernel).body s from rfl,
    gridStride_body_eq] at hExec
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 0 s))]
    at hExec
  rw [stepStmts.cons_some (numTiles0_step s0 tile_size0 _)] at hExec
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_numPrograms 0 _))]
    at hExec
  rw [stepStmts_singleton, stepForRangeAux.forRangeDyn_unfold] at hExec
  simp only [evalOp_constNat, Option.bind_some] at hExec
  obtain ⟨h1, _, _⟩ := gs_loop_readback in0_ptr out0_ptr in0_stride0
    out0_stride0 s0 tile_size0 hStride hDisj (s.pids 0) (s.numPids 0) hGrid
    tiles_per_cta tiles_per_cta 0 (by omega) _
    (by simp) (by simp) s' hExec
  intro j i hj hact
  rw [h1 j i (Nat.zero_le j) hj hact]
  rfl

/-- Compute-facing correctness for the `one_tile_per_cta = false` grid-stride
branch: one `Realizes` over all `(iteration, lane)` pairs. -/
theorem relu_grid_stride_compute_correct
    (in0_ptr out0_ptr : RegionName)
    (in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta tile_size0 : Nat)
    (s : BlockState) (hStride : 0 < out0_stride0) (hDisj : in0_ptr ≠ out0_ptr)
    (hGrid : 0 < s.numPids 0) :
    ComputeCorrect.Realizes
      (kernel := relu_forward_kernel_rank_1_grid_stride_surface in0_ptr
        out0_ptr in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta tile_size0)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun p : Fin tiles_per_cta × Fin tile_size0 =>
          taskIndex (s.pids 0 + p.1.val * s.numPids 0) tile_size0 p.2 < s0)
        (fun p => (out0_ptr,
          taskIndex (s.pids 0 + p.1.val * s.numPids 0) tile_size0 p.2
            * out0_stride0)))
      (expected := fun p =>
        reluSpec s in0_ptr in0_stride0
          (taskIndex (s.pids 0 + p.1.val * s.numPids 0) tile_size0 p.2)) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [relu_forward_kernel_rank_1_grid_stride_surface,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0' s' hExec hs0
  subst s0'
  intro p hActive
  have h := relu_grid_stride_exec_correct in0_ptr out0_ptr in0_stride0
    out0_stride0 s0 num_tasks tiles_per_cta tile_size0 s hStride hDisj hGrid
    s' hExec p.1.val p.2 p.1.isLt hActive
  simpa using h

/-! ### ════════ ★ MAIN THEOREM ★ ════════ -/

/-- **Dimension-general** correctness summary for `relu_strided_buffer.py`'s
`relu_forward_kernel_rank_1`, against the **genuine closed form**
`reluSpec = relu(in0[t·in0_stride0]) = max 0 (…)` — a pure function of INPUT
memory, never a read-back of the kernel's own output — for arbitrary `s0`,
strides, `tile_size0` and `tiles_per_cta`. It packages, for **both**
`one_tile_per_cta` constexpr branches:

* both surfaces lower to the algorithm layer;
* the `one_tile_per_cta = true` branch: every active lane
  (`pid·tile_size0 + i < s0`) of the program's single tile holds
  `relu(in0[t·in0_stride0])` at `out0[t·out0_stride0]`;
* the `one_tile_per_cta = false` grid-stride branch: for **every** iteration
  `j < tiles_per_cta` and lane `i` of tile `pid + j·num_ctas`, the active
  cells hold the genuine ReLU value — the full multi-iteration loop is
  verified end-to-end (loop invariant `gs_loop_readback`).

Honest side-conditions: `0 < out0_stride0` (store-footprint injectivity —
torch strides of a non-degenerate rank-1 buffer are ≥ 1), and for the
grid-stride branch `in0_ptr ≠ out0_ptr` (later iterations load after earlier
stores) and `0 < s.numPids 0` (a launched grid has at least one program). -/
theorem relu_strided_buffer_output_summary_general
    (in0_ptr out0_ptr : RegionName)
    (in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta tile_size0 : Nat)
    (s : BlockState)
    (hStride : 0 < out0_stride0)
    (hDisj : in0_ptr ≠ out0_ptr)
    (hGrid : 0 < s.numPids 0) :
    -- (1) both branch surfaces lower to the algorithm layer
    (∃ alg, (relu_forward_kernel_rank_1_one_tile_surface in0_ptr out0_ptr
      in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta
      tile_size0).toAlgorithm? = Except.ok alg) ∧
    (∃ alg, (relu_forward_kernel_rank_1_grid_stride_surface in0_ptr out0_ptr
      in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta
      tile_size0).toAlgorithm? = Except.ok alg) ∧
    -- (2) one_tile_per_cta = true: genuine elementwise ReLU
    ComputeCorrect.Realizes
      (kernel := relu_forward_kernel_rank_1_one_tile_surface in0_ptr out0_ptr
        in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta tile_size0)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin tile_size0 => taskIndex (s.pids 0) tile_size0 i < s0)
        (fun i => (out0_ptr, taskIndex (s.pids 0) tile_size0 i * out0_stride0)))
      (expected := fun i =>
        reluSpec s in0_ptr in0_stride0 (taskIndex (s.pids 0) tile_size0 i)) ∧
    -- (3) one_tile_per_cta = false: genuine elementwise ReLU across the
    --     whole grid-stride loop
    ComputeCorrect.Realizes
      (kernel := relu_forward_kernel_rank_1_grid_stride_surface in0_ptr
        out0_ptr in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta tile_size0)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun p : Fin tiles_per_cta × Fin tile_size0 =>
          taskIndex (s.pids 0 + p.1.val * s.numPids 0) tile_size0 p.2 < s0)
        (fun p => (out0_ptr,
          taskIndex (s.pids 0 + p.1.val * s.numPids 0) tile_size0 p.2
            * out0_stride0)))
      (expected := fun p =>
        reluSpec s in0_ptr in0_stride0
          (taskIndex (s.pids 0 + p.1.val * s.numPids 0) tile_size0 p.2)) :=
  ⟨relu_one_tile_surface_toAlgorithm_supported in0_ptr out0_ptr in0_stride0
      out0_stride0 s0 num_tasks tiles_per_cta tile_size0,
    relu_grid_stride_surface_toAlgorithm_supported in0_ptr out0_ptr in0_stride0
      out0_stride0 s0 num_tasks tiles_per_cta tile_size0,
    relu_one_tile_compute_correct in0_ptr out0_ptr in0_stride0 out0_stride0
      s0 num_tasks tiles_per_cta tile_size0 s hStride,
    relu_grid_stride_compute_correct in0_ptr out0_ptr in0_stride0 out0_stride0
      s0 num_tasks tiles_per_cta tile_size0 s hStride hDisj hGrid⟩

end Correct

end VeriTile.Bench.TritonBenchG.ReluStridedBuffer
