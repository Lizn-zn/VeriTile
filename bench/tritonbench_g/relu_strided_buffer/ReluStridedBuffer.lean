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

**Both constexpr branches have an IO-surface headline.** The grid-stride branch
needs the grid width in its *signature*, not merely in its proof: its windows
stride by `num_ctas`, and a skin pinning only `pids` provably cannot state such
a contract. It is therefore stated on `StreamGridStrideEmitMasked2DKernelIO₁`,
whose `⊨[R]` pins `s₀.numPids 0` alongside the pids — this port is that skin's
first consumer.

## Proof architecture

```
relu_strided_buffer_grid_stride_io_correctnessR   ← TOP THEOREM (⊨[R], the
  ├─ relu_grid_stride_flattenOk                     grid-stride branch on the
  ├─ relu_grid_stride_traceSafeR                    grid-width-aware skin)
  │    └─ perTile_traceSafeR                    R-side per-tile safety walk
  ├─ relu_grid_stride_execR_eq_exec             cast-free: execR R = exec
  │    └─ rsForDyn_castFree → rsLoopBody_castFree → rsPerTile_stmt_castFree
  ├─ relu_grid_stride_run_exact                 termination + cell frame
  │    └─ gs_loop_cells                         per-cell loop invariant
  └─ relu_grid_stride_exec_correct              per-(step, lane) values (below)

relu_strided_buffer_one_tile_io_correctness       ⊨, the one-tile branch
  ├─ relu_one_tile_flattenOk                    inside the flat-memory bridge
  ├─ relu_one_tile_traceSafe                    block-pointer safety walk
  │    └─ perTile_traceSafe                     shared per-tile body
  └─ relu_one_tile_region_run                   region-model run + cell frame
       └─ perTile_steps                         (shared engine, below)

relu_strided_buffer_output_summary_general       both branches, per write map
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
open scoped VeriTile.Triton.MaskedTileKernelIO₁

set_option linter.unusedSimpArgs false
set_option linter.unusedVariables false

/-! **★ Main theorem:** `relu_strided_buffer_one_tile_io_correctness` — the
`one_tile_per_cta = true` branch on the tile-indexed IO surface (`⊨`), which
additionally pins the flat-memory placement. `relu_strided_buffer_output_summary_general`
below it keeps the per-write-map summary of **both** constexpr branches. Both
are dimension-general in `s0`, `in0_stride0`, `out0_stride0`, `tile_size0`,
`tiles_per_cta`; the Python benchmark shapes are instantiations. -/

/-! # ══════════ CORRECT — genuine / dimension-general (review this) ══════════ -/

section Correct_without_Rounding

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

/-- **Cell-level** frame of a masked scatter: a cell the fold never writes —
either because it lives in a different region, or because no active lane
targets its offset — keeps its `mem` value verbatim (not merely its decoded
`readMem` value). `bench` files are standalone, so this induction is a private
copy rather than an import. -/
private theorem foldl_writeMem_frame {α : Type} {region : RegionName}
    (offsetFn : α → Nat) (valueFn : α → ℝ) (P : α → Prop) [DecidablePred P]
    (R : RegionName) (off : Nat) :
    ∀ l : List α, (R ≠ region ∨ ∀ k ∈ l, P k → offsetFn k ≠ off) →
      ∀ s : BlockState,
        ((l.foldl (fun acc k =>
            if P k then acc.writeMem region (offsetFn k) (valueFn k) else acc)
            s).mem R off) = s.mem R off := by
  intro l
  induction l with
  | nil => intro _ s; rfl
  | cons hd tl ih =>
      intro hc s
      have htl : R ≠ region ∨ ∀ k ∈ tl, P k → offsetFn k ≠ off := by
        rcases hc with h | h
        · exact Or.inl h
        · exact Or.inr fun k hk => h k (List.mem_cons_of_mem hd hk)
      rw [List.foldl_cons, ih htl]
      by_cases hP : P hd
      · rw [if_pos hP, BlockState.writeMem_mem, if_neg ?_]
        rintro ⟨h1, h2⟩
        rcases hc with h | h
        · exact h h1
        · exact h hd List.mem_cons_self hP h2.symm
      · rw [if_neg hP]

/-! ## One tile of work — shared by both branches

`perTile_steps` executes the seven shared statements (`tile_id0` bookkeeping →
block-ptr load → `relu_forward` → block-ptr store) from any state whose
`tile_id` register holds `T`, and characterizes the result: active lanes of
tile `T` receive the genuine ReLU value, everything else — other `out0` cells,
other regions, all registers except the six per-tile scratch registers, and
(cell-level, for the flat-memory IO face) every `mem` cell off the written
window — is untouched. -/

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
          t'.regs dtype shape name = t.regs dtype shape name)
      ∧ (∀ (R : RegionName) (o : Nat),
          (R ≠ out0_ptr ∨ ∀ i : Fin tile_size0, taskIndex T tile_size0 i < s0 →
            o ≠ taskIndex T tile_size0 i * out0_stride0) →
          t'.mem R o = t.mem R o) := by
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
  refine ⟨_, hstep, ?_, ?_, ?_, ?_, ?_⟩
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
  · -- cell-level frame off the written window (the flat-memory IO face)
    intro R o hcond
    rw [foldl_writeMem_frame
      (region := out0_ptr)
      (fun idx : TileIndex [tile_size0] =>
        (T * tile_size0 + idx.1.val) * out0_stride0)
      (fun idx : TileIndex [tile_size0] => (outT.data idx).unbotD 0)
      (fun idx : TileIndex [tile_size0] => T * tile_size0 + idx.1.val < s0)
      R o (TileShape.allIndices [tile_size0]) ?_]
    · simp [ht6, ht5, ht4, ht3, ht2, ht1]
    · rcases hcond with h | h
      · exact Or.inl h
      · exact Or.inr fun idx _ hidx =>
          Ne.symm (h idx.1 (by simpa [taskIndex] using hidx))

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
    ComputeCorrect.Realizes_without_Rounding
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
      obtain ⟨t', hstepBody, hact, hpresOut, hpresReg, hregs, _⟩ :=
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
branch: one `Realizes_without_Rounding` over all `(iteration, lane)` pairs. -/
theorem relu_grid_stride_compute_correct
    (in0_ptr out0_ptr : RegionName)
    (in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta tile_size0 : Nat)
    (s : BlockState) (hStride : 0 < out0_stride0) (hDisj : in0_ptr ≠ out0_ptr)
    (hGrid : 0 < s.numPids 0) :
    ComputeCorrect.Realizes_without_Rounding
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

/-! ### ════════ Per-write-map summary, both constexpr branches ════════ -/

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
specification relu_strided_buffer_output_summary_general
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
    ComputeCorrect.Realizes_without_Rounding
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
    ComputeCorrect.Realizes_without_Rounding
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

end Correct_without_Rounding

/-! # ════════ `⊨` IO face — the flat-memory placement ════════

The summary above is stated per *declared write map*. This section restates the
`one_tile_per_cta = true` branch on the audit-once IO surface
`MaskedTileKernelIO₁.Implements` (`⊨`), which additionally pins the **flat
memory** placement: for every disjoint placement of the two buffers, every
program id whose active lanes are in bounds, and every launch state whose input
window holds `xs` at the active lanes, the translated pointer kernel terminates,
every active output lane holds `relu (xs i)`, and every other memory **cell** is
unchanged.

This is the first port on the **tile-indexed** skin. The contiguous
`MaskedKernelIO₁` cannot express this footprint at all: the lane addresses are
`t·in0_stride0` / `t·out0_stride0` for a task index `t`, i.e. two *different*
strides on the two buffers, and neither is forced to be `1`. `MaskedTileKernelIO₁`
takes the address functions verbatim.

The spec function is the pure `TiledActivation.relu` oracle applied to the
loaded lane value — on this surface the loads are quantified over as `xs`, so
the statement is a closed form in the *values*, not in memory. -/

section IOFace

/-- The one-tile branch sits inside the flat-memory bridge's covered fragment
(block pointers included). -/
theorem relu_one_tile_flattenOk
    (in0_ptr out0_ptr : RegionName)
    (in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta tile_size0 : Nat) :
    ((relu_forward_kernel_rank_1_one_tile_surface in0_ptr out0_ptr
      in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta
      tile_size0).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [relu_forward_kernel_rank_1_one_tile_surface, ComputeKernel.toAlgKernel,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk]

/-- Inversion of a successful `assign` step: it fixes the assigned value and
the successor state. The exact-semantics twin of `stepStmtR_assign_inv`. -/
private theorem stepStmt_assign_inv' {dtype : TileDType} {shape : TileShape}
    {name : RegName} {e : Op dtype shape} {s s' : BlockState}
    (h : stepStmt (.assign dtype shape name e) s = some s') :
    ∃ v, evalOp e s = some v ∧ s' = s.setReg name dtype shape v := by
  cases hv : evalOp e s with
  | none => simp [stepStmt, hv] at h
  | some v =>
      refine ⟨v, rfl, ?_⟩
      rw [stepStmt_assign_eq_some hv] at h
      exact (Option.some.inj h).symm

/-- Per-execution safety walk for the shared per-tile body: the two
`tl.make_block_ptr` assigns and the three register-only assigns impose nothing;
the block-pointer load's and the block-pointer store's active lanes are exactly
the `boundary_check=(0,)` guard `T·tile_size0 + i < s0`, in bounds by the read /
write window bounds. -/
theorem perTile_traceSafe
    (in0_ptr out0_ptr : RegionName)
    (in0_stride0 out0_stride0 s0 tile_size0 : Nat)
    (bounds : RegionBounds) (T : Nat) (t : BlockState) (V : Tile .nat [])
    (hT : t.regs .nat [] "tile_id" = some V)
    (hV : V.data PUnit.unit = T)
    (hin : ∀ i : Fin tile_size0, T * tile_size0 + i.val < s0 →
      (T * tile_size0 + i.val) * in0_stride0 < bounds in0_ptr)
    (hout : ∀ i : Fin tile_size0, T * tile_size0 + i.val < s0 →
      (T * tile_size0 + i.val) * out0_stride0 < bounds out0_ptr) :
    Stmt.TraceSafeList bounds
      (perTileStmts in0_ptr out0_ptr in0_stride0 out0_stride0 s0 tile_size0)
      t := by
  set offV : Tile .nat [] :=
    Tile.bop (NumericDType.mul .nat) Broadcast.nil V (Tile.scalar tile_size0)
    with hoffV
  have hoffData : offV.data PUnit.unit = T * tile_size0 := by
    rw [hoffV, Tile.bop_data]
    show NumericDType.mul .nat (V.data PUnit.unit) tile_size0 = T * tile_size0
    rw [hV]
    rfl
  simp only [perTileStmts]
  -- `tile_id0 = tile_id`
  refine Stmt.TraceSafeList.cons_intro
    (by simp only [Stmt.TraceSafe, Op.SafeAt]) (fun t1 ht1 => ?_)
  rw [stepStmt_assign_eq_some
    (show evalOp (Op.ref .nat [] "tile_id") t = some V by simp [hT])] at ht1
  obtain rfl := Option.some.inj ht1
  -- `offset0 = tile_id0 * tile_size0`
  refine Stmt.TraceSafeList.cons_intro
    (by simp only [Stmt.TraceSafe, Op.SafeAt, and_self]) (fun t2 ht2 => ?_)
  rw [stepStmt_assign_eq_some
    (show evalOp (Op.mul NumericDType.nat Broadcast.nil
        (Op.ref .nat [] "tile_id0") (Op.constNat tile_size0))
        (t.setReg "tile_id0" .nat [] V) = some offV by
      simp [hoffV])] at ht2
  obtain rfl := Option.some.inj ht2
  -- `in0_bptr = tl.make_block_ptr(...)`
  refine Stmt.TraceSafeList.cons_intro
    (by simp only [Stmt.TraceSafe, Op.SafeAt, List.mem_cons, List.not_mem_nil,
      or_false, forall_eq, and_self]) (fun t3 ht3 => ?_)
  rw [stepStmt_assign_eq_some
    (makeBlockPtr_1d_eval in0_ptr s0 in0_stride0 tile_size0 "offset0"
      ((t.setReg "tile_id0" .nat [] V).setReg "offset0" .nat [] offV) offV
      (T * tile_size0) (by simp) hoffData)] at ht3
  obtain rfl := Option.some.inj ht3
  -- the boundary-checked block-pointer load
  refine Stmt.TraceSafeList.cons_intro ?_ (fun t4 ht4 => ?_)
  · simp only [Stmt.TraceSafe, Op.SafeAt, MaskOpt.Active, Op.MemorySafe,
      MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe]
    refine ⟨trivial, trivial, trivial, ?_⟩
    intro ptrs hptrs idx _
    rw [evalOp_ref, BlockState.setReg_same] at hptrs
    obtain rfl := Option.some.inj hptrs
    intro hib
    simp only [TileShape.indexToList, BlockPtr.address_1d, Nat.zero_add]
    refine hin idx.1 ?_
    simpa [TileShape.indexToList] using hib
  · obtain ⟨v4, -, rfl⟩ := stepStmt_assign_inv' ht4
    -- `out0 = tl.where(in0 > 0, in0, 0)`
    refine Stmt.TraceSafeList.cons_intro
      (by simp only [Stmt.TraceSafe, Op.SafeAt, and_self]) (fun t5 ht5 => ?_)
    obtain ⟨v5, -, rfl⟩ := stepStmt_assign_inv' ht5
    -- `out0_bptr = tl.make_block_ptr(...)`
    refine Stmt.TraceSafeList.cons_intro
      (by simp only [Stmt.TraceSafe, Op.SafeAt, List.mem_cons, List.not_mem_nil,
        or_false, forall_eq, and_self]) (fun t6 ht6 => ?_)
    rw [stepStmt_assign_eq_some
      (makeBlockPtr_1d_eval out0_ptr s0 out0_stride0 tile_size0 "offset0" _ offV
        (T * tile_size0) (by simp) hoffData)] at ht6
    obtain rfl := Option.some.inj ht6
    -- the boundary-checked block-pointer store
    refine Stmt.TraceSafeList.cons_intro ?_
      (fun _ _ => Stmt.TraceSafeList.nil_intro)
    simp only [Stmt.TraceSafe, Op.SafeAt, MaskOpt.SafeAt, MaskOpt.Active,
      MaskOpt.MemorySafe, Op.MemorySafe, MemAccess.SafeAt, MemAccess.MemorySafe,
      memAccessMemorySafe, MemAccess.ActiveAddressSafe,
      memAccessActiveAddressSafe]
    refine ⟨trivial, trivial, trivial, trivial, ?_⟩
    intro ptrs hptrs idx _
    rw [evalOp_ref, BlockState.setReg_same] at hptrs
    obtain rfl := Option.some.inj hptrs
    intro hib
    simp only [TileShape.indexToList, BlockPtr.address_1d, Nat.zero_add]
    refine hout idx.1 ?_
    simpa [TileShape.indexToList] using hib

/-- Per-execution safety walk for the whole one-tile kernel: the three prologue
assigns impose nothing, and the per-tile body is `perTile_traceSafe` at
`T = pid`. -/
theorem relu_one_tile_traceSafe
    (in0_ptr out0_ptr : RegionName)
    (in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta tile_size0 : Nat)
    (bounds : RegionBounds) (s : BlockState)
    (hin : ∀ i : Fin tile_size0, taskIndex (s.pids 0) tile_size0 i < s0 →
      taskIndex (s.pids 0) tile_size0 i * in0_stride0 < bounds in0_ptr)
    (hout : ∀ i : Fin tile_size0, taskIndex (s.pids 0) tile_size0 i < s0 →
      taskIndex (s.pids 0) tile_size0 i * out0_stride0 < bounds out0_ptr) :
    ((relu_forward_kernel_rank_1_one_tile_surface in0_ptr out0_ptr
        in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta
        tile_size0).toAlgKernel).TraceSafe bounds s := by
  unfold Kernel.TraceSafe
  rw [oneTile_body_eq]
  refine Stmt.TraceSafeList.cons_intro
    (by simp only [Stmt.TraceSafe, Op.SafeAt]) (fun t1 ht1 => ?_)
  rw [stepStmt_assign_eq_some (evalOp_programId 0 s)] at ht1
  obtain rfl := Option.some.inj ht1
  refine Stmt.TraceSafeList.cons_intro
    (by simp only [numTiles0Stmt, Stmt.TraceSafe, Op.SafeAt, and_self])
    (fun t2 ht2 => ?_)
  rw [numTiles0_step s0 tile_size0 _] at ht2
  obtain rfl := Option.some.inj ht2
  refine Stmt.TraceSafeList.cons_intro
    (by simp only [Stmt.TraceSafe, Op.SafeAt]) (fun t3 ht3 => ?_)
  rw [stepStmt_assign_eq_some
    (show evalOp (Op.ref .nat [] "pid")
        ((s.setReg "pid" .nat [] (Tile.scalar (s.pids 0))).setReg
          "num_tiles0" .nat [] (numTiles0Val s0 tile_size0))
      = some (Tile.scalar (s.pids 0)) by simp)] at ht3
  obtain rfl := Option.some.inj ht3
  refine perTile_traceSafe in0_ptr out0_ptr in0_stride0 out0_stride0 s0
    tile_size0 bounds (s.pids 0) _ (Tile.scalar (s.pids 0))
    (BlockState.setReg_same _ _ _ _ _) rfl
    (fun i hi => hin i hi) (fun i hi => hout i hi)

/-- Region-model run of the one-tile branch, in the shape
`MaskedTileKernelIO₁.Implements.intro` consumes: termination, the per-lane
readback against the pinned load values, and the **cell-level** frame. -/
theorem relu_one_tile_region_run
    (in0_ptr out0_ptr : RegionName)
    (in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta tile_size0 : Nat)
    (hStride : 0 < out0_stride0)
    (s₀ : BlockState) (xs : TileIndex [tile_size0] → ℝ)
    (hx : ∀ i : TileIndex [tile_size0],
      taskIndex (s₀.pids 0) tile_size0 i.1 < s0 →
      s₀.readMem in0_ptr (taskIndex (s₀.pids 0) tile_size0 i.1 * in0_stride0)
        = xs i) :
    ∃ s1, exec ((relu_forward_kernel_rank_1_one_tile_surface in0_ptr out0_ptr
        in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta
        tile_size0).toAlgKernel) s₀ = some s1
      ∧ (∀ i : TileIndex [tile_size0],
          taskIndex (s₀.pids 0) tile_size0 i.1 < s0 →
          s1.readMem out0_ptr
              (taskIndex (s₀.pids 0) tile_size0 i.1 * out0_stride0)
            = TiledActivation.relu (xs i))
      ∧ (∀ (r : RegionName) (o : Nat),
          (r ≠ out0_ptr ∨ ∀ i : TileIndex [tile_size0],
            taskIndex (s₀.pids 0) tile_size0 i.1 < s0 →
            o ≠ taskIndex (s₀.pids 0) tile_size0 i.1 * out0_stride0) →
          s1.mem r o = s₀.mem r o) := by
  set sp3 := ((s₀.setReg "pid" .nat [] (Tile.scalar (s₀.pids 0))).setReg
      "num_tiles0" .nat [] (numTiles0Val s0 tile_size0)).setReg
      "tile_id" .nat [] (Tile.scalar (s₀.pids 0)) with hsp3
  obtain ⟨t', hstep, hact, _, _, _, hcell⟩ :=
    perTile_steps in0_ptr out0_ptr in0_stride0 out0_stride0 s0 tile_size0
      hStride (s₀.pids 0) sp3 (Tile.scalar (s₀.pids 0)) (by simp [hsp3]) rfl
  have hexec : exec ((relu_forward_kernel_rank_1_one_tile_surface in0_ptr
      out0_ptr in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta
      tile_size0).toAlgKernel) s₀ = some t' := by
    rw [show exec ((relu_forward_kernel_rank_1_one_tile_surface in0_ptr out0_ptr
          in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta
          tile_size0).toAlgKernel) s₀
        = stepStmts ((relu_forward_kernel_rank_1_one_tile_surface in0_ptr
          out0_ptr in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta
          tile_size0).toAlgKernel).body s₀ from rfl,
      oneTile_body_eq]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 0 s₀))]
    rw [stepStmts.cons_some (numTiles0_step s0 tile_size0 _)]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (show evalOp (Op.ref .nat [] "pid")
          ((s₀.setReg "pid" .nat [] (Tile.scalar (s₀.pids 0))).setReg
            "num_tiles0" .nat [] (numTiles0Val s0 tile_size0))
        = some (Tile.scalar (s₀.pids 0)) by simp))]
    exact hstep
  refine ⟨t', hexec, ?_, ?_⟩
  · intro i hi
    rw [hact i.1 hi]
    simp only [reluSpec, hsp3, BlockState.setReg_readMem]
    rw [hx i hi]
  · intro r o hcond
    rw [hcell r o ?_]
    · simp [hsp3]
    · rcases hcond with h | h
      · exact Or.inl h
      · exact Or.inr fun i hi => h (i, PUnit.unit) hi

/-- IO signature of the `one_tile_per_cta = true` branch on the **tile-indexed**
surface: lane `i` of program `pid` covers task `t = pid·tile_size0 + i`, reads
`in0_ptr` at `t·in0_stride0`, writes `out0_ptr` at `t·out0_stride0`, and is
active exactly on the `boundary_check=(0,)` guard `t < s0`. -/
def reluOneTileIO (in0_ptr out0_ptr : RegionName)
    (in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta tile_size0 : Nat) :
    MaskedTileKernelIO₁ where
  kernel := relu_forward_kernel_rank_1_one_tile_surface in0_ptr out0_ptr
    in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta tile_size0
  inp := in0_ptr
  out := out0_ptr
  shape := [tile_size0]
  read := fun pid i => taskIndex pid tile_size0 i.1 * in0_stride0
  write := fun pid i => taskIndex pid tile_size0 i.1 * out0_stride0
  mask := fun pid i => taskIndex pid tile_size0 i.1 < s0

/-! ### ════════ ★ MAIN THEOREM ★ ════════ -/

/-- **The headline on the IO surface** for `relu_strided_buffer.py`'s
`relu_forward_kernel_rank_1`, `one_tile_per_cta = true` branch: for every
disjoint flat placement of the two buffers, every program id whose active lanes
are in bounds, and every launch state whose input window holds `xs`, the
translated pointer kernel terminates, every active lane of the output tile holds
`relu (xs i) = max 0 (xs i)`, and every other memory cell is unchanged.

Dimension-general in `s0`, both strides, `tile_size0`; the only side-condition
is `0 < out0_stride0` (store-footprint injectivity — a torch stride of a
non-degenerate rank-1 buffer is ≥ 1). The strided, two-stride footprint is
carried verbatim by `MaskedTileKernelIO₁`'s address functions. -/
specification relu_strided_buffer_one_tile_io_correctness
    (in0_ptr out0_ptr : RegionName)
    (in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta tile_size0 : Nat)
    (hStride : 0 < out0_stride0) :
    reluOneTileIO in0_ptr out0_ptr in0_stride0 out0_stride0 s0 num_tasks
        tiles_per_cta tile_size0
      ⊨ fun _pid xs i => TiledActivation.relu (xs i) := by
  refine MaskedTileKernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact relu_one_tile_flattenOk in0_ptr out0_ptr in0_stride0 out0_stride0 s0
      num_tasks tiles_per_cta tile_size0
  · intro bounds s h1 h2
    exact relu_one_tile_traceSafe in0_ptr out0_ptr in0_stride0 out0_stride0 s0
      num_tasks tiles_per_cta tile_size0 bounds s
      (fun i hi => h1 (i, PUnit.unit) hi) (fun i hi => h2 (i, PUnit.unit) hi)
  · intro s₀ xs hin
    exact relu_one_tile_region_run in0_ptr out0_ptr in0_stride0 out0_stride0 s0
      num_tasks tiles_per_cta tile_size0 hStride s₀ xs hin

open scoped VeriTile.Triton.StreamGridStrideEmitMasked2DKernelIO₁

/-! # ════════ `⊨[R]` IO face — the grid-stride branch ════════

The section above states the `one_tile_per_cta = true` branch. This one states
the `one_tile_per_cta = false` **grid-stride** branch, which needs a wider
signature than any pid-only skin can provide: its windows stride by
`num_ctas = tl.num_programs(0)`, and a skin that pins only `pids` provably
cannot state such a contract (instantiate it at two launch states with equal
inputs and `numPids 0 = 1` vs `2`: one cell must both hold the readback value
and be untouched by the frame). See the genre note on
`StreamGridStrideEmitMasked2DKernelIO₁` in `Memory/KernelSpec.lean`; this is
that skin's first consumer.

The streaming genre's only surface is `⊨[R]`. With the default
`outDType := .real` every per-step store is exact under `execR R`, so the
statement below **is** the exact grid-stride streaming contract — `R` is
threaded through the whole kernel but never rounds anything, which is faithful
here precisely because ReLU is a *selection* (`tl.where(x > 0, x, 0)` returns a
loaded value or the literal `0`); no arithmetic manufactures a real that would
have to be quantized. -/

/-- The grid-stride surface sits inside the flat-memory bridge's covered
fragment (`FlattenOk` recurses into the `forRangeDyn` body). -/
theorem relu_grid_stride_flattenOk
    (in0_ptr out0_ptr : RegionName)
    (in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta tile_size0 : Nat) :
    ((relu_forward_kernel_rank_1_grid_stride_surface in0_ptr out0_ptr
      in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta
      tile_size0).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  rw [gridStride_body_eq]
  simp [numTiles0Stmt, perTileStmts, StmtList.FlattenOk, Stmt.FlattenOk,
    Op.FlattenOk]

/-! ### Cast-freeness: `execR R` collapses onto the exact stepper

The kernel's only dtype casts are the source's `.to(...element_ty)` pair, which
the DSL erases to `.real`, and every store is `.real`-typed
(`writeMemTypedR R .real` is definitionally the exact write). So `execR R` and
`exec` step identically — statement by statement, and through the grid-stride
loop via `stepForRangeAuxR_castFree`. -/

/-- An assign steps identically under `stepStmtR R` once its operand does. -/
private theorem rsAssign_castFree {R : RoundingModel} {d : TileDType}
    {sh : TileShape} {nm : RegName} (e : Op d sh) (u : BlockState)
    (h : evalOpR R e u = evalOp e u) :
    stepStmtR R (Stmt.assign d sh nm e) u = stepStmt (Stmt.assign d sh nm e) u := by
  simp only [stepStmtR, stepStmt, h]

/-- Statement lists whose members all step identically step identically. -/
private theorem rsStepList_castFree {R : RoundingModel} :
    ∀ (l : List Stmt),
      (∀ st ∈ l, ∀ u : BlockState, stepStmtR R st u = stepStmt st u) →
      ∀ u : BlockState, stepStmtsR R l u = stepStmts l u
  | [], _, u => by simp only [stepStmtsR, stepStmts]
  | st :: rest, h, u => by
      cases hv : stepStmt st u with
      | none => simp only [stepStmtsR, stepStmts, h st List.mem_cons_self u, hv]
      | some u' =>
          simp only [stepStmtsR, stepStmts, h st List.mem_cons_self u, hv]
          exact rsStepList_castFree rest
            (fun s' hs' => h s' (List.mem_cons_of_mem _ hs')) u'

/-- The two per-tile `tl.make_block_ptr` operands evaluate identically. A
dedicated lemma: unfolding `evalOpR`/`evalOp` inside a broad `simp` at symbolic
dimensions blows up `whnf`. -/
private theorem rsEvalR_blockPtr (R : RoundingModel) (Reg : RegionName)
    (s0 tile_size0 stride : Nat) (u : BlockState) :
    evalOpR R (Op.makeBlockPtrDynOffsets Reg (Op.constNat 0) [s0] [tile_size0]
        [stride] [Op.ref .nat [] "offset0"]) u
      = evalOp (Op.makeBlockPtrDynOffsets Reg (Op.constNat 0) [s0] [tile_size0]
        [stride] [Op.ref .nat [] "offset0"]) u := by
  simp only [evalOpR, evalOp, List.mapM_cons, List.mapM_nil]

/-- The boundary-checked `.real` block-pointer load evaluates identically. -/
private theorem rsEvalR_load (R : RoundingModel) (tile_size0 : Nat)
    (u : BlockState) :
    evalOpR R (Op.load .real
        (MemAccess.blockPtr (Op.ref .blockPtr [tile_size0] "in0_bptr") [0])
        MaskOpt.none) u
      = evalOp (Op.load .real
        (MemAccess.blockPtr (Op.ref .blockPtr [tile_size0] "in0_bptr") [0])
        MaskOpt.none) u := by
  simp only [evalOpR, evalOp, evalOpR_ref, evalOp_ref]

/-- The inlined `relu_forward` `tl.where` evaluates identically (a selection of
already-loaded values and the literal `0`; nothing to round). -/
private theorem rsEvalR_where (R : RoundingModel) (tile_size0 : Nat)
    (u : BlockState) :
    evalOpR R ((Op.gt ComparableDType.real Broadcast.scalarR
          (Op.ref .real [tile_size0] "in0") (Op.const 0)).where
        (Op.ref .real [tile_size0] "in0")
        ((Op.const 0).broadcast [tile_size0])) u
      = evalOp ((Op.gt ComparableDType.real Broadcast.scalarR
          (Op.ref .real [tile_size0] "in0") (Op.const 0)).where
        (Op.ref .real [tile_size0] "in0")
        ((Op.const 0).broadcast [tile_size0])) u := by
  simp only [evalOpR, evalOp, evalOpR_ref, evalOp_ref]

/-- Every statement of the shared per-tile body is cast-free, unconditionally
in the state — which is what `stepForRangeAuxR_castFree` needs. -/
private theorem rsPerTile_stmt_castFree (R : RoundingModel)
    (in0_ptr out0_ptr : RegionName) (in0_stride0 out0_stride0 s0 tile_size0 : Nat) :
    ∀ st ∈ perTileStmts in0_ptr out0_ptr in0_stride0 out0_stride0 s0 tile_size0,
      ∀ u : BlockState, stepStmtR R st u = stepStmt st u := by
  intro st hst u
  simp only [perTileStmts, List.mem_cons, List.not_mem_nil, or_false] at hst
  rcases hst with rfl | rfl | rfl | rfl | rfl | rfl | rfl
  · exact rsAssign_castFree _ u (by simp only [evalOpR, evalOp])
  · exact rsAssign_castFree _ u (by simp only [evalOpR, evalOp])
  · exact rsAssign_castFree _ u (rsEvalR_blockPtr R in0_ptr s0 tile_size0 in0_stride0 u)
  · exact rsAssign_castFree _ u (rsEvalR_load R tile_size0 u)
  · exact rsAssign_castFree _ u (rsEvalR_where R tile_size0 u)
  · exact rsAssign_castFree _ u (rsEvalR_blockPtr R out0_ptr s0 tile_size0 out0_stride0 u)
  · simp only [stepStmtR, stepStmt, evalOpR_ref, evalOp_ref,
      BlockState.writeMemTypedR]

/-- The grid-stride loop body (the `tile_id` recomputation plus the shared
per-tile body) is cast-free. -/
private theorem rsLoopBody_castFree (R : RoundingModel)
    (in0_ptr out0_ptr : RegionName) (in0_stride0 out0_stride0 s0 tile_size0 : Nat)
    (u : BlockState) :
    stepStmtsR R
        (Stmt.assign .nat [] "tile_id"
            (Op.add NumericDType.nat Broadcast.nil (Op.ref .nat [] "pid")
              (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "j")
                (Op.ref .nat [] "num_ctas")))
          :: perTileStmts in0_ptr out0_ptr in0_stride0 out0_stride0 s0 tile_size0)
        u
      = stepStmts
        (Stmt.assign .nat [] "tile_id"
            (Op.add NumericDType.nat Broadcast.nil (Op.ref .nat [] "pid")
              (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "j")
                (Op.ref .nat [] "num_ctas")))
          :: perTileStmts in0_ptr out0_ptr in0_stride0 out0_stride0 s0 tile_size0)
        u := by
  refine rsStepList_castFree _ ?_ u
  intro st hst v
  rcases List.mem_cons.mp hst with rfl | hrest
  · exact rsAssign_castFree _ v (by simp only [evalOpR.eq_def, evalOp.eq_def])
  · exact rsPerTile_stmt_castFree R in0_ptr out0_ptr in0_stride0 out0_stride0 s0
      tile_size0 st hrest v

/-- The grid-stride `forRangeDyn` steps identically under `stepStmtR R`. -/
private theorem rsForDyn_castFree (R : RoundingModel)
    (in0_ptr out0_ptr : RegionName)
    (in0_stride0 out0_stride0 s0 tiles_per_cta tile_size0 : Nat)
    (u : BlockState) :
    stepStmtR R
        (Stmt.forRangeDyn "j" (Op.constNat 0) (Op.constNat tiles_per_cta)
          (Op.constNat 1)
          (Stmt.assign .nat [] "tile_id"
              (Op.add NumericDType.nat Broadcast.nil (Op.ref .nat [] "pid")
                (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "j")
                  (Op.ref .nat [] "num_ctas")))
            :: perTileStmts in0_ptr out0_ptr in0_stride0 out0_stride0 s0
              tile_size0)) u
      = stepStmt
        (Stmt.forRangeDyn "j" (Op.constNat 0) (Op.constNat tiles_per_cta)
          (Op.constNat 1)
          (Stmt.assign .nat [] "tile_id"
              (Op.add NumericDType.nat Broadcast.nil (Op.ref .nat [] "pid")
                (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "j")
                  (Op.ref .nat [] "num_ctas")))
            :: perTileStmts in0_ptr out0_ptr in0_stride0 out0_stride0 s0
              tile_size0)) u := by
  rw [stepForRangeAux.forRangeDyn_unfold]
  simp only [stepStmtR, evalOpR, evalOp, Option.bind]
  exact stepForRangeAuxR_castFree R _
    (rsLoopBody_castFree R in0_ptr out0_ptr in0_stride0 out0_stride0 s0 tile_size0)
    "j" _ _ _ u

/-- **The whole grid-stride kernel is cast-free**: `execR R` is the exact
`exec`. -/
private theorem relu_grid_stride_execR_eq_exec (R : RoundingModel)
    (in0_ptr out0_ptr : RegionName)
    (in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta tile_size0 : Nat)
    (u : BlockState) :
    execR R ((relu_forward_kernel_rank_1_grid_stride_surface in0_ptr out0_ptr
        in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta
        tile_size0).toAlgKernel) u
      = exec ((relu_forward_kernel_rank_1_grid_stride_surface in0_ptr out0_ptr
        in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta
        tile_size0).toAlgKernel) u := by
  unfold execR exec
  rw [gridStride_body_eq]
  refine rsStepList_castFree _ ?_ u
  intro st hst v
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hst
  rcases hst with rfl | rfl | rfl | rfl
  · exact rsAssign_castFree _ v (by simp only [evalOpR.eq_def, evalOp.eq_def])
  · exact rsAssign_castFree _ v
      (by simp only [numTiles0Stmt, evalOpR.eq_def, evalOp.eq_def])
  · exact rsAssign_castFree _ v (by simp only [evalOpR.eq_def, evalOp.eq_def])
  · exact rsForDyn_castFree R in0_ptr out0_ptr in0_stride0 out0_stride0 s0
      tiles_per_cta tile_size0 v

/-! ### The `R`-side safety walk

`TraceSafeR` is not `TraceSafe` (the address/mask obligations quantify over
`evalOpR`-evaluated pointers), so the walk is redone rather than transported.
Cast-freeness still pays: it identifies the intermediate states, so each step
reuses the exact stack's eval lemmas. -/

set_option maxHeartbeats 1000000 in
/-- `R`-side per-execution safety walk for the shared per-tile body — the
mirror of `perTile_traceSafe`. -/
theorem perTile_traceSafeR (R : RoundingModel)
    (in0_ptr out0_ptr : RegionName)
    (in0_stride0 out0_stride0 s0 tile_size0 : Nat)
    (bounds : RegionBounds) (T : Nat) (t : BlockState) (V : Tile .nat [])
    (hT : t.regs .nat [] "tile_id" = some V)
    (hV : V.data PUnit.unit = T)
    (hin : ∀ i : Fin tile_size0, T * tile_size0 + i.val < s0 →
      (T * tile_size0 + i.val) * in0_stride0 < bounds in0_ptr)
    (hout : ∀ i : Fin tile_size0, T * tile_size0 + i.val < s0 →
      (T * tile_size0 + i.val) * out0_stride0 < bounds out0_ptr) :
    Stmt.TraceSafeListR R bounds
      (perTileStmts in0_ptr out0_ptr in0_stride0 out0_stride0 s0 tile_size0)
      t := by
  set offV : Tile .nat [] :=
    Tile.bop (NumericDType.mul .nat) Broadcast.nil V (Tile.scalar tile_size0)
    with hoffV
  have hoffData : offV.data PUnit.unit = T * tile_size0 := by
    rw [hoffV, Tile.bop_data]
    show NumericDType.mul .nat (V.data PUnit.unit) tile_size0 = T * tile_size0
    rw [hV]
    rfl
  simp only [perTileStmts]
  -- `tile_id0 = tile_id`
  refine Stmt.TraceSafeListR.cons_intro
    (by simp only [Stmt.TraceSafeR, Op.SafeAtR]) (fun t1 ht1 => ?_)
  rw [rsAssign_castFree _ t (by simp only [evalOpR, evalOp]),
    stepStmt_assign_eq_some
      (show evalOp (Op.ref .nat [] "tile_id") t = some V by simp [hT])] at ht1
  obtain rfl := Option.some.inj ht1
  -- `offset0 = tile_id0 * tile_size0`
  refine Stmt.TraceSafeListR.cons_intro
    (by simp only [Stmt.TraceSafeR, Op.SafeAtR, and_self]) (fun t2 ht2 => ?_)
  rw [rsAssign_castFree _ _ (by simp only [evalOpR, evalOp]),
    stepStmt_assign_eq_some
      (show evalOp (Op.mul NumericDType.nat Broadcast.nil
          (Op.ref .nat [] "tile_id0") (Op.constNat tile_size0))
          (t.setReg "tile_id0" .nat [] V) = some offV by
        simp [hoffV])] at ht2
  obtain rfl := Option.some.inj ht2
  -- `in0_bptr = tl.make_block_ptr(...)`
  refine Stmt.TraceSafeListR.cons_intro
    (by simp only [Stmt.TraceSafeR, Op.SafeAtR, List.mem_cons, List.not_mem_nil,
      or_false, forall_eq, and_self]) (fun t3 ht3 => ?_)
  rw [rsAssign_castFree _ _ (rsEvalR_blockPtr R in0_ptr s0 tile_size0 in0_stride0 _),
    stepStmt_assign_eq_some
      (makeBlockPtr_1d_eval in0_ptr s0 in0_stride0 tile_size0 "offset0"
        ((t.setReg "tile_id0" .nat [] V).setReg "offset0" .nat [] offV) offV
        (T * tile_size0) (by simp) hoffData)] at ht3
  obtain rfl := Option.some.inj ht3
  -- the boundary-checked block-pointer load
  refine Stmt.TraceSafeListR.cons_intro ?_ (fun t4 ht4 => ?_)
  · simp only [Stmt.TraceSafeR, Op.SafeAtR, MaskOpt.ActiveR,
      MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
    refine ⟨trivial, trivial, ?_⟩
    intro ptrs hptrs idx _
    rw [evalOpR_ref, BlockState.setReg_same] at hptrs
    obtain rfl := Option.some.inj hptrs
    intro hib
    simp only [TileShape.indexToList, BlockPtr.address_1d, Nat.zero_add]
    refine hin idx.1 ?_
    simpa [TileShape.indexToList] using hib
  · rw [rsAssign_castFree _ _ (rsEvalR_load R tile_size0 _)] at ht4
    obtain ⟨v4, -, rfl⟩ := stepStmt_assign_inv' ht4
    -- `out0 = tl.where(in0 > 0, in0, 0)`
    refine Stmt.TraceSafeListR.cons_intro
      (by simp only [Stmt.TraceSafeR, Op.SafeAtR, and_self]) (fun t5 ht5 => ?_)
    rw [rsAssign_castFree _ _ (rsEvalR_where R tile_size0 _)] at ht5
    obtain ⟨v5, -, rfl⟩ := stepStmt_assign_inv' ht5
    -- `out0_bptr = tl.make_block_ptr(...)`
    refine Stmt.TraceSafeListR.cons_intro
      (by simp only [Stmt.TraceSafeR, Op.SafeAtR, List.mem_cons, List.not_mem_nil,
        or_false, forall_eq, and_self]) (fun t6 ht6 => ?_)
    rw [rsAssign_castFree _ _
        (rsEvalR_blockPtr R out0_ptr s0 tile_size0 out0_stride0 _),
      stepStmt_assign_eq_some
        (makeBlockPtr_1d_eval out0_ptr s0 out0_stride0 tile_size0 "offset0" _ offV
          (T * tile_size0) (by simp) hoffData)] at ht6
    obtain rfl := Option.some.inj ht6
    -- the boundary-checked block-pointer store
    refine Stmt.TraceSafeListR.cons_intro ?_
      (fun _ _ => Stmt.TraceSafeListR.nil_intro)
    simp only [Stmt.TraceSafeR, Op.SafeAtR, MemAccess.SafeAtR, MaskOpt.SafeAtR,
      MaskOpt.ActiveR, MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
    refine ⟨trivial, trivial, trivial, ?_⟩
    intro ptrs hptrs idx _
    rw [evalOpR_ref, BlockState.setReg_same] at hptrs
    obtain rfl := Option.some.inj hptrs
    intro hib
    simp only [TileShape.indexToList, BlockPtr.address_1d, Nat.zero_add]
    refine hout idx.1 ?_
    simpa [TileShape.indexToList] using hib

set_option maxHeartbeats 1000000 in
/-- `R`-side per-execution safety walk for the whole grid-stride kernel: the
three prologue assigns impose nothing, and the grid-stride loop is driven by
`Stmt.forRangeTraceSafeR_inv` on the invariant "`pid` and `num_ctas` still hold
the launch values", whose body step is `perTile_traceSafeR` at the iteration's
grid-stride tile. -/
theorem relu_grid_stride_traceSafeR (R : RoundingModel)
    (in0_ptr out0_ptr : RegionName)
    (in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta tile_size0 : Nat)
    (hStride : 0 < out0_stride0) (bounds : RegionBounds) (s : BlockState)
    (hin : ∀ (j : Fin tiles_per_cta) (i : Fin tile_size0),
      taskIndex (s.pids 0 + j.val * s.numPids 0) tile_size0 i < s0 →
      taskIndex (s.pids 0 + j.val * s.numPids 0) tile_size0 i * in0_stride0
        < bounds in0_ptr)
    (hout : ∀ (j : Fin tiles_per_cta) (i : Fin tile_size0),
      taskIndex (s.pids 0 + j.val * s.numPids 0) tile_size0 i < s0 →
      taskIndex (s.pids 0 + j.val * s.numPids 0) tile_size0 i * out0_stride0
        < bounds out0_ptr) :
    ((relu_forward_kernel_rank_1_grid_stride_surface in0_ptr out0_ptr
        in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta
        tile_size0).toAlgKernel).TraceSafeR R bounds s := by
  unfold Kernel.TraceSafeR
  rw [gridStride_body_eq]
  -- `pid = tl.program_id(0)`
  refine Stmt.TraceSafeListR.cons_intro
    (by simp only [Stmt.TraceSafeR, Op.SafeAtR]) (fun t1 ht1 => ?_)
  rw [rsAssign_castFree _ s (by simp only [evalOpR, evalOp]),
    stepStmt_assign_eq_some (evalOp_programId 0 s)] at ht1
  obtain rfl := Option.some.inj ht1
  -- `num_tiles0 = tl.cdiv(s0, tile_size0)`
  refine Stmt.TraceSafeListR.cons_intro
    (by simp only [numTiles0Stmt, Stmt.TraceSafeR, Op.SafeAtR, and_self])
    (fun t2 ht2 => ?_)
  rw [show stepStmtR R (numTiles0Stmt s0 tile_size0)
        (s.setReg "pid" .nat [] (Tile.scalar (s.pids 0)))
      = stepStmt (numTiles0Stmt s0 tile_size0)
        (s.setReg "pid" .nat [] (Tile.scalar (s.pids 0))) from by
      rw [numTiles0Stmt]
      exact rsAssign_castFree _ _ (by simp only [evalOpR, evalOp]),
    numTiles0_step s0 tile_size0 _] at ht2
  obtain rfl := Option.some.inj ht2
  -- `num_ctas = tl.num_programs(0)`
  refine Stmt.TraceSafeListR.cons_intro
    (by simp only [Stmt.TraceSafeR, Op.SafeAtR]) (fun t3 ht3 => ?_)
  rw [rsAssign_castFree _ _ (by simp only [evalOpR, evalOp]),
    stepStmt_assign_eq_some (evalOp_numPrograms 0 _)] at ht3
  obtain rfl := Option.some.inj ht3
  -- the grid-stride loop
  refine Stmt.TraceSafeListR.cons_intro ?_
    (fun _ _ => Stmt.TraceSafeListR.nil_intro)
  simp only [Stmt.TraceSafeR]
  refine ⟨by simp only [Op.SafeAtR], by simp only [Op.SafeAtR],
    by simp only [Op.SafeAtR], ?_⟩
  rw [show evalOpR R (Op.constNat 0) _ = some (Tile.scalar 0) from by
      simp [evalOpR],
    show evalOpR R (Op.constNat tiles_per_cta) _
        = some (Tile.scalar tiles_per_cta) from by simp [evalOpR],
    show evalOpR R (Op.constNat 1) _ = some (Tile.scalar 1) from by simp [evalOpR]]
  refine Stmt.forRangeTraceSafeR_inv R bounds "j" tiles_per_cta 1 _
    (fun _ st => st.regs .nat [] "pid" = some (Tile.scalar (s.pids 0)) ∧
      st.regs .nat [] "num_ctas" = some (Tile.scalar (s.numPids 0)))
    ?_ 0 _ ⟨by simp, by simp⟩
  intro c sc hc hP
  obtain ⟨hpidC, hctasC⟩ := hP
  set scj := sc.setReg "j" .nat [] (Tile.scalar c) with hscj
  have hpidJ : scj.regs .nat [] "pid" = some (Tile.scalar (s.pids 0)) := by
    rw [hscj, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hpidC
  have hctasJ : scj.regs .nat [] "num_ctas"
      = some (Tile.scalar (s.numPids 0)) := by
    rw [hscj, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hctasC
  set addV : Tile .nat [] := Tile.bop (NumericDType.add .nat) Broadcast.nil
    (Tile.scalar (s.pids 0)) (Tile.bop (NumericDType.mul .nat) Broadcast.nil
      (Tile.scalar c) (Tile.scalar (s.numPids 0))) with haddV
  have hAssign : stepStmt (Stmt.assign .nat [] "tile_id"
      (Op.add NumericDType.nat Broadcast.nil (Op.ref .nat [] "pid")
        (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "j")
          (Op.ref .nat [] "num_ctas")))) scj
      = some (scj.setReg "tile_id" .nat [] addV) :=
    stepStmt_assign_eq_some (by
      simp only [evalOp_add, evalOp_mul, evalOp_ref, hpidJ, hctasJ,
        show scj.regs .nat [] "j" = some (Tile.scalar c) by simp [hscj],
        Option.bind_some, haddV]
      rfl)
  have haddData : addV.data PUnit.unit = s.pids 0 + c * s.numPids 0 := by
    rw [haddV]; rfl
  refine ⟨?_, ?_⟩
  · -- body safety
    refine Stmt.TraceSafeListR.cons_intro
      (by simp only [Stmt.TraceSafeR, Op.SafeAtR, and_self]) (fun tb htb => ?_)
    rw [rsAssign_castFree _ _ (by simp only [evalOpR, evalOp]), hAssign] at htb
    obtain rfl := Option.some.inj htb
    exact perTile_traceSafeR R in0_ptr out0_ptr in0_stride0 out0_stride0 s0
      tile_size0 bounds (s.pids 0 + c * s.numPids 0) _ addV
      (BlockState.setReg_same _ _ _ _ _) haddData
      (fun i hi => hin ⟨c, hc⟩ i hi) (fun i hi => hout ⟨c, hc⟩ i hi)
  · -- body run, invariant preserved
    obtain ⟨tb', hstepBody, -, -, -, hregs, -⟩ :=
      perTile_steps in0_ptr out0_ptr in0_stride0 out0_stride0 s0 tile_size0
        hStride (s.pids 0 + c * s.numPids 0)
        (scj.setReg "tile_id" .nat [] addV) addV (by simp) haddData
    refine ⟨tb', ?_, ?_, ?_⟩
    · rw [rsLoopBody_castFree R in0_ptr out0_ptr in0_stride0 out0_stride0 s0
        tile_size0, stepStmts.cons_some hAssign]
      exact hstepBody
    · rw [hregs _ _ _ (by decide) (by decide) (by decide) (by decide) (by decide)
        (by decide), BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
      exact hpidJ
    · rw [hregs _ _ _ (by decide) (by decide) (by decide) (by decide) (by decide)
        (by decide), BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
      exact hctasJ

/-! ### Termination and the **cell-level** frame of the grid-stride loop

`gs_loop_readback` already proves the per-lane values (and the `readMem`-level
`out0` frame). The `⊨[R]` triple additionally needs a per-**cell** frame over
every region, which `perTile_steps`' last conjunct supplies per iteration; the
loop lemma below threads it and, on the way, gives termination. -/

set_option maxHeartbeats 1000000 in
private theorem gs_loop_cells
    (in0_ptr out0_ptr : RegionName)
    (in0_stride0 out0_stride0 s0 tile_size0 : Nat)
    (hStride : 0 < out0_stride0) (P C : Nat) (tiles_per_cta : Nat) :
    ∀ (n c : Nat), c + n = tiles_per_cta →
    ∀ (t : BlockState),
      t.regs .nat [] "pid" = some (Tile.scalar P) →
      t.regs .nat [] "num_ctas" = some (Tile.scalar C) →
      ∃ t', stepForRangeAux "j" c tiles_per_cta 1
          (Stmt.assign .nat [] "tile_id"
              (Op.add NumericDType.nat Broadcast.nil (Op.ref .nat [] "pid")
                (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "j")
                  (Op.ref .nat [] "num_ctas")))
            :: perTileStmts in0_ptr out0_ptr in0_stride0 out0_stride0 s0 tile_size0)
          t = some t'
        ∧ (∀ (r : RegionName) (o : Nat),
            (r ≠ out0_ptr ∨ ∀ (j : Nat) (i : Fin tile_size0), c ≤ j →
              j < tiles_per_cta → taskIndex (P + j * C) tile_size0 i < s0 →
              o ≠ taskIndex (P + j * C) tile_size0 i * out0_stride0) →
            t'.mem r o = t.mem r o) := by
  intro n
  induction n with
  | zero =>
      intro c hc t hpid hctas
      exact ⟨t, stepForRangeAux.step_ge one_ne_zero (by omega), fun r o _ => rfl⟩
  | succ n ih =>
      intro c hc t hpid hctas
      have hlt : c < tiles_per_cta := by omega
      set tJ := t.setReg "j" .nat [] (Tile.scalar c) with htJ
      set addV : Tile .nat [] := Tile.bop (NumericDType.add .nat) Broadcast.nil
        (Tile.scalar P) (Tile.bop (NumericDType.mul .nat) Broadcast.nil
          (Tile.scalar c) (Tile.scalar C)) with haddV
      have hpidJ : tJ.regs .nat [] "pid" = some (Tile.scalar P) := by
        rw [htJ, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hpid
      have hctasJ : tJ.regs .nat [] "num_ctas" = some (Tile.scalar C) := by
        rw [htJ, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
        exact hctas
      have haddData : addV.data PUnit.unit = P + c * C := by rw [haddV]; rfl
      have hAssign : stepStmt (Stmt.assign .nat [] "tile_id"
          (Op.add NumericDType.nat Broadcast.nil (Op.ref .nat [] "pid")
            (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "j")
              (Op.ref .nat [] "num_ctas")))) tJ
          = some (tJ.setReg "tile_id" .nat [] addV) :=
        stepStmt_assign_eq_some (by
          simp only [evalOp_add, evalOp_mul, evalOp_ref, hpidJ, hctasJ,
            show tJ.regs .nat [] "j" = some (Tile.scalar c) by simp [htJ],
            Option.bind_some, haddV]
          rfl)
      obtain ⟨tb, hstepBody, -, -, -, hregs, hcell⟩ :=
        perTile_steps in0_ptr out0_ptr in0_stride0 out0_stride0 s0 tile_size0
          hStride (P + c * C) (tJ.setReg "tile_id" .nat [] addV) addV
          (by simp) haddData
      have hbody : stepStmts
          (Stmt.assign .nat [] "tile_id"
              (Op.add NumericDType.nat Broadcast.nil (Op.ref .nat [] "pid")
                (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "j")
                  (Op.ref .nat [] "num_ctas")))
            :: perTileStmts in0_ptr out0_ptr in0_stride0 out0_stride0 s0
              tile_size0) tJ = some tb := by
        rw [stepStmts.cons_some hAssign]; exact hstepBody
      have hpidB : tb.regs .nat [] "pid" = some (Tile.scalar P) := by
        rw [hregs _ _ _ (by decide) (by decide) (by decide) (by decide)
          (by decide) (by decide),
          BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
        exact hpidJ
      have hctasB : tb.regs .nat [] "num_ctas" = some (Tile.scalar C) := by
        rw [hregs _ _ _ (by decide) (by decide) (by decide) (by decide)
          (by decide) (by decide),
          BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
        exact hctasJ
      obtain ⟨t', hrest, hcellRest⟩ := ih (c + 1) (by omega) tb hpidB hctasB
      refine ⟨t', ?_, ?_⟩
      · rw [stepForRangeAux.step_lt one_ne_zero hlt, hbody, Option.bind_some]
        exact hrest
      · intro r o hcond
        have hcRest : r ≠ out0_ptr ∨ ∀ (j : Nat) (i : Fin tile_size0), c + 1 ≤ j →
            j < tiles_per_cta → taskIndex (P + j * C) tile_size0 i < s0 →
            o ≠ taskIndex (P + j * C) tile_size0 i * out0_stride0 := by
          rcases hcond with h | h
          · exact Or.inl h
          · exact Or.inr fun j i hcj hj hi => h j i (by omega) hj hi
        have hcHere : r ≠ out0_ptr ∨ ∀ i : Fin tile_size0,
            taskIndex (P + c * C) tile_size0 i < s0 →
            o ≠ taskIndex (P + c * C) tile_size0 i * out0_stride0 := by
          rcases hcond with h | h
          · exact Or.inl h
          · exact Or.inr fun i hi => h c i (Nat.le_refl c) hlt hi
        rw [hcellRest r o hcRest, hcell r o hcHere, htJ]
        simp

set_option maxHeartbeats 1000000 in
/-- Termination and the cell-level frame of the whole grid-stride kernel under
the exact stepper, from an arbitrary launch state. -/
private theorem relu_grid_stride_run_exact
    (in0_ptr out0_ptr : RegionName)
    (in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta tile_size0 : Nat)
    (hStride : 0 < out0_stride0) (s₀ : BlockState) :
    ∃ s1, exec ((relu_forward_kernel_rank_1_grid_stride_surface in0_ptr out0_ptr
        in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta
        tile_size0).toAlgKernel) s₀ = some s1
      ∧ (∀ (r : RegionName) (o : Nat),
          (r ≠ out0_ptr ∨ ∀ (j : Nat) (i : Fin tile_size0), j < tiles_per_cta →
            taskIndex (s₀.pids 0 + j * s₀.numPids 0) tile_size0 i < s0 →
            o ≠ taskIndex (s₀.pids 0 + j * s₀.numPids 0) tile_size0 i
                  * out0_stride0) →
          s1.mem r o = s₀.mem r o) := by
  set sp3 := ((s₀.setReg "pid" .nat [] (Tile.scalar (s₀.pids 0))).setReg
      "num_tiles0" .nat [] (numTiles0Val s0 tile_size0)).setReg
      "num_ctas" .nat [] (Tile.scalar (s₀.numPids 0)) with hsp3
  obtain ⟨s1, hloop, hcells⟩ :=
    gs_loop_cells in0_ptr out0_ptr in0_stride0 out0_stride0 s0 tile_size0
      hStride (s₀.pids 0) (s₀.numPids 0) tiles_per_cta tiles_per_cta 0 (by omega)
      sp3 (by simp [hsp3]) (by simp [hsp3])
  refine ⟨s1, ?_, ?_⟩
  · rw [show exec ((relu_forward_kernel_rank_1_grid_stride_surface in0_ptr
          out0_ptr in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta
          tile_size0).toAlgKernel) s₀
        = stepStmts ((relu_forward_kernel_rank_1_grid_stride_surface in0_ptr
          out0_ptr in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta
          tile_size0).toAlgKernel).body s₀ from rfl,
      gridStride_body_eq]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 0 s₀))]
    rw [stepStmts.cons_some (numTiles0_step s0 tile_size0 _)]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_numPrograms 0 _))]
    rw [stepStmts_singleton, stepForRangeAux.forRangeDyn_unfold]
    simp only [evalOp_constNat, Option.bind_some]
    exact hloop
  · intro r o hcond
    rw [hcells r o ?_, hsp3]
    · simp
    · rcases hcond with h | h
      · exact Or.inl h
      · exact Or.inr fun j i _ hj hi => h j i hj hi

/-- IO signature of the `one_tile_per_cta = false` **grid-stride** branch on the
grid-width-aware streaming surface: step `j` of program `(pid₀, _)` in a launch
grid of width `nCtas` covers the flat tile `pid₀ + j·nCtas`, whose lane `i` reads
`in0_ptr` at `t·in0_stride0` and writes `out0_ptr` at `t·out0_stride0` for the
task index `t`, active exactly on the `boundary_check=(0,)` guard `t < s0`.

`pre` is the grid-stride idiom's own launch legality `pid₀ < nCtas` — a program's
id is below its grid's width. `BlockState` carries no invariant tying `pids` to
`numPids`, and the kernel's loop genuinely needs `0 < nCtas` (at `nCtas = 0`
every step would revisit tile `pid₀`), so it is assumed here rather than
pretended free. `outDType` stays at the default `.real`: both source casts are
`.to(...element_ty)`, which the DSL erases. -/
def reluGridStrideIO (in0_ptr out0_ptr : RegionName)
    (in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta tile_size0 : Nat) :
    StreamGridStrideEmitMasked2DKernelIO₁ where
  kernel := relu_forward_kernel_rank_1_grid_stride_surface in0_ptr out0_ptr
    in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta tile_size0
  inp1 := in0_ptr
  out := out0_ptr
  T := tiles_per_cta
  B1 := tile_size0
  C := tile_size0
  pre := fun pid₀ _ nCtas => pid₀ < nCtas
  read1 := fun p₀ _ nCtas t j =>
    taskIndex (p₀ + t.val * nCtas) tile_size0 j * in0_stride0
  write := fun p₀ _ nCtas t j =>
    taskIndex (p₀ + t.val * nCtas) tile_size0 j * out0_stride0
  mask1 := fun p₀ _ nCtas t j => taskIndex (p₀ + t.val * nCtas) tile_size0 j < s0
  writeMask := fun p₀ _ nCtas t j =>
    taskIndex (p₀ + t.val * nCtas) tile_size0 j < s0

/-! ### ════════ ★ MAIN THEOREM ★ ════════ -/

/-- **The headline on the grid-width-aware IO surface** for
`relu_strided_buffer.py`'s `relu_forward_kernel_rank_1`,
`one_tile_per_cta = false` (grid-stride) branch: for every disjoint flat
placement of the two buffers, every program id and **launch grid width**, and
every launch state whose per-step input windows hold `xs`, the translated
pointer kernel terminates and every write-active lane of every grid-stride step
holds `relu (xs t i) = max 0 (xs t i)`, with every other memory **cell**
unchanged.

Dimension-general in `s0`, both strides, `tile_size0` and `tiles_per_cta`, and
**universally quantified over the launch grid width** — the point of the skin:
the windows stride by `nCtas = tl.num_programs(0)`, pinned inside the relation
to `s₀.numPids 0`.

Honest side-conditions: `0 < out0_stride0` (store-footprint injectivity — a
torch stride of a non-degenerate rank-1 buffer is ≥ 1); `in0_ptr ≠ out0_ptr`
(later grid-stride steps load after earlier steps stored, and the two buffers
carry *different* strides, so an aliased pair genuinely breaks the contract —
the IO surface's placement disjointness does not supply this, since the core
allows a skin to name one region twice for in-place kernels); and the skin's
`pre` `pid₀ < nCtas` (launch legality, which also supplies the `0 < nCtas` the
loop arithmetic needs).

`R` is threaded through the whole kernel but rounds nothing — with
`outDType := .real` every per-step store is exact under `execR R`, so this is
the exact grid-stride streaming contract. That is faithful rather than vacuous
because ReLU is a *selection* (`tl.where`), so no arithmetic manufactures a real
that would need quantizing. -/
specification relu_strided_buffer_grid_stride_io_correctnessR (R : RoundingModel)
    (in0_ptr out0_ptr : RegionName)
    (in0_stride0 out0_stride0 s0 num_tasks tiles_per_cta tile_size0 : Nat)
    (hStride : 0 < out0_stride0) (hDisj : in0_ptr ≠ out0_ptr) :
    reluGridStrideIO in0_ptr out0_ptr in0_stride0 out0_stride0 s0 num_tasks
        tiles_per_cta tile_size0
      ⊨[R] fun _pid₀ _pid₁ _nCtas xs t i => TiledActivation.relu (xs t i) := by
  refine StreamGridStrideEmitMasked2DKernelIO₁.ImplementsR.intro _ ?_ ?_ ?_
  · exact relu_grid_stride_flattenOk in0_ptr out0_ptr in0_stride0 out0_stride0 s0
      num_tasks tiles_per_cta tile_size0
  · intro bounds s xs _ _ hbr1 hbw
    exact relu_grid_stride_traceSafeR R in0_ptr out0_ptr in0_stride0 out0_stride0
      s0 num_tasks tiles_per_cta tile_size0 hStride bounds s hbr1 hbw
  · intro s₀ xs hpre _ hx
    simp only [reluGridStrideIO] at hpre hx ⊢
    have hGrid : 0 < s₀.numPids 0 := Nat.lt_of_le_of_lt (Nat.zero_le _) hpre
    obtain ⟨s1, hexec, hcells⟩ :=
      relu_grid_stride_run_exact in0_ptr out0_ptr in0_stride0 out0_stride0 s0
        num_tasks tiles_per_cta tile_size0 hStride s₀
    refine ⟨s1, ?_, ?_, ?_⟩
    · rw [relu_grid_stride_execR_eq_exec R in0_ptr out0_ptr in0_stride0
        out0_stride0 s0 num_tasks tiles_per_cta tile_size0 s₀]
      exact hexec
    · intro t i hact
      have hval := relu_grid_stride_exec_correct in0_ptr out0_ptr in0_stride0
        out0_stride0 s0 num_tasks tiles_per_cta tile_size0 s₀ hStride hDisj
        hGrid s1 hexec t.val i t.isLt hact
      rw [BlockState.readMemAs_real]
      refine congrArg some ?_
      rw [hval]
      simp only [reluSpec, FloatDType.ofReal, RoundingModel.round_real_apply]
      rw [hx t i hact]
    · intro r o hcond
      refine hcells r o ?_
      rcases hcond with h | h
      · exact Or.inl h
      · exact Or.inr fun j i hj hi => h ⟨j, hj⟩ i hi

end IOFace

end VeriTile.Bench.TritonBenchG.ReluStridedBuffer
