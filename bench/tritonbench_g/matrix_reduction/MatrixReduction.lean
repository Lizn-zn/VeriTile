import VeriTile.Triton

/-!
# `matrix_reduction` — strict per-kernel correctness

`load_reduce_kernel` loads a `BLOCK_M × BLOCK_N` tile of `x_ptr` through a
block pointer, computes the row-wise maximum `tl.max(x, axis=1)`, and stores the
resulting length-`BLOCK_M` vector to `y_ptr + tl.arange(0, BLOCK_M)`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`load_reduce_kernel[(1,)](...)`, the grid size, the
strides chosen on the host, and how the runtime composes per-program writes into
one buffer) is the *trusted boundary*, not a proof obligation here. Because the
program id `pid` is universally quantified (it enters only via `BlockState`), the
per-program statement covers every program of the grid.

## Proof architecture

```
matrix_reduction_io_correctness                 ← TOP THEOREM (`⊨`)
  ├─ matrix_reduction_flattenOk                 inside the flat-memory bridge
  ├─ matrix_reduction_traceSafe                 per-execution address safety
  └─ matrix_reduction_region_run                region-model run
       ├─ matrix_reduction_terminates
       ├─ load_reduce_kernel_correct            per-row readback (shared, below)
       ├─ matrixReduceSpec_eq_of                memory spec = value spec under the pins
       └─ matrix_reduction_frame                cell-level frame (only `y_ptr[i]` is touched)

load_reduce_kernel_output_summary               per-write-map summary
  ├─ (toAlgorithm? = Except.ok _)               surface lowers to the algorithm layer
  └─ load_reduce_kernel_compute_correct         ← ComputeCorrect over the row-vector store
       └─ load_reduce_kernel_correct            ← algorithm-layer readback per row lane
            └─ matrixReduceSpec / matrixReduceInputTile (row-wise `Tile.reduceMax`)
```

The spec `matrixReduceSpec` is the row-wise `Tile.reduceMax` of the input tile
read by the block pointer.

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float). The Python test mixes `float16`/`float32` and passes
`check_dtype=False`; post-erasure all dtypes unify to `ℝ`, so the cast is the
identity here. The unused `stride_y` argument is kept as `_stride_y`. The store
target `y_ptr` is taken disjoint enough that the row-vector scatter is injective
(`BlockState.tileIndex1d_offset_injective`).
-/

namespace VeriTile.Bench.TritonBenchG.MatrixReduction

open VeriTile.Triton
open scoped VeriTile.Triton.MaskedTileKernelIO₁

/-- Faithful 1:1 transcription of `matrix_reduction.py`'s `load_reduce_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_M: tl.constexpr` / `BLOCK_N: tl.constexpr` → Lean `Nat`
  parameters.
- `stride_y` is kept as `_stride_y`: the upstream Triton kernel accepts it but
  stores to `y_ptr + tl.arange(0, BLOCK_M)` and does not use it. -/
def load_reduce_kernel
    (x_ptr y_ptr : RegionName)
    (stride_xm stride_xn _stride_y BLOCK_M BLOCK_N : Nat) :
    ComputeKernel := triton {
  x_ptr = tl.make_block_ptr(base=x_ptr,
    shape=($(BLOCK_M), $(BLOCK_N)),
    strides=($(stride_xm), $(stride_xn)),
    offsets=($(0), $(0)),
    block_shape=($(BLOCK_M), $(BLOCK_N)),
    order=(1, 0))
  x = tl.load(x_ptr)
  y = tl.max(x, axis=1)
  tl.store(y_ptr + tl.arange(0, $(BLOCK_M)), y)
}

/-- Input tile read by the block pointer in `load_reduce_kernel`. -/
noncomputable def matrixReduceInputTile
    (s : BlockState) (x_ptr : RegionName)
    (stride_xm stride_xn BLOCK_M BLOCK_N : Nat) :
    Tile .real [BLOCK_M, BLOCK_N] :=
  { data := fun idx =>
      let bp : BlockPtr :=
        { region := x_ptr, baseOffset := 0, parentShape := [BLOCK_M, BLOCK_N],
          blockShape := [BLOCK_M, BLOCK_N], strides := [stride_xm, stride_xn],
          offsets := [0, 0] }
      some (s.readMem x_ptr (bp.address (TileShape.indexToList [BLOCK_M, BLOCK_N] idx))) }

/-- Exact row-wise max written by `load_reduce_kernel` at row lane `i`. -/
noncomputable def matrixReduceSpec
    (s : BlockState) (x_ptr : RegionName)
    (stride_xm stride_xn BLOCK_M BLOCK_N : Nat) (i : Fin BLOCK_M) : ℝ :=
  match Tile.reduceMax (shape := [BLOCK_M, BLOCK_N]) ⟨1, by simp⟩ Bool.false
      (matrixReduceInputTile s x_ptr stride_xm stride_xn BLOCK_M BLOCK_N) with
  | some out => WithBot.unbotD 0 (out.data (i, PUnit.unit))
  | none => 0

/-- Algorithm-layer row-wise max correctness for `load_reduce_kernel`. -/
theorem load_reduce_kernel_correct
    (x_ptr y_ptr : RegionName)
    (stride_xm stride_xn stride_y BLOCK_M BLOCK_N : Nat)
    (s s' : BlockState)
    (hExec : exec (load_reduce_kernel x_ptr y_ptr stride_xm stride_xn stride_y
          BLOCK_M BLOCK_N) s = some s') :
    ∀ i : Fin BLOCK_M,
      s'.readMem y_ptr i.val =
        matrixReduceSpec s x_ptr stride_xm stride_xn BLOCK_M BLOCK_N i := by
  intro i
  have h_inj := BlockState.tileIndex1d_offset_injective (BLOCK := BLOCK_M)
  by_cases hBM : 0 < BLOCK_M
  · by_cases hBN : 0 < BLOCK_N
    · simp [exec, load_reduce_kernel, stepStmts, stepStmt, evalOp, evalOp.eq_def,
            Tile.reduceMax, Tile.reduceMaxDrop,
            TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
            BlockPtr.address, BlockPtr.inBounds, hBN] at hExec
      subst s'
      rw [BlockState.scatter_readback_nd _ _ _ h_inj (i, PUnit.unit)]
      simp [matrixReduceSpec, matrixReduceInputTile, Tile.reduceMax, Tile.reduceMaxDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        BlockPtr.address, hBN]
      congr with x
    · simp [exec, load_reduce_kernel, stepStmts, stepStmt, evalOp, evalOp.eq_def,
            Tile.reduceMax, Tile.reduceMaxDrop,
            TileShape.axisDim, TileShape.eraseAxis, hBN] at hExec
  · exact False.elim (hBM (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Compute-facing row-wise max correctness for `load_reduce_kernel`. -/
theorem load_reduce_kernel_compute_correct
    (x_ptr y_ptr : RegionName)
    (stride_xm stride_xn stride_y BLOCK_M BLOCK_N : Nat)
    (s : BlockState) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := load_reduce_kernel x_ptr y_ptr stride_xm stride_xn stride_y
        BLOCK_M BLOCK_N)
      (initialState := s)
      (write := fun i : Fin BLOCK_M => some (y_ptr, i.val))
      (expected := fun i => matrixReduceSpec s x_ptr stride_xm stride_xn BLOCK_M BLOCK_N i) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel rfl
  intro s0 s' hExec hs0
  subst s0
  intro i
  exact load_reduce_kernel_correct x_ptr y_ptr stride_xm stride_xn stride_y
    BLOCK_M BLOCK_N s s' hExec i

/-- Per-kernel output summary for `load_reduce_kernel`: the DSL surface lowers to
the algorithm layer, and the row-vector store to `y_ptr` is compute-correct —
every row lane `i` holds the row-wise maximum `matrixReduceSpec`. -/
specification load_reduce_kernel_output_summary
    (x_ptr y_ptr : RegionName)
    (stride_xm stride_xn stride_y BLOCK_M BLOCK_N : Nat)
    (s : BlockState) :
    (∃ alg, (load_reduce_kernel x_ptr y_ptr stride_xm stride_xn stride_y
        BLOCK_M BLOCK_N).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel := load_reduce_kernel x_ptr y_ptr stride_xm stride_xn stride_y
        BLOCK_M BLOCK_N)
      (initialState := s)
      (write := fun i : Fin BLOCK_M => some (y_ptr, i.val))
      (expected := fun i =>
        matrixReduceSpec s x_ptr stride_xm stride_xn BLOCK_M BLOCK_N i) := by
  refine ⟨⟨_, rfl⟩, ?_⟩
  exact load_reduce_kernel_compute_correct x_ptr y_ptr stride_xm stride_xn
    stride_y BLOCK_M BLOCK_N s

/-! ## ════════ `⊨` IO face ════════

The summary above is stated per *declared write map*. This section restates the
kernel on the audit-once IO surface `MaskedTileKernelIO₁.Implements` (`⊨`), which
additionally pins the **flat memory** placement.

The footprint is a genuine 2-D tile — `i·stride_xm + j·stride_xn`, read through
the block pointer with no `boundary_check`, hence every lane read-active — so
lanes are `TileIndex [BLOCK_M, BLOCK_N]` and nothing is flattened. The output is
the length-`BLOCK_M` row vector `y_ptr + i`, expressed on the same lane set by
making **column 0 the write-active lanes** and taking the write address to be the
row index: a reduction's write footprint is a sub-slice of its read tile, and
that is exactly what an independent `writeMask` / `write` pair says (the same
shape as `max_reduction`'s one-lane write mask in #568). -/

section IOFace

/-- Cell-level frame of an **unmasked** scatter: a cell the fold never writes
keeps its `mem` value verbatim. `bench` files are standalone, so this induction
is a private copy rather than an import. -/
private theorem foldl_writeMem_frame_unmasked {α : Type} {region : RegionName}
    (offsetFn : α → Nat) (valueFn : α → ℝ) (R : RegionName) (off : Nat) :
    ∀ l : List α, (R ≠ region ∨ ∀ k ∈ l, offsetFn k ≠ off) →
      ∀ s : BlockState,
        ((l.foldl (fun acc k => acc.writeMem region (offsetFn k) (valueFn k))
            s).mem R off) = s.mem R off := by
  intro l
  induction l with
  | nil => intro _ s; rfl
  | cons hd tl ih =>
      intro hc s
      have htl : R ≠ region ∨ ∀ k ∈ tl, offsetFn k ≠ off := by
        rcases hc with h | h
        · exact Or.inl h
        · exact Or.inr fun k hk => h k (List.mem_cons_of_mem hd hk)
      rw [List.foldl_cons, ih htl, BlockState.writeMem_mem, if_neg ?_]
      rintro ⟨h1, h2⟩
      rcases hc with h | h
      · exact h h1
      · exact h hd List.mem_cons_self h2.symm

/-- `Op.FlattenOk` of a `reduceMax` reduces to its argument's obligation. Stated
as its own lemma and used as a *term*: the per-case equation does not fire under
`simp` on a `reduceMax` node, only under `rw`, and `rw` only succeeds in an
isolated goal (third instance of this pattern — see #566 `castFloat`, #567
`expandDim`, #568 `reduceMax`). -/
private theorem flattenOk_reduceMax {shape : TileShape}
    (ax : Fin shape.length) (keepDims : Bool) (e : Op .real shape)
    (h : e.FlattenOk) : (Op.reduceMax ax keepDims e).FlattenOk := by
  rw [Op.FlattenOk]
  exact h

/-- `Op.SafeAt` of a `reduceMax` is its argument's obligation (same caveat). -/
private theorem safeAt_reduceMax {shape : TileShape} (bounds : RegionBounds)
    (s : BlockState) (ax : Fin shape.length) (keepDims : Bool)
    (e : Op .real shape) (h : Op.SafeAt bounds s e) :
    Op.SafeAt bounds s (Op.reduceMax ax keepDims e) := by
  rw [Op.SafeAt]
  exact h

/-- `Op.FlattenOk` of a register reference is vacuous. -/
private theorem flattenOk_ref (dtype : TileDType) (shape : TileShape)
    (name : RegName) : (Op.ref dtype shape name).FlattenOk := by
  rw [Op.FlattenOk]
  trivial

/-- `Op.SafeAt` of a register reference is vacuous. -/
private theorem safeAt_ref (bounds : RegionBounds) (s : BlockState)
    (dtype : TileDType) (shape : TileShape) (name : RegName) :
    Op.SafeAt bounds s (Op.ref dtype shape name) := by
  rw [Op.SafeAt]
  trivial

/-- Lane address of the input tile: the block pointer's `[i, j]` cell. -/
def xOffset (stride_xm stride_xn BLOCK_M BLOCK_N : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_N]) : Nat :=
  idx.1.val * stride_xm + idx.2.1.val * stride_xn

/-- Value-level row-wise max spec: the `Tile.reduceMax` along axis 1 of the tile
that holds `xs`, read off at lane `idx`'s **row**. Written over the loaded values
rather than over memory, which is what the IO surface quantifies. -/
noncomputable def matrixReduceSpecOf (BLOCK_M BLOCK_N : Nat)
    (xs : TileIndex [BLOCK_M, BLOCK_N] → ℝ)
    (idx : TileIndex [BLOCK_M, BLOCK_N]) : ℝ :=
  match Tile.reduceMax (shape := [BLOCK_M, BLOCK_N]) ⟨1, by simp⟩ Bool.false
      ⟨fun k => some (xs k)⟩ with
  | some out => WithBot.unbotD 0 (out.data (idx.1, PUnit.unit))
  | none => 0

/-- The memory-level and value-level specs agree once the input tile is pinned to
`xs`. -/
theorem matrixReduceSpec_eq_of (x_ptr : RegionName)
    (stride_xm stride_xn BLOCK_M BLOCK_N : Nat) (s : BlockState)
    (xs : TileIndex [BLOCK_M, BLOCK_N] → ℝ)
    (hx : ∀ idx : TileIndex [BLOCK_M, BLOCK_N],
      s.readMem x_ptr (xOffset stride_xm stride_xn BLOCK_M BLOCK_N idx) = xs idx)
    (idx : TileIndex [BLOCK_M, BLOCK_N]) :
    matrixReduceSpec s x_ptr stride_xm stride_xn BLOCK_M BLOCK_N idx.1
      = matrixReduceSpecOf BLOCK_M BLOCK_N xs idx := by
  have htile :
      matrixReduceInputTile s x_ptr stride_xm stride_xn BLOCK_M BLOCK_N
        = ⟨fun k : TileIndex [BLOCK_M, BLOCK_N] => some (xs k)⟩ := by
    refine congrArg Tile.mk (funext fun k => ?_)
    simp only [matrixReduceInputTile]
    rw [show (BlockPtr.address
          { region := x_ptr, baseOffset := 0,
            parentShape := [BLOCK_M, BLOCK_N], blockShape := [BLOCK_M, BLOCK_N],
            strides := [stride_xm, stride_xn], offsets := [0, 0] }
          (TileShape.indexToList [BLOCK_M, BLOCK_N] k))
        = xOffset stride_xm stride_xn BLOCK_M BLOCK_N k from by
      rw [show TileShape.indexToList [BLOCK_M, BLOCK_N] k
            = [k.1.val, k.2.1.val] from rfl,
        BlockPtr.address_2d_zero_offsets]
      simp [xOffset]]
    rw [hx k]
  rw [matrixReduceSpec, matrixReduceSpecOf, htile]

/-- The kernel sits inside the flat-memory bridge's covered fragment. -/
theorem matrix_reduction_flattenOk (x_ptr y_ptr : RegionName)
    (stride_xm stride_xn stride_y BLOCK_M BLOCK_N : Nat) :
    ((load_reduce_kernel x_ptr y_ptr stride_xm stride_xn stride_y BLOCK_M
      BLOCK_N).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [load_reduce_kernel, ComputeKernel.toAlgKernel, StmtList.FlattenOk,
    Stmt.FlattenOk, Op.FlattenOk]
  exact flattenOk_reduceMax _ _ _ (flattenOk_ref _ _ _)

/-- Termination (a non-degenerate reduction axis is genuinely needed: with
`BLOCK_N = 0` the `tl.max` has no axis to reduce and the kernel faults). -/
theorem matrix_reduction_terminates (x_ptr y_ptr : RegionName)
    (stride_xm stride_xn stride_y BLOCK_M BLOCK_N : Nat) (hBN : 0 < BLOCK_N)
    (s : BlockState) :
    ∃ s1, exec (load_reduce_kernel x_ptr y_ptr stride_xm stride_xn stride_y
      BLOCK_M BLOCK_N) s = some s1 := by
  simp [exec, load_reduce_kernel, stepStmts, stepStmt, evalOp.eq_def,
    Tile.reduceMax, Tile.reduceMaxDrop, TileShape.axisDim, TileShape.eraseAxis,
    TileShape.insertAxisIndex, BlockPtr.address, BlockPtr.inBounds, hBN]

/-- Per-execution safety walk. -/
theorem matrix_reduction_traceSafe (x_ptr y_ptr : RegionName)
    (stride_xm stride_xn stride_y BLOCK_M BLOCK_N : Nat) (hBN : 0 < BLOCK_N)
    (bounds : RegionBounds) (s : BlockState)
    (hX : ∀ idx : TileIndex [BLOCK_M, BLOCK_N],
      xOffset stride_xm stride_xn BLOCK_M BLOCK_N idx < bounds x_ptr)
    (hY : ∀ i : Fin BLOCK_M, i.val < bounds y_ptr) :
    ((load_reduce_kernel x_ptr y_ptr stride_xm stride_xn stride_y BLOCK_M
      BLOCK_N).toAlgKernel).TraceSafe bounds s := by
  simp [Kernel.TraceSafe, load_reduce_kernel, Stmt.TraceSafeList, Stmt.TraceSafe,
    Op.SafeAt, MaskOpt.SafeAt, MaskOpt.Active, MemAccess.SafeAt,
    MemAccess.MemorySafe, memAccessMemorySafe, MemAccess.ActiveAddressSafe,
    memAccessActiveAddressSafe, Op.PointerAddressesSafeOn, Op.MemorySafe,
    stepStmt, evalOp, evalOp.eq_def, Tile.bop, Tile.cop, Tile.ptrAdd,
    Tile.reduceMax, Tile.reduceMaxDrop, TileShape.axisDim, TileShape.eraseAxis,
    TileShape.insertAxisIndex, NumericDType.add, NumericDType.mul,
    ComparableDType.lt, hBN]
  -- what survives: the tile window's bound, the `reduceMax` node (simp cannot
  -- peel it), and the row-vector store's bound
  refine ⟨fun a b => ?_, safeAt_reduceMax _ _ _ _ _ (safeAt_ref _ _ _ _ _),
    fun a => hY a⟩
  simpa [xOffset] using hX (a, b, PUnit.unit)

/-- Cell-level frame: the only cells touched are `y_ptr[i]`, `i < BLOCK_M`. -/
theorem matrix_reduction_frame (x_ptr y_ptr : RegionName)
    (stride_xm stride_xn stride_y BLOCK_M BLOCK_N : Nat) (hBN : 0 < BLOCK_N)
    (s s' : BlockState)
    (hExec : exec (load_reduce_kernel x_ptr y_ptr stride_xm stride_xn stride_y
      BLOCK_M BLOCK_N) s = some s') :
    ∀ (r : RegionName) (o : Nat), (r ≠ y_ptr ∨ ∀ i : Fin BLOCK_M, o ≠ i.val) →
      s'.mem r o = s.mem r o := by
  intro r o hcond
  simp [exec, load_reduce_kernel, stepStmts, stepStmt, evalOp.eq_def,
    Tile.reduceMax, Tile.reduceMaxDrop, TileShape.axisDim, TileShape.eraseAxis,
    TileShape.insertAxisIndex, hBN] at hExec
  subst hExec
  rw [foldl_writeMem_frame_unmasked (region := y_ptr)
    (fun i : TileIndex [BLOCK_M] => i.1.val) _ r o
    (TileShape.allIndices [BLOCK_M]) ?_]
  · simp
  · rcases hcond with h | h
    · exact Or.inl h
    · exact Or.inr fun i _ => Ne.symm (h i.1)

/-- Region-model run, in the shape `Implements.intro` consumes. -/
theorem matrix_reduction_region_run (x_ptr y_ptr : RegionName)
    (stride_xm stride_xn stride_y BLOCK_M BLOCK_N : Nat) (hBN : 0 < BLOCK_N)
    (s₀ : BlockState) (xs : TileIndex [BLOCK_M, BLOCK_N] → ℝ)
    (hx : ∀ idx : TileIndex [BLOCK_M, BLOCK_N],
      s₀.readMem x_ptr (xOffset stride_xm stride_xn BLOCK_M BLOCK_N idx)
        = xs idx) :
    ∃ s1, exec (load_reduce_kernel x_ptr y_ptr stride_xm stride_xn stride_y
        BLOCK_M BLOCK_N) s₀ = some s1
      ∧ (∀ idx : TileIndex [BLOCK_M, BLOCK_N],
          s1.readMem y_ptr idx.1.val
            = matrixReduceSpecOf BLOCK_M BLOCK_N xs idx)
      ∧ (∀ (r : RegionName) (o : Nat),
          (r ≠ y_ptr ∨ ∀ i : Fin BLOCK_M, o ≠ i.val) →
          s1.mem r o = s₀.mem r o) := by
  obtain ⟨s1, hexec⟩ := matrix_reduction_terminates x_ptr y_ptr stride_xm
    stride_xn stride_y BLOCK_M BLOCK_N hBN s₀
  refine ⟨s1, hexec, ?_, matrix_reduction_frame x_ptr y_ptr stride_xm stride_xn
    stride_y BLOCK_M BLOCK_N hBN s₀ s1 hexec⟩
  intro idx
  rw [load_reduce_kernel_correct x_ptr y_ptr stride_xm stride_xn stride_y
      BLOCK_M BLOCK_N s₀ s1 hexec idx.1,
    matrixReduceSpec_eq_of x_ptr stride_xm stride_xn BLOCK_M BLOCK_N s₀ xs hx idx]

/-- IO signature on the tile-indexed surface: every lane of the
`BLOCK_M × BLOCK_N` tile reads `x_ptr` at `i·stride_xm + j·stride_xn` and is
read-active (no `boundary_check`); **column 0** is write-active and writes
`y_ptr[i]`. -/
def matrixReduceIO (x_ptr y_ptr : RegionName)
    (stride_xm stride_xn stride_y BLOCK_M BLOCK_N : Nat) :
    MaskedTileKernelIO₁ where
  kernel := load_reduce_kernel x_ptr y_ptr stride_xm stride_xn stride_y BLOCK_M
    BLOCK_N
  inp := x_ptr
  out := y_ptr
  shape := [BLOCK_M, BLOCK_N]
  read := fun _pid idx => xOffset stride_xm stride_xn BLOCK_M BLOCK_N idx
  write := fun _pid idx => idx.1.val
  mask := fun _pid _ => True
  writeMask := fun _pid idx => idx.2.1.val = 0

/-! ### ════════ ★ MAIN THEOREM ★ ════════ -/

/-- **The headline on the IO surface** for `matrix_reduction.py`'s
`load_reduce_kernel`: for every disjoint flat placement of the two buffers, every
launch state whose `BLOCK_M × BLOCK_N` input tile holds `xs`, the translated
pointer kernel terminates, `y_ptr[i]` holds the genuine row-wise maximum of row
`i` of `xs`, and every other memory cell is unchanged.

Dimension-general in both strides, `BLOCK_M` and `BLOCK_N`. Honest
side-condition: `0 < BLOCK_N` — with an empty reduction axis `tl.max` faults, so
termination genuinely fails there; the same hypothesis witnesses a write-active
lane, turning the skin's lane-wise frame disjunct into the row-vector one. -/
specification matrix_reduction_io_correctness (x_ptr y_ptr : RegionName)
    (stride_xm stride_xn stride_y BLOCK_M BLOCK_N : Nat) (hBN : 0 < BLOCK_N) :
    matrixReduceIO x_ptr y_ptr stride_xm stride_xn stride_y BLOCK_M BLOCK_N
      ⊨ fun _pid xs idx => matrixReduceSpecOf BLOCK_M BLOCK_N xs idx := by
  refine MaskedTileKernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact matrix_reduction_flattenOk x_ptr y_ptr stride_xm stride_xn stride_y
      BLOCK_M BLOCK_N
  · intro bounds s h1 h2
    exact matrix_reduction_traceSafe x_ptr y_ptr stride_xm stride_xn stride_y
      BLOCK_M BLOCK_N hBN bounds s (fun idx => h1 idx trivial)
      (fun i => h2 (i, ⟨0, hBN⟩, PUnit.unit) rfl)
  · intro s₀ xs hin
    obtain ⟨s1, hexec, hval, hframe⟩ :=
      matrix_reduction_region_run x_ptr y_ptr stride_xm stride_xn stride_y
        BLOCK_M BLOCK_N hBN s₀ xs (fun idx => hin idx trivial)
    refine ⟨s1, hexec, fun idx _ => hval idx, ?_⟩
    intro r o hcond
    refine hframe r o ?_
    rcases hcond with h | h
    · exact Or.inl h
    · exact Or.inr fun i => h (i, ⟨0, hBN⟩, PUnit.unit) rfl

end IOFace

end VeriTile.Bench.TritonBenchG.MatrixReduction
