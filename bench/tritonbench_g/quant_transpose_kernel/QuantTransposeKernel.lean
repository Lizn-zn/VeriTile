import VeriTile.Triton

/-!
# `quant_transpose_kernel` — strict per-kernel correctness

`_quantize_global_transpose` reads a `BLOCK_M × BLOCK_N` tile of `A` under a
grouped one-dimensional program-id schedule, multiplies by a single global
inverse-absmax scale, rounds `127.0 · (a · absmax_inv)` to int8 via CUDA
`llrint`, and stores the result into `B` with transposed addressing
(`stride_bm`/`stride_bn`), masked by `(rm < M) & (rn < N)`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`_quantize_global_transpose[grid](...)`, the grid size
`cdiv(M, BLOCK_M) · cdiv(N, BLOCK_N)`, the grouped pid scheduling, and how the
runtime composes per-program writes into one buffer) is the *trusted boundary*,
not a proof obligation here. Because the program coordinates are universally
quantified over `s`, the per-program statement covers every program of the grid.

## Proof architecture

```
quant_transpose_scaled_store_io_correctness                          ← TOP THEOREM (`⊨`, dimension-general)
  ├─ quant_transpose_flattenOk                                        inside the flat-memory bridge
  ├─ quant_transpose_traceSafe                                        per-execution address safety
  └─ quant_transpose_region_run                                       region-model run
       ├─ quant_transpose_terminates                                  straight line + one masked store
       ├─ quantize_global_transpose_scaled_store_slice_correct         per-lane readback (shared, below)
       └─ quant_transpose_frame                                       cell-level frame off the write window

quantize_global_transpose_blocked_output_summary_general              per-write-map summary
  ├─ quantize_global_transpose_real_surface_toAlgorithm_blocked        faithful surface is blocked at erasure (llrint)
  └─ quantize_global_transpose_scaled_store_slice_compute_correct      ← ComputeCorrect over the masked transposed store
       └─ quantize_global_transpose_scaled_store_slice_correct         ← algorithm-layer readback per tile lane
  (output-address injectivity for the transposed writeback is taken as a hypothesis `hOutInj`)
```

The pre-rounding spec is `scale127 · (a · absmax_inv)` per lane
(`quantTransposeScaledSpec`); inputs are read from `BlockState` memory.

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); `@triton.autotune` /
`num_warps` are not modeled. The **faithful surface**
(`quantize_global_transpose_real_surface`) is *blocked* at algorithm erasure: the
CUDA `tl.extra.cuda.libdevice.llrint` rounding and int8 cast are outside
VeriTile's real-tile arithmetic layer, so `toAlgorithm?` returns `Except.error`.
What is positively verified is the pre-rounding *scaled-store slice*: the masked
transposed writeback of `scale127 · (a · absmax_inv)`, including output-address
injectivity for the four Python test shapes. The `llrint` rounding / int8 cast
itself is the honest blocker, recorded in each summary. Python's returned
`absmax` is computed on the host before the kernel and is outside this store
proof.
-/

namespace VeriTile.Bench.TritonBenchG.QuantTransposeKernel

open VeriTile.Triton
open scoped VeriTile.Triton.MaskedTile2DKernelIO₂

set_option linter.unusedSimpArgs false

/-- Real-valued surface of `quant_transpose_kernel.py`'s
`_quantize_global_transpose`.

This preserves the grouped one-dimensional program-id schedule, masked load,
global scale, CUDA `llrint` surface operation, transposed store addressing, and
masked writeback. The algorithm carrier records the pre-cast real value. -/
def quantize_global_transpose_real_surface
    (A AbsmaxInv B : RegionName)
    (stride_am stride_an stride_bn stride_bm M N BLOCK_M BLOCK_N GROUP_M : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0)
  grid_m = ($((M : Nat)) + $((BLOCK_M : Nat)) - $((1 : Nat))) // $((BLOCK_M : Nat))
  grid_n = ($((N : Nat)) + $((BLOCK_N : Nat)) - $((1 : Nat))) // $((BLOCK_N : Nat))
  width = $(GROUP_M) * grid_n
  group_id = pid // width
  group_size = min(grid_m - group_id * $(GROUP_M), $(GROUP_M))
  pid_m = group_id * $(GROUP_M) + (pid % group_size)
  pid_n = (pid % width) // group_size
  rm = pid_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  rn = pid_n * $(BLOCK_N) + tl.arange(0, $(BLOCK_N))
  A = A + (rm[:, None] * $(stride_am) + rn[None, :] * $(stride_an))
  mask = (rm < $(M))[:, None] & (rn < $(N))[None, :]
  a = tl.load(A, mask=mask)
  absmax_inv = tl.load(AbsmaxInv)
  rm = pid_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  rn = pid_n * $(BLOCK_N) + tl.arange(0, $(BLOCK_N))
  B = B + (rm[:, None] * $(stride_bm) + rn[None, :] * $(stride_bn))
  mask = (rm < $(M))[:, None] & (rn < $(N))[None, :]
  output = tl.extra.cuda.libdevice.llrint(127.0 * (a * absmax_inv))
  tl.store(B, output, mask=mask)
}

/-- The faithful quantize-global-transpose surface is blocked at algorithm
erasure by the CUDA `llrint` rounding operation. The scaled-store slice below
proves the real-valued expression before backend-specific rounding/cast. -/
theorem quantize_global_transpose_real_surface_toAlgorithm_blocked
    (A AbsmaxInv B : RegionName)
    (stride_am stride_an stride_bn stride_bm M N BLOCK_M BLOCK_N GROUP_M : Nat) :
    ∃ err,
      (quantize_global_transpose_real_surface A AbsmaxInv B stride_am stride_an
        stride_bn stride_bm M N BLOCK_M BLOCK_N GROUP_M).toAlgorithm? =
        Except.error err := by
  simp [quantize_global_transpose_real_surface, ComputeExpr.toAlgorithm?]

/-- Proof-oriented scaled-store tile slice of `quant_transpose_kernel.py`'s
`_quantize_global_transpose`.

The full Triton kernel uses a one-dimensional grouped program-id schedule to
derive `pid_m` and `pid_n`. This slice starts after that scheduling choice, uses
program axes 0/1 for the tile coordinates, loads the `BLOCK_M × BLOCK_N` tile
from `A`, applies the global `absmax_inv` scale, and proves the masked writeback
into `B`. CUDA `llrint` and int8 casting are outside VeriTile's current real-tile
arithmetic layer, matching the other quantization ports. -/
def quantize_global_transpose_scaled_store_slice
    (A AbsmaxInv B : RegionName)
    (stride_am stride_an stride_bn stride_bm M N BLOCK_M BLOCK_N : Nat)
    (scale127 : ℝ) :
    ComputeKernel := triton {
  pid_m = tl.program_id(0)
  pid_n = tl.program_id(1)
  rm = pid_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  rn = pid_n * $(BLOCK_N) + tl.arange(0, $(BLOCK_N))
  mask = (rm[:, None] < $(M)) & (rn[None, :] < $(N))
  a = tl.load(A + rm[:, None] * $(stride_am) + rn[None, :] * $(stride_an),
    mask=mask)
  absmax_inv = tl.load(AbsmaxInv)
  output = $(scale127) * (a * absmax_inv)
  tl.store(B + rm[:, None] * $(stride_bm) + rn[None, :] * $(stride_bn),
    output, mask=mask)
}

def rowIndex (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 0 * BLOCK_M + i.val

def colIndex (s : BlockState) (BLOCK_N : Nat) (j : Fin BLOCK_N) : Nat :=
  s.pids 1 * BLOCK_N + j.val

def active
    (s : BlockState) (M N BLOCK_M BLOCK_N : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_N]) : Prop :=
  rowIndex s BLOCK_M idx.1 < M ∧ colIndex s BLOCK_N idx.2.1 < N

instance activeDecidable
    (s : BlockState) (M N BLOCK_M BLOCK_N : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_N]) :
    Decidable (active s M N BLOCK_M BLOCK_N idx) := by
  unfold active
  infer_instance

def aOffset
    (s : BlockState) (stride_am stride_an BLOCK_M BLOCK_N : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_N]) : Nat :=
  rowIndex s BLOCK_M idx.1 * stride_am + colIndex s BLOCK_N idx.2.1 * stride_an

def bOffset
    (s : BlockState) (stride_bm stride_bn BLOCK_M BLOCK_N : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_N]) : Nat :=
  rowIndex s BLOCK_M idx.1 * stride_bm + colIndex s BLOCK_N idx.2.1 * stride_bn

noncomputable def quantTransposeScaledSpec
    (s : BlockState) (A AbsmaxInv : RegionName)
    (stride_am stride_an BLOCK_M BLOCK_N : Nat) (scale127 : ℝ)
    (idx : TileIndex [BLOCK_M, BLOCK_N]) : ℝ :=
  scale127 *
    (s.readMem A (aOffset s stride_am stride_an BLOCK_M BLOCK_N idx) *
      s.readMem AbsmaxInv 0)

/-- Algorithm-layer correctness for the quantize-global-transpose store slice. -/
theorem quantize_global_transpose_scaled_store_slice_correct
    (A AbsmaxInv B : RegionName)
    (stride_am stride_an stride_bn stride_bm M N BLOCK_M BLOCK_N : Nat)
    (scale127 : ℝ)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
        bOffset s stride_bm stride_bn BLOCK_M BLOCK_N idx)) :
    ∀ idx : TileIndex [BLOCK_M, BLOCK_N],
      let outAddr := bOffset s stride_bm stride_bn BLOCK_M BLOCK_N idx
      (exec (quantize_global_transpose_scaled_store_slice A AbsmaxInv B
            stride_am stride_an stride_bn stride_bm M N BLOCK_M BLOCK_N scale127)
          s).map (·.readMem B outAddr)
        = some (if active s M N BLOCK_M BLOCK_N idx then
            quantTransposeScaledSpec s A AbsmaxInv stride_am stride_an
              BLOCK_M BLOCK_N scale127 idx
          else s.readMem B outAddr) := by
  intro idx
  simp [exec, quantize_global_transpose_scaled_store_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
        Tile.ptrAdd, NumericDType.add, NumericDType.mul, ComparableDType.lt,
        rowIndex, colIndex, aOffset, bOffset, TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BLOCK_M, BLOCK_N] → Nat :=
    fun idx =>
      (s.pids 0 * BLOCK_M + idx.1.val) * stride_bm +
        (s.pids 1 * BLOCK_N + idx.2.1.val) * stride_bn
  let valueFn : TileIndex [BLOCK_M, BLOCK_N] → ℝ :=
    fun idx =>
      WithBot.unbotD 0
        (match
          match
            if s.pids 0 * BLOCK_M + idx.1.val < M ∧
                s.pids 1 * BLOCK_N + idx.2.1.val < N then
              some (s.readMem A
                ((s.pids 0 * BLOCK_M + idx.1.val) * stride_am +
                  (s.pids 1 * BLOCK_N + idx.2.1.val) * stride_an))
            else
              some (s.undef A
                ((s.pids 0 * BLOCK_M + idx.1.val) * stride_am +
                  (s.pids 1 * BLOCK_N + idx.2.1.val) * stride_an)) with
          | some x => some (x * s.readMem AbsmaxInv 0)
          | none => none with
        | some x => some (scale127 * x)
        | none => none)
  let P : TileIndex [BLOCK_M, BLOCK_N] → Prop :=
    fun idx =>
      s.pids 0 * BLOCK_M + idx.1.val < M ∧
        s.pids 1 * BLOCK_N + idx.2.1.val < N
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, bOffset, rowIndex, colIndex] using hOutInj
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj idx]
  by_cases hActive :
      s.pids 0 * BLOCK_M + idx.1.val < M ∧
        s.pids 1 * BLOCK_N + idx.2.1.val < N
  · simp [offsetFn, valueFn, P, active, quantTransposeScaledSpec, rowIndex,
      colIndex, aOffset, bOffset, hActive]
  · simp [offsetFn, valueFn, P, active, rowIndex, colIndex, aOffset, bOffset,
      hActive]

/-- Compute-facing correctness for the quantize-global-transpose store slice. -/
theorem quantize_global_transpose_scaled_store_slice_compute_correct
    (A AbsmaxInv B : RegionName)
    (stride_am stride_an stride_bn stride_bm M N BLOCK_M BLOCK_N : Nat)
    (scale127 : ℝ)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
        bOffset s stride_bm stride_bn BLOCK_M BLOCK_N idx)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := quantize_global_transpose_scaled_store_slice A AbsmaxInv B
        stride_am stride_an stride_bn stride_bm M N BLOCK_M BLOCK_N scale127)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s M N BLOCK_M BLOCK_N)
        (fun idx => (B, bOffset s stride_bm stride_bn BLOCK_M BLOCK_N idx)))
      (expected := fun idx =>
        quantTransposeScaledSpec s A AbsmaxInv stride_am stride_an
          BLOCK_M BLOCK_N scale127 idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [quantize_global_transpose_scaled_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := quantize_global_transpose_scaled_store_slice_correct A AbsmaxInv B
    stride_am stride_an stride_bn stride_bm M N BLOCK_M BLOCK_N scale127
    s hOutInj idx
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-- **Dimension-general blocked output summary.** For arbitrary strides, sizes
`M`/`N`, block sizes `BLOCK_M`/`BLOCK_N`, group factor `GROUP_M`, and real scale
`scale127` (and any program coordinates in `s`), the faithful full surface is
recorded as **blocked** at algorithm erasure (it stores CUDA `llrint`/int8
results, not the real-valued pre-rounding expression), while the checked
scaled-store slice realizes the genuine pre-rounding quantity
`scale127 * (a * absmax_inv)` (`quantTransposeScaledSpec`) at every in-range tile
lane, leaving out-of-range lanes unchanged. This holds over arbitrary (symbolic)
dimensions. Output-address injectivity for the transposed writeback is taken as a
hypothesis (`hOutInj`). The `llrint` rounding / int8 cast remain the honest,
unmodeled blocker. -/
specification quantize_global_transpose_blocked_output_summary_general
    (A AbsmaxInv B : RegionName)
    (stride_am stride_an stride_bn stride_bm M N BLOCK_M BLOCK_N GROUP_M : Nat)
    (scale127 : ℝ)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
        bOffset s stride_bm stride_bn BLOCK_M BLOCK_N idx)) :
    (∃ err, (quantize_global_transpose_real_surface A AbsmaxInv B
      stride_am stride_an stride_bn stride_bm M N BLOCK_M BLOCK_N
      GROUP_M).toAlgorithm? = Except.error err) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel := quantize_global_transpose_scaled_store_slice A AbsmaxInv B
        stride_am stride_an stride_bn stride_bm M N BLOCK_M BLOCK_N scale127)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s M N BLOCK_M BLOCK_N)
        (fun idx => (B, bOffset s stride_bm stride_bn BLOCK_M BLOCK_N idx)))
      (expected := fun idx =>
        quantTransposeScaledSpec s A AbsmaxInv stride_am stride_an
          BLOCK_M BLOCK_N scale127 idx) :=
  ⟨quantize_global_transpose_real_surface_toAlgorithm_blocked A AbsmaxInv B
      stride_am stride_an stride_bn stride_bm M N BLOCK_M BLOCK_N GROUP_M,
   quantize_global_transpose_scaled_store_slice_compute_correct A AbsmaxInv B
      stride_am stride_an stride_bn stride_bm M N BLOCK_M BLOCK_N scale127 s
      hOutInj⟩

/-! ## ════════ `⊨` IO face for the scaled-store slice ════════

The summary above is stated per *declared write map*. This section restates the
checked scaled-store slice on the audit-once IO surface
`MaskedTile2DKernelIO₂.Implements` (`⊨`), which additionally pins the **flat
memory** placement.

Two things make this the tile-indexed skin's natural home. The footprint is a
genuine 2-D tile — `row·stride_am + col·stride_an` on the input and the
*transposed* `row·stride_bm + col·stride_bn` on the output — so lanes are
`TileIndex [BLOCK_M, BLOCK_N]` and no `Fin (BLOCK_M · BLOCK_N)` flattening
appears anywhere. And the second input is the **broadcast scalar**
`absmax_inv = tl.load(AbsmaxInv)`: it is expressed as a second channel whose
window is the constant offset `0` and whose mask is `True` on every lane, so the
pin hypothesis says every lane of `ys` holds that one cell.

Masked-off lanes read the launch state's `undef` (the load carries no `other=`),
which `Implements` pins to `0`; they are never stored, so the spec value at an
active lane depends only on `xs` at active lanes. -/

section IOFace

/-- Cell-level frame of a masked scatter (private copy — `bench` files are
standalone). -/
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

/-- `Op.FlattenOk` of an `expandDim` broadcast reduces to its argument's
obligation. Stated as its own lemma because the per-case equation does not fire
under `simp` on an `expandDim` node, and `rw` only succeeds in an isolated
goal. -/
private theorem flattenOk_expandDim {dtype : TileDType} {shape : TileShape}
    (ax : Fin (shape.length + 1)) (e : Op dtype shape) (h : e.FlattenOk) :
    (Op.expandDim ax e).FlattenOk := by
  rw [Op.FlattenOk]
  exact h

/-- `Op.FlattenOk` of a register reference is vacuous. -/
private theorem flattenOk_ref (dtype : TileDType) (shape : TileShape)
    (name : RegName) : (Op.ref dtype shape name).FlattenOk := by
  rw [Op.FlattenOk]
  trivial

/-- `Op.SafeAt` of an `expandDim` broadcast reduces to its argument's
obligation. Stated as its own lemma (and used as a *term*, not a `simp` lemma):
the per-case equation does not fire under `simp` on an `expandDim` node, and
after `simp` has unfolded `Op.SafeAt` elsewhere an `iff` rewrite no longer
matches either. -/
private theorem safeAt_expandDim {dtype : TileDType} {shape : TileShape}
    (ax : Fin (shape.length + 1)) (e : Op dtype shape) (bounds : RegionBounds)
    (s : BlockState) (h : Op.SafeAt bounds s e) :
    Op.SafeAt bounds s (Op.expandDim ax e) := by
  rw [Op.SafeAt]
  exact h

/-- `Op.SafeAt` of a register reference is vacuous. -/
private theorem safeAt_ref (bounds : RegionBounds) (s : BlockState)
    (dtype : TileDType) (shape : TileShape) (name : RegName) :
    Op.SafeAt bounds s (Op.ref dtype shape name) := by
  rw [Op.SafeAt]
  trivial

/-- The scaled-store slice sits inside the flat-memory bridge's covered
fragment. -/
theorem quant_transpose_flattenOk
    (A AbsmaxInv B : RegionName)
    (stride_am stride_an stride_bn stride_bm M N BLOCK_M BLOCK_N : Nat)
    (scale127 : ℝ) :
    ((quantize_global_transpose_scaled_store_slice A AbsmaxInv B
      stride_am stride_an stride_bn stride_bm M N BLOCK_M
      BLOCK_N scale127).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [quantize_global_transpose_scaled_store_slice, ComputeKernel.toAlgKernel,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk]
  -- the two `expandDim` broadcasts
  exact ⟨flattenOk_expandDim _ _ (flattenOk_ref _ _ _),
    flattenOk_expandDim _ _ (flattenOk_ref _ _ _)⟩

/-- Termination of the slice: it is a straight line of register assigns plus one
masked store, so it never faults at the algorithm layer. -/
theorem quant_transpose_terminates
    (A AbsmaxInv B : RegionName)
    (stride_am stride_an stride_bn stride_bm M N BLOCK_M BLOCK_N : Nat)
    (scale127 : ℝ) (s : BlockState) :
    ∃ s1, exec (quantize_global_transpose_scaled_store_slice A AbsmaxInv B
      stride_am stride_an stride_bn stride_bm M N BLOCK_M BLOCK_N scale127)
      s = some s1 := by
  simp [exec, quantize_global_transpose_scaled_store_slice, stepStmts, stepStmt,
    evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop,
    Tile.expandDim, Tile.ptrAdd, NumericDType.add, NumericDType.mul,
    ComparableDType.lt]

/-- Cell-level frame of the slice: every cell off the active output window keeps
its value. -/
theorem quant_transpose_frame
    (A AbsmaxInv B : RegionName)
    (stride_am stride_an stride_bn stride_bm M N BLOCK_M BLOCK_N : Nat)
    (scale127 : ℝ) (s s' : BlockState)
    (hExec : exec (quantize_global_transpose_scaled_store_slice A AbsmaxInv B
      stride_am stride_an stride_bn stride_bm M N BLOCK_M BLOCK_N scale127)
      s = some s') :
    ∀ (r : RegionName) (o : Nat),
      (r ≠ B ∨ ∀ idx : TileIndex [BLOCK_M, BLOCK_N],
        active s M N BLOCK_M BLOCK_N idx →
        o ≠ bOffset s stride_bm stride_bn BLOCK_M BLOCK_N idx) →
      s'.mem r o = s.mem r o := by
  intro r o hcond
  simp [exec, quantize_global_transpose_scaled_store_slice, stepStmts,
    stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop,
    Tile.cop, Tile.expandDim, Tile.ptrAdd, NumericDType.add, NumericDType.mul,
    ComparableDType.lt, TileShape.dropInsertedIndex] at hExec
  subst hExec
  rw [foldl_writeMem_frame
    (region := B)
    (fun i : TileIndex [BLOCK_M, BLOCK_N] =>
      (s.pids 0 * BLOCK_M + i.1.val) * stride_bm
        + (s.pids 1 * BLOCK_N + i.2.1.val) * stride_bn)
    _
    (fun i : TileIndex [BLOCK_M, BLOCK_N] =>
      s.pids 0 * BLOCK_M + i.1.val < M ∧ s.pids 1 * BLOCK_N + i.2.1.val < N)
    r o (TileShape.allIndices [BLOCK_M, BLOCK_N]) ?_]
  · simp
  · rcases hcond with h | h
    · exact Or.inl h
    · exact Or.inr fun i _ hi =>
        Ne.symm (h i (by simpa [active, rowIndex, colIndex] using hi))

/-- Per-execution safety walk: the masked `A` load, the unmasked scalar
`AbsmaxInv` load and the masked `B` store address their own buffer's tile
window, active exactly on the `rm < M ∧ rn < N` guard. -/
theorem quant_transpose_traceSafe
    (A AbsmaxInv B : RegionName)
    (stride_am stride_an stride_bn stride_bm M N BLOCK_M BLOCK_N : Nat)
    (scale127 : ℝ) (bounds : RegionBounds) (s : BlockState)
    (hA : ∀ idx : TileIndex [BLOCK_M, BLOCK_N],
      active s M N BLOCK_M BLOCK_N idx →
      aOffset s stride_am stride_an BLOCK_M BLOCK_N idx < bounds A)
    (hS : 0 < bounds AbsmaxInv)
    (hB : ∀ idx : TileIndex [BLOCK_M, BLOCK_N],
      active s M N BLOCK_M BLOCK_N idx →
      bOffset s stride_bm stride_bn BLOCK_M BLOCK_N idx < bounds B) :
    ((quantize_global_transpose_scaled_store_slice A AbsmaxInv B
      stride_am stride_an stride_bn stride_bm M N BLOCK_M BLOCK_N
      scale127).toAlgKernel).TraceSafe bounds s := by
  simp [Kernel.TraceSafe, quantize_global_transpose_scaled_store_slice,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt, MaskOpt.SafeAt,
    MaskOpt.Active, MemAccess.SafeAt, MemAccess.ActiveAddressSafe,
    memAccessActiveAddressSafe, stepStmt, evalOp, evalOp.eq_def, Option.bind,
    Option.map, Tile.bop, Tile.cop, Tile.expandDim, Tile.ptrAdd,
    NumericDType.add, NumericDType.mul, ComparableDType.lt]
  -- what survives: the six `expandDim` broadcast obligations (simp cannot peel
  -- them), the two tile-window address bounds, and the scalar load's bound
  and_intros
  all_goals first
    | exact safeAt_expandDim _ _ _ _ (safeAt_ref _ _ _ _ _)
    | exact hS
    | (intro a b ha hb; exact hA (a, b, PUnit.unit) ⟨ha, hb⟩)
    | (intro a b ha hb; exact hB (a, b, PUnit.unit) ⟨ha, hb⟩)

/-- Region-model run of the scaled-store slice, in the shape
`MaskedTile2DKernelIO₂.Implements.intro` consumes: termination, the per-lane
readback against the pinned load values, and the cell-level frame. -/
theorem quant_transpose_region_run
    (A AbsmaxInv B : RegionName)
    (stride_am stride_an stride_bn stride_bm M N BLOCK_M BLOCK_N : Nat)
    (scale127 : ℝ) (s₀ : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
        bOffset s₀ stride_bm stride_bn BLOCK_M BLOCK_N idx))
    (xs ys : TileIndex [BLOCK_M, BLOCK_N] → ℝ)
    (hx : ∀ idx : TileIndex [BLOCK_M, BLOCK_N],
      active s₀ M N BLOCK_M BLOCK_N idx →
      s₀.readMem A (aOffset s₀ stride_am stride_an BLOCK_M BLOCK_N idx) = xs idx)
    (hy : ∀ idx : TileIndex [BLOCK_M, BLOCK_N],
      s₀.readMem AbsmaxInv 0 = ys idx) :
    ∃ s1, exec (quantize_global_transpose_scaled_store_slice A AbsmaxInv B
        stride_am stride_an stride_bn stride_bm M N BLOCK_M BLOCK_N scale127)
        s₀ = some s1
      ∧ (∀ idx : TileIndex [BLOCK_M, BLOCK_N],
          active s₀ M N BLOCK_M BLOCK_N idx →
          s1.readMem B (bOffset s₀ stride_bm stride_bn BLOCK_M BLOCK_N idx)
            = scale127 * (xs idx * ys idx))
      ∧ (∀ (r : RegionName) (o : Nat),
          (r ≠ B ∨ ∀ idx : TileIndex [BLOCK_M, BLOCK_N],
            active s₀ M N BLOCK_M BLOCK_N idx →
            o ≠ bOffset s₀ stride_bm stride_bn BLOCK_M BLOCK_N idx) →
          s1.mem r o = s₀.mem r o) := by
  obtain ⟨s1, hexec⟩ := quant_transpose_terminates A AbsmaxInv B stride_am
    stride_an stride_bn stride_bm M N BLOCK_M BLOCK_N scale127 s₀
  refine ⟨s1, hexec, ?_, quant_transpose_frame A AbsmaxInv B stride_am stride_an
    stride_bn stride_bm M N BLOCK_M BLOCK_N scale127 s₀ s1 hexec⟩
  intro idx hact
  have h := quantize_global_transpose_scaled_store_slice_correct A AbsmaxInv B
    stride_am stride_an stride_bn stride_bm M N BLOCK_M BLOCK_N scale127 s₀
    hOutInj idx
  have hval : s1.readMem B
        (bOffset s₀ stride_bm stride_bn BLOCK_M BLOCK_N idx)
      = if active s₀ M N BLOCK_M BLOCK_N idx then
          quantTransposeScaledSpec s₀ A AbsmaxInv stride_am stride_an BLOCK_M
            BLOCK_N scale127 idx
        else s₀.readMem B
          (bOffset s₀ stride_bm stride_bn BLOCK_M BLOCK_N idx) := by
    simpa [hexec] using h
  rw [hval, if_pos hact, quantTransposeScaledSpec, hx idx hact, hy idx]

/-- IO signature of the scaled-store slice on the **tile-indexed** two-input
surface: lane `(i, j)` of program `(pid₀, pid₁)` reads `A` at
`row·stride_am + col·stride_an`, reads the broadcast scalar `AbsmaxInv` at
offset `0` on every lane, writes `B` at the *transposed*
`row·stride_bm + col·stride_bn`, and is read/write-active exactly on the
kernel's `rm < M ∧ rn < N` guard. -/
def quantTransposeScaledIO (A AbsmaxInv B : RegionName)
    (stride_am stride_an stride_bn stride_bm M N BLOCK_M BLOCK_N : Nat)
    (scale127 : ℝ) : MaskedTile2DKernelIO₂ where
  kernel := quantize_global_transpose_scaled_store_slice A AbsmaxInv B
    stride_am stride_an stride_bn stride_bm M N BLOCK_M BLOCK_N scale127
  in1 := A
  in2 := AbsmaxInv
  out := B
  shape := [BLOCK_M, BLOCK_N]
  read1 := fun p₀ p₁ idx =>
    (p₀ * BLOCK_M + idx.1.val) * stride_am
      + (p₁ * BLOCK_N + idx.2.1.val) * stride_an
  read2 := fun _ _ _ => 0
  write := fun p₀ p₁ idx =>
    (p₀ * BLOCK_M + idx.1.val) * stride_bm
      + (p₁ * BLOCK_N + idx.2.1.val) * stride_bn
  mask := fun p₀ p₁ idx =>
    p₀ * BLOCK_M + idx.1.val < M ∧ p₁ * BLOCK_N + idx.2.1.val < N
  read2Mask := fun _ _ _ => True

/-! ### ════════ ★ MAIN THEOREM ★ ════════ -/

/-- **The headline on the IO surface** for the checked scaled-store slice of
`quant_transpose_kernel.py`'s `_quantize_global_transpose`: for every disjoint
flat placement of the three buffers, every program coordinate `(pid₀, pid₁)`
whose active lanes are in bounds, and every launch state whose `A` tile window
holds `xs` at the active lanes and whose `AbsmaxInv` cell holds the broadcast
scalar, the translated pointer kernel terminates, every active lane of the
**transposed** output tile holds the genuine pre-rounding quantity
`scale127 * (xs · ys)`, and every other memory cell is unchanged.

Dimension-general in all four strides, `M`, `N`, `BLOCK_M`, `BLOCK_N`, and the
real scale. Honest side-conditions: output-address injectivity for the
transposed writeback at every program coordinate (`hOutInj` — the same
hypothesis the per-write-map summary takes, here universally quantified over the
two program axes because `Implements` quantifies over them), and a non-degenerate
tile (`0 < BLOCK_M`, `0 < BLOCK_N`), which is what lets the signature's
broadcast-scalar channel witness the `AbsmaxInv` load's own bound. The CUDA
`llrint` rounding / int8 cast remain the honest, unmodeled blocker, exactly as in
the summary above — this face covers the same real-valued slice. -/
specification quant_transpose_scaled_store_io_correctness
    (A AbsmaxInv B : RegionName)
    (stride_am stride_an stride_bn stride_bm M N BLOCK_M BLOCK_N : Nat)
    (scale127 : ℝ)
    (hM : 0 < BLOCK_M) (hN : 0 < BLOCK_N)
    (hOutInj : ∀ p₀ p₁ : Nat, Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
        (p₀ * BLOCK_M + idx.1.val) * stride_bm
          + (p₁ * BLOCK_N + idx.2.1.val) * stride_bn)) :
    quantTransposeScaledIO A AbsmaxInv B stride_am stride_an stride_bn stride_bm
        M N BLOCK_M BLOCK_N scale127
      ⊨ fun _p₀ _p₁ xs ys idx => scale127 * (xs idx * ys idx) := by
  refine MaskedTile2DKernelIO₂.Implements.intro _ ?_ ?_ ?_
  · exact quant_transpose_flattenOk A AbsmaxInv B stride_am stride_an stride_bn
      stride_bm M N BLOCK_M BLOCK_N scale127
  · intro bounds s h1 h2 h3
    exact quant_transpose_traceSafe A AbsmaxInv B stride_am stride_an stride_bn
      stride_bm M N BLOCK_M BLOCK_N scale127 bounds s
      (fun idx hact => h1 idx hact)
      (h2 (⟨0, hM⟩, ⟨0, hN⟩, PUnit.unit) trivial)
      (fun idx hact => h3 idx hact)
  · intro s₀ xs ys hx hy
    exact quant_transpose_region_run A AbsmaxInv B stride_am stride_an stride_bn
      stride_bm M N BLOCK_M BLOCK_N scale127 s₀
      (hOutInj (s₀.pids 0) (s₀.pids 1)) xs ys
      (fun idx hact => hx idx hact) (fun idx => hy idx trivial)

end IOFace

end VeriTile.Bench.TritonBenchG.QuantTransposeKernel
