import VeriTile.Triton

/-!
# `matrix_transpose` — strict per-kernel correctness

`kernel` transposes a `SIZE_M × D_HEAD` matrix into a `D_HEAD × SIZE_M` output:
it builds 2-D strided pointer tiles `matrix_ptr` (row `size_m`, col `d_head`) and
`out_ptr` (row `d_head`, col `size_m`), loads the whole matrix tile, and stores
`tl.trans(matrix)` to `out_ptr`. The whole matrix is processed by a single
program (grid `(1,)`).

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`kernel[(1,)](...)`, the supplied strides from
`matrix.stride()` / `out.stride()`, and tensor allocation) is the *trusted
boundary*, not a proof obligation here. Since the grid has a single program,
the per-program statement is the whole computation.

The headline is stated on the kernel's **IO signature** `transposeIO`
(`Masked2DKernelIO₁`): which buffer is which argument, and — the whole point of
this kernel — the **transposed per-lane windows**. The `D_HEAD × SIZE_M` output
tile is flattened row-major into `B = D_HEAD * SIZE_M` lanes (`laneIdx`); lane
`j` (logical output cell `(d, m) = (j / SIZE_M, j % SIZE_M)`) writes
`d * out_stridex + m * out_stridey` and reads `m * matrix_stridex +
d * matrix_stridey` — the source cell with the two coordinates *swapped*. Every
lane is active (the kernel is unmasked), so `mask = writeMask = True`. Because
the read window already carries the transposition, the implemented per-lane
function is the identity `fun _ _ xs j => xs j`: the audit content lives in the
declared windows, and the headline **proves** the kernel's actual addressing
matches them.

`⊨` (`Masked2DKernelIO₁.Implements`) is the audit-once masked Hoare-triple
combinator: for **every** disjoint flat placement of the two buffers, **every**
program id whose lanes are in bounds, and **every** launch state whose input
window holds the tile `xs`, the translated pointer kernel terminates, every
output lane holds `xs j`, and every other memory cell is unchanged. The grid is
1-D (indeed a single program), so both program-id axes are quantified but
unused by every field.

## Proof architecture

```
kernel_correctness                       ← TOP SPECIFICATION (transposeIO ⊨ identity-on-transposed-windows)
  ├─ kernel_flattenOk                    bridge fragment membership
  ├─ kernel_traceSafe                    per-execution lane-wise safety walk
  └─ kernel_region_run                   region-model Hoare triple
       ├─ kernel_exec_isSome             termination
       ├─ kernel_correct                 ← algorithm-layer readback per cell
       └─ kernel_frame                   scatter-store cell frame
```

## Modeling boundary

Arithmetic over addresses is exact `Nat`; element values are over `ℝ` (the
upstream `float16` dtype is erased — post-erasure all dtypes unify to `ℝ`, so the
transpose is value-preserving and not bit-accurate). The headline carries an
explicit no-alias side condition `hOutInj`: the output address map must be
injective over the output tile's lanes (the standard non-overlapping-store
assumption). It is **required for truth** — if two lanes aliased the same output
cell, only the last store would survive, and the per-lane claim would be false.
The host's contiguous `out` buffer satisfies it.
-/

namespace VeriTile.Bench.TritonBenchG.MatrixTranspose

open VeriTile.Triton
open scoped VeriTile.Triton.Masked2DKernelIO₁

/-- Faithful 1:1 transcription of `matrix_transpose.py`'s `kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `SIZE_M: tl.constexpr` / `D_HEAD: tl.constexpr` → Lean `Nat`
  parameters. -/
def kernel
    (M Out : RegionName)
    (matrix_stridex matrix_stridey out_stridex out_stridey
      SIZE_M D_HEAD : Nat) :
    ComputeKernel := triton {
  size_m_arange = tl.arange(0, $(SIZE_M))
  d_head_arange = tl.arange(0, $(D_HEAD))
  matrix_ptr = M + size_m_arange[:, None] * $(matrix_stridex)
                + d_head_arange[None, :] * $(matrix_stridey)
  out_ptr = Out + d_head_arange[:, None] * $(out_stridex)
             + size_m_arange[None, :] * $(out_stridey)
  matrix = tl.load(matrix_ptr)
  tl.store(out_ptr, tl.trans(matrix))
}

/-! ### Row-major flattening of a 2-D tile into `⊨` lanes -/

/-- Lane `j` of the row-major flattening of an `R × C` tile. -/
def laneIdx (R C : Nat) (j : Fin (R * C)) : TileIndex [R, C] :=
  (⟨j.val / C, Nat.div_lt_of_lt_mul (Nat.lt_of_lt_of_le j.isLt (Nat.le_of_eq (Nat.mul_comm R C)))⟩,
   ⟨j.val % C, Nat.mod_lt _ (Nat.pos_of_ne_zero (by
      rintro rfl
      exact absurd j.isLt (by simp)))⟩,
   PUnit.unit)

/-- The lane of a logical `R × C` tile index — inverse of `laneIdx`. -/
def laneOf (R C : Nat) (idx : TileIndex [R, C]) : Fin (R * C) :=
  ⟨idx.1.val * C + idx.2.1.val, by
    have h1 : idx.1.val + 1 ≤ R := idx.1.isLt
    have h2 : idx.1.val * C + idx.2.1.val < idx.1.val * C + C :=
      Nat.add_lt_add_left idx.2.1.isLt _
    refine Nat.lt_of_lt_of_le h2 ?_
    rw [← Nat.succ_mul]
    exact Nat.mul_le_mul_right C h1⟩

@[simp] theorem laneIdx_laneOf (R C : Nat) (idx : TileIndex [R, C]) :
    laneIdx R C (laneOf R C idx) = idx := by
  obtain ⟨a, b, u⟩ := idx
  have hC : 0 < C := Nat.lt_of_le_of_lt (Nat.zero_le _) b.isLt
  have hdiv : (a.val * C + b.val) / C = a.val := by
    rw [Nat.mul_comm, Nat.mul_add_div hC, Nat.div_eq_of_lt b.isLt, Nat.add_zero]
  have hmod : (a.val * C + b.val) % C = b.val := by
    rw [Nat.mul_comm, Nat.mul_add_mod, Nat.mod_eq_of_lt b.isLt]
  simp only [laneIdx, laneOf, hdiv, hmod]

theorem laneOf_injective (R C : Nat) :
    Function.Injective (laneOf R C) :=
  Function.LeftInverse.injective (laneIdx_laneOf R C)

/-- Source address read by `kernel` at a logical output tile index. -/
def matrixAddr (matrix_stridex matrix_stridey : Nat)
    (idx : TileIndex [D_HEAD, SIZE_M]) : Nat :=
  idx.2.1.val * matrix_stridex + idx.1.val * matrix_stridey

/-- Output address written by `kernel` at a logical output tile index. -/
def outAddr (out_stridex out_stridey : Nat)
    (idx : TileIndex [D_HEAD, SIZE_M]) : Nat :=
  idx.1.val * out_stridex + idx.2.1.val * out_stridey

/-- Algorithm-layer cellwise correctness for `kernel`.

The injectivity hypothesis is the standard no-alias condition for the output
tile: if two logical output indices map to the same memory cell, a cellwise
postcondition cannot distinguish them after the store fold. -/
theorem kernel_correct
    (M Out : RegionName)
    (matrix_stridex matrix_stridey out_stridex out_stridey
      SIZE_M D_HEAD : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [D_HEAD, SIZE_M] => outAddr out_stridex out_stridey idx))
    (s' : BlockState)
    (hExec : exec (kernel M Out matrix_stridex matrix_stridey out_stridex out_stridey
        SIZE_M D_HEAD) s = some s') :
    ∀ idx : TileIndex [D_HEAD, SIZE_M],
      s'.readMem Out (outAddr out_stridex out_stridey idx)
        = s.readMem M (matrixAddr matrix_stridex matrix_stridey idx) := by
  intro idx
  simp [exec, kernel, stepStmts, stepStmt, evalOp.eq_def,
        Tile.bop, Tile.ptrAdd, Tile.expandDim, Tile.transpose,
        NumericDType.add, NumericDType.mul,
        TileShape.insertAxis, TileShape.dropInsertedIndex] at hExec
  subst s'
  have hRawInj : Function.Injective
      (fun idx : TileIndex [D_HEAD, SIZE_M] =>
        idx.1.val * out_stridex + idx.2.1.val * out_stridey) := by
    simpa [outAddr] using hOutInj
  simp [matrixAddr, outAddr]
  rw [BlockState.scatter_readback_nd _ _ _ hRawInj idx]

/-- An unmasked scatter-store `foldl` leaves every memory cell it does not hit
unchanged (cell-level frame for the store). -/
private theorem foldl_store_preserve_cell {α : Type} {region : RegionName}
    (offsetFn : α → Nat) (valueFn : α → ℝ)
    (r : RegionName) (o : Nat) (l : List α) (s : BlockState)
    (hnot : ∀ k ∈ l, ¬(region = r ∧ offsetFn k = o)) :
    (l.foldl (fun acc k => acc.writeMem region (offsetFn k) (valueFn k))
      s).mem r o = s.mem r o := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
      rw [List.foldl_cons,
        ih _ (fun k hk => hnot k (List.mem_cons_of_mem hd hk)),
        BlockState.writeMem_mem]
      exact if_neg (fun hc =>
        hnot hd List.mem_cons_self ⟨hc.1.symm, hc.2.symm⟩)

/-- Frame half: every memory cell not written by the output store — every cell
of every region other than `Out`, and every cell of `Out` outside the output
window — is preserved by the run. -/
private theorem kernel_frame
    (M Out : RegionName)
    (matrix_stridex matrix_stridey out_stridex out_stridey
      SIZE_M D_HEAD : Nat)
    (s s1 : BlockState)
    (hExec : exec ((kernel M Out matrix_stridex matrix_stridey out_stridex
        out_stridey SIZE_M D_HEAD).toAlgKernel) s = some s1)
    (r : RegionName) (o : Nat)
    (hmiss : ∀ idx : TileIndex [D_HEAD, SIZE_M],
      ¬(Out = r ∧ outAddr out_stridex out_stridey idx = o)) :
    s1.mem r o = s.mem r o := by
  simp [exec, kernel, ComputeKernel.toAlgKernel, stepStmts, stepStmt, evalOp.eq_def,
        Tile.bop, Tile.ptrAdd, Tile.expandDim, Tile.transpose,
        NumericDType.add, NumericDType.mul,
        TileShape.insertAxis, TileShape.dropInsertedIndex] at hExec
  subst hExec
  refine Eq.trans (foldl_store_preserve_cell _ _ r o _ _ ?_) rfl
  intro k _ hc
  exact hmiss k (by simpa [outAddr] using hc)

/-- Termination: the kernel executes to completion from any state (a
straight-line unmasked load/transpose/store). -/
private theorem kernel_exec_isSome
    (M Out : RegionName)
    (matrix_stridex matrix_stridey out_stridex out_stridey
      SIZE_M D_HEAD : Nat)
    (s : BlockState) :
    ∃ s1, exec ((kernel M Out matrix_stridex matrix_stridey out_stridex
        out_stridey SIZE_M D_HEAD).toAlgKernel) s = some s1 := by
  simp [exec, kernel, ComputeKernel.toAlgKernel, stepStmts, stepStmt, evalOp.eq_def,
        Tile.bop, Tile.ptrAdd, Tile.expandDim, Tile.transpose,
        NumericDType.add, NumericDType.mul,
        TileShape.insertAxis, TileShape.dropInsertedIndex]

/-- The kernel sits inside the flat-memory bridge's covered fragment
(pointer arithmetic, an unmasked load, a transpose, an unmasked store). -/
theorem kernel_flattenOk
    (M Out : RegionName)
    (matrix_stridex matrix_stridey out_stridex out_stridey
      SIZE_M D_HEAD : Nat) :
    ((kernel M Out matrix_stridex matrix_stridey out_stridex out_stridey
      SIZE_M D_HEAD).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [kernel, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
    StmtList.FlattenOk, Stmt.FlattenOk,
    Op.FlattenOk.eq_def]

/-- Per-execution safety walk: the two `arange`s and the two pointer-tile
computations are memory-silent; the unmasked load of the `[SIZE_M, D_HEAD]`
source tile and the unmasked store of the `[D_HEAD, SIZE_M]` output tile
reduce to the lane-wise bounds hypotheses (each 2-D tile index is the image
of exactly one `⊨` lane under `laneIdx`). -/
theorem kernel_traceSafe
    (M Out : RegionName)
    (matrix_stridex matrix_stridey out_stridex out_stridey
      SIZE_M D_HEAD : Nat)
    (bounds : RegionBounds) (s : BlockState)
    (h1 : ∀ j : Fin (D_HEAD * SIZE_M),
      matrixAddr matrix_stridex matrix_stridey (laneIdx D_HEAD SIZE_M j)
        < bounds M)
    (h2 : ∀ j : Fin (D_HEAD * SIZE_M),
      outAddr out_stridex out_stridey (laneIdx D_HEAD SIZE_M j)
        < bounds Out) :
    Kernel.TraceSafe bounds
      ((kernel M Out matrix_stridex matrix_stridey out_stridex out_stridey
        SIZE_M D_HEAD).toAlgKernel) s := by
  have h1' : ∀ (a : Fin SIZE_M) (b : Fin D_HEAD),
      a.val * matrix_stridex + b.val * matrix_stridey < bounds M := by
    intro a b
    have := h1 (laneOf D_HEAD SIZE_M (b, a, PUnit.unit))
    simpa [matrixAddr] using this
  have h2' : ∀ (a : Fin D_HEAD) (b : Fin SIZE_M),
      a.val * out_stridex + b.val * out_stridey < bounds Out := by
    intro a b
    have := h2 (laneOf D_HEAD SIZE_M (a, b, PUnit.unit))
    simpa [outAddr] using this
  unfold Kernel.TraceSafe
  simp [kernel, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
    Stmt.TraceSafeList, Stmt.TraceSafe,
    Op.SafeAt.eq_def, Op.PointerAddressesSafeOn, Op.MemorySafe, MaskOpt.SafeAt,
    MemAccess.SafeAt, stepStmt, evalOp.eq_def, MemAccess.ActiveAddressSafe,
    memAccessActiveAddressSafe, MaskOpt.Active, BlockState.setReg,
    Tile.bop, Tile.ptrAdd, Tile.expandDim, Tile.transpose,
    NumericDType.add, NumericDType.mul,
    TileShape.insertAxis, TileShape.dropInsertedIndex]
  exact ⟨fun a b => h1' a b, fun a b => h2' a b⟩

/-- **The region-model Hoare triple** — termination, per-lane output values,
and frame off the output window, from any launch state whose (transposed)
input window holds the tile `xs`. This is the `hrun` obligation of
`Masked2DKernelIO₁.Implements.intro`. -/
theorem kernel_region_run
    (M Out : RegionName)
    (matrix_stridex matrix_stridey out_stridex out_stridey
      SIZE_M D_HEAD : Nat)
    (s₀ : BlockState) (xs : Fin (D_HEAD * SIZE_M) → ℝ)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [D_HEAD, SIZE_M] => outAddr out_stridex out_stridey idx))
    (hx : ∀ j : Fin (D_HEAD * SIZE_M),
      s₀.readMem M (matrixAddr matrix_stridex matrix_stridey
        (laneIdx D_HEAD SIZE_M j)) = xs j) :
    ∃ s1, exec ((kernel M Out matrix_stridex matrix_stridey out_stridex
        out_stridey SIZE_M D_HEAD).toAlgKernel) s₀ = some s1
      ∧ (∀ j : Fin (D_HEAD * SIZE_M),
          s1.readMem Out (outAddr out_stridex out_stridey
            (laneIdx D_HEAD SIZE_M j)) = xs j)
      ∧ (∀ r o,
          (r ≠ Out ∨ ∀ j : Fin (D_HEAD * SIZE_M),
            o ≠ outAddr out_stridex out_stridey (laneIdx D_HEAD SIZE_M j)) →
          s1.mem r o = s₀.mem r o) := by
  obtain ⟨s1, hs1⟩ := kernel_exec_isSome M Out matrix_stridex matrix_stridey
    out_stridex out_stridey SIZE_M D_HEAD s₀
  have hs1' : exec (kernel M Out matrix_stridex matrix_stridey out_stridex
      out_stridey SIZE_M D_HEAD) s₀ = some s1 := hs1
  refine ⟨s1, hs1, fun j => ?_, fun r o hcond => ?_⟩
  · have h := kernel_correct M Out matrix_stridex matrix_stridey out_stridex
      out_stridey SIZE_M D_HEAD s₀ hOutInj s1 hs1' (laneIdx D_HEAD SIZE_M j)
    rw [h, hx j]
  · refine kernel_frame M Out matrix_stridex matrix_stridey out_stridex
      out_stridey SIZE_M D_HEAD s₀ s1 hs1 r o (fun idx ⟨hr, ho⟩ => ?_)
    rcases hcond with hne | hno
    · exact hne hr.symm
    · exact hno (laneOf D_HEAD SIZE_M idx) (by rw [laneIdx_laneOf]; exact ho.symm)

/-- `kernel`'s **IO signature** — the whole kernel-specific audit surface of
the `⊨` headline:

* `inp`/`out` — which buffer is which argument (`M`, `Out`);
* `B = D_HEAD * SIZE_M` — the output tile flattened row-major into lanes,
  lane `j` = logical output cell `(j / SIZE_M, j % SIZE_M)` (`laneIdx`);
* `read` — lane `j`'s **transposed** source address
  `m * matrix_stridex + d * matrix_stridey`;
* `write` — lane `j`'s output address `d * out_stridex + m * out_stridey`;
* `mask` — every lane (the kernel's load and store are both unmasked), so
  `writeMask` keeps the struct default.

The grid is `(1,)`, so no field depends on either program id. -/
def transposeIO (M Out : RegionName)
    (matrix_stridex matrix_stridey out_stridex out_stridey
      SIZE_M D_HEAD : Nat) :
    Masked2DKernelIO₁ where
  kernel := kernel M Out matrix_stridex matrix_stridey out_stridex out_stridey
    SIZE_M D_HEAD
  inp := M
  out := Out
  B := D_HEAD * SIZE_M
  read := fun _ _ j =>
    matrixAddr matrix_stridex matrix_stridey (laneIdx D_HEAD SIZE_M j)
  write := fun _ _ j =>
    outAddr out_stridex out_stridey (laneIdx D_HEAD SIZE_M j)
  mask := fun _ _ _ => True

/-- **The headline**: `kernel` implements the matrix transpose on its IO
signature — for every disjoint flat placement of `M`/`Out`, every program id
whose lanes are in bounds, and every launch state whose transposed source
window holds the tile `xs`, the translated pointer kernel terminates, output
lane `j` holds `xs j` (i.e. the output cell `(d, m)` holds the source cell
`(m, d)` — the transposition lives in the signature's windows, which this
theorem proves the kernel actually uses), and every other memory cell is
unchanged.

`hOutInj` is an honest, **truth-required** side condition: two distinct output
lanes must not alias the same memory cell, otherwise only the last store
survives and the per-lane claim is false. The host's contiguous `out` buffer
satisfies it. -/
specification kernel_correctness
    (M Out : RegionName)
    (matrix_stridex matrix_stridey out_stridex out_stridey
      SIZE_M D_HEAD : Nat)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [D_HEAD, SIZE_M] =>
        outAddr out_stridex out_stridey idx)) :
    transposeIO M Out matrix_stridex matrix_stridey out_stridex out_stridey
        SIZE_M D_HEAD
      ⊨ fun _ _ xs j => xs j := by
  refine Masked2DKernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact kernel_flattenOk M Out matrix_stridex matrix_stridey out_stridex
      out_stridey SIZE_M D_HEAD
  · intro bounds s hin hout _
    exact kernel_traceSafe M Out matrix_stridex matrix_stridey out_stridex
      out_stridey SIZE_M D_HEAD bounds s
      (fun j => hin j trivial) (fun j => hout j trivial)
  · intro s₀ xs hx
    obtain ⟨s1, hs1, hval, hframe⟩ :=
      kernel_region_run M Out matrix_stridex matrix_stridey out_stridex
        out_stridey SIZE_M D_HEAD s₀ xs hOutInj (fun j => hx j trivial)
    exact ⟨s1, hs1, fun j _ => hval j, fun r o hcond _ =>
      hframe r o (by
        rcases hcond with hne | hno
        · exact Or.inl hne
        · exact Or.inr fun j => hno j trivial)⟩

end VeriTile.Bench.TritonBenchG.MatrixTranspose
