import VeriTile.Triton
import VeriTile.Examples.Common

/-!
# `square_matrix` — strict per-kernel correctness

`square_kernel` squares a matrix row-wise: program `row_idx` loads one row of
`input_ptr` (a `BLOCK_SIZE`-wide tile starting at `row_idx·input_row_stride`,
masked by `col_offsets < n_cols` with out-of-bounds lanes defaulting to `-inf`),
computes `row * row`, and stores `x²` to the corresponding row of `output_ptr`,
masked by `col_offsets < n_cols`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`square_kernel[(n_rows,)](...)`, the one-program-per-row
grid, the host-side `BLOCK_SIZE = next_power_of_2(n_cols)` choice, `num_warps`,
and how the runtime composes per-program writes into one buffer) is the *trusted
boundary*, not a proof obligation here. Because `row_idx` is universally
quantified (as the program id), the per-program statement covers every row of
the grid.

## Proof architecture

```
square_kernel_correctness                   ← TOP SPECIFICATION (squareIO ⊨ lane-wise square)
  ├─ square_kernel_flattenOk                bridge fragment membership
  ├─ square_kernel_traceSafe                per-execution lane-wise safety walk
  └─ square_kernel_region_run               region-model masked Hoare triple
       ├─ square_kernel_correct             ← algorithm-layer readback per column
       └─ square_kernel_frame               masked scatter-store cell frame
```

The headline is stated on the kernel's masked **IO signature** `squareIO`
(`MaskedKernelIO₁`): which buffer is which argument, where program `pid` reads
its row (`pid * input_row_stride`) and writes it (`pid * output_row_stride`),
and the active-lane predicate `j < n_cols` (the same for every program — the
row prefix that actually exists in the matrix). `⊨` is the audit-once masked
Hoare-triple combinator (`MaskedKernelIO₁.Implements`): for **every** disjoint
placement of the two buffers in flat memory, **every** program id all of whose
*active* lanes are in bounds, and **every** launch state whose active
input-row lanes hold `xs`, the translated pointer kernel terminates, every
active output-row lane holds `xs i * xs i`, and every other memory cell is
unchanged. The spec is plain elementwise `xs i * xs i` — no
optimizer/reduction oracle applies.

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); the manual `num_warps`
heuristic is not modeled. The `other=-float('inf')` masked-load default does
not affect any in-bounds (`col_offsets < n_cols`) lane, which is all the spec
constrains; out-of-bounds output columns are preserved verbatim. No
output/input disjointness is assumed in the region model: the row is read into
registers before the masked scatter.
-/

namespace VeriTile.Bench.TritonBenchG.SquareMatrix

open VeriTile.Triton VeriTile.Examples
open scoped VeriTile.Triton.MaskedKernelIO₁

/-- Faithful 1:1 transcription of `square_matrix.py`'s `square_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` → Lean `Nat` parameter. -/
def square_kernel
    (output_ptr input_ptr : RegionName)
    (input_row_stride output_row_stride n_cols BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  row_idx = tl.program_id(0)
  row_start_ptr = input_ptr + row_idx * $(input_row_stride)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  input_ptrs = row_start_ptr + col_offsets
  row = tl.load(input_ptrs, mask=col_offsets < $(n_cols), other=-float("inf"))
  square_output = row * row
  output_row_start_ptr = output_ptr + row_idx * $(output_row_stride)
  output_ptrs = output_row_start_ptr + col_offsets
  tl.store(output_ptrs, square_output, mask=col_offsets < $(n_cols))
}

/-- Algorithm-layer correctness for `square_kernel`.

For the current row pid, active columns write `x^2`; inactive tail columns are
preserved. -/
theorem square_kernel_correct
    (output_ptr input_ptr : RegionName)
    (input_row_stride output_row_stride n_cols BLOCK_SIZE : Nat)
    (s : BlockState) (xs : Fin BLOCK_SIZE → ℝ)
    (h_x : InputRowLoadedAt s input_ptr input_row_stride BLOCK_SIZE xs) :
    ∀ i : Fin BLOCK_SIZE,
      let outAddr := s.pid * output_row_stride + i.val
      (exec (square_kernel output_ptr input_ptr input_row_stride output_row_stride
            n_cols BLOCK_SIZE) s).map (·.readMem output_ptr outAddr)
        = some (if i.val < n_cols then xs i * xs i else s.readMem output_ptr outAddr) := by
  intro i
  simp [exec, square_kernel, stepStmts, stepStmt, evalOp.eq_def,
        Tile.bop, Tile.cop, Tile.ptrAdd, NumericDType.mul,
        ComparableDType.lt]
  unfold InputRowLoadedAt at h_x
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
        (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
  by_cases hi : i.val < n_cols
  · simp [hi, h_x]
  · simp [hi]

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

set_option maxHeartbeats 1600000 in
/-- Frame half: every memory cell not actively written by the masked output
store — every cell of every region other than `output_ptr`, and the *inactive*
columns of the output row itself — is preserved by the run. -/
private theorem square_kernel_frame
    (output_ptr input_ptr : RegionName)
    (input_row_stride output_row_stride n_cols BLOCK_SIZE : Nat)
    (s s1 : BlockState)
    (hExec : exec ((square_kernel output_ptr input_ptr input_row_stride
        output_row_stride n_cols BLOCK_SIZE).toAlgKernel) s = some s1)
    (r : RegionName) (o : Nat)
    (hmiss : ∀ i : Fin BLOCK_SIZE, i.val < n_cols →
      ¬(output_ptr = r ∧ s.pid * output_row_stride + i.val = o)) :
    s1.mem r o = s.mem r o := by
  simp [exec, square_kernel, ComputeKernel.toAlgKernel, stepStmts, stepStmt,
        evalOp.eq_def, Tile.bop, Tile.cop, Tile.ptrAdd,
        NumericDType.mul, ComparableDType.lt] at hExec
  subst hExec
  refine Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_) rfl
  intro k _ hmk hc
  exact hmiss k.1 (by simpa using hmk) hc

/-- **The region-model masked Hoare triple** — termination, active-lane output
values, and frame off the active output lanes, from any launch state whose
input row is loaded at the **active lanes only** (`j < n_cols`; the masked
load defaults inactive lanes to `-inf`, so they are never consulted). This is
the `hrun` obligation of the `⊨` headline; the value half reuses
`square_kernel_correct` (instantiated at the row the state actually holds). -/
theorem square_kernel_region_run
    (output_ptr input_ptr : RegionName)
    (input_row_stride output_row_stride n_cols BLOCK_SIZE : Nat)
    (s₀ : BlockState) (xs : Fin BLOCK_SIZE → ℝ)
    (hx : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s₀.readMem input_ptr (s₀.pid * input_row_stride + j.val) = xs j) :
    ∃ s1, exec ((square_kernel output_ptr input_ptr input_row_stride
          output_row_stride n_cols BLOCK_SIZE).toAlgKernel) s₀ = some s1
      ∧ (∀ j : Fin BLOCK_SIZE, j.val < n_cols →
          s1.readMem output_ptr (s₀.pid * output_row_stride + j.val)
            = xs j * xs j)
      ∧ (∀ r o,
          (r ≠ output_ptr ∨ ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
            o ≠ s₀.pid * output_row_stride + j.val) →
          s1.mem r o = s₀.mem r o) := by
  have hobs := square_kernel_correct output_ptr input_ptr input_row_stride
    output_row_stride n_cols BLOCK_SIZE s₀
    (fun j => s₀.readMem input_ptr (s₀.pid * input_row_stride + j.val))
    (fun _ => rfl)
  rw [show exec (square_kernel output_ptr input_ptr input_row_stride
        output_row_stride n_cols BLOCK_SIZE) s₀
      = exec ((square_kernel output_ptr input_ptr input_row_stride
          output_row_stride n_cols BLOCK_SIZE).toAlgKernel) s₀
      from rfl] at hobs
  cases hsrc : exec ((square_kernel output_ptr input_ptr input_row_stride
      output_row_stride n_cols BLOCK_SIZE).toAlgKernel) s₀ with
  | none =>
      exact absurd hsrc (by
        simp [exec, square_kernel, ComputeKernel.toAlgKernel, stepStmts,
          stepStmt, evalOp.eq_def, Tile.bop, Tile.cop, Tile.ptrAdd,
          NumericDType.mul, ComparableDType.lt])
  | some s1 =>
      refine ⟨s1, rfl, fun j hj => ?_, fun r o hcond => ?_⟩
      · have hje := hobs j
        rw [hsrc] at hje
        simp only [Option.map_some, Option.some_inj, if_pos hj] at hje
        rw [hje, hx j hj]
      · refine square_kernel_frame output_ptr input_ptr input_row_stride
          output_row_stride n_cols BLOCK_SIZE s₀ s1 hsrc r o
          (fun i hi ⟨hr, ho⟩ => ?_)
        rcases hcond with hne | hno
        · exact hne hr.symm
        · exact hno i hi ho.symm

set_option maxHeartbeats 1600000 in
/-- Per-execution safety walk: the masked row load and masked row store address
their strided windows (`pid * input_row_stride + j` / `pid * output_row_stride
+ j`), active only when `j < n_cols`, so the bounds contract is **lane-wise**:
every *active* lane's address is below the region bound of the buffer it
touches. -/
theorem square_kernel_traceSafe
    (output_ptr input_ptr : RegionName)
    (input_row_stride output_row_stride n_cols BLOCK_SIZE : Nat)
    (bounds : RegionBounds) (s : BlockState)
    (hin : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s.pid * input_row_stride + j.val < bounds input_ptr)
    (hout : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s.pid * output_row_stride + j.val < bounds output_ptr) :
    Kernel.TraceSafe bounds
      ((square_kernel output_ptr input_ptr input_row_stride output_row_stride
        n_cols BLOCK_SIZE).toAlgKernel) s := by
  unfold Kernel.TraceSafe
  -- Computational unroll: walks all nine statements, discharging every
  -- load-free `SafeAt` and reducing the two masked accesses' lane-wise
  -- address obligations to the bounds hypotheses below.
  simp [square_kernel, ComputeKernel.toAlgKernel,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
    MaskOpt.SafeAt, MemAccess.SafeAt, stepStmt, evalOp.eq_def,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    MaskOpt.Active, Op.PointerAddressesSafeOn, Op.MemorySafe,
    BlockState.setReg,
    Tile.bop, Tile.cop, Tile.ptrAdd,
    NumericDType.mul, ComparableDType.lt]
  exact ⟨fun a ha => hin a ha, fun a ha => hout a ha⟩

/-- The kernel sits inside the flat-memory bridge's covered fragment (pointer
arithmetic, masked load with `other`, register multiply, masked store). -/
theorem square_kernel_flattenOk
    (output_ptr input_ptr : RegionName)
    (input_row_stride output_row_stride n_cols BLOCK_SIZE : Nat) :
    ((square_kernel output_ptr input_ptr input_row_stride output_row_stride
      n_cols BLOCK_SIZE).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [square_kernel, ComputeKernel.toAlgKernel,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk]

/-- `square_kernel`'s masked **IO signature** — the whole kernel-specific
audit surface of the `⊨` headline:

* `inp`/`out` — which buffer is which argument (the wiring);
* `B = BLOCK_SIZE` — the row window each program owns;
* `read`/`write` — program `pid` reads its row at `pid * input_row_stride` and
  writes it at `pid * output_row_stride` (the host-side one-program-per-row
  launch convention);
* `mask` — the active lanes `j < n_cols`, **the same for every program**: the
  row prefix that actually exists in the matrix. Inactive lanes (the padding
  of `BLOCK_SIZE = next_power_of_2(n_cols)`) carry no obligations on either
  side.

The windows and mask are declared, not parsed from the kernel; the headline
**proves** the kernel's actual addressing and masking match them. Buffer sizes
are not signature content: the headline quantifies over every allocation whose
extents cover the active lanes. -/
def squareIO (output_ptr input_ptr : RegionName)
    (input_row_stride output_row_stride n_cols BLOCK_SIZE : Nat) :
    MaskedKernelIO₁ where
  kernel := square_kernel output_ptr input_ptr input_row_stride
    output_row_stride n_cols BLOCK_SIZE
  inp := input_ptr
  out := output_ptr
  B := BLOCK_SIZE
  read := fun pid => pid * input_row_stride
  write := fun pid => pid * output_row_stride
  mask := fun _ j => j.val < n_cols

/-- **The headline**: `square_kernel` implements the lane-wise square
`xs i * xs i` on its masked IO signature — for every disjoint flat placement
of the two buffers, every program id whose active lanes are in bounds, and
every launch state whose active input-row lanes hold `xs`, the translated
pointer kernel terminates, every active output-row lane holds `xs i * xs i`,
and every other memory cell is unchanged. Proof:
`MaskedKernelIO₁.Implements.intro` assembles the region-model masked triple
with the flat-memory bridge side conditions. -/
specification square_kernel_correctness
    (output_ptr input_ptr : RegionName)
    (input_row_stride output_row_stride n_cols BLOCK_SIZE : Nat) :
    squareIO output_ptr input_ptr input_row_stride output_row_stride
        n_cols BLOCK_SIZE ⊨
      fun xs i => xs i * xs i := by
  refine MaskedKernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact square_kernel_flattenOk output_ptr input_ptr input_row_stride
      output_row_stride n_cols BLOCK_SIZE
  · intro bounds s h1 h2 _
    exact square_kernel_traceSafe output_ptr input_ptr input_row_stride
      output_row_stride n_cols BLOCK_SIZE bounds s h1 h2
  · intro s₀ xs hx
    obtain ⟨s1, hexec, hval, hframe⟩ := square_kernel_region_run output_ptr
      input_ptr input_row_stride output_row_stride n_cols BLOCK_SIZE s₀ xs hx
    -- scratch is empty, so its frame side condition is vacuous
    exact ⟨s1, hexec, hval, fun r o hout _ => hframe r o hout⟩

end VeriTile.Bench.TritonBenchG.SquareMatrix
