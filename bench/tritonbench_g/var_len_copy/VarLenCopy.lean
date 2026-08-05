import VeriTile.Triton

/-!
# `var_len_copy` — strict per-kernel correctness

`var_len_copy_kernel_triton` is a variable-length segment copy: program `a_id`
reads the segment `length` and the `old_start` / `new_start` offsets from the
per-segment metadata buffers, then loops over `BLOCK_SIZE`-wide chunks of the
segment, copying `old_a_location[old_start + i + offset]` into
`new_a_location[new_start + i + offset]`, masked by `offset < length`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (grid over segments, the start/length metadata inputs,
and how the runtime composes per-segment copies into the destination buffer) is
the *trusted boundary*, not a proof obligation here. Because `a_id = pid` is
universally quantified, the per-program statement covers every program of the
grid.

## Proof architecture

```
var_len_copy_one_chunk_io_correctness                    ← TOP THEOREM (`⊨`, one chunk)
  ├─ var_len_one_chunk_flattenOk                           inside the flat-memory bridge
  ├─ var_len_one_chunk_traceSafe                           per-execution address safety
  └─ var_len_one_chunk_region_run                          region-model run
       ├─ var_len_one_chunk_terminates
       ├─ var_len_copy_one_chunk_correct                    per-lane readback (shared, below)
       ├─ destOffset_inj                                    output injectivity, discharged
       └─ var_len_one_chunk_frame                           cell-level frame

var_len_copy_kernel_triton_small_length_output_summary    per-write-map summary
  ├─ (toAlgorithm? = Except.ok _)                          surface lowers (incl. the for-loop)
  └─ var_len_copy_kernel_triton_small_length_compute_correct  ← ComputeCorrect over the copy
       └─ var_len_copy_kernel_triton_small_length_correct     algorithm-layer readback per lane
```
The per-chunk slice `var_len_copy_one_chunk_{correct,compute_correct}` proves a
single fixed chunk index; it is an independent proof that parallels (rather than
feeds) the single-iteration full-kernel proof. The `⊨` face above sits on that
slice, and additionally pins the flat-memory placement and the value-dependent
metadata window.

## Modeling boundary

Arithmetic/values are over `ℝ` (not bit-accurate IEEE float); the `int32`
metadata is typed as Nat regions (loads recover the `.nat` channel). This is a
**partial / blocked** verification: the full kernel is proved only under the
Python-tested **small-length regime** `0 < length ≤ BLOCK_SIZE` (lengths
50/150/200 with `BLOCK_SIZE = 256`), where the `range(0, length, BLOCK_SIZE)`
loop runs exactly once (`chunk = 0`); longer multi-chunk segments are not
covered by the full-kernel theorem (only by the per-chunk slice). The proofs
carry an `hOutInj` injectivity side condition on the destination offsets (no two
lanes of one program alias).
-/

namespace VeriTile.Bench.TritonBenchG.VarLenCopy

open VeriTile.Triton
open scoped VeriTile.Triton.Meta3MaskedTileKernelIO₁

set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

/-- Faithful transcription of `var_len_copy.py`'s `var_len_copy_kernel_triton`.

Allowed mechanical Lean-syntax-only change:
- The Python test creates the start/length metadata as `int32`; the Lean
  parameters type those metadata buffers as Nat regions so their `tl.load`
  calls do not need extra `dtype=` kwargs. -/
def var_len_copy_kernel_triton
    (old_a_start old_a_len : Region .nat) (old_a_location : RegionName)
    (new_a_start : Region .nat) (new_a_location : RegionName)
    (BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  a_id = tl.program_id(0)
  length = tl.load(old_a_len + a_id)
  old_start = tl.load(old_a_start + a_id)
  new_start = tl.load(new_a_start + a_id)
  old_offset = tl.arange(0, $(BLOCK_SIZE))
  new_offset = tl.arange(0, $(BLOCK_SIZE))
  for i in range($(0), length, $(BLOCK_SIZE)) {
    v = tl.load(old_a_location + old_start + i + old_offset,
      mask=old_offset < length)
    tl.store(new_a_location + new_start + i + new_offset, v,
      mask=new_offset < length)
  }
}

/-- Proof-oriented one-chunk slice of `var_len_copy.py`'s
`var_len_copy_kernel_triton`.

The Python kernel loops over chunks of a variable-length segment. This slice
fixes one chunk index and proves the masked vector copy from
`old_start + chunk * BLOCK_SIZE` to `new_start + chunk * BLOCK_SIZE`.

The metadata buffers are typed Nat regions so loads from them recover the
`.nat` channel without explicit `dtype=` kwargs. -/
def var_len_copy_one_chunk
    (old_a_start old_a_len : Region .nat) (old_a_location : RegionName)
    (new_a_start : Region .nat) (new_a_location : RegionName)
    (chunk BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  a_id = tl.program_id(0)
  length = tl.load(old_a_len + a_id)
  old_start = tl.load(old_a_start + a_id)
  new_start = tl.load(new_a_start + a_id)
  offset = tl.arange(0, $(BLOCK_SIZE))
  chunk_base = $(chunk) * $(BLOCK_SIZE)
  v = tl.load(old_a_location + old_start + chunk_base + offset,
    mask=offset < length)
  tl.store(new_a_location + new_start + chunk_base + offset, v,
    mask=offset < length)
}

def preStoreState
    (old_a_start old_a_len old_a_location new_a_start _new_a_location : RegionName)
    (chunk BLOCK_SIZE : Nat) (s : BlockState) : BlockState :=
  let s1 := s.setReg "a_id" TileDType.nat [] (Tile.scalar (s.pids 0))
  let s2 := s1.setReg "length" TileDType.nat []
    (Tile.scalar (s.readMemValue .nat old_a_len (s.pids 0)))
  let s3 := s2.setReg "old_start" TileDType.nat []
    (Tile.scalar (s.readMemValue .nat old_a_start (s.pids 0)))
  let s4 := s3.setReg "new_start" TileDType.nat []
    (Tile.scalar (s.readMemValue .nat new_a_start (s.pids 0)))
  let s5 := s4.setReg "offset" TileDType.nat [BLOCK_SIZE] (Tile.vec fun i => i.val)
  let s6 := s5.setReg "chunk_base" TileDType.nat []
    (Tile.scalar (chunk * BLOCK_SIZE))
  s6.setReg "v" TileDType.real [BLOCK_SIZE]
    { data := fun i =>
      if i.1.val < s.readMemValue .nat old_a_len (s.pids 0) then
        some (s.readMem old_a_location
          (s.readMemValue .nat old_a_start (s.pids 0) + chunk * BLOCK_SIZE + i.1.val))
      else some (s.undef old_a_location
          (s.readMemValue .nat old_a_start (s.pids 0) + chunk * BLOCK_SIZE + i.1.val)) }

def laneOffset (chunk BLOCK_SIZE : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  chunk * BLOCK_SIZE + i.val

def segmentLength (s : BlockState) (old_a_len : RegionName) : Nat :=
  s.readMemValue .nat old_a_len (s.pids 0)

def oldStart (s : BlockState) (old_a_start : RegionName) : Nat :=
  s.readMemValue .nat old_a_start (s.pids 0)

def newStart (s : BlockState) (new_a_start : RegionName) : Nat :=
  s.readMemValue .nat new_a_start (s.pids 0)

def active (s : BlockState) (old_a_len : RegionName) (_chunk BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) : Prop :=
  i.val < segmentLength s old_a_len

instance activeDecidable
    (s : BlockState) (old_a_len : RegionName) (chunk BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) :
    Decidable (active s old_a_len chunk BLOCK_SIZE i) := by
  unfold active
  infer_instance

def sourceOffset
    (s : BlockState) (old_a_start : RegionName) (chunk BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) : Nat :=
  oldStart s old_a_start + chunk * BLOCK_SIZE + i.val

def destOffset
    (s : BlockState) (new_a_start : RegionName) (chunk BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) : Nat :=
  newStart s new_a_start + chunk * BLOCK_SIZE + i.val

/-- Algorithm-layer correctness for the one-chunk variable-length copy. -/
theorem var_len_copy_one_chunk_correct
    (old_a_start old_a_len old_a_location new_a_start new_a_location : RegionName)
    (chunk BLOCK_SIZE : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => destOffset s new_a_start chunk BLOCK_SIZE i))
    (hExec : exec (var_len_copy_one_chunk old_a_start old_a_len old_a_location
        new_a_start new_a_location chunk BLOCK_SIZE) s = some s') :
    ∀ i : Fin BLOCK_SIZE,
      s'.readMem new_a_location (destOffset s new_a_start chunk BLOCK_SIZE i) =
        if active s old_a_len chunk BLOCK_SIZE i then
          s.readMem old_a_location (sourceOffset s old_a_start chunk BLOCK_SIZE i)
        else
          s.readMem new_a_location (destOffset s new_a_start chunk BLOCK_SIZE i) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE] =>
        s.readMemValue .nat new_a_start (s.pids 0) + chunk * BLOCK_SIZE +
          idx.1.val) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [destOffset, newStart, laneOffset] using h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  by_cases hBS : 0 < BLOCK_SIZE
  · simp [exec, var_len_copy_one_chunk, stepStmts, stepStmt, evalOp, evalOp.eq_def,
          Option.bind, Option.map, Tile.bop, Tile.ptrAdd, Tile.uop,
          NumericDType.add, NumericDType.mul, ComparableDType.lt,
          BlockState.readMemValue, hBS] at hExec
    rw [← hExec]
    have hScatter :=
      (BlockState.scatter_readback_prop_masked_nd
        (region := new_a_location)
        (shape := [BLOCK_SIZE])
        (s := preStoreState old_a_start old_a_len old_a_location
          new_a_start new_a_location chunk BLOCK_SIZE s)
        (offsetFn := fun idx : TileIndex [BLOCK_SIZE] =>
          s.readMemValue .nat new_a_start (s.pids 0) + chunk * BLOCK_SIZE +
            idx.1.val)
        (valueFn := fun idx : TileIndex [BLOCK_SIZE] =>
          WithBot.unbotD 0
            (if idx.1.val < s.readMemValue .nat old_a_len (s.pids 0) then
              some (s.readMem old_a_location
                (s.readMemValue .nat old_a_start (s.pids 0) +
                  chunk * BLOCK_SIZE + idx.1.val))
            else some (s.undef old_a_location
                (s.readMemValue .nat old_a_start (s.pids 0) +
                  chunk * BLOCK_SIZE + idx.1.val))))
        (P := fun idx : TileIndex [BLOCK_SIZE] =>
          idx.1.val < s.readMemValue .nat old_a_len (s.pids 0))
        hRawInj (i, PUnit.unit))
    simp [preStoreState, BlockState.readMemValue, Region.cast] at hScatter
    simp only [destOffset, sourceOffset, active, segmentLength, oldStart,
      newStart, laneOffset, BlockState.readMemValue]
    rw [hScatter]
    split <;> simp_all
  · exact False.elim (hBS (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Compute-facing correctness for the one-chunk variable-length copy. -/
theorem var_len_copy_one_chunk_compute_correct
    (old_a_start old_a_len old_a_location new_a_start new_a_location : RegionName)
    (chunk BLOCK_SIZE : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => destOffset s new_a_start chunk BLOCK_SIZE i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := var_len_copy_one_chunk old_a_start old_a_len old_a_location
        new_a_start new_a_location chunk BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => active s old_a_len chunk BLOCK_SIZE i)
        (fun i => (new_a_location, destOffset s new_a_start chunk BLOCK_SIZE i)))
      (expected := fun i =>
        s.readMem old_a_location (sourceOffset s old_a_start chunk BLOCK_SIZE i)) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [var_len_copy_one_chunk]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := var_len_copy_one_chunk_correct old_a_start old_a_len old_a_location
    new_a_start new_a_location chunk BLOCK_SIZE s s' hOutInj hExec i
  simpa [hActive] using h

/-! ## Full-kernel small-length proof

The Python test exercises `length ≤ BLOCK_SIZE` (lengths 50/150/200 with
`BLOCK_SIZE = 256`), so the `for i in range(0, length, BLOCK_SIZE)` loop runs
exactly once at `i = 0`. We use the `forRangeDyn_single_step` infrastructure
plus aggressive simp through the body to land on the same scatter-readback
form as `var_len_copy_one_chunk_correct` at `chunk = 0`. -/

/-- Pre-store state of `var_len_copy_kernel_triton` after the pre-loop binds
and one body iteration at `i = 0`, modulo the final memory writes. Used to
align the simp trace with `scatter_readback_prop_masked_nd`. -/
def fullKernelPreStoreState
    (old_a_start old_a_len old_a_location new_a_start _new_a_location : RegionName)
    (BLOCK_SIZE : Nat) (s : BlockState) : BlockState :=
  let s1 := s.setReg "a_id" TileDType.nat [] (Tile.scalar (s.pids 0))
  let s2 := s1.setReg "length" TileDType.nat []
    (Tile.scalar (s.readMemValue .nat old_a_len (s.pids 0)))
  let s3 := s2.setReg "old_start" TileDType.nat []
    (Tile.scalar (s.readMemValue .nat old_a_start (s.pids 0)))
  let s4 := s3.setReg "new_start" TileDType.nat []
    (Tile.scalar (s.readMemValue .nat new_a_start (s.pids 0)))
  let s5 := s4.setReg "old_offset" TileDType.nat [BLOCK_SIZE]
    (Tile.vec fun i => i.val)
  let s6 := s5.setReg "new_offset" TileDType.nat [BLOCK_SIZE]
    (Tile.vec fun i => i.val)
  let s7 := s6.setReg "i" TileDType.nat [] (Tile.scalar 0)
  s7.setReg "v" TileDType.real [BLOCK_SIZE]
    { data := fun i =>
      if i.1.val < s.readMemValue .nat old_a_len (s.pids 0) then
        some (s.readMem old_a_location
          (s.readMemValue .nat old_a_start (s.pids 0) + i.1.val))
      else some (s.undef old_a_location
          (s.readMemValue .nat old_a_start (s.pids 0) + i.1.val)) }

/-- Algorithm-layer correctness for `var_len_copy_kernel_triton` under the
Python-tested small-length regime `0 < length ≤ BLOCK_SIZE`. -/
theorem var_len_copy_kernel_triton_small_length_correct
    (old_a_start old_a_len : Region .nat) (old_a_location : RegionName)
    (new_a_start : Region .nat) (new_a_location : RegionName)
    (BLOCK_SIZE : Nat)
    (s s' : BlockState)
    (hBS : 0 < BLOCK_SIZE)
    (hLen :
      s.readMemValue .nat (Region.cast old_a_len : RegionName) (s.pids 0)
        ≤ BLOCK_SIZE)
    (hLenPos :
      0 < s.readMemValue .nat (Region.cast old_a_len : RegionName) (s.pids 0))
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE =>
        s.readMemValue .nat (Region.cast new_a_start : RegionName) (s.pids 0)
          + i.val))
    (hExec :
      exec (var_len_copy_kernel_triton old_a_start old_a_len old_a_location
        new_a_start new_a_location BLOCK_SIZE) s = some s') :
    ∀ i : Fin BLOCK_SIZE,
      s'.readMem new_a_location
          (s.readMemValue .nat (Region.cast new_a_start : RegionName) (s.pids 0)
            + i.val) =
        if i.val < s.readMemValue .nat (Region.cast old_a_len : RegionName) (s.pids 0) then
          s.readMem old_a_location
            (s.readMemValue .nat (Region.cast old_a_start : RegionName) (s.pids 0)
              + i.val)
        else
          s.readMem new_a_location
            (s.readMemValue .nat (Region.cast new_a_start : RegionName) (s.pids 0)
              + i.val) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE] =>
        s.readMemValue .nat (Region.cast new_a_start : RegionName) (s.pids 0)
          + idx.1.val) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa using h
    cases a; cases b
    simp only at hab
    cases hab; rfl
  have hBSne : BLOCK_SIZE ≠ 0 := Nat.pos_iff_ne_zero.mp hBS
  simp only [Region.cast] at hLen hLenPos hRawInj ⊢
  simp [exec, var_len_copy_kernel_triton, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.ptrAdd, Tile.uop,
        NumericDType.add, NumericDType.mul, ComparableDType.lt,
        hBS, hBSne, hLenPos, hLen,
        stepForRangeAux.forRangeDyn_unfold, stepForRangeAux.step_lt,
        stepForRangeAux.step_ge, Nat.zero_add, Region.cast] at hExec
  rw [← hExec]
  have hScatter :=
    (BlockState.scatter_readback_prop_masked_nd
      (region := new_a_location)
      (shape := [BLOCK_SIZE])
      (s := fullKernelPreStoreState
        (Region.cast old_a_start : RegionName)
        (Region.cast old_a_len : RegionName) old_a_location
        (Region.cast new_a_start : RegionName) new_a_location BLOCK_SIZE s)
      (offsetFn := fun idx : TileIndex [BLOCK_SIZE] =>
        s.readMemValue .nat (Region.cast new_a_start : RegionName) (s.pids 0)
          + idx.1.val)
      (valueFn := fun idx : TileIndex [BLOCK_SIZE] =>
        WithBot.unbotD 0
          (if idx.1.val < s.readMemValue .nat (Region.cast old_a_len : RegionName) (s.pids 0) then
            some (s.readMem old_a_location
              (s.readMemValue .nat (Region.cast old_a_start : RegionName) (s.pids 0) + idx.1.val))
          else some (s.undef old_a_location
              (s.readMemValue .nat (Region.cast old_a_start : RegionName) (s.pids 0) + idx.1.val))))
      (P := fun idx : TileIndex [BLOCK_SIZE] =>
        idx.1.val < s.readMemValue .nat (Region.cast old_a_len : RegionName) (s.pids 0))
      hRawInj (i, PUnit.unit))
  simp [fullKernelPreStoreState, Region.cast] at hScatter
  rw [hScatter]
  split <;> simp_all [BlockState.readMemValue]

/-- Compute-facing correctness for `var_len_copy_kernel_triton` under the
Python-tested small-length regime. Covers the test cases with
`length ∈ {50, 150, 200}` and `BLOCK_SIZE = 256`. -/
theorem var_len_copy_kernel_triton_small_length_compute_correct
    (old_a_start old_a_len : Region .nat) (old_a_location : RegionName)
    (new_a_start : Region .nat) (new_a_location : RegionName)
    (BLOCK_SIZE : Nat)
    (s : BlockState)
    (hBS : 0 < BLOCK_SIZE)
    (hLen :
      s.readMemValue .nat (Region.cast old_a_len : RegionName) (s.pids 0)
        ≤ BLOCK_SIZE)
    (hLenPos :
      0 < s.readMemValue .nat (Region.cast old_a_len : RegionName) (s.pids 0))
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE =>
        s.readMemValue .nat (Region.cast new_a_start : RegionName) (s.pids 0)
          + i.val)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := var_len_copy_kernel_triton old_a_start old_a_len old_a_location
        new_a_start new_a_location BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE =>
          i.val < s.readMemValue .nat (Region.cast old_a_len : RegionName) (s.pids 0))
        (fun i =>
          (new_a_location,
            s.readMemValue .nat (Region.cast new_a_start : RegionName) (s.pids 0)
              + i.val)))
      (expected := fun i =>
        s.readMem old_a_location
          (s.readMemValue .nat (Region.cast old_a_start : RegionName) (s.pids 0)
            + i.val)) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [var_len_copy_kernel_triton]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := var_len_copy_kernel_triton_small_length_correct
    old_a_start old_a_len old_a_location new_a_start new_a_location BLOCK_SIZE
    s s' hBS hLen hLenPos hOutInj hExec i
  simpa [hActive] using h

/-- Per-kernel output summary for `var_len_copy_kernel_triton` under the
Python-tested small-length regime `0 < length ≤ BLOCK_SIZE`: the DSL surface
lowers to the algorithm layer, and the masked segment copy to `new_a_location`
is compute-correct — every active lane (`< length`) holds the matching
`old_a_location` lane, out-of-segment lanes are preserved. -/
specification var_len_copy_kernel_triton_small_length_output_summary
    (old_a_start old_a_len : Region .nat) (old_a_location : RegionName)
    (new_a_start : Region .nat) (new_a_location : RegionName)
    (BLOCK_SIZE : Nat)
    (s : BlockState)
    (hBS : 0 < BLOCK_SIZE)
    (hLen :
      s.readMemValue .nat (Region.cast old_a_len : RegionName) (s.pids 0)
        ≤ BLOCK_SIZE)
    (hLenPos :
      0 < s.readMemValue .nat (Region.cast old_a_len : RegionName) (s.pids 0))
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE =>
        s.readMemValue .nat (Region.cast new_a_start : RegionName) (s.pids 0)
          + i.val)) :
    (∃ alg, (var_len_copy_kernel_triton old_a_start old_a_len old_a_location
        new_a_start new_a_location BLOCK_SIZE).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel := var_len_copy_kernel_triton old_a_start old_a_len old_a_location
        new_a_start new_a_location BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE =>
          i.val < s.readMemValue .nat (Region.cast old_a_len : RegionName) (s.pids 0))
        (fun i =>
          (new_a_location,
            s.readMemValue .nat (Region.cast new_a_start : RegionName) (s.pids 0)
              + i.val)))
      (expected := fun i =>
        s.readMem old_a_location
          (s.readMemValue .nat (Region.cast old_a_start : RegionName) (s.pids 0)
            + i.val)) := by
  refine ⟨?_, ?_⟩
  · simp [var_len_copy_kernel_triton, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  · exact var_len_copy_kernel_triton_small_length_compute_correct
      old_a_start old_a_len old_a_location new_a_start new_a_location BLOCK_SIZE
      s hBS hLen hLenPos hOutInj

/-! ## ════════ `⊨` IO face for the one-chunk copy ════════

The summary above is stated per *declared write map*. This section restates the
one-chunk slice on the audit-once IO surface
`Meta3MaskedTileKernelIO₁.Implements` (`⊨`), which additionally pins the **flat
memory** placement.

This is the variable-length genre: the kernel first loads three `.nat` scalars —
the segment `length`, the source base `old_start`, the destination base
`new_start` — and *then* uses them in the data window's address **and** in its
mask. No pid-only window can express that, which is why the metadata skin exists;
the three scalars are universally quantified in the statement and pinned by the
launch state, so the headline reads "for whatever the metadata says, the copy is
that copy". -/

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

/-- The destination window is injective outright — it is `base + lane` — so the
one-chunk readback's `hOutInj` precondition is a theorem here, not a hypothesis
the headline carries. -/
private theorem destOffset_inj (s : BlockState) (new_a_start : RegionName)
    (chunk BLOCK_SIZE : Nat) :
    Function.Injective
      (fun i : Fin BLOCK_SIZE => destOffset s new_a_start chunk BLOCK_SIZE i) := by
  intro a b h
  simp only [destOffset] at h
  exact Fin.ext (Nat.add_left_cancel h)

/-- The one-chunk slice sits inside the flat-memory bridge's covered fragment. -/
theorem var_len_one_chunk_flattenOk
    (old_a_start old_a_len old_a_location new_a_start new_a_location : RegionName)
    (chunk BLOCK_SIZE : Nat) :
    ((var_len_copy_one_chunk old_a_start old_a_len old_a_location new_a_start
      new_a_location chunk BLOCK_SIZE).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [var_len_copy_one_chunk, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, StmtList.FlattenOk,
    Stmt.FlattenOk, Op.FlattenOk]

/-- Termination of the one-chunk slice. -/
theorem var_len_one_chunk_terminates
    (old_a_start old_a_len old_a_location new_a_start new_a_location : RegionName)
    (chunk BLOCK_SIZE : Nat) (s : BlockState) :
    ∃ s1, exec (var_len_copy_one_chunk old_a_start old_a_len old_a_location
      new_a_start new_a_location chunk BLOCK_SIZE) s = some s1 := by
  simp [exec, var_len_copy_one_chunk, stepStmts, stepStmt, evalOp.eq_def,
    Option.bind, Option.map, Tile.bop, Tile.ptrAdd, Tile.uop, NumericDType.add,
    NumericDType.mul, ComparableDType.lt]

/-- Cell-level frame of the one-chunk slice. -/
theorem var_len_one_chunk_frame
    (old_a_start old_a_len old_a_location new_a_start new_a_location : RegionName)
    (chunk BLOCK_SIZE : Nat) (s s' : BlockState)
    (hExec : exec (var_len_copy_one_chunk old_a_start old_a_len old_a_location
      new_a_start new_a_location chunk BLOCK_SIZE) s = some s') :
    ∀ (r : RegionName) (o : Nat),
      (r ≠ new_a_location ∨ ∀ i : Fin BLOCK_SIZE,
        active s old_a_len chunk BLOCK_SIZE i →
        o ≠ destOffset s new_a_start chunk BLOCK_SIZE i) →
      s'.mem r o = s.mem r o := by
  intro r o hcond
  simp [exec, var_len_copy_one_chunk, stepStmts, stepStmt, evalOp.eq_def,
    Option.bind, Option.map, Tile.bop, Tile.ptrAdd, Tile.uop, NumericDType.add,
    NumericDType.mul, ComparableDType.lt] at hExec
  subst hExec
  rw [foldl_writeMem_frame (region := new_a_location)
    (fun i : TileIndex [BLOCK_SIZE] =>
      s.readMemValue .nat new_a_start (s.pids 0) + chunk * BLOCK_SIZE + i.1.val)
    _ (fun i : TileIndex [BLOCK_SIZE] =>
      i.1.val < s.readMemValue .nat old_a_len (s.pids 0)) r o
    (TileShape.allIndices [BLOCK_SIZE]) ?_]
  · simp [preStoreState]
  · rcases hcond with h | h
    · exact Or.inl h
    · exact Or.inr fun i _ hi => Ne.symm (h i.1 hi)

/-- Per-execution safety walk for the one-chunk slice. -/
theorem var_len_one_chunk_traceSafe
    (old_a_start old_a_len old_a_location new_a_start new_a_location : RegionName)
    (chunk BLOCK_SIZE : Nat) (bounds : RegionBounds) (s : BlockState)
    (hlen : s.pids 0 < bounds old_a_len)
    (hos : s.pids 0 < bounds old_a_start)
    (hns : s.pids 0 < bounds new_a_start)
    (hsrc : ∀ i : Fin BLOCK_SIZE, active s old_a_len chunk BLOCK_SIZE i →
      sourceOffset s old_a_start chunk BLOCK_SIZE i < bounds old_a_location)
    (hdst : ∀ i : Fin BLOCK_SIZE, active s old_a_len chunk BLOCK_SIZE i →
      destOffset s new_a_start chunk BLOCK_SIZE i < bounds new_a_location) :
    ((var_len_copy_one_chunk old_a_start old_a_len old_a_location new_a_start
      new_a_location chunk BLOCK_SIZE).toAlgKernel).TraceSafe bounds s := by
  simp [Kernel.TraceSafe, var_len_copy_one_chunk, Stmt.TraceSafeList,
    Stmt.TraceSafe, Op.SafeAt, MaskOpt.SafeAt, MaskOpt.Active,
    MaskOpt.MemorySafe, MemAccess.SafeAt, MemAccess.MemorySafe,
    memAccessMemorySafe, MemAccess.ActiveAddressSafe,
    memAccessActiveAddressSafe, Op.PointerAddressesSafeOn, Op.MemorySafe,
    stepStmts, stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map,
    Tile.bop, Tile.ptrAdd, Tile.uop, NumericDType.add, NumericDType.mul,
    ComparableDType.lt]
  exact ⟨hlen, hos, hns, fun a ha => hsrc a ha, fun a ha => hdst a ha⟩

/-- Region-model run of the one-chunk slice, in the shape
`Meta3MaskedTileKernelIO₁.Implements.intro` consumes. -/
theorem var_len_one_chunk_region_run
    (old_a_start old_a_len old_a_location new_a_start new_a_location : RegionName)
    (chunk BLOCK_SIZE : Nat) (s₀ : BlockState) (len olds news : Nat)
    (hlen : s₀.readMemValue .nat old_a_len s₀.pid = len)
    (hos : s₀.readMemValue .nat old_a_start s₀.pid = olds)
    (hns : s₀.readMemValue .nat new_a_start s₀.pid = news)
    (xs : TileIndex [BLOCK_SIZE] → ℝ)
    (hx : ∀ i : TileIndex [BLOCK_SIZE], i.1.val < len →
      s₀.readMem old_a_location (olds + chunk * BLOCK_SIZE + i.1.val) = xs i) :
    ∃ s1, exec (var_len_copy_one_chunk old_a_start old_a_len old_a_location
        new_a_start new_a_location chunk BLOCK_SIZE) s₀ = some s1
      ∧ (∀ i : TileIndex [BLOCK_SIZE], i.1.val < len →
          s1.readMem new_a_location (news + chunk * BLOCK_SIZE + i.1.val)
            = xs i)
      ∧ (∀ (r : RegionName) (o : Nat),
          (r ≠ new_a_location ∨ ∀ i : TileIndex [BLOCK_SIZE], i.1.val < len →
            o ≠ news + chunk * BLOCK_SIZE + i.1.val) →
          s1.mem r o = s₀.mem r o) := by
  obtain ⟨s1, hexec⟩ := var_len_one_chunk_terminates old_a_start old_a_len
    old_a_location new_a_start new_a_location chunk BLOCK_SIZE s₀
  refine ⟨s1, hexec, ?_, ?_⟩
  · intro i hi
    have h := var_len_copy_one_chunk_correct old_a_start old_a_len
      old_a_location new_a_start new_a_location chunk BLOCK_SIZE s₀ s1
      (destOffset_inj s₀ new_a_start chunk BLOCK_SIZE) hexec i.1
    rw [show destOffset s₀ new_a_start chunk BLOCK_SIZE i.1
          = news + chunk * BLOCK_SIZE + i.1.val from by
        simp only [destOffset, newStart, hns]] at h
    rw [h, if_pos (show active s₀ old_a_len chunk BLOCK_SIZE i.1 from by
        simpa [active, segmentLength, hlen] using hi),
      show sourceOffset s₀ old_a_start chunk BLOCK_SIZE i.1
          = olds + chunk * BLOCK_SIZE + i.1.val from by
        simp only [sourceOffset, oldStart, hos],
      hx i hi]
  · intro r o hcond
    refine var_len_one_chunk_frame old_a_start old_a_len old_a_location
      new_a_start new_a_location chunk BLOCK_SIZE s₀ s1 hexec r o ?_
    rcases hcond with h | h
    · exact Or.inl h
    · refine Or.inr fun i hact => ?_
      have hi : i.val < len := by simpa [active, segmentLength, hlen] using hact
      have := h (i, PUnit.unit) hi
      simpa [destOffset, newStart, hns] using this

/-- IO signature of the one-chunk slice on the **three-metadata** surface: the
segment length, the source base and the destination base are read at the
program's own cell, and both data windows plus the mask are functions of them. -/
def varLenOneChunkIO
    (old_a_start old_a_len old_a_location new_a_start new_a_location : RegionName)
    (chunk BLOCK_SIZE : Nat) : Meta3MaskedTileKernelIO₁ where
  kernel := var_len_copy_one_chunk old_a_start old_a_len old_a_location
    new_a_start new_a_location chunk BLOCK_SIZE
  mbuf1 := old_a_len
  mbuf2 := old_a_start
  mbuf3 := new_a_start
  inp := old_a_location
  out := new_a_location
  shape := [BLOCK_SIZE]
  mwin1 := fun pid => pid
  mwin2 := fun pid => pid
  mwin3 := fun pid => pid
  read := fun _pid _len olds _news i => olds + chunk * BLOCK_SIZE + i.1.val
  write := fun _pid _len _olds news i => news + chunk * BLOCK_SIZE + i.1.val
  mask := fun _pid len _olds _news i => i.1.val < len

/-! ### ════════ ★ MAIN THEOREM ★ ════════ -/

/-- **The headline on the IO surface** for `var_len_copy.py`'s
`var_len_copy_kernel_triton`, one-chunk slice: for every disjoint flat placement
of the five buffers, every program id whose metadata cells and active data lanes
are in bounds, **whatever** the three metadata scalars say, and every launch state
whose source window holds `xs`, the translated pointer kernel terminates, every
active destination lane holds the copied value `xs i`, and every other memory cell
is unchanged.

The three scalars are universally quantified and pinned by the launch state, so
the mask (`lane < length`) and both window bases (`old_start` / `new_start`) are
value-dependent — the shape the metadata skin exists for.

Dimension-general in `BLOCK_SIZE` and the chunk index, with **no**
side-condition: the destination window is `base + lane`, so the one-chunk
readback's output-injectivity precondition is discharged by `destOffset_inj`
rather than assumed. -/
specification var_len_copy_one_chunk_io_correctness
    (old_a_start old_a_len old_a_location new_a_start new_a_location : RegionName)
    (chunk BLOCK_SIZE : Nat) :
    varLenOneChunkIO old_a_start old_a_len old_a_location new_a_start
        new_a_location chunk BLOCK_SIZE
      ⊨ fun _pid _len _olds _news xs i => xs i := by
  refine Meta3MaskedTileKernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact var_len_one_chunk_flattenOk old_a_start old_a_len old_a_location
      new_a_start new_a_location chunk BLOCK_SIZE
  · intro bounds s len olds news hm1 hm2 hm3 hb1 hb2 hb3 hsrc hdst
    -- substituting the three pins turns the metadata scalars back into the
    -- `readMemValue` forms the per-slice lemma is phrased in, and `BlockState.pid`
    -- is reducibly `pids 0`, so the two phrasings are the same term
    subst hm1
    subst hm2
    subst hm3
    exact var_len_one_chunk_traceSafe old_a_start old_a_len old_a_location
      new_a_start new_a_location chunk BLOCK_SIZE bounds s hb1 hb2 hb3
      (fun i hact => hsrc (i, PUnit.unit) hact)
      (fun i hact => hdst (i, PUnit.unit) hact)
  · intro s₀ len olds news xs hm1 hm2 hm3 hx
    exact var_len_one_chunk_region_run old_a_start old_a_len old_a_location
      new_a_start new_a_location chunk BLOCK_SIZE s₀ len olds news hm1 hm2 hm3 xs
      (fun i hi => hx i hi)

end IOFace

end VeriTile.Bench.TritonBenchG.VarLenCopy
