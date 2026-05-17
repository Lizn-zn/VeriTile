import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.VarLenCopy

open VeriTile.Triton

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
  · simp [exec, var_len_copy_one_chunk, stepStmts, stepStmt, evalOp,
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
    ComputeCorrect.Realizes
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
  simp [exec, var_len_copy_kernel_triton, stepStmts, stepStmt, evalOp,
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
    ComputeCorrect.Realizes
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

end VeriTile.Bench.TritonBenchG.VarLenCopy
