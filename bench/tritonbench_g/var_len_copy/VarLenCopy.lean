import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.VarLenCopy

open VeriTile.Triton

set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

/-- Proof-oriented one-chunk slice of `var_len_copy.py`'s
`var_len_copy_kernel_triton`.

The Python kernel loops over chunks of a variable-length segment. This slice
fixes one chunk index and proves the masked vector copy from
`old_start + chunk * BLOCK_SIZE` to `new_start + chunk * BLOCK_SIZE`.

The in-body `tl.region` directive declares each int64 metadata buffer
(`old_a_len`, `old_a_start`, `new_a_start`) so loads from them recover
the `.nat` channel without explicit `dtype=` kwargs. -/
def var_len_copy_one_chunk
    (old_a_start old_a_len old_a_location new_a_start new_a_location : RegionName)
    (chunk BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  tl.region old_a_len = tl.uint64, old_a_start = tl.uint64, new_a_start = tl.uint64
  a_id = tl.program_id(0)
  length = tl.load(old_a_len + a_id)
  old_start = tl.load(old_a_start + a_id)
  new_start = tl.load(new_a_start + a_id)
  offset = tl.arange(0, $(BLOCK_SIZE))
  chunk_base = $(chunk) * $(BLOCK_SIZE)
  v = tl.load(old_a_location + old_start + chunk_base + offset,
    mask=chunk_base + offset < length, other=0.0)
  tl.store(new_a_location + new_start + chunk_base + offset, v,
    mask=chunk_base + offset < length)
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
      if chunk * BLOCK_SIZE + i.1.val < s.readMemValue .nat old_a_len (s.pids 0) then
        some (s.readMem old_a_location
          (s.readMemValue .nat old_a_start (s.pids 0) + chunk * BLOCK_SIZE + i.1.val))
      else some (0.0 : ℝ) }

def laneOffset (chunk BLOCK_SIZE : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  chunk * BLOCK_SIZE + i.val

def segmentLength (s : BlockState) (old_a_len : RegionName) : Nat :=
  s.readMemValue .nat old_a_len (s.pids 0)

def oldStart (s : BlockState) (old_a_start : RegionName) : Nat :=
  s.readMemValue .nat old_a_start (s.pids 0)

def newStart (s : BlockState) (new_a_start : RegionName) : Nat :=
  s.readMemValue .nat new_a_start (s.pids 0)

def active (s : BlockState) (old_a_len : RegionName) (chunk BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) : Prop :=
  laneOffset chunk BLOCK_SIZE i < segmentLength s old_a_len

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
            (if chunk * BLOCK_SIZE + idx.1.val <
                s.readMemValue .nat old_a_len (s.pids 0) then
              some (s.readMem old_a_location
                (s.readMemValue .nat old_a_start (s.pids 0) +
                  chunk * BLOCK_SIZE + idx.1.val))
            else some (0.0 : ℝ)))
        (P := fun idx : TileIndex [BLOCK_SIZE] =>
          chunk * BLOCK_SIZE + idx.1.val <
            s.readMemValue .nat old_a_len (s.pids 0))
        hRawInj (i, PUnit.unit))
    simp [preStoreState, BlockState.readMemValue] at hScatter
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

end VeriTile.Bench.TritonBenchG.VarLenCopy
