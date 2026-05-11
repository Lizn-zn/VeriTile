import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.RotaryTransformOps

open VeriTile.Triton

set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

/-- Proof-oriented one-row first-half slice of `rotary_transform_ops.py`'s
`rotary_kernel`.

This models the non-varlen, non-interleaved, non-conjugate branch for a single
row `pid_m`: `o0 = x0 * cos - x1 * sin`, followed by the first-half output
store. -/
def rotary_kernel_o0_row
    (OUT X COS SIN : RegionName)
    (SEQLEN_OFFSETS seqlen rotary_dim_half seqlen_ro
      stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim
      stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim
      BLOCK_M BLOCK_HALF : Nat) :
    ComputeKernel := triton {
  pid_m = tl.program_id(0)
  pid_batch = tl.program_id(1)
  pid_head = tl.program_id(2)
  rm = pid_m * $(BLOCK_M)
  rm_cs = rm + $(SEQLEN_OFFSETS)
  rk_half = tl.arange(0, $(BLOCK_HALF))
  x_base = X + pid_batch * $(stride_x_batch) + pid_head * $(stride_x_nheads)
  out_base = OUT + pid_batch * $(stride_out_batch) + pid_head * $(stride_out_nheads)
  cos = tl.load(COS + rm_cs * $(rotary_dim_half) + rk_half,
    mask=(rm_cs < $(seqlen_ro)) and (rk_half < $(rotary_dim_half)), other=1.0)
  sin = tl.load(SIN + rm_cs * $(rotary_dim_half) + rk_half,
    mask=(rm_cs < $(seqlen_ro)) and (rk_half < $(rotary_dim_half)), other=0.0)
  x0 = tl.load(x_base + rm * $(stride_x_seqlen) + rk_half * $(stride_x_headdim),
    mask=(rm < $(seqlen)) and (rk_half < $(rotary_dim_half)), other=0.0)
  x1 = tl.load(x_base + rm * $(stride_x_seqlen) +
      (rk_half + $(rotary_dim_half)) * $(stride_x_headdim),
    mask=(rm < $(seqlen)) and (rk_half < $(rotary_dim_half)), other=0.0)
  o0 = x0 * cos - x1 * sin
  tl.store(out_base + rm * $(stride_out_seqlen) + rk_half * $(stride_out_headdim),
    o0, mask=(rm < $(seqlen)) and (rk_half < $(rotary_dim_half)))
}

def rowIndex (s : BlockState) (BLOCK_M : Nat) : Nat :=
  s.pids 0 * BLOCK_M

def dimIndex (i : Fin BLOCK_HALF) : Nat :=
  i.val

def active (s : BlockState) (seqlen rotary_dim_half BLOCK_M : Nat)
    (i : Fin BLOCK_HALF) : Prop :=
  rowIndex s BLOCK_M < seqlen ∧ dimIndex i < rotary_dim_half

instance activeDecidable (s : BlockState) (seqlen rotary_dim_half BLOCK_M : Nat)
    (i : Fin BLOCK_HALF) :
    Decidable (active s seqlen rotary_dim_half BLOCK_M i) := by
  unfold active
  infer_instance

def outOffset
    (s : BlockState)
    (stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim
      BLOCK_M : Nat)
    (i : Fin BLOCK_HALF) : Nat :=
  s.pids 1 * stride_out_batch + s.pids 2 * stride_out_nheads +
    rowIndex s BLOCK_M * stride_out_seqlen + dimIndex i * stride_out_headdim

def x0Offset
    (s : BlockState)
    (stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim
      BLOCK_M : Nat)
    (i : Fin BLOCK_HALF) : Nat :=
  s.pids 1 * stride_x_batch + s.pids 2 * stride_x_nheads +
    rowIndex s BLOCK_M * stride_x_seqlen + dimIndex i * stride_x_headdim

def x1Offset
    (s : BlockState)
    (stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim
      rotary_dim_half BLOCK_M : Nat)
    (i : Fin BLOCK_HALF) : Nat :=
  s.pids 1 * stride_x_batch + s.pids 2 * stride_x_nheads +
    rowIndex s BLOCK_M * stride_x_seqlen +
    (dimIndex i + rotary_dim_half) * stride_x_headdim

def rotOffset (s : BlockState) (SEQLEN_OFFSETS rotary_dim_half BLOCK_M : Nat)
    (i : Fin BLOCK_HALF) : Nat :=
  (rowIndex s BLOCK_M + SEQLEN_OFFSETS) * rotary_dim_half + dimIndex i

noncomputable def rotaryO0Spec
    (s : BlockState) (X COS SIN : RegionName)
    (SEQLEN_OFFSETS seqlen_ro stride_x_batch stride_x_seqlen stride_x_nheads
      stride_x_headdim rotary_dim_half BLOCK_M : Nat)
    (i : Fin BLOCK_HALF) : ℝ :=
  let cosVal :=
    if rowIndex s BLOCK_M + SEQLEN_OFFSETS < seqlen_ro ∧
        dimIndex i < rotary_dim_half then
      s.readMem COS (rotOffset s SEQLEN_OFFSETS rotary_dim_half BLOCK_M i)
    else
      1.0
  let sinVal :=
    if rowIndex s BLOCK_M + SEQLEN_OFFSETS < seqlen_ro ∧
        dimIndex i < rotary_dim_half then
      s.readMem SIN (rotOffset s SEQLEN_OFFSETS rotary_dim_half BLOCK_M i)
    else
      0.0
  s.readMem X
      (x0Offset s stride_x_batch stride_x_seqlen stride_x_nheads
        stride_x_headdim BLOCK_M i) *
    cosVal -
  s.readMem X
      (x1Offset s stride_x_batch stride_x_seqlen stride_x_nheads
        stride_x_headdim rotary_dim_half BLOCK_M i) *
    sinVal

/-- Algorithm-layer correctness for the one-row `o0` rotary store. -/
theorem rotary_kernel_o0_row_correct
    (OUT X COS SIN : RegionName)
    (SEQLEN_OFFSETS seqlen rotary_dim_half seqlen_ro
      stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim
      stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim
      BLOCK_M BLOCK_HALF : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_HALF =>
        outOffset s stride_out_batch stride_out_seqlen stride_out_nheads
          stride_out_headdim BLOCK_M i))
    (hExec : exec (rotary_kernel_o0_row OUT X COS SIN SEQLEN_OFFSETS
        seqlen rotary_dim_half seqlen_ro stride_out_batch stride_out_seqlen
        stride_out_nheads stride_out_headdim stride_x_batch stride_x_seqlen
        stride_x_nheads stride_x_headdim BLOCK_M BLOCK_HALF) s = some s') :
    ∀ i : Fin BLOCK_HALF,
      s'.readMem OUT
          (outOffset s stride_out_batch stride_out_seqlen stride_out_nheads
            stride_out_headdim BLOCK_M i) =
        if active s seqlen rotary_dim_half BLOCK_M i then
          rotaryO0Spec s X COS SIN SEQLEN_OFFSETS seqlen_ro stride_x_batch
            stride_x_seqlen stride_x_nheads stride_x_headdim rotary_dim_half
            BLOCK_M i
        else
          s.readMem OUT
            (outOffset s stride_out_batch stride_out_seqlen stride_out_nheads
              stride_out_headdim BLOCK_M i) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_HALF] =>
        s.pids 1 * stride_out_batch + s.pids 2 * stride_out_nheads +
          s.pids 0 * BLOCK_M * stride_out_seqlen +
          idx.1.val * stride_out_headdim) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [outOffset, rowIndex, dimIndex, Nat.mul_assoc] using h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  by_cases hBH : 0 < BLOCK_HALF
  · simp [exec, rotary_kernel_o0_row, stepStmts, stepStmt, evalOp,
          Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
          NumericDType.add, NumericDType.mul, NumericDType.sub,
          ComparableDType.lt, hBH] at hExec
    rw [← hExec]
    simp only [outOffset, rowIndex, dimIndex]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hRawInj (i, PUnit.unit)]
    by_cases hRow : s.pids 0 * BLOCK_M < seqlen
    · by_cases hDim : i.val < rotary_dim_half
      · by_cases hRot : s.pids 0 * BLOCK_M + SEQLEN_OFFSETS < seqlen_ro
        · simp [active, rotaryO0Spec, outOffset, x0Offset, x1Offset, rotOffset,
                rowIndex, dimIndex, hRow, hDim, hRot, Option.map₂,
                Option.bind, Option.map]
        · simp [active, rotaryO0Spec, outOffset, x0Offset, x1Offset, rotOffset,
                rowIndex, dimIndex, hRow, hDim, hRot, Option.map₂,
                Option.bind, Option.map]
      · simp [active, rowIndex, dimIndex, hRow, hDim]
    · simp [active, rowIndex, dimIndex, hRow]
  · exact False.elim (hBH (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Compute-facing correctness for the one-row `o0` rotary store. -/
theorem rotary_kernel_o0_row_compute_correct
    (OUT X COS SIN : RegionName)
    (SEQLEN_OFFSETS seqlen rotary_dim_half seqlen_ro
      stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim
      stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim
      BLOCK_M BLOCK_HALF : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_HALF =>
        outOffset s stride_out_batch stride_out_seqlen stride_out_nheads
          stride_out_headdim BLOCK_M i)) :
    ComputeCorrect.Realizes
      (kernel := rotary_kernel_o0_row OUT X COS SIN SEQLEN_OFFSETS
        seqlen rotary_dim_half seqlen_ro stride_out_batch stride_out_seqlen
        stride_out_nheads stride_out_headdim stride_x_batch stride_x_seqlen
        stride_x_nheads stride_x_headdim BLOCK_M BLOCK_HALF)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_HALF => active s seqlen rotary_dim_half BLOCK_M i)
        (fun i => (OUT,
          outOffset s stride_out_batch stride_out_seqlen stride_out_nheads
            stride_out_headdim BLOCK_M i)))
      (expected := fun i =>
        rotaryO0Spec s X COS SIN SEQLEN_OFFSETS seqlen_ro stride_x_batch
          stride_x_seqlen stride_x_nheads stride_x_headdim rotary_dim_half
          BLOCK_M i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [rotary_kernel_o0_row]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := rotary_kernel_o0_row_correct OUT X COS SIN SEQLEN_OFFSETS
    seqlen rotary_dim_half seqlen_ro stride_out_batch stride_out_seqlen
    stride_out_nheads stride_out_headdim stride_x_batch stride_x_seqlen
    stride_x_nheads stride_x_headdim BLOCK_M BLOCK_HALF s s' hOutInj hExec i
  simpa [hActive] using h

end VeriTile.Bench.TritonBenchG.RotaryTransformOps
