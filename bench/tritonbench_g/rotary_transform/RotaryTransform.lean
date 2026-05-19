import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.RotaryTransform

open VeriTile.Triton

set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

/-- Faithful DSL port of `rotary_transform.py`'s `rotary_kernel`.

Python's `SEQLEN_OFFSETS` argument is a union of scalar offset and tensor
pointer. The surface keeps `SEQLEN_OFFSETS` as the tensor region used by the
tensor-offset path and uses `SEQLEN_OFFSETS_SCALAR` for the scalar-offset path. -/
def rotary_kernel_surface
    (OUT X COS SIN : RegionName) (CU_SEQLENS SEQLEN_OFFSETS : Region .nat)
    (SEQLEN_OFFSETS_SCALAR seqlen nheads rotary_dim seqlen_ro CACHE_KEY_SEQLEN
      stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim
      stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim
      BLOCK_K BLOCK_M : Nat)
    (IS_SEQLEN_OFFSETS_TENSOR IS_VARLEN INTERLEAVED CONJUGATE : Bool) :
    ComputeKernel := triton {
  pid_m = tl.program_id(axis=0)
  pid_batch = tl.program_id(axis=1)
  pid_head = tl.program_id(axis=2)
  rotary_dim_half = rotary_dim // $(2)

  if not IS_VARLEN {
    X = X + pid_batch * $(stride_x_batch) + pid_head * $(stride_x_nheads)
    OUT = OUT + pid_batch * $(stride_out_batch) + pid_head * $(stride_out_nheads)
  } else {
    start_idx = tl.load(CU_SEQLENS + pid_batch)
    seqlen = tl.load(CU_SEQLENS + pid_batch + $(1)) - start_idx
    X = X + start_idx * $(stride_x_seqlen) + pid_head * $(stride_x_nheads)
    OUT = OUT + start_idx * $(stride_out_seqlen) + pid_head * $(stride_out_nheads)
  }

  if pid_m * $(BLOCK_M) >= seqlen {
    return
  }
  rm = pid_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  if not IS_SEQLEN_OFFSETS_TENSOR {
    rm_cs = rm + $(SEQLEN_OFFSETS_SCALAR)
  } else {
    rm_cs = rm + tl.load(SEQLEN_OFFSETS + pid_batch)
  }
  rk = tl.arange(0, $(BLOCK_K))
  rk_half = tl.arange(0, $(BLOCK_K) // $(2))

  if not INTERLEAVED {
    X = X + (rm[:, None] * $(stride_x_seqlen) +
      rk_half[None, :] * $(stride_x_headdim))
    COS = COS + (rm_cs[:, None] * rotary_dim_half + rk_half[None, :])
    SIN = SIN + (rm_cs[:, None] * rotary_dim_half + rk_half[None, :])
    cos = tl.load(COS,
      mask=(rm_cs[:, None] < $(seqlen_ro)) & (rk_half[None, :] < rotary_dim_half),
      other=1.0).to(tl.float32)
    sin = tl.load(SIN,
      mask=(rm_cs[:, None] < $(seqlen_ro)) & (rk_half[None, :] < rotary_dim_half),
      other=0.0).to(tl.float32)
    x0 = tl.load(X,
      mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half),
      other=0.0).to(tl.float32)
    x1 = tl.load(X + rotary_dim_half * $(stride_x_headdim),
      mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half),
      other=0.0).to(tl.float32)
    if CONJUGATE {
      sin = -sin
    }
    o0 = x0 * cos - x1 * sin
    o1 = x0 * sin + x1 * cos
    OUT = OUT + (rm[:, None] * $(stride_out_seqlen) +
      rk_half[None, :] * $(stride_out_headdim))
    tl.store(OUT, o0,
      mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half))
    tl.store(OUT + rotary_dim_half * $(stride_out_headdim), o1,
      mask=(rm[:, None] < seqlen) & (rk_half[None, :] < rotary_dim_half))
  } else {
    rk_swap = rk + ((rk + $(1)) % $(2)) * $(2) - $(1)
    rk_repeat = tl.arange(0, $(BLOCK_K)) // $(2)
    X0 = X + (rm[:, None] * $(stride_x_seqlen) + rk[None, :] * $(stride_x_headdim))
    X1 = X + (rm[:, None] * $(stride_x_seqlen) + rk_swap[None, :] * $(stride_x_headdim))
    COS = COS + (rm_cs[:, None] * rotary_dim_half + rk_repeat[None, :])
    SIN = SIN + (rm_cs[:, None] * rotary_dim_half + rk_repeat[None, :])
    cos = tl.load(COS,
      mask=(rm_cs[:, None] < $(seqlen_ro)) & (rk_repeat[None, :] < rotary_dim_half),
      other=1.0).to(tl.float32)
    sin = tl.load(SIN,
      mask=(rm_cs[:, None] < $(seqlen_ro)) & (rk_repeat[None, :] < rotary_dim_half),
      other=0.0).to(tl.float32)
    x0 = tl.load(X0,
      mask=(rm[:, None] < seqlen) & (rk[None, :] < rotary_dim),
      other=0.0).to(tl.float32)
    x1 = tl.load(X1,
      mask=(rm[:, None] < seqlen) & (rk_swap[None, :] < rotary_dim),
      other=0.0).to(tl.float32)
    if CONJUGATE {
      sin = -sin
    }
    x0_cos = x0 * cos
    x1_sin = x1 * sin
    out = tl.where(rk[None, :] % $(2) == $(0), x0_cos - x1_sin, x0_cos + x1_sin)
    OUT = OUT + (rm[:, None] * $(stride_out_seqlen) + rk[None, :] * $(stride_out_headdim))
    tl.store(OUT, out, mask=(rm[:, None] < seqlen) & (rk[None, :] < rotary_dim))
  }
}

/-- The full rotary-transform surface lowers to the algorithm layer, including
varlen/scalar-offset branches, interleaved mode, and conjugation. -/
theorem rotary_kernel_surface_toAlgorithm_supported
    (OUT X COS SIN : RegionName) (CU_SEQLENS SEQLEN_OFFSETS : Region .nat)
    (SEQLEN_OFFSETS_SCALAR seqlen nheads rotary_dim seqlen_ro CACHE_KEY_SEQLEN
      stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim
      stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim
      BLOCK_K BLOCK_M : Nat)
    (IS_SEQLEN_OFFSETS_TENSOR IS_VARLEN INTERLEAVED CONJUGATE : Bool) :
    ∃ alg, (rotary_kernel_surface OUT X COS SIN CU_SEQLENS SEQLEN_OFFSETS
      SEQLEN_OFFSETS_SCALAR seqlen nheads rotary_dim seqlen_ro CACHE_KEY_SEQLEN
      stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim
      stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim
      BLOCK_K BLOCK_M IS_SEQLEN_OFFSETS_TENSOR IS_VARLEN INTERLEAVED
      CONJUGATE).toAlgorithm? = Except.ok alg := by
  simp [rotary_kernel_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-- Surface transcription of the non-varlen, scalar-offset, non-interleaved,
non-conjugate branch of `rotary_transform.py`'s `rotary_kernel`.

This keeps the full `[BLOCK_M, BLOCK_K / 2]` tile shape of the Python branch and
writes both `o0` and `o1`; the proof below focuses on the one-row `o0`
projection of this surface. -/
def rotary_kernel_non_interleaved
    (OUT X COS SIN : RegionName)
    (SEQLEN_OFFSETS seqlen rotary_dim_half seqlen_ro
      stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim
      stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim
      BLOCK_M BLOCK_HALF : Nat) :
    ComputeKernel := triton {
    pid_m = tl.program_id(axis=0)
    pid_batch = tl.program_id(axis=1)
    pid_head = tl.program_id(axis=2)
    X = X + pid_batch * $(stride_x_batch) + pid_head * $(stride_x_nheads)
    OUT = OUT + pid_batch * $(stride_out_batch) + pid_head * $(stride_out_nheads)
    if pid_m * $(BLOCK_M) >= $(seqlen) {
      return
    }
    rm = pid_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
    rm_cs = rm + $(SEQLEN_OFFSETS)
    rk_half = tl.arange(0, $(BLOCK_HALF))
    X = X + (rm[:, None] * $(stride_x_seqlen) +
      rk_half[None, :] * $(stride_x_headdim))
    COS = COS + (rm_cs[:, None] * $(rotary_dim_half) + rk_half[None, :])
    SIN = SIN + (rm_cs[:, None] * $(rotary_dim_half) + rk_half[None, :])
    cos = tl.load(COS, mask=(rm_cs[:, None] < $(seqlen_ro)) &
      (rk_half[None, :] < $(rotary_dim_half)), other=1.0).to(tl.float32)
    sin = tl.load(SIN, mask=(rm_cs[:, None] < $(seqlen_ro)) &
      (rk_half[None, :] < $(rotary_dim_half)), other=0.0).to(tl.float32)
    x0 = tl.load(X, mask=(rm[:, None] < $(seqlen)) &
      (rk_half[None, :] < $(rotary_dim_half)), other=0.0).to(tl.float32)
    x1 = tl.load(X + $(rotary_dim_half) * $(stride_x_headdim),
      mask=(rm[:, None] < $(seqlen)) &
        (rk_half[None, :] < $(rotary_dim_half)), other=0.0).to(tl.float32)
    o0 = x0 * cos - x1 * sin
    o1 = x0 * sin + x1 * cos
    OUT = OUT + (rm[:, None] * $(stride_out_seqlen) +
      rk_half[None, :] * $(stride_out_headdim))
    tl.store(OUT, o0, mask=(rm[:, None] < $(seqlen)) &
      (rk_half[None, :] < $(rotary_dim_half)))
    tl.store(OUT + $(rotary_dim_half) * $(stride_out_headdim), o1,
      mask=(rm[:, None] < $(seqlen)) & (rk_half[None, :] < $(rotary_dim_half)))
}

/-- Proof-oriented one-row first-half slice of `rotary_transform.py`'s
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
  pid_m = tl.program_id(axis=0)
  pid_batch = tl.program_id(axis=1)
  pid_head = tl.program_id(axis=2)
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

/-- Proof-oriented one-row `o1` slice of `rotary_transform.py`'s
`rotary_kernel`.

Captures the second-half companion store `o1 = x0 * sin + x1 * cos` to offset
`out_base + rm * stride_out_seqlen + (rk_half + rotary_dim_half) *
stride_out_headdim`. -/
def rotary_kernel_o1_row
    (OUT X COS SIN : RegionName)
    (SEQLEN_OFFSETS seqlen rotary_dim_half seqlen_ro
      stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim
      stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim
      BLOCK_M BLOCK_HALF : Nat) :
    ComputeKernel := triton {
  pid_m = tl.program_id(axis=0)
  pid_batch = tl.program_id(axis=1)
  pid_head = tl.program_id(axis=2)
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
  o1 = x0 * sin + x1 * cos
  tl.store(out_base + rm * $(stride_out_seqlen) +
      (rk_half + $(rotary_dim_half)) * $(stride_out_headdim),
    o1, mask=(rm < $(seqlen)) and (rk_half < $(rotary_dim_half)))
}

def out1Offset
    (s : BlockState)
    (stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim
      rotary_dim_half BLOCK_M : Nat)
    (i : Fin BLOCK_HALF) : Nat :=
  s.pids 1 * stride_out_batch + s.pids 2 * stride_out_nheads +
    rowIndex s BLOCK_M * stride_out_seqlen +
    (dimIndex i + rotary_dim_half) * stride_out_headdim

noncomputable def rotaryO1Spec
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
    sinVal +
  s.readMem X
      (x1Offset s stride_x_batch stride_x_seqlen stride_x_nheads
        stride_x_headdim rotary_dim_half BLOCK_M i) *
    cosVal

/-- Algorithm-layer correctness for the one-row `o1` rotary store. -/
theorem rotary_kernel_o1_row_correct
    (OUT X COS SIN : RegionName)
    (SEQLEN_OFFSETS seqlen rotary_dim_half seqlen_ro
      stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim
      stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim
      BLOCK_M BLOCK_HALF : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_HALF =>
        out1Offset s stride_out_batch stride_out_seqlen stride_out_nheads
          stride_out_headdim rotary_dim_half BLOCK_M i))
    (hExec : exec (rotary_kernel_o1_row OUT X COS SIN SEQLEN_OFFSETS
        seqlen rotary_dim_half seqlen_ro stride_out_batch stride_out_seqlen
        stride_out_nheads stride_out_headdim stride_x_batch stride_x_seqlen
        stride_x_nheads stride_x_headdim BLOCK_M BLOCK_HALF) s = some s') :
    ∀ i : Fin BLOCK_HALF,
      s'.readMem OUT
          (out1Offset s stride_out_batch stride_out_seqlen stride_out_nheads
            stride_out_headdim rotary_dim_half BLOCK_M i) =
        if active s seqlen rotary_dim_half BLOCK_M i then
          rotaryO1Spec s X COS SIN SEQLEN_OFFSETS seqlen_ro stride_x_batch
            stride_x_seqlen stride_x_nheads stride_x_headdim rotary_dim_half
            BLOCK_M i
        else
          s.readMem OUT
            (out1Offset s stride_out_batch stride_out_seqlen stride_out_nheads
              stride_out_headdim rotary_dim_half BLOCK_M i) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_HALF] =>
        s.pids 1 * stride_out_batch + s.pids 2 * stride_out_nheads +
          s.pids 0 * BLOCK_M * stride_out_seqlen +
          (idx.1.val + rotary_dim_half) * stride_out_headdim) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [out1Offset, rowIndex, dimIndex, Nat.mul_assoc] using h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  by_cases hBH : 0 < BLOCK_HALF
  · simp [exec, rotary_kernel_o1_row, stepStmts, stepStmt, evalOp,
          Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
          NumericDType.add, NumericDType.mul, NumericDType.sub,
          ComparableDType.lt, hBH] at hExec
    rw [← hExec]
    simp only [out1Offset, rowIndex, dimIndex]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hRawInj (i, PUnit.unit)]
    by_cases hRow : s.pids 0 * BLOCK_M < seqlen
    · by_cases hDim : i.val < rotary_dim_half
      · by_cases hRot : s.pids 0 * BLOCK_M + SEQLEN_OFFSETS < seqlen_ro
        · simp [active, rotaryO1Spec, out1Offset, x0Offset, x1Offset, rotOffset,
                rowIndex, dimIndex, hRow, hDim, hRot, Option.map₂,
                Option.bind, Option.map]
        · simp [active, rotaryO1Spec, out1Offset, x0Offset, x1Offset, rotOffset,
                rowIndex, dimIndex, hRow, hDim, hRot, Option.map₂,
                Option.bind, Option.map]
      · simp [active, rowIndex, dimIndex, hRow, hDim]
    · simp [active, rowIndex, dimIndex, hRow]
  · exact False.elim (hBH (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Compute-facing correctness for the one-row `o1` rotary store. -/
theorem rotary_kernel_o1_row_compute_correct
    (OUT X COS SIN : RegionName)
    (SEQLEN_OFFSETS seqlen rotary_dim_half seqlen_ro
      stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim
      stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim
      BLOCK_M BLOCK_HALF : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_HALF =>
        out1Offset s stride_out_batch stride_out_seqlen stride_out_nheads
          stride_out_headdim rotary_dim_half BLOCK_M i)) :
    ComputeCorrect.Realizes
      (kernel := rotary_kernel_o1_row OUT X COS SIN SEQLEN_OFFSETS
        seqlen rotary_dim_half seqlen_ro stride_out_batch stride_out_seqlen
        stride_out_nheads stride_out_headdim stride_x_batch stride_x_seqlen
        stride_x_nheads stride_x_headdim BLOCK_M BLOCK_HALF)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_HALF => active s seqlen rotary_dim_half BLOCK_M i)
        (fun i => (OUT,
          out1Offset s stride_out_batch stride_out_seqlen stride_out_nheads
            stride_out_headdim rotary_dim_half BLOCK_M i)))
      (expected := fun i =>
        rotaryO1Spec s X COS SIN SEQLEN_OFFSETS seqlen_ro stride_x_batch
          stride_x_seqlen stride_x_nheads stride_x_headdim rotary_dim_half
          BLOCK_M i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [rotary_kernel_o1_row]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := rotary_kernel_o1_row_correct OUT X COS SIN SEQLEN_OFFSETS
    seqlen rotary_dim_half seqlen_ro stride_out_batch stride_out_seqlen
    stride_out_nheads stride_out_headdim stride_x_batch stride_x_seqlen
    stride_x_nheads stride_x_headdim BLOCK_M BLOCK_HALF s s' hOutInj hExec i
  simpa [hActive] using h

/-- Combined one-row first-and-second-half slice of `rotary_transform.py`'s
`rotary_kernel`.

Performs BOTH the `o0 = x0 * cos - x1 * sin` first-half store and the
`o1 = x0 * sin + x1 * cos` second-half store in a single kernel, matching
the non-varlen, non-interleaved, non-conjugate branch of the Python source. -/
def rotary_kernel_o0o1_row
    (OUT X COS SIN : RegionName)
    (SEQLEN_OFFSETS seqlen rotary_dim_half seqlen_ro
      stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim
      stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim
      BLOCK_M BLOCK_HALF : Nat) :
    ComputeKernel := triton {
  pid_m = tl.program_id(axis=0)
  pid_batch = tl.program_id(axis=1)
  pid_head = tl.program_id(axis=2)
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
  o1 = x0 * sin + x1 * cos
  tl.store(out_base + rm * $(stride_out_seqlen) + rk_half * $(stride_out_headdim),
    o0, mask=(rm < $(seqlen)) and (rk_half < $(rotary_dim_half)))
  tl.store(out_base + rm * $(stride_out_seqlen) +
      (rk_half + $(rotary_dim_half)) * $(stride_out_headdim),
    o1, mask=(rm < $(seqlen)) and (rk_half < $(rotary_dim_half)))
}

/-- Algorithm-layer correctness for the combined one-row `o0` + `o1` rotary
store, projected on the first-half (`o0`) output position. -/
theorem rotary_kernel_o0o1_row_o0_correct
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
    (hStrideHd : stride_out_headdim ≠ 0)
    (hHalfBound : BLOCK_HALF ≤ rotary_dim_half)
    (hExec : exec (rotary_kernel_o0o1_row OUT X COS SIN SEQLEN_OFFSETS
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
  -- Disjointness: the `o1` write at `(j.val + rotary_dim_half) * stride_headdim`
  -- never collides with the `o0` read offset `i.val * stride_headdim`, because
  -- `i.val < BLOCK_HALF ≤ rotary_dim_half ≤ j.val + rotary_dim_half`.
  have hDisjoint : ∀ k : TileIndex [BLOCK_HALF],
      s.pids 1 * stride_out_batch + s.pids 2 * stride_out_nheads +
          s.pids 0 * BLOCK_M * stride_out_seqlen +
          i.val * stride_out_headdim
        ≠
      s.pids 1 * stride_out_batch + s.pids 2 * stride_out_nheads +
          s.pids 0 * BLOCK_M * stride_out_seqlen +
          (k.1.val + rotary_dim_half) * stride_out_headdim := by
    intro k hEq
    have hMul : i.val * stride_out_headdim =
        (k.1.val + rotary_dim_half) * stride_out_headdim :=
      Nat.add_left_cancel hEq
    have hVal : i.val = k.1.val + rotary_dim_half :=
      Nat.eq_of_mul_eq_mul_right (Nat.pos_of_ne_zero hStrideHd) hMul
    have hI : i.val < rotary_dim_half :=
      Nat.lt_of_lt_of_le i.isLt hHalfBound
    have hGe : k.1.val + rotary_dim_half ≥ rotary_dim_half := Nat.le_add_left _ _
    omega
  by_cases hBH : 0 < BLOCK_HALF
  · simp [exec, rotary_kernel_o0o1_row, stepStmts, stepStmt, evalOp,
          Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
          NumericDType.add, NumericDType.mul, NumericDType.sub,
          ComparableDType.lt, hBH] at hExec
    rw [← hExec]
    simp only [outOffset, rowIndex, dimIndex]
    -- Strip the outer `o1` foldl: same region, disjoint offsets.
    rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
          (region := OUT)
          (P := fun idx : TileIndex [BLOCK_HALF] =>
            s.pids 0 * BLOCK_M < seqlen ∧ idx.1.val < rotary_dim_half)
          (offsetFn := fun idx : TileIndex [BLOCK_HALF] =>
            s.pids 1 * stride_out_batch + s.pids 2 * stride_out_nheads +
              s.pids 0 * BLOCK_M * stride_out_seqlen +
              (idx.1.val + rotary_dim_half) * stride_out_headdim)
          (hOff := fun k _ _ => hDisjoint k)]
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

/-- Algorithm-layer correctness for the combined one-row `o0` + `o1` rotary
store, projected on the second-half (`o1`) output position. -/
theorem rotary_kernel_o0o1_row_o1_correct
    (OUT X COS SIN : RegionName)
    (SEQLEN_OFFSETS seqlen rotary_dim_half seqlen_ro
      stride_out_batch stride_out_seqlen stride_out_nheads stride_out_headdim
      stride_x_batch stride_x_seqlen stride_x_nheads stride_x_headdim
      BLOCK_M BLOCK_HALF : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_HALF =>
        out1Offset s stride_out_batch stride_out_seqlen stride_out_nheads
          stride_out_headdim rotary_dim_half BLOCK_M i))
    (hStrideHd : stride_out_headdim ≠ 0)
    (hHalfBound : BLOCK_HALF ≤ rotary_dim_half)
    (hExec : exec (rotary_kernel_o0o1_row OUT X COS SIN SEQLEN_OFFSETS
        seqlen rotary_dim_half seqlen_ro stride_out_batch stride_out_seqlen
        stride_out_nheads stride_out_headdim stride_x_batch stride_x_seqlen
        stride_x_nheads stride_x_headdim BLOCK_M BLOCK_HALF) s = some s') :
    ∀ i : Fin BLOCK_HALF,
      s'.readMem OUT
          (out1Offset s stride_out_batch stride_out_seqlen stride_out_nheads
            stride_out_headdim rotary_dim_half BLOCK_M i) =
        if active s seqlen rotary_dim_half BLOCK_M i then
          rotaryO1Spec s X COS SIN SEQLEN_OFFSETS seqlen_ro stride_x_batch
            stride_x_seqlen stride_x_nheads stride_x_headdim rotary_dim_half
            BLOCK_M i
        else
          s.readMem OUT
            (out1Offset s stride_out_batch stride_out_seqlen stride_out_nheads
              stride_out_headdim rotary_dim_half BLOCK_M i) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_HALF] =>
        s.pids 1 * stride_out_batch + s.pids 2 * stride_out_nheads +
          s.pids 0 * BLOCK_M * stride_out_seqlen +
          (idx.1.val + rotary_dim_half) * stride_out_headdim) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [out1Offset, rowIndex, dimIndex, Nat.mul_assoc] using h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  -- Disjointness: the `o0` write at `j.val * stride_headdim` never collides
  -- with the `o1` read offset `(i.val + rotary_dim_half) * stride_headdim`.
  have hDisjoint : ∀ k : TileIndex [BLOCK_HALF],
      s.pids 1 * stride_out_batch + s.pids 2 * stride_out_nheads +
          s.pids 0 * BLOCK_M * stride_out_seqlen +
          (i.val + rotary_dim_half) * stride_out_headdim
        ≠
      s.pids 1 * stride_out_batch + s.pids 2 * stride_out_nheads +
          s.pids 0 * BLOCK_M * stride_out_seqlen +
          k.1.val * stride_out_headdim := by
    intro k hEq
    have hMul : (i.val + rotary_dim_half) * stride_out_headdim =
        k.1.val * stride_out_headdim :=
      Nat.add_left_cancel hEq
    have hVal : i.val + rotary_dim_half = k.1.val :=
      Nat.eq_of_mul_eq_mul_right (Nat.pos_of_ne_zero hStrideHd) hMul
    have hK : k.1.val < rotary_dim_half :=
      Nat.lt_of_lt_of_le k.1.isLt hHalfBound
    omega
  by_cases hBH : 0 < BLOCK_HALF
  · simp [exec, rotary_kernel_o0o1_row, stepStmts, stepStmt, evalOp,
          Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
          NumericDType.add, NumericDType.mul, NumericDType.sub,
          ComparableDType.lt, hBH] at hExec
    rw [← hExec]
    simp only [out1Offset, rowIndex, dimIndex]
    -- Outer foldl is the `o1` store; close via scatter_readback.
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hRawInj (i, PUnit.unit)]
    -- Now the `else` branch leaves a `(o0_foldl _).readMem OUT o1_offset`.
    -- Strip the `o0` foldl via the disjoint-offsets lemma.
    rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
          (region := OUT)
          (P := fun idx : TileIndex [BLOCK_HALF] =>
            s.pids 0 * BLOCK_M < seqlen ∧ idx.1.val < rotary_dim_half)
          (offsetFn := fun idx : TileIndex [BLOCK_HALF] =>
            s.pids 1 * stride_out_batch + s.pids 2 * stride_out_nheads +
              s.pids 0 * BLOCK_M * stride_out_seqlen +
              idx.1.val * stride_out_headdim)
          (hOff := fun k _ _ => hDisjoint k)]
    by_cases hRow : s.pids 0 * BLOCK_M < seqlen
    · by_cases hDim : i.val < rotary_dim_half
      · by_cases hRot : s.pids 0 * BLOCK_M + SEQLEN_OFFSETS < seqlen_ro
        · simp [active, rotaryO1Spec, out1Offset, x0Offset, x1Offset, rotOffset,
                rowIndex, dimIndex, hRow, hDim, hRot, Option.map₂,
                Option.bind, Option.map]
        · simp [active, rotaryO1Spec, out1Offset, x0Offset, x1Offset, rotOffset,
                rowIndex, dimIndex, hRow, hDim, hRot, Option.map₂,
                Option.bind, Option.map]
      · simp [active, rowIndex, dimIndex, hRow, hDim]
    · simp [active, rowIndex, dimIndex, hRow]
  · exact False.elim (hBH (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

end VeriTile.Bench.TritonBenchG.RotaryTransform
