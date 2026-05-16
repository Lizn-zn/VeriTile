import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems

open VeriTile.Triton

/-- Faithful transcription of `softmax_flaggems.py`'s
`softmax_kernel_non_inner`.

This keeps both the one-tile CTA path and the multi-tile online fallback for
the non-inner softmax dimension. -/
def softmax_kernel_non_inner_surface
    (output_ptr input_ptr : RegionName)
    (M N K TILE_N TILE_K : Nat)
    (ONE_TILE_PER_CTA : Bool) :
    ComputeKernel := triton {
  pid_k = tl.program_id(1)
  pid_m = tl.program_id(0)

  k_offsets = pid_k * $(TILE_K) + tl.arange(0, $(TILE_K))

  if ONE_TILE_PER_CTA {
    n_offsets = tl.arange(0, $(TILE_N))
    offset = pid_m * $(N) * $(K) + n_offsets[:, None] * $(K) + k_offsets
    mask = (n_offsets[:, None] < $(N)) & (k_offsets < $(K))
    input_ptrs = input_ptr + offset
    inp = tl.load(input_ptrs, mask=mask, other=-float("inf"))
    m = tl.max(inp, 0)
    e = tl.exp(inp - m[None, :])
    z = tl.sum(e, 0)
    out = e / z
    output_ptrs = output_ptr + offset
    tl.store(output_ptrs, out, mask=mask)
  } else {
    m = tl.full([$(TILE_N), $(TILE_K)], value=float("-inf"), dtype=tl.float32)
    z = tl.full([$(TILE_N), $(TILE_K)], value=0.0, dtype=tl.float32)

    for start_n in range($(0), $(N), $(TILE_N)) {
      n_offsets = start_n + tl.arange(0, $(TILE_N))
      offsets = pid_m * $(N) * $(K) + n_offsets[:, None] * $(K) + k_offsets
      mask = (n_offsets[:, None] < $(N)) & (k_offsets < $(K))
      inp = tl.load(input_ptr + offsets, mask=mask, other=-float("inf"))
      m_new = tl.maximum(m, inp)
      alpha = tl.exp(m - m_new)
      z = z * alpha + tl.exp(inp - m_new)
      m = m_new
    }

    m_reduced = tl.max(m, 0)
    z = tl.sum(z * tl.exp(m - m_reduced[None, :]), 0)
    m = m_reduced

    previous_multiple = $((N + TILE_N - 1)) // $(TILE_N) * $(TILE_N) - $(TILE_N)
    for start_n in range($(0), $(N), $(TILE_N)) {
      n_offsets = (previous_multiple - start_n) + tl.arange(0, $(TILE_N))
      offsets = pid_m * $(N) * $(K) + n_offsets[:, None] * $(K) + k_offsets
      mask = (n_offsets[:, None] < $(N)) & (k_offsets[None, :] < $(K))
      inp = tl.load(input_ptr + offsets, mask=mask, other=-float("inf"))
      o = tl.exp(inp - m[None, :]) / z[None, :]
      tl.store(output_ptr + offsets, o, mask=mask)
    }
  }
}

/-- Proof-oriented specialization of `softmax_kernel_non_inner` to
`ONE_TILE_PER_CTA=true`.

This covers the `K > 1` forward path used when the softmax dimension is not the
innermost physical dimension: each CTA handles one `(m, k-block)`, loads the
`[TILE_N, TILE_K]` tile, reduces along `N`, and stores the normalized tile.
The multi-tile fallback's `tl.float32` running `m/z` state is outside this
specialized surface. -/
def softmax_kernel_non_inner_one_tile_surface
    (output_ptr input_ptr : RegionName)
    (N K TILE_N TILE_K : Nat) :
    ComputeKernel := triton {
  pid_k = tl.program_id(1)
  pid_m = tl.program_id(0)
  k_offsets = pid_k * $(TILE_K) + tl.arange(0, $(TILE_K))
  n_offsets = tl.arange(0, $(TILE_N))
  offset = pid_m * $(N) * $(K) + n_offsets[:, None] * $(K) + k_offsets
  mask = (n_offsets[:, None] < $(N)) & (k_offsets < $(K))
  input_ptrs = input_ptr + offset
  inp = tl.load(input_ptrs, mask=mask, other=-float("inf"))
  m = tl.max(inp, 0)
  e = tl.exp(inp - m[None, :])
  z = tl.sum(e, 0)
  out = e / z
  output_ptrs = output_ptr + offset
  tl.store(output_ptrs, out, mask=mask)
}

/-- Proof-oriented `ONE_TILE_PER_CTA=true` slice of
`softmax_flaggems.py`'s `softmax_kernel_inner`.

This covers the inner-dimension fast path where one CTA covers a full row. It
preserves the source kernel's row program id, masked load with `-inf`, stable
softmax max/sum normalization, output-dtype load cast, and masked store. The
multi-tile online fallback remains future work. -/
def softmax_kernel_inner_one_tile
    (output_ptr input_ptr : RegionName)
    (N TILE_N : Nat) :
    ComputeKernel := triton {
  pid_m = tl.program_id(0)
  n_offsets = tl.arange(0, $(TILE_N))
  offset = pid_m * $(N) + n_offsets
  mask = n_offsets < $(N)
  input_ptrs = input_ptr + offset
  inp = (tl.load(input_ptrs, mask=mask, other=-float("inf"))).to(output_ptr.dtype.element_ty)
  m = tl.max(inp, 0)
  e = tl.exp(inp - m)
  z = tl.sum(e, 0)
  out = e / z
  output_ptrs = output_ptr + offset
  tl.store(output_ptrs, out, mask=mask)
}

/-- Surface transcription of `softmax_flaggems.py`'s
`softmax_backward_kernel_non_inner`, specialized to `ONE_TILE_PER_CTA=true`.

This preserves the non-inner `(m, k-block)` addressing and the standard softmax
VJP formula `out * (out_grad - sum(out * out_grad, axis=0))`. -/
def softmax_backward_kernel_non_inner_one_tile_surface
    (out_ptr out_grad_ptr in_grad_ptr : RegionName)
    (N K TILE_N TILE_K : Nat) :
    ComputeKernel := triton {
  pid_m = tl.program_id(0)
  pid_k = tl.program_id(1)
  offsets_k = pid_k * $(TILE_K) + tl.arange(0, $(TILE_K))
  offsets_n = tl.arange(0, $(TILE_N))
  offsets = pid_m * $(N) * $(K) + offsets_n[:, None] * $(K) + offsets_k
  mask = (offsets_n < $(N))[:, None] & (offsets_k < $(K))
  out_tile = tl.load(out_ptr + offsets, mask=mask)
  out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask)
  scale = tl.sum(out_tile * out_grad_tile, axis=0)
  in_grad_tile = out_tile * (out_grad_tile - scale[None, :])
  tl.store(in_grad_ptr + offsets, in_grad_tile, mask=mask)
}

/-- Surface transcription of `softmax_flaggems.py`'s
`softmax_backward_kernel_inner`, specialized to `ONE_TILE_PER_CTA=true`.

This is the contiguous-row backward path: a CTA handles `[TILE_M, TILE_N]`,
reduces along `N`, and writes the masked input-gradient tile. -/
def softmax_backward_kernel_inner_one_tile_surface
    (out_ptr out_grad_ptr in_grad_ptr : RegionName)
    (M N TILE_M TILE_N : Nat) :
    ComputeKernel := triton {
  pid_m = tl.program_id(0)
  m_offsets = pid_m * $(TILE_M) + tl.arange(0, $(TILE_M))
  n_offsets = tl.arange(0, $(TILE_N))
  offsets = m_offsets[:, None] * $(N) + n_offsets
  mask = (m_offsets[:, None] < $(M)) & (n_offsets < $(N))
  out_tile = tl.load(out_ptr + offsets, mask=mask)
  out_grad_tile = tl.load(out_grad_ptr + offsets, mask=mask)
  scale = tl.sum(out_tile * out_grad_tile, 1)
  in_grad_tile = out_tile * (out_grad_tile - scale[:, None])
  tl.store(in_grad_ptr + offsets, in_grad_tile, mask=mask)
}

def outOffset (s : BlockState) (N : Nat) (i : Fin TILE_N) : Nat :=
  s.pid * N + i.val

noncomputable def softmaxFlaggemsInputTile
    (s : BlockState) (input_ptr : RegionName) (N TILE_N : Nat) :
    Tile .real [TILE_N] :=
  { data := fun idx =>
      if idx.1.val < N then
        some (s.readMem input_ptr (outOffset s N idx.1))
      else none }

noncomputable def softmaxFlaggemsSpec
    (s : BlockState) (input_ptr : RegionName)
    (N TILE_N : Nat) (idx : Fin TILE_N) : ℝ :=
  let row := softmaxFlaggemsInputTile s input_ptr N TILE_N
  match Tile.reduceMax (shape := [TILE_N]) ⟨0, by simp⟩ Bool.false row with
  | some rowMax =>
      let shifted := Tile.bop (NumericDType.sub .real) Broadcast.scalarR row rowMax
      let e := Tile.uop WithBot.realExp shifted
      let z := Tile.reduceSum (shape := [TILE_N]) ⟨0, by simp⟩ Bool.false e
      WithBot.unbotD 0
        ((Tile.bop (NumericDType.div .real) Broadcast.scalarR e z).data
          (idx, PUnit.unit))
  | none => 0

/-- Algorithm-layer correctness for the inner one-tile FlagGems softmax. -/
theorem softmax_kernel_inner_one_tile_correct
    (output_ptr input_ptr : RegionName)
    (N TILE_N : Nat)
    (s s' : BlockState)
    (hExec : exec (softmax_kernel_inner_one_tile output_ptr input_ptr N TILE_N) s =
        some s') :
    ∀ i : Fin TILE_N,
      s'.readMem output_ptr (outOffset s N i) =
        if i.val < N then
          softmaxFlaggemsSpec s input_ptr N TILE_N i
        else s.readMem output_ptr (outOffset s N i) := by
  intro i
  by_cases hT : 0 < TILE_N
  · simp [exec, softmax_kernel_inner_one_tile, stepStmts, stepStmt, evalOp,
          Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
          Tile.reduceMax, Tile.reduceMaxDrop, Tile.reduceSum, Tile.reduceSumDrop,
          TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
          NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
          ComparableDType.lt, hT] at hExec
    subst s'
    simp [BlockState.pid_eq, outOffset]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
          (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
    by_cases hi : i.val < N
    · simp [hi, softmaxFlaggemsSpec, softmaxFlaggemsInputTile, outOffset,
            Tile.reduceMax, Tile.reduceMaxDrop, Tile.reduceSum,
            Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis,
            TileShape.insertAxisIndex, hT]
      congr
    · simp [hi]
  · exact False.elim (hT (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Compute-facing correctness for the inner one-tile FlagGems softmax. -/
theorem softmax_kernel_inner_one_tile_compute_correct
    (output_ptr input_ptr : RegionName)
    (N TILE_N : Nat)
    (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := softmax_kernel_inner_one_tile output_ptr input_ptr N TILE_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin TILE_N => i.val < N)
        (fun i => (output_ptr, outOffset s N i)))
      (expected := fun i => softmaxFlaggemsSpec s input_ptr N TILE_N i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [softmax_kernel_inner_one_tile]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := softmax_kernel_inner_one_tile_correct output_ptr input_ptr N TILE_N
    s s' hExec i
  simpa [hActive] using h

/-! ## Non-inner one-tile forward — store-side proof skeleton

This block provides offset/value defs and a non-trivial `_writes_at_idx`
lemma showing that, for in-range lanes, the kernel writes exactly the
softmax-of-column value at every `(n, k)` lane. The full
`ComputeCorrect.Realizes` lift to a math-level spec for the 2D non-inner
softmax is left as future work — the underlying difficulty is the per-column
streaming reduction along axis 0 with a precomputed-tile dependence. -/

def nonInnerOffset (s : BlockState) (N K TILE_K : Nat)
    (idx : TileIndex [TILE_N, TILE_K]) : Nat :=
  s.pids 0 * N * K + idx.1.val * K + (s.pids 1 * TILE_K + idx.2.1.val)

/-- Offset injectivity for the 2D non-inner store under `TILE_K ≤ K` and
`s.pids 1 * TILE_K + TILE_K ≤ K`. -/
theorem nonInnerOffset_injective {N K TILE_N TILE_K : Nat}
    (s : BlockState) (hRange : s.pids 1 * TILE_K + TILE_K ≤ K) :
    Function.Injective (fun idx : TileIndex [TILE_N, TILE_K] =>
      nonInnerOffset s N K TILE_K idx) := by
  rintro ⟨⟨a1, ha1⟩, ⟨a2, ha2⟩, _⟩ ⟨⟨b1, hb1⟩, ⟨b2, hb2⟩, _⟩ h
  simp only [nonInnerOffset] at h
  -- The K-column-stride bounds: `a2 < TILE_K`, so `(pid1*TILE_K + a2) < K`.
  have ha2K : s.pids 1 * TILE_K + a2 < K :=
    lt_of_lt_of_le (Nat.add_lt_add_left ha2 _) hRange
  have hb2K : s.pids 1 * TILE_K + b2 < K :=
    lt_of_lt_of_le (Nat.add_lt_add_left hb2 _) hRange
  have hK_pos : 0 < K := lt_of_le_of_lt (Nat.zero_le _) ha2K
  -- Massage to: base + a1 * K + a2' = base + b1 * K + b2' with a2', b2' < K.
  -- Let base = s.pids 0 * N * K.
  set base := s.pids 0 * N * K
  -- h : base + a1 * K + (pids1*TILE_K + a2) = base + b1 * K + (pids1*TILE_K + b2)
  have hsum : a1 * K + (s.pids 1 * TILE_K + a2) =
      b1 * K + (s.pids 1 * TILE_K + b2) := by
    have hh : base + (a1 * K + (s.pids 1 * TILE_K + a2)) =
        base + (b1 * K + (s.pids 1 * TILE_K + b2)) := by
      simp only [Nat.add_assoc] at h ⊢
      exact h
    exact Nat.add_left_cancel hh
  -- div by K: a1 = b1
  have ha_div : (a1 * K + (s.pids 1 * TILE_K + a2)) / K = a1 := by
    rw [Nat.add_comm (a1 * K) _, Nat.mul_comm a1 K,
        Nat.add_mul_div_left _ _ hK_pos, Nat.div_eq_of_lt ha2K, Nat.zero_add]
  have hb_div : (b1 * K + (s.pids 1 * TILE_K + b2)) / K = b1 := by
    rw [Nat.add_comm (b1 * K) _, Nat.mul_comm b1 K,
        Nat.add_mul_div_left _ _ hK_pos, Nat.div_eq_of_lt hb2K, Nat.zero_add]
  have h1 : a1 = b1 := by rw [← ha_div, hsum, hb_div]
  -- mod by K: (pid1*TILE_K + a2) = (pid1*TILE_K + b2)
  have ha_mod : (a1 * K + (s.pids 1 * TILE_K + a2)) % K = s.pids 1 * TILE_K + a2 := by
    rw [Nat.add_comm (a1 * K) _, Nat.mul_comm a1 K,
        Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt ha2K]
  have hb_mod : (b1 * K + (s.pids 1 * TILE_K + b2)) % K = s.pids 1 * TILE_K + b2 := by
    rw [Nat.add_comm (b1 * K) _, Nat.mul_comm b1 K,
        Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hb2K]
  have hcol : s.pids 1 * TILE_K + a2 = s.pids 1 * TILE_K + b2 := by
    rw [← ha_mod, hsum, hb_mod]
  have h2 : a2 = b2 := Nat.add_left_cancel hcol
  subst h1; subst h2; rfl

/-- Per-`(n, k)` input tile for the 2D non-inner softmax: along the `n`
axis the column at fixed `k` ranges over `n ∈ [0, N)`; out-of-range lanes
default to `-∞` to match `tl.load(..., other=-float("inf"))`. -/
noncomputable def softmaxFlaggemsNonInnerInputTile
    (s : BlockState) (input_ptr : RegionName) (N K TILE_N TILE_K : Nat) :
    Tile .real [TILE_N, TILE_K] :=
  { data := fun idx =>
      if idx.1.val < N ∧ s.pids 1 * TILE_K + idx.2.1.val < K then
        some (s.readMem input_ptr (nonInnerOffset s N K TILE_K idx))
      else ⊥ }

/-- Spec for the non-inner one-tile softmax output at lane `(n, k)`:
softmax along the `n` axis (axis 0) at column `k`. -/
noncomputable def softmaxFlaggemsNonInnerSpec
    (s : BlockState) (input_ptr : RegionName)
    (N K TILE_N TILE_K : Nat) (idx : TileIndex [TILE_N, TILE_K]) : ℝ :=
  let row := softmaxFlaggemsNonInnerInputTile s input_ptr N K TILE_N TILE_K
  match Tile.reduceMax (shape := [TILE_N, TILE_K]) ⟨0, by simp⟩ Bool.false row with
  | some colMax =>
      let bc : Broadcast [TILE_N, TILE_K] [TILE_K] [TILE_N, TILE_K] :=
        Broadcast.leadR (Broadcast.consSame Broadcast.nil)
      let shifted := Tile.bop (NumericDType.sub .real) bc row colMax
      let e := Tile.uop WithBot.realExp shifted
      let z := Tile.reduceSum (shape := [TILE_N, TILE_K]) ⟨0, by simp⟩ Bool.false e
      WithBot.unbotD 0
        ((Tile.bop (NumericDType.div .real) bc e z).data idx)
  | none => 0

/-- Algorithm-layer correctness for the non-inner one-tile FlagGems softmax.

Uses `nonInnerOffset_injective` for the masked-store readback and lifts the
1D `softmax_kernel_inner_one_tile_correct` template to the 2D `[TILE_N, TILE_K]`
non-inner shape. The reduction axis is 0 (the `n` dimension); each `(n, k)`
output is the softmax along its column. -/
theorem softmax_kernel_non_inner_one_tile_correct
    (output_ptr input_ptr : RegionName)
    (N K TILE_N TILE_K : Nat)
    (s s' : BlockState)
    (hRange : s.pids 1 * TILE_K + TILE_K ≤ K)
    (hExec : exec (softmax_kernel_non_inner_one_tile_surface
        output_ptr input_ptr N K TILE_N TILE_K) s = some s') :
    ∀ idx : TileIndex [TILE_N, TILE_K],
      s'.readMem output_ptr (nonInnerOffset s N K TILE_K idx) =
        if idx.1.val < N ∧ s.pids 1 * TILE_K + idx.2.1.val < K then
          softmaxFlaggemsNonInnerSpec s input_ptr N K TILE_N TILE_K idx
        else s.readMem output_ptr (nonInnerOffset s N K TILE_K idx) := by
  intro idx
  by_cases hN : 0 < TILE_N
  · by_cases hK : 0 < TILE_K
    · simp [exec, softmax_kernel_non_inner_one_tile_surface,
            stepStmts, stepStmt, evalOp, Option.bind,
            Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
            Tile.reduceMax, Tile.reduceMaxDrop,
            Tile.reduceSum, Tile.reduceSumDrop,
            TileShape.axisDim, TileShape.eraseAxis,
            TileShape.insertAxisIndex,
            NumericDType.add, NumericDType.mul, NumericDType.sub,
            NumericDType.div, ComparableDType.lt, hN, hK] at hExec
      rw [← hExec]
      -- Restate the writeMem offset in terms of `nonInnerOffset` so the
      -- scatter readback can fire.
      simp only [show ∀ idx : TileIndex [TILE_N, TILE_K],
          s.pids 0 * N * K + idx.1.val * K + (s.pids 1 * TILE_K + idx.2.1.val) =
            nonInnerOffset s N K TILE_K idx from fun _ => rfl]
      rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
            (nonInnerOffset_injective s hRange) idx]
      by_cases hmask : idx.1.val < N ∧ s.pids 1 * TILE_K + idx.2.1.val < K
      · simp [hmask, softmaxFlaggemsNonInnerSpec,
              softmaxFlaggemsNonInnerInputTile, nonInnerOffset,
              Tile.reduceMax, Tile.reduceMaxDrop,
              Tile.reduceSum, Tile.reduceSumDrop,
              TileShape.axisDim, TileShape.eraseAxis,
              TileShape.insertAxisIndex,
              NumericDType.div, NumericDType.sub,
              hN, hmask.2]
        rfl
      · simp [hmask]
    · exact False.elim (hK (Nat.lt_of_le_of_lt (Nat.zero_le _) idx.2.1.isLt))
  · exact False.elim (hN (Nat.lt_of_le_of_lt (Nat.zero_le _) idx.1.isLt))

/-- Compute-facing correctness for the non-inner one-tile FlagGems softmax. -/
theorem softmax_kernel_non_inner_one_tile_compute_correct
    (output_ptr input_ptr : RegionName)
    (N K TILE_N TILE_K : Nat)
    (s : BlockState)
    (hRange : s.pids 1 * TILE_K + TILE_K ≤ K) :
    ComputeCorrect.Realizes
      (kernel := softmax_kernel_non_inner_one_tile_surface
        output_ptr input_ptr N K TILE_N TILE_K)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [TILE_N, TILE_K] =>
          idx.1.val < N ∧ s.pids 1 * TILE_K + idx.2.1.val < K)
        (fun idx => (output_ptr, nonInnerOffset s N K TILE_K idx)))
      (expected := fun idx =>
        softmaxFlaggemsNonInnerSpec s input_ptr N K TILE_N TILE_K idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [softmax_kernel_non_inner_one_tile_surface]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := softmax_kernel_non_inner_one_tile_correct output_ptr input_ptr
    N K TILE_N TILE_K s s' hRange hExec idx
  simpa [hActive] using h

end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems
