import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-- Surface transcription of `fused_recurrent_hgrn.py`'s
`fused_recurrent_hgrn_fwd_kernel`.

The forward recurrence is a regular `0..T` loop and is represented directly,
including the optional initial-state load and final-state store. The backward
kernel iterates `range(T - 1, -1, -1)`, which requires signed negative-step loop
support and is intentionally not folded into this forward surface. -/
def fused_recurrent_hgrn_fwd_surface
    (X G O H0 HT : RegionName) (T D BD : Nat)
    (USE_INITIAL_STATE STORE_FINAL_STATE : Bool) :
    ComputeKernel := triton {
  i_d = tl.program_id(axis=0)
  i_bh = tl.program_id(axis=1)
  o_d = i_d * $(BD) + tl.arange(0, $(BD))
  mask = o_d < $(D)
  p_x = X + i_bh * $(T) * $(D) + o_d
  p_g = G + i_bh * $(T) * $(D) + o_d
  p_o = O + i_bh * $(T) * $(D) + o_d
  b_h = tl.zeros([$(BD)], dtype=tl.float32)
  if USE_INITIAL_STATE {
    p_h0 = H0 + i_bh * $(D) + o_d
    b_h += tl.load(p_h0, mask=mask, other=0.0).to(tl.float32)
  }
  for _i in range($(0), $(T), $(1)) {
    b_x = tl.load(p_x, mask=mask, other=0.0).to(tl.float32)
    b_g = tl.load(p_g, mask=mask, other=0.0).to(tl.float32)
    b_h = b_g * b_h + b_x
    tl.store(p_o, (b_h).to(O.dtype.element_ty), mask=mask)
    p_x += $(D)
    p_g += $(D)
    p_o += $(D)
  }
  if STORE_FINAL_STATE {
    p_ht = HT + i_bh * $(D) + o_d
    tl.store(p_ht, (b_h).to(HT.dtype.element_ty), mask=mask)
  }
}

/-- Proof-oriented per-step output-store slice of
`fused_recurrent_hgrn.py`'s `fused_recurrent_hgrn_fwd_kernel`.

The full kernel updates `b_h` recurrently over `T`. This slice models one
iteration's `p_o` writeback from a precomputed `BH` vector into `O`. -/
def fused_recurrent_hgrn_output_store_slice
    (BH O : RegionName) (i_t T D BD : Nat) :
    ComputeKernel := triton {
  i_d = tl.program_id(axis=0)
  i_bh = tl.program_id(axis=1)
  offs_d = i_d * $(BD) + tl.arange(0, $(BD))
  mask = offs_d < $(D)
  b_h = tl.load(BH + i_bh * $(D) + offs_d, mask=mask, other=0.0)
  tl.store(O + (i_bh * $(T) + $(i_t)) * $(D) + offs_d,
    (b_h).to(O.dtype.element_ty), mask=mask)
}

def dIndex (s : BlockState) (BD : Nat) (i : Fin BD) : Nat :=
  s.pids 0 * BD + i.val

def active (s : BlockState) (D BD : Nat) (i : Fin BD) : Prop :=
  dIndex s BD i < D

instance activeDecidable (s : BlockState) (D BD : Nat) (i : Fin BD) :
    Decidable (active s D BD i) := by
  unfold active
  infer_instance

def bhOffset (s : BlockState) (D BD : Nat) (i : Fin BD) : Nat :=
  s.pids 1 * D + dIndex s BD i

def outOffset (s : BlockState) (i_t T D BD : Nat) (i : Fin BD) : Nat :=
  (s.pids 1 * T + i_t) * D + dIndex s BD i

noncomputable def storeValue (s : BlockState) (BH : RegionName) (D BD : Nat)
    (i : Fin BD) : ℝ :=
  WithBot.unbotD 0
    (if active s D BD i then some (s.readMem BH (bhOffset s D BD i))
    else some (0.0 : ℝ))

theorem fused_recurrent_hgrn_output_store_slice_correct
    (BH O : RegionName) (i_t T D BD : Nat) (s : BlockState)
    (hOutInj : Function.Injective (fun i : Fin BD => outOffset s i_t T D BD i)) :
    ∀ i : Fin BD,
      let outAddr := outOffset s i_t T D BD i
      (exec (fused_recurrent_hgrn_output_store_slice BH O i_t T D BD) s).map
          (·.readMem O outAddr)
        = some (if active s D BD i then storeValue s BH D BD i
          else s.readMem O outAddr) := by
  intro i
  simp [exec, fused_recurrent_hgrn_output_store_slice, stepStmts, stepStmt,
        evalOp, Option.bind, Option.map, Tile.bop, Tile.ptrAdd,
        NumericDType.add, NumericDType.mul, ComparableDType.lt, dIndex,
        active, bhOffset, outOffset]
  let offsetFn : TileIndex [BD] → Nat :=
    fun idx => (s.pids 1 * T + i_t) * D + (s.pids 0 * BD + idx.1.val)
  let valueFn : TileIndex [BD] → ℝ :=
    fun idx => WithBot.unbotD 0
      (if s.pids 0 * BD + idx.1.val < D then
        some (s.readMem BH (s.pids 1 * D + (s.pids 0 * BD + idx.1.val)))
      else some (0.0 : ℝ))
  let P : TileIndex [BD] → Prop := fun idx => s.pids 0 * BD + idx.1.val < D
  have hOffsetInj : Function.Injective offsetFn := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    have habFin : outOffset s i_t T D BD a = outOffset s i_t T D BD b := by
      simpa [offsetFn, outOffset, dIndex] using hab
    obtain rfl : a = b := hOutInj habFin
    rfl
  change (List.foldl
      (fun (acc : BlockState) idx =>
        if P idx then acc.writeMem O (offsetFn idx) (valueFn idx) else acc)
      _ (TileShape.allIndices [BD])).readMem O (offsetFn (i, PUnit.unit)) =
    if active s D BD i then storeValue s BH D BD i
    else s.readMem O (offsetFn (i, PUnit.unit))
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj (i, PUnit.unit)]
  by_cases hi : s.pids 0 * BD + i.val < D
  · simp [P, valueFn, active, storeValue, bhOffset, dIndex, hi]
  · simp [P, active, storeValue, dIndex, hi]

theorem fused_recurrent_hgrn_output_store_slice_compute_correct
    (BH O : RegionName) (i_t T D BD : Nat) (s : BlockState)
    (hOutInj : Function.Injective (fun i : Fin BD => outOffset s i_t T D BD i)) :
    ComputeCorrect.Realizes
      (kernel := fused_recurrent_hgrn_output_store_slice BH O i_t T D BD)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BD => active s D BD i)
        (fun i => (O, outOffset s i_t T D BD i)))
      (expected := fun i : Fin BD => storeValue s BH D BD i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [fused_recurrent_hgrn_output_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := fused_recurrent_hgrn_output_store_slice_correct BH O i_t T D BD s hOutInj i
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

end VeriTile.Bench.TritonBenchG.FusedRecurrentHgrn
