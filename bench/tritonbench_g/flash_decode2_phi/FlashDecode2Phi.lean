import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.FlashDecode2Phi

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-- Proof-oriented final output-store slice of
`flash_decode2_phi.py`'s `_fwd_kernel_flash_decode_stage2`.

The full stage2 kernel reduces partial sequence-block outputs into
`acc / sum_exp`. This slice starts from a precomputed normalized `Final` vector
and proves the final `offs_d < head_dim` masked writeback into `Out`. -/
def flash_decode2_phi_final_store_slice
    (Final Out : RegionName)
    (head_dim
      stride_final_b stride_final_h stride_final_d
      stride_obs stride_oh stride_od
      BLOCK_DMODEL : Nat) :
    ComputeKernel := triton {
  cur_batch = tl.program_id(axis=0)
  cur_head = tl.program_id(axis=1)
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  mask = offs_d < $(head_dim)
  final = tl.load(Final + cur_batch * $(stride_final_b) +
      cur_head * $(stride_final_h) + offs_d * $(stride_final_d),
      mask=mask, other=0.0)
  tl.store(Out + cur_batch * $(stride_obs) + cur_head * $(stride_oh) +
      offs_d * $(stride_od), final, mask=mask)
}

def dIndex (_s : BlockState) (i : Fin BLOCK_DMODEL) : Nat :=
  i.val

def active (_s : BlockState) (head_dim : Nat) (i : Fin BLOCK_DMODEL) : Prop :=
  i.val < head_dim

instance activeDecidable (s : BlockState) (head_dim : Nat) (i : Fin BLOCK_DMODEL) :
    Decidable (active s head_dim i) := by
  unfold active
  infer_instance

def finalOffset
    (s : BlockState)
    (stride_final_b stride_final_h stride_final_d : Nat)
    (i : Fin BLOCK_DMODEL) : Nat :=
  s.pids 0 * stride_final_b + s.pids 1 * stride_final_h +
    dIndex s i * stride_final_d

def outOffset
    (s : BlockState)
    (stride_obs stride_oh stride_od : Nat)
    (i : Fin BLOCK_DMODEL) : Nat :=
  s.pids 0 * stride_obs + s.pids 1 * stride_oh + dIndex s i * stride_od

noncomputable def finalStoreValue
    (s : BlockState) (Final : RegionName)
    (head_dim stride_final_b stride_final_h stride_final_d : Nat)
    (i : Fin BLOCK_DMODEL) : ℝ :=
  WithBot.unbotD 0
    (if active s head_dim i then
      some (s.readMem Final
        (finalOffset s stride_final_b stride_final_h stride_final_d i))
    else some (0.0 : ℝ))

/-- Algorithm-layer correctness for the masked final flash-decode writeback. -/
theorem flash_decode2_phi_final_store_slice_correct
    (Final Out : RegionName)
    (head_dim
      stride_final_b stride_final_h stride_final_d
      stride_obs stride_oh stride_od
      BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_DMODEL => outOffset s stride_obs stride_oh stride_od i)) :
    ∀ i : Fin BLOCK_DMODEL,
      let outAddr := outOffset s stride_obs stride_oh stride_od i
      (exec (flash_decode2_phi_final_store_slice Final Out head_dim
            stride_final_b stride_final_h stride_final_d stride_obs stride_oh
            stride_od BLOCK_DMODEL) s).map (·.readMem Out outAddr)
        = some (if active s head_dim i then
            finalStoreValue s Final head_dim stride_final_b stride_final_h
              stride_final_d i
          else s.readMem Out outAddr) := by
  intro i
  simp [exec, flash_decode2_phi_final_store_slice, stepStmts, stepStmt, evalOp,
        Option.bind, Option.map, Tile.bop, Tile.ptrAdd, NumericDType.add,
        NumericDType.mul, ComparableDType.lt, dIndex, active, finalOffset,
        outOffset]
  let offsetFn : TileIndex [BLOCK_DMODEL] → Nat :=
    fun idx => s.pids 0 * stride_obs + s.pids 1 * stride_oh + idx.1.val * stride_od
  let valueFn : TileIndex [BLOCK_DMODEL] → ℝ :=
    fun idx =>
      WithBot.unbotD 0
        (if idx.1.val < head_dim then
          some (s.readMem Final
            (s.pids 0 * stride_final_b + s.pids 1 * stride_final_h +
              idx.1.val * stride_final_d))
        else some (0.0 : ℝ))
  let P : TileIndex [BLOCK_DMODEL] → Prop :=
    fun idx => idx.1.val < head_dim
  have hOffsetInj : Function.Injective offsetFn := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    have habFin : outOffset s stride_obs stride_oh stride_od a =
        outOffset s stride_obs stride_oh stride_od b := by
      simpa [offsetFn, outOffset, dIndex] using hab
    obtain rfl : a = b := hOutInj habFin
    rfl
  change (List.foldl
      (fun (acc : BlockState) idx =>
        if P idx then acc.writeMem Out (offsetFn idx) (valueFn idx) else acc)
      _ (TileShape.allIndices [BLOCK_DMODEL])).readMem Out
        (offsetFn (i, PUnit.unit)) =
    if active s head_dim i then
      finalStoreValue s Final head_dim stride_final_b stride_final_h
        stride_final_d i
    else s.readMem Out (offsetFn (i, PUnit.unit))
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj (i, PUnit.unit)]
  by_cases hi : i.val < head_dim
  · simp [P, valueFn, active, finalStoreValue, finalOffset, dIndex, hi]
  · simp [P, active, finalStoreValue, dIndex, hi]

/-- Compute-facing correctness for the masked final flash-decode writeback. -/
theorem flash_decode2_phi_final_store_slice_compute_correct
    (Final Out : RegionName)
    (head_dim
      stride_final_b stride_final_h stride_final_d
      stride_obs stride_oh stride_od
      BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_DMODEL => outOffset s stride_obs stride_oh stride_od i)) :
    ComputeCorrect.Realizes
      (kernel := flash_decode2_phi_final_store_slice Final Out head_dim
        stride_final_b stride_final_h stride_final_d stride_obs stride_oh
        stride_od BLOCK_DMODEL)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_DMODEL => active s head_dim i)
        (fun i => (Out, outOffset s stride_obs stride_oh stride_od i)))
      (expected := fun i : Fin BLOCK_DMODEL =>
        finalStoreValue s Final head_dim stride_final_b stride_final_h
          stride_final_d i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [flash_decode2_phi_final_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := flash_decode2_phi_final_store_slice_correct Final Out head_dim
    stride_final_b stride_final_h stride_final_d stride_obs stride_oh stride_od
    BLOCK_DMODEL s hOutInj i
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

end VeriTile.Bench.TritonBenchG.FlashDecode2Phi
