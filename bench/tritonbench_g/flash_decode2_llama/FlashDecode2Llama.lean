import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.FlashDecode2Llama

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-- Proof-oriented final output-store slice of
`flash_decode2_llama.py`'s `_fwd_kernel_flash_decode_stage2`.

The full stage2 kernel reduces per-sequence-block partial outputs into a final
`acc / sum_exp` vector. This slice starts from a precomputed normalized `Final`
vector and proves the unmasked writeback into `O`. -/
def flash_decode2_llama_final_store_slice
    (Final O : RegionName)
    (stride_final_b stride_final_h stride_final_d
      stride_obs stride_oh stride_od
      BLOCK_DMODEL : Nat) :
    ComputeKernel := triton {
  cur_batch = tl.program_id(axis=0)
  cur_head = tl.program_id(axis=1)
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  final = tl.load(Final + cur_batch * $(stride_final_b) +
      cur_head * $(stride_final_h) + offs_d * $(stride_final_d))
  tl.store(O + cur_batch * $(stride_obs) + cur_head * $(stride_oh) +
      offs_d * $(stride_od), final)
}

def dIndex (_s : BlockState) (i : Fin BLOCK_DMODEL) : Nat :=
  i.val

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

/-- Algorithm-layer correctness for the final flash-decode writeback. -/
theorem flash_decode2_llama_final_store_slice_correct
    (Final O : RegionName)
    (stride_final_b stride_final_h stride_final_d
      stride_obs stride_oh stride_od
      BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_DMODEL => outOffset s stride_obs stride_oh stride_od i)) :
    ∀ i : Fin BLOCK_DMODEL,
      let outAddr := outOffset s stride_obs stride_oh stride_od i
      (exec (flash_decode2_llama_final_store_slice Final O stride_final_b
            stride_final_h stride_final_d stride_obs stride_oh stride_od
            BLOCK_DMODEL) s).map (·.readMem O outAddr)
        = some (s.readMem Final
            (finalOffset s stride_final_b stride_final_h stride_final_d i)) := by
  intro i
  simp [exec, flash_decode2_llama_final_store_slice, stepStmts, stepStmt,
        evalOp, Option.bind, Option.map, Tile.bop, Tile.ptrAdd,
        NumericDType.add, NumericDType.mul, dIndex, finalOffset, outOffset]
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_DMODEL] =>
        s.pids 0 * stride_obs + s.pids 1 * stride_oh + idx.1.val * stride_od) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    have habFin : outOffset s stride_obs stride_oh stride_od a =
        outOffset s stride_obs stride_oh stride_od b := by
      simpa [outOffset, dIndex] using hab
    obtain rfl : a = b := hOutInj habFin
    rfl
  rw [BlockState.scatter_readback_nd _ _ _ hRawInj (i, PUnit.unit)]

/-- Compute-facing correctness for the final flash-decode writeback. -/
theorem flash_decode2_llama_final_store_slice_compute_correct
    (Final O : RegionName)
    (stride_final_b stride_final_h stride_final_d
      stride_obs stride_oh stride_od
      BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_DMODEL => outOffset s stride_obs stride_oh stride_od i)) :
    ComputeCorrect.Realizes
      (kernel := flash_decode2_llama_final_store_slice Final O stride_final_b
        stride_final_h stride_final_d stride_obs stride_oh stride_od
        BLOCK_DMODEL)
      (initialState := s)
      (write := fun i : Fin BLOCK_DMODEL =>
        some (O, outOffset s stride_obs stride_oh stride_od i))
      (expected := fun i : Fin BLOCK_DMODEL =>
        s.readMem Final
          (finalOffset s stride_final_b stride_final_h stride_final_d i)) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [flash_decode2_llama_final_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro i
  have h := flash_decode2_llama_final_store_slice_correct Final O
    stride_final_b stride_final_h stride_final_d stride_obs stride_oh stride_od
    BLOCK_DMODEL s hOutInj i
  rw [hExec] at h
  exact Option.some.inj h

end VeriTile.Bench.TritonBenchG.FlashDecode2Llama
