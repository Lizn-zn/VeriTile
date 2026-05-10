import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.ChunkGlaSimple

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-- Proof-oriented final output-store slice of `chunk_gla_simple.py`'s
`chunk_simple_gla_fwd_kernel_o`.

The full kernel computes a GLA output tile from Q/K/V/H/G. This slice starts
from a precomputed `BO` tile and proves the final boundary-checked writeback
into `O`. -/
def chunk_gla_simple_output_store_slice
    (BO O : RegionName) (s_v_h s_v_t T V BT BV : Nat) :
    ComputeKernel := triton {
  i_v = tl.program_id(axis=0)
  i_t = tl.program_id(axis=1)
  i_bh = tl.program_id(axis=2)
  offs_t = i_t * $(BT) + tl.arange(0, $(BT))
  offs_v = i_v * $(BV) + tl.arange(0, $(BV))
  mask = (offs_t[:, None] < $(T)) & (offs_v[None, :] < $(V))
  b_o = tl.load(BO + i_bh * $(s_v_h) + offs_t[:, None] * $(s_v_t) +
      offs_v[None, :], mask=mask, other=0.0)
  tl.store(O + i_bh * $(s_v_h) + offs_t[:, None] * $(s_v_t) +
      offs_v[None, :], b_o, mask=mask)
}

def tIndex (s : BlockState) (BT : Nat) (i : Fin BT) : Nat :=
  s.pids 1 * BT + i.val

def vIndex (s : BlockState) (BV : Nat) (j : Fin BV) : Nat :=
  s.pids 0 * BV + j.val

def active (s : BlockState) (T V BT BV : Nat) (idx : TileIndex [BT, BV]) : Prop :=
  tIndex s BT idx.1 < T ∧ vIndex s BV idx.2.1 < V

instance activeDecidable (s : BlockState) (T V BT BV : Nat)
    (idx : TileIndex [BT, BV]) : Decidable (active s T V BT BV idx) := by
  unfold active
  infer_instance

def tileOffset (s : BlockState) (s_v_h s_v_t BT BV : Nat)
    (idx : TileIndex [BT, BV]) : Nat :=
  s.pids 2 * s_v_h + tIndex s BT idx.1 * s_v_t + vIndex s BV idx.2.1

noncomputable def storeValue (s : BlockState) (BO : RegionName)
    (s_v_h s_v_t T V BT BV : Nat) (idx : TileIndex [BT, BV]) : ℝ :=
  WithBot.unbotD 0
    (if active s T V BT BV idx then
      some (s.readMem BO (tileOffset s s_v_h s_v_t BT BV idx))
    else some (0.0 : ℝ))

theorem chunk_gla_simple_output_store_slice_correct
    (BO O : RegionName) (s_v_h s_v_t T V BT BV : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BT, BV] => tileOffset s s_v_h s_v_t BT BV idx)) :
    ∀ idx : TileIndex [BT, BV],
      let outAddr := tileOffset s s_v_h s_v_t BT BV idx
      (exec (chunk_gla_simple_output_store_slice BO O s_v_h s_v_t T V BT BV)
          s).map (·.readMem O outAddr)
        = some (if active s T V BT BV idx then
            storeValue s BO s_v_h s_v_t T V BT BV idx
          else s.readMem O outAddr) := by
  intro idx
  simp [exec, chunk_gla_simple_output_store_slice, stepStmts, stepStmt, evalOp,
        Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
        Tile.ptrAdd, NumericDType.add, NumericDType.mul, ComparableDType.lt,
        tIndex, vIndex, active, tileOffset, TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BT, BV] → Nat :=
    fun idx => s.pids 2 * s_v_h + (s.pids 1 * BT + idx.1.val) * s_v_t +
      (s.pids 0 * BV + idx.2.1.val)
  let valueFn : TileIndex [BT, BV] → ℝ :=
    fun idx => WithBot.unbotD 0
      (if s.pids 1 * BT + idx.1.val < T ∧
          s.pids 0 * BV + idx.2.1.val < V then
        some (s.readMem BO (offsetFn idx))
      else some (0.0 : ℝ))
  let P : TileIndex [BT, BV] → Prop :=
    fun idx => s.pids 1 * BT + idx.1.val < T ∧
      s.pids 0 * BV + idx.2.1.val < V
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, tileOffset, tIndex, vIndex] using hOutInj
  change (List.foldl
      (fun (acc : BlockState) i =>
        if P i then acc.writeMem O (offsetFn i) (valueFn i) else acc)
      _ (TileShape.allIndices [BT, BV])).readMem O (offsetFn idx) =
    if P idx then storeValue s BO s_v_h s_v_t T V BT BV idx
    else s.readMem O (offsetFn idx)
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj idx]
  by_cases hActive :
      s.pids 1 * BT + idx.1.val < T ∧ s.pids 0 * BV + idx.2.1.val < V
  · rfl
  · rfl

theorem chunk_gla_simple_output_store_slice_compute_correct
    (BO O : RegionName) (s_v_h s_v_t T V BT BV : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BT, BV] => tileOffset s s_v_h s_v_t BT BV idx)) :
    ComputeCorrect.Realizes
      (kernel := chunk_gla_simple_output_store_slice BO O s_v_h s_v_t T V BT BV)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BT, BV] => active s T V BT BV idx)
        (fun idx : TileIndex [BT, BV] => (O, tileOffset s s_v_h s_v_t BT BV idx)))
      (expected := fun idx : TileIndex [BT, BV] =>
        storeValue s BO s_v_h s_v_t T V BT BV idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [chunk_gla_simple_output_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := chunk_gla_simple_output_store_slice_correct BO O s_v_h s_v_t T V BT BV
    s hOutInj idx
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

end VeriTile.Bench.TritonBenchG.ChunkGlaSimple
