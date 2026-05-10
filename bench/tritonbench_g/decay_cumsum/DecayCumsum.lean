import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.DecayCumsum

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-- Proof-oriented masked qg writeback slice of `decay_cumsum.py`'s
`prepare_qg_kg`.

The full kernel computes `tl.math.exp2(_g) * scale` inside a `BT` loop and also
writes `kg`. VeriTile's current arithmetic layer does not model that exp2
operation directly, so this slice fixes one loop row `t_rel`, starts from a
precomputed `QDecay` multiplier, and proves the masked `qg = q * decay`
writeback over the `BK` vector. -/
def prepare_qg_decay_store_slice
    (Q QDecay QG : RegionName)
    (s_qk_h DK t_rel BT BK : Nat) :
    ComputeKernel := triton {
  i_k = tl.program_id(axis=0)
  i_c = tl.program_id(axis=1)
  i_bh = tl.program_id(axis=2)
  offs = tl.arange(0, $(BK))
  base = i_bh * $(s_qk_h) + (i_c * $(BT) + $(t_rel)) * $(DK) + i_k * $(BK)
  mask = (i_k * $(BK) + offs) < $(DK)
  q = tl.load(Q + base + offs, mask=mask, other=0.0)
  decay = tl.load(QDecay + base + offs, mask=mask, other=0.0)
  qg = q * decay
  tl.store(QG + base + offs, qg, mask=mask)
}

def elemIndex (s : BlockState) (BK : Nat) (i : Fin BK) : Nat :=
  s.pids 0 * BK + i.val

def baseOffset (s : BlockState) (s_qk_h DK t_rel BT BK : Nat) : Nat :=
  s.pids 2 * s_qk_h + (s.pids 1 * BT + t_rel) * DK + s.pids 0 * BK

def offset
    (s : BlockState) (s_qk_h DK t_rel BT BK : Nat) (i : Fin BK) : Nat :=
  baseOffset s s_qk_h DK t_rel BT BK + i.val

def active (s : BlockState) (DK BK : Nat) (i : Fin BK) : Prop :=
  elemIndex s BK i < DK

instance activeDecidable (s : BlockState) (DK BK : Nat) (i : Fin BK) :
    Decidable (active s DK BK i) := by
  unfold active
  infer_instance

noncomputable def prepareQgDecaySpec
    (s : BlockState) (Q QDecay : RegionName)
    (s_qk_h DK t_rel BT BK : Nat) (i : Fin BK) : ℝ :=
  s.readMem Q (offset s s_qk_h DK t_rel BT BK i) *
    s.readMem QDecay (offset s s_qk_h DK t_rel BT BK i)

/-- Algorithm-layer correctness for the masked qg decay store slice. -/
theorem prepare_qg_decay_store_slice_correct
    (Q QDecay QG : RegionName)
    (s_qk_h DK t_rel BT BK : Nat)
    (s : BlockState) :
    ∀ i : Fin BK,
      let outAddr := offset s s_qk_h DK t_rel BT BK i
      (exec (prepare_qg_decay_store_slice Q QDecay QG s_qk_h DK t_rel BT BK)
          s).map (·.readMem QG outAddr)
        = some (if active s DK BK i then
            prepareQgDecaySpec s Q QDecay s_qk_h DK t_rel BT BK i
          else s.readMem QG outAddr) := by
  intro i
  have hInj : Function.Injective
      (fun idx : TileIndex [BK] =>
        s.pids 2 * s_qk_h + (s.pids 1 * BT + t_rel) * DK +
          s.pids 0 * BK + idx.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp [exec, prepare_qg_decay_store_slice, stepStmts, stepStmt, evalOp,
        Option.bind, Option.map, Tile.bop, Tile.ptrAdd, NumericDType.add,
        NumericDType.mul, ComparableDType.lt, elemIndex, baseOffset, offset]
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hInj (i, PUnit.unit)]
  by_cases hi : s.pids 0 * BK + i.val < DK
  · simp [active, prepareQgDecaySpec, elemIndex, baseOffset, offset, hi]
  · simp [active, elemIndex, baseOffset, offset, hi]

/-- Compute-facing correctness for the masked qg decay store slice. -/
theorem prepare_qg_decay_store_slice_compute_correct
    (Q QDecay QG : RegionName)
    (s_qk_h DK t_rel BT BK : Nat)
    (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := prepare_qg_decay_store_slice Q QDecay QG s_qk_h DK t_rel BT BK)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s DK BK)
        (fun i => (QG, offset s s_qk_h DK t_rel BT BK i)))
      (expected := fun i =>
        prepareQgDecaySpec s Q QDecay s_qk_h DK t_rel BT BK i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [prepare_qg_decay_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := prepare_qg_decay_store_slice_correct Q QDecay QG
    s_qk_h DK t_rel BT BK s i
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

end VeriTile.Bench.TritonBenchG.DecayCumsum
