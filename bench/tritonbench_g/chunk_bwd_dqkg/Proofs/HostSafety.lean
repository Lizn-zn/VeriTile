import bench.tritonbench_g.chunk_bwd_dqkg.Proofs.RealStep
import VeriTile.Meta.Disprove

/-! Address-level checks for the original Python fixtures. The first fixture
allocates H and DH as `[2,4,64,64]`, but a program's block pointer declares
`(V, NT*K) = (64,128)`. Its boundary check therefore admits addresses beyond
that allocation. The counterexample is to the actual active-address window,
not to real correctness under a sufficient memory-bound precondition.
No allocation, pointer or mask is repaired here. -/
namespace VeriTile.Bench.TritonBenchG.ChunkBwdDqkg.HostSafety
open VeriTile.Triton RealMath
noncomputable section

/-- An address window required by both H and DH boundary-checked loads.
The pointer is the same one used in `RealStep.executes`. -/
def HistoryWindowSafe (p : Layout) (l : Launch) (r : RegionName)
    (count allocation : Nat) : Prop :=
  ∀ v : Fin p.BV, ∀ e : Fin p.BK,
    let bp := (RealStep.hPointer p l r count).data (v, e, PUnit.unit)
    bp.inBounds [v.val, e.val] [0, 1] = Bool.true →
      bp.address [v.val, e.val] < allocation

def firstFixture : Layout where
  T := 128
  K := 64
  V := 64
  BT := 64
  BK := 64
  BV := 64
  NT := 2
  s_k_h := 128 * 64
  s_k_t := 64
  s_v_h := 128 * 64
  s_v_t := 64
  s_h_h := 64 * 64
  s_h_t := 64
  scale := 1 / 10

/-- A genuine host grid coordinate: key block zero, time chunk one, last
batch/head. The head index is flattened as in `program_id(2)`. -/
def lastProgram : Launch := ⟨0, 1, 7, 8⟩

theorem first_fixture_active_lane (r : RegionName) :
    ((RealStep.hPointer firstFixture lastProgram r 0).data
      (⟨0, by decide⟩, ⟨0, by decide⟩, PUnit.unit)).inBounds [0, 0] [0, 1] = Bool.true := by
  rfl

theorem first_fixture_address (r : RegionName) :
    ((RealStep.hPointer firstFixture lastProgram r 0).data
      (⟨0, by decide⟩, ⟨0, by decide⟩, PUnit.unit)).address [0, 0] = 32768 := by
  norm_num [RealStep.hPointer, firstFixture, lastProgram, BlockPtr.address_2d_offsets]

/-- The first active H/DH lane lies exactly one past the last allocated
element: valid offsets are 0 through 32767. -/
refutation first_fixture_history_window_safe (r : RegionName) :
    HistoryWindowSafe firstFixture lastProgram r 0 (2 * 4 * 64 * 64) := by
  intro h
  have bad := h ⟨0, by decide⟩ ⟨0, by decide⟩ (first_fixture_active_lane r)
  rw [first_fixture_address] at bad
  exact (by decide : ¬ (32768 < 2 * 4 * 64 * 64)) bad

/-- A separate mask distinction: for K=65, key-block lane one at key block
one is outside logical K, yet H/DH accepts it within the NT*K column extent.
This is why the independent formula keeps the wider source mask. -/
theorem history_key_tail_is_active (r : RegionName) :
    let p := { firstFixture with K := 65, NT := 2 }
    let l : Launch := ⟨1, 0, 0, 8⟩
    ((RealStep.hPointer p l r 0).data
      (⟨0, by decide⟩, ⟨1, by decide⟩, PUnit.unit)).inBounds [0, 1] [0, 1] = Bool.true ∧
    ¬ (l.key * p.BK + 1 < p.K) := by
  exact ⟨rfl, by decide⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkBwdDqkg.HostSafety
