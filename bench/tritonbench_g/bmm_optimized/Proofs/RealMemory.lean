import bench.tritonbench_g.bmm_optimized.Proofs.RealExecution

/-! Valid host launch geometry and complete real memory results. The actual
runtime grid is pinned to the host's ceiling-divided dimensions. Divisibility
flags justify omitted masks; active output addresses are proved distinct. -/
namespace VeriTile.Bench.TritonBenchG.BmmOptimized.RealMemory
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false

structure Host (BM BN BK : Nat) (p : RealProgram.Parameters) : Prop where
  mPositive : 0 < BM
  nPositive : 0 < BN
  kPositive : 0 < BK
  groupPositive : 0 < p.groupM
  divisibleM : p.divM = Bool.true → BM ∣ p.M
  divisibleN : p.divN = Bool.true → BN ∣ p.N
  divisibleK : p.divK = Bool.true → BK ∣ p.K

structure Launch (s : BlockState) (BM BN : Nat) (p : RealProgram.Parameters) : Prop where
  gridX : s.numPids 0 = BmmChunkBwd.RealMath.blocks p.M BM
  gridY : s.numPids 1 = BmmChunkBwd.RealMath.blocks p.N BN
  x : s.pids 0 < s.numPids 0
  y : s.pids 1 < s.numPids 1

theorem tile_bounds (s : BlockState) (BM BN BK : Nat) (p : RealProgram.Parameters)
    (hh : Host BM BN BK p) (hl : Launch s BM BN p) :
    RealSetup.tileM s p < BmmChunkBwd.RealMath.blocks p.M BM ∧
      RealSetup.tileN s p < BmmChunkBwd.RealMath.blocks p.N BN := by
  have h := RealGrid.coordinates_in_grid _ _ _ _ _ hh.groupPositive hl.x hl.y
  change RealSetup.tileM s p < s.numPids 0 ∧ RealSetup.tileN s p < s.numPids 1 at h
  simpa only [hl.gridX, hl.gridY] using h

theorem row_active (s : BlockState) (BM BN BK : Nat) (p : RealProgram.Parameters)
    (hh : Host BM BN BK p) (hl : Launch s BM BN p) (i : Fin BM) :
    RealMath.active p.divM (RealSetup.rows s BM p i) p.M ↔ RealSetup.rows s BM p i < p.M := by
  exact RealMath.contraction_mask p.M BM (RealSetup.tileM s p) p.divM hh.mPositive hh.divisibleM
    (tile_bounds s BM BN BK p hh hl).1 i

theorem column_active (s : BlockState) (BM BN BK : Nat) (p : RealProgram.Parameters)
    (hh : Host BM BN BK p) (hl : Launch s BM BN p) (j : Fin BN) :
    RealMath.active p.divN (RealSetup.columns s BN p j) p.N ↔ RealSetup.columns s BN p j < p.N := by
  exact RealMath.contraction_mask p.N BN (RealSetup.tileN s p) p.divN hh.nPositive hh.divisibleN
    (tile_bounds s BM BN BK p hh hl).2 j

def active (s : BlockState) (BM BN : Nat) (p : RealProgram.Parameters)
    (i : TileIndex [BM,BN]) : Prop := RealSetup.rows s BM p i.1 < p.M ∧ RealSetup.columns s BN p i.2.1 < p.N

def address (s : BlockState) (BM BN : Nat) (p : RealProgram.Parameters)
    (i : TileIndex [BM,BN]) : Nat :=
  RealSetup.oBase s p + RealSetup.rows s BM p i.1 * p.N + RealSetup.columns s BN p i.2.1

def formula (s : BlockState) (BM BN : Nat) (p : RealProgram.Parameters)
    (i : TileIndex [BM,BN]) : ℝ :=
  RealMath.contraction p.K (RealLoop.aEntry s BM p i.1) (RealLoop.bEntry s BN p i.2.1)

theorem active_iff (s : BlockState) (BM BN BK : Nat) (p : RealProgram.Parameters)
    (hh : Host BM BN BK p) (hl : Launch s BM BN p) (i : TileIndex [BM,BN]) :
    RealStore.active p (RealSetup.rows s BM p) (RealSetup.columns s BN p) i ↔ active s BM BN p i := by
  simp only [RealStore.active, active, row_active s BM BN BK p hh hl, column_active s BM BN BK p hh hl]

theorem address_injective (s : BlockState) (BM BN : Nat) (p : RealProgram.Parameters)
    (a b : TileIndex [BM,BN]) (ha : active s BM BN p a) (hb : active s BM BN p b)
    (he : address s BM BN p a = address s BM BN p b) : a = b := by
  have hN : 0 < p.N := Nat.zero_lt_of_lt ha.2
  have he' : RealSetup.rows s BM p a.1 * p.N + RealSetup.columns s BN p a.2.1 =
      RealSetup.rows s BM p b.1 * p.N + RealSetup.columns s BN p b.2.1 := by
    exact Nat.add_left_cancel (by simpa only [address, Nat.add_assoc] using he)
  have hrow := congrArg (fun x => x / p.N) he'
  rw [Nat.mul_comm (RealSetup.rows s BM p a.1) p.N,
    Nat.mul_comm (RealSetup.rows s BM p b.1) p.N] at hrow
  simp only [Nat.mul_add_div hN, Nat.div_eq_of_lt ha.2, Nat.div_eq_of_lt hb.2, Nat.add_zero] at hrow
  have hcol : RealSetup.columns s BN p a.2.1 = RealSetup.columns s BN p b.2.1 := by
    rw [hrow] at he'
    exact Nat.add_left_cancel he'
  have hr : a.1 = b.1 := Fin.ext (Nat.add_left_cancel hrow)
  have hc : a.2.1 = b.2.1 := Fin.ext (Nat.add_left_cancel hcol)
  rcases a with ⟨i,j,⟨⟩⟩
  rcases b with ⟨i',j',⟨⟩⟩
  simpa only [Prod.mk.injEq, and_true] using And.intro hr hc

theorem executes (swapped : Bool) (s : BlockState) (BM BN BK : Nat) (p : RealProgram.Parameters)
    (hh : Host BM BN BK p) (hl : Launch s BM BN p) (hu : ∀ r o, s.undef r o = 0) :
    ∃ t, exec (RealProgram.kernel swapped BM BN BK p).toAlgKernel s = some t ∧
      (∀ i, active s BM BN p i → t.mem p.O (address s BM BN p i) = MemCell.real (formula s BM BN p i)) ∧
      (∀ r o, (∀ i, active s BM BN p i → ¬ (p.O = r ∧ address s BM BN p i = o)) → t.mem r o = s.mem r o) := by
  obtain ⟨current,hc,he⟩ := RealExecution.kernel_executes swapped s BM BN BK p hh.kPositive hh.divisibleK hu
  refine ⟨RealExecution.finished s current BM BN BK p, he, ?_, ?_⟩
  · intro i hi
    exact RealExecution.readback s current BM BN BK p hh.kPositive i
      ((active_iff s BM BN BK p hh hl i).mpr hi)
      (fun a b ha hb he => address_injective s BM BN p a b
        ((active_iff s BM BN BK p hh hl a).mp ha) ((active_iff s BM BN BK p hh hl b).mp hb) he)
  · intro r o h
    exact RealExecution.frame s current BM BN BK p hc.mem r o
      (fun i hi => h i ((active_iff s BM BN BK p hh hl i).mp hi))

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BmmOptimized.RealMemory
