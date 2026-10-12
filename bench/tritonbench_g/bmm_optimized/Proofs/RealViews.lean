import bench.tritonbench_g.bmm_optimized.Proofs.RealIO
import bench.tritonbench_g.bmm_optimized.Proofs.RealSafety

/-! Host-grid windows for two independent initial arrays. Runtime grid pins
identify these windows with the actual swizzled source coordinates. The
reference formula is the ordinary finite matrix product over those arrays. -/
namespace VeriTile.Bench.TritonBenchG.BmmOptimized.RealViews
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false

def row (BM BN : Nat) (p : RealProgram.Parameters) (px py : Nat) (i : Fin BM) : Nat :=
  RealGrid.rowTile px py (BmmChunkBwd.RealMath.blocks p.M BM) (BmmChunkBwd.RealMath.blocks p.N BN) p.groupM * BM + i.val

def column (BM BN : Nat) (p : RealProgram.Parameters) (px py : Nat) (j : Fin BN) : Nat :=
  RealGrid.columnTile px py (BmmChunkBwd.RealMath.blocks p.M BM) (BmmChunkBwd.RealMath.blocks p.N BN) p.groupM * BN + j.val

def io (swapped : Bool) (BM BN BK : Nat) (p : RealProgram.Parameters) : RealIO.IO BM BN p.K where
  kernel := RealProgram.kernel swapped BM BN BK p
  projection := by
    have hs : (RealProgram.source .fp32 swapped BM BN BK p).toAlgorithm? =
        .ok (RealProgram.source .fp32 swapped BM BN BK p).toAlgKernel :=
      RealSource.source_projection .fp32 p.divM p.divN p.divK swapped p.A p.B p.O p.M p.N p.K BM BN BK p.groupM
    simp only [RealProgram.kernel, ComputeKernel.eraseDType, hs]
    simp [ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?_fromKernelBody]
  A := p.A
  B := p.B
  O := p.O
  gridX := BmmChunkBwd.RealMath.blocks p.M BM
  gridY := BmmChunkBwd.RealMath.blocks p.N BN
  aWindow := fun px py pb i => pb * p.M * p.K + row BM BN p px py i.1 * p.K + i.2.1.val
  bWindow := fun px py pb i => pb * p.K * p.N + i.1.val * p.N + column BM BN p px py i.2.1
  outputWindow := fun px py pb i => pb * p.M * p.N + row BM BN p px py i.1 * p.N + column BM BN p px py i.2.1
  aMask := fun px py _ i => row BM BN p px py i.1 < p.M
  bMask := fun px py _ i => column BM BN p px py i.2.1 < p.N
  outputMask := fun px py _ i => row BM BN p px py i.1 < p.M ∧ column BM BN p px py i.2.1 < p.N

theorem launch_matches (swapped : Bool) (s : BlockState) (BM BN BK : Nat) (p : RealProgram.Parameters)
    (hl : (io swapped BM BN BK p).Launch s) : RealMemory.Launch s BM BN p := by
  exact ⟨hl.gridX, hl.gridY, hl.gridX.symm ▸ hl.x, hl.gridY.symm ▸ hl.y⟩

theorem row_matches (s : BlockState) (BM BN : Nat) (p : RealProgram.Parameters)
    (hl : RealMemory.Launch s BM BN p) (i : Fin BM) :
    row BM BN p (s.pids 0) (s.pids 1) i = RealSetup.rows s BM p i := by
  simp only [row, RealSetup.rows, RealSetup.tileM, hl.gridX, hl.gridY]

theorem column_matches (s : BlockState) (BM BN : Nat) (p : RealProgram.Parameters)
    (hl : RealMemory.Launch s BM BN p) (j : Fin BN) :
    column BM BN p (s.pids 0) (s.pids 1) j = RealSetup.columns s BN p j := by
  simp only [column, RealSetup.columns, RealSetup.tileN, hl.gridX, hl.gridY]

theorem formula_of_pins (swapped : Bool) (s : BlockState) (BM BN BK : Nat) (p : RealProgram.Parameters)
    (hl : RealMemory.Launch s BM BN p) (xs : RealIO.Inputs BM BN p.K)
    (hp : RealIO.Pins (io swapped BM BN BK p) s xs)
    (i : TileIndex [BM,BN]) (hi : RealMemory.active s BM BN p i) :
    RealMemory.formula s BM BN p i = RealIO.expected xs i := by
  simp only [RealMemory.formula, RealMath.contraction, RealIO.expected]
  rw [Finset.sum_range]
  apply Finset.sum_congr rfl
  intro k _
  have ha := hp.a i.1 k (by simpa only [io, row_matches s BM BN p hl] using hi.1)
  have hb := hp.b k i.2.1 (by simpa only [io, column_matches s BM BN p hl] using hi.2)
  apply congrArg₂ (· * ·)
  · simpa only [io, row_matches s BM BN p hl, RealLoop.aEntry, RealSetup.aBase] using ha
  · simpa only [io, column_matches s BM BN p hl, RealLoop.bEntry, RealSetup.bBase] using hb

theorem region_executes (swapped : Bool) (s : BlockState) (BM BN BK : Nat) (p : RealProgram.Parameters)
    (hh : RealMemory.Host BM BN BK p) (hl : (io swapped BM BN BK p).Launch s)
    (hu : s.undef = (fun _ _ => 0)) (xs : RealIO.Inputs BM BN p.K)
    (hp : RealIO.Pins (io swapped BM BN BK p) s xs) :
    ∃ t, exec (io swapped BM BN BK p).kernel.toAlgKernel s = some t ∧
      RealIO.Results (io swapped BM BN BK p) s t (RealIO.expected xs) := by
  have hl' := launch_matches swapped s BM BN BK p hl
  obtain ⟨t,he,hv,hf⟩ := RealMemory.executes swapped s BM BN BK p hh hl' (by simp [hu])
  refine ⟨t,he,?_,?_⟩
  · intro i hi
    have ha : RealMemory.active s BM BN p i := by
      simpa only [io, row_matches s BM BN p hl', column_matches s BM BN p hl', RealMemory.active] using hi
    have hv' := hv i ha
    have hr : t.readMem p.O (RealMemory.address s BM BN p i) = RealMemory.formula s BM BN p i := by
      simp [BlockState.readMem, hv']
    simpa only [io, row_matches s BM BN p hl', column_matches s BM BN p hl', RealMemory.address, RealSetup.oBase] using
      hr.trans (formula_of_pins swapped s BM BN BK p hl' xs hp i ha)
  · intro r a h
    apply hf r a
    intro i hi ⟨hr,ha⟩
    have hi' : (io swapped BM BN BK p).outputMask (s.pids 0) (s.pids 1) (s.pids 2) i := by
      simpa only [io, row_matches s BM BN p hl', column_matches s BM BN p hl', RealMemory.active] using hi
    have hw : (io swapped BM BN BK p).outputWindow (s.pids 0) (s.pids 1) (s.pids 2) i = a := by
      simpa only [io, row_matches s BM BN p hl', column_matches s BM BN p hl', RealMemory.address, RealSetup.oBase] using ha
    rcases h i hi' with hreg | hoff
    · exact hreg hr.symm
    · exact hoff hw.symm

theorem safe (swapped : Bool) (s : BlockState) (BM BN BK : Nat) (p : RealProgram.Parameters)
    (hh : RealMemory.Host BM BN BK p) (hl : (io swapped BM BN BK p).Launch s)
    (hu : s.undef = (fun _ _ => 0)) (bounds : RegionBounds)
    (hb : RealIO.Bounds (io swapped BM BN BK p) s bounds) :
    (io swapped BM BN BK p).kernel.toAlgKernel.TraceSafe bounds s := by
  have hl' := launch_matches swapped s BM BN BK p hl
  apply RealSafety.safe swapped s BM BN BK p hh.kPositive hh.divisibleK (by simp [hu]) bounds
  refine ⟨⟨?_,?_⟩,?_⟩
  · intro i k hi hk
    have hr := (RealMemory.row_active s BM BN BK p hh hl' i).mp hi
    have h := hb.a (i,⟨k,hk⟩,PUnit.unit) (by simpa only [io, row_matches s BM BN p hl'] using hr)
    simpa only [io, row_matches s BM BN p hl', RealSetup.aBase] using h
  · intro k j hk hj
    have hc := (RealMemory.column_active s BM BN BK p hh hl' j).mp hj
    have h := hb.b (⟨k,hk⟩,j,PUnit.unit) (by simpa only [io, column_matches s BM BN p hl'] using hc)
    simpa only [io, column_matches s BM BN p hl', RealSetup.bBase] using h
  · intro i j hi hj
    have hr := (RealMemory.row_active s BM BN BK p hh hl' i).mp hi
    have hc := (RealMemory.column_active s BM BN BK p hh hl' j).mp hj
    have h := hb.output (i,j,PUnit.unit)
      (by simpa only [io, row_matches s BM BN p hl', column_matches s BM BN p hl'] using And.intro hr hc)
    simpa only [io, row_matches s BM BN p hl', column_matches s BM BN p hl', RealSetup.oBase] using h

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BmmOptimized.RealViews
