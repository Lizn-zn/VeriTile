import bench.tritonbench_g.bmm_chunk_bwd.Proofs.RealIO
import bench.tritonbench_g.bmm_chunk_bwd.Proofs.RealMemory

/-! The concrete initial-array views use all three program IDs and preserve
all source strides. Dout's row load mask is chunk_size, while residual/store
use the actual shortened length; the distinction matters in a partial chunk. -/
namespace VeriTile.Bench.TritonBenchG.BmmChunkBwd.RealViews
open VeriTile.Triton
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

def chunk (p : RealProgram.Parameters) (pid₂ : Nat) := pid₂ / p.groups
def head (p : RealProgram.Parameters) (pid₂ : Nat) := pid₂ - chunk p pid₂ * p.groups
def length (p : RealProgram.Parameters) (pid₂ : Nat) := RealMath.chunkLimit p.seqlen p.chunk (chunk p pid₂)
def row (BM BN : Nat) (p : RealProgram.Parameters) (pid₀ : Nat) (i : Fin BM) :=
  pid₀ / RealSetup.tilesN BN p * BM + i.val
def column (BN : Nat) (p : RealProgram.Parameters) (pid₀ : Nat) (j : Fin BN) :=
  pid₀ % RealSetup.tilesN BN p * BN + j.val

def offset (BM BN : Nat) (p : RealProgram.Parameters) (batch rowStride headStride columnStride : Nat)
    (pid₀ pid₁ pid₂ : Nat) (i : TileIndex [BM, BN]) : Nat :=
  (pid₁ * batch + chunk p pid₂ * p.chunk * rowStride + head p pid₂ * headStride) +
    (row BM BN p pid₀ i.1 * rowStride + column BN p pid₀ i.2.1 * columnStride)

def io (swapped : Bool) (BM BN BCS : Nat) (p : RealProgram.Parameters) : RealIO.IO BM BN p.chunk where
  kernel := RealProgram.kernel swapped BM BN BCS p
  projection := by
    simp only [RealProgram.kernel, ComputeKernel.eraseDType, RealProgram.source_projects]
    simp [ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?_fromKernelBody]
  Dout := p.Dout
  A := p.A
  Res := p.Res
  Db := p.Db
  doutWindow := fun pid₀ pid₁ pid₂ i =>
    (pid₁ * p.dBatch + chunk p pid₂ * p.dChunk + head p pid₂ * p.dHead) +
      (row BM BN p pid₀ i.1 * p.dColumn + i.2.1.val * p.dRow)
  aWindow := fun pid₀ pid₁ pid₂ i =>
    (pid₁ * p.aBatch + chunk p pid₂ * p.chunk * p.aRow + head p pid₂ * p.aHead) +
      (i.1.val * p.aRow + column BN p pid₀ i.2.1 * p.aColumn)
  residualWindow := offset BM BN p p.rBatch p.rRow p.rHead p.rColumn
  outputWindow := offset BM BN p p.oBatch p.oRow p.oHead p.oColumn
  doutMask := fun pid₀ _ pid₂ i => row BM BN p pid₀ i.1 < p.chunk ∧ i.2.1.val < length p pid₂
  aMask := fun pid₀ _ pid₂ i => i.1.val < length p pid₂ ∧ column BN p pid₀ i.2.1 < p.K
  outputMask := fun pid₀ _ pid₂ i => row BM BN p pid₀ i.1 < length p pid₂ ∧ column BN p pid₀ i.2.1 < p.K
  hasResidual := p.hasResidual

def expected (BM BN : Nat) (p : RealProgram.Parameters) (_pid₀ _pid₁ pid₂ : Nat)
    (xs : RealInputs.Inputs BM BN p.chunk) (i : TileIndex [BM, BN]) : ℝ :=
  RealInputs.formula (length p pid₂) p.hasResidual xs i.1 i.2.1

/-- Only addresses of simultaneous active output writes need be distinct.
This is a stride/layout condition, independent of all numerical values. -/
def OutputDistinct (BM BN : Nat) (p : RealProgram.Parameters) : Prop :=
  ∀ pid₀ pid₁ pid₂ a b,
    row BM BN p pid₀ a.1 < length p pid₂ ∧ column BN p pid₀ a.2.1 < p.K →
    row BM BN p pid₀ b.1 < length p pid₂ ∧ column BN p pid₀ b.2.1 < p.K →
    offset BM BN p p.oBatch p.oRow p.oHead p.oColumn pid₀ pid₁ pid₂ a =
      offset BM BN p p.oBatch p.oRow p.oHead p.oColumn pid₀ pid₁ pid₂ b → a = b

theorem formula_of_pins (swapped : Bool) (s : BlockState) (BM BN BCS : Nat) (p : RealProgram.Parameters)
    (xs : RealInputs.Inputs BM BN p.chunk) (hp : RealIO.Pins (io swapped BM BN BCS p) s xs)
    (i : TileIndex [BM, BN]) (hi : RealMemory.active s BM BN p i) :
    RealMemory.formula s BM BN p i = expected BM BN p (s.pids 0) (s.pids 1) (s.pids 2) xs i := by
  have hlen : RealSetup.limit s p ≤ p.chunk := Nat.min_le_left _ _
  have hrow := lt_of_lt_of_le hi.1 hlen
  have hsum := RealInputs.sum_of_pins p.chunk (RealSetup.limit s p) hlen
    (fun k => xs.dout i.1 k * xs.a k i.2.1) (RealLoop.product s BM BN p i.1 i.2.1)
    (by
      intro k hk
      have hd := hp.dout i.1 k ⟨hrow, hk⟩
      have ha := hp.a k i.2.1 ⟨hk, hi.2⟩
      exact congrArg₂ (· * ·) hd ha)
  unfold RealMemory.formula expected RealInputs.formula
  rw [hsum]
  congr 1
  split
  · exact hp.residual (by assumption) i.1 i.2.1 hi
  · rfl

theorem region_executes (swapped : Bool) (s : BlockState) (BM BN BCS : Nat) (p : RealProgram.Parameters)
    (hBCS : 0 < BCS) (hinj : OutputDistinct BM BN p)
    (xs : RealInputs.Inputs BM BN p.chunk) (hp : RealIO.Pins (io swapped BM BN BCS p) s xs) :
    ∃ t, exec (io swapped BM BN BCS p).kernel.toAlgKernel s = some t ∧
      RealIO.Results (io swapped BM BN BCS p) s t (expected BM BN p (s.pids 0) (s.pids 1) (s.pids 2) xs) := by
  obtain ⟨t, he, hv, hf⟩ := RealMemory.executes swapped s BM BN BCS p hBCS
    (hinj (s.pids 0) (s.pids 1) (s.pids 2))
  refine ⟨t, ?_, ?_, ?_⟩
  · exact he
  · intro i hi
    have h := hv i hi
    have hr : t.readMem p.Db (RealMemory.address s BM BN p p.oBatch p.oRow p.oHead p.oColumn i) =
        RealMemory.formula s BM BN p i := by simp [BlockState.readMem, h]
    exact hr.trans (formula_of_pins swapped s BM BN BCS p xs hp i hi)
  · intro r a h
    apply hf r a
    intro i hi ⟨hr, ha⟩
    rcases h i hi with hr' | ha'
    · exact hr' hr.symm
    · exact ha' ha.symm

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BmmChunkBwd.RealViews
