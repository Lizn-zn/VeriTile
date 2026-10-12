import bench.tritonbench_g.bmm_chunk_fwd.Proofs.RealIO
import bench.tritonbench_g.bmm_chunk_fwd.Proofs.RealMemory

/-! Initial-array windows for all three grid coordinates. Data and metadata
loads stop at the actual chunk length, while the output includes padded cells
within chunk_size. A skipped causal tile has no active read or write window. -/
namespace VeriTile.Bench.TritonBenchG.BmmChunkFwd.RealViews
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false

def chunk (p : RealProgram.Parameters) (pid₂ : Nat) := pid₂ / p.groups
def head (p : RealProgram.Parameters) (pid₂ : Nat) := pid₂ - chunk p pid₂ * p.groups
def length (p : RealProgram.Parameters) (pid₂ : Nat) := min p.chunk (p.seqlen - chunk p pid₂ * p.chunk)
def row (BM BN : Nat) (p : RealProgram.Parameters) (pid₀ : Nat) (i : Fin BM) :=
  pid₀ / RealSetup.tilesN BN p * BM + i.val
def column (BN : Nat) (p : RealProgram.Parameters) (pid₀ : Nat) (j : Fin BN) :=
  pid₀ % RealSetup.tilesN BN p * BN + j.val

def activeTile (BM BN : Nat) (p : RealProgram.Parameters) (pid₀ : Nat) : Bool :=
  !p.causal || decide (pid₀ % RealSetup.tilesN BN p * BN < (pid₀ / RealSetup.tilesN BN p + 1) * BM)

def outputOffset (BM BN : Nat) (p : RealProgram.Parameters)
    (pid₀ pid₁ pid₂ : Nat) (i : TileIndex [BM, BN]) : Nat :=
  (pid₁ * p.oBatch + chunk p pid₂ * p.oChunk + head p pid₂ * p.oHead) +
    (row BM BN p pid₀ i.1 * p.oRow + column BN p pid₀ i.2.1 * p.oColumn)

def io (swapped : Bool) (BM BN BK : Nat) (p : RealProgram.Parameters) : RealIO.IO BM BN p.K where
  kernel := RealProgram.kernel swapped BM BN BK p
  projection := by
    simp only [RealProgram.kernel, ComputeKernel.eraseDType, RealProgram.source_projects]
    simp [ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?_fromKernelBody]
  A := p.A
  B := p.B
  Seq := Region.cast p.Seq
  Out := p.Out
  aWindow := fun pid₀ pid₁ pid₂ i =>
    (pid₁ * p.aBatch + chunk p pid₂ * p.chunk * p.aRow + head p pid₂ * p.aHead) +
      (row BM BN p pid₀ i.1 * p.aRow + i.2.1.val * p.aColumn)
  bWindow := fun pid₀ pid₁ pid₂ i =>
    (pid₁ * p.bBatch + chunk p pid₂ * p.chunk * p.bRow + head p pid₂ * p.bHead) +
      (i.1.val * p.bColumn + column BN p pid₀ i.2.1 * p.bRow)
  rowIDWindow := fun pid₀ pid₁ pid₂ i =>
    (pid₁ * p.sBatch + chunk p pid₂ * p.chunk * p.sRow) + row BM BN p pid₀ i.1 * p.sRow
  columnIDWindow := fun pid₀ pid₁ pid₂ i =>
    (pid₁ * p.sBatch + chunk p pid₂ * p.chunk * p.sRow) + column BN p pid₀ i.1 * p.sRow
  outputWindow := outputOffset BM BN p
  aMask := fun pid₀ _ pid₂ i => activeTile BM BN p pid₀ ∧ row BM BN p pid₀ i.1 < length p pid₂
  bMask := fun pid₀ _ pid₂ i => activeTile BM BN p pid₀ ∧ column BN p pid₀ i.2.1 < length p pid₂
  rowIDMask := fun pid₀ _ pid₂ i => activeTile BM BN p pid₀ ∧ row BM BN p pid₀ i.1 < length p pid₂
  columnIDMask := fun pid₀ _ pid₂ i => activeTile BM BN p pid₀ ∧ column BN p pid₀ i.1 < length p pid₂
  outputMask := fun pid₀ _ _ i => activeTile BM BN p pid₀ ∧
    row BM BN p pid₀ i.1 < p.chunk ∧ column BN p pid₀ i.2.1 < p.chunk
  hasSeq := p.hasSeq

def expected (BM BN : Nat) (p : RealProgram.Parameters) (pid₀ _pid₁ pid₂ : Nat)
    (xs : RealInputs.Inputs BM BN p.K) (i : TileIndex [BM, BN]) : ℝ :=
  RealInputs.formula (length p pid₂) (row BM BN p pid₀) (column BN p pid₀) p.hasSeq xs i.1 i.2.1

def OutputDistinct (BM BN : Nat) (p : RealProgram.Parameters) : Prop :=
  ∀ pid₀ pid₁ pid₂ a b,
    activeTile BM BN p pid₀ ∧ row BM BN p pid₀ a.1 < p.chunk ∧ column BN p pid₀ a.2.1 < p.chunk →
    activeTile BM BN p pid₀ ∧ row BM BN p pid₀ b.1 < p.chunk ∧ column BN p pid₀ b.2.1 < p.chunk →
    outputOffset BM BN p pid₀ pid₁ pid₂ a = outputOffset BM BN p pid₀ pid₁ pid₂ b → a = b

theorem formula_of_pins (swapped : Bool) (s : BlockState) (BM BN BK : Nat) (p : RealProgram.Parameters)
    (xs : RealInputs.Inputs BM BN p.K) (hp : RealIO.Pins (io swapped BM BN BK p) s xs)
    (i : TileIndex [BM, BN]) (hi : RealMemory.active s BM BN p i) :
    RealMemory.formula s BM BN p i = expected BM BN p (s.pids 0) (s.pids 1) (s.pids 2) xs i := by
  have hr : row BM BN p (s.pids 0) i.1 = RealSetup.rows s BM BN p i.1 := rfl
  have hc : column BN p (s.pids 0) i.2.1 = RealSetup.columns s BN p i.2.1 := rfl
  have hl : length p (s.pids 2) = RealSetup.limit s p := rfl
  by_cases hrow : RealSetup.rows s BM BN p i.1 < RealSetup.limit s p
  · by_cases hcol : RealSetup.columns s BN p i.2.1 < RealSetup.limit s p
    · have hsum : (∑ k ∈ Finset.range p.K, RealLoop.product s BM BN p i.1 i.2.1 k) =
          ∑ k : Fin p.K, xs.a i.1 k * xs.b k i.2.1 := by
        rw [Finset.sum_range]
        apply Finset.sum_congr rfl
        intro k _
        exact congrArg₂ (· * ·) (hp.a i.1 k ⟨hi.1, hrow⟩) (hp.b k i.2.1 ⟨hi.1, hcol⟩)
      cases hs : p.hasSeq
      · simp [RealMemory.formula, expected, RealInputs.formula, hr, hc, hl, hrow, hcol, hs, hsum]
      · have hm := hp.rowIDs hs i.1 ⟨hi.1, hrow⟩
        have hn := hp.columnIDs hs i.2.1 ⟨hi.1, hcol⟩
        change RealMemory.sequenceID s p (RealSetup.rows s BM BN p i.1) = xs.rowIDs i.1 at hm
        change RealMemory.sequenceID s p (RealSetup.columns s BN p i.2.1) = xs.columnIDs i.2.1 at hn
        simp [RealMemory.formula, expected, RealInputs.formula, hr, hc, hl, hrow, hcol, hs, hsum, hm, hn]
    · simp [RealMemory.formula, expected, RealInputs.formula, hr, hc, hl, hcol]
  · simp [RealMemory.formula, expected, RealInputs.formula, hr, hc, hl, hrow]

theorem region_executes (swapped : Bool) (s : BlockState) (BM BN BK : Nat) (p : RealProgram.Parameters)
    (hBK : 0 < BK) (hinj : OutputDistinct BM BN p)
    (xs : RealInputs.Inputs BM BN p.K) (hp : RealIO.Pins (io swapped BM BN BK p) s xs) :
    ∃ t, exec (io swapped BM BN BK p).kernel.toAlgKernel s = some t ∧
      RealIO.Results (io swapped BM BN BK p) s t (expected BM BN p (s.pids 0) (s.pids 1) (s.pids 2) xs) := by
  obtain ⟨t, he, hv, hf⟩ := RealMemory.executes swapped s BM BN BK p hBK
    (hinj (s.pids 0) (s.pids 1) (s.pids 2))
  refine ⟨t, he, ?_, ?_⟩
  · intro i hi
    have h := hv i hi
    have hr : t.readMem p.Out (RealMemory.address s BM BN p i) = RealMemory.formula s BM BN p i := by
      simp [BlockState.readMem, h]
    exact hr.trans (formula_of_pins swapped s BM BN BK p xs hp i hi)
  · intro r a h
    apply hf r a
    intro i hi ⟨hr, ha⟩
    rcases h i hi with hr' | ha'
    · exact hr' hr.symm
    · exact ha' ha.symm

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BmmChunkFwd.RealViews
