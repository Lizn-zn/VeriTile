import bench.tritonbench_g.bmm_chunk_fwd.Kernels
import bench.tritonbench_g.bmm_chunk_fwd.Proofs.RealContract
import VeriTile.Meta.StatementAudit

/-! Real correctness of the complete chunked A·Bᵀ source and its dot-first
candidate. The formula uses independent initial arrays and signed sequence
IDs. Partial K blocks and all three program IDs are retained. Causal mode
skips whole tiles; surviving tiles have no extra elementwise triangular mask.
Data loads stop at the actual chunk length, while stores cover chunk_size:
padded final-chunk output cells are written as zero.

The IO contract includes successful flat execution, actual masked-address
bounds and preservation of every cell outside active outputs. Arithmetic is
real after dtype erasure. Host launch validity, fixed-width address overflow
and launch-wide race freedom are separate from this per-program statement. -/
namespace VeriTile.Bench.TritonBenchG.BmmChunkFwd.Correctness
open VeriTile.Triton
open scoped BigOperators VeriTile.Bench.TritonBenchG.BmmChunkFwd.RealIO
noncomputable section

/-- A/B/Out regions, signed Seq metadata, all dimensions and independent
strides, plus the causal and sequence-filter switches. -/
abbrev Parameters := RealProgram.Parameters
abbrev Inputs := RealInputs.Inputs

/-- Read windows stop at the actual final-chunk length; output windows stop
at chunk_size. Metadata windows are used only when hasSeq is true. A skipped
causal tile has no active IO. The underlying contract keeps A/B/Seq/Out
allocations disjoint and preserves cells outside the active output window. -/
def originalIO (aStorage bStorage : TypedSource.Storage) (outputDType : FloatDType)
    (BM BN BK : Nat) (p : Parameters) : RealIO.IO BM BN p.K where
  kernel := (RealProgram.source aStorage bStorage Bool.false outputDType BM BN BK p).eraseDType
  projection := by
    rw [RealProgram.storage_erasure]
    exact (RealViews.io Bool.false BM BN BK p).projection
  A := p.A
  B := p.B
  Seq := Region.cast p.Seq
  Out := p.Out
  aWindow := fun pid₀ pid₁ pid₂ i =>
    (pid₁ * p.aBatch + RealViews.chunk p pid₂ * p.chunk * p.aRow + RealViews.head p pid₂ * p.aHead) +
      (RealViews.row BM BN p pid₀ i.1 * p.aRow + i.2.1.val * p.aColumn)
  bWindow := fun pid₀ pid₁ pid₂ i =>
    (pid₁ * p.bBatch + RealViews.chunk p pid₂ * p.chunk * p.bRow + RealViews.head p pid₂ * p.bHead) +
      (i.1.val * p.bColumn + RealViews.column BN p pid₀ i.2.1 * p.bRow)
  rowIDWindow := fun pid₀ pid₁ pid₂ i =>
    (pid₁ * p.sBatch + RealViews.chunk p pid₂ * p.chunk * p.sRow) + RealViews.row BM BN p pid₀ i.1 * p.sRow
  columnIDWindow := fun pid₀ pid₁ pid₂ i =>
    (pid₁ * p.sBatch + RealViews.chunk p pid₂ * p.chunk * p.sRow) + RealViews.column BN p pid₀ i.1 * p.sRow
  outputWindow := RealViews.outputOffset BM BN p
  aMask := fun pid₀ _ pid₂ i => RealViews.activeTile BM BN p pid₀ ∧ RealViews.row BM BN p pid₀ i.1 < RealViews.length p pid₂
  bMask := fun pid₀ _ pid₂ i => RealViews.activeTile BM BN p pid₀ ∧ RealViews.column BN p pid₀ i.2.1 < RealViews.length p pid₂
  rowIDMask := fun pid₀ _ pid₂ i => RealViews.activeTile BM BN p pid₀ ∧ RealViews.row BM BN p pid₀ i.1 < RealViews.length p pid₂
  columnIDMask := fun pid₀ _ pid₂ i => RealViews.activeTile BM BN p pid₀ ∧ RealViews.column BN p pid₀ i.1 < RealViews.length p pid₂
  outputMask := fun pid₀ _ _ i => RealViews.activeTile BM BN p pid₀ ∧
    RealViews.row BM BN p pid₀ i.1 < p.chunk ∧ RealViews.column BN p pid₀ i.2.1 < p.chunk
  hasSeq := p.hasSeq

/-- The candidate reverses only each fp32 carry-plus-dot addition. -/
def dotFirstIO (aStorage bStorage : TypedSource.Storage) (outputDType : FloatDType)
    (BM BN BK : Nat) (p : Parameters) : RealIO.IO BM BN p.K :=
  { originalIO aStorage bStorage outputDType BM BN BK p with
    kernel := (RealProgram.source aStorage bStorage Bool.true outputDType BM BN BK p).eraseDType
    projection := by
      rw [RealProgram.storage_erasure]
      exact (RealViews.io Bool.true BM BN BK p).projection }

/-- Independent matrix-product formula with the actual row/column length
and optional signed sequence equality. Invalid padded rows/columns are zero;
negative sequence IDs are compared literally, without excluding them. -/
def expected (BM BN : Nat) (p : Parameters) (pid₀ _pid₁ pid₂ : Nat)
    (xs : Inputs BM BN p.K) (i : TileIndex [BM, BN]) : ℝ :=
  let length := min p.chunk (p.seqlen - (pid₂ / p.groups) * p.chunk)
  if RealViews.row BM BN p pid₀ i.1 < length ∧
      RealViews.column BN p pid₀ i.2.1 < length ∧
      (p.hasSeq = Bool.false ∨ xs.rowIDs i.1 = xs.columnIDs i.2.1)
  then ∑ k : Fin p.K, xs.a i.1 k * xs.b k i.2.1 else 0

specification bmm_chunk_fwd_correctness
    (aStorage bStorage : TypedSource.Storage) (outputDType : FloatDType)
    (BM BN BK : Nat) (p : Parameters) (hBK : 0 < BK)
    (hout : RealViews.OutputDistinct BM BN p) :
    Spec.Real (originalIO aStorage bStorage outputDType BM BN BK p ⊨ expected BM BN p) := by
  simpa only [originalIO, RealProgram.storage_erasure] using
    RealContract.implements Bool.false BM BN BK p hBK hout

specification bmm_dot_first_correctness
    (aStorage bStorage : TypedSource.Storage) (outputDType : FloatDType)
    (BM BN BK : Nat) (p : Parameters) (hBK : 0 < BK)
    (hout : RealViews.OutputDistinct BM BN p) :
    Spec.Real (dotFirstIO aStorage bStorage outputDType BM BN BK p ⊨ expected BM BN p) := by
  simpa only [dotFirstIO, originalIO, RealProgram.storage_erasure] using
    RealContract.implements Bool.true BM BN BK p hBK hout

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BmmChunkFwd.Correctness
