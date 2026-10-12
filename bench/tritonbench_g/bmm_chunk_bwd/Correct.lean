import bench.tritonbench_g.bmm_chunk_bwd.Kernels
import bench.tritonbench_g.bmm_chunk_bwd.Proofs.RealContract
import VeriTile.Meta.StatementAudit

/-! Real correctness of complete chunked Doutᵀ·A, with an optional residual.
Both sources keep the runtime minimum, partial contraction/output tails,
independent strides and all three program IDs. Initial arrays alone determine
the output formula. The IO contract includes successful flat execution,
active-address bounds and preservation of every cell outside active outputs.

Dtype erasure is deliberate here; admitted floating-point equivalence is in
FPEquiv. The claim is per program, with mathematical integer addresses. Valid
host launch dimensions, fixed-width overflow and launch-wide race freedom are
separate from this real-semantics statement. -/
namespace VeriTile.Bench.TritonBenchG.BmmChunkBwd.Correctness
open VeriTile.Triton
open scoped BigOperators VeriTile.Bench.TritonBenchG.BmmChunkBwd.RealIO
noncomputable section

/-- Four regions; sequence/chunk length, K and group count; independent
batch, row, group and column strides; and the residual switch. -/
abbrev Parameters := RealProgram.Parameters
abbrev Inputs := RealInputs.Inputs

/-- Dout is transposed in the contraction: its row mask uses chunk_size.
The residual and output use the shorter runtime chunk length. Inactive array
padding is arbitrary; all windows come from source indices and strides. -/
def originalIO (aStorage doutStorage : TypedSource.Storage) (residualDType outputDType : FloatDType)
    (BM BN BCS : Nat) (p : Parameters) : RealIO.IO BM BN p.chunk where
  kernel := (RealProgram.source aStorage doutStorage Bool.false residualDType outputDType BM BN BCS p).eraseDType
  projection := by
    rw [RealProgram.storage_erasure]
    exact (RealViews.io Bool.false BM BN BCS p).projection
  Dout := p.Dout
  A := p.A
  Res := p.Res
  Db := p.Db
  doutWindow := fun pid₀ pid₁ pid₂ i =>
    (pid₁ * p.dBatch + RealViews.chunk p pid₂ * p.dChunk + RealViews.head p pid₂ * p.dHead) +
      (RealViews.row BM BN p pid₀ i.1 * p.dColumn + i.2.1.val * p.dRow)
  aWindow := fun pid₀ pid₁ pid₂ i =>
    (pid₁ * p.aBatch + RealViews.chunk p pid₂ * p.chunk * p.aRow + RealViews.head p pid₂ * p.aHead) +
      (i.1.val * p.aRow + RealViews.column BN p pid₀ i.2.1 * p.aColumn)
  residualWindow := RealViews.offset BM BN p p.rBatch p.rRow p.rHead p.rColumn
  outputWindow := RealViews.offset BM BN p p.oBatch p.oRow p.oHead p.oColumn
  doutMask := fun pid₀ _ pid₂ i => RealViews.row BM BN p pid₀ i.1 < p.chunk ∧ i.2.1.val < RealViews.length p pid₂
  aMask := fun pid₀ _ pid₂ i => i.1.val < RealViews.length p pid₂ ∧ RealViews.column BN p pid₀ i.2.1 < p.K
  outputMask := fun pid₀ _ pid₂ i => RealViews.row BM BN p pid₀ i.1 < RealViews.length p pid₂ ∧ RealViews.column BN p pid₀ i.2.1 < p.K
  hasResidual := p.hasResidual

/-- The candidate changes only the later residual addition order. -/
def residualLeftIO (aStorage doutStorage : TypedSource.Storage) (residualDType outputDType : FloatDType)
    (BM BN BCS : Nat) (p : Parameters) : RealIO.IO BM BN p.chunk :=
  { originalIO aStorage doutStorage residualDType outputDType BM BN BCS p with
    kernel := (RealProgram.source aStorage doutStorage Bool.true residualDType outputDType BM BN BCS p).eraseDType
    projection := by
      rw [RealProgram.storage_erasure]
      exact (RealViews.io Bool.true BM BN BCS p).projection }

/-- Independent initial-array formula, including the final partial chunk.
The first grid axis chooses tiles, the second batches, and the third chunks
and groups. The pinned windows already select this tile's initial inputs. -/
def expected (BM BN : Nat) (p : Parameters) (_pid₀ _pid₁ pid₂ : Nat)
    (xs : Inputs BM BN p.chunk) (i : TileIndex [BM, BN]) : ℝ :=
  let length := min p.chunk (p.seqlen - (pid₂ / p.groups) * p.chunk)
  (∑ k : Fin p.chunk, if k.val < length then xs.dout i.1 k * xs.a k i.2.1 else 0) +
    if p.hasResidual then xs.residual i.1 i.2.1 else 0

specification bmm_chunk_bwd_correctness
    (aStorage doutStorage : TypedSource.Storage) (residualDType outputDType : FloatDType)
    (BM BN BCS : Nat) (p : Parameters) (hBCS : 0 < BCS)
    (hout : RealViews.OutputDistinct BM BN p) :
    Spec.Real (originalIO aStorage doutStorage residualDType outputDType BM BN BCS p ⊨ expected BM BN p) := by
  simpa only [originalIO, RealProgram.storage_erasure] using
    RealContract.implements Bool.false BM BN BCS p hBCS hout

specification bmm_residual_left_correctness
    (aStorage doutStorage : TypedSource.Storage) (residualDType outputDType : FloatDType)
    (BM BN BCS : Nat) (p : Parameters) (hBCS : 0 < BCS)
    (hout : RealViews.OutputDistinct BM BN p) :
    Spec.Real (residualLeftIO aStorage doutStorage residualDType outputDType BM BN BCS p ⊨ expected BM BN p) := by
  simpa only [residualLeftIO, originalIO, RealProgram.storage_erasure] using
    RealContract.implements Bool.true BM BN BCS p hBCS hout

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BmmChunkBwd.Correctness
