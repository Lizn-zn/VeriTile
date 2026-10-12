import bench.tritonbench_g.destindex_copy.Proofs.SourceIO
/-! Real correctness of the complete unmasked dual scatter in Kernels.lean.
The source ignores head counts and head strides: only the first head is copied.
The read/write windows below include every BLOCK_DMODEL lane, so rounded-up
padding needs allocated storage. Destination metadata is a signed row index;
nonnegativity prevents clipping at the model's pointer boundary.

The IO contract quantifies over independent initial arrays, finite disjoint
flat allocations, actual-access bounds and a zero unspecified-value oracle.
It proves successful execution, both output windows and the full tagged-cell
frame. Input/output aliases are allowed; the output regions and per-output
lane addresses must be distinct. It is a per-program real statement, without
fixed-width overflow, host-allocation or concurrent-launch claims. -/
namespace VeriTile.Bench.TritonBenchG.DestindexCopy.Correctness
open VeriTile.Triton
open scoped RealIO
noncomputable section

def originalIO (p : Layout) (nopeType ropeType : TypedSource.Storage) : RealIO.IO p where
  kernel := (p.source nopeType ropeType Bool.false).eraseDType
  projection := (SourceIO.sourceIO p nopeType ropeType Bool.false).projection
  readMetadata := id
  readNope := fun pid j => pid*p.stride_kv_nope_bs + p.stride_kv_nope_d*j.val
  readRope := fun pid j => pid*p.stride_kv_rope_bs + p.stride_kv_rope_d*j.val
  writeNope := fun row j => row.toNat*p.stride_o_nope_bs + p.stride_o_nope_d*j.val
  writeRope := fun row j => row.toNat*p.stride_o_rope_bs + p.stride_o_rope_d*j.val

def indexCommutedIO (p : Layout) (nopeType ropeType : TypedSource.Storage) : RealIO.IO p :=
  { originalIO p nopeType ropeType with
    kernel := (p.source nopeType ropeType Bool.true).eraseDType
    projection := (SourceIO.sourceIO p nopeType ropeType Bool.true).projection }

def expected {N R : Nat} (xs : RealIO.Inputs N R) : (Fin N → ℝ) × (Fin R → ℝ) :=
  (xs.nope,xs.rope)

variable (p : Layout) (nopeType ropeType : TypedSource.Storage)
    (hOut : p.O_nope ≠ p.O_rope)
    (hNope : Function.Injective (fun j : Fin p.BLOCK_DMODEL_NOPE => p.stride_o_nope_d*j.val))
    (hRope : Function.Injective (fun j : Fin p.BLOCK_DMODEL_ROPE => p.stride_o_rope_d*j.val))
include hOut hNope hRope

specification destindex_copy_correctness :
    Spec.Real (originalIO p nopeType ropeType ⊨ expected) := by
  exact SourceIO.correct p nopeType ropeType Bool.false hOut hNope hRope

specification index_commuted_correctness :
    Spec.Real (indexCommutedIO p nopeType ropeType ⊨ expected) := by
  exact SourceIO.correct p nopeType ropeType Bool.true hOut hNope hRope

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.DestindexCopy.Correctness
