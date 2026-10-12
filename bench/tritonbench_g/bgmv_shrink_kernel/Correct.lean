import bench.tritonbench_g.bgmv_shrink_kernel.Kernels
import bench.tritonbench_g.bgmv_shrink_kernel.Proofs.RealIO
import VeriTile.Meta.StatementAudit

/-! Real correctness of both complete sources in Kernels. Each program
computes its scaled rank-slice GEMV contribution. SPLIT_K=1 stores that value;
the atomic branch adds it to INITIAL output. Signed -1 skips all data work.

The contract includes successful flat execution, actual masked-address bounds
and preservation of every other typed cell. It requires four disjoint
allocations, signed metadata either -1 or nonnegative, initial input pins and
zero undef. Dimensions and strides are symbolic; positive BLOCK_K*SPLIT_K,
positive output stride and N≤BLOCK_N are stated below. K may have a partial
final block. Input loads remain active when N=0. The inputStride argument is
unused, matching the original source's contiguous input addressing.

Both supported input/weight storage types and every output dtype are covered
after precision erasure. These are per-program real results; cross-program
atomic interleaving, launch-wide reduction, fixed-width address overflow and
concrete IEEE behavior are outside this specification. -/
namespace VeriTile.Bench.TritonBenchG.BgmvShrinkKernel.Correctness
open VeriTile.Triton
open scoped VeriTile.Bench.TritonBenchG.BgmvShrinkKernel.RealIO
noncomputable section

abbrev Parameters := RealSetupSource.Parameters
abbrev Inputs := RealIO.Inputs

section
variable (p : Parameters) (storage : TypedSource.Storage) (out_dtype : FloatDType)

/-- Initial X/W values are pinned only on this split's rank indices below K.
All N output lanes are allocated; initial output is pinned only for atomic
addition. Metadata is read at batch. Allocation roles are vector, weights,
output and signed metadata, with the read/write output allocated once. -/
def originalIO : RealIO.IO p where
  kernel := (TypedSource.original storage out_dtype p.input p.weights p.output p.N p.K p.indices p.scaling
    p.inputBatchStride p.inputStride p.slotStride p.rowStride p.colStride p.outputBatchStride
    p.outputStride p.BN p.BK p.splits).eraseDType
  projection := RealIO.projection p storage out_dtype Bool.false
  readIndex := fun batch => batch
  readInput := fun batch k => batch * p.inputBatchStride + k.val
  readWeight := fun slot n k => p.slotStride * slot.toNat + n.val * p.rowStride + k.val * p.colStride
  write := fun batch n => batch * p.outputBatchStride + n.val * p.outputStride

def scalarLeftIO : RealIO.IO p :=
  { originalIO p storage out_dtype with
    kernel := (TypedSource.scalarLeft storage out_dtype p.input p.weights p.output p.N p.K p.indices p.scaling
      p.inputBatchStride p.inputStride p.slotStride p.rowStride p.colStride p.outputBatchStride
      p.outputStride p.BN p.BK p.splits).eraseDType
    projection := RealIO.projection p storage out_dtype Bool.true }

end

/-- Independent rank-slice formula. Every field of xs is an INITIAL input,
including initialOutput; no computed output is assumed. Each block contributes
BK ranks separated from the next by BK*SPLIT_K, with zero beyond K. -/
def expected (p : Parameters) (split : Nat) (xs : Inputs p) (n : Fin p.N) : ℝ :=
  let v := p.scaling * ∑ c ∈ Finset.range ((p.K + p.BK * p.splits - 1) / (p.BK * p.splits)),
    ∑ e : Fin p.BK,
      let k := c * (p.BK * p.splits) + (e.val + split * p.BK)
      if hk : k < p.K then xs.vector ⟨k, hk⟩ * xs.weights n ⟨k, hk⟩ else 0
  if p.splits = 1 then v else xs.initialOutput n + v

specification bgmv_shrink_correctness (p : Parameters) (storage : TypedSource.Storage) (out_dtype : FloatDType)
    (hS : 0 < p.BK * p.splits) (hstride : 0 < p.outputStride) (hN : p.N ≤ p.BN) :
    Spec.Real (originalIO p storage out_dtype ⊨ expected p) := by
  exact RealIO.implements p storage out_dtype Bool.false hS hstride hN

specification bgmv_scalar_left_correctness (p : Parameters) (storage : TypedSource.Storage) (out_dtype : FloatDType)
    (hS : 0 < p.BK * p.splits) (hstride : 0 < p.outputStride) (hN : p.N ≤ p.BN) :
    Spec.Real (scalarLeftIO p storage out_dtype ⊨ expected p) := by
  exact RealIO.implements p storage out_dtype Bool.true hS hstride hN

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BgmvShrinkKernel.Correctness
