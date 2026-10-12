import bench.tritonbench_g.chunk_cumsum_vector.Kernels
import bench.tritonbench_g.chunk_cumsum_vector.Proofs.RealIO
import bench.tritonbench_g.chunk_cumsum_vector.Proofs.RealPadding
import VeriTile.Meta.StatementAudit

/-! Both complete Kernels sources implement column-wise prefix sums after
precision erasure. The IO window covers all T rows processed by one program;
the source still executes BT-by-BS chunks, including both partial tails.
Positive BT, distinct input/output, active bounds, nonoverlapping active output
addresses and zero unspecified real loads are explicit flat IO conditions.
The zero-oracle condition belongs to the shared flat IO adapter.
`RealPadding.source_outputs` additionally proves full region execution, valid
outputs and the memory frame for arbitrary unspecified real values on both axes.
The IO relation includes successful flat execution and the full tagged-cell
memory frame. The third program coordinate is unused. -/
namespace VeriTile.Bench.TritonBenchG.ChunkCumsumVector.Correctness
open VeriTile.Triton
open scoped BigOperators VeriTile.Triton.Masked3DTileKernelIO₁
noncomputable section

/-- Symbolic head/time/channel strides, logical T/S sizes and BT/BS block sizes. -/
abbrev Layout := RealProgram.Layout

/-- The complete logical input/output window of one feature-block/head program.
The same strides and masks are used by the original source's block pointers. -/
def originalIO (p : Layout) (s o : RegionName) (dtype out_dtype : FloatDType) :
    Masked3DTileKernelIO₁ where
  kernel := (TypedSource.original dtype out_dtype s o p.headStride p.timeStride p.channelStride
    p.T p.S p.BT p.BS).eraseDType
  projection := (RealIO.io Bool.false p s o dtype out_dtype).projection
  inp := s
  out := o
  shape := [p.T, p.BS]
  read := fun feature head _ i =>
    head * p.headStride + i.1.val * p.timeStride + (feature * p.BS + i.2.1.val) * p.channelStride
  write := fun feature head _ i =>
    head * p.headStride + i.1.val * p.timeStride + (feature * p.BS + i.2.1.val) * p.channelStride
  mask := fun feature _ _ i => feature * p.BS + i.2.1.val < p.S

def sumLeftIO (p : Layout) (s o : RegionName) (dtype out_dtype : FloatDType) :
    Masked3DTileKernelIO₁ :=
  { originalIO p s o dtype out_dtype with
    kernel := (TypedSource.sumLeft dtype out_dtype s o p.headStride p.timeStride p.channelStride
      p.T p.S p.BT p.BS).eraseDType
    projection := (RealIO.io Bool.true p s o dtype out_dtype).projection }

/-- Initial input values up to the output time index, independently per column. -/
def expected (p : Layout) (_feature _head : Nat) (xs : TileIndex [p.T,p.BS] → ℝ)
    (i : TileIndex [p.T,p.BS]) : ℝ :=
  ∑ k : Fin p.T, if k.val ≤ i.1.val then xs (k,i.2.1,PUnit.unit) else 0

/-- Distinct active output elements have distinct addresses over the whole time
span, including different chunks. Masked feature lanes need not be injective. -/
def noOutputOverlap (p : Layout) : Prop :=
  ∀ (feature head t u : Nat) (i j : Fin p.BS),
    (t < p.T ∧ feature * p.BS + i.val < p.S) →
    (u < p.T ∧ feature * p.BS + j.val < p.S) →
    head * p.headStride + t * p.timeStride + (feature * p.BS + i.val) * p.channelStride =
      head * p.headStride + u * p.timeStride + (feature * p.BS + j.val) * p.channelStride →
    t = u ∧ i = j

specification chunk_cumsum_vector_correctness (p : Layout) (s o : RegionName)
    (dtype out_dtype : FloatDType) (hSO : s ≠ o) (hBT : 0 < p.BT)
    (hLayout : noOutputOverlap p) :
    Spec.Real (originalIO p s o dtype out_dtype ⊨ expected p) := by
  exact RealIO.original_implements p s o dtype out_dtype hSO hBT hLayout

specification chunk_sum_left_correctness (p : Layout) (s o : RegionName)
    (dtype out_dtype : FloatDType) (hSO : s ≠ o) (hBT : 0 < p.BT)
    (hLayout : noOutputOverlap p) :
    Spec.Real (sumLeftIO p s o dtype out_dtype ⊨ expected p) := by
  exact RealIO.candidate_implements p s o dtype out_dtype hSO hBT hLayout

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkCumsumVector.Correctness
