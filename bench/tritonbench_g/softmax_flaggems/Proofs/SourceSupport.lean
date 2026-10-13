import bench.tritonbench_g.softmax_flaggems.Kernels
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.FullSource

/-- The original fallback loops initialize fp32 carried accumulators. An fp64
input changes their tensor types, which Triton v3.5.1 rejects. This declaration
records source-profile eligibility; it is not a claim of GPU compilation. The
one-tile branch has no such carried accumulator. -/
def SourceProfileSupported (input : Storage) (oneTile : Bool) : Prop :=
  oneTile = Bool.true ∨ input ≠ .fp64

/-- Every fallback lane receives a finite input in its first tile. Without
this bound, a never-active lane forms negative-infinity minus itself before
its value is included in the denominator reduction. -/
def ForwardSourceDomain (oneTile : Bool) (N B : Nat) : Prop :=
  oneTile = Bool.false → B ≤ N

theorem forward_source_domain_nonempty (oneTile : Bool) : ForwardSourceDomain oneTile 1 1 :=
  fun _ ↦ Nat.le_refl 1

theorem fp32_source_supported (oneTile : Bool) : SourceProfileSupported .fp32 oneTile :=
  Or.inr (by decide)
theorem fp16_source_supported (oneTile : Bool) : SourceProfileSupported .fp16 oneTile :=
  Or.inr (by decide)
theorem bf16_source_supported (oneTile : Bool) : SourceProfileSupported .bf16 oneTile :=
  Or.inr (by decide)
theorem one_tile_source_supported (input : Storage) : SourceProfileSupported input Bool.true :=
  Or.inl rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.FullSource
