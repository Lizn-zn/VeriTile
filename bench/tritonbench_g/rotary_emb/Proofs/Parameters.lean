import bench.tritonbench_g.rotary_emb.RotaryEmb
import bench.tritonbench_g.rotary_emb.Proofs.Math
import bench.tritonbench_g.rotary_emb.Proofs.Scatter

namespace VeriTile.Bench.TritonBenchG.RotaryEmb.FullMath
open VeriTile.Triton

/-- Source arguments are kept independently, including the two unused sine strides. -/
structure Parameters where
  qBatch : Nat
  qHead : Nat
  qDim : Nat
  kBatch : Nat
  kHead : Nat
  kDim : Nat
  cosBatch : Nat
  cosDim : Nat
  sinBatch : Nat
  sinDim : Nat
  length : Nat
  headsQ : Nat
  headsK : Nat
  blockHead : Nat
  blockSeq : Nat
  dimension : Nat

abbrev Parameters.Lane (p : Parameters) := Lane p.blockSeq p.blockHead p.dimension
abbrev Parameters.TrigLane (p : Parameters) := Lane p.blockSeq 1 p.dimension

def mathKernel (p : Parameters) (Q K Cos Sin : RegionName) : Kernel :=
  (rotary_kernel_surface Q K Cos Sin p.qBatch p.qHead p.qDim p.kBatch p.kHead p.kDim
    p.cosBatch p.cosDim p.sinBatch p.sinDim p.length p.headsQ p.headsK
    p.blockHead p.blockSeq p.dimension).toAlgKernel

def Parameters.qAddress (p : Parameters) (s : BlockState) (odd : Bool) (i : p.Lane) : Nat :=
  address s p.qBatch p.qHead p.qDim p.blockSeq p.blockHead p.dimension odd i

def Parameters.kAddress (p : Parameters) (s : BlockState) (odd : Bool) (i : p.Lane) : Nat :=
  address s p.kBatch p.kHead p.kDim p.blockSeq p.blockHead p.dimension odd i

def Parameters.trigAddress (p : Parameters) (s : BlockState) (odd : Bool) (i : p.Lane) : Nat :=
  FullMath.trigAddress s p.cosBatch p.cosDim p.blockSeq p.blockHead p.dimension odd i

def Parameters.activeQ (p : Parameters) (s : BlockState) (i : p.Lane) : Prop :=
  active s p.length p.headsQ p.blockSeq p.blockHead p.dimension i

def Parameters.activeK (p : Parameters) (s : BlockState) (i : p.Lane) : Prop :=
  active s p.length p.headsK p.blockSeq p.blockHead p.dimension i

noncomputable def Parameters.expectedQ (p : Parameters) (s : BlockState)
    (Q Cos Sin : RegionName) (odd : Bool) (i : p.Lane) : ℝ :=
  expected s Q Cos Sin p.qBatch p.qHead p.qDim p.cosBatch p.cosDim
    p.blockSeq p.blockHead p.dimension odd i

noncomputable def Parameters.expectedK (p : Parameters) (s : BlockState)
    (K Cos Sin : RegionName) (odd : Bool) (i : p.Lane) : ℝ :=
  expected s K Cos Sin p.kBatch p.kHead p.kDim p.cosBatch p.cosDim
    p.blockSeq p.blockHead p.dimension odd i

/-- Collision freedom concerns the lanes that actually write. Inactive padding
may alias any address, and zero strides are allowed when this condition holds. -/
def Parameters.Geometry (p : Parameters) (s : BlockState) : Prop :=
  Scatter.PairDisjoint (p.qAddress s) (p.activeQ s) ∧
  Scatter.PairDisjoint (p.kAddress s) (p.activeK s)

theorem active_pair_geometry (s : BlockState) (sb sh sd BS BH D L H : Nat)
    (hinj : PairInjective s sb sh sd BS BH D) :
    Scatter.PairDisjoint (address s sb sh sd BS BH D) (active s L H BS BH D) := by
  intro a b i j _ _ he
  have hp : (i,a)=(j,b) := hinj he
  exact ⟨congrArg Prod.snd hp, congrArg Prod.fst hp⟩

/-- Allocation bounds follow the source's distinct masks: trigonometric loads
are gated by sequence alone, even when the data head is inactive. -/
def Parameters.Bounds (p : Parameters) (s : BlockState)
    (bounds : RegionName → Nat) (Q K Cos Sin : RegionName) : Prop :=
  (∀ odd i, p.activeQ s i → p.qAddress s odd i < bounds Q) ∧
  (∀ odd i, p.activeK s i → p.kAddress s odd i < bounds K) ∧
  (∀ (odd : Bool) (i : p.TrigLane), sequence s p.blockSeq i < p.length →
    FullMath.trigAddress s p.cosBatch p.cosDim p.blockSeq 1 p.dimension odd i < bounds Cos ∧
    FullMath.trigAddress s p.cosBatch p.cosDim p.blockSeq 1 p.dimension odd i < bounds Sin)

def Parameters.footprint (p : Parameters) (s : BlockState) (Q K : RegionName) :
    WriteFootprint :=
  FullMath.footprint s Q K p.qBatch p.qHead p.qDim p.kBatch p.kHead p.kDim
    p.length p.headsQ p.headsK p.blockSeq p.blockHead p.dimension

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RotaryEmb.FullMath
