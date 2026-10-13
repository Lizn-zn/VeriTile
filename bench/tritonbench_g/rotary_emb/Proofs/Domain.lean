import bench.tritonbench_g.rotary_emb.Proofs.Parameters

namespace VeriTile.Bench.TritonBenchG.RotaryEmb.FullMath
open VeriTile.Triton

/-- Two sequences, two heads and two dimension pairs; noncontiguous batch
storage and deliberately unused sine strides exercise the full geometry. -/
def exampleParameters : Parameters :=
  ⟨16,4,1,16,4,1,4,1,999,983,2,2,2,2,2,4⟩

theorem example_geometry (s : BlockState) : exampleParameters.Geometry s := by
  constructor <;> exact active_pair_geometry s 16 4 1 2 2 4 2 2
    (pair_geometry_nonempty s)

theorem example_bounds (s : BlockState) (h0 : s.pids 0=0) (h1 : s.pids 1=0) :
    exampleParameters.Bounds s (fun _ ↦ 32) "Q" "K" "Cos" "Sin" := by
  constructor
  · intro odd i _
    have hs := i.1.isLt
    have hh := i.2.1.isLt
    have hd := dimension_lt odd i
    simp only [exampleParameters, Parameters.qAddress, address, sequence, head,
      h0, h1, Nat.zero_mul, Nat.zero_add, Nat.mul_one]
    omega
  constructor
  · intro odd i _
    have hs := i.1.isLt
    have hh := i.2.1.isLt
    have hd := dimension_lt odd i
    simp only [exampleParameters, Parameters.kAddress, address, sequence, head,
      h0, h1, Nat.zero_mul, Nat.zero_add, Nat.mul_one]
    omega
  · intro odd i _
    have hs := i.1.isLt
    have hd := dimension_lt odd i
    simp only [exampleParameters, trigAddress, sequence, h1, Nat.zero_mul,
      Nat.zero_add, Nat.mul_one]
    constructor <;> omega

/-- The formula and access premises are jointly satisfiable for a nonempty
multisequence, multihead block, independently of memory and register contents. -/
theorem example_domain (s : BlockState) (h0 : s.pids 0=0) (h1 : s.pids 1=0) :
    RegionsSeparate "Q" "K" "Cos" "Sin" ∧ exampleParameters.Geometry s ∧
      exampleParameters.Bounds s (fun _ ↦ 32) "Q" "K" "Cos" "Sin" :=
  ⟨regions_nonempty, example_geometry s, example_bounds s h0 h1⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RotaryEmb.FullMath
