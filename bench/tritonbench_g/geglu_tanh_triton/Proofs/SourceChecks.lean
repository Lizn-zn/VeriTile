import bench.tritonbench_g.geglu_tanh_triton.Kernels

namespace VeriTile.Bench.TritonBenchG.GegluTanhTriton.SourceChecks
open VeriTile.Triton
set_option maxHeartbeats 8000000
set_option maxRecDepth 16000

/-- Python imports tanh from libdevice, and the argument is computed in fp32
regardless of the partner's or upstream gradient's storage dtype. -/
def expectedTanh (B : Nat) : ComputeStmt :=
  .assign .real [B] "tanh_result" (.compute (.alg .fp32
    (.libdeviceTanh (.ref .real [B] "tanh_arg"))))

theorem forward_tanh (storage partner : Storage) (v : Bool) (a b c : RegionName)
    (stride N B : Nat) :
    (typedFwd storage partner v a b c stride N B).surfaceBody[11]? =
      some (expectedTanh B) := by
  cases storage <;> cases partner <;> cases v <;> rfl

theorem backward_tanh (storage partner gradient : Storage) (v : Bool) (dc a b : RegionName)
    (stride N B : Nat) :
    (typedBwd storage partner gradient v dc a b stride N B).surfaceBody[12]? =
      some (expectedTanh B) := by
  cases storage <;> cases partner <;> cases gradient <;> cases v <;> rfl

/-- Both variants retain every source statement; storage metadata adds no execution step. -/
theorem forward_shape (storage partner : Storage) (v : Bool) (a b c : RegionName)
    (stride N B : Nat) :
    (typedFwd storage partner v a b c stride N B).surfaceBody.length = 15 ∧
    (typedFwd storage partner v a b c stride N B).inputs = [a, b] ∧
    (typedFwd storage partner v a b c stride N B).outputs = [c] := by
  cases storage <;> cases partner <;> cases v <;> exact ⟨rfl, rfl, rfl⟩

/-- Both backward outputs alias their corresponding inputs, with no duplicated
input role or omitted source statement in any storage combination. -/
theorem backward_shape (storage partner gradient : Storage) (v : Bool) (dc a b : RegionName)
    (stride N B : Nat) :
    (typedBwd storage partner gradient v dc a b stride N B).surfaceBody.length = 21 ∧
    (typedBwd storage partner gradient v dc a b stride N B).inputs = [dc, a, b] ∧
    (typedBwd storage partner gradient v dc a b stride N B).outputs = [a, b] := by
  cases storage <;> cases partner <;> cases gradient <;> cases v <;> exact ⟨rfl, rfl, rfl⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.GegluTanhTriton.SourceChecks
