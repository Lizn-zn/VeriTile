import bench.tritonbench_g.swiglu_triton.Kernels

namespace VeriTile.Bench.TritonBenchG.SwigluTriton.ArithmeticPrecision
open VeriTile.Triton
set_option maxHeartbeats 8000000
set_option maxRecDepth 16000

/-- The parent double product cannot widen the inlined fp32 silu computation. -/
def expectedForward64 (B : Nat) : ComputeStmt :=
  .assign .real [B] "c_row" (.compute (.alg .fp64
    (.mul .real (.consSame .nil)
      (.withPrecision .real .fp32
        (.mul .real (.consSame .nil) (.ref .real [B] "a_row")
          (.sigmoid (.ref .real [B] "a_row"))))
      (.ref .real [B] "b_row"))))

theorem forward_double_partner (storage : Storage) (v : Bool)
    (a b c : RegionName) (stride N B : Nat) :
    (typedFwd storage .fp64 v a b c stride N B).surfaceBody[8]? =
      some (expectedForward64 B) := by
  cases storage <;> cases v <;> rfl

def factor (B : Nat) : Op .real [B] :=
  .add .real (.consSame .nil)
    (.mul .real (.consSame .nil) (.ref .real [B] "silu_a")
      (.sub .real .scalarL (.const 1) (.ref .real [B] "sig_a")))
    (.ref .real [B] "sig_a")

/-- Both small inputs widen separately; there is no half/bfloat intermediate product. -/
def expectedHalfDa (reverse : Bool) (B : Nat) : ComputeStmt :=
  let d : FloatDType := if reverse then .bf16 else .fp16
  let b : FloatDType := if reverse then .fp16 else .bf16
  .assign .real [B] "da_row" (.compute (.alg .fp32
    (.mul .real (.consSame .nil)
      (.mul .real (.consSame .nil)
        (.castFloat d .real (.ref d.toTileDType [B] "dc_row")) (factor B))
      (.castFloat b .real (.ref b.toTileDType [B] "b_row")))))

theorem backward_small_inputs (storage : Storage) (v reverse : Bool)
    (dc a b : RegionName) (stride N B : Nat) :
    (typedBwd storage (if reverse then .fp16 else .bf16)
      (if reverse then .bf16 else .fp16) v dc a b stride N B).surfaceBody[12]? =
      some (expectedHalfDa reverse B) := by
  cases storage <;> cases v <;> cases reverse <;> rfl

/-- A double partner widens after the first product; a double upstream gradient
widens before it. The independent sigmoid factor remains fp32 in both cases. -/
def expectedDoubleDa (doubleGradient : Bool) (B : Nat) : ComputeStmt :=
  .assign .real [B] "da_row" (.compute (.alg .fp64
    (if doubleGradient then
      .mul .real (.consSame .nil)
        (.mul .real (.consSame .nil) (.ref .real [B] "dc_row")
          (.withPrecision .real .fp32 (factor B)))
        (.withPrecision .real .fp32 (.ref .real [B] "b_row"))
    else
      .mul .real (.consSame .nil)
        (.withPrecision .real .fp32
          (.mul .real (.consSame .nil) (.ref .real [B] "dc_row") (factor B)))
        (.ref .real [B] "b_row"))))

theorem backward_double_inputs (storage : Storage) (v doubleGradient : Bool)
    (dc a b : RegionName) (stride N B : Nat) :
    (typedBwd storage (if doubleGradient then .fp32 else .fp64)
      (if doubleGradient then .fp64 else .fp32) v dc a b stride N B).surfaceBody[12]? =
      some (expectedDoubleDa doubleGradient B) := by
  cases storage <;> cases v <;> cases doubleGradient <;> rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SwigluTriton.ArithmeticPrecision
