import VeriTile.Triton.DSL
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.ReduceMaxPromotion
open Lean VeriTile.Triton

/-! Triton v3.5.1 standard.py, max (without indices), widens sub-32-bit
floating input to fp32 before reducing. These regressions check both the
operation and the result precision consumed by following arithmetic.
https://github.com/triton-lang/triton/blob/v3.5.1/python/triton/language/standard.py#L162-L176 -/

syntax "smallMax%" "[" tritonDType "]" : term
set_option hygiene false in
macro_rules
  | `(smallMax%[$dtype:tritonDType]) =>
    `(fun (x : RegionName) (B : Nat) => triton {
      tl.region x = $dtype
      inp = tl.load(x + tl.arange(0, $(B)))
      positional = tl.max(inp, 0)
      named = tl.max(inp, axis=0)
      all_axes = tl.max(inp)
      kept = tl.max(inp, axis=0, keep_dims=true)
      delta = inp - positional
    })

def halfMax := smallMax%[tl.float16]
def bfloatMax := smallMax%[tl.bfloat16]
def e4Max := smallMax%[tl.float8e4nv]
def e5Max := smallMax%[tl.float8e5]

private def reduction (dtype : FloatDType) (name : String) (B : Nat) : ComputeStmt :=
  .assign .real [] name (.compute (.alg .fp32
    (.reduceMax ⟨0, by simp⟩ Bool.false
      (.castFloat dtype .real (.ref dtype.toTileDType [B] "inp")))))

private def keptReduction (dtype : FloatDType) (B : Nat) : ComputeStmt :=
  .assign .real [1] "kept" (.compute (.alg .fp32
    (.reduceMax ⟨0, by simp⟩ Bool.true
      (.castFloat dtype .real (.ref dtype.toTileDType [B] "inp")))))

private def difference (dtype : FloatDType) (B : Nat) : ComputeStmt :=
  .assign .real [B] "delta" (.compute (.alg .fp32
    (.sub .real .scalarR
      (.castFloat dtype .real (.ref dtype.toTileDType [B] "inp"))
      (.ref .real [] "positional"))))

private def expectedTail (dtype : FloatDType) (B : Nat) : List ComputeStmt :=
  [reduction dtype "positional" B, reduction dtype "named" B,
    reduction dtype "all_axes" B, keptReduction dtype B, difference dtype B]

example (x : RegionName) (B : Nat) :
    (halfMax x B).surfaceBody.drop 1 = expectedTail .fp16 B := rfl
example (x : RegionName) (B : Nat) :
    (bfloatMax x B).surfaceBody.drop 1 = expectedTail .bf16 B := rfl
example (x : RegionName) (B : Nat) :
    (e4Max x B).surfaceBody.drop 1 = expectedTail .f8e4 B := rfl
example (x : RegionName) (B : Nat) :
    (e5Max x B).surfaceBody.drop 1 = expectedTail .f8e5 B := rfl

def singleMax (x : RegionName) (B : Nat) : ComputeKernel := triton {
  tl.region x = tl.float32
  inp = tl.load(x + tl.arange(0, $(B)))
  m = tl.max(inp, 0)
}

/-- The existing explicit fp32 profile and absence of an extra cast are retained. -/
example (x : RegionName) (B : Nat) :
    (singleMax x B).surfaceBody.getLast? = some
      (.assign .real [] "m" (.compute (.alg .fp32
        (.reduceMax ⟨0, by simp⟩ Bool.false (.ref .real [B] "inp"))))) := rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Tests.ReduceMaxPromotion
