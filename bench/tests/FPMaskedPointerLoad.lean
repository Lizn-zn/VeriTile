import VeriTile.Triton.DSL
import VeriTile.Meta.StatementAudit

/-! Positional masks must use the same pointer storage type as keyword masks.
Both binding spellings, pointer aliases, and implicit `other` conversions are
covered; erasing types cannot detect these frontend regressions. -/
namespace FPMaskedPointerLoadTests
open VeriTile.Triton

def positional (X : RegionName) (B : Nat) : ComputeKernel := triton {
  tl.region X = tl.float16
  offsets = tl.arange(0, $(B))
  ptrs = X + offsets
  copy_ptrs = ptrs
  x = tl.load(copy_ptrs, offsets < $(B), other=0.0)
}
def keyword (X : RegionName) (B : Nat) : ComputeKernel := triton {
  tl.region X = tl.float16
  offsets = tl.arange(0, $(B))
  ptrs = X + offsets
  copy_ptrs = ptrs
  x = tl.load(copy_ptrs, mask=offsets < $(B), other=0.0)
}

theorem positional_and_keyword_agree (X : RegionName) (B : Nat) :
    positional X B = keyword X B := rfl

theorem fp16_load_and_zero_conversion (X : RegionName) (B : Nat) :
    (positional X B).surfaceBody.getLast? = some
      (.assign .fp16 [B] "x" (.alg
        (.load .fp16 (.ptr (.ref .ptr [B] "copy_ptrs"))
          (.maskOther (.lt .nat .scalarR (.ref .nat [B] "offsets") (.constNat B))
            (.broadcast (.castFloat .real .fp16 (.const 0.0)) [B]))))) := rfl

def bfloatPositional (X : RegionName) (B : Nat) : ComputeKernel := triton {
  tl.region X = tl.bfloat16
  offsets := tl.arange(0, $(B))
  ptrs := X + offsets
  x := tl.load(ptrs, offsets < $(B), other=0.0)
}

theorem bf16_load_and_zero_conversion (X : RegionName) (B : Nat) :
    (bfloatPositional X B).surfaceBody.getLast? = some
      (.assign .bf16 [B] "x" (.alg
        (.load .bf16 (.ptr (.ref .ptr [B] "ptrs"))
          (.maskOther (.lt .nat .scalarR (.ref .nat [B] "offsets") (.constNat B))
            (.broadcast (.castFloat .real .bf16 (.const 0.0)) [B]))))) := rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end FPMaskedPointerLoadTests
