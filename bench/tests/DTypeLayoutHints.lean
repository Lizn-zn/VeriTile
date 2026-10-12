import VeriTile.Triton.DSL
import VeriTile.Triton.Source
import VeriTile.Meta.StatementAudit

/-! A layout hint around a pointer must preserve its element storage. These
full parser examples exercise nested hints, a following pointer alias and a
load/cast chain. Unknown storage must keep the existing real fallback. -/
namespace VeriTile.Bench.Tests.DTypeLayoutHints
open Lean VeriTile.Triton

syntax "hintedPointer%" "[" tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(hintedPointer%[$dtype:tritonDType, $kind:ident]) => do
    let bind : TSyntax `tritonStmt ← if kind.getId == `hinted then
      `(tritonStmt| p = tl.max_contiguous(tl.multiple_of(G + offs, $(N)), $(N)))
      else `(tritonStmt| p = G + offs)
    `(fun (G O : RegionName) (N : Nat) => (triton {
      tl.region G = $dtype, O = $dtype
      offs = tl.arange(0, $(N))
      $bind
      p_next = p + $(1)
      value = tl.load(p_next, mask=offs < $(N), other=0)
      wide = tl.load(p_next, mask=offs < $(N), other=0).to(tl.float32)
      tl.store(O + offs, value)
    } : ComputeKernel))

def halfHinted := hintedPointer%[tl.float16, hinted]
def halfPlain := hintedPointer%[tl.float16, plain]
def bf16Hinted := hintedPointer%[tl.bfloat16, hinted]
def bf16Plain := hintedPointer%[tl.bfloat16, plain]

theorem half_storage_preserved (G O : RegionName) (N : Nat) :
    (halfHinted G O N).surfaceBody = (halfPlain G O N).surfaceBody := rfl

theorem bf16_storage_preserved (G O : RegionName) (N : Nat) :
    (bf16Hinted G O N).surfaceBody = (bf16Plain G O N).surfaceBody := rfl

private def assignment (body : List ComputeStmt) (name : RegName) :=
  (body.drop (Source.prefixBefore body name).length).head?

theorem half_load_before_widening (G O : RegionName) (N : Nat) :
    assignment (halfHinted G O N).surfaceBody "wide" = some
      (.assign .real [N] "wide" (.compute (.alg .fp32 (.castFloat .fp16 .real
        (.load .fp16 (.ptr (.ref .ptr [N] "p_next"))
          (.maskOther (.lt .nat .scalarR (.ref .nat [N] "offs") (.constNat N))
            (.broadcast (.castFloat .real .fp16 (.const 0)) [N]))))))) := rfl

def untypedHinted (G : RegionName) (N : Nat) : ComputeKernel := triton {
  p = tl.multiple_of(tl.max_contiguous(G + tl.arange(0, $(N)), $(N)), $(N))
  value = tl.load(p)
}
def untypedPlain (G : RegionName) (N : Nat) : ComputeKernel := triton {
  p = G + tl.arange(0, $(N))
  value = tl.load(p)
}

theorem untyped_fallback_preserved (G : RegionName) (N : Nat) :
    (untypedHinted G N).surfaceBody = (untypedPlain G N).surfaceBody := rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Tests.DTypeLayoutHints
