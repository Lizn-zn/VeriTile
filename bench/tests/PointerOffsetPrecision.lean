import VeriTile.Triton.DSL
import VeriTile.Triton.Source
import VeriTile.Meta.StatementAudit

/-! Offset expressions must inherit their pointer's element dtype at the
current program point, including nested control flow and pointer rebinding. -/
namespace PointerOffsetPrecisionTests
open VeriTile.Triton

private def assignments := triton {
  tl.region A = tl.float16, B = tl.bfloat16, C = tl.float32
  p = A + tl.arange(0, 4)
  a = tl.load(p + $(1))
  p = B + tl.arange(0, 4)
  b = tl.load((p + $(2)) + $(3))
  p = C + tl.arange(0, 4)
  c = tl.load(p + $(4))
}
where
  A : RegionName := "A"
  B : RegionName := "B"
  C : RegionName := "C"

private def assignment (code : List ComputeStmt) (name : RegName) :=
  (code.drop (Source.prefixBefore code name).length).head?

private def loadDType? : ComputeStmt → Option TileDType
  | .assign _ _ _ (.alg (.load dtype _ _)) => some dtype
  | .assign _ _ _ (.compute (.load .fp32 _ _)) => some .fp32
  | _ => none

private def assignedLoad (code : List ComputeStmt) (name : RegName) :=
  (assignment code name).bind loadDType?

theorem offsets_keep_each_binding :
    assignedLoad assignments.surfaceBody "a" = some .fp16 ∧
    assignedLoad assignments.surfaceBody "b" = some .bf16 ∧
    assignedLoad assignments.surfaceBody "c" = some .fp32 := by
  exact ⟨rfl, rfl, rfl⟩

private def nested (enabled : Bool) : ComputeKernel :=
  let A : RegionName := "A"
  triton {
    tl.region A = tl.float16
    p = A + $(1)
    limit = $(2)
    if enabled {
      for i in range($(0), limit, $(1)) {
        q = p + i
        x = tl.load(q + tl.arange(0, 4))
      }
    }
  }

private def loopBody (enabled : Bool) : List ComputeStmt :=
  match Source.conditional? (nested enabled).surfaceBody (fun _ => Bool.true) with
  | none => []
  | some branch => ((Source.dynamicRange? branch.yes "i").map (fun (r : Source.DynamicRange) => r.body)).getD []

theorem nested_offset_load (enabled : Bool) :
    assignedLoad (loopBody enabled) "x" = some .fp16 := rfl

/-- Unknown untyped bases keep their existing interpretation; finiteness or
pointer arithmetic alone supplies no storage-precision declaration. -/
private def untyped : ComputeKernel :=
  let A : RegionName := "A"
  triton {
    p = A + $(1)
    x = tl.load(p + tl.arange(0, 4))
  }

theorem untyped_base_stays_untyped : assignedLoad untyped.surfaceBody "x" = some .real := rfl

/-- Parameterized storage declarations preserve the parameter on both aliased
loads. The output conversion stays explicit in the source. -/
private def parametric (dtype out_dtype : FloatDType) : ComputeKernel :=
  let A : RegionName := "A"
  let O : RegionName := "O"
  triton {
    tl.region A = DTYPE, O = OUT_DTYPE
    p = A + $(1)
    q = O + $(2)
    x = tl.load(p + tl.arange(0, 4))
    y = tl.load(q + tl.arange(0, 4))
    tl.store(q + tl.arange(0, 4), (x).to(OUT_DTYPE))
  }

theorem parametric_loads (dtype out_dtype : FloatDType) :
    assignedLoad (parametric dtype out_dtype).surfaceBody "x" = some dtype.toTileDType ∧
    assignedLoad (parametric dtype out_dtype).surfaceBody "y" = some out_dtype.toTileDType := by
  cases dtype <;> cases out_dtype <;> exact ⟨rfl, rfl⟩

theorem parametric_store (dtype out_dtype : FloatDType) :
    (parametric dtype out_dtype).surfaceBody.getLast? = some
      (.store out_dtype.toTileDType [4]
        (.ptr (.ptrAdd .scalarL (.ref .ptr [] "q") (.arange 4)))
        (.alg (.castFloat dtype out_dtype (.ref dtype.toTileDType [4] "x"))) .none) := rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end PointerOffsetPrecisionTests
