import bench.tritonbench_g.dropout_triton.Kernels
import VeriTile.Triton.Float.Erasure
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.DropoutTriton.Erasure
open Lean VeriTile.Triton TypedSource
set_option maxHeartbeats 3000000
set_option maxRecDepth 20000

/-- Canonical real body: identity precision conversions are absent after erasure. -/
syntax "dropoutReference%" "[" tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(dropoutReference%[$storage:tritonDType, $keep:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `indexCommuted do
      Macro.throwError "expected original or indexCommuted"
    let blockStart : TSyntax `tritonStmt ←
      if version.getId == `indexCommuted then `(tritonStmt| block_start = $(BLOCK_SIZE) * pid)
      else `(tritonStmt| block_start = pid * $(BLOCK_SIZE))
    `(fun (x_ptr x_keep_ptr output_ptr : RegionName) (n_elements BLOCK_SIZE : Nat) (p : ℝ) =>
      ComputeKernel.mk [x_ptr, x_keep_ptr] [output_ptr] (triton {
  tl.region x_ptr = $storage, x_keep_ptr = $keep, output_ptr = $storage
  pid = tl.program_id(axis=0)
  $blockStart
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  x = tl.load(x_ptr + offsets, mask=mask)
  x_keep = tl.load(x_keep_ptr + offsets, mask=mask)
  output = tl.where(x_keep, x / (1 - $(p)), 0.0)
  tl.store(output_ptr + offsets, output, mask=mask)
}).surfaceBody)

def source (keep : KeepStorage) (swapped : Bool) : sourceType :=
  match keep, swapped with
  | .int32, Bool.false => dropoutReference%[tl.float64, tl.int32, original]
  | .int32, Bool.true => dropoutReference%[tl.float64, tl.int32, indexCommuted]
  | .bool, Bool.false => dropoutReference%[tl.float64, tl.int1, original]
  | .bool, Bool.true => dropoutReference%[tl.float64, tl.int1, indexCommuted]

theorem body (storage : Storage) (keep : KeepStorage) (swapped : Bool)
    (x mask out : RegionName) (n B : Nat) (p : ℝ) :
    (TypedSource.source storage keep swapped x mask out n B p).eraseDType.toAlgKernel.body =
      (source keep swapped x mask out n B p).toAlgKernel.body := by
  cases storage <;> cases keep <;> cases swapped
  all_goals simp only [TypedSource.source,source,
    ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType,List.map_cons,List.map_nil,Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType,MemAccess.eraseDType,MaskOpt.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    Except.bind,bind]
  all_goals simp only [Op.eraseDType.eq_def,VeriTile.Triton.eraseDType,FloatDType.toTileDType]
  all_goals rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.DropoutTriton.Erasure
