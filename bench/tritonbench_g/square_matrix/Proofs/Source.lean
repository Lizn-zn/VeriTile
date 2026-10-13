import bench.tritonbench_g.square_matrix.Kernels
import VeriTile.Triton
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.SquareMatrix
open VeriTile.Triton
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000

def mathKernel (commuted : Bool) : Source :=
  if commuted then squareSource%[tl.float64, indexCommuted]
  else fun out inp si so n B =>
    ComputeKernel.mk [inp] [out] (square_kernel out inp si so n B).surfaceBody

theorem typed_projects (storage : Storage) (v : Bool) (out inp : RegionName)
    (si so n B : Nat) :
    (typedKernel storage v out inp si so n B).toAlgorithm? =
      .ok (typedKernel storage v out inp si so n B).toAlgKernel := by
  cases storage <;> cases v <;> rfl

theorem typed_projection (storage : Storage) (v : Bool) (out inp : RegionName)
    (si so n B : Nat) :
    (typedKernel storage v out inp si so n B).eraseDType.toAlgKernel =
      (mathKernel v out inp si so n B).toAlgKernel := by
  simp only [ComputeKernel.eraseDType, typed_projects, ComputeKernel.toAlgKernel_fromKernelBody]
  cases storage <;> cases v
  all_goals simp only [typedKernel, mathKernel, Bool.false_eq_true, ↓reduceIte, square_kernel,
    ComputeKernel.surfaceBody, ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, Except.bind, bind,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType,
    VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
    FloatDType.toTileDType, ComputeDType.eraseDType]
  all_goals try simp only [Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def,
    MaskOpt.eraseDType.eq_def, VeriTile.Triton.eraseDType, NumericDType.eraseDType,
    FloatDType.toTileDType, ComputeDType.eraseDType]
  all_goals rfl

theorem typed_supported (storage : Storage) (v : Bool) (out inp : RegionName)
    (si so n B : Nat) :
    (typedKernel storage v out inp si so n B).eraseDType.toAlgorithm? =
      .ok (typedKernel storage v out inp si so n B).eraseDType.toAlgKernel := by
  simp only [ComputeKernel.eraseDType, typed_projects,
    ComputeKernel.toAlgorithm?_fromKernelBody, ComputeKernel.toAlgKernel_fromKernelBody]

/-- Alias-based pointers retain the actual input/output metadata. -/
theorem io_signature (storage : Storage) (v : Bool) (out inp : RegionName)
    (si so n B : Nat) :
    (typedKernel storage v out inp si so n B).inputs = [inp] ∧
      (typedKernel storage v out inp si so n B).outputs = [out] := by
  cases storage <;> cases v <;> exact ⟨rfl, rfl⟩

def expectedStore (storage : Storage) (n B : Nat) : ComputeStmt :=
  match storage with
  | .fp32 => .store .real [B] (.ptr (.ref .ptr [B] "output_ptrs"))
      (.compute (.alg .fp32 (.ref .real [B] "square_output")))
      (.mask (.lt .nat .scalarR (.ref .nat [B] "col_offsets") (.constNat n)))
  | .fp16 => .store .fp16 [B] (.ptr (.ref .ptr [B] "output_ptrs"))
      (.alg (.ref .fp16 [B] "square_output"))
      (.mask (.lt .nat .scalarR (.ref .nat [B] "col_offsets") (.constNat n)))
  | .bf16 => .store .bf16 [B] (.ptr (.ref .ptr [B] "output_ptrs"))
      (.alg (.ref .bf16 [B] "square_output"))
      (.mask (.lt .nat .scalarR (.ref .nat [B] "col_offsets") (.constNat n)))

/-- Multiplication and output storage retain the source precision through aliases. -/
theorem store_precision (storage : Storage) (v : Bool) (out inp : RegionName)
    (si so n B : Nat) :
    (typedKernel storage v out inp si so n B).surfaceBody.getLast? =
      some (expectedStore storage n B) := by
  cases storage <;> cases v <;> rfl

def expectedProduct (storage : Storage) (B : Nat) : ComputeStmt :=
  match storage with
  | .fp32 => .assign .real [B] "square_output" (.compute (.alg .fp32
      (.mul .real (.consSame .nil) (.ref .real [B] "row") (.ref .real [B] "row"))))
  | .fp16 => .assign .fp16 [B] "square_output" (.alg
      (.mul .fp16 (.consSame .nil) (.ref .fp16 [B] "row") (.ref .fp16 [B] "row")))
  | .bf16 => .assign .bf16 [B] "square_output" (.alg
      (.mul .bf16 (.consSame .nil) (.ref .bf16 [B] "row") (.ref .bf16 [B] "row")))

/-- The product itself uses the source dtype, without a widening and narrowing pair. -/
theorem multiplication_precision (storage : Storage) (v : Bool) (out inp : RegionName)
    (si so n B : Nat) :
    (typedKernel storage v out inp si so n B).surfaceBody[5]? =
      some (expectedProduct storage B) := by
  cases storage <;> cases v <;> rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SquareMatrix
