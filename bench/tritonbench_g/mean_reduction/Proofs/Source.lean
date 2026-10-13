import bench.tritonbench_g.mean_reduction.Kernels
import VeriTile.Triton
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.MeanReduction
open VeriTile.Triton
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000

def indexStmt (commuted : Bool) (BM : Nat) : Stmt :=
  .assign .nat [BM, 1] "pid" (.add .nat .scalarL
    (if commuted then .mul .nat .nil (.constNat BM) (.programId 0)
      else .mul .nat .nil (.programId 0) (.constNat BM))
    (.expandDim ⟨1, by simp⟩ (.arange BM)))

def mathKernel (commuted : Bool) (X Mean : RegionName) (M N BM BN : Nat) : Kernel :=
  .mk [X] [Mean] (indexStmt commuted BM ::
    (mean_dim_kernel X Mean M N BM BN).toAlgKernel.body.drop 1)

theorem typed_projects (input output : Storage) (v : Bool) (X Mean : RegionName)
    (M N BM BN : Nat) :
    (typedKernel input output v X Mean M N BM BN).toAlgorithm? =
      .ok (typedKernel input output v X Mean M N BM BN).toAlgKernel := by
  cases input <;> cases output <;> cases v <;> rfl

theorem typed_supported (input output : Storage) (v : Bool) (X Mean : RegionName)
    (M N BM BN : Nat) :
    (typedKernel input output v X Mean M N BM BN).eraseDType.toAlgorithm? =
      .ok (typedKernel input output v X Mean M N BM BN).eraseDType.toAlgKernel := by
  simp only [ComputeKernel.eraseDType, typed_projects,
    ComputeKernel.toAlgorithm?_fromKernelBody, ComputeKernel.toAlgKernel_fromKernelBody]

theorem typed_projection (input output : Storage) (v : Bool) (X Mean : RegionName)
    (M N BM BN : Nat) :
    (typedKernel input output v X Mean M N BM BN).eraseDType.toAlgKernel =
      mathKernel v X Mean M N BM BN := by
  simp only [ComputeKernel.eraseDType, typed_projects, ComputeKernel.toAlgKernel_fromKernelBody]
  cases input <;> cases output <;> cases v
  all_goals simp only [typedKernel, mathKernel, indexStmt, Bool.false_eq_true, ↓reduceIte,
    mean_dim_kernel, ComputeKernel.surfaceBody, ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, Except.bind, bind, Kernel.eraseDType,
    Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType,
    VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
    FloatDType.toTileDType, ComputeDType.eraseDType]
  all_goals try simp only [Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def,
    MaskOpt.eraseDType.eq_def, VeriTile.Triton.eraseDType, NumericDType.eraseDType,
    FloatDType.toTileDType, ComputeDType.eraseDType]
  all_goals rfl

/-- Rebound pointers retain the intended source ports. -/
theorem io_signature (input output : Storage) (v : Bool) (X Mean : RegionName)
    (M N BM BN : Nat) :
    (typedKernel input output v X Mean M N BM BN).inputs = [X] ∧
      (typedKernel input output v X Mean M N BM BN).outputs = [Mean] := by
  cases input <;> cases output <;> cases v <;> exact ⟨rfl, rfl⟩

def loopAssignment (kernel : ComputeKernel) (index : Nat) : Option ComputeStmt :=
  match (kernel.surfaceBody[5]? : Option ComputeStmt) with
  | some (.forRange _ _ _ _ body) => body[index]?
  | _ => none

def loadAddress (dtype : TileDType) (BM BN : Nat) : MemAccess dtype [BM, BN] :=
  .ptr (.ptrAdd Broadcast.nil.consL.consR
    (.ref .ptr [BM, 1] "X") (.ref .nat [1, BN] "cols"))

def expectedLoad (input : Storage) (BM BN : Nat) : ComputeStmt :=
  match input with
  | .fp32 => .assign .real [BM, BN] "a" (.compute (.alg .fp32
      (.castFloat .real .real (.load .real (loadAddress .real BM BN)
        (.maskOther (.ref .bool [BM, BN] "mask")
          ((Op.const 0.0).broadcast [BM, BN]))))))
  | .fp64 => .assign .real [BM, BN] "a" (.compute (.alg .fp32
      (.castFloat .real .real (.withPrecision .real .fp64
        (.load .real (loadAddress .real BM BN)
          (.maskOther (.ref .bool [BM, BN] "mask")
            ((Op.const 0.0).broadcast [BM, BN])))))))
  | .fp16 => .assign .real [BM, BN] "a" (.compute (.alg .fp32
      (.castFloat .fp16 .real (.load .fp16 (loadAddress .fp16 BM BN)
        (.maskOther (.ref .bool [BM, BN] "mask")
          ((Op.castFloat .real .fp16 (.const 0.0)).broadcast [BM, BN]))))))
  | .bf16 => .assign .real [BM, BN] "a" (.compute (.alg .fp32
      (.castFloat .bf16 .real (.load .bf16 (loadAddress .bf16 BM BN)
        (.maskOther (.ref .bool [BM, BN] "mask")
          ((Op.castFloat .real .bf16 (.const 0.0)).broadcast [BM, BN]))))))

/-- Every input profile converts the masked zero-padded load to fp32, including
the wrapper's fp64-input case. Half/bfloat reads retain their source storage. -/
theorem load_precision (input output : Storage) (v : Bool) (X Mean : RegionName)
    (M N BM BN : Nat) :
    loopAssignment (typedKernel input output v X Mean M N BM BN) 3 =
      some (expectedLoad input BM BN) := by
  cases input <;> cases output <;> cases v <;> rfl

def expectedStore (output : Storage) (BM : Nat) : ComputeStmt :=
  match output with
  | .fp32 => .store .real [BM, 1] (.ptr (.ref .ptr [BM, 1] "Mean"))
      (.alg (.ref .real [BM, 1] "mean"))
      (.mask (.ref .bool [BM, 1] "row_mask"))
  | .fp64 => .store .real [BM, 1] (.ptr (.ref .ptr [BM, 1] "Mean"))
      (.compute (.alg .fp64 (.castFloat .real .real
        (.withPrecision .real .fp32 (.ref .real [BM, 1] "mean")))))
      (.mask (.ref .bool [BM, 1] "row_mask"))
  | .fp16 => .store .fp16 [BM, 1] (.ptr (.ref .ptr [BM, 1] "Mean"))
      (.alg (.castFloat .real .fp16 (.ref .real [BM, 1] "mean")))
      (.mask (.ref .bool [BM, 1] "row_mask"))
  | .bf16 => .store .bf16 [BM, 1] (.ptr (.ref .ptr [BM, 1] "Mean"))
      (.alg (.castFloat .real .bf16 (.ref .real [BM, 1] "mean")))
      (.mask (.ref .bool [BM, 1] "row_mask"))

/-- Stores use the requested output carrier and precisely one explicit
conversion when leaving the fp32 accumulator carrier. -/
theorem store_precision (input output : Storage) (v : Bool) (X Mean : RegionName)
    (M N BM BN : Nat) :
    (typedKernel input output v X Mean M N BM BN).surfaceBody.getLast? =
      some (expectedStore output BM) := by
  cases input <;> cases output <;> cases v <;> rfl

/-- The full loop accumulates the converted tile in fp32 without reassociation. -/
theorem accumulator_precision (input output : Storage) (v : Bool) (X Mean : RegionName)
    (M N BM BN : Nat) :
    loopAssignment (typedKernel input output v X Mean M N BM BN) 4 =
      some (.assign .real [BM, BN] "_mean" (.compute (.alg .fp32
        (.add .real Broadcast.nil.consSame.consSame
          (.ref .real [BM, BN] "_mean") (.ref .real [BM, BN] "a"))))) := by
  cases input <;> cases output <;> cases v <;> rfl

/-- The source sums the accumulated column lanes before one fp32 division by N. -/
theorem reduction_precision (input output : Storage) (v : Bool) (X Mean : RegionName)
    (M N BM BN : Nat) :
    (typedKernel input output v X Mean M N BM BN).surfaceBody[6]? =
      some (.assign .real [BM] "mean" (.compute (.alg .fp32
        (.div .real .scalarR
          (.reduceSum ⟨1, by simp⟩ Bool.false (.ref .real [BM, BN] "_mean"))
          (.const (N : ℝ)))))) := by
  cases input <;> cases output <;> cases v <;> rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.MeanReduction
