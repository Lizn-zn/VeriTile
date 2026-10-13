import bench.tritonbench_g.kldiv_triton.Kernels
import bench.tritonbench_g.kldiv_triton.Proofs.IndexRewrite

namespace VeriTile.Bench.TritonBenchG.KldivTriton.FPEquiv
open VeriTile Triton FP.ProfiledRewrite TypedSource
open scoped VeriTile.Spec
set_option maxHeartbeats 16000000
set_option maxRecDepth 16000

def assumptions : Spec.Assumptions FP.GuardedFragment := []

def forwardProgram (pred : Storage) (target : MathStorage) (output : Storage) (v : Bool)
    (Y T O : RegionName) (sy st so N B : Nat) (logTarget : Bool) (reduction : Nat) :
    Program where
  kernel := forward pred target output v Y T O sy st so N B logTarget reduction
  domain := []

def backwardDefaultProgram (target output : Storage) (v : Bool)
    (T O : RegionName) (st so N B : Nat) : Program where
  kernel := backwardDefault target output v T O st so N B
  domain := []

def backwardLogProgram (target : MathStorage) (output : Storage) (v : Bool)
    (T O : RegionName) (st so N B : Nat) : Program where
  kernel := backwardLog target output v T O st so N B
  domain := []

/-- Full forward state/failure equivalence retains every chunk store and both source
branches. It imposes no floating law, successful-execution or output-value premise. -/
specification forward_equivalence (pred : Storage) (target : MathStorage) (output : Storage)
    (Y T O : RegionName) (sy st so N B : Nat) (logTarget : Bool) (reduction : Nat) :
    forwardProgram pred target output Bool.false Y T O sy st so N B logTarget reduction
      ≡[assumptions]
    forwardProgram pred target output Bool.true Y T O sy st so N B logTarget reduction := by
  cases pred <;> cases target <;> cases output <;> equiv_decompose <;> fp_prove

/-- Complete default backward AST, including aliases, arbitrary backend behavior,
masked-load values, empty rows and multiple blocks. The fp8 target branches are
a formal extension; pinned source support is recorded by BackwardSourceSupported. -/
specification backward_default_equivalence (target output : Storage)
    (T O : RegionName) (st so N B : Nat) :
    backwardDefaultProgram target output Bool.false T O st so N B ≡[assumptions]
      backwardDefaultProgram target output Bool.true T O st so N B := by
  cases target <;> cases output <;> equiv_decompose <;> fp_prove

/-- Complete log-target backward source; the only rewrite is signed address multiplication. -/
specification backward_log_equivalence (target : MathStorage) (output : Storage)
    (T O : RegionName) (st so N B : Nat) :
    backwardLogProgram target output Bool.false T O st so N B ≡[assumptions]
      backwardLogProgram target output Bool.true T O st so N B := by
  cases target <;> cases output <;> equiv_decompose <;> fp_prove

#print_fp_assumptions forward_equivalence
#print_fp_assumptions backward_default_equivalence
#print_fp_assumptions backward_log_equivalence
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.KldivTriton.FPEquiv
