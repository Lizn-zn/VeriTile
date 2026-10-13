import bench.tritonbench_g.swiglu_backward.Kernels
import bench.tritonbench_g.swiglu_backward.Proofs.Source
namespace VeriTile.Bench.TritonBenchG.SwigluBackward.Precision
open VeriTile.Triton
set_option maxHeartbeats 8000000
set_option maxRecDepth 16000

/-- Literal full-source quotations independently check every boundary carrier. -/
def homogeneous (storage : Storage) (v : Bool) : Source :=
  match storage, v with
  | .fp32, Bool.false => swigluBackward%[tl.float32, tl.float32, tl.float32, tl.float32, original]
  | .fp32, Bool.true => swigluBackward%[tl.float32, tl.float32, tl.float32, tl.float32, commuted]
  | .fp16, Bool.false => swigluBackward%[tl.float16, tl.float16, tl.float16, tl.float16, original]
  | .fp16, Bool.true => swigluBackward%[tl.float16, tl.float16, tl.float16, tl.float16, commuted]
  | .bf16, Bool.false => swigluBackward%[tl.bfloat16, tl.bfloat16, tl.bfloat16, tl.bfloat16, original]
  | .bf16, Bool.true => swigluBackward%[tl.bfloat16, tl.bfloat16, tl.bfloat16, tl.bfloat16, commuted]
  | .fp64, Bool.false => swigluBackward%[tl.float64, tl.float64, tl.float64, tl.float64, original]
  | .fp64, Bool.true => swigluBackward%[tl.float64, tl.float64, tl.float64, tl.float64, commuted]
  | .f8e4, Bool.false => swigluBackward%[tl.float8e4nv, tl.float8e4nv, tl.float8e4nv, tl.float8e4nv, original]
  | .f8e4, Bool.true => swigluBackward%[tl.float8e4nv, tl.float8e4nv, tl.float8e4nv, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => swigluBackward%[tl.float8e5, tl.float8e5, tl.float8e5, tl.float8e5, original]
  | .f8e5, Bool.true => swigluBackward%[tl.float8e5, tl.float8e5, tl.float8e5, tl.float8e5, commuted]

/-- The factored source is exactly the complete quotation for every homogeneous
storage, with both constexpr blocks retained for a symbolic flag. -/
theorem homogeneous_source (storage : Storage) (v : Bool)
    (X Y DOUT OUT DX DY : RegionName) (sx sy sd so sdx sdy N B : Nat) (R : Bool) :
    typedKernel storage storage storage storage v X Y DOUT OUT DX DY sx sy sd so sdx sdy N B R =
      homogeneous storage v X Y DOUT OUT DX DY sx sy sd so sdx sdy N B R := by
  cases storage <;> cases v <;> rfl

/-- Independent mixed full-source quotations expose cast interactions. -/
def mixed (v : Bool) : Source :=
  if v then swigluBackward%[tl.float16, tl.float64, tl.bfloat16, tl.float8e5, commuted]
  else swigluBackward%[tl.float16, tl.float64, tl.bfloat16, tl.float8e5, original]

theorem mixed_source (v : Bool)
    (X Y DOUT OUT DX DY : RegionName) (sx sy sd so sdx sdy N B : Nat) (R : Bool) :
    typedKernel .fp16 .fp64 .bf16 .f8e5 v X Y DOUT OUT DX DY sx sy sd so sdx sdy N B R =
      mixed v X Y DOUT OUT DX DY sx sy sd so sdx sdy N B R := by
  cases v <;> rfl

def mixedWide (v : Bool) : Source :=
  if v then swigluBackward%[tl.float64, tl.float8e4nv, tl.float64, tl.float32, commuted]
  else swigluBackward%[tl.float64, tl.float8e4nv, tl.float64, tl.float32, original]

theorem mixed_wide_source (v : Bool)
    (X Y DOUT OUT DX DY : RegionName) (sx sy sd so sdx sdy N B : Nat) (R : Bool) :
    typedKernel .fp64 .f8e4 .fp64 .fp32 v X Y DOUT OUT DX DY sx sy sd so sdx sdy N B R =
      mixedWide v X Y DOUT OUT DX DY sx sy sd so sdx sdy N B R := by
  cases v <;> rfl

/-- Every profile has all eighteen outer statements, including both conditional
blocks. Inputs and all output roles remain explicit even when OUT is inactive. -/
theorem source_shape (input upstream gradient output : Storage) (v : Bool)
    (X Y DOUT OUT DX DY : RegionName) (sx sy sd so sdx sdy N B : Nat) (R : Bool) :
    (typedKernel input upstream gradient output v X Y DOUT OUT DX DY sx sy sd so sdx sdy N B R).surfaceBody.length = 18 ∧
    (typedKernel input upstream gradient output v X Y DOUT OUT DX DY sx sy sd so sdx sdy N B R).inputs = [X,Y,DOUT] ∧
    (typedKernel input upstream gradient output v X Y DOUT OUT DX DY sx sy sd so sdx sdy N B R).outputs = [DX,DY,OUT] := by
  cases v <;> refine ⟨rfl, ?_⟩
  all_goals simp [ComputeKernel.inputs, ComputeKernel.outputs, typedKernel, template, ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?,
    ComputeKernel.surfaceBody, ComputeStmt.listToAlgorithm?, load_projects, store_projects,
    ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    recomputedValue, Except.bind, bind]

/-- All profiles preserve the full typed load order and precisely the original
three masks and zero-padding behavior. Both input tensors precede upstream. -/
theorem source_loads (input upstream gradient output : Storage) (v : Bool)
    (X Y DOUT OUT DX DY : RegionName) (sx sy sd so sdx sdy N B : Nat) (R : Bool) :
    ((typedKernel input upstream gradient output v X Y DOUT OUT DX DY sx sy sd so sdx sdy N B R).surfaceBody.drop 9).take 3 =
      [typedLoad "X" "x" input B N, typedLoad "Y" "y" input B N,
       typedLoad "DOUT" "dout" upstream B N] := by cases v <;> rfl

/-- DX is stored before DY; the optional recomputation and OUT store follow
both gradient stores. This checks the complete suffix for every profile. -/
theorem source_stores (input upstream gradient output : Storage) (v : Bool)
    (X Y DOUT OUT DX DY : RegionName) (sx sy sd so sdx sdy N B : Nat) (R : Bool) :
    (typedKernel input upstream gradient output v X Y DOUT OUT DX DY sx sy sd so sdx sdy N B R).surfaceBody.drop 15 =
      [typedStore "DX" "dx" gradient B N, typedStore "DY" "dy" gradient B N,
       .ifThen (.constBool R) [recomputedValue B, typedStore "OUT" "out" output B N]] := by
  cases v <;> rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SwigluBackward.Precision
