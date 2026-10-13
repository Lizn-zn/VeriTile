import bench.tritonbench_g.log_softmax.Kernels
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.LogSoftmax.Precision
open VeriTile.Triton
set_option maxHeartbeats 12000000
set_option maxRecDepth 16000

def expectedLoad (pointer name : RegName) (storage : Storage) (other : Bool)
    (R C : Nat) : ComputeStmt :=
  match storage with
  | .fp32 => .assign .real [R,C] name (.compute (.alg .fp32
      (.castFloat .real .real (.load .real (.ptr (.ref .ptr [R,C] pointer))
        (if other then .maskOther (.ref .bool [R,C] "mask") ((Op.negInf).broadcast [R,C])
          else .mask (.ref .bool [R,C] "mask"))))))
  | .fp16 => .assign .real [R,C] name (.compute (.alg .fp32
      (.castFloat .fp16 .real (.load .fp16 (.ptr (.ref .ptr [R,C] pointer))
        (if other then .maskOther (.ref .bool [R,C] "mask") ((Op.castFloat .real .fp16 Op.negInf).broadcast [R,C])
          else .mask (.ref .bool [R,C] "mask"))))))
  | .bf16 => .assign .real [R,C] name (.compute (.alg .fp32
      (.castFloat .bf16 .real (.load .bf16 (.ptr (.ref .ptr [R,C] pointer))
        (if other then .maskOther (.ref .bool [R,C] "mask") ((Op.castFloat .real .bf16 Op.negInf).broadcast [R,C])
          else .mask (.ref .bool [R,C] "mask"))))))
  | .fp64 => .assign .real [R,C] name (.compute (.alg .fp32
      (.castFloat .real .real (.withPrecision .real .fp64 (.load .real (.ptr (.ref .ptr [R,C] pointer))
        (if other then .maskOther (.ref .bool [R,C] "mask") ((Op.negInf).broadcast [R,C])
          else .mask (.ref .bool [R,C] "mask")))))))
  | .f8e4 => .assign .real [R,C] name (.compute (.alg .fp32
      (.castFloat .f8e4 .real (.load .f8e4 (.ptr (.ref .ptr [R,C] pointer))
        (if other then .maskOther (.ref .bool [R,C] "mask") ((Op.castFloat .real .f8e4 Op.negInf).broadcast [R,C])
          else .mask (.ref .bool [R,C] "mask"))))))
  | .f8e5 => .assign .real [R,C] name (.compute (.alg .fp32
      (.castFloat .f8e5 .real (.load .f8e5 (.ptr (.ref .ptr [R,C] pointer))
        (if other then .maskOther (.ref .bool [R,C] "mask") ((Op.castFloat .real .f8e5 Op.negInf).broadcast [R,C])
          else .mask (.ref .bool [R,C] "mask"))))))

def expectedStore (pointer name : RegName) (storage : Storage) (R C : Nat) : ComputeStmt :=
  match storage with
  | .fp32 => .store .real [R,C] (.ptr (.ref .ptr [R,C] pointer))
      (.compute (.alg .fp32 (.ref .real [R,C] name))) (.mask (.ref .bool [R,C] "mask"))
  | .fp16 => .store .fp16 [R,C] (.ptr (.ref .ptr [R,C] pointer))
      (.alg (.castFloat .real .fp16 (.ref .real [R,C] name))) (.mask (.ref .bool [R,C] "mask"))
  | .bf16 => .store .bf16 [R,C] (.ptr (.ref .ptr [R,C] pointer))
      (.alg (.castFloat .real .bf16 (.ref .real [R,C] name))) (.mask (.ref .bool [R,C] "mask"))
  | .fp64 => .store .real [R,C] (.ptr (.ref .ptr [R,C] pointer))
      (.compute (.alg .fp64 (.castFloat .real .real
        (.withPrecision .real .fp32 (.ref .real [R,C] name))))) (.mask (.ref .bool [R,C] "mask"))
  | .f8e4 => .store .f8e4 [R,C] (.ptr (.ref .ptr [R,C] pointer))
      (.alg (.castFloat .real .f8e4 (.ref .real [R,C] name))) (.mask (.ref .bool [R,C] "mask"))
  | .f8e5 => .store .f8e5 [R,C] (.ptr (.ref .ptr [R,C] pointer))
      (.alg (.castFloat .real .f8e5 (.ref .real [R,C] name))) (.mask (.ref .bool [R,C] "mask"))

theorem forward_load (input output : Storage) (v : Bool)
    (OUT X : RegionName) (M N K R C : Nat) :
    (typedForward input output v OUT X M N K R C).surfaceBody[7]? =
      some (expectedLoad "input_ptrs" "inp" input Bool.true R C) := by
  cases input <;> cases output <;> cases v <;> rfl

theorem forward_store (input output : Storage) (v : Bool)
    (OUT X : RegionName) (M N K R C : Nat) :
    (typedForward input output v OUT X M N K R C).surfaceBody[13]? =
      some (expectedStore "output_ptrs" "softmax_output" output R C) := by
  cases input <;> cases output <;> cases v <;> rfl

theorem backward_out_load (input upstream output : Storage) (v : Bool)
    (Y G DX : RegionName) (M N K R C : Nat) :
    (typedBackward input upstream output v Y G DX M N K R C).surfaceBody[7]? =
      some (expectedLoad "out_ptrs" "out" input Bool.false R C) := by
  cases input <;> cases upstream <;> cases output <;> cases v <;> rfl

theorem backward_grad_load (input upstream output : Storage) (v : Bool)
    (Y G DX : RegionName) (M N K R C : Nat) :
    (typedBackward input upstream output v Y G DX M N K R C).surfaceBody[9]? =
      some (expectedLoad "out_grad_ptrs" "out_grad" upstream Bool.false R C) := by
  cases input <;> cases upstream <;> cases output <;> cases v <;> rfl

theorem backward_store (input upstream output : Storage) (v : Bool)
    (Y G DX : RegionName) (M N K R C : Nat) :
    (typedBackward input upstream output v Y G DX M N K R C).surfaceBody[13]? =
      some (expectedStore "in_grad_ptrs" "in_grad" output R C) := by
  cases input <;> cases upstream <;> cases output <;> cases v <;> rfl

def expectedForwardFormula (R C : Nat) : ComputeStmt :=
  .assign .real [R,C] "softmax_output" (.compute (.alg .fp32
    (.log (.div .real (.consSame (.consR .nil))
      (.ref .real [R,C] "numerator") (.ref .real [R,1] "denominator")))))

def expectedBackwardFormula (R C : Nat) : ComputeStmt :=
  .assign .real [R,C] "in_grad" (.compute (.alg .fp32
    (.sub .real (.consSame (.consSame .nil)) (.ref .real [R,C] "out_grad")
      (.mul .real (.consSame (.consR .nil))
        (.exp (.castFloat .real .real (.ref .real [R,C] "out")))
        (.expandDim ⟨1, by simp⟩ (.ref .real [R] "scale"))))))

theorem forward_formula (input output : Storage) (v : Bool)
    (OUT X : RegionName) (M N K R C : Nat) :
    (typedForward input output v OUT X M N K R C).surfaceBody[11]? =
      some (expectedForwardFormula R C) := by
  cases input <;> cases output <;> cases v <;> rfl

theorem backward_formula (input upstream output : Storage) (v : Bool)
    (Y G DX : RegionName) (M N K R C : Nat) :
    (typedBackward input upstream output v Y G DX M N K R C).surfaceBody[11]? =
      some (expectedBackwardFormula R C) := by
  cases input <;> cases upstream <;> cases output <;> cases v <;> rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.LogSoftmax.Precision
