import bench.tritonbench_g.l2_norm_triton1.Kernels
namespace VeriTile.Bench.TritonBenchG.L2NormTriton1
open VeriTile.Triton
def loadAddress (dtype : TileDType) (B _N : Nat) : MemAccess dtype [B] :=
  .ptr (.ptrAdd Broadcast.scalarL
    (.ref .ptr [] "X") (.ref .nat [B] "cols"))

def expectedLoad (input : Storage) (B N : Nat) : ComputeStmt :=
  match input with
  | .fp32 => .assign .real [B] "x" (.compute (.alg .fp32
      (.castFloat .real .real (.load .real (loadAddress .real B N)
        (.maskOther (.lt .nat .scalarR (.ref .nat [B] "cols") (.constNat N))
          ((Op.const 0.0).broadcast [B]))))))
  | .fp64 => .assign .real [B] "x" (.compute (.alg .fp32
      (.castFloat .real .real (.withPrecision .real .fp64
        (.load .real (loadAddress .real B N)
          (.maskOther (.lt .nat .scalarR (.ref .nat [B] "cols") (.constNat N))
            ((Op.const 0.0).broadcast [B])))))))
  | .fp16 => .assign .real [B] "x" (.compute (.alg .fp32
      (.castFloat .fp16 .real (.load .fp16 (loadAddress .fp16 B N)
        (.maskOther (.lt .nat .scalarR (.ref .nat [B] "cols") (.constNat N))
          ((Op.castFloat .real .fp16 (.const 0.0)).broadcast [B]))))))
  | .bf16 => .assign .real [B] "x" (.compute (.alg .fp32
      (.castFloat .bf16 .real (.load .bf16 (loadAddress .bf16 B N)
        (.maskOther (.lt .nat .scalarR (.ref .nat [B] "cols") (.constNat N))
          ((Op.castFloat .real .bf16 (.const 0.0)).broadcast [B]))))))


theorem load_precision (storage : Storage) (v : Bool) (X Y : RegionName)
    (stride N : Nat) (eps : ℝ) (B : Nat) :
    (typedKernel storage v X Y stride N eps B).surfaceBody[4]? =
      some (expectedLoad storage B N) := by
  cases storage <;> cases v <;> rfl

def storeAddress (dtype : TileDType) (B : Nat) : MemAccess dtype [B] :=
  .ptr (.ptrAdd .scalarL (.ref .ptr [] "Y") (.ref .nat [B] "cols"))

def expectedStore (storage : Storage) (B : Nat) : ComputeStmt :=
  match storage with
  | .fp32 => .store .real [B] (storeAddress .real B)
      (.compute (.alg .fp32 (.ref .real [B] "y"))) (.mask (.ref .bool [B] "mask"))
  | .fp64 => .store .real [B] (storeAddress .real B)
      (.compute (.alg .fp64 (.castFloat .real .real
        (.withPrecision .real .fp32 (.ref .real [B] "y")))))
      (.mask (.ref .bool [B] "mask"))
  | .fp16 => .store .fp16 [B] (storeAddress .fp16 B)
      (.alg (.castFloat .real .fp16 (.ref .real [B] "y"))) (.mask (.ref .bool [B] "mask"))
  | .bf16 => .store .bf16 [B] (storeAddress .bf16 B)
      (.alg (.castFloat .real .bf16 (.ref .real [B] "y"))) (.mask (.ref .bool [B] "mask"))

theorem store_precision (storage : Storage) (v : Bool) (X Y : RegionName)
    (stride N : Nat) (eps : ℝ) (B : Nat) :
    (typedKernel storage v X Y stride N eps B).surfaceBody[10]? =
      some (expectedStore storage B) := by
  cases storage <;> cases v <;> rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.L2NormTriton1
