import bench.tritonbench_g.l2_norm_bwd.Kernels
namespace VeriTile.Bench.TritonBenchG.L2NormBwd
open VeriTile.Triton
def loadAddress (pointer : RegName) (dtype : TileDType) (B _N : Nat) : MemAccess dtype [B] :=
  .ptr (.ptrAdd Broadcast.scalarL
    (.ref .ptr [] pointer) (.ref .nat [B] "cols"))

def expectedLoad (pointer name : RegName) (input : Storage) (B N : Nat) : ComputeStmt :=
  match input with
  | .fp32 => .assign .real [B] name (.compute (.alg .fp32
      (.castFloat .real .real (.load .real (loadAddress pointer .real B N)
        (.maskOther (.lt .nat .scalarR (.ref .nat [B] "cols") (.constNat N))
          ((Op.const 0.0).broadcast [B]))))))
  | .fp64 => .assign .real [B] name (.compute (.alg .fp32
      (.castFloat .real .real (.withPrecision .real .fp64
        (.load .real (loadAddress pointer .real B N)
          (.maskOther (.lt .nat .scalarR (.ref .nat [B] "cols") (.constNat N))
            ((Op.const 0.0).broadcast [B])))))))
  | .fp16 => .assign .real [B] name (.compute (.alg .fp32
      (.castFloat .fp16 .real (.load .fp16 (loadAddress pointer .fp16 B N)
        (.maskOther (.lt .nat .scalarR (.ref .nat [B] "cols") (.constNat N))
          ((Op.castFloat .real .fp16 (.const 0.0)).broadcast [B]))))))
  | .bf16 => .assign .real [B] name (.compute (.alg .fp32
      (.castFloat .bf16 .real (.load .bf16 (loadAddress pointer .bf16 B N)
        (.maskOther (.lt .nat .scalarR (.ref .nat [B] "cols") (.constNat N))
          ((Op.castFloat .real .bf16 (.const 0.0)).broadcast [B]))))))


def storeAddress (pointer : RegName) (dtype : TileDType) (B : Nat) : MemAccess dtype [B] :=
  .ptr (.ptrAdd .scalarL (.ref .ptr [] pointer) (.ref .nat [B] "cols"))

def expectedStore (pointer name : RegName) (storage : Storage) (B : Nat) : ComputeStmt :=
  match storage with
  | .fp32 => .store .real [B] (storeAddress pointer .real B)
      (.compute (.alg .fp32 (.ref .real [B] name))) (.mask (.ref .bool [B] "mask"))
  | .fp64 => .store .real [B] (storeAddress pointer .real B)
      (.compute (.alg .fp64 (.castFloat .real .real
        (.withPrecision .real .fp32 (.ref .real [B] name)))))
      (.mask (.ref .bool [B] "mask"))
  | .fp16 => .store .fp16 [B] (storeAddress pointer .fp16 B)
      (.alg (.castFloat .real .fp16 (.ref .real [B] name))) (.mask (.ref .bool [B] "mask"))
  | .bf16 => .store .bf16 [B] (storeAddress pointer .bf16 B)
      (.alg (.castFloat .real .bf16 (.ref .real [B] name))) (.mask (.ref .bool [B] "mask"))

theorem bwd_x_load_precision (storage gradient : Storage) (v : Bool) (X DY DX : RegionName)
    (stride N : Nat) (eps : ℝ) (B : Nat) :
    (typedBwd storage gradient v X DY DX stride N eps B).surfaceBody[5]? =
      some (expectedLoad "X" "x" storage B N) := by
  cases storage <;> cases gradient <;> cases v <;> rfl

theorem bwd_dy_load_precision (storage gradient : Storage) (v : Bool) (X DY DX : RegionName)
    (stride N : Nat) (eps : ℝ) (B : Nat) :
    (typedBwd storage gradient v X DY DX stride N eps B).surfaceBody[10]? =
      some (expectedLoad "DY" "dy" gradient B N) := by
  cases storage <;> cases gradient <;> cases v <;> rfl

theorem bwd_store_precision (storage gradient : Storage) (v : Bool) (X DY DX : RegionName)
    (stride N : Nat) (eps : ℝ) (B : Nat) :
    (typedBwd storage gradient v X DY DX stride N eps B).surfaceBody[13]? =
      some (expectedStore "DX" "dx" storage B) := by
  cases storage <;> cases gradient <;> cases v <;> rfl

def expectedRstd (eps : ℝ) : ComputeStmt :=
  .assign .real [] "rstd" (.compute (.alg .fp32 (.div .real .nil (.const 1)
    (.sqrt (.add .real .nil (.ref .real [] "var")
      (.castFloat .real .real (.const eps)))))))

theorem bwd_epsilon_precision (storage gradient : Storage) (v : Bool) (X DY DX : RegionName)
    (stride N : Nat) (eps : ℝ) (B : Nat) :
    (typedBwd storage gradient v X DY DX stride N eps B).surfaceBody[8]? =
      some (expectedRstd eps) := by
  cases storage <;> cases gradient <;> cases v <;> rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.L2NormBwd
