import bench.tritonbench_g.swiglu_fwd.Kernels
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.SwigluFwd.Precision
open VeriTile.Triton
set_option maxHeartbeats 8000000
set_option maxRecDepth 16000

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
  | .f8e4 => .assign .real [B] name (.compute (.alg .fp32
      (.castFloat .f8e4 .real (.load .f8e4 (loadAddress pointer .f8e4 B N)
        (.maskOther (.lt .nat .scalarR (.ref .nat [B] "cols") (.constNat N))
          ((Op.castFloat .real .f8e4 (.const 0.0)).broadcast [B]))))))
  | .f8e5 => .assign .real [B] name (.compute (.alg .fp32
      (.castFloat .f8e5 .real (.load .f8e5 (loadAddress pointer .f8e5 B N)
        (.maskOther (.lt .nat .scalarR (.ref .nat [B] "cols") (.constNat N))
          ((Op.castFloat .real .f8e5 (.const 0.0)).broadcast [B]))))))


def storeAddress (pointer : RegName) (dtype : TileDType) (B : Nat) : MemAccess dtype [B] :=
  .ptr (.ptrAdd .scalarL (.ref .ptr [] pointer) (.ref .nat [B] "cols"))

def expectedStore (pointer name : RegName) (storage : Storage) (B N : Nat) : ComputeStmt :=
  match storage with
  | .fp32 => .store .real [B] (storeAddress pointer .real B)
      (.compute (.alg .fp32 (.ref .real [B] name)))
      (.mask (.lt .nat .scalarR (.ref .nat [B] "cols") (.constNat N)))
  | .fp64 => .store .real [B] (storeAddress pointer .real B)
      (.compute (.alg .fp64 (.castFloat .real .real
        (.withPrecision .real .fp32 (.ref .real [B] name)))))
      (.mask (.lt .nat .scalarR (.ref .nat [B] "cols") (.constNat N)))
  | .fp16 => .store .fp16 [B] (storeAddress pointer .fp16 B)
      (.alg (.castFloat .real .fp16 (.ref .real [B] name)))
      (.mask (.lt .nat .scalarR (.ref .nat [B] "cols") (.constNat N)))
  | .bf16 => .store .bf16 [B] (storeAddress pointer .bf16 B)
      (.alg (.castFloat .real .bf16 (.ref .real [B] name)))
      (.mask (.lt .nat .scalarR (.ref .nat [B] "cols") (.constNat N)))

  | .f8e4 => .store .f8e4 [B] (storeAddress pointer .f8e4 B)
      (.alg (.castFloat .real .f8e4 (.ref .real [B] name)))
      (.mask (.lt .nat .scalarR (.ref .nat [B] "cols") (.constNat N)))
  | .f8e5 => .store .f8e5 [B] (storeAddress pointer .f8e5 B)
      (.alg (.castFloat .real .f8e5 (.ref .real [B] name)))
      (.mask (.lt .nat .scalarR (.ref .nat [B] "cols") (.constNat N)))

theorem x_load (input partner output : Storage) (v : Bool) (X Y OUT : RegionName)
    (sx sy so N B : Nat) :
    (typedKernel input partner output v X Y OUT sx sy so N B).surfaceBody[6]? =
      some (expectedLoad "X" "x" input B N) := by
  cases input <;> cases partner <;> cases output <;> cases v <;> rfl

theorem y_load (input partner output : Storage) (v : Bool) (X Y OUT : RegionName)
    (sx sy so N B : Nat) :
    (typedKernel input partner output v X Y OUT sx sy so N B).surfaceBody[7]? =
      some (expectedLoad "Y" "y" partner B N) := by
  cases input <;> cases partner <;> cases output <;> cases v <;> rfl

theorem output_store (input partner output : Storage) (v : Bool) (X Y OUT : RegionName)
    (sx sy so N B : Nat) :
    (typedKernel input partner output v X Y OUT sx sy so N B).surfaceBody[9]? =
      some (expectedStore "OUT" "out" output B N) := by
  cases input <;> cases partner <;> cases output <;> cases v <;> rfl

def expectedFormula (B : Nat) : ComputeStmt :=
  .assign .real [B] "out" (.compute (.alg .fp32
    (.mul .real (.consSame .nil)
      (.mul .real (.consSame .nil) (.ref .real [B] "x") (.sigmoid (.ref .real [B] "x")))
      (.ref .real [B] "y"))))

theorem formula_precision (input partner output : Storage) (v : Bool) (X Y OUT : RegionName)
    (sx sy so N B : Nat) :
    (typedKernel input partner output v X Y OUT sx sy so N B).surfaceBody[8]? =
      some (expectedFormula B) := by
  cases input <;> cases partner <;> cases output <;> cases v <;> rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SwigluFwd.Precision
