import bench.tritonbench_g.swiglu_triton.Kernels

namespace VeriTile.Bench.TritonBenchG.SwigluTriton.Precision
open VeriTile.Triton
set_option maxHeartbeats 12000000
set_option maxRecDepth 16000

def loadAddress (dtype : TileDType) (pointer : RegName) (B : Nat) : MemAccess dtype [B] :=
  .ptr (.ptrAdd .scalarL (.ref .ptr [] pointer) (.ref .nat [B] "col_offsets"))

/-- Exact raw loads retain storage and masked zero padding. -/
def rawLoad (storage : Storage) (pointer name : RegName) (B : Nat) : ComputeStmt :=
  match storage with
  | .fp32 => .assign .real [B] name
      (.compute (.load .fp32 (loadAddress .real pointer B)
            (.maskOther (.ref .bool [B] "mask")
              ((Op.const 0).broadcast [B]))))
  | .fp16 => .assign .fp16 [B] name
      (.alg
          (.load .fp16 (loadAddress .fp16 pointer B)
            (.maskOther (.ref .bool [B] "mask")
              ((Op.castFloat .real .fp16 (.const 0)).broadcast [B]))))
  | .bf16 => .assign .bf16 [B] name
      (.alg
          (.load .bf16 (loadAddress .bf16 pointer B)
            (.maskOther (.ref .bool [B] "mask")
              ((Op.castFloat .real .bf16 (.const 0)).broadcast [B]))))
  | .fp64 => .assign .real [B] name
      (.compute (.alg .fp64
          (.load .real (loadAddress .real pointer B)
            (.maskOther (.ref .bool [B] "mask")
              ((Op.const 0).broadcast [B])))))
  | .f8e4 => .assign .f8e4 [B] name
      (.alg
          (.load .f8e4 (loadAddress .f8e4 pointer B)
            (.maskOther (.ref .bool [B] "mask")
              ((Op.castFloat .real .f8e4 (.const 0)).broadcast [B]))))
  | .f8e5 => .assign .f8e5 [B] name
      (.alg
          (.load .f8e5 (loadAddress .f8e5 pointer B)
            (.maskOther (.ref .bool [B] "mask")
              ((Op.castFloat .real .f8e5 (.const 0)).broadcast [B]))))

/-- The gate load alone explicitly converts to fp32. A double source is read
under its inner fp64 scope before conversion. -/
def gateLoad (storage : Storage) (B : Nat) : ComputeStmt :=
  match storage with
  | .fp32 => .assign .real [B] "a_row"
      (.compute (.alg .fp32
        (.castFloat .real .real
          (.load .real (loadAddress .real "a_ptr" B)
            (.maskOther (.ref .bool [B] "mask")
              ((Op.const 0).broadcast [B]))))))
  | .fp16 => .assign .real [B] "a_row"
      (.compute (.alg .fp32
        (.castFloat .fp16 .real
          (.load .fp16 (loadAddress .fp16 "a_ptr" B)
            (.maskOther (.ref .bool [B] "mask")
              ((Op.castFloat .real .fp16 (.const 0)).broadcast [B]))))))
  | .bf16 => .assign .real [B] "a_row"
      (.compute (.alg .fp32
        (.castFloat .bf16 .real
          (.load .bf16 (loadAddress .bf16 "a_ptr" B)
            (.maskOther (.ref .bool [B] "mask")
              ((Op.castFloat .real .bf16 (.const 0)).broadcast [B]))))))
  | .fp64 => .assign .real [B] "a_row"
      (.compute (.alg .fp32
        (.castFloat .real .real
          (.withPrecision .real .fp64
          (.load .real (loadAddress .real "a_ptr" B)
            (.maskOther (.ref .bool [B] "mask")
              ((Op.const 0).broadcast [B])))))))
  | .f8e4 => .assign .real [B] "a_row"
      (.compute (.alg .fp32
        (.castFloat .f8e4 .real
          (.load .f8e4 (loadAddress .f8e4 "a_ptr" B)
            (.maskOther (.ref .bool [B] "mask")
              ((Op.castFloat .real .f8e4 (.const 0)).broadcast [B]))))))
  | .f8e5 => .assign .real [B] "a_row"
      (.compute (.alg .fp32
        (.castFloat .f8e5 .real
          (.load .f8e5 (loadAddress .f8e5 "a_ptr" B)
            (.maskOther (.ref .bool [B] "mask")
              ((Op.castFloat .real .f8e5 (.const 0)).broadcast [B]))))))

/-- Output conversions retain the precision of the computed gradient/value. -/
def expectedStore (storage : Storage) (wide : Bool) (pointer name : RegName)
    (B : Nat) : ComputeStmt :=
  match storage, wide with
  | .fp32, Bool.false => .store .real [B] (loadAddress .real pointer B)
      (.compute (.alg .fp32 (.ref .real [B] name))) (.mask (.ref .bool [B] "mask"))
  | .fp32, Bool.true => .store .real [B] (loadAddress .real pointer B)
     
      (.compute (.alg .fp32
        (.castFloat .real .real
          (.withPrecision .real .fp64 (.ref .real [B] name))))) (.mask (.ref .bool [B] "mask"))
  | .fp16, Bool.false => .store .fp16 [B] (loadAddress .fp16 pointer B)
      (.alg (.castFloat .real .fp16 (.ref .real [B] name))) (.mask (.ref .bool [B] "mask"))
  | .fp16, Bool.true => .store .fp16 [B] (loadAddress .fp16 pointer B)
      (.alg
        (.castFloat .real .fp16
          (.withPrecision .real .fp64 (.ref .real [B] name)))) (.mask (.ref .bool [B] "mask"))
  | .bf16, Bool.false => .store .bf16 [B] (loadAddress .bf16 pointer B)
      (.alg (.castFloat .real .bf16 (.ref .real [B] name))) (.mask (.ref .bool [B] "mask"))
  | .bf16, Bool.true => .store .bf16 [B] (loadAddress .bf16 pointer B)
      (.alg
        (.castFloat .real .bf16
          (.withPrecision .real .fp64 (.ref .real [B] name)))) (.mask (.ref .bool [B] "mask"))
  | .fp64, Bool.false => .store .real [B] (loadAddress .real pointer B)
     
      (.compute (.alg .fp64
        (.castFloat .real .real
          (.withPrecision .real .fp32 (.ref .real [B] name))))) (.mask (.ref .bool [B] "mask"))
  | .fp64, Bool.true => .store .real [B] (loadAddress .real pointer B)
      (.compute (.alg .fp64 (.ref .real [B] name))) (.mask (.ref .bool [B] "mask"))
  | .f8e4, Bool.false => .store .f8e4 [B] (loadAddress .f8e4 pointer B)
      (.alg (.castFloat .real .f8e4 (.ref .real [B] name))) (.mask (.ref .bool [B] "mask"))
  | .f8e4, Bool.true => .store .f8e4 [B] (loadAddress .f8e4 pointer B)
      (.alg
        (.castFloat .real .f8e4
          (.withPrecision .real .fp64 (.ref .real [B] name)))) (.mask (.ref .bool [B] "mask"))
  | .f8e5, Bool.false => .store .f8e5 [B] (loadAddress .f8e5 pointer B)
      (.alg (.castFloat .real .f8e5 (.ref .real [B] name))) (.mask (.ref .bool [B] "mask"))
  | .f8e5, Bool.true => .store .f8e5 [B] (loadAddress .f8e5 pointer B)
      (.alg
        (.castFloat .real .f8e5
          (.withPrecision .real .fp64 (.ref .real [B] name)))) (.mask (.ref .bool [B] "mask"))

theorem forward_loads (storage partner : Storage) (v : Bool) (a b c : RegionName)
    (stride N B : Nat) :
    (typedFwd storage partner v a b c stride N B).surfaceBody[6]? = some (gateLoad storage B) ∧
    (typedFwd storage partner v a b c stride N B).surfaceBody[7]? =
      some (rawLoad partner "b_ptr" "b_row" B) := by
  cases storage <;> cases partner <;> cases v <;> exact ⟨rfl,rfl⟩

theorem backward_loads (storage partner gradient : Storage) (v : Bool) (dc a b : RegionName)
    (stride N B : Nat) :
    (typedBwd storage partner gradient v dc a b stride N B).surfaceBody[6]? =
      some (rawLoad gradient "dc_ptr" "dc_row" B) ∧
    (typedBwd storage partner gradient v dc a b stride N B).surfaceBody[7]? =
      some (gateLoad storage B) ∧
    (typedBwd storage partner gradient v dc a b stride N B).surfaceBody[8]? =
      some (rawLoad partner "b_ptr" "b_row" B) := by
  cases storage <;> cases partner <;> cases gradient <;> cases v <;> exact ⟨rfl,rfl,rfl⟩

theorem forward_store (storage partner : Storage) (v : Bool) (a b c : RegionName)
    (stride N B : Nat) :
    (typedFwd storage partner v a b c stride N B).surfaceBody[9]? =
      some (expectedStore storage (partner == .fp64) "c_ptr" "c_row" B) := by
  cases storage <;> cases partner <;> cases v <;> rfl

theorem backward_stores (storage partner gradient : Storage) (v : Bool) (dc a b : RegionName)
    (stride N B : Nat) :
    (typedBwd storage partner gradient v dc a b stride N B).surfaceBody[13]? =
      some (expectedStore storage (partner == .fp64 || gradient == .fp64) "a_ptr" "da_row" B) ∧
    (typedBwd storage partner gradient v dc a b stride N B).surfaceBody[14]? =
      some (expectedStore partner (gradient == .fp64) "b_ptr" "db_row" B) := by
  cases storage <;> cases partner <;> cases gradient <;> cases v <;> exact ⟨rfl,rfl⟩

/-- The complete source statement and IO-role lists survive dtype specialization. -/
theorem source_shapes (storage partner gradient : Storage) (v : Bool)
    (a b c dc : RegionName) (stride N B : Nat) :
    (typedFwd storage partner v a b c stride N B).surfaceBody.length = 10 ∧
    (typedBwd storage partner gradient v dc a b stride N B).surfaceBody.length = 15 ∧
    (typedFwd storage partner v a b c stride N B).inputs = [a,b] ∧
    (typedFwd storage partner v a b c stride N B).outputs = [c] ∧
    (typedBwd storage partner gradient v dc a b stride N B).inputs = [dc,a,b] ∧
    (typedBwd storage partner gradient v dc a b stride N B).outputs = [a,b] := by
  cases storage <;> cases partner <;> cases gradient <;> cases v <;>
    exact ⟨rfl,rfl,rfl,rfl,rfl,rfl⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SwigluTriton.Precision
