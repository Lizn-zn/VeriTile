import bench.tritonbench_g.geglu_tanh_triton.Kernels

namespace VeriTile.Bench.TritonBenchG.GegluTanhTriton.Precision
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
          (.load .real (loadAddress .real "a" B)
            (.maskOther (.ref .bool [B] "mask")
              ((Op.const 0).broadcast [B]))))))
  | .fp16 => .assign .real [B] "a_row"
      (.compute (.alg .fp32
        (.castFloat .fp16 .real
          (.load .fp16 (loadAddress .fp16 "a" B)
            (.maskOther (.ref .bool [B] "mask")
              ((Op.castFloat .real .fp16 (.const 0)).broadcast [B]))))))
  | .bf16 => .assign .real [B] "a_row"
      (.compute (.alg .fp32
        (.castFloat .bf16 .real
          (.load .bf16 (loadAddress .bf16 "a" B)
            (.maskOther (.ref .bool [B] "mask")
              ((Op.castFloat .real .bf16 (.const 0)).broadcast [B]))))))
  | .fp64 => .assign .real [B] "a_row"
      (.compute (.alg .fp32
        (.castFloat .real .real
          (.withPrecision .real .fp64
          (.load .real (loadAddress .real "a" B)
            (.maskOther (.ref .bool [B] "mask")
              ((Op.const 0).broadcast [B])))))))
  | .f8e4 => .assign .real [B] "a_row"
      (.compute (.alg .fp32
        (.castFloat .f8e4 .real
          (.load .f8e4 (loadAddress .f8e4 "a" B)
            (.maskOther (.ref .bool [B] "mask")
              ((Op.castFloat .real .f8e4 (.const 0)).broadcast [B]))))))
  | .f8e5 => .assign .real [B] "a_row"
      (.compute (.alg .fp32
        (.castFloat .f8e5 .real
          (.load .f8e5 (loadAddress .f8e5 "a" B)
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
      some (rawLoad partner "b" "b_row" B) := by
  cases storage <;> cases partner <;> cases v <;> exact ⟨rfl,rfl⟩

theorem backward_loads (storage partner gradient : Storage) (v : Bool) (dc a b : RegionName)
    (stride N B : Nat) :
    (typedBwd storage partner gradient v dc a b stride N B).surfaceBody[6]? =
      some (rawLoad gradient "dc" "dc_row" B) ∧
    (typedBwd storage partner gradient v dc a b stride N B).surfaceBody[7]? =
      some (gateLoad storage B) ∧
    (typedBwd storage partner gradient v dc a b stride N B).surfaceBody[8]? =
      some (rawLoad partner "b" "b_row" B) := by
  cases storage <;> cases partner <;> cases gradient <;> cases v <;> exact ⟨rfl,rfl,rfl⟩

theorem forward_store (storage partner : Storage) (v : Bool) (a b c : RegionName)
    (stride N B : Nat) :
    (typedFwd storage partner v a b c stride N B).surfaceBody[14]? =
      some (expectedStore storage (partner == .fp64) "c" "c_row" B) := by
  cases storage <;> cases partner <;> cases v <;> rfl

theorem backward_stores (storage partner gradient : Storage) (v : Bool) (dc a b : RegionName)
    (stride N B : Nat) :
    (typedBwd storage partner gradient v dc a b stride N B).surfaceBody[19]? =
      some (expectedStore storage (partner == .fp64 || gradient == .fp64) "a" "da_row" B) ∧
    (typedBwd storage partner gradient v dc a b stride N B).surfaceBody[20]? =
      some (expectedStore partner (gradient == .fp64) "b" "db_row" B) := by
  cases storage <;> cases partner <;> cases gradient <;> cases v <;> exact ⟨rfl,rfl⟩

/-- Literal-only Python multiplication folds before tensor conversion. The
coefficient is one constant, with all tensor operations in their source order. -/
def expectedTerm2 (B : Nat) : ComputeStmt :=
  .assign .real [B] "term2" (.compute (.alg .fp32
    (.mul .real Broadcast.nil.consSame
      (.mul .real Broadcast.nil.consSame
        (.mul .real .scalarL (.const 0.5) (.ref .real [B] "a_row"))
        (.sub .real .scalarL (.const 1) (.ref .real [B] "tanh_sq")))
      (.mul .real .scalarL (.ref .real [] "sqrt_2_over_pi")
        (.add .real .scalarL (.const 1)
          (.mul .real Broadcast.nil.consSame
            (.mul .real .scalarL (.const 0.134145) (.ref .real [B] "a_row"))
            (.ref .real [B] "a_row")))))))

theorem backward_term2 (storage partner gradient : Storage) (v : Bool) (dc a b : RegionName)
    (stride N B : Nat) :
    (typedBwd storage partner gradient v dc a b stride N B).surfaceBody[17]? =
      some (expectedTerm2 B) := by
  cases storage <;> cases partner <;> cases gradient <;> cases v <;> rfl

/-- The half/bfloat product is evaluated in half before widening to fp32. -/
def expectedHalfDa (reverse : Bool) (B : Nat) : ComputeStmt :=
  .assign .real [B] "da_row" (.compute (.alg .fp32
    (.mul .real Broadcast.nil.consSame
      (.castFloat .fp16 .real
        (.mul .fp16 Broadcast.nil.consSame
          (if reverse then .castFloat .bf16 .fp16 (.ref .bf16 [B] "dc_row")
           else .ref .fp16 [B] "dc_row")
          (if reverse then .ref .fp16 [B] "b_row"
           else .castFloat .bf16 .fp16 (.ref .bf16 [B] "b_row"))))
      (.add .real Broadcast.nil.consSame (.ref .real [B] "term1") (.ref .real [B] "term2")))))

theorem backward_half_bfloat_da (storage : Storage) (v : Bool) (dc a b : RegionName)
    (stride N B : Nat) :
    (typedBwd storage .bf16 .fp16 v dc a b stride N B).surfaceBody[18]? =
      some (expectedHalfDa Bool.false B) ∧
    (typedBwd storage .fp16 .bf16 v dc a b stride N B).surfaceBody[18]? =
      some (expectedHalfDa Bool.true B) := by
  cases storage <;> cases v <;> exact ⟨rfl,rfl⟩

/-- The mixed double product widens the fp32 operand while the sum of gate
terms stays under an inner fp32 scope. -/
def expectedDoubleDa (reverse : Bool) (B : Nat) : ComputeStmt :=
  .assign .real [B] "da_row" (.compute (.alg .fp64
    (.mul .real Broadcast.nil.consSame
      (.mul .real Broadcast.nil.consSame
        (if reverse then .ref .real [B] "dc_row"
         else .withPrecision .real .fp32 (.ref .real [B] "dc_row"))
        (if reverse then .withPrecision .real .fp32 (.ref .real [B] "b_row")
         else .ref .real [B] "b_row"))
      (.withPrecision .real .fp32
        (.add .real Broadcast.nil.consSame (.ref .real [B] "term1") (.ref .real [B] "term2"))))))

theorem backward_double_single_da (storage : Storage) (v : Bool) (dc a b : RegionName)
    (stride N B : Nat) :
    (typedBwd storage .fp64 .fp32 v dc a b stride N B).surfaceBody[18]? =
      some (expectedDoubleDa Bool.false B) ∧
    (typedBwd storage .fp32 .fp64 v dc a b stride N B).surfaceBody[18]? =
      some (expectedDoubleDa Bool.true B) := by
  cases storage <;> cases v <;> exact ⟨rfl,rfl⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.GegluTanhTriton.Precision
