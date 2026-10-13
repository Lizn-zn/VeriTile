import VeriTile.Triton.DSL
import VeriTile.Triton.Float.ExecutionProfile
import VeriTile.Triton.Float.Erasure
import VeriTile.Meta.StatementAudit

/-! Pinned Triton 3.5.1 `standard.py` lines 134–183:
https://github.com/triton-lang/triton/blob/v3.5.1/python/triton/language/standard.py
The indexed max promotes bf16, retains fp16, and defaults to smallest-index ties.
The numeric backend below witnesses execution on exact finite integer payloads;
it does not assert an IEEE implementation or supply a numerical rewrite axiom. -/
namespace TypedArgMaxTests
open VeriTile.Triton FP.Structural
set_option maxHeartbeats 1000000

private def numericArgMax (dtype : FloatDType) {shape : TileShape}
    (ax : Fin shape.length) (v : Values Nat dtype.toTileDType shape) :
    Option (Values Nat .nat (TileShape.eraseAxis shape ax)) :=
  if h : 0 < TileShape.axisDim shape ax then
    some (fun out => ((List.finRange (TileShape.axisDim shape ax)).foldl
      (fun best next => if toFloat dtype (v (TileShape.insertAxisIndex shape ax out best)) <
          toFloat dtype (v (TileShape.insertAxisIndex shape ax out next)) then next else best)
      ⟨0,h⟩).val)
  else none

private def model : Algebra Nat where
  literal := fun _ _ _ => 0
  negInf := 0
  binary := fun _ _ _ a b => a + b
  unary := fun _ _ a => a
  cast := fun _ _ _ a => a + 1000
  fromNat := fun _ n => n
  fromInt := fun _ n => n.toNat
  fp32Bits := fun _ => 0
  fp32Load := id
  reduceMax := fun _ _ _ _ _ => 0
  reduceSum := fun _ _ _ _ _ => 0
  indexedMax := fun _ dtype {shape} ax v => do
    let indices ← numericArgMax dtype ax v
    if h : 0 < TileShape.axisDim shape ax then
      some ((fun out => v (TileShape.insertAxisIndex shape ax out
        ⟨indices out % TileShape.axisDim shape ax, Nat.mod_lt _ h⟩)), indices)
    else none

private def initial : State Nat where
  mem := fun _ _ => .mk .real 0
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

private def sample (dtype : FloatDType) : State Nat :=
  initial.setReg "x" dtype.toTileDType [3]
    (fun i => ofFloat dtype (if i.1.val = 0 then 4 else 9))

private def typed (dtype : FloatDType) : Op .nat [] :=
  .argMaxTyped dtype (shape := [3]) ⟨0, by simp⟩ (.ref dtype.toTileDType [3] "x")

theorem nonempty_half_returns_first_maximum :
    evalOp model none (typed .fp16) (sample .fp16) = some (fun _ => 1) := rfl

theorem nonempty_real_returns_first_maximum :
    evalOp model none (.argMax (shape := [3]) ⟨0, by simp⟩ (.ref .real [3] "x"))
      (sample .real) = some (fun _ => 1) := rfl


/-- Indexed selected values use the paired callback, never ordinary maximum. -/
theorem selected_value_and_index_share_backend :
    evalOp model none (.indexedMaxValue .fp16 (shape := [3]) ⟨0, by simp⟩
      (.ref .fp16 [3] "x")) (sample .fp16) = some (fun _ => ofFloat .fp16 9) := rfl

/-- A backend can distinguish ordinary maximum from indexed selection.
For this witness, zero represents a NaN in the first lane and nine a finite
value in the other lanes: ordinary propagation and indexed right selection
need not agree. This is a dispatch witness, not a general NaN implementation. -/
theorem ordinary_and_indexed_backends_are_distinct :
    let s := initial.setReg "x" .real [3] (fun i => if i.1.val = 0 then 0 else 9)
    evalOp model none (.reduceMax (shape := [3]) ⟨0, by simp⟩ Bool.false
      (.ref .real [3] "x")) s = some (fun _ => 0) ∧
    evalOp model none (.indexedMaxValue .real (shape := [3]) ⟨0, by simp⟩
      (.ref .real [3] "x")) s = some (fun _ => ofFloat .real 9) := ⟨rfl, rfl⟩

theorem absent_backend_fails :
    evalOp { model with indexedMax := fun _ _ {_} _ _ => none }
      none (typed .fp16) (sample .fp16) = none := rfl

theorem empty_does_not_call_backend :
    evalOp { model with indexedMax := fun _ d {_} _ _ => some ((fun _ => ofFloat d 123), fun _ => 77) }
      none (.argMaxTyped .fp16 (shape := [0]) ⟨0, by simp⟩
        (.full [0] (.constFloat .fp16 0))) initial = none := rfl

theorem precision_and_dtype_reach_backend :
    evalOp ({ model with indexedMax := fun p d {_} _ _ =>
      if (p = (some ComputeDType.fp64) ∧ d = FloatDType.fp16) then some ((fun _ => ofFloat d 19), fun _ => 2) else none }.withDefaultPrecision .fp64)
      none (typed .fp16) (sample .fp16) = some (fun _ => 2) := rfl

private def paired32 : Algebra Nat :=
  { model with indexedMax := fun p d {_} _ _ =>
      if (p = (some ComputeDType.fp32)) then some ((fun _ => ofFloat d 42), fun _ => 2) else none }

/-- The fp32 selected value and natural index use the same explicit precision,
even under a conflicting enclosing default. -/
theorem paired_projections_keep_fp32 :
    evalExpr (paired32.withDefaultPrecision .fp64)
      (.compute (.alg .fp32 (.indexedMaxValue .real (shape := [3]) ⟨0, by simp⟩
        (.ref .real [3] "x")))) (sample .real) = some (fun _ => (42 : Nat)) ∧
    evalOp (paired32.withDefaultPrecision .fp64) none
      (.argMaxTyped .real (shape := [3]) ⟨0, by simp⟩ (.ref .real [3] "x") (some .fp32))
      (sample .real) = some (fun _ => 2) := ⟨rfl, rfl⟩

def halfTuple : ComputeKernel := triton {
  x = tl.zeros([3], dtype=tl.float16)
  value, index = tl.max(x, axis=0, return_indices=True)
  direct = tl.argmax(x, axis=0, tie_break_left=True)
}

theorem half_tuple_exact : halfTuple.surfaceBody[1]? = some
    (.ifThen (.constBool Bool.true)
      [.assign .fp16 [] "value" (.alg (.indexedMaxValue .fp16 ⟨0, by simp⟩ (.ref .fp16 [3] "x"))),
       .assign .nat [] "index" (.alg (.argMaxTyped .fp16 ⟨0, by simp⟩ (.ref .fp16 [3] "x")))]) := rfl

theorem half_direct_exact : halfTuple.surfaceBody[2]? = some
    (.assign .nat [] "direct" (.alg (.argMaxTyped .fp16 ⟨0, by simp⟩ (.ref .fp16 [3] "x")))) := rfl

def bfloatTuple : ComputeKernel := triton {
  x = tl.zeros([3], dtype=tl.bfloat16)
  value, index := tl.max(x, axis=0, return_indices=True, return_indices_tie_break_left=True)
}

theorem bfloat_tuple_exact : bfloatTuple.surfaceBody[1]? = some
    (.ifThen (.constBool Bool.true)
      [.assign .real [] "value" (.compute (.alg .fp32 (.indexedMaxValue .real ⟨0, by simp⟩
        (.castFloat .bf16 .real (.ref .bf16 [3] "x"))))),
       .assign .nat [] "index" (.alg (.argMaxTyped .real ⟨0, by simp⟩
        (.castFloat .bf16 .real (.ref .bf16 [3] "x")) (some .fp32)))]) := rfl

def fp32Tuple : ComputeKernel := triton {
  x = tl.zeros([3], dtype=tl.float32)
  value, index = tl.max(x, axis=0, return_indices=True)
}

theorem fp32_tuple_uses_indexed_backend : fp32Tuple.surfaceBody[1]? = some
    (.ifThen (.constBool Bool.true)
      [.assign .real [] "value" (.compute (.alg .fp32 (.indexedMaxValue .real ⟨0, by simp⟩
        (.ref .real [3] "x")))),
       .assign .nat [] "index" (.alg (.argMaxTyped .real ⟨0, by simp⟩
        (.ref .real [3] "x") (some .fp32)))]) := rfl

def e4Tuple : ComputeKernel := triton {
  x = tl.zeros([3], dtype=tl.float8e4nv)
  value, index = tl.max(x, axis=0, return_indices=True)
}

def e5Tuple : ComputeKernel := triton {
  x = tl.zeros([3], dtype=tl.float8e5)
  value, index = tl.max(x, axis=0, return_indices=True)
}

theorem e4_tuple_keeps_channel : e4Tuple.surfaceBody[1]? = some
    (.ifThen (.constBool Bool.true)
      [.assign .f8e4 [] "value" (.alg (.indexedMaxValue .f8e4 ⟨0, by simp⟩ (.ref .f8e4 [3] "x"))),
       .assign .nat [] "index" (.alg (.argMaxTyped .f8e4 ⟨0, by simp⟩ (.ref .f8e4 [3] "x")))]) := rfl

theorem e5_tuple_keeps_channel : e5Tuple.surfaceBody[1]? = some
    (.ifThen (.constBool Bool.true)
      [.assign .f8e5 [] "value" (.alg (.indexedMaxValue .f8e5 ⟨0, by simp⟩ (.ref .f8e5 [3] "x"))),
       .assign .nat [] "index" (.alg (.argMaxTyped .f8e5 ⟨0, by simp⟩ (.ref .f8e5 [3] "x")))]) := rfl

/-- error: tl.max return_indices: only literal True leftmost tie breaking is supported -/
#guard_msgs in
example : ComputeKernel := triton {
  x = tl.zeros([3], dtype=tl.float32)
  v, i = tl.max(x, axis=0, return_indices=True, return_indices_tie_break_left=False)
}

/-- error: tl.argmax: only literal True leftmost tie breaking is supported -/
#guard_msgs in
example : ComputeKernel := triton {
  x = tl.zeros([3], dtype=tl.float16)
  i = tl.argmax(x, axis=0, tie_break_left=False)
}

#guard_msgs (drop info) in
#auditModuleAxioms
end TypedArgMaxTests
