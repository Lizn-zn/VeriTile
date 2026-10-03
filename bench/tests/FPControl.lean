/- Control-flow boundary checks use an arbitrary, non-associative numerical
interpretation, so loop execution cannot borrow Real algebraic identities. -/
import VeriTile.Triton.Float.Control
import VeriTile.Meta.StatementAudit

namespace FPControlTests
open VeriTile Triton FP.Structural

private def M : Algebra Nat where
  literal := fun _ _ _ => 0
  negInf := 0
  binary := fun p _ _ x y => if p = some .fp32 then 10 * x + y else 100 + x + y
  unary := fun _ _ x => x + 7
  cast := fun _ _ _ x => x + 1
  fromNat := fun _ n => n + 20
  fromInt := fun _ n => n.toNat
  fp32Bits := fun x => x.bits.toNat
  fp32Load := id
  reduceMax := fun _ _ _ _ _ => 0
  reduceSum := fun _ _ _ _ _ => 0

private def initial : State Nat where
  mem := fun _ _ => .mk .real 7
  regs := fun _ _ _ => none
  pids := fun _ => 2
  numPids := fun _ => 5
  undef := fun d _ _ => defaultValue d

private def seed := initial.setReg "acc" .real [] (fun _ => 1)

private def accumulate : ComputeStmt := .assign .real [] "acc" (.compute (.alg .fp32
  (.add .real .nil (.ref .real [] "acc") (.natToReal (.ref .nat [] "i")))))

private def overwriteIndex : ComputeStmt := .assign .nat [] "i" (.alg (.constNat 999))

private def readAcc (s : Option (State Nat)) : Option Nat :=
  s.bind fun s => (s.regs .real [] "acc").map (fun v => v PUnit.unit)

-- Every iteration resets the loop index even when the body overwrites it.
example : readAcc (step M (.forLoop "i" 3 [accumulate, overwriteIndex]) seed) = some 3232 := by
  simp [readAcc, step, loop, run, accumulate, overwriteIndex, evalExpr, evalComputeOp, ComputeDType.eraseDType,
    evalOp_unfold, numeric, FP.Structural.bop, M, seed, State.setReg]

-- Non-unit ranges execute 1, 4, 7; they do not execute the stop index.
example : readAcc (step M (.forRange "i" 1 8 3 [accumulate]) seed) = some 3367 := by
  simp [readAcc, step, range, run, accumulate, evalExpr, evalComputeOp, ComputeDType.eraseDType,
    evalOp_unfold, numeric, FP.Structural.bop, M, seed, State.setReg]

-- Dynamic bounds are captured on entry, before a body can overwrite them.
private def changeStop : ComputeStmt := .assign .nat [] "stop" (.alg (.constNat 0))
example : readAcc (step M
    (.forRangeDyn "i" (.constNat 1) (.ref .nat [] "stop") (.constNat 3)
      [accumulate, changeStop]) (seed.setReg "stop" .nat [] (fun _ => 8))) = some 3367 := by
  have frozen : step M
      (.forRangeDyn "i" (.constNat 1) (.ref .nat [] "stop") (.constNat 3)
        [accumulate, changeStop]) (seed.setReg "stop" .nat [] (fun _ => 8)) =
      range M "i" 1 8 3 [accumulate, changeStop]
        (seed.setReg "stop" .nat [] (fun _ => 8)) := by
    rw [step]
    simp [evalOp_unfold]
  rw [frozen]
  simp [readAcc, step, range, run, accumulate, changeStop, evalExpr, evalComputeOp, ComputeDType.eraseDType,
    evalOp_unfold, numeric, FP.Structural.bop, M, seed, State.setReg]

-- Empty loops and zero-stride ranges skip unsupported bodies.
example : step M (.forLoop "i" 0 [.effectMarker "unsupported"]) seed = some seed := by
  simp [step, loop]
example : step M (.forRange "i" 3 3 1 [.effectMarker "unsupported"]) seed = some seed := by
  simp [step, range]
example : step M (.forRange "i" 0 3 0 [.effectMarker "unsupported"]) seed = some seed := by
  simp [step, range]

-- Reached failures propagate; neither two failures nor a missing register
-- become a successful execution certificate.
example : step M (.forLoop "i" 1 [.effectMarker "unsupported"]) seed = none := by
  simp [step, loop, run]
example : step M (.forLoop "i" 1 [accumulate]) initial = none := by
  simp [step, loop, run, accumulate, evalExpr, evalComputeOp, ComputeDType.eraseDType, evalOp_unfold,
    initial, State.setReg]
example : step M (.forRangeDyn "i" (.constNat 0) (.ref .nat [] "missing")
    (.constNat 1) []) initial = none := by
  simp [step, evalOp_unfold, initial]

-- Only the selected branch executes; nesting does not erase casts or tags.
private def castAcc : ComputeStmt := .assign .bf16 [] "cast"
  (.alg (.castFloat .real .bf16 (.ref .real [] "acc")))
example : (step M (.forLoop "outer" 2
    [.ifThenElse (.constBool true) [.forLoop "i" 1 [accumulate]]
      [.effectMarker "unsupported"], castAcc]) seed).bind
    (fun s => (s.regs .bf16 [] "cast").map (fun v => v PUnit.unit)) = some 321 := by
  simp [step, loop, run, accumulate, castAcc, evalExpr, evalComputeOp, ComputeDType.eraseDType, evalOp_unfold,
    numeric, FP.Structural.bop, M, seed, State.setReg, ofFloat, toFloat]

example : step M (.ifThen (.constBool false) [.effectMarker "unsupported"]) seed = some seed := by
  simp [step, evalOp_unfold]

#axiomsClean loop_invariant
#axiomsClean forLoop_invariant
#axiomsClean range_invariant

end FPControlTests
