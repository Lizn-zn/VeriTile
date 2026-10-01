/- Boundary regressions for structural FP steps. These are deliberately
non-numerical countermodels: no algebraic law may enter through the evaluator. -/
import VeriTile.Triton.Float.StructuralIO
import VeriTile.Meta.StatementAudit

namespace FPStructuralTests
open VeriTile Triton
open FP.Structural

private def M : Algebra Nat where
  literal := fun _ _ _ => 0
  negInf := 0
  binary := fun p _ op x y => match op, p with
    | .add, none => x - y
    | .add, some _ => x + y
    | _, _ => x * y
  unary := fun _ _ x => x + 1
  cast := fun _ _ _ x => x + 1
  fromNat := fun _ n => n
  fromInt := fun _ n => n.toNat
  fp32Bits := fun x => x.bits.toNat
  fp32Load := fun x => x + 1

private def initial : State Nat where
  mem := fun _ _ => .mk .real 0
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

private def abc : State Nat :=
  ((initial.setReg "a" .real [] (fun _ => 10)).setReg "b" .real [] (fun _ => 3)).setReg
    "c" .real [] (fun _ => 1)

private def a : Op .real [] := .ref .real [] "a"
private def b : Op .real [] := .ref .real [] "b"
private def c : Op .real [] := .ref .real [] "c"
private def plus (x y : Op .real []) := Op.add .real .nil x y

-- An arbitrary interpretation need not commute or associate.
example : evalOp M none (plus a b) abc ≠ evalOp M none (plus b a) abc := by
  intro h
  have hscalar := congrArg (Option.map (fun v => v PUnit.unit)) h
  simp [plus, a, b, abc, initial, evalOp_unfold, numeric, FP.Structural.bop,
    M, State.setReg] at hscalar
example : evalOp M none (plus (plus a b) c) abc ≠ evalOp M none (plus a (plus b c)) abc := by
  intro h
  have hscalar := congrArg (Option.map (fun v => v PUnit.unit)) h
  simp [plus, a, b, c, abc, initial, evalOp_unfold, numeric, FP.Structural.bop,
    M, State.setReg] at hscalar

-- Compute precision and repeated casts remain observable primitive calls.
example : evalExpr M (.alg (plus a b)) abc ≠
    evalExpr M (.compute (.alg .fp32 (plus a b))) abc := by
  intro h
  have hscalar := congrArg (Option.map (fun v => v PUnit.unit)) h
  simp [evalExpr, evalComputeOp, ComputeDType.eraseDType, plus, a, b, abc, initial,
    evalOp_unfold, numeric, FP.Structural.bop, M, State.setReg] at hscalar
example : evalOp M none (.castFloat .real .bf16 (.const 0)) initial ≠
    evalOp M none (.castFloat .bf16 .bf16 (.castFloat .real .bf16 (.const 0))) initial := by
  intro h
  have hscalar := congrArg (Option.map (fun v => v PUnit.unit)) h
  simp [evalOp_unfold, ofFloat, toFloat, M] at hscalar

-- A register assignment shadows old bindings of the same name at other types.
example : ((initial.setReg "v" .real [] (fun _ => 1)).setReg
    "v" .nat [2] (fun _ => 4)).regs .real [] "v" = none := by
  simp [State.setReg]

-- Memory forwarding preserves the dtype; reading at another dtype is not a cast.
example : ((initial.write "tmp" 3 (.mk .bf16 7)).mem "tmp" 3).read .bf16 = 7 := by simp
example : ((initial.write "tmp" 3 (.mk .bf16 7)).mem "tmp" 3).read .real = 0 := by
  simp [Cell.read, defaultValue]

private def unsupportedIO : KernelIO₃ where
  kernel := .mk [] [] [.effectMarker "tl.debug_barrier"]
  projection := by rfl
  in1 := "x"
  in2 := "g"
  in3 := "r"
  out := "out"
  B1 := 0
  B2 := 0
  B3 := 0
  Bout := 0
  read1 := fun _ => 0
  read2 := fun _ => 0
  read3 := fun _ => 0
  write := fun _ => 0

-- Two unsupported executions cannot supply a structural certificate.
example : ¬ IO₃Equiv unsupportedIO unsupportedIO := by
  intro h
  obtain ⟨t, _, he, _⟩ := h.2.2 Nat M initial
  simp [unsupportedIO, FP.Structural.exec, run, step] at he

-- A same-body numerical derivation cannot modify private scratch metadata.
example : ¬ Spec.ProgramSyntax.sameContext unsupportedIO
    { unsupportedIO with scratch := [{ buf := "tmp", win := fun _ => 0, len := 1 }] } := by
  change ¬ ([] : List ScratchSpec) = [_]
  simp

-- Declaring scratch cannot hide public output, or a different offset in the
-- same scratch region. Framing is over individual cells, not whole buffers.
example : ¬ PrivateScratch
    { unsupportedIO with scratch := [{ buf := "out", win := fun _ => 0, len := 1 }] } := by
  simp [PrivateScratch, unsupportedIO]

example : ¬ Frame
    { unsupportedIO with scratch := [{ buf := "tmp", win := fun _ => 0, len := 1 }] }
    initial (initial.write "tmp" 1 (.mk .real 9)) := by
  intro h
  have hcell := h "tmp" 1 (Or.inl (by decide)) (by
    intro p hp _ i
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hp
    subst p
    simp)
  simp [initial] at hcell

end FPStructuralTests
