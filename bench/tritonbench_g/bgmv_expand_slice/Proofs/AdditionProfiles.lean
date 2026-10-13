import VeriTile.Triton.Float.GuardedIO
import VeriTile.Triton.Float.ReportedAdmission
import VeriTile.Meta.StatementAudit

/-! Typed admission plumbing for the existing ADD-COMMUTE experiment profiles.
The probe widens its input to fp32 and implements low precision by explicit
round trips. These fragments retain those casts and observe the typed output.
They do not identify that experiment with native bf16 arithmetic in BGMV, and
are not a replacement for the pending general BGMV equivalence theorem. -/
namespace VeriTile.Bench.TritonBenchG.BgmvExpandSlice.AdditionProfiles
open VeriTile Triton FP FP.Structural FP.Guarded

inductive Profile where
  | bf16 | bf16_fp32 | fp32
  deriving DecidableEq, Repr

def Profile.input : Profile → FloatDType
  | .bf16 | .bf16_fp32 => .bf16
  | .fp32 => .fp32

def Profile.output := Profile.input

def Profile.row : Profile → ReportedRule
  | .bf16 => ReportedAdmission.bf16_add_commute
  | .bf16_fp32 => ReportedAdmission.bf16_fp32_add_commute
  | .fp32 => ReportedAdmission.fp32_add_commute

/-- Unsupported native formats have no entry; no report row is manufactured. -/
def nativeReport? : FloatDType → Option ReportedRule
  | .fp32 => some ReportedAdmission.fp32_add_commute
  | _ => none

theorem fp16_unsupported : nativeReport? .fp16 = none := rfl
theorem fp64_unsupported : nativeReport? .real = none := rfl
/-- The bf16 report tests rounded widened arithmetic, not an opaque bf16 primitive. -/
theorem native_bf16_needs_correspondence : nativeReport? .bf16 = none := rfl

/-- Unlike the older GuardedFragment, both register channels are explicit. -/
structure Fragment where
  input : FloatDType
  output : FloatDType
  guards : List OperandGuard
  code : List ComputeStmt

def widened (p : Profile) (name : RegName) : Op .real [] :=
  .castFloat p.input .real (.ref p.input.toTileDType [] name)

def sum (p : Profile) (swapped : Bool) : Op .real [] :=
  if swapped then .add .real .nil (widened p "b") (widened p "a")
  else .add .real .nil (widened p "a") (widened p "b")

/-- Exact `rnd` branch in experiments/floating_point/kernels.py. -/
def rounded (p : Profile) (e : Op .real []) : Op .real [] :=
  match p with
  | .bf16 => .castFloat .bf16 .real (.castFloat .real .bf16 e)
  | .bf16_fp32 | .fp32 => .castFloat .real .real e

/-- Store conversion follows the probe's fp32 addition and explicit `rnd`. -/
def fragment (p : Profile) (swapped : Bool) : Fragment :=
  { input := p.input, output := p.output
    guards := [⟨"a", .finite⟩, ⟨"b", .finite⟩]
    code := [.assign .real [] "rounded" (.compute (.alg .fp32 (rounded p (sum p swapped)))),
      .assign p.output.toTileDType [] "out"
        (.alg (.castFloat .real p.output (.ref .real [] "rounded")))] }

def DomainHolds {α : Type} (D : Domain α) (f : Fragment) (s : State α) : Prop :=
  ∀ g ∈ f.guards, ∃ v, s.regs f.input.toTileDType [] g.operand = some v ∧
    D g.kind (toFloat f.input (v PUnit.unit))

/-- A typed scalar observation, including execution failure. -/
noncomputable def observe {α : Type} [Inhabited α] (M : Algebra α)
    (f : Fragment) (s : State α) : Option (Option α) :=
  (run M f.code s).map fun t ↦
    (t.regs f.output.toTileDType [] "out").map fun v ↦ toFloat f.output (v PUnit.unit)

def entry (p : Profile) : Spec.RuleEntry Fragment :=
  p.row.bind [fragment p false] [fragment p true]

/-- Validation is explicitly for the typed, cast-preserving fragments above. -/
structure Rules where
  validated : ∀ p, Spec.EvidenceValidated (entry p).rule (entry p).evidence

def Rules.assumptions (_ : Rules) : Spec.Assumptions Fragment :=
  [entry .bf16, entry .bf16_fp32, entry .fp32]

theorem admitted (R : Rules) (p : Profile) :
    Spec.Derivation R.assumptions [fragment p false] [fragment p true] := by
  apply Spec.Derivation.atom (entry p)
  · cases p <;> simp [Rules.assumptions]
  · exact p.row.admit _ _ (R.validated p)

/-- A model premise is scoped to the typed admitted theory, not asserted as IEEE truth. -/
def Models {α : Type} [Inhabited α] (R : Rules) (M : Algebra α) (D : Domain α) : Prop :=
  ∀ lhs rhs, Spec.Derivation R.assumptions [lhs] [rhs] → ∀ s,
    DomainHolds D lhs s → DomainHolds D rhs s → observe M lhs s = observe M rhs s

theorem commute_observation {α : Type} [Inhabited α] (R : Rules)
    (M : Algebra α) (D : Domain α) (hM : Models R M D) (p : Profile) (s : State α)
    (hd : DomainHolds D (fragment p false) s) :
    observe M (fragment p false) s = observe M (fragment p true) s :=
  hM _ _ (admitted R p) s hd hd

/-- Source metadata is checked, including the probe's fp32 accumulator. -/
theorem report_precision (p : Profile) :
    p.row.ruleID = "ADD-COMMUTE" ∧ p.row.accumulator = "fp32" ∧
    p.row.input = (match p with | .fp32 => "fp32" | _ => "bf16") ∧
    p.row.compute = (match p with | .bf16 => "bf16" | _ => "fp32") ∧
    p.row.output = (match p with | .fp32 => "fp32" | _ => "bf16") := by
  cases p <;> exact ⟨rfl, rfl, rfl, rfl, rfl⟩

#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.BgmvExpandSlice.AdditionProfiles
