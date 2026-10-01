/- Imported numerical results are data. Their use as atomic assumptions is a
scoped model premise, never an axiom about IEEE values. The Python exporter
freezes the selected report before the proof/comparator runs. -/
import VeriTile.Spec

namespace VeriTile.Triton.FP

/-- One accepted row from a named external report. `key` identifies the report
snapshot and row; it is not a fabricated raw-bundle instance/PTX digest. -/
structure ReportedRule where
  ruleID : String
  format : String
  key : String
  artifact : String
  configuration : Lean.Json
  description : String
  shape : List Nat
  block : Nat
  input : String
  compute : String
  accumulator : String
  output : String

/-- Instantiate the selected atomic assumption as explicit syntax fragments.
Experimental shape/launch remain provenance, not restrictions on syntax sizes.
Binding is data construction only; it does not validate arbitrary fragments. -/
def ReportedRule.bind (row : ReportedRule) (lhs rhs : List Statement) :
    Spec.RuleEntry Statement where
  rule := { lhs, rhs, contract := {
    ruleID := row.ruleID
    instanceKey := row.key
    configuration := row.configuration
    warningPolicy := .passOnly
    description := row.description } }
  evidence := { instanceKey := row.key, artifact := some row.artifact,
                bias := .pass, vars := .pass }

/-- Choosing to trust the imported row for these particular fragments is the
only external premise. Identity, artifact presence and the gate policy are
checked by Lean. No unverified numerical equality is introduced. -/
theorem ReportedRule.admit (row : ReportedRule) (lhs rhs : List Statement)
    (h : Spec.EvidenceValidated (row.bind lhs rhs).rule (row.bind lhs rhs).evidence) :
    Spec.AcceptedAtom (row.bind lhs rhs) :=
  ⟨h, rfl, rfl, rfl, rfl⟩

end VeriTile.Triton.FP
