/- Domain requirements are part of an atomic fragment, not just report prose.
An accepted scalar identity cannot be instantiated without retaining its guard.
Experimental shape/distribution still select rules rather than restrict syntax. -/
import VeriTile.Triton.Float.ReportedRules
import VeriTile.Triton.Core.Ast

namespace VeriTile.Triton.FP

inductive GuardKind where
  | finite | nonzero | positive
  deriving DecidableEq, Repr

structure OperandGuard where
  operand : RegName
  kind : GuardKind
  deriving DecidableEq, Repr

/-- A scalar rewrite carries the same operand domain on both sides. -/
structure GuardedFragment where
  guards : List OperandGuard
  code : List ComputeStmt

/-- A report row together with its machine-readable scalar domain. -/
structure ReportedScalarRule where
  report : ReportedRule
  guards : List OperandGuard

def ReportedScalarRule.bind (row : ReportedScalarRule) (lhs rhs : List ComputeStmt) :
    Spec.RuleEntry GuardedFragment :=
  row.report.bind [⟨row.guards, lhs⟩] [⟨row.guards, rhs⟩]

theorem ReportedScalarRule.admit (row : ReportedScalarRule) (lhs rhs : List ComputeStmt)
    (h : Spec.EvidenceValidated (row.bind lhs rhs).rule (row.bind lhs rhs).evidence) :
    Spec.AcceptedAtom (row.bind lhs rhs) :=
  row.report.admit _ _ h

end VeriTile.Triton.FP
