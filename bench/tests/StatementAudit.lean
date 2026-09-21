import VeriTile.Meta.StatementAudit

-- Namespace spelling must not exempt project definitions from either gate.
def auditKernel : Nat := 1
namespace Nat
def auditWrapper : Nat := auditKernel
end Nat
namespace Real
def auditWrapper : Nat := Nat.auditWrapper
end Real
def instAuditWrapper : Nat := Real.auditWrapper
def auditCircularSpec : Nat := instAuditWrapper

/-- error: auditCircularSpec: SELF-REFERENTIAL — definition transitively uses kernel(s) [auditKernel] -/
#guard_msgs in
#specNonCircular auditCircularSpec avoiding [auditKernel]

/-- error: auditKernel: SELF-REFERENTIAL — definition transitively uses kernel(s) [auditKernel] -/
#guard_msgs in
#specNonCircular auditKernel avoiding [auditKernel]

theorem auditHiddenStatement : Nat.auditWrapper = Nat.auditWrapper := rfl
/-- error: auditHiddenStatement: statement mentions non-allowlisted project constants:
[Nat.auditWrapper] -/
#guard_msgs in
#stmtSurfaceSubset auditHiddenStatement ⊆ []

/-- info: auditHiddenStatement: statement's project surface ⊆ allowlist ✓ -/
#guard_msgs in
#stmtSurfaceSubset auditHiddenStatement ⊆ [Nat.auditWrapper]

theorem auditCoreStatement (n : Nat) : n + 0 = n := rfl
/-- info: auditCoreStatement: statement's project surface ⊆ allowlist ✓ -/
#guard_msgs in
#stmtSurfaceSubset auditCoreStatement ⊆ []

def auditIndependentSpec : Nat := 1
/-- info: auditIndependentSpec: definition does not reference [auditKernel] — non-circular ✓ -/
#guard_msgs in
#specNonCircular auditIndependentSpec avoiding [auditKernel]

-- Origin tracking must also work after importing a project declaration.
-- This definition is from VeriTile.Meta.StatementAudit, with a public name.
theorem auditImportedStatement :
    VeriTile.Meta.valueConsts = VeriTile.Meta.valueConsts := rfl
/-- error: auditImportedStatement: statement mentions non-allowlisted project constants:
[VeriTile.Meta.valueConsts] -/
#guard_msgs in
#stmtSurfaceSubset auditImportedStatement ⊆ []
