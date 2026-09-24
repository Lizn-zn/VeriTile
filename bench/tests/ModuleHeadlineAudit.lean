import VeriTile.Meta.StatementAudit

namespace HeadlineAuditRegression

-- Registration must be independent of line breaks, attributes, privacy,
-- quoted/Unicode identifiers, and user namespaces.
@[simp] specification
  multiline (n : Nat) : n + 0 = n := rfl

private specification «私有结论» : True := True.intro

namespace Nested
specification «quoted headline» : True := True.intro
end Nested

theorem
  legacy_correct : True := True.intro

#auditModuleAxioms

end HeadlineAuditRegression
