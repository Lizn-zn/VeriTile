/-
VeriTile.Meta.Specification

The `specification` declaration keyword. A **specification** is a public
headline theorem — the statement a reader audits as the file's trust
surface. It elaborates *identically* to `theorem` (same kernel object, same
axiom footprint, `#axiomsClean`/`#print axioms` unaffected); the keyword is
a machine-readable marker, so headline discovery in the audit tooling
(`scripts/spec_sheet.py`, `bench/check_proof_gap_manifest.py`,
`bench/audit_trust_prep.py`) can key on syntax instead of name-suffix
heuristics.

Modeled on Mathlib's `lemma` command macro.
-/

/-- `specification` declares a public headline theorem. Identical to
`theorem` after elaboration; the keyword marks the declaration as a file's
public spec surface for readers and for the audit tooling. -/
syntax (name := specification) declModifiers
  group("specification " declId ppIndent(declSig) declVal) : command

macro_rules
  | `($mods:declModifiers specification%$tk $id:declId $sig:declSig $val:declVal) =>
    `($mods:declModifiers theorem%$tk $id $sig $val)
