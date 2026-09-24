/-
VeriTile.Meta.Specification

The `specification` declaration keyword. A **specification** is a public
headline theorem — the statement a reader audits as the file's trust
surface. It elaborates *identically* to `theorem` (same kernel object, same
axiom footprint, `#axiomsClean`/`#print axioms` unaffected); the keyword is
a machine-readable marker registered in Lean's environment by `kernel_headline`.
The trust audit enumerates this metadata, independently of source formatting.
Source inventories (`scripts/spec_sheet.py`, `bench/check_proof_gap_manifest.py`)
also use the keyword as a visible declaration marker.

Modeled on Mathlib's `lemma` command macro.

This module also defines the dual marker **`denotation`**: the declaration
keyword for a kernel's denotation ⟦·⟧ (issue #487) — the one audited
definition a denotation-style specification's statement depends on. It
elaborates identically to `def`; combine with modifiers as usual
(`noncomputable denotation ⟦k⟧ ... := ...`).
-/

import Lean

namespace VeriTile.Meta

/-- Headline declarations are registered during elaboration, independently of
source layout, identifier spelling, and declaration modifiers. -/
initialize headlineAttr : Lean.TagAttribute ←
  Lean.registerTagAttribute `kernel_headline "A public kernel specification theorem."

end VeriTile.Meta

/-- `specification` declares a public headline theorem. Identical to
`theorem` after elaboration; the keyword marks the declaration as a file's
public spec surface for readers and for the audit tooling. -/
syntax (name := specification) declModifiers
  group("specification " declId ppIndent(declSig) declVal) : command

open Lean in
macro_rules
  | `($mods:declModifiers specification%$tk $id:declId $sig:declSig $val:declVal) => do
    let decl ← `($mods:declModifiers theorem%$tk $id $sig $val)
    let name : TSyntax `ident := ⟨id.raw[0]⟩
    let tag ← `(attribute [kernel_headline] $name)
    return mkNullNode #[decl, tag]

/-- `denotation` declares a kernel's denotation ⟦·⟧ — the single audited
definition carrying all addressing/layout content of a denotation-style
specification (issue #487). Identical to `def` after elaboration; the
keyword marks the declaration for readers and for the audit tooling, dual
to `specification`. -/
syntax (name := denotation) declModifiers
  group("denotation " declId ppIndent(declSig) declVal) : command

open Lean in
macro_rules
  | `($mods:declModifiers denotation%$tk $id:declId $sig:declSig $val:declVal) => do
    -- `def`'s signature slot is `optDeclSig`; repackage the mandatory
    -- `declSig` (denotations always state their type) into it.
    let optSig := mkNode ``Lean.Parser.Command.optDeclSig
      #[sig.raw[0], mkNullNode #[sig.raw[1]]]
    `($mods:declModifiers def%$tk $id:declId $(⟨optSig⟩) $val:declVal)
