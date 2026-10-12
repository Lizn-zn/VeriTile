import VeriTile.Triton.Semantics.Step
import VeriTile.Meta.StatementAudit

/-! Register-only fragment frames. A syntactic derivation lists every register
that may change, including both arms of every branch. These helpers preserve
memory, launch coordinates and all other registers without expanding numerical
expressions. Loops and memory writes require their separate execution proofs. -/
namespace VeriTile.Bench.TritonBenchG.AttentionScore.RealFrame
open VeriTile.Triton

structure Frame (writes : List RegName) (before after : BlockState) : Prop where
  mem : after.mem = before.mem
  pids : after.pids = before.pids
  numPids : after.numPids = before.numPids
  undef : after.undef = before.undef
  regs : ∀ d sh name, name ∉ writes → after.regs d sh name = before.regs d sh name

theorem Frame.refl (writes : List RegName) (s : BlockState) : Frame writes s s :=
  ⟨rfl,rfl,rfl,rfl,fun _ _ _ _ => rfl⟩
theorem Frame.trans {writes : List RegName} {s t u : BlockState}
    (a : Frame writes s t) (b : Frame writes t u) : Frame writes s u :=
  ⟨b.mem.trans a.mem, b.pids.trans a.pids, b.numPids.trans a.numPids,
    b.undef.trans a.undef, fun d sh name hn => (b.regs d sh name hn).trans (a.regs d sh name hn)⟩
theorem Frame.mono {writes more : List RegName} {s t : BlockState}
    (a : Frame writes s t) (h : writes ⊆ more) : Frame more s t :=
  ⟨a.mem,a.pids,a.numPids,a.undef,fun d sh name hn => a.regs d sh name (fun hw => hn (h hw))⟩
theorem Frame.setReg (writes : List RegName) (s : BlockState)
    (name : RegName) (d : TileDType) (sh : TileShape) (value : Tile d sh)
    (hw : name ∈ writes) : Frame writes s (s.setReg name d sh value) := by
  refine ⟨rfl,rfl,rfl,rfl,?_⟩
  intro dt shape key hk
  have hn : key ≠ name := by rintro rfl; exact hk hw
  simp [hn]

inductive Code (writes : List RegName) : List Stmt → Prop
  | nil : Code writes []
  | assign (d : TileDType) (sh : TileShape) (name : RegName) (value : Op d sh)
      (tail : List Stmt) (member : name ∈ writes) (next : Code writes tail) :
      Code writes (.assign d sh name value :: tail)
  | ifThen (condition : Op .bool []) (body tail : List Stmt)
      (inside : Code writes body) (next : Code writes tail) :
      Code writes (.ifThen condition body :: tail)
  | ifThenElse (condition : Op .bool []) (yes no tail : List Stmt)
      (yesCode : Code writes yes) (noCode : Code writes no) (next : Code writes tail) :
      Code writes (.ifThenElse condition yes no :: tail)

/-- Any successful execution of a checked register-only fragment has the
full declared frame. No premise refers to the fragment's numerical output. -/
theorem Code.frame {writes : List RegName} {code : List Stmt} (hc : Code writes code)
    {s t : BlockState} (he : stepStmts code s = some t) : Frame writes s t := by
  induction hc generalizing s t with
  | nil =>
      simp only [stepStmts.nil, Option.some.injEq] at he
      subst t
      exact Frame.refl _ _
  | assign d sh name value tail member next ih =>
      cases hv : evalOp value s with
      | none => simp [stepStmts,stepStmt,hv] at he
      | some v =>
          have hs : stepStmt (.assign d sh name value) s = some (s.setReg name d sh v) := by
            simp [stepStmt,hv]
          rw [stepStmts.cons_some hs] at he
          exact (Frame.setReg writes s name d sh v member).trans (ih he)
  | ifThen condition body tail inside next ihBody ihTail =>
      cases hv : evalOp condition s with
      | none => simp [stepStmts,stepStmt,hv] at he
      | some v =>
          by_cases hb : v.data PUnit.unit = Bool.true
          · cases hi : stepStmts body s with
            | none => simp [stepStmts,stepStmt,hv,hb,hi] at he
            | some u =>
                have ht : stepStmts tail u = some t := by
                  simpa [stepStmts,stepStmt,hv,hb,hi] using he
                exact (ihBody hi).trans (ihTail ht)
          · have ht : stepStmts tail s = some t := by
              simpa [stepStmts,stepStmt,hv,hb] using he
            exact ihTail ht
  | ifThenElse condition yes no tail yesCode noCode next ihYes ihNo ihTail =>
      cases hv : evalOp condition s with
      | none => simp [stepStmts,stepStmt,hv] at he
      | some v =>
          by_cases hb : v.data PUnit.unit = Bool.true
          · cases hi : stepStmts yes s with
            | none => simp [stepStmts,stepStmt,hv,hb,hi] at he
            | some u =>
                have ht : stepStmts tail u = some t := by
                  simpa [stepStmts,stepStmt,hv,hb,hi] using he
                exact (ihYes hi).trans (ihTail ht)
          · cases hi : stepStmts no s with
            | none => simp [stepStmts,stepStmt,hv,hb,hi] at he
            | some u =>
                have ht : stepStmts tail u = some t := by
                  simpa [stepStmts,stepStmt,hv,hb,hi] using he
                exact (ihNo hi).trans (ihTail ht)

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionScore.RealFrame
