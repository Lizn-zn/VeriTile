import VeriTile.Triton.Float.ProfiledSoundness
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.VarLenCopy.FPProof
open VeriTile.Triton FP.Structural

/-- Signed source-base commutation is exact, including missing registers. -/
theorem signed_add_commutes {α : Type} [Inhabited α] (M : Algebra α)
    (p : Option ComputeDType) (s : State α) (a b : Op .int []) :
    evalOp M p (.add .int .nil a b) s = evalOp M p (.add .int .nil b a) s := by
  change (evalOp M p a s >>= fun x ↦ evalOp M p b s >>= fun y ↦
      some (bop Int.add .nil x y)) =
    (evalOp M p b s >>= fun y ↦ evalOp M p a s >>= fun x ↦
      some (bop Int.add .nil y x))
  cases ha : evalOp M p a s <;> cases hb : evalOp M p b s
  all_goals first
    | rfl
    | apply congrArg some
      funext i
      exact Int.add_comm _ _

 def offsets (v : Bool) (B : Nat) : Op .nat [B] :=
  .castIntToNat (.add .int .scalarL
    (if v then .add .int .nil (.ref .int [] "i") (.ref .int [] "old_start")
     else .add .int .nil (.ref .int [] "old_start") (.ref .int [] "i"))
    (.castNatToInt (.ref .nat [B] "old_offset")))

 theorem offsets_commute {α : Type} [Inhabited α] (M : Algebra α)
    (p : Option ComputeDType) (s : State α) (B : Nat) :
    evalOp M p (offsets Bool.false B) s = evalOp M p (offsets Bool.true B) s := by
  simp only [offsets, Bool.false_eq_true, ↓reduceIte]
  change ((evalOp M p (.add .int .nil (.ref .int [] "old_start") (.ref .int [] "i")) s
      >>= fun x ↦ evalOp M p (.castNatToInt (.ref .nat [B] "old_offset")) s
      >>= fun y ↦ some (bop Int.add .scalarL x y)) >>= fun z ↦
        some (fun i ↦ (z i).toNat)) = _
  rw [signed_add_commutes M p s (.ref .int [] "old_start") (.ref .int [] "i")]
  rfl

 theorem load_commutes {α : Type} [Inhabited α] (M : Algebra α)
    (p : Option ComputeDType) (s : State α) (d : TileDType) (src : Region d)
    (B : Nat) (mask : Op .bool [B]) (padding : PaddingOption) :
    evalOp M p (.load d (.region src (offsets Bool.false B)) (.mask mask) padding) s =
      evalOp M p (.load d (.region src (offsets Bool.true B)) (.mask mask) padding) s := by
  cases d
  all_goals
    conv_lhs => rw [evalOp_unfold]
    conv_rhs => rw [evalOp_unfold]
    dsimp only
    rw [offsets_commute]

theorem load_commutes_expanded {α : Type} [Inhabited α] (M : Algebra α)
    (p : Option ComputeDType) (s : State α) (d : TileDType) (src : Region d)
    (B : Nat) (mask : Op .bool [B]) (padding : PaddingOption) :
    evalOp M p (.load d (.region src (.castIntToNat (.add .int .scalarL
      (.add .int .nil (.ref .int [] "old_start") (.ref .int [] "i"))
      (.castNatToInt (.ref .nat [B] "old_offset"))))) (.mask mask) padding) s =
    evalOp M p (.load d (.region src (.castIntToNat (.add .int .scalarL
      (.add .int .nil (.ref .int [] "i") (.ref .int [] "old_start"))
      (.castNatToInt (.ref .nat [B] "old_offset"))))) (.mask mask) padding) s :=
  load_commutes M p s d src B mask padding

end VeriTile.Bench.TritonBenchG.VarLenCopy.FPProof

#guard_msgs (drop info) in
#auditModuleAxioms
