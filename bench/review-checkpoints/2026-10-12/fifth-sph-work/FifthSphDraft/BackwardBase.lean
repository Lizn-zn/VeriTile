import FifthSphDraft.BackwardSupport
namespace VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.BackwardBase
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

theorem assigns_exists {d sh name} {e : Op d sh} {s : BlockState} {rest : List Stmt}
    (he : ∃ v,evalOp e s=some v)
    (hr : ∀ v,∃ t,stepStmts rest (s.setReg name d sh v)=some t) :
    ∃ t,stepStmts (.assign d sh name e :: rest) s=some t := by
  obtain ⟨v,hv⟩ := he
  obtain ⟨t,ht⟩ := hr v
  refine ⟨t,?_⟩
  simpa only [stepStmts,stepStmt,hv,Option.bind_some] using ht

theorem executes (X dX g : RegionName) (B N M off stride : Nat) (u : BlockState) :
    ∃ v,exec (source X dX g B N M off stride) u=some v := by
  rw [source_eq_algorithm]
  change ∃ v,stepStmts (NormalizedSource.algorithm X dX g B N M off stride).body u=some v
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t0
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t1
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t2
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t3
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t4
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t5
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t6
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t7
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t8
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t9
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t10
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t11
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t12
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t13
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t14
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t15
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t16
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t17
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t18
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t19
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t20
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t21
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t22
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t23
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t24
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t25
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t26
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t27
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t28
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t29
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t30
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t31
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t32
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t33
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t34
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t35
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t36
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t37
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t38
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t39
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t40
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t41
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t42
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t43
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t44
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t45
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t46
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t47
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t48
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t49
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t50
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t51
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t52
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t53
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t54
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t55
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t56
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t57
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t58
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t59
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t60
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t61
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t62
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t63
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t64
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t65
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t66
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t67
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t68
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t69
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t70
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t71
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t72
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t73
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t74
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t75
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t76
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t77
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t78
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t79
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t80
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t81
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t82
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t83
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t84
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t85
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t86
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t87
  apply assigns_exists
  case he =>
    simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte]
    exact ⟨_,rfl⟩
  intro t88
  all_goals simp (maxSteps := 3000000) [stepStmts,stepStmt,evalOp.eq_def]


#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.BackwardBase
