import FifthSphDraft.NormalizedSource
import FifthSphDraft.RealMath
import VeriTile.Triton.Float.Erasure
import VeriTile.Triton.Memory.TraceProofs
namespace VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.BackwardBase
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
set_option linter.unusedTactic false
set_option linter.unreachableTactic false

def source (X dX g : RegionName) (B N M off stride : Nat) : Kernel :=
  (TypedSource.backward .fp32 .fp32 .fp32 Bool.false X dX g B N M off stride).eraseDType.toAlgKernel

theorem source_eq_algorithm (X dX g : RegionName) (B N M off stride : Nat) :
    source X dX g B N M off stride=NormalizedSource.algorithm X dX g B N M off stride :=
  NormalizedSource.original_eq_algorithm X dX g B N M off stride

def coord (pid B j c : Nat) : Nat := j*3+B*3*pid+c
def output (pid B stride off j c : Nat) : Nat := j*stride+B*stride*pid+off+c

theorem assigns_inv {d sh name} {e : Op d sh} {s t : BlockState} {rest : List Stmt}
    (h : stepStmts (.assign d sh name e :: rest) s=some t) :
    ∃ v,evalOp e s=some v ∧ stepStmts rest (s.setReg name d sh v)=some t := by
  simp only [stepStmts,stepStmt] at h
  cases hv : evalOp e s with
  | none => simp [hv] at h
  | some v => exact ⟨v,rfl,by simpa [hv] using h⟩

theorem frame (X dX g : RegionName) (B N M off stride : Nat) (u v : BlockState)
    (he : exec (source X dX g B N M off stride) u=some v) (r : RegionName) (o : Nat)
    (hmiss : ∀ c : Fin 3,∀ j : Fin B,coord u.pid B j.val c.val<N → ¬(dX=r ∧ coord u.pid B j.val c.val=o)) :
    v.mem r o=u.mem r o := by
  rw [source_eq_algorithm] at he
  change stepStmts (NormalizedSource.algorithm X dX g B N M off stride).body u=some v at he
  obtain ⟨t0,ht0,he⟩ := assigns_inv he
  simp [evalOp.eq_def,tile_elementwise] at ht0
  subst t0
  obtain ⟨t1,ht1,he⟩ := assigns_inv he
  simp [evalOp.eq_def,tile_elementwise] at ht1
  subst t1
  obtain ⟨t2,ht2,he⟩ := assigns_inv he
  simp [evalOp.eq_def,tile_elementwise] at ht2
  subst t2
  obtain ⟨t3,ht3,he⟩ := assigns_inv he
  simp [evalOp.eq_def,tile_elementwise] at ht3
  subst t3
  obtain ⟨t4,ht4,he⟩ := assigns_inv he
  clear ht4
  obtain ⟨t5,ht5,he⟩ := assigns_inv he
  clear ht5
  obtain ⟨t6,ht6,he⟩ := assigns_inv he
  clear ht6
  obtain ⟨t7,ht7,he⟩ := assigns_inv he
  simp [evalOp.eq_def,tile_elementwise] at ht7
  subst t7
  obtain ⟨t8,ht8,he⟩ := assigns_inv he
  simp [evalOp.eq_def,tile_elementwise] at ht8
  subst t8
  obtain ⟨t9,ht9,he⟩ := assigns_inv he
  clear ht9
  obtain ⟨t10,ht10,he⟩ := assigns_inv he
  clear ht10
  obtain ⟨t11,ht11,he⟩ := assigns_inv he
  clear ht11
  obtain ⟨t12,ht12,he⟩ := assigns_inv he
  clear ht12
  obtain ⟨t13,ht13,he⟩ := assigns_inv he
  clear ht13
  obtain ⟨t14,ht14,he⟩ := assigns_inv he
  clear ht14
  obtain ⟨t15,ht15,he⟩ := assigns_inv he
  clear ht15
  obtain ⟨t16,ht16,he⟩ := assigns_inv he
  clear ht16
  obtain ⟨t17,ht17,he⟩ := assigns_inv he
  clear ht17
  obtain ⟨t18,ht18,he⟩ := assigns_inv he
  clear ht18
  obtain ⟨t19,ht19,he⟩ := assigns_inv he
  clear ht19
  obtain ⟨t20,ht20,he⟩ := assigns_inv he
  clear ht20
  obtain ⟨t21,ht21,he⟩ := assigns_inv he
  clear ht21
  obtain ⟨t22,ht22,he⟩ := assigns_inv he
  clear ht22
  obtain ⟨t23,ht23,he⟩ := assigns_inv he
  clear ht23
  obtain ⟨t24,ht24,he⟩ := assigns_inv he
  clear ht24
  obtain ⟨t25,ht25,he⟩ := assigns_inv he
  clear ht25
  obtain ⟨t26,ht26,he⟩ := assigns_inv he
  clear ht26
  obtain ⟨t27,ht27,he⟩ := assigns_inv he
  clear ht27
  obtain ⟨t28,ht28,he⟩ := assigns_inv he
  clear ht28
  obtain ⟨t29,ht29,he⟩ := assigns_inv he
  clear ht29
  obtain ⟨t30,ht30,he⟩ := assigns_inv he
  clear ht30
  obtain ⟨t31,ht31,he⟩ := assigns_inv he
  clear ht31
  obtain ⟨t32,ht32,he⟩ := assigns_inv he
  clear ht32
  obtain ⟨t33,ht33,he⟩ := assigns_inv he
  clear ht33
  obtain ⟨t34,ht34,he⟩ := assigns_inv he
  clear ht34
  obtain ⟨t35,ht35,he⟩ := assigns_inv he
  clear ht35
  obtain ⟨t36,ht36,he⟩ := assigns_inv he
  clear ht36
  obtain ⟨t37,ht37,he⟩ := assigns_inv he
  clear ht37
  obtain ⟨t38,ht38,he⟩ := assigns_inv he
  clear ht38
  obtain ⟨t39,ht39,he⟩ := assigns_inv he
  clear ht39
  obtain ⟨t40,ht40,he⟩ := assigns_inv he
  clear ht40
  obtain ⟨t41,ht41,he⟩ := assigns_inv he
  clear ht41
  obtain ⟨t42,ht42,he⟩ := assigns_inv he
  clear ht42
  obtain ⟨t43,ht43,he⟩ := assigns_inv he
  clear ht43
  obtain ⟨t44,ht44,he⟩ := assigns_inv he
  clear ht44
  obtain ⟨t45,ht45,he⟩ := assigns_inv he
  clear ht45
  obtain ⟨t46,ht46,he⟩ := assigns_inv he
  clear ht46
  obtain ⟨t47,ht47,he⟩ := assigns_inv he
  clear ht47
  obtain ⟨t48,ht48,he⟩ := assigns_inv he
  clear ht48
  obtain ⟨t49,ht49,he⟩ := assigns_inv he
  clear ht49
  obtain ⟨t50,ht50,he⟩ := assigns_inv he
  clear ht50
  obtain ⟨t51,ht51,he⟩ := assigns_inv he
  clear ht51
  obtain ⟨t52,ht52,he⟩ := assigns_inv he
  clear ht52
  obtain ⟨t53,ht53,he⟩ := assigns_inv he
  clear ht53
  obtain ⟨t54,ht54,he⟩ := assigns_inv he
  clear ht54
  obtain ⟨t55,ht55,he⟩ := assigns_inv he
  clear ht55
  obtain ⟨t56,ht56,he⟩ := assigns_inv he
  clear ht56
  obtain ⟨t57,ht57,he⟩ := assigns_inv he
  clear ht57
  obtain ⟨t58,ht58,he⟩ := assigns_inv he
  clear ht58
  obtain ⟨t59,ht59,he⟩ := assigns_inv he
  clear ht59
  obtain ⟨t60,ht60,he⟩ := assigns_inv he
  clear ht60
  obtain ⟨t61,ht61,he⟩ := assigns_inv he
  clear ht61
  obtain ⟨t62,ht62,he⟩ := assigns_inv he
  clear ht62
  obtain ⟨t63,ht63,he⟩ := assigns_inv he
  clear ht63
  obtain ⟨t64,ht64,he⟩ := assigns_inv he
  clear ht64
  obtain ⟨t65,ht65,he⟩ := assigns_inv he
  clear ht65
  obtain ⟨t66,ht66,he⟩ := assigns_inv he
  clear ht66
  obtain ⟨t67,ht67,he⟩ := assigns_inv he
  clear ht67
  obtain ⟨t68,ht68,he⟩ := assigns_inv he
  clear ht68
  obtain ⟨t69,ht69,he⟩ := assigns_inv he
  clear ht69
  obtain ⟨t70,ht70,he⟩ := assigns_inv he
  clear ht70
  obtain ⟨t71,ht71,he⟩ := assigns_inv he
  clear ht71
  obtain ⟨t72,ht72,he⟩ := assigns_inv he
  clear ht72
  obtain ⟨t73,ht73,he⟩ := assigns_inv he
  clear ht73
  obtain ⟨t74,ht74,he⟩ := assigns_inv he
  clear ht74
  obtain ⟨t75,ht75,he⟩ := assigns_inv he
  clear ht75
  obtain ⟨t76,ht76,he⟩ := assigns_inv he
  clear ht76
  obtain ⟨t77,ht77,he⟩ := assigns_inv he
  clear ht77
  obtain ⟨t78,ht78,he⟩ := assigns_inv he
  clear ht78
  obtain ⟨t79,ht79,he⟩ := assigns_inv he
  clear ht79
  obtain ⟨t80,ht80,he⟩ := assigns_inv he
  clear ht80
  obtain ⟨t81,ht81,he⟩ := assigns_inv he
  clear ht81
  obtain ⟨t82,ht82,he⟩ := assigns_inv he
  clear ht82
  obtain ⟨t83,ht83,he⟩ := assigns_inv he
  clear ht83
  obtain ⟨t84,ht84,he⟩ := assigns_inv he
  clear ht84
  obtain ⟨t85,ht85,he⟩ := assigns_inv he
  clear ht85
  obtain ⟨t86,ht86,he⟩ := assigns_inv he
  clear ht86
  obtain ⟨t87,ht87,he⟩ := assigns_inv he
  clear ht87
  obtain ⟨t88,ht88,he⟩ := assigns_inv he
  clear ht88
  simp (maxSteps := 3000000) [stepStmts,stepStmt,evalOp.eq_def,tile_elementwise] at he
  subst v
  refine Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_)
    (Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_)
      (Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_) rfl))
  · intro j _ hj ho
    exact hmiss 2 j.1 (by simpa [coord] using hj) (by simpa [coord] using ho)
  · intro j _ hj ho
    exact hmiss 1 j.1 (by simpa [coord] using hj) (by simpa [coord] using ho)
  · intro j _ hj ho
    exact hmiss 0 j.1 (by simpa [coord] using hj) (by simpa [coord] using ho)



theorem flatten_ok (X dX g : RegionName) (B N M off stride : Nat) :
    (source X dX g B N M off stride).FlattenOk := by
  unfold Kernel.FlattenOk
  simp (maxSteps := 3000000) [source,TypedSource.backward,TypedSource.backward_fp32_fp32_fp32,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    IntegralDType.eraseDType,ComputeDType.eraseDType,Region.cast,Bool.false_eq_true,Except.bind,bind,
    exec,stepStmts,stepStmt,evalOp.eq_def,tile_elementwise,StmtList.FlattenOk,Stmt.FlattenOk,Op.FlattenOk.eq_def]

structure Bounds (X dX g : RegionName) (B N M off stride : Nat) (u : BlockState) (bounds : RegionBounds) : Prop where
  coordinates : ∀ c : Fin 3,∀ j : Fin B,coord u.pid B j.val c.val<N → coord u.pid B j.val c.val<bounds X
  gradient : ∀ c : Fin 3,∀ j : Fin B,coord u.pid B j.val c.val<N → coord u.pid B j.val c.val<bounds dX
  upstream : ∀ c : Fin 11,∀ j : Fin B,output u.pid B stride off j.val c.val<M → output u.pid B stride off j.val c.val<bounds g

theorem trace_safe (X dX g : RegionName) (B N M off stride : Nat) (u : BlockState) (bounds : RegionBounds)
    (hb : Bounds X dX g B N M off stride u bounds) :
    Kernel.TraceSafe bounds (source X dX g B N M off stride) u := by
  rw [source_eq_algorithm]
  change Stmt.TraceSafeList bounds (NormalizedSource.algorithm X dX g B N M off stride).body u
  refine TraceProofs.assign_eval_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t0 ht0
  simp [evalOp.eq_def,tile_elementwise] at ht0
  subst t0
  refine TraceProofs.assign_eval_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t1 ht1
  simp [evalOp.eq_def,tile_elementwise] at ht1
  subst t1
  refine TraceProofs.assign_eval_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t2 ht2
  simp [evalOp.eq_def,tile_elementwise] at ht2
  subst t2
  refine TraceProofs.assign_eval_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t3 ht3
  simp [evalOp.eq_def,tile_elementwise] at ht3
  subst t3
  refine TraceProofs.assign_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t4
  refine TraceProofs.assign_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t5
  refine TraceProofs.assign_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t6
  refine TraceProofs.assign_eval_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t7 ht7
  simp [evalOp.eq_def,tile_elementwise] at ht7
  subst t7
  refine TraceProofs.assign_eval_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t8 ht8
  simp [evalOp.eq_def,tile_elementwise] at ht8
  subst t8
  refine TraceProofs.assign_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t9
  refine TraceProofs.assign_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t10
  refine TraceProofs.assign_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t11
  refine TraceProofs.assign_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t12
  refine TraceProofs.assign_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t13
  refine TraceProofs.assign_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t14
  refine TraceProofs.assign_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t15
  refine TraceProofs.assign_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t16
  refine TraceProofs.assign_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t17
  refine TraceProofs.assign_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t18
  refine TraceProofs.assign_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t19
  refine TraceProofs.assign_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t20
  refine TraceProofs.assign_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t21
  refine TraceProofs.assign_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t22
  refine TraceProofs.assign_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t23
  refine TraceProofs.assign_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t24
  refine TraceProofs.assign_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t25
  refine TraceProofs.assign_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t26
  refine TraceProofs.assign_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t27
  refine TraceProofs.assign_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t28
  refine TraceProofs.assign_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t29
  refine TraceProofs.assign_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t30
  refine TraceProofs.assign_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t31
  refine TraceProofs.assign_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t32
  refine TraceProofs.assign_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t33
  refine TraceProofs.assign_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t34
  refine TraceProofs.assign_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t35
  refine TraceProofs.assign_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t36
  refine TraceProofs.assign_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t37
  refine TraceProofs.assign_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t38
  refine TraceProofs.assign_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t39
  refine TraceProofs.assign_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t40
  refine TraceProofs.assign_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t41
  refine TraceProofs.assign_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t42
  refine TraceProofs.assign_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t43
  refine TraceProofs.assign_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t44
  refine TraceProofs.assign_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t45
  refine TraceProofs.assign_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t46
  refine TraceProofs.assign_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t47
  refine TraceProofs.assign_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t48
  refine TraceProofs.assign_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t49
  refine TraceProofs.assign_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t50
  refine TraceProofs.assign_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t51
  refine TraceProofs.assign_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t52
  refine TraceProofs.assign_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t53
  refine TraceProofs.assign_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t54
  refine TraceProofs.assign_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t55
  refine TraceProofs.assign_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t56
  refine TraceProofs.assign_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t57
  refine TraceProofs.assign_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t58
  refine TraceProofs.assign_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t59
  refine TraceProofs.assign_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t60
  refine TraceProofs.assign_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t61
  refine TraceProofs.assign_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t62
  refine TraceProofs.assign_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t63
  refine TraceProofs.assign_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t64
  refine TraceProofs.assign_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t65
  refine TraceProofs.assign_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t66
  refine TraceProofs.assign_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t67
  refine TraceProofs.assign_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t68
  refine TraceProofs.assign_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t69
  refine TraceProofs.assign_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t70
  refine TraceProofs.assign_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t71
  refine TraceProofs.assign_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t72
  refine TraceProofs.assign_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t73
  refine TraceProofs.assign_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t74
  refine TraceProofs.assign_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t75
  refine TraceProofs.assign_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t76
  refine TraceProofs.assign_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t77
  refine TraceProofs.assign_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t78
  refine TraceProofs.assign_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t79
  refine TraceProofs.assign_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t80
  refine TraceProofs.assign_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t81
  refine TraceProofs.assign_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t82
  refine TraceProofs.assign_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t83
  refine TraceProofs.assign_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t84
  refine TraceProofs.assign_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t85
  refine TraceProofs.assign_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t86
  refine TraceProofs.assign_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t87
  refine TraceProofs.assign_cons (by
    simp (maxSteps := 3000000) [Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
    all_goals intro j hj
    all_goals first
      | simpa [coord,output] using hb.coordinates 0 j hj
      | simpa [coord,output] using hb.coordinates 1 j hj
      | simpa [coord,output] using hb.coordinates 2 j hj
      | simpa [coord,output] using hb.gradient 0 j hj
      | simpa [coord,output] using hb.gradient 1 j hj
      | simpa [coord,output] using hb.gradient 2 j hj
      | simpa [coord,output] using hb.upstream 0 j hj
      | simpa [coord,output] using hb.upstream 1 j hj
      | simpa [coord,output] using hb.upstream 2 j hj
      | simpa [coord,output] using hb.upstream 3 j hj
      | simpa [coord,output] using hb.upstream 4 j hj
      | simpa [coord,output] using hb.upstream 5 j hj
      | simpa [coord,output] using hb.upstream 6 j hj
      | simpa [coord,output] using hb.upstream 7 j hj
      | simpa [coord,output] using hb.upstream 8 j hj
      | simpa [coord,output] using hb.upstream 9 j hj
      | simpa [coord,output] using hb.upstream 10 j hj
  ) ?_
  intro t88
  simp (maxSteps := 3000000) [Stmt.TraceSafeList,Stmt.TraceSafe,stepStmts,stepStmt,Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,evalOp.eq_def,tile_elementwise,
      Tile.bop,Tile.cop,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt]
  repeat' apply And.intro
  all_goals intro j hj
  all_goals first
    | simpa [coord,output] using hb.coordinates 0 j hj
    | simpa [coord,output] using hb.coordinates 1 j hj
    | simpa [coord,output] using hb.coordinates 2 j hj
    | simpa [coord,output] using hb.gradient 0 j hj
    | simpa [coord,output] using hb.gradient 1 j hj
    | simpa [coord,output] using hb.gradient 2 j hj
    | simpa [coord,output] using hb.upstream 0 j hj
    | simpa [coord,output] using hb.upstream 1 j hj
    | simpa [coord,output] using hb.upstream 2 j hj
    | simpa [coord,output] using hb.upstream 3 j hj
    | simpa [coord,output] using hb.upstream 4 j hj
    | simpa [coord,output] using hb.upstream 5 j hj
    | simpa [coord,output] using hb.upstream 6 j hj
    | simpa [coord,output] using hb.upstream 7 j hj
    | simpa [coord,output] using hb.upstream 8 j hj
    | simpa [coord,output] using hb.upstream 9 j hj
    | simpa [coord,output] using hb.upstream 10 j hj


#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.BackwardBase
