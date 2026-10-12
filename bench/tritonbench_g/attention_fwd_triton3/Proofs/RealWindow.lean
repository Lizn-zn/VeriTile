import bench.tritonbench_g.attention_fwd_triton3.Proofs.RealInitSource

/-! Execute the complete source window-selection branch. Bounds retain signed
truncating division; only the lower bound is clamped at zero. The nested
complement arm is kept in the AST even though the outer condition excludes it.
RealWindowSource checks this statement against both complete implementations. -/
namespace VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealWindow
open VeriTile.Triton RealScores

def lowerExpr (BM BN : Nat) (offset size : Int) : Op .int [] :=
  .mul .int .nil
    (.truncDiv .nil
      (.add .int .nil
        (.sub .int .nil
          (.add .int .nil (.mul .int .nil (.castNatToInt (.ref .nat [] "start_m")) (.constInt BM))
            (.constInt offset)) (.constInt size)) (.constInt 1))
      (.constInt BN)) (.constInt BN)

def upperExpr (BM BN : Nat) (offset : Int) : Op .int [] :=
  .mul .int .nil
    (.truncDiv .nil
      (.add .int .nil
        (.add .int .nil
          (.sub .int .nil
            (.mul .int .nil (.add .int .nil (.castNatToInt (.ref .nat [] "start_m")) (.constInt 1))
              (.constInt BM)) (.constInt 1)) (.constInt offset)) (.constInt BN))
      (.constInt BN)) (.constInt BN)

def complementUpperExpr (BM BN : Nat) (offset size : Int) : Op .int [] :=
  .mul .int .nil
    (.truncDiv .nil
      (.sub .int .nil
        (.add .int .nil
          (.sub .int .nil
            (.add .int .nil
              (.mul .int .nil (.add .int .nil (.castNatToInt (.ref .nat [] "start_m")) (.constInt 1))
                (.constInt BM)) (.constInt offset)) (.constInt size)) (.constInt BN)) (.constInt 1))
      (.constInt BN)) (.constInt BN)

def clampLower : Stmt :=
  .ifThen (.lt .int .nil (.ref .int [] "lo") (.constInt 0)) [.assign .int [] "lo" (.constInt 0)]

def clampUpper (context : Nat) : Stmt :=
  .ifThen (.gt .int .nil (.ref .int [] "hi") (.castNatToInt (.constNat context)))
    [.assign .int [] "hi" (.constInt context)]

def advance (BN BD : Nat) : List Stmt :=
  [.assign .int [] "lo" (.ref .int [] "lo"),
   .assign .blockPtr [BD, BN] "K_block_ptr"
     (.advanceBlockPtrDyn (.ref .blockPtr [BD, BN] "K_block_ptr") [.constInt 0, .ref .int [] "lo"]),
   .assign .blockPtr [BN, BD] "V_block_ptr"
     (.advanceBlockPtrDyn (.ref .blockPtr [BN, BD] "V_block_ptr") [.ref .int [] "lo", .constInt 0])]

def code (BM BN BD context window complement : Nat) (offset size : Int) : Stmt :=
  .ifThenElse (.boolAnd .nil (.ne .nat .nil (.constNat window) (.constNat 0))
      (.eq .nat .nil (.constNat complement) (.constNat 0)))
    [.ifThenElse (.ne .nat .nil (.constNat complement) (.constNat 0))
      [.assign .int [] "lo" (.constInt 0), .assign .int [] "hi" (complementUpperExpr BM BN offset size)]
      ([.assign .int [] "lo" (lowerExpr BM BN offset size),
        .assign .int [] "hi" (upperExpr BM BN offset), clampLower, clampUpper context] ++ advance BN BD)]
    [.assign .int [] "lo" (.constInt 0), .assign .int [] "hi" (.constInt context)]

def rawLower (startM BM BN : Nat) (offset size : Int) : Int :=
  (((startM * BM : Nat) : Int) + offset - size + 1).tdiv BN * BN

def rawUpper (startM BM BN : Nat) (offset : Int) : Int :=
  ((((startM + 1) * BM : Nat) : Int) - 1 + offset + BN).tdiv BN * BN

/-- Unlike the mathematical key-set bound, this value retains a negative hi. -/
def upper (startM BM BN context window complement : Nat) (offset : Int) : Int :=
  if window ≠ 0 ∧ complement = 0 then min (rawUpper startM BM BN offset) context else context

theorem lower_eval (s : BlockState) (startM BM BN : Nat) (offset size : Int)
    (hs : s.regs .nat [] "start_m" = some (Tile.scalar startM)) :
    evalOp (lowerExpr BM BN offset size) s = some (Tile.scalar (rawLower startM BM BN offset size)) := by
  simp only [lowerExpr, evalOp_mul, eval_truncDiv]
  simp only [evalOp.eq_def, hs, bind, Option.bind_some]
  congr 1

theorem upper_eval (s : BlockState) (startM BM BN : Nat) (offset : Int)
    (hs : s.regs .nat [] "start_m" = some (Tile.scalar startM)) :
    evalOp (upperExpr BM BN offset) s = some (Tile.scalar (rawUpper startM BM BN offset)) := by
  simp only [upperExpr, evalOp_mul, eval_truncDiv]
  simp only [evalOp.eq_def, hs, bind, Option.bind_some]
  congr 1

theorem upper_matches (startM BM BN context window complement : Nat) (offset : Int) :
    (upper startM BM BN context window complement offset).toNat =
      RealKeys.upper startM BM BN context window complement offset := by
  by_cases h : window ≠ 0 ∧ complement = 0 <;>
    simp [upper, RealKeys.upper, WindowBounds.upperBound, rawUpper, h]

theorem clamp_lower_executes (s : BlockState) (lo : Int)
    (hlo : s.regs .int [] "lo" = some (Tile.scalar lo)) :
    ∃ s', stepStmt clampLower s = some s' ∧
      s'.regs .int [] "lo" = some (Tile.scalar (lo.toNat : Int)) ∧ Frame ["lo"] s s' := by
  have hr : stepStmt clampLower s =
      if lo < 0 then some (s.setReg "lo" .int [] (Tile.scalar 0)) else some s := by
    unfold clampLower
    conv_lhs => unfold stepStmt
    simp only [evalOp.eq_def, hlo, bind, Option.bind_some, Tile.scalar]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (show evalOp (.constInt 0) s = some (Tile.scalar 0) by simp [evalOp.eq_def])),
      stepStmts.nil]
    simp
  rw [hr]
  by_cases h : lo < 0
  · rw [if_pos h]
    exact ⟨_, rfl, by simp [Int.toNat_of_nonpos (by omega : lo ≤ 0)],
      Frame.setReg _ s "lo" _ _ _ (by simp)⟩
  · rw [if_neg h]
    exact ⟨s, rfl, by simpa [Int.toNat_of_nonneg (by omega : 0 ≤ lo)] using hlo, Frame.refl _ s⟩

theorem clamp_upper_executes (s : BlockState) (hi : Int) (context : Nat)
    (hhi : s.regs .int [] "hi" = some (Tile.scalar hi)) :
    ∃ s', stepStmt (clampUpper context) s = some s' ∧
      s'.regs .int [] "hi" = some (Tile.scalar (min hi context)) ∧ Frame ["hi"] s s' := by
  have hr : stepStmt (clampUpper context) s =
      if (context : Int) < hi then some (s.setReg "hi" .int [] (Tile.scalar context)) else some s := by
    unfold clampUpper
    conv_lhs => unfold stepStmt
    simp only [evalOp.eq_def, hhi, bind, Option.bind_some, Tile.scalar]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (show evalOp (.constInt context) s = some (Tile.scalar (context : Int)) by simp [evalOp.eq_def])), stepStmts.nil]
    simp
  rw [hr]
  by_cases h : (context : Int) < hi
  · rw [if_pos h]
    exact ⟨_, rfl, by simp [min_eq_right (by omega : (context : Int) ≤ hi)],
      Frame.setReg _ s "hi" _ _ _ (by simp)⟩
  · rw [if_neg h]
    exact ⟨s, rfl, by simpa [min_eq_left (by omega : hi ≤ (context : Int))] using hhi, Frame.refl _ s⟩

def shifted {shape : TileShape} (ptr : Tile .blockPtr shape) (deltas : List Int) : Tile .blockPtr shape :=
  ⟨fun i => (ptr.data i).advance deltas⟩

theorem advance_executes (s : BlockState) (BN BD : Nat) (lo : Int)
    (kp : Tile .blockPtr [BD, BN]) (vp : Tile .blockPtr [BN, BD])
    (hl : s.regs .int [] "lo" = some (Tile.scalar lo))
    (hk : s.regs .blockPtr [BD, BN] "K_block_ptr" = some kp)
    (hv : s.regs .blockPtr [BN, BD] "V_block_ptr" = some vp) :
    ∃ s', stepStmts (advance BN BD) s = some s' ∧
      s'.regs .int [] "lo" = some (Tile.scalar lo) ∧
      s'.regs .blockPtr [BD, BN] "K_block_ptr" = some (shifted kp [0, lo]) ∧
      s'.regs .blockPtr [BN, BD] "V_block_ptr" = some (shifted vp [lo, 0]) ∧
      Frame ["lo", "K_block_ptr", "V_block_ptr"] s s' := by
  let sl := s.setReg "lo" .int [] (Tile.scalar lo)
  let sk := sl.setReg "K_block_ptr" .blockPtr [BD, BN] (shifted kp [0, lo])
  have hc (t : BlockState) : evalOp (.constInt 0) t = some (Tile.scalar 0) := by simp [evalOp.eq_def]
  have ek : evalOp (.advanceBlockPtrDyn (.ref .blockPtr [BD, BN] "K_block_ptr")
        [.constInt 0, .ref .int [] "lo"]) sl = some (shifted kp [0, lo]) := by
    simp [advanceBlockPtrDyn_eval, hc, sl, hk, Tile.scalar, List.mapM_cons, List.mapM_nil, shifted, -Tile.scalar_eta]
  have ev : evalOp (.advanceBlockPtrDyn (.ref .blockPtr [BN, BD] "V_block_ptr")
        [.ref .int [] "lo", .constInt 0]) sk = some (shifted vp [lo, 0]) := by
    simp [advanceBlockPtrDyn_eval, hc, sk, sl, hv, Tile.scalar, List.mapM_cons, List.mapM_nil, shifted, -Tile.scalar_eta]
  rw [advance, stepStmts.cons_some (stepStmt_assign_eq_some (by simpa using hl)),
    stepStmts.cons_some (stepStmt_assign_eq_some ek), stepStmts.cons_some (stepStmt_assign_eq_some ev),
    stepStmts.nil]
  refine ⟨_, rfl, by simp [sk, sl], by simp [sk], by simp, ?_⟩
  exact ((Frame.setReg _ s "lo" _ _ _ (by simp)).trans
    (Frame.setReg _ sl "K_block_ptr" _ _ _ (by simp))).trans
      (Frame.setReg _ sk "V_block_ptr" _ _ _ (by simp))

def writes : List RegName := ["lo", "hi", "K_block_ptr", "V_block_ptr"]

theorem regular_executes (s : BlockState) (startM BM BN BD context : Nat) (offset size : Int)
    (kp : Tile .blockPtr [BD, BN]) (vp : Tile .blockPtr [BN, BD])
    (hm : s.regs .nat [] "start_m" = some (Tile.scalar startM))
    (hk : s.regs .blockPtr [BD, BN] "K_block_ptr" = some kp)
    (hv : s.regs .blockPtr [BN, BD] "V_block_ptr" = some vp) :
    let lo := (rawLower startM BM BN offset size).toNat
    let hi := min (rawUpper startM BM BN offset) context
    ∃ s', stepStmts
        ([.assign .int [] "lo" (lowerExpr BM BN offset size),
          .assign .int [] "hi" (upperExpr BM BN offset), clampLower, clampUpper context] ++ advance BN BD) s = some s' ∧
      s'.regs .int [] "lo" = some (Tile.scalar (lo : Int)) ∧
      s'.regs .int [] "hi" = some (Tile.scalar hi) ∧
      s'.regs .blockPtr [BD, BN] "K_block_ptr" = some (shifted kp [0, lo]) ∧
      s'.regs .blockPtr [BN, BD] "V_block_ptr" = some (shifted vp [lo, 0]) ∧ Frame writes s s' := by
  dsimp only
  let sl := s.setReg "lo" .int [] (Tile.scalar (rawLower startM BM BN offset size))
  let sr := sl.setReg "hi" .int [] (Tile.scalar (rawUpper startM BM BN offset))
  obtain ⟨sc, hcl, hl, fc⟩ := clamp_lower_executes sr (rawLower startM BM BN offset size) (by simp [sr, sl])
  obtain ⟨sd, hcu, hh, fd⟩ := clamp_upper_executes sc (rawUpper startM BM BN offset) context
    (by rw [fc.regs .int [] "hi" (by simp)]; simp [sr])
  have fsc : Frame writes s sc := ((Frame.setReg writes s "lo" _ _ _ (by simp [writes])).trans
    (Frame.setReg writes sl "hi" _ _ _ (by simp [writes]))).trans (fc.mono (by simp [writes]))
  have fsd := fsc.trans (fd.mono (by simp [writes]))
  obtain ⟨sf, ha, hlf, hkf, hvf, ff⟩ := advance_executes sd BN BD (rawLower startM BM BN offset size).toNat kp vp
    ((fd.regs .int [] "lo" (by simp)).trans hl)
    ((fd.regs .blockPtr [BD, BN] "K_block_ptr" (by simp)).trans
      ((fc.regs .blockPtr [BD, BN] "K_block_ptr" (by simp)).trans (by simp [sr, sl, hk])))
    ((fd.regs .blockPtr [BN, BD] "V_block_ptr" (by simp)).trans
      ((fc.regs .blockPtr [BN, BD] "V_block_ptr" (by simp)).trans (by simp [sr, sl, hv])))
  refine ⟨sf, ?_, hlf, (ff.regs .int [] "hi" (by simp)).trans hh, hkf, hvf,
    fsd.trans (ff.mono (by simp [writes]))⟩
  simp only [List.cons_append, List.nil_append]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (lower_eval s startM BM BN offset size hm)),
    stepStmts.cons_some (stepStmt_assign_eq_some (upper_eval sl startM BM BN offset (by simp [sl, hm]))),
    stepStmts.cons_some hcl, stepStmts.cons_some hcu]
  exact ha

theorem executes (s : BlockState) (startM BM BN BD context window complement : Nat) (offset size : Int)
    (kp : Tile .blockPtr [BD, BN]) (vp : Tile .blockPtr [BN, BD])
    (hm : s.regs .nat [] "start_m" = some (Tile.scalar startM))
    (hk : s.regs .blockPtr [BD, BN] "K_block_ptr" = some kp)
    (hv : s.regs .blockPtr [BN, BD] "V_block_ptr" = some vp) :
    let lo := RealKeys.lower startM BM BN window complement offset size
    ∃ s', stepStmt (code BM BN BD context window complement offset size) s = some s' ∧
      s'.regs .int [] "lo" = some (Tile.scalar (lo : Int)) ∧
      s'.regs .int [] "hi" = some (Tile.scalar (upper startM BM BN context window complement offset)) ∧
      s'.regs .blockPtr [BD, BN] "K_block_ptr" =
        some (if window ≠ 0 ∧ complement = 0 then shifted kp [0, lo] else kp) ∧
      s'.regs .blockPtr [BN, BD] "V_block_ptr" =
        some (if window ≠ 0 ∧ complement = 0 then shifted vp [lo, 0] else vp) ∧ Frame writes s s' := by
  have hout (yes no : List Stmt) :
      stepStmt (.ifThenElse (.boolAnd .nil (.ne .nat .nil (.constNat window) (.constNat 0))
        (.eq .nat .nil (.constNat complement) (.constNat 0))) yes no) s =
        if window ≠ 0 ∧ complement = 0 then stepStmts yes s else stepStmts no s := by
    conv_lhs => unfold stepStmt
    simp only [evalOp.eq_def, bind, Option.bind_some, Tile.scalar, Tile.cop, ComparableDType.ne,
      ComparableDType.eq]
    simp
  dsimp only
  rw [code, hout]
  by_cases h : window ≠ 0 ∧ complement = 0
  · rw [if_pos h]
    have hin (yes no : List Stmt) :
        stepStmt (.ifThenElse (.ne .nat .nil (.constNat complement) (.constNat 0)) yes no) s = stepStmts no s := by
      conv_lhs => unfold stepStmt
      simp [evalOp.eq_def, h.2, Tile.cop, ComparableDType.ne, Tile.scalar, -Tile.scalar_eta]
    obtain ⟨sf, hs, hl, hh, hkf, hvf, hf⟩ := regular_executes s startM BM BN BD context offset size kp vp hm hk hv
    refine ⟨sf, ?_, ?_, ?_, ?_, ?_, hf⟩
    · rw [stepStmts.cons_some ((hin _ _).trans hs), stepStmts.nil]
    · simpa [RealKeys.lower, WindowBounds.lowerBound, rawLower, h] using hl
    · simpa [upper, h] using hh
    · simpa [RealKeys.lower, WindowBounds.lowerBound, rawLower, h] using hkf
    · simpa [RealKeys.lower, WindowBounds.lowerBound, rawLower, h] using hvf
  · rw [if_neg h]
    let sl := s.setReg "lo" .int [] (Tile.scalar 0)
    rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (show evalOp (.constInt 0) s = some (Tile.scalar 0) by simp [evalOp.eq_def])),
      stepStmts.cons_some (stepStmt_assign_eq_some
        (show evalOp (.constInt context) sl = some (Tile.scalar (context : Int)) by simp [evalOp.eq_def])),
      stepStmts.nil]
    refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_⟩
    · simp [sl, RealKeys.lower, h]
    · simp [upper, h]
    · simp [sl, h, hk]
    · simp [sl, h, hv]
    · exact (Frame.setReg writes s "lo" _ _ _ (by simp [writes])).trans
        (Frame.setReg writes sl "hi" _ _ _ (by simp [writes]))

open RealLoop in
/-- Establish exactly the bounds and K/V metadata required at loop entry.
The source header is responsible for the unadvanced pointers supplied here. -/
theorem enters_loop {BM BN BD : Nat} (s : BlockState) (p : Parameters BM BN BD)
    (hlo : p.lo = RealKeys.lower p.startM BM BN p.window p.complement p.offset p.size)
    (hhi : p.hi = upper p.startM BM BN p.context p.window p.complement p.offset)
    (hm : s.regs .nat [] "start_m" = some (Tile.scalar p.startM))
    (hk : s.regs .blockPtr [BD, BN] "K_block_ptr" = some (keyPointer { p with lo := 0 } 0))
    (hv : s.regs .blockPtr [BN, BD] "V_block_ptr" = some (valuePointer { p with lo := 0 } 0)) :
    ∃ s', stepStmt (code BM BN BD p.context p.window p.complement p.offset p.size) s = some s' ∧
      s'.regs .int [] "lo" = some (Tile.scalar (p.lo : Int)) ∧
      s'.regs .int [] "hi" = some (Tile.scalar p.hi) ∧
      s'.regs .blockPtr [BD, BN] "K_block_ptr" = some (keyPointer p 0) ∧
      s'.regs .blockPtr [BN, BD] "V_block_ptr" = some (valuePointer p 0) ∧ Frame writes s s' := by
  obtain ⟨sf, hs, hl, hh, hkf, hvf, hf⟩ := executes s p.startM BM BN BD p.context p.window p.complement
    p.offset p.size (keyPointer { p with lo := 0 } 0) (valuePointer { p with lo := 0 } 0) hm hk hv
  refine ⟨sf, hs, by simpa [hlo] using hl, by simpa [hhi] using hh, ?_, ?_, hf⟩
  · rw [hkf]
    congr 1
    by_cases h : p.window ≠ 0 ∧ p.complement = 0
    · rw [if_pos h]
      apply Tile.ext
      intro i
      simp [shifted, keyPointer, cursor, hlo]
    · rw [if_neg h]
      simp [keyPointer, cursor, hlo, RealKeys.lower, h]
  · rw [hvf]
    congr 1
    by_cases h : p.window ≠ 0 ∧ p.complement = 0
    · rw [if_pos h]
      apply Tile.ext
      intro i
      simp [shifted, valuePointer, cursor, hlo]
    · rw [if_neg h]
      simp [valuePointer, cursor, hlo, RealKeys.lower, h]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealWindow
