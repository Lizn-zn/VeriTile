import bench.tritonbench_g.attn_fwd_causal.Proofs.RealLoopFormula

/-! The diagonal helper's six setup statements. The first helper advances
its local copies; the second resets from the caller's preserved pointer bases.
No state left in the first helper's advanced pointers is used by the second. -/
namespace VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealSetup
open VeriTile.Triton RealLoads RealMemory RealIteration
open AttentionFwdTriton3.RealScores (Frame)
noncomputable section
set_option maxHeartbeats 800000
set_option linter.unusedSimpArgs false

def lowerName (second : Bool) : RegName := if second then "second_lo" else "first_lo"
def upperName (second : Bool) : RegName := if second then "second_hi" else "first_hi"
def keyBase (second : Bool) : RegName := if second then "K_ptrs" else "K_ptrs_first"
def scaleBase (second : Bool) : RegName := if second then "K_scale_ptr" else "K_scale_ptr_first"
def valueBase (second : Bool) : RegName := if second then "V_ptrs" else "V_ptrs_first"

def lowerExpr (BM : Nat) : Op .nat [] :=
  .mul .nat .nil (.ref .nat [] "start_m") (.constNat BM)
def upperExpr (BM : Nat) : Op .nat [] :=
  .mul .nat .nil (.add .nat .nil (.ref .nat [] "start_m") (.constNat 1)) (.constNat BM)

def code (BM BN BD HD : Nat) (second : Bool) : List Stmt :=
  [.assign .nat [] (lowerName second) (lowerExpr BM),
   .assign .nat [] (upperName second) (upperExpr BM),
   .assign .nat [] (lowerName second) (.ref .nat [] (lowerName second)),
   .assign .ptr [] (scaleName second) (.ptrAdd .nil (.ref .ptr [] (scaleBase second))
     (.floorDiv .nat .nil (.ref .nat [] (lowerName second)) (.constNat BN))),
   .assign .ptr [BD,BN] (keyName second) (.ptrAdd .scalarR (.ref .ptr [BD,BN] (keyBase second))
     (.mul .nat .nil (.constNat HD) (.ref .nat [] (lowerName second)))),
   .assign .ptr [BN,BD] (valueName second) (.ptrAdd .scalarR (.ref .ptr [BN,BD] (valueBase second))
     (.mul .nat .nil (.constNat HD) (.ref .nat [] (lowerName second))))]

def writes (second : Bool) : List RegName :=
  [lowerName second,upperName second,scaleName second,keyName second,valueName second]

theorem key_shift (s : BlockState) (c : Config) (BN BD start : Nat) (ptr low : RegName)
    (hp : s.regs .ptr [BD,BN] ptr = some (keyPointer c BD BN 0))
    (hl : s.regs .nat [] low = some (Tile.scalar start)) :
    evalOp (.ptrAdd .scalarR (.ref .ptr [BD,BN] ptr)
      (.mul .nat .nil (.constNat c.headDim) (.ref .nat [] low))) s = some (keyPointer c BD BN start) := by
  simp only [evalOp.eq_def, hp, hl, bind, Option.bind]
  congr 1

theorem value_shift (s : BlockState) (c : Config) (BN BD start : Nat) (ptr low : RegName)
    (hp : s.regs .ptr [BN,BD] ptr = some (valuePointer c BN BD 0))
    (hl : s.regs .nat [] low = some (Tile.scalar start)) :
    evalOp (.ptrAdd .scalarR (.ref .ptr [BN,BD] ptr)
      (.mul .nat .nil (.constNat c.headDim) (.ref .nat [] low))) s = some (valuePointer c BN BD start) := by
  simp only [evalOp.eq_def, hp, hl, bind, Option.bind]
  congr 1

theorem scale_shift (s : BlockState) (c : Config) (BN start : Nat) (ptr low : RegName)
    (hp : s.regs .ptr [] ptr = some (keyScalePointer c BN 0))
    (hl : s.regs .nat [] low = some (Tile.scalar start)) :
    evalOp (.ptrAdd .nil (.ref .ptr [] ptr)
      (.floorDiv .nat .nil (.ref .nat [] low) (.constNat BN))) s = some (keyScalePointer c BN start) := by
  simp only [evalOp.eq_def, hp, hl, bind, Option.bind]
  congr 1
  apply Tile.ext
  rintro ⟨⟩
  simp [keyScalePointer, Tile.scalar, IntegralDType.floorDiv, -Tile.scalar_eta]

theorem executes (s : BlockState) (c : Config) (BM BN BD : Nat) (second : Bool)
    (hm : s.regs .nat [] "start_m" = some (Tile.scalar c.startM))
    (hk : s.regs .ptr [BD,BN] (keyBase second) = some (keyPointer c BD BN 0))
    (hs : s.regs .ptr [] (scaleBase second) = some (keyScalePointer c BN 0))
    (hv : s.regs .ptr [BN,BD] (valueBase second) = some (valuePointer c BN BD 0)) :
    ∃ t, stepStmts (code BM BN BD c.headDim second) s = some t ∧
      t.regs .nat [] (lowerName second) = some (Tile.scalar (c.startM*BM)) ∧
      t.regs .nat [] (upperName second) = some (Tile.scalar ((c.startM+1)*BM)) ∧
      Pointers t c BN BD (c.startM*BM) second ∧ Frame (writes second) s t := by
  let sl := s.setReg (lowerName second) .nat [] (Tile.scalar (c.startM*BM))
  let sh := sl.setReg (upperName second) .nat [] (Tile.scalar ((c.startM+1)*BM))
  let sl' := sh.setReg (lowerName second) .nat [] (Tile.scalar (c.startM*BM))
  let ss := sl'.setReg (scaleName second) .ptr [] (keyScalePointer c BN (c.startM*BM))
  let sk := ss.setReg (keyName second) .ptr [BD,BN] (keyPointer c BD BN (c.startM*BM))
  have hlow : evalOp (lowerExpr BM) s = some (Tile.scalar (c.startM*BM)) := by
    simp [lowerExpr, evalOp.eq_def, hm]; rfl
  have hhigh : evalOp (upperExpr BM) sl = some (Tile.scalar ((c.startM+1)*BM)) := by
    cases second <;> simp [upperExpr, evalOp.eq_def, sl, lowerName, hm] <;> rfl
  have hcopy : evalOp (.ref .nat [] (lowerName second)) sh = some (Tile.scalar (c.startM*BM)) := by
    cases second <;> simp [sh, sl, lowerName, upperName]
  have hscale := scale_shift sl' c BN (c.startM*BM) (scaleBase second) (lowerName second)
    (by cases second <;> simpa [sl', sh, sl, scaleBase, lowerName, upperName] using hs)
    (by simp [sl'])
  have hkey := key_shift ss c BN BD (c.startM*BM) (keyBase second) (lowerName second)
    (by cases second <;> simpa [ss, sl', sh, sl, keyBase, scaleName, lowerName, upperName] using hk)
    (by cases second <;> simp [ss, sl', scaleName, lowerName])
  have hvalue := value_shift sk c BN BD (c.startM*BM) (valueBase second) (lowerName second)
    (by cases second <;> simpa [sk, ss, sl', sh, sl, valueBase, keyName, scaleName, lowerName, upperName] using hv)
    (by cases second <;> simp [sk, ss, sl', keyName, scaleName, lowerName])
  simp only [code]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some hlow),
    stepStmts.cons_some (stepStmt_assign_eq_some hhigh),
    stepStmts.cons_some (stepStmt_assign_eq_some hcopy),
    stepStmts.cons_some (stepStmt_assign_eq_some hscale),
    stepStmts.cons_some (stepStmt_assign_eq_some hkey),
    stepStmts.cons_some (stepStmt_assign_eq_some hvalue), stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ⟨?_, ?_, ?_⟩, rfl, rfl, rfl, ?_⟩
  · cases second <;> simp [sk, ss, sl', valueName, keyName, scaleName, lowerName]
  · cases second <;> simp [sk, ss, sl', sh, valueName, keyName, scaleName, lowerName, upperName]
  · cases second <;> simp [sk, valueName, keyName]
  · cases second <;> simp [sk, ss, valueName, keyName, scaleName]
  · simp
  · intro dt shape name hn
    simp only [writes, List.mem_cons, not_or] at hn
    simp [sk, ss, sl', sh, sl, hn.1, hn.2.1, hn.2.2.1, hn.2.2.2.1, hn.2.2.2.2.1]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealSetup
