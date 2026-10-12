import bench.tritonbench_g.attn_fwd_causal.Proofs.RealExpressions
import VeriTile.Triton.Float.Erasure

/-! Named fragments of both actual helpers after precision erasure. The source
identities below retain masks, pointer names, scales, the finite penalty and
the probability-cast assignment. These are execution helpers, not a new spec. -/
namespace VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealBody
open VeriTile.Triton
set_option maxRecDepth 16000
set_option maxHeartbeats 1500000
set_option linter.unusedSimpArgs false

def keyMask (BD BN N HA : Nat) : Op .bool [BD,BN] :=
  .boolAnd (.consL (.consR .nil))
    (.lt .nat .scalarR (.expandDim ⟨0, by simp⟩ (.ref .nat [BN] "offs_n"))
      (.sub .nat .nil (.constNat N) (.ref .nat [] "start_n")))
    (.expandDim ⟨1, by simp⟩ (.lt .nat .scalarR (.arange BD) (.constNat HA)))

def valueMask (BN BD N HA : Nat) : Op .bool [BN,BD] :=
  .boolAnd (.consR (.consL .nil))
    (.lt .nat .scalarR (.expandDim ⟨1, by simp⟩ (.ref .nat [BN] "offs_n"))
      (.sub .nat .nil (.constNat N) (.ref .nat [] "start_n")))
    (.expandDim ⟨0, by simp⟩ (.lt .nat .scalarR (.arange BD) (.constNat HA)))

def loadPrefix (keyPtr scalePtr : RegName) (BD BN N HA : Nat) : List Stmt :=
  [.assign .nat [] "start_n" (.ref .nat [] "start_n"),
   .assign .bool [BD,BN] "k_mask" (keyMask BD BN N HA),
   .assign .real [BD,BN] "k" (.load .real (.ptr (.ref .ptr [BD,BN] keyPtr))
     (.mask (.ref .bool [BD,BN] "k_mask"))),
   .assign .real [] "k_scale" (.load .real (.ptr (.ref .ptr [] scalePtr)) .none)]

def scoreExpr (BM BN BD : Nat) : Op .real [BM,BN] :=
  .mul .real .scalarR
    (.mul .real .scalarR (.dot (batch := [])
      (.ref .real [BM,BD] "q") (.ref .real [BD,BN] "k")) (.ref .real [] "q_scale"))
    (.ref .real [] "k_scale")

def score (BM BN BD : Nat) : Stmt := .assign .real [BM,BN] "qk" (scoreExpr BM BN BD)

def diagonal (BM BN : Nat) (mask : RegName) : List Stmt :=
  [.assign .bool [BM,BN] mask (RealExpressions.maskExpr BM BN),
   .assign .real [BM,BN] "qk" (RealExpressions.penaltyExpr BM BN mask)]

def centerExpr (BM BN : Nat) : Op .real [BM] :=
  .where (.gt .real (.consSame .nil) (.ref .real [BM] "m_i")
    (.reduceMax ⟨1, by simp⟩ Bool.false (.ref .real [BM,BN] "qk")))
    (.ref .real [BM] "m_i")
    (.reduceMax ⟨1, by simp⟩ Bool.false (.ref .real [BM,BN] "qk"))

def center (BM BN : Nat) : List Stmt :=
  [.assign .real [BM] "m_ij" (centerExpr BM BN),
   .assign .real [BM,BN] "qk" (.sub .real (.consSame (.consR .nil))
     (.ref .real [BM,BN] "qk") (.expandDim ⟨1, by simp⟩ (.ref .real [BM] "m_ij")))]

def probability (BM BN : Nat) : Stmt :=
  .assign .real [BM,BN] "p" (.exp2 (.ref .real [BM,BN] "qk"))

def rescale (BM BN BD : Nat) : List Stmt :=
  [.assign .real [BM] "l_ij" (.reduceSum ⟨1, by simp⟩ Bool.false (.ref .real [BM,BN] "p")),
   .assign .real [BM] "alpha" (.exp2 (.sub .real (.consSame .nil)
     (.ref .real [BM] "m_i") (.ref .real [BM] "m_ij"))),
   .assign .real [BM] "l_i" (.add .real (.consSame .nil)
     (.mul .real (.consSame .nil) (.ref .real [BM] "l_i") (.ref .real [BM] "alpha"))
     (.ref .real [BM] "l_ij")),
   .assign .real [BM,BD] "acc" (.mul .real (.consSame (.consR .nil))
     (.ref .real [BM,BD] "acc") (.expandDim ⟨1, by simp⟩ (.ref .real [BM] "alpha")))]

def loadValue (ptr : RegName) (BN BD N HA : Nat) : Stmt :=
  .assign .real [BN,BD] "v" (.load .real (.ptr (.ref .ptr [BN,BD] ptr))
    (.mask (valueMask BN BD N HA)))

def accumulate (BM BN BD : Nat) : List Stmt :=
  [.assign .real [BM,BN] "p" (.ref .real [BM,BN] "p"),
   .assign .real [BM,BD] "acc" (.add .real (.consSame (.consSame .nil))
     (.ref .real [BM,BD] "acc") (.dot (options := {outDType := some .fp16}) (batch := [])
       (.ref .real [BM,BN] "p") (.ref .real [BN,BD] "v"))),
   .assign .real [BM] "m_i" (.ref .real [BM] "m_ij")]

def advance (keyPtr scalePtr valuePtr : RegName) (BN BD HD : Nat) : List Stmt :=
  [.assign .ptr [BD,BN] keyPtr (.ptrAdd .scalarR (.ref .ptr [BD,BN] keyPtr)
      (.mul .nat .nil (.constNat BN) (.constNat HD))),
   .assign .ptr [] scalePtr (.ptrAdd .nil (.ref .ptr [] scalePtr) (.constNat 1)),
   .assign .ptr [BN,BD] valuePtr (.ptrAdd .scalarR (.ref .ptr [BN,BD] valuePtr)
      (.mul .nat .nil (.constNat BN) (.constNat HD)))]

def suffix (keyPtr scalePtr valuePtr : RegName) (BM BN BD N HA HD : Nat) : List Stmt :=
  [probability BM BN] ++ rescale BM BN BD ++ [loadValue valuePtr BN BD N HA] ++
    accumulate BM BN BD ++ advance keyPtr scalePtr valuePtr BN BD HD

def first (BM BN BD N HA HD : Nat) (diagonalFirst : Bool) : List Stmt :=
  loadPrefix "K_ptrs_first" "K_scale_ptr_first" BD BN N HA ++ [score BM BN BD,
    .ifThenElse (.constBool diagonalFirst)
      (diagonal BM BN "mask_first" ++ center BM BN) (center BM BN)] ++
    suffix "K_ptrs_first" "K_scale_ptr_first" "V_ptrs_first" BM BN BD N HA HD

def second (BM BN BD N HA HD : Nat) : List Stmt :=
  loadPrefix "K_ptrs_second" "K_scale_ptr_second" BD BN N HA ++ [score BM BN BD] ++
    diagonal BM BN "mask" ++ center BM BN ++
    suffix "K_ptrs_second" "K_scale_ptr_second" "V_ptrs_second" BM BN BD N HA HD

/-- Locate a helper body by loop name and occurrence, preserving the source
order. Failure stays none rather than producing an assumed loop body. -/
def sourceLoop? (index : RegName) : List Stmt → Nat → Option (List Stmt)
  | [], _ => none
  | .forRangeDyn name _ _ _ loop :: rest, n =>
      if name = index then match n with
        | 0 => some loop
        | n+1 => sourceLoop? index rest n
      else sourceLoop? index rest n
  | _ :: rest, n => sourceLoop? index rest n

section
variable (Q K V QS KS Out : RegionName)
    (sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      Z H N HD BM BN BD HA stage : Nat) (hs : stage = 2 ∨ stage = 3)

theorem first_source (candidate : Bool) :
    sourceLoop? "start_n" ((if candidate then TypedSource.reciprocal else TypedSource.original)
      Q K V QS KS Out sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      Z H N HD BM BN BD HA stage hs).eraseDType.toAlgKernel.body 0 =
      some (first BM BN BD N HA HD (decide (stage=2))) := by
  cases candidate
  all_goals
    simp only [TypedSource.original, TypedSource.reciprocal, Bool.false_eq_true, Bool.true_eq,
      ↓reduceIte, ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?,
      ComputeKernel.surfaceBody, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
      ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
      Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
      VeriTile.Triton.eraseDType, FloatDType.toTileDType, NumericDType.eraseDType,
      ComparableDType.eraseDType, IntegralDType.eraseDType, Except.bind, bind]
    simp [sourceLoop?, first, suffix, loadPrefix, keyMask, valueMask, score, scoreExpr,
      diagonal, center, centerExpr, probability, rescale, loadValue, accumulate, advance,
      RealExpressions.maskExpr, RealExpressions.penaltyExpr]

theorem second_source (candidate : Bool) :
    sourceLoop? "start_n" ((if candidate then TypedSource.reciprocal else TypedSource.original)
      Q K V QS KS Out sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      Z H N HD BM BN BD HA stage hs).eraseDType.toAlgKernel.body 1 =
      some (second BM BN BD N HA HD) := by
  cases candidate
  all_goals
    simp only [TypedSource.original, TypedSource.reciprocal, Bool.false_eq_true, Bool.true_eq,
      ↓reduceIte, ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?,
      ComputeKernel.surfaceBody, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
      ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
      Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
      VeriTile.Triton.eraseDType, FloatDType.toTileDType, NumericDType.eraseDType,
      ComparableDType.eraseDType, IntegralDType.eraseDType, Except.bind, bind]
    simp [sourceLoop?, second, suffix, loadPrefix, keyMask, valueMask, score, scoreExpr,
      diagonal, center, centerExpr, probability, rescale, loadValue, accumulate, advance,
      RealExpressions.maskExpr, RealExpressions.penaltyExpr]

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealBody
