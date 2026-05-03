/-
VeriTile.Triton.DSL.Typing

Macro-time dtype, shape, and environment helpers for Triton DSL expansion.
-/

import VeriTile.Triton.Core
import VeriTile.Triton.DSL.Syntax

open Lean

namespace VeriTile.Triton.DSL

inductive DInfo where
  | real
  | fp32
  | fp16
  | bf16
  | int
  | nat
  | bool
  | ptr
  | blockPtr
  deriving BEq, Inhabited

inductive SInfo where
  | dims : List (TSyntax `term) → SInfo
  deriving Inhabited

abbrev Env := List (String × DInfo × SInfo)

def termKey (t : TSyntax `term) : String :=
  toString t.raw

namespace SInfo

def scalar : SInfo := .dims []

def vec (n : TSyntax `term) : SInfo := .dims [n]

end SInfo

def termListEq : List (TSyntax `term) → List (TSyntax `term) → Bool
  | [], [] => Bool.true
  | a :: as, b :: bs => (termKey a == termKey b) && termListEq as bs
  | _, _ => Bool.false

def SInfo.eq : SInfo → SInfo → Bool
  | SInfo.dims a, SInfo.dims b => termListEq a b

def eraseNth {α : Type} : List α → Nat → MacroM (List α)
  | [], _ => Macro.throwError "internal error: eraseNth index out of bounds"
  | _ :: xs, 0 => pure xs
  | x :: xs, n + 1 => return x :: (← eraseNth xs n)

def setNthOne : List (TSyntax `term) → Nat → MacroM (List (TSyntax `term))
  | [], _ => Macro.throwError "internal error: setNthOne index out of bounds"
  | _ :: xs, 0 => do
      let oneLit : TSyntax `term ← `((1 : Nat))
      pure (oneLit :: xs)
  | x :: xs, n + 1 => return x :: (← setNthOne xs n)

def DInfo.term : DInfo → MacroM (TSyntax `term)
  | .real => `(TileDType.real)
  | .fp32 => `(TileDType.fp32)
  | .fp16 => `(TileDType.fp16)
  | .bf16 => `(TileDType.bf16)
  | .int => `(TileDType.int)
  | .nat => `(TileDType.nat)
  | .bool => `(TileDType.bool)
  | .ptr => `(TileDType.ptr)
  | .blockPtr => `(TileDType.blockPtr)

def DInfo.floatProof : DInfo → MacroM (TSyntax `term)
  | .real => `(FloatDType.real)
  | .fp32 => `(FloatDType.fp32)
  | .fp16 => `(FloatDType.fp16)
  | .bf16 => `(FloatDType.bf16)
  | .int => Macro.throwError "tl.cast: signed integer casts are not modeled yet"
  | .nat => Macro.throwError "tl.cast: Nat casts are not modeled yet; use tl.toReal for Nat to real"
  | .bool => Macro.throwError "tl.cast: Bool casts are not supported"
  | .ptr => Macro.throwError "tl.cast: pointer casts are not supported"
  | .blockPtr => Macro.throwError "tl.cast: block pointer casts are not supported"

def SInfo.term : SInfo → MacroM (TSyntax `term)
  | SInfo.dims ds => do
      let rec go : List (TSyntax `term) → MacroM (TSyntax `term)
        | [] => `(([] : TileShape))
        | d :: rest => do
            let tail ← go rest
            `($d :: $tail)
      go ds

def DInfo.numericProof : DInfo → MacroM (TSyntax `term)
  | .real => `(NumericDType.real)
  | .fp32 => `(NumericDType.fp32)
  | .fp16 => `(NumericDType.fp16)
  | .bf16 => `(NumericDType.bf16)
  | .int => `(NumericDType.int)
  | .nat => `(NumericDType.nat)
  | .bool => Macro.throwError "arithmetic on Bool values is not supported"
  | .ptr => Macro.throwError "arithmetic on pointer values is not supported; use pointer + Nat offset"
  | .blockPtr => Macro.throwError "arithmetic on block pointers is not supported; use tl.advance"

def DInfo.integralProof : DInfo → MacroM (TSyntax `term)
  | .int => `(IntegralDType.int)
  | .nat => `(IntegralDType.nat)
  | .real => Macro.throwError "integer division/remainder on Real values is not supported"
  | .fp32 => Macro.throwError "integer division/remainder on floating values is not supported"
  | .fp16 => Macro.throwError "integer division/remainder on floating values is not supported"
  | .bf16 => Macro.throwError "integer division/remainder on floating values is not supported"
  | .bool => Macro.throwError "integer division/remainder on Bool values is not supported"
  | .ptr => Macro.throwError "integer division/remainder on pointer values is not supported"
  | .blockPtr => Macro.throwError "integer division/remainder on block pointers is not supported"

def DInfo.comparableProof : DInfo → MacroM (TSyntax `term)
  | .real => `(ComparableDType.real)
  | .fp32 => `(ComparableDType.fp32)
  | .fp16 => `(ComparableDType.fp16)
  | .bf16 => `(ComparableDType.bf16)
  | .int => `(ComparableDType.int)
  | .nat => `(ComparableDType.nat)
  | .bool => Macro.throwError "comparison on Bool values is not supported"
  | .ptr => Macro.throwError "comparison on pointer values is not supported"
  | .blockPtr => Macro.throwError "comparison on block pointers is not supported"

def expandDType : TSyntax `tritonDType → MacroM DInfo
  | `(tritonDType| tl.float64) => pure .real
  | `(tritonDType| tl.float32) => pure .fp32
  | `(tritonDType| tl.float16) => pure .fp16
  | `(tritonDType| tl.bfloat16) => pure .bf16
  | `(tritonDType| tl.int32) => pure .int
  | `(tritonDType| tl.int64) => pure .int
  | `(tritonDType| tl.uint32) => pure .nat
  | `(tritonDType| tl.uint64) => pure .nat
  | _ => Macro.throwUnsupported

def lookupEnv (env : Env) (name : String) : MacroM (DInfo × SInfo) := do
  match env.find? (fun entry => entry.1 == name) with
  | some (_, dtype, shape) => pure (dtype, shape)
  | none => Macro.throwError ("unknown Triton identifier `" ++ name ++ "`")

def ensureDType (expected actual : DInfo) (ctx : String) : MacroM Unit := do
  unless expected == actual do
    Macro.throwError (ctx ++ ": dtype mismatch")

def ensureShape (expected actual : SInfo) (ctx : String) : MacroM Unit := do
  unless expected.eq actual do
    Macro.throwError (ctx ++ ": shape mismatch")

/-- Recognize a dim term as the literal `1`, tolerating common syntactic
wrappings (`1`, `(1)`, `(1 : Nat)`, `((1 : Nat))`, ...) so that unit-dim
broadcasts fire regardless of how the dim was emitted. -/
partial def termIsOne (t : TSyntax `term) : Bool :=
  match t with
  | `($n:num) => n.getNat == 1
  | `(($e:term)) => termIsOne e
  | `(($e:term : $_:term)) => termIsOne e
  | _ => Bool.false

def broadcastTerm (a b : SInfo) (ctx : String) :
    MacroM (TSyntax `term × SInfo) := do
  match a, b with
  | SInfo.dims [], SInfo.dims [] => pure (← `(Broadcast.nil), SInfo.dims [])
  | SInfo.dims [], SInfo.dims (_ :: _) => pure (← `(Broadcast.scalarL), b)
  | SInfo.dims (_ :: _), SInfo.dims [] => pure (← `(Broadcast.scalarR), a)
  | SInfo.dims (ad :: ads), SInfo.dims (bd :: bds) =>
      let (subBc, subShape) ← broadcastTerm (SInfo.dims ads) (SInfo.dims bds) ctx
      if termKey ad == termKey bd then
        match subShape with
        | SInfo.dims outRest => pure (← `(Broadcast.consSame $subBc), SInfo.dims (ad :: outRest))
      else if termIsOne ad then
        match subShape with
        | SInfo.dims outRest => pure (← `(Broadcast.consL $subBc), SInfo.dims (bd :: outRest))
      else if termIsOne bd then
        match subShape with
        | SInfo.dims outRest => pure (← `(Broadcast.consR $subBc), SInfo.dims (ad :: outRest))
      else
        Macro.throwError (ctx ++ ": incompatible shapes")

def coerceShape (e : TSyntax `term) (src target : SInfo) (ctx : String) :
    MacroM (TSyntax `term) := do
  if src.eq target then
    pure e
  else
    match src with
    | SInfo.dims [] =>
        let st ← target.term
        `(Op.broadcast $e $st)
    | _ => Macro.throwError (ctx ++ ": cannot broadcast non-scalar to target shape")

end VeriTile.Triton.DSL
