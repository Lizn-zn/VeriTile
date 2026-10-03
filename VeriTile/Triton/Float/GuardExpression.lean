/- Numerical-domain expressions are syntax, not arbitrary semantic predicates.
Every floating call keeps its precision and dtype. Conditions can inspect
values, combine checks and enumerate indices; they cannot assume an equality. -/
import VeriTile.Triton.Float.GuardedIO

namespace VeriTile.Triton.FP.GuardExpression
open Structural

inductive Expr (Input : Type) where
  | input (name : Input)
  | literal (precision : Option ComputeDType) (dtype : FloatDType) (value : ℝ)
  | negInf
  | binary (precision : Option ComputeDType) (dtype : FloatDType) (op : Binary)
      (left right : Expr Input)
  | unary (precision : Option ComputeDType) (op : Unary) (value : Expr Input)
  | cast (precision : Option ComputeDType) (source target : FloatDType) (value : Expr Input)
  | fromNat (precision : Option ComputeDType) (value : Nat)
  | fromInt (precision : Option ComputeDType) (value : Int)
  | fp32Bits (bits : Float32Bits)
  | fp32Load (value : Expr Input)
  | reduceMax (precision : Option ComputeDType) (shape : TileShape)
      (axis : Fin shape.length) (keepDims : Bool) (values : TileIndex shape → Expr Input)
      (outputIndex : TileIndex (TileShape.reduceShape shape axis keepDims))
  | reduceSum (precision : Option ComputeDType) (shape : TileShape)
      (axis : Fin shape.length) (keepDims : Bool) (values : TileIndex shape → Expr Input)
      (outputIndex : TileIndex (TileShape.reduceShape shape axis keepDims))

def Expr.eval {Input α : Type} (M : Algebra α) (read : Input → α) : Expr Input → α
  | .input name => read name
  | .literal p d r => M.literal p d r
  | .negInf => M.negInf
  | .binary p d op a b => M.binary p d op (a.eval M read) (b.eval M read)
  | .unary p op a => M.unary p op (a.eval M read)
  | .cast p src dst a => M.cast p src dst (a.eval M read)
  | .fromNat p n => M.fromNat p n
  | .fromInt p n => M.fromInt p n
  | .fp32Bits b => M.fp32Bits b
  | .fp32Load a => M.fp32Load (a.eval M read)
  | .reduceMax p _ axis keepDims xs i => M.reduceMax p axis keepDims (fun j => (xs j).eval M read) i
  | .reduceSum p _ axis keepDims xs i => M.reduceSum p axis keepDims (fun j => (xs j).eval M read) i

/-- Reify numerical expressions without interpreting any floating law. -/
def algebra (Input : Type) : Algebra (Expr Input) where
  literal := .literal
  negInf := .negInf
  binary := .binary
  unary := .unary
  cast := .cast
  fromNat := .fromNat
  fromInt := .fromInt
  fp32Bits := .fp32Bits
  fp32Load := .fp32Load
  reduceMax := fun p {shape} => .reduceMax p shape
  reduceSum := fun p {shape} => .reduceSum p shape

structure MemoryInput where
  region : RegionName
  offset : Nat → Nat
  dtype : FloatDType := .real

def MemoryInput.read {α : Type} [Inhabited α] (s : State α) (i : MemoryInput) : α :=
  toFloat i.dtype ((s.mem i.region (i.offset (s.pids 0))).read i.dtype.toTileDType)

/-- A finite collection of domain checks. There is no equation constructor. -/
inductive Requirements (Value : Type) where
  | guard (kind : GuardKind) (value : Value)
  | top
  | both (left right : Requirements Value)
  | each (n : Nat) (checks : Fin n → Requirements Value)

def Requirements.all {Value : Type} (checks : List (Requirements Value)) : Requirements Value :=
  checks.foldr .both .top

def Requirements.Holds {Value : Type} (D : GuardKind → Value → Prop) : Requirements Value → Prop
  | .guard kind value => D kind value
  | .top => True
  | .both a b => a.Holds D ∧ b.Holds D
  | .each _ checks => ∀ i, (checks i).Holds D

@[simp] theorem Requirements.holds_guard {Value : Type} (D : GuardKind → Value → Prop)
    (kind : GuardKind) (value : Value) :
    (.guard kind value : Requirements Value).Holds D ↔ D kind value := Iff.rfl

@[simp] theorem Requirements.holds_top {Value : Type} (D : GuardKind → Value → Prop) :
    (.top : Requirements Value).Holds D ↔ True := Iff.rfl

@[simp] theorem Requirements.holds_both {Value : Type} (D : GuardKind → Value → Prop)
    (a b : Requirements Value) : (Requirements.both a b).Holds D ↔ a.Holds D ∧ b.Holds D := Iff.rfl

@[simp] theorem Requirements.holds_all {Value : Type} (D : GuardKind → Value → Prop)
    (checks : List (Requirements Value)) :
    (Requirements.all checks).Holds D ↔ ∀ c ∈ checks, c.Holds D := by
  induction checks with
  | nil => simp [all]
  | cons c cs ih => simp [all, ← ih, List.foldr]

@[simp] theorem Requirements.holds_each {Value : Type} (D : GuardKind → Value → Prop)
    (n : Nat) (checks : Fin n → Requirements Value) :
    (.each n checks : Requirements Value).Holds D ↔ ∀ i, (checks i).Holds D := Iff.rfl

def Requirements.map {Value Other : Type} (f : Value → Other) :
    Requirements Value → Requirements Other
  | .guard k v => .guard k (f v)
  | .top => .top
  | .both a b => .both (a.map f) (b.map f)
  | .each n cs => .each n (fun i => (cs i).map f)

@[simp] theorem Requirements.map_all {Value Other : Type} (f : Value → Other)
    (checks : List (Requirements Value)) :
    (Requirements.all checks).map f = Requirements.all (checks.map (Requirements.map f)) := by
  induction checks with
  | nil => rfl
  | cons c cs ih =>
      change Requirements.both (c.map f) ((all cs).map f) =
        Requirements.both (c.map f) (all (cs.map (map f)))
      rw [ih]

/-- Interpreting reified requirements preserves exactly their domain checks. -/
theorem Requirements.holds_map {Value Other : Type} (f : Value → Other)
    (D : GuardKind → Other → Prop) (c : Requirements Value) :
    (c.map f).Holds D ↔ c.Holds (fun k v => D k (f v)) := by
  induction c with
  | guard => rfl
  | top => rfl
  | both a b ha hb => simp only [map, holds_both, ha, hb]
  | each n cs ih => simp only [map, holds_each, ih]

abbrev Condition := Requirements (Expr MemoryInput)

def Condition.Holds {α : Type} [Inhabited α] (M : Algebra α) (D : Guarded.Domain α)
    (s : State α) (c : Condition) : Prop :=
  Requirements.Holds (fun k e => D k (e.eval M (MemoryInput.read s))) c

end VeriTile.Triton.FP.GuardExpression
