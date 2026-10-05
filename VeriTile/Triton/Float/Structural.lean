/-
Structural execution for FP proofs. Floating values inhabit an arbitrary type;
numerical primitives have no algebraic laws. This is separate from the Real
interpreter and does not assert IEEE equality. In particular, reassociation,
commutation and cast idempotence cannot be obtained from this evaluator.

The interpreter preserves typed cells, register shadowing, masks and compute
precision. A store copies a value already typed by its expression; casts remain
explicit numerical operations. Unsupported syntax fails. A structural proof
must establish successful executions, not equality of two failures.
-/
import VeriTile.Triton.Core.Ast

namespace VeriTile.Triton.FP.Structural

/-- Every floating channel holds an opaque value; address/control channels keep
exact discrete carriers. No Real arithmetic is available on floating values. -/
abbrev Value (α : Type) : TileDType → Type
  | .real | .fp32 | .fp16 | .bf16 | .f8e4 | .f8e5 => α
  | .nat => Nat
  | .int => Int
  | .bool => Bool
  | .ptr => RegionName × Nat
  | .blockPtr => BlockPtr

abbrev Values (α : Type) (dtype : TileDType) (shape : TileShape) :=
  TileIndex shape → Value α dtype

def defaultValue {α : Type} [Inhabited α] : (dtype : TileDType) → Value α dtype
  | .real | .fp32 | .fp16 | .bf16 | .f8e4 | .f8e5 => default
  | .nat => 0
  | .int => 0
  | .bool => Bool.false
  | .ptr => ("", 0)
  | .blockPtr => ⟨"", 0, [], [], [], []⟩

def ofFloat {α : Type} : (d : FloatDType) → α → Value α d.toTileDType
  | .real | .fp32 | .fp16 | .bf16 | .f8e4 | .f8e5 => id

def toFloat {α : Type} : (d : FloatDType) → Value α d.toTileDType → α
  | .real | .fp32 | .fp16 | .bf16 | .f8e4 | .f8e5 => id

inductive Binary where
  | add | sub | mul | div | max | pow
  deriving DecidableEq, Repr

inductive Unary where
  | exp | libdeviceExp | libdeviceLog | libdeviceExpm1 | libdeviceLog1p
  | exp2 | log | log2 | sigmoid | sqrt | rsqrt | tanh
  | sin | cos | tan | atan | cosh | sinh | erf
  deriving DecidableEq, Repr

/-- Numerical operations are arbitrary functions. `none` records an ordinary
algorithm-typed operation; `some dtype` preserves a ComputeOp precision tag.
Even `.real` operations here are opaque, never mathematical ring operations. -/
structure Algebra (α : Type) where
  literal : Option ComputeDType → FloatDType → ℝ → α
  negInf : α
  binary : Option ComputeDType → FloatDType → Binary → α → α → α
  unary : Option ComputeDType → Unary → α → α
  /-- Floating comparisons stay opaque and retain precision/dtype. Missing
  support still fails; no order on the floating carrier is assumed. -/
  compareLt : Option ComputeDType → FloatDType → Option (α → α → Bool) := fun _ _ => none
  compareLe : Option ComputeDType → FloatDType → Option (α → α → Bool) := fun _ _ => none
  cast : Option ComputeDType → FloatDType → FloatDType → α → α
  fromNat : Option ComputeDType → Nat → α
  fromInt : Option ComputeDType → Int → α
  fp32Bits : Float32Bits → α
  /-- Compute loads retain their explicit load precision. -/
  fp32Load : α → α
  /-- Preserve the complete reduction operation and layout. No max identity,
  permutation invariance, scalar fold law or reduction schedule is assumed. -/
  reduceMax : Option ComputeDType → {shape : TileShape} →
    (axis : Fin shape.length) → (keepDims : Bool) → Values α .real shape →
    Values α .real (TileShape.reduceShape shape axis keepDims)
  /-- A structural proof cannot change this opaque reduction's input order.
  The equational interpreter separately expands an explicit addition tree. -/
  reduceSum : Option ComputeDType → {shape : TileShape} →
    (axis : Fin shape.length) → (keepDims : Bool) → Values α .real shape →
    Values α .real (TileShape.reduceShape shape axis keepDims)

inductive Cell (α : Type) where
  | mk (dtype : TileDType) (value : Value α dtype)

def Cell.read {α : Type} [Inhabited α] (dtype : TileDType) : Cell α → Value α dtype
  | .mk d v => if h : d = dtype then h ▸ v else defaultValue dtype

@[simp] theorem Cell.read_same {α : Type} [Inhabited α] (d : TileDType) (v : Value α d) :
    (Cell.mk d v).read d = v := by simp [Cell.read]

structure State (α : Type) where
  mem : RegionName → Nat → Cell α
  regs : (dtype : TileDType) → (shape : TileShape) → RegName → Option (Values α dtype shape)
  pids : Nat → Nat
  numPids : Nat → Nat
  undef : (dtype : TileDType) → RegionName → Nat → Value α dtype

namespace State

def setReg {α : Type} (s : State α) (name : RegName)
    (dtype : TileDType) (shape : TileShape) (v : Values α dtype shape) : State α :=
  { s with regs := fun d sh n =>
      if n = name then
        if hd : dtype = d then
          if hs : shape = sh then some (hd ▸ hs ▸ v) else none
        else none
      else s.regs d sh n }

@[simp] theorem setReg_same {α : Type} (s : State α) (n : RegName)
    (d : TileDType) (sh : TileShape) (v : Values α d sh) :
    (s.setReg n d sh v).regs d sh n = some v := by simp [setReg]

@[simp] theorem setReg_ne_name {α : Type} (s : State α) (n n' : RegName)
    (d d' : TileDType) (sh sh' : TileShape) (v : Values α d sh) (h : n' ≠ n) :
    (s.setReg n d sh v).regs d' sh' n' = s.regs d' sh' n' := by simp [setReg, h]

@[simp] theorem setReg_mem {α : Type} (s : State α) (n : RegName)
    (d : TileDType) (sh : TileShape) (v : Values α d sh) :
    (s.setReg n d sh v).mem = s.mem := rfl

@[simp] theorem setReg_pids {α : Type} (s : State α) (n : RegName)
    (d : TileDType) (sh : TileShape) (v : Values α d sh) :
    (s.setReg n d sh v).pids = s.pids := rfl

@[simp] theorem setReg_undef {α : Type} (s : State α) (n : RegName)
    (d : TileDType) (sh : TileShape) (v : Values α d sh) :
    (s.setReg n d sh v).undef = s.undef := rfl

def write {α : Type} (s : State α) (r : RegionName) (o : Nat) (v : Cell α) : State α :=
  { s with mem := fun r' o' => if r' = r ∧ o' = o then v else s.mem r' o' }

@[simp] theorem write_same {α : Type} (s : State α) (r : RegionName) (o : Nat) (v : Cell α) :
    (s.write r o v).mem r o = v := by simp [write]

@[simp] theorem write_other {α : Type} (s : State α) (r r' : RegionName) (o o' : Nat)
    (v : Cell α) (h : r' ≠ r ∨ o' ≠ o) :
    (s.write r o v).mem r' o' = s.mem r' o' := by
  simp only [write]
  rw [if_neg (by tauto)]


/-- Cell framing does not discard dtype tags or offsets. -/
theorem scatter_frame {α ι : Type} (r : RegionName) (off : ι → Nat)
    (val : ι → Cell α) (l : List ι) (r' : RegionName) (o : Nat)
    (hmiss : r' ≠ r ∨ ∀ i ∈ l, o ≠ off i) (s : State α) :
    (l.foldl (fun t i => t.write r (off i) (val i)) s).mem r' o = s.mem r' o := by
  induction l generalizing s with
  | nil => rfl
  | cons a rest ih =>
    rw [List.foldl_cons, ih]
    · apply write_other
      rcases hmiss with h | h
      · exact Or.inl h
      · exact Or.inr (h a (by simp))
    · rcases hmiss with h | h
      · exact Or.inl h
      · exact Or.inr fun i hi => h i (List.mem_cons_of_mem _ hi)

theorem scatter_readback_list {α ι : Type} (r : RegionName) (off : ι → Nat)
    (val : ι → Cell α) (l : List ι) (s : State α)
    (i : ι) (hnd : l.Nodup) (hi : i ∈ l) (hinj : Function.Injective off) :
    (l.foldl (fun t k => t.write r (off k) (val k)) s).mem r (off i) = val i := by
  obtain ⟨before, after, hl⟩ := List.append_of_mem hi
  rw [hl, List.nodup_append, List.nodup_cons] at hnd
  obtain ⟨_, ⟨hnot, _⟩, _⟩ := hnd
  rw [hl, List.foldl_append, List.foldl_cons, scatter_frame]
  · exact write_same _ _ _ _
  · right
    intro k hk heq
    exact hnot ((hinj heq) ▸ hk)

theorem scatter_readback {α : Type} (r : RegionName) (off : TileIndex shape → Nat)
    (val : TileIndex shape → Cell α) (s : State α)
    (i : TileIndex shape) (hinj : Function.Injective off) :
    ((TileShape.allIndices shape).foldl (fun t k => t.write r (off k) (val k)) s).mem r (off i) = val i :=
  scatter_readback_list r off val _ s i (TileShape.allIndices_nodup shape)
    (TileShape.mem_allIndices shape i) hinj

@[simp] theorem scatter_pids {α ι : Type} (r : RegionName) (off : ι → Nat)
    (val : ι → Cell α) (l : List ι) (s : State α) :
    (l.foldl (fun t i => t.write r (off i) (val i)) s).pids = s.pids := by
  induction l generalizing s with
  | nil => rfl
  | cons a rest ih => exact ih _

/-- Masked stores are exactly the scatter over active lanes. Inactive lanes
neither overwrite existing cells nor provide a value for load forwarding. -/
theorem masked_scatter_eq_filter {α ι : Type} (r : RegionName) (off : ι → Nat)
    (val : ι → Cell α) (active : ι → Prop) [DecidablePred active]
    (l : List ι) (s : State α) :
    l.foldl (fun t i => if active i then t.write r (off i) (val i) else t) s =
      (l.filter (fun i => decide (active i))).foldl (fun t i => t.write r (off i) (val i)) s := by
  induction l generalizing s with
  | nil => rfl
  | cons a rest ih =>
    by_cases ha : active a <;> simp [ha, ih]

theorem masked_scatter_frame {α ι : Type} (r : RegionName) (off : ι → Nat)
    (val : ι → Cell α) (active : ι → Prop) [DecidablePred active]
    (l : List ι) (r' : RegionName) (o : Nat)
    (hmiss : r' ≠ r ∨ ∀ i ∈ l, active i → o ≠ off i) (s : State α) :
    (l.foldl (fun t i => if active i then t.write r (off i) (val i) else t) s).mem r' o =
      s.mem r' o := by
  rw [masked_scatter_eq_filter]
  apply scatter_frame
  rcases hmiss with hr | ho
  · exact Or.inl hr
  · right
    intro i hi
    simp only [List.mem_filter, decide_eq_true_eq] at hi
    exact ho i hi.1 hi.2

theorem masked_scatter_readback {α : Type} (r : RegionName)
    (off : TileIndex shape → Nat) (val : TileIndex shape → Cell α)
    (active : TileIndex shape → Prop) [DecidablePred active] (s : State α)
    (i : TileIndex shape) (hi : active i) (hinj : Function.Injective off) :
    ((TileShape.allIndices shape).foldl
      (fun t k => if active k then t.write r (off k) (val k) else t) s).mem r (off i) = val i := by
  rw [masked_scatter_eq_filter]
  exact scatter_readback_list r off val _ s i
    ((TileShape.allIndices_nodup shape).filter _)
    (by simp [TileShape.mem_allIndices, hi]) hinj

@[simp] theorem masked_scatter_pids {α ι : Type} (r : RegionName) (off : ι → Nat)
    (val : ι → Cell α) (active : ι → Prop) [DecidablePred active]
    (l : List ι) (s : State α) :
    (l.foldl (fun t i => if active i then t.write r (off i) (val i) else t) s).pids = s.pids := by
  rw [masked_scatter_eq_filter, scatter_pids]

end State

/-- Binary address arithmetic stays exact; floating arithmetic is opaque. -/
def numeric {α : Type} (M : Algebra α) (p : Option ComputeDType) (op : Binary) :
    {dtype : TileDType} → NumericDType dtype → Value α dtype → Value α dtype → Value α dtype
  | _, .real => M.binary p .real op
  | _, .fp32 => M.binary p .fp32 op
  | _, .fp16 => M.binary p .fp16 op
  | _, .bf16 => M.binary p .bf16 op
  | _, .f8e4 => M.binary p .f8e4 op
  | _, .f8e5 => M.binary p .f8e5 op
  | _, .nat => match op with
      | .add => (· + ·) | .sub => (· - ·) | .mul => (· * ·)
      | .div => (· / ·) | .max => max | .pow => (· ^ ·)
  | _, .int => match op with
      | .add => (· + ·) | .sub => (· - ·) | .mul => (· * ·)
      | .div => (· / ·) | .max => max | .pow => fun a b => a ^ b.toNat

def bop {A B C : Type} (f : A → B → C)
    (bc : Broadcast a b out) (va : TileIndex a → A) (vb : TileIndex b → B) : TileIndex out → C :=
  fun i => f (va (bc.leftIndex i)) (vb (bc.rightIndex i))

def natLt {α : Type} : {dtype : TileDType} → ComparableDType dtype →
    Option (Value α dtype → Value α dtype → Bool)
  | _, .nat => some (fun a b => decide (a < b))
  | _, _ => none

def natLe {α : Type} : {dtype : TileDType} → ComparableDType dtype →
    Option (Value α dtype → Value α dtype → Bool)
  | _, .nat => some (fun a b => decide (a ≤ b))
  | _, _ => none

def numericLt {α : Type} (M : Algebra α) (p : Option ComputeDType) :
    {dtype : TileDType} → ComparableDType dtype → Option (Value α dtype → Value α dtype → Bool)
  | _, .real => M.compareLt p .real
  | _, .fp32 => M.compareLt p .fp32
  | _, .fp16 => M.compareLt p .fp16
  | _, .bf16 => M.compareLt p .bf16
  | _, .f8e4 => M.compareLt p .f8e4
  | _, .f8e5 => M.compareLt p .f8e5
  | _, h => natLt h

def numericLe {α : Type} (M : Algebra α) (p : Option ComputeDType) :
    {dtype : TileDType} → ComparableDType dtype → Option (Value α dtype → Value α dtype → Bool)
  | _, .real => M.compareLe p .real
  | _, .fp32 => M.compareLe p .fp32
  | _, .fp16 => M.compareLe p .fp16
  | _, .bf16 => M.compareLe p .bf16
  | _, .f8e4 => M.compareLe p .f8e4
  | _, .f8e5 => M.compareLe p .f8e5
  | _, h => natLe h

@[simp] theorem numericLt_nat {α : Type} (M : Algebra α) (p : Option ComputeDType) :
    numericLt M p .nat = natLt (α := α) .nat := rfl

@[simp] theorem numericLe_nat {α : Type} (M : Algebra α) (p : Option ComputeDType) :
    numericLe M p .nat = natLe (α := α) .nat := rfl

set_option maxHeartbeats 1600000 in
/-- Evaluate the supported expression fragment without any floating laws.
The final failure branch is deliberate: never fall back to the Real evaluator. -/
noncomputable def evalOp {α : Type} [Inhabited α] (M : Algebra α) (p : Option ComputeDType) :
    Op dtype shape → State α → Option (Values α dtype shape)
  | .const c, _ => some (fun _ => M.literal p .real c)
  | .constFloat d c, _ => some (fun _ => ofFloat d (M.literal p d c))
  | .constNat n, _ => some (fun _ => n)
  | .constInt n, _ => some (fun _ => n)
  | .constBool b, _ => some (fun _ => b)
  | .negInf, _ => some (fun _ => M.negInf)
  | .programId a, s => some (fun _ => s.pids a)
  | .numPrograms a, s => some (fun _ => s.numPids a)
  | .ref d sh n, s => s.regs d sh n
  | .arange _, _ => some (fun i => i.1.val)
  | .broadcast e _, s | .full _ e, s => do
      let v ← evalOp M p e s
      return fun _ => v PUnit.unit
  | .castFloat src dst e, s => do
      let v ← evalOp M p e s
      return fun i => ofFloat dst (M.cast p src dst (toFloat src (v i)))
  | .castNatToInt e, s => do
      let v ← evalOp M p e s
      return fun i => Int.ofNat (v i)
  | .castIntToNat e, s => do
      let v ← evalOp M p e s
      return fun i => (v i).toNat
  | .add d bc a b, s => return bop (numeric M p .add d) bc (← evalOp M p a s) (← evalOp M p b s)
  | .sub d bc a b, s => return bop (numeric M p .sub d) bc (← evalOp M p a s) (← evalOp M p b s)
  | .mul d bc a b, s => return bop (numeric M p .mul d) bc (← evalOp M p a s) (← evalOp M p b s)
  | .div d bc a b, s => return bop (numeric M p .div d) bc (← evalOp M p a s) (← evalOp M p b s)
  | .exp a, s => return (M.unary p .exp) ∘ (← evalOp M p a s)
  | .libdeviceExp a, s => return (M.unary p .libdeviceExp) ∘ (← evalOp M p a s)
  | .libdeviceLog a, s => return (M.unary p .libdeviceLog) ∘ (← evalOp M p a s)
  | .libdeviceExpm1 a, s => return (M.unary p .libdeviceExpm1) ∘ (← evalOp M p a s)
  | .libdeviceLog1p a, s => return (M.unary p .libdeviceLog1p) ∘ (← evalOp M p a s)
  | .exp2 a, s => return (M.unary p .exp2) ∘ (← evalOp M p a s)
  | .log a, s => return (M.unary p .log) ∘ (← evalOp M p a s)
  | .log2 a, s => return (M.unary p .log2) ∘ (← evalOp M p a s)
  | .sigmoid a, s => return (M.unary p .sigmoid) ∘ (← evalOp M p a s)
  | .sqrt a, s => return (M.unary p .sqrt) ∘ (← evalOp M p a s)
  | .rsqrt a, s => return (M.unary p .rsqrt) ∘ (← evalOp M p a s)
  | .tanh a, s => return (M.unary p .tanh) ∘ (← evalOp M p a s)
  | .sin a, s => return (M.unary p .sin) ∘ (← evalOp M p a s)
  | .cos a, s => return (M.unary p .cos) ∘ (← evalOp M p a s)
  | .tan a, s => return (M.unary p .tan) ∘ (← evalOp M p a s)
  | .atan a, s => return (M.unary p .atan) ∘ (← evalOp M p a s)
  | .cosh a, s => return (M.unary p .cosh) ∘ (← evalOp M p a s)
  | .sinh a, s => return (M.unary p .sinh) ∘ (← evalOp M p a s)
  | .erf a, s => return (M.unary p .erf) ∘ (← evalOp M p a s)
  | .max2 bc a b, s => return bop (M.binary p .real .max) bc (← evalOp M p a s) (← evalOp M p b s)
  | .pow bc a b, s => return bop (M.binary p .real .pow) bc (← evalOp M p a s) (← evalOp M p b s)
  | .natToReal e, s => do
      let v ← evalOp M p e s
      return fun i => M.fromNat p (v i)
  | .intToReal e, s => do
      let v ← evalOp M p e s
      return fun i => M.fromInt p (v i)
  | .boolAnd bc a b, s => return bop (· && ·) bc (← evalOp M p a s) (← evalOp M p b s)
  | .boolOr bc a b, s => return bop (· || ·) bc (← evalOp M p a s) (← evalOp M p b s)
  | .boolNot a, s => do
      let v ← evalOp M p a s
      return fun i => !(v i)
  | .lt h bc a b, s => do
      let f ← numericLt M p h
      return bop f bc (← evalOp M p a s) (← evalOp M p b s)
  | .le h bc a b, s => do
      let f ← numericLe M p h
      return bop f bc (← evalOp M p a s) (← evalOp M p b s)
  | .ge h bc a b, s => do
      let f ← numericLe M p h
      return bop (fun a b => f b a) bc (← evalOp M p a s) (← evalOp M p b s)
  | .ptrBase r, _ => some (fun _ => (Region.cast r, 0))
  | .ptrAdd bc a b, s => return bop (fun (a : RegionName × Nat) (b : Nat) => (a.1, a.2 + b)) bc (← evalOp M p a s) (← evalOp M p b s)
  | .where c a b, s => do
      let vc ← evalOp M p c s
      let va ← evalOp M p a s
      let vb ← evalOp M p b s
      return fun i => if vc i then va i else vb i
  | .ite c a b, s => do
      let vc ← evalOp M p c s
      if vc PUnit.unit then evalOp M p a s else evalOp M p b s
  | .load d mem mask, s => do
      let addressResult : Option (TileIndex _ → RegionName × Nat) := match mem with
        | .region r off => (evalOp M p off s).map (fun offsets i => (Region.cast r, offsets i))
        | .ptr e => evalOp M p e s
        | .blockPtr .. => none
      let addresses ← addressResult
      let activeResult : Option (TileIndex _ → Bool) := match mask with
        | .none => some (fun _ => Bool.true)
        | .mask e | .maskOther e _ => evalOp M p e s
      let active ← activeResult
      let otherResult : Option (Values α d _) := match mask with
        | .maskOther _ e => evalOp M p e s
        | _ => some (fun i => s.undef d (addresses i).1 (addresses i).2)
      let other ← otherResult
      return fun i => if active i then (s.mem (addresses i).1 (addresses i).2).read d else other i
  | .reduceMax axis keepDims e, s => do
      let v ← evalOp M p e s
      if 0 < TileShape.axisDim _ axis then
        return M.reduceMax p axis keepDims v
      else none
  | .reduceSum axis keepDims e, s => do
      let v ← evalOp M p e s
      return M.reduceSum p axis keepDims v
  | .castRealToInt8 .., _ => none
  | .floorDiv .., _ => none
  | .mod .., _ => none
  | .bitAnd .., _ => none
  | .bitOr .., _ => none
  | .bitXor .., _ => none
  | .shiftLeft .., _ => none
  | .shiftRight .., _ => none
  | .eq .., _ => none
  | .gt .., _ => none
  | .ne .., _ => none
  | .reduceMaxNat .., _ => none
  | .scan .., _ => none
  | .argMax .., _ => none
  | .argMin .., _ => none
  | .sort .., _ => none
  | .dot .., _ => none
  | .dotInt .., _ => none
  | .transpose .., _ => none
  | .reshape .., _ => none
  | .remap .., _ => none
  | .join .., _ => none
  | .split .., _ => none
  | .expandDim .., _ => none
  | .ptrSub .., _ => none
  | .makeBlockPtr .., _ => none
  | .makeBlockPtrDyn .., _ => none
  | .makeBlockPtrDynOffsets .., _ => none
  | .advanceBlockPtr .., _ => none

-- Materialize the dependent unfold theorem in its defining module. Imported
-- lazy equation generation uses Lean's default heartbeat limit instead of the
-- evaluator's declaration options.
abbrev evalOp_unfold := @evalOp.eq_def

noncomputable def evalComputeOp {α : Type} [Inhabited α] (M : Algebra α) :
    ComputeOp dtype shape → State α → Option (Values α dtype.eraseDType shape)
  | .alg d e, s => evalOp M (some d) e s
  | .const (dtype := .fp64) _, _ => none
  | .const (dtype := .fp32) c, _ => some (fun _ => M.fp32Bits c)
  | .const (dtype := .int32) c, _ => some (fun _ => c.toInt)
  | .const (dtype := .uint32) c, _ => some (fun _ => c.toNat)
  | .full _ e, s => do
      let v ← evalComputeOp M e s
      return fun _ => v PUnit.unit
  | .load .fp64 .., _ => none
  | .load .fp32 mem mask, s => do
      let v ← evalOp M (some .fp32) (.load .real mem mask) s
      return fun i => M.fp32Load (v i)
  | .load .int32 mem mask, s => evalOp M (some .int32) (.load .int mem mask) s
  | .load .uint32 mem mask, s => evalOp M (some .uint32) (.load .nat mem mask) s
  | .bitcast .., _ => none

noncomputable def evalExpr {α : Type} [Inhabited α] (M : Algebra α) :
    ComputeExpr dtype shape → State α → Option (Values α dtype shape)
  | .alg e, s => evalOp M none e s
  | .compute e, s => evalComputeOp M e s
  | _, _ => none

/-- The address and mask are evaluated before writes, as in the operational
semantics. Each active lane copies its already typed value into memory. -/
noncomputable def store {α : Type} [Inhabited α] (M : Algebra α)
    (mem : MemAccess dtype shape) (mask : MaskOpt dtype shape)
    (values : Values α dtype shape) (s : State α) : Option (State α) := do
  let addressResult : Option (TileIndex shape → RegionName × Nat) := match mem with
    | .region r off => (evalOp M none off s).map (fun offsets i => (Region.cast r, offsets i))
    | .ptr e => evalOp M none e s
    | .blockPtr .. => none
  let addresses ← addressResult
  let activeResult : Option (TileIndex shape → Bool) := match mask with
    | .none => some (fun _ => Bool.true)
    | .mask e | .maskOther e _ => evalOp M none e s
  let active ← activeResult
  return (TileShape.allIndices shape).foldl (fun t i =>
    if active i then t.write (addresses i).1 (addresses i).2 (.mk dtype (values i)) else t) s

noncomputable def stepAlg {α : Type} [Inhabited α] (M : Algebra α) :
    Stmt → State α → Option (State α)
  | .assign d sh n e, s => do
      let v ← evalOp M none e s
      return s.setReg n d sh v
  | .store _ _ mem e mask, s => do
      store M mem mask (← evalOp M none e s) s
  | _, _ => none

mutual

noncomputable def step {α : Type} [Inhabited α] (M : Algebra α) :
    ComputeStmt → State α → Option (State α)
  | .alg st, s => stepAlg M st s
  | .assign d sh n e, s => do
      let v ← evalExpr M e s
      return s.setReg n d sh v
  | .store _ _ mem e mask, s => do
      store M mem mask (← evalExpr M e s) s
  | .forLoop idx n body, s => loop M idx 0 n body s
  | .forRange idx start stop stride body, s => range M idx start stop stride body s
  | .forRangeDyn idx start stop stride body, s => do
      let a ← evalOp M none start s
      let b ← evalOp M none stop s
      let d ← evalOp M none stride s
      range M idx (a PUnit.unit) (b PUnit.unit) (d PUnit.unit) body s
  | .ifThen cond body, s => do
      let c ← evalOp M none cond s
      if c PUnit.unit then run M body s else some s
  | .ifThenElse cond yes no, s => do
      let c ← evalOp M none cond s
      if c PUnit.unit then run M yes s else run M no s
  | _, _ => none
termination_by st _ => (sizeOf st, 0)
decreasing_by
  all_goals simp_wf
  all_goals (try omega)
  all_goals (have : 0 < sizeOf idx := by cases idx; simp)
  all_goals omega

noncomputable def run {α : Type} [Inhabited α] (M : Algebra α) :
    List ComputeStmt → State α → Option (State α)
  | [], s => some s
  | st :: rest, s => do run M rest (← step M st s)
termination_by code _ => (sizeOf code, 0)
decreasing_by all_goals (simp_wf; omega)

/-- Counted loops bind the exact natural index before every iteration. The
floating operations inside the body still use the same opaque algebra. -/
noncomputable def loop {α : Type} [Inhabited α] (M : Algebra α)
    (idx : RegName) (start stop : Nat) (body : List ComputeStmt) :
    State α → Option (State α)
  | s => if start < stop then do
      let t ← run M body (s.setReg idx .nat [] (fun _ => start))
      loop M idx (start + 1) stop body t
    else some s
termination_by _ => (sizeOf body + 1, stop - start)
decreasing_by all_goals omega

/-- Range bounds are evaluated once on entry. As in the existing operational
semantics, a zero stride or an empty interval executes no iterations. -/
noncomputable def range {α : Type} [Inhabited α] (M : Algebra α)
    (idx : RegName) (cur stop stride : Nat) (body : List ComputeStmt) :
    State α → Option (State α)
  | s => if stride = 0 then some s
    else if cur < stop then do
      let t ← run M body (s.setReg idx .nat [] (fun _ => cur))
      range M idx (cur + stride) stop stride body t
    else some s
termination_by _ => (sizeOf body + 1, stop - cur)
decreasing_by all_goals omega

end

noncomputable def exec {α : Type} [Inhabited α] (M : Algebra α) :
    ComputeKernel → State α → Option (State α)
  | .mk _ _ code, s => run M code s

theorem run_append {α : Type} [Inhabited α] (M : Algebra α)
    (before after : List ComputeStmt) (s : State α) :
    run M (before ++ after) s = (run M before s).bind (run M after) := by
  induction before generalizing s with
  | nil => simp [run]
  | cons st rest ih =>
    simp only [List.cons_append, run]
    cases step M st s with
    | none => rfl
    | some t => exact ih t

theorem exec_seq_cons {α : Type} [Inhabited α] (M : Algebra α)
    (ins outs : List RegionName) (k : ComputeKernel) (ks : List ComputeKernel) (s : State α) :
    exec M (ComputeKernel.seq ins outs (k :: ks)) s =
      (exec M k s).bind (exec M (ComputeKernel.seq ins outs ks)) := by
  cases k
  simp only [ComputeKernel.seq, ComputeKernel.surfaceBody, List.flatMap_cons,
    exec, run_append]

theorem exec_seq_nil {α : Type} [Inhabited α] (M : Algebra α)
    (ins outs : List RegionName) (s : State α) :
    exec M (ComputeKernel.seq ins outs []) s = some s := by
  simp [ComputeKernel.seq, exec, run]

end VeriTile.Triton.FP.Structural
