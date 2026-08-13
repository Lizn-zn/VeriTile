/-
VeriTile.Triton.Float.EvalOpR

Rounding-model-parametric expression evaluation (#447).

`evalOpR R` is `evalOp` with one difference: the `Op.castFloat` case applies
the model's rounding at the destination dtype (`RoundingModel.cast`) instead
of the identity round-trip (`FloatDType.cast`). Every other case is a verbatim
copy that threads `R` through the recursion. The degeneration theorem
`evalOpR_triv` states that the trivial model recovers `evalOp` exactly, so
the existing semantics is the `R = .triv` instance of this one.

This file deliberately does NOT modify `evalOp` or anything it depends on:
the parametric evaluator is an addition, not a replacement.
-/

import VeriTile.Triton.Semantics.EvalOp
import VeriTile.Triton.Float.RoundingModel

namespace VeriTile.Triton

set_option maxHeartbeats 800000 in
/-- `evalOp` parametrized by a `RoundingModel`: float casts round at the
destination dtype (rounding-event site 1 of 2; site 2 is the narrowing
store, see `StepR`). Verbatim `evalOp` otherwise. -/
noncomputable def evalOpR (R : RoundingModel) :
    Op dtype shape → BlockState → Option (Tile dtype shape)
  | .const c, _ => some (Tile.scalar (some c : WithBot ℝ))
  | .constFloat h c, _ => some (Tile.scalar (h.ofReal c))
  | .constNat n, _ => some (Tile.scalar n)
  | .constInt n, _ => some (Tile.scalar n)
  | .constBool b, _ => some (Tile.scalar b)
  | .negInf, _ => some (Tile.scalar (none : WithBot ℝ))
  | .programId axis, s => some (Tile.scalar (s.pids axis))
  | .numPrograms axis, s => some (Tile.scalar (s.numPids axis))
  | .ref dtype shape name, s => s.regs dtype shape name
  | .arange n, _ => some (Tile.vec (fun i => i.val))
  | .broadcast e shape, s => do
      let v ← evalOpR R e s
      some ⟨fun _ => v.data PUnit.unit⟩
  | .full shape e, s => do
      let v ← evalOpR R e s
      some ⟨fun _ => v.data PUnit.unit⟩
  | .castFloat src dst a, s => do
      let va ← evalOpR R a s
      some ⟨fun i => R.cast src dst (va.data i)⟩
  | .castNatToInt a, s => do
      let va ← evalOpR R a s
      some ⟨fun i => Int.ofNat (va.data i)⟩
  | .castIntToNat a, s => do
      let va ← evalOpR R a s
      some ⟨fun i => (va.data i).toNat⟩
  | .castRealToInt8 a, s => do
      let va ← evalOpR R a s
      some ⟨fun i => WithBot.realToInt8 (va.data i)⟩
  | .add h bc a b, s => do
      let va ← evalOpR R a s
      let vb ← evalOpR R b s
      some (Tile.bop h.add bc va vb)
  | .sub h bc a b, s => do
      let va ← evalOpR R a s
      let vb ← evalOpR R b s
      some (Tile.bop h.sub bc va vb)
  | .mul h bc a b, s => do
      let va ← evalOpR R a s
      let vb ← evalOpR R b s
      some (Tile.bop h.mul bc va vb)
  | .div h bc a b, s => do
      let va ← evalOpR R a s
      let vb ← evalOpR R b s
      some (Tile.bop h.div bc va vb)
  | .floorDiv h bc a b, s => do
      let va ← evalOpR R a s
      let vb ← evalOpR R b s
      some (Tile.bop h.floorDiv bc va vb)
  | .mod h bc a b, s => do
      let va ← evalOpR R a s
      let vb ← evalOpR R b s
      some (Tile.bop h.mod bc va vb)
  | .bitAnd bc a b, s => return Tile.bop (· &&& ·) bc (← evalOpR R a s) (← evalOpR R b s)
  | .bitOr bc a b, s => return Tile.bop (· ||| ·) bc (← evalOpR R a s) (← evalOpR R b s)
  | .bitXor bc a b, s => return Tile.bop (· ^^^ ·) bc (← evalOpR R a s) (← evalOpR R b s)
  | .shiftLeft bc a b, s => return Tile.bop (· <<< ·) bc (← evalOpR R a s) (← evalOpR R b s)
  | .shiftRight bc a b, s => return Tile.bop (· >>> ·) bc (← evalOpR R a s) (← evalOpR R b s)
  | .exp a, s => return Tile.uop WithBot.realExp (← evalOpR R a s)
  | .exp2 a, s => return Tile.uop WithBot.realExp2 (← evalOpR R a s)
  | .log a, s => return Tile.uop WithBot.realLog (← evalOpR R a s)
  | .log2 a, s => return Tile.uop WithBot.realLog2 (← evalOpR R a s)
  | .sigmoid a, s => return Tile.uop WithBot.realSigmoid (← evalOpR R a s)
  | .sqrt a, s => return Tile.uop WithBot.realSqrt (← evalOpR R a s)
  | .rsqrt a, s => return Tile.uop WithBot.realRsqrt (← evalOpR R a s)
  | .tanh a, s => return Tile.uop WithBot.realTanh (← evalOpR R a s)
  | .sin a, s => return Tile.uop WithBot.realSin (← evalOpR R a s)
  | .cos a, s => return Tile.uop WithBot.realCos (← evalOpR R a s)
  | .tan a, s => return Tile.uop WithBot.realTan (← evalOpR R a s)
  | .atan a, s => return Tile.uop WithBot.realAtan (← evalOpR R a s)
  | .cosh a, s => return Tile.uop WithBot.realCosh (← evalOpR R a s)
  | .sinh a, s => return Tile.uop WithBot.realSinh (← evalOpR R a s)
  | .erf a, s => return Tile.uop WithBot.realErf (← evalOpR R a s)
  | .lt h bc a b, s => return Tile.cop h.lt bc (← evalOpR R a s) (← evalOpR R b s)
  | .le h bc a b, s => return Tile.cop h.le bc (← evalOpR R a s) (← evalOpR R b s)
  | .eq h bc a b, s => return Tile.cop h.eq bc (← evalOpR R a s) (← evalOpR R b s)
  | .gt h bc a b, s => return Tile.cop h.gt bc (← evalOpR R a s) (← evalOpR R b s)
  | .ge h bc a b, s => return Tile.cop h.ge bc (← evalOpR R a s) (← evalOpR R b s)
  | .ne h bc a b, s => return Tile.cop h.ne bc (← evalOpR R a s) (← evalOpR R b s)
  | .boolAnd bc a b, s => do
      let va ← evalOpR R a s
      let vb ← evalOpR R b s
      some (Tile.bop (fun x y : Bool => x && y) bc va vb)
  | .boolOr bc a b, s => do
      let va ← evalOpR R a s
      let vb ← evalOpR R b s
      some (Tile.bop (fun x y : Bool => x || y) bc va vb)
  | .boolNot a, s => do
      let va ← evalOpR R a s
      some (Tile.uop (fun x : Bool => !x) va)
  | .max2 bc a b, s => do
      let va ← evalOpR R a s
      let vb ← evalOpR R b s
      some (Tile.bop max bc va vb)
  | .pow bc a b, s => do
      let va ← evalOpR R a s
      let vb ← evalOpR R b s
      some (Tile.bop WithBot.realPow bc va vb)
  | .reduceMax axis keepDims a, s => do
      let va ← evalOpR R a s
      Tile.reduceMax axis keepDims va
  | .reduceMaxNat axis keepDims a, s => do
      let va ← evalOpR R a s
      Tile.reduceMaxNat axis keepDims va
  | .reduceSum axis keepDims a, s => return Tile.reduceSum axis keepDims (← evalOpR R a s)
  | .scan op axis dir a, s => return Tile.scan op axis dir (← evalOpR R a s)
  | .argMax axis a, s => return Tile.argMaxDrop axis (← evalOpR R a s)
  | .argMin axis a, s => return Tile.argMinDrop axis (← evalOpR R a s)
  | .sort axis a, s => return Tile.sortAxis axis (← evalOpR R a s)
  | .dot (batch := batch) a b, s => do
      let va ← evalOpR R a s
      let vb ← evalOpR R b s
      some (Tile.dot batch va vb)
  | .expandDim axis a, s => return Tile.expandDim axis (← evalOpR R a s)
  | .where c a b, s => do
      let vc ← evalOpR R c s
      let va ← evalOpR R a s
      let vb ← evalOpR R b s
      some (Tile.select vc va vb)
  | .ite c a b, s => do
      let vc ← evalOpR R c s
      if vc.data PUnit.unit then evalOpR R a s else evalOpR R b s
  | .transpose (batch := batch) a, s => do
      let va ← evalOpR R a s
      some (Tile.transpose batch va)
  | .reshape _ a, s => return Tile.reshape (← evalOpR R a s)
  | .remap _ map a, s => return Tile.remap map (← evalOpR R a s)
  | .join a b, s => return Tile.join (← evalOpR R a s) (← evalOpR R b s)
  | .split side a, s => return Tile.split side (← evalOpR R a s)
  | .ptrBase region, _ => some (Tile.scalar (Region.cast region, 0))
  | .ptrAdd bc ptr off, s => do
      let ptrs ← evalOpR R ptr s
      let offs ← evalOpR R off s
      some (Tile.ptrAdd bc ptrs offs)
  | .ptrSub bc ptr off, s => do
      let ptrs ← evalOpR R ptr s
      let offs ← evalOpR R off s
      some (Tile.ptrSub bc ptrs offs)
  | .makeBlockPtr region baseOffset parentShape blockShape strides offsets, _ =>
      some ⟨fun _ =>
        { region := Region.cast region
        , baseOffset := baseOffset
        , parentShape := parentShape
        , blockShape := blockShape
        , strides := strides
        , offsets := offsets }⟩
  | .makeBlockPtrDyn region baseOffset parentShape blockShape strides offsets, s => do
      let base ← evalOpR R baseOffset s
      some ⟨fun _ =>
        { region := region
        , baseOffset := base.data PUnit.unit
        , parentShape := parentShape
        , blockShape := blockShape
        , strides := strides
        , offsets := offsets }⟩
  | .makeBlockPtrDynOffsets region baseOffset parentShape blockShape strides offsets, s => do
      let base ← evalOpR R baseOffset s
      let offsets ← offsets.mapM (fun off => do
        let v ← evalOpR R off s
        some (v.data PUnit.unit))
      some ⟨fun _ =>
        { region := region
        , baseOffset := base.data PUnit.unit
        , parentShape := parentShape
        , blockShape := blockShape
        , strides := strides
        , offsets := offsets }⟩
  | .advanceBlockPtr ptr deltas, s => do
      let p ← evalOpR R ptr s
      some ⟨fun i => (p.data i).advance deltas⟩
  | .load dtype mem mask, s => do
      let addr : TileIndex shape → RegionName × Nat × Bool ←
        match mem with
        | .region region off => do
            let offsets ← evalOpR R off s
            some fun i => (Region.cast region, offsets.data i, true)
        | .ptr ptr => do
            let ptrs ← evalOpR R ptr s
            some fun i =>
              let p := ptrs.data i
              (p.1, p.2, true)
        | .blockPtr ptr boundaryCheck => do
            let ptrs ← evalOpR R ptr s
            some fun i =>
              let bp := ptrs.data i
              let idx := TileShape.indexToList shape i
              (bp.region, bp.address idx, bp.inBounds idx boundaryCheck)
      let read := fun i =>
        let (region, offset, ok) := addr i
        if ok then s.readMemValue dtype region offset else BlockState.defaultCarrier dtype
      match mask with
      | .none => some ⟨read⟩
      | .mask mask => do
          let masks ← evalOpR R mask s
          some ⟨fun i =>
            let (region, offset, ok) := addr i
            if masks.data i then
              if ok then s.readMemValue dtype region offset else BlockState.defaultCarrier dtype
            else
              match dtype with
              | .real => if ok then some (s.undef region offset) else some 0
              | .fp32 => if ok then FloatDType.fp32.ofReal (s.undef region offset) else FloatDType.fp32.ofReal 0
              | .fp16 => if ok then FloatDType.fp16.ofReal (s.undef region offset) else FloatDType.fp16.ofReal 0
              | .bf16 => if ok then FloatDType.bf16.ofReal (s.undef region offset) else FloatDType.bf16.ofReal 0
              | .f8e4 => if ok then FloatDType.f8e4.ofReal (s.undef region offset) else FloatDType.f8e4.ofReal 0
              | .f8e5 => if ok then FloatDType.f8e5.ofReal (s.undef region offset) else FloatDType.f8e5.ofReal 0
              | dtype => BlockState.defaultCarrier dtype⟩
      | .maskOther mask other => do
          let masks ← evalOpR R mask s
          let others ← evalOpR R other s
          some ⟨fun i => if masks.data i then read i else others.data i⟩
  | .natToReal a, s => return Tile.natToReal (← evalOpR R a s)

/-! ## Degeneration to the trivial model

At `RoundingModel.triv` every rounding function is the identity, so the
parametric evaluator collapses onto `evalOp` case by case:
`RoundingModel.triv_cast` discharges the single differing case
(`Op.castFloat`), and every other arm is a verbatim copy whose recursive
occurrences rewrite by the induction hypothesis. Stated as an
equation-compiler recursion (one arm per constructor), mutually with a
pointwise `List.mapM` congruence for the nested offsets list of
`Op.makeBlockPtrDynOffsets`. -/

set_option maxHeartbeats 1600000 in
mutual

/-- The rounding-model-parametric evaluator degenerates to the existing
evaluator at the trivial model: `evalOpR .triv` *is* `evalOp`. -/
theorem evalOpR_triv : ∀ {dtype : TileDType} {shape : TileShape} (op : Op dtype shape)
    (s : BlockState), evalOpR RoundingModel.triv op s = evalOp op s
  | _, _, .const _, _ => by simp only [evalOpR, evalOp]
  | _, _, .constFloat _ _, _ => by simp only [evalOpR, evalOp]
  | _, _, .constNat _, _ => by simp only [evalOpR, evalOp]
  | _, _, .constInt _, _ => by simp only [evalOpR, evalOp]
  | _, _, .constBool _, _ => by simp only [evalOpR, evalOp]
  | _, _, .negInf, _ => by simp only [evalOpR, evalOp]
  | _, _, .programId _, _ => by simp only [evalOpR, evalOp]
  | _, _, .numPrograms _, _ => by simp only [evalOpR, evalOp]
  | _, _, .ref _ _ _, _ => by simp only [evalOpR, evalOp]
  | _, _, .arange _, _ => by simp only [evalOpR, evalOp]
  | _, _, .broadcast e _, s => by simp only [evalOpR, evalOp, evalOpR_triv e s]
  | _, _, .full _ e, s => by simp only [evalOpR, evalOp, evalOpR_triv e s]
  | _, _, .castFloat src dst a, s => by
      simp only [evalOpR, evalOp, evalOpR_triv a s, RoundingModel.triv_cast]
  | _, _, .castNatToInt a, s => by simp only [evalOpR, evalOp, evalOpR_triv a s]
  | _, _, .castIntToNat a, s => by simp only [evalOpR, evalOp, evalOpR_triv a s]
  | _, _, .castRealToInt8 a, s => by simp only [evalOpR, evalOp, evalOpR_triv a s]
  | _, _, .add _ _ a b, s => by
      simp only [evalOpR, evalOp, evalOpR_triv a s, evalOpR_triv b s]
  | _, _, .sub _ _ a b, s => by
      simp only [evalOpR, evalOp, evalOpR_triv a s, evalOpR_triv b s]
  | _, _, .mul _ _ a b, s => by
      simp only [evalOpR, evalOp, evalOpR_triv a s, evalOpR_triv b s]
  | _, _, .div _ _ a b, s => by
      simp only [evalOpR, evalOp, evalOpR_triv a s, evalOpR_triv b s]
  | _, _, .floorDiv _ _ a b, s => by
      simp only [evalOpR, evalOp, evalOpR_triv a s, evalOpR_triv b s]
  | _, _, .mod _ _ a b, s => by
      simp only [evalOpR, evalOp, evalOpR_triv a s, evalOpR_triv b s]
  | _, _, .bitAnd _ a b, s => by
      simp only [evalOpR, evalOp, evalOpR_triv a s, evalOpR_triv b s]
  | _, _, .bitOr _ a b, s => by
      simp only [evalOpR, evalOp, evalOpR_triv a s, evalOpR_triv b s]
  | _, _, .bitXor _ a b, s => by
      simp only [evalOpR, evalOp, evalOpR_triv a s, evalOpR_triv b s]
  | _, _, .shiftLeft _ a b, s => by
      simp only [evalOpR, evalOp, evalOpR_triv a s, evalOpR_triv b s]
  | _, _, .shiftRight _ a b, s => by
      simp only [evalOpR, evalOp, evalOpR_triv a s, evalOpR_triv b s]
  | _, _, .exp a, s => by simp only [evalOpR, evalOp, evalOpR_triv a s]
  | _, _, .exp2 a, s => by simp only [evalOpR, evalOp, evalOpR_triv a s]
  | _, _, .log a, s => by simp only [evalOpR, evalOp, evalOpR_triv a s]
  | _, _, .log2 a, s => by simp only [evalOpR, evalOp, evalOpR_triv a s]
  | _, _, .sigmoid a, s => by simp only [evalOpR, evalOp, evalOpR_triv a s]
  | _, _, .sqrt a, s => by simp only [evalOpR, evalOp, evalOpR_triv a s]
  | _, _, .rsqrt a, s => by simp only [evalOpR, evalOp, evalOpR_triv a s]
  | _, _, .tanh a, s => by simp only [evalOpR, evalOp, evalOpR_triv a s]
  | _, _, .sin a, s => by simp only [evalOpR, evalOp, evalOpR_triv a s]
  | _, _, .cos a, s => by simp only [evalOpR, evalOp, evalOpR_triv a s]
  | _, _, .tan a, s => by simp only [evalOpR, evalOp, evalOpR_triv a s]
  | _, _, .atan a, s => by simp only [evalOpR, evalOp, evalOpR_triv a s]
  | _, _, .cosh a, s => by simp only [evalOpR, evalOp, evalOpR_triv a s]
  | _, _, .sinh a, s => by simp only [evalOpR, evalOp, evalOpR_triv a s]
  | _, _, .erf a, s => by simp only [evalOpR, evalOp, evalOpR_triv a s]
  | _, _, .lt _ _ a b, s => by
      simp only [evalOpR, evalOp, evalOpR_triv a s, evalOpR_triv b s]
  | _, _, .le _ _ a b, s => by
      simp only [evalOpR, evalOp, evalOpR_triv a s, evalOpR_triv b s]
  | _, _, .eq _ _ a b, s => by
      simp only [evalOpR, evalOp, evalOpR_triv a s, evalOpR_triv b s]
  | _, _, .gt _ _ a b, s => by
      simp only [evalOpR, evalOp, evalOpR_triv a s, evalOpR_triv b s]
  | _, _, .ge _ _ a b, s => by
      simp only [evalOpR, evalOp, evalOpR_triv a s, evalOpR_triv b s]
  | _, _, .ne _ _ a b, s => by
      simp only [evalOpR, evalOp, evalOpR_triv a s, evalOpR_triv b s]
  | _, _, .boolAnd _ a b, s => by
      simp only [evalOpR, evalOp, evalOpR_triv a s, evalOpR_triv b s]
  | _, _, .boolOr _ a b, s => by
      simp only [evalOpR, evalOp, evalOpR_triv a s, evalOpR_triv b s]
  | _, _, .boolNot a, s => by simp only [evalOpR, evalOp, evalOpR_triv a s]
  | _, _, .max2 _ a b, s => by
      simp only [evalOpR, evalOp, evalOpR_triv a s, evalOpR_triv b s]
  | _, _, .pow _ a b, s => by
      simp only [evalOpR, evalOp, evalOpR_triv a s, evalOpR_triv b s]
  | _, _, .reduceMax _ _ a, s => by simp only [evalOpR, evalOp, evalOpR_triv a s]
  | _, _, .reduceMaxNat _ _ a, s => by simp only [evalOpR, evalOp, evalOpR_triv a s]
  | _, _, .reduceSum _ _ a, s => by simp only [evalOpR, evalOp, evalOpR_triv a s]
  | _, _, .scan _ _ _ a, s => by simp only [evalOpR, evalOp, evalOpR_triv a s]
  | _, _, .argMax _ a, s => by simp only [evalOpR, evalOp, evalOpR_triv a s]
  | _, _, .argMin _ a, s => by simp only [evalOpR, evalOp, evalOpR_triv a s]
  | _, _, .sort _ a, s => by simp only [evalOpR, evalOp, evalOpR_triv a s]
  | _, _, .dot a b, s => by
      simp only [evalOpR, evalOp, evalOpR_triv a s, evalOpR_triv b s]
  | _, _, .expandDim _ a, s => by simp only [evalOpR, evalOp, evalOpR_triv a s]
  | _, _, .where c a b, s => by
      simp only [evalOpR, evalOp, evalOpR_triv c s, evalOpR_triv a s, evalOpR_triv b s]
  | _, _, .ite c a b, s => by
      simp only [evalOpR, evalOp, evalOpR_triv c s, evalOpR_triv a s, evalOpR_triv b s]
  | _, _, .transpose a, s => by simp only [evalOpR, evalOp, evalOpR_triv a s]
  | _, _, .reshape _ a, s => by simp only [evalOpR, evalOp, evalOpR_triv a s]
  | _, _, .remap _ _ a, s => by simp only [evalOpR, evalOp, evalOpR_triv a s]
  | _, _, .join a b, s => by
      simp only [evalOpR, evalOp, evalOpR_triv a s, evalOpR_triv b s]
  | _, _, .split _ a, s => by simp only [evalOpR, evalOp, evalOpR_triv a s]
  | _, _, .ptrBase _, _ => by simp only [evalOpR, evalOp]
  | _, _, .ptrAdd _ p o, s => by
      simp only [evalOpR, evalOp, evalOpR_triv p s, evalOpR_triv o s]
  | _, _, .ptrSub _ p o, s => by
      simp only [evalOpR, evalOp, evalOpR_triv p s, evalOpR_triv o s]
  | _, _, .makeBlockPtr _ _ _ _ _ _, _ => by simp only [evalOpR, evalOp]
  | _, _, .makeBlockPtrDyn _ base _ _ _ _, s => by
      simp only [evalOpR, evalOp, evalOpR_triv base s]
  | _, _, .makeBlockPtrDynOffsets _ base _ _ _ offsets, s => by
      simp only [evalOpR, evalOp, evalOpR_triv base s, evalOpR_triv_offsets offsets s]
  | _, _, .advanceBlockPtr p _, s => by simp only [evalOpR, evalOp, evalOpR_triv p s]
  | _, _, .load _ (.region region off) .none, s => by
      simp only [evalOpR, evalOp, evalOpR_triv off s]
  | _, _, .load dt (.region region off) (.mask m), s => by
      cases dt <;> simp only [evalOpR, evalOp, evalOpR_triv off s, evalOpR_triv m s]
  | _, _, .load _ (.region region off) (.maskOther m other), s => by
      simp only [evalOpR, evalOp, evalOpR_triv off s, evalOpR_triv m s, evalOpR_triv other s]
  | _, _, .load _ (.ptr p) .none, s => by
      simp only [evalOpR, evalOp, evalOpR_triv p s]
  | _, _, .load dt (.ptr p) (.mask m), s => by
      cases dt <;> simp only [evalOpR, evalOp, evalOpR_triv p s, evalOpR_triv m s]
  | _, _, .load _ (.ptr p) (.maskOther m other), s => by
      simp only [evalOpR, evalOp, evalOpR_triv p s, evalOpR_triv m s, evalOpR_triv other s]
  | _, _, .load _ (.blockPtr p _) .none, s => by
      simp only [evalOpR, evalOp, evalOpR_triv p s]
  | _, _, .load dt (.blockPtr p _) (.mask m), s => by
      cases dt <;> simp only [evalOpR, evalOp, evalOpR_triv p s, evalOpR_triv m s]
  | _, _, .load _ (.blockPtr p _) (.maskOther m other), s => by
      simp only [evalOpR, evalOp, evalOpR_triv p s, evalOpR_triv m s, evalOpR_triv other s]
  | _, _, .natToReal a, s => by simp only [evalOpR, evalOp, evalOpR_triv a s]
termination_by dtype shape op _ => sizeOf op
decreasing_by all_goals (simp_wf <;> omega)

/-- Pointwise `evalOpR_triv` congruence across the `List.mapM` in the
`Op.makeBlockPtrDynOffsets` case (the nested-list recursion of `Op`). -/
private theorem evalOpR_triv_offsets : ∀ (offsets : List (Op .nat [])) (s : BlockState),
    offsets.mapM (fun off => do
      let v ← evalOpR RoundingModel.triv off s
      some (v.data PUnit.unit)) =
    offsets.mapM (fun off => do
      let v ← evalOp off s
      some (v.data PUnit.unit))
  | [], _ => rfl
  | off :: rest, s => by
      simp only [List.mapM_cons, evalOpR_triv off s, evalOpR_triv_offsets rest s]
termination_by offsets _ => sizeOf offsets
decreasing_by all_goals (simp_wf <;> omega)

end

@[simp] theorem evalOpR_castFloat (R : RoundingModel) (src dst : FloatDType) {shape : TileShape}
    (a : Op src.toTileDType shape) (s : BlockState) :
    evalOpR R (.castFloat src dst a) s = (do
      let va ← evalOpR R a s
      some (⟨fun i => R.cast src dst (va.data i)⟩ : Tile dst.toTileDType shape)) := by
  simp [evalOpR]

end VeriTile.Triton
