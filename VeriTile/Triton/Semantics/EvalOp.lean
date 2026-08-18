/-
VeriTile.Triton.Semantics.EvalOp

Expression evaluation for the typed Triton AST.
-/

import VeriTile.Triton.Semantics.TileOps
import Batteries.Data.Nat.Bitwise

namespace VeriTile.Triton

@[simp] theorem option_match_bind {α β : Type}
    (x : Option α) (f : α → Option β) :
    (match x, f with
      | none, _ => none
      | some a, f => f a) = x.bind f := by
  cases x <;> rfl

set_option maxHeartbeats 800000 in
noncomputable def evalOp : Op dtype shape → BlockState → Option (Tile dtype shape)
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
      let v ← evalOp e s
      some ⟨fun _ => v.data PUnit.unit⟩
  | .full shape e, s => do
      let v ← evalOp e s
      some ⟨fun _ => v.data PUnit.unit⟩
  | .castFloat src dst a, s => do
      let va ← evalOp a s
      some ⟨fun i => src.cast dst (va.data i)⟩
  | .castNatToInt a, s => do
      let va ← evalOp a s
      some ⟨fun i => Int.ofNat (va.data i)⟩
  | .castIntToNat a, s => do
      let va ← evalOp a s
      some ⟨fun i => (va.data i).toNat⟩
  | .castRealToInt8 a, s => do
      let va ← evalOp a s
      some ⟨fun i => WithBot.realToInt8 (va.data i)⟩
  | .add h bc a b, s => do
      let va ← evalOp a s
      let vb ← evalOp b s
      some (Tile.bop h.add bc va vb)
  | .sub h bc a b, s => do
      let va ← evalOp a s
      let vb ← evalOp b s
      some (Tile.bop h.sub bc va vb)
  | .mul h bc a b, s => do
      let va ← evalOp a s
      let vb ← evalOp b s
      some (Tile.bop h.mul bc va vb)
  | .div h bc a b, s => do
      let va ← evalOp a s
      let vb ← evalOp b s
      some (Tile.bop h.div bc va vb)
  | .floorDiv h bc a b, s => do
      let va ← evalOp a s
      let vb ← evalOp b s
      some (Tile.bop h.floorDiv bc va vb)
  | .mod h bc a b, s => do
      let va ← evalOp a s
      let vb ← evalOp b s
      some (Tile.bop h.mod bc va vb)
  | .bitAnd bc a b, s => return Tile.bop (· &&& ·) bc (← evalOp a s) (← evalOp b s)
  | .bitOr bc a b, s => return Tile.bop (· ||| ·) bc (← evalOp a s) (← evalOp b s)
  | .bitXor bc a b, s => return Tile.bop (· ^^^ ·) bc (← evalOp a s) (← evalOp b s)
  | .shiftLeft bc a b, s => return Tile.bop (· <<< ·) bc (← evalOp a s) (← evalOp b s)
  | .shiftRight bc a b, s => return Tile.bop (· >>> ·) bc (← evalOp a s) (← evalOp b s)
  | .exp a, s => return Tile.uop WithBot.realExp (← evalOp a s)
  | .exp2 a, s => return Tile.uop WithBot.realExp2 (← evalOp a s)
  | .log a, s => return Tile.uop WithBot.realLog (← evalOp a s)
  | .log2 a, s => return Tile.uop WithBot.realLog2 (← evalOp a s)
  | .sigmoid a, s => return Tile.uop WithBot.realSigmoid (← evalOp a s)
  | .sqrt a, s => return Tile.uop WithBot.realSqrt (← evalOp a s)
  | .rsqrt a, s => return Tile.uop WithBot.realRsqrt (← evalOp a s)
  | .tanh a, s => return Tile.uop WithBot.realTanh (← evalOp a s)
  | .sin a, s => return Tile.uop WithBot.realSin (← evalOp a s)
  | .cos a, s => return Tile.uop WithBot.realCos (← evalOp a s)
  | .tan a, s => return Tile.uop WithBot.realTan (← evalOp a s)
  | .atan a, s => return Tile.uop WithBot.realAtan (← evalOp a s)
  | .cosh a, s => return Tile.uop WithBot.realCosh (← evalOp a s)
  | .sinh a, s => return Tile.uop WithBot.realSinh (← evalOp a s)
  | .erf a, s => return Tile.uop WithBot.realErf (← evalOp a s)
  | .lt h bc a b, s => return Tile.cop h.lt bc (← evalOp a s) (← evalOp b s)
  | .le h bc a b, s => return Tile.cop h.le bc (← evalOp a s) (← evalOp b s)
  | .eq h bc a b, s => return Tile.cop h.eq bc (← evalOp a s) (← evalOp b s)
  | .gt h bc a b, s => return Tile.cop h.gt bc (← evalOp a s) (← evalOp b s)
  | .ge h bc a b, s => return Tile.cop h.ge bc (← evalOp a s) (← evalOp b s)
  | .ne h bc a b, s => return Tile.cop h.ne bc (← evalOp a s) (← evalOp b s)
  | .boolAnd bc a b, s => do
      let va ← evalOp a s
      let vb ← evalOp b s
      some (Tile.bop (fun x y : Bool => x && y) bc va vb)
  | .boolOr bc a b, s => do
      let va ← evalOp a s
      let vb ← evalOp b s
      some (Tile.bop (fun x y : Bool => x || y) bc va vb)
  | .boolNot a, s => do
      let va ← evalOp a s
      some (Tile.uop (fun x : Bool => !x) va)
  | .max2 bc a b, s => do
      let va ← evalOp a s
      let vb ← evalOp b s
      some (Tile.bop max bc va vb)
  | .pow bc a b, s => do
      let va ← evalOp a s
      let vb ← evalOp b s
      some (Tile.bop WithBot.realPow bc va vb)
  | .reduceMax axis keepDims a, s => do
      let va ← evalOp a s
      Tile.reduceMax axis keepDims va
  | .reduceMaxNat axis keepDims a, s => do
      let va ← evalOp a s
      Tile.reduceMaxNat axis keepDims va
  | .reduceSum axis keepDims a, s => return Tile.reduceSum axis keepDims (← evalOp a s)
  | .scan op axis dir a, s => return Tile.scan op axis dir (← evalOp a s)
  | .argMax axis a, s => return Tile.argMaxDrop axis (← evalOp a s)
  | .argMin axis a, s => return Tile.argMinDrop axis (← evalOp a s)
  | .sort axis a, s => return Tile.sortAxis axis (← evalOp a s)
  | .dot (batch := batch) a b, s => do
      let va ← evalOp a s
      let vb ← evalOp b s
      some (Tile.dot batch va vb)
  | .dotInt (batch := batch) a b, s => do
      let va ← evalOp a s
      let vb ← evalOp b s
      some (Tile.dotInt batch va vb)
  | .expandDim axis a, s => return Tile.expandDim axis (← evalOp a s)
  | .where c a b, s => do
      let vc ← evalOp c s
      let va ← evalOp a s
      let vb ← evalOp b s
      some (Tile.select vc va vb)
  | .ite c a b, s => do
      let vc ← evalOp c s
      if vc.data PUnit.unit then evalOp a s else evalOp b s
  | .transpose (batch := batch) a, s => do
      let va ← evalOp a s
      some (Tile.transpose batch va)
  | .reshape _ a, s => return Tile.reshape (← evalOp a s)
  | .remap _ map a, s => return Tile.remap map (← evalOp a s)
  | .join a b, s => return Tile.join (← evalOp a s) (← evalOp b s)
  | .split side a, s => return Tile.split side (← evalOp a s)
  | .ptrBase region, _ => some (Tile.scalar (Region.cast region, 0))
  | .ptrAdd bc ptr off, s => do
      let ptrs ← evalOp ptr s
      let offs ← evalOp off s
      some (Tile.ptrAdd bc ptrs offs)
  | .ptrSub bc ptr off, s => do
      let ptrs ← evalOp ptr s
      let offs ← evalOp off s
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
      let base ← evalOp baseOffset s
      some ⟨fun _ =>
        { region := region
        , baseOffset := base.data PUnit.unit
        , parentShape := parentShape
        , blockShape := blockShape
        , strides := strides
        , offsets := offsets }⟩
  | .makeBlockPtrDynOffsets region baseOffset parentShape blockShape strides offsets, s => do
      let base ← evalOp baseOffset s
      let offsets ← offsets.mapM (fun off => do
        let v ← evalOp off s
        some (v.data PUnit.unit))
      some ⟨fun _ =>
        { region := region
        , baseOffset := base.data PUnit.unit
        , parentShape := parentShape
        , blockShape := blockShape
        , strides := strides
        , offsets := offsets }⟩
  | .advanceBlockPtr ptr deltas, s => do
      let p ← evalOp ptr s
      some ⟨fun i => (p.data i).advance deltas⟩
  | .load dtype mem mask, s => do
      let addr : TileIndex shape → RegionName × Nat × Bool ←
        match mem with
        | .region region off => do
            let offsets ← evalOp off s
            some fun i => (Region.cast region, offsets.data i, true)
        | .ptr ptr => do
            let ptrs ← evalOp ptr s
            some fun i =>
              let p := ptrs.data i
              (p.1, p.2, true)
        | .blockPtr ptr boundaryCheck => do
            let ptrs ← evalOp ptr s
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
          let masks ← evalOp mask s
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
          let masks ← evalOp mask s
          let others ← evalOp other s
          some ⟨fun i => if masks.data i then read i else others.data i⟩
  | .natToReal a, s => return Tile.natToReal (← evalOp a s)
  | .intToReal a, s => return Tile.intToReal (← evalOp a s)

@[simp] theorem evalOp_programId (axis : Nat) (s : BlockState) :
    evalOp (.programId axis) s = some (Tile.scalar (s.pids axis)) := by
  simp [evalOp]

@[simp] theorem evalOp_numPrograms (axis : Nat) (s : BlockState) :
    evalOp (.numPrograms axis) s = some (Tile.scalar (s.numPids axis)) := by
  simp [evalOp]

@[simp] theorem evalOp_const (c : ℝ) (s : BlockState) :
    evalOp (.const c) s = some (Tile.scalar (some c : WithBot ℝ)) := by
  simp [evalOp]

@[simp] theorem evalOp_constNat (n : Nat) (s : BlockState) :
    evalOp (.constNat n) s = some (Tile.scalar n) := by
  simp [evalOp]

@[simp] theorem evalOp_negInf (s : BlockState) :
    evalOp .negInf s = some (Tile.scalar (none : WithBot ℝ)) := by
  simp [evalOp]

@[simp] theorem evalOp_arange (n : Nat) (s : BlockState) :
    evalOp (.arange n) s = some (Tile.vec (fun i => i.val)) := by
  simp [evalOp]

@[simp] theorem evalOp_ref (dtype : TileDType) (shape : TileShape)
    (name : RegName) (s : BlockState) :
    evalOp (.ref dtype shape name) s = s.regs dtype shape name := by
  simp [evalOp]

@[simp] theorem evalOp_full (shape : TileShape) (e : Op dtype [])
    (s : BlockState) :
    evalOp (.full shape e) s = (do
      let v ← evalOp e s
      some ⟨fun _ => v.data PUnit.unit⟩) := by
  simp [evalOp]

@[simp] theorem evalOp_expandDim (axis : Fin (shape.length + 1))
    (e : Op dtype shape) (s : BlockState) :
    evalOp (.expandDim axis e) s = (do
      let v ← evalOp e s
      some (Tile.expandDim axis v)) := by
  simp [evalOp]

@[simp 1200] theorem evalOp_expandDim_ref
    (dtype : TileDType) (shape : TileShape) (axis : Fin (shape.length + 1))
    (name : RegName) (s : BlockState) :
    evalOp (.expandDim axis (.ref dtype shape name)) s = (do
      let v ← s.regs dtype shape name
      some (Tile.expandDim axis v)) := by
  simp [evalOp]

@[simp 1400] theorem evalOp_expandDim_ref_of_regs
    (dtype : TileDType) (shape : TileShape) (axis : Fin (shape.length + 1))
    (name : RegName) (s : BlockState) (v : Tile dtype shape)
    (h : s.regs dtype shape name = some v) :
    evalOp (.expandDim axis (.ref dtype shape name)) s =
      some (Tile.expandDim axis v) := by
  simp [evalOp, h]

@[simp] theorem evalOp_castFloat (src dst : FloatDType) {shape : TileShape}
    (a : Op src.toTileDType shape) (s : BlockState) :
    evalOp (.castFloat src dst a) s = (do
      let va ← evalOp a s
      some (⟨fun i => src.cast dst (va.data i)⟩ : Tile dst.toTileDType shape)) := by
  simp [evalOp]

@[simp] theorem evalOp_add (h : NumericDType dtype)
    (bc : Broadcast a b shape) (x : Op dtype a) (y : Op dtype b)
    (s : BlockState) :
    evalOp (.add h bc x y) s = (do
      let vx ← evalOp x s
      let vy ← evalOp y s
      some (Tile.bop h.add bc vx vy)) := by
  simp [evalOp]

@[simp] theorem evalOp_mul (h : NumericDType dtype)
    (bc : Broadcast a b shape) (x : Op dtype a) (y : Op dtype b)
    (s : BlockState) :
    evalOp (.mul h bc x y) s = (do
      let vx ← evalOp x s
      let vy ← evalOp y s
      some (Tile.bop h.mul bc vx vy)) := by
  simp [evalOp]

@[simp] theorem evalOp_sub (h : NumericDType dtype)
    (bc : Broadcast a b shape) (x : Op dtype a) (y : Op dtype b)
    (s : BlockState) :
    evalOp (.sub h bc x y) s = (do
      let vx ← evalOp x s
      let vy ← evalOp y s
      some (Tile.bop h.sub bc vx vy)) := by
  simp [evalOp]

@[simp] theorem evalOp_div (h : NumericDType dtype)
    (bc : Broadcast a b shape) (x : Op dtype a) (y : Op dtype b)
    (s : BlockState) :
    evalOp (.div h bc x y) s = (do
      let vx ← evalOp x s
      let vy ← evalOp y s
      some (Tile.bop h.div bc vx vy)) := by
  simp [evalOp]

@[simp] theorem evalOp_exp (x : Op .real shape) (s : BlockState) :
    evalOp (.exp x) s = (do
      let vx ← evalOp x s
      some (Tile.uop WithBot.realExp vx)) := by
  simp [evalOp]

@[simp] theorem evalOp_tanh (x : Op .real shape) (s : BlockState) :
    evalOp (.tanh x) s = (do
      let vx ← evalOp x s
      some (Tile.uop WithBot.realTanh vx)) := by
  simp [evalOp]

@[simp] theorem evalOp_natToReal (x : Op .nat shape) (s : BlockState) :
    evalOp (.natToReal x) s = (do
      let vx ← evalOp x s
      some (Tile.natToReal vx)) := by
  simp [evalOp]

@[simp] theorem evalOp_intToReal (x : Op .int shape) (s : BlockState) :
    evalOp (.intToReal x) s = (do
      let vx ← evalOp x s
      some (Tile.intToReal vx)) := by
  simp [evalOp]

@[simp] theorem evalOp_lt
    (h : ComparableDType dtype) (bc : Broadcast a b shape)
    (x : Op dtype a) (y : Op dtype b) (s : BlockState) :
    evalOp (.lt h bc x y) s = (do
      let vx ← evalOp x s
      let vy ← evalOp y s
      some (Tile.cop h.lt bc vx vy)) := by
  simp [evalOp]

@[simp] theorem evalOp_gt
    (h : ComparableDType dtype) (bc : Broadcast a b shape)
    (x : Op dtype a) (y : Op dtype b) (s : BlockState) :
    evalOp (.gt h bc x y) s = (do
      let vx ← evalOp x s
      let vy ← evalOp y s
      some (Tile.cop h.gt bc vx vy)) := by
  simp [evalOp]

@[simp] theorem evalOp_where
    (c : Op .bool shape) (x y : Op dtype shape) (s : BlockState) :
    evalOp (.where c x y) s = (do
      let vc ← evalOp c s
      let vx ← evalOp x s
      let vy ← evalOp y s
      some (Tile.select vc vx vy)) := by
  simp [evalOp]

@[simp] theorem evalOp_max2
    (bc : Broadcast a b shape) (x : Op .real a) (y : Op .real b)
    (s : BlockState) :
    evalOp (.max2 bc x y) s = (do
      let vx ← evalOp x s
      let vy ← evalOp y s
      some (Tile.bop max bc vx vy)) := by
  simp [evalOp]

@[simp] theorem evalOp_pow
    (bc : Broadcast a b shape) (x : Op .real a) (y : Op .real b)
    (s : BlockState) :
    evalOp (.pow bc x y) s = (do
      let vx ← evalOp x s
      let vy ← evalOp y s
      some (Tile.bop WithBot.realPow bc vx vy)) := by
  simp [evalOp]

@[simp] theorem evalOp_reduceMax
    (axis : Fin shape.length) (keepDims : Bool) (x : Op .real shape)
    (s : BlockState) :
    evalOp (.reduceMax axis keepDims x) s = (do
      let vx ← evalOp x s
      Tile.reduceMax axis keepDims vx) := by
  simp [evalOp]

@[simp] theorem evalOp_reduceSum
    (axis : Fin shape.length) (keepDims : Bool) (x : Op .real shape)
    (s : BlockState) :
    evalOp (.reduceSum axis keepDims x) s = (do
      let vx ← evalOp x s
      some (Tile.reduceSum axis keepDims vx)) := by
  simp [evalOp]

@[simp] theorem evalOp_dot
    (batch : TileShape) (x : Op .real (batch ++ [M, K]))
    (y : Op .real (batch ++ [K, N])) (s : BlockState) :
    evalOp (.dot (batch := batch) x y) s = (do
      let vx ← evalOp x s
      let vy ← evalOp y s
      some (Tile.dot batch vx vy)) := by
  simp [evalOp]

@[simp] theorem evalOp_dotInt
    (batch : TileShape) (x : Op .int (batch ++ [M, K]))
    (y : Op .int (batch ++ [K, N])) (s : BlockState) :
    evalOp (.dotInt (batch := batch) x y) s = (do
      let vx ← evalOp x s
      let vy ← evalOp y s
      some (Tile.dotInt batch vx vy)) := by
  simp [evalOp]

@[simp] theorem evalOp_transpose
    (batch : TileShape) (x : Op dtype (batch ++ [M, N])) (s : BlockState) :
    evalOp (.transpose (batch := batch) x) s = (do
      let vx ← evalOp x s
      some (Tile.transpose batch vx)) := by
  simp [evalOp]

@[simp] theorem evalOp_load_region_none {dtype : TileDType} {shape : TileShape}
    (region : Region dtype) (offsets : Op .nat shape) (s : BlockState) :
    evalOp (.load dtype (MemAccess.region region offsets) MaskOpt.none) s = (do
      let offs ← evalOp offsets s
      some ⟨fun i => s.readMemValue dtype (Region.cast region) (offs.data i)⟩) := by
  simp [evalOp]

@[simp 1100] theorem evalOp_ref_setReg_same (s : BlockState) (name : RegName)
    (dtype : TileDType) (shape : TileShape) (v : Tile dtype shape) :
    evalOp (.ref dtype shape name) (s.setReg name dtype shape v) = some v := by
  simp [evalOp]

@[simp 1100] theorem evalOp_ref_setReg (s : BlockState) (name name' : RegName)
    (dtype dtype' : TileDType) (shape shape' : TileShape) (v : Tile dtype shape) :
    evalOp (.ref dtype' shape' name') (s.setReg name dtype shape v) =
      if hName : name' = name then
        if hDType : dtype' = dtype then
          if hShape : shape' = shape then
            some (castTile hDType hShape v)
          else
            none
        else
          none
      else
        evalOp (.ref dtype' shape' name') s := by
  simp [evalOp, BlockState.setReg]

@[simp 1100] theorem evalOp_ref_setReg_ne_name (s : BlockState) (name name' : RegName)
    (dtype dtype' : TileDType) (shape shape' : TileShape) (v : Tile dtype shape)
    (h : name' ≠ name) :
    evalOp (.ref dtype' shape' name') (s.setReg name dtype shape v) =
      evalOp (.ref dtype' shape' name') s := by
  simp [evalOp, h]

@[simp 1100] theorem evalOp_ref_setReg_ne_dtype (s : BlockState) (name name' : RegName)
    (dtype dtype' : TileDType) (shape shape' : TileShape) (v : Tile dtype shape)
    (hName : name' = name) (h : dtype' ≠ dtype) :
    evalOp (.ref dtype' shape' name') (s.setReg name dtype shape v) = none := by
  simp [evalOp, hName, h]

@[simp 1100] theorem evalOp_ref_setReg_ne_shape (s : BlockState) (name name' : RegName)
    (dtype dtype' : TileDType) (shape shape' : TileShape) (v : Tile dtype shape)
    (hName : name' = name) (hDType : dtype' = dtype) (h : shape' ≠ shape) :
    evalOp (.ref dtype' shape' name') (s.setReg name dtype shape v) = none := by
  simp [evalOp, hName, hDType, h]

@[simp 1500] theorem evalOp_expandDim_ref_setReg_same
    (s : BlockState) (name : RegName)
    (dtype : TileDType) (shape : TileShape) (axis : Fin (shape.length + 1))
    (v : Tile dtype shape) :
    evalOp (.expandDim axis (.ref dtype shape name)) (s.setReg name dtype shape v) =
      some (Tile.expandDim axis v) := by
  simp

@[simp 1500] theorem evalOp_expandDim_ref_setReg
    (s : BlockState) (name name' : RegName)
    (dtype dtype' : TileDType) (shape shape' : TileShape)
    (axis : Fin (shape'.length + 1)) (v : Tile dtype shape) :
    evalOp (.expandDim axis (.ref dtype' shape' name')) (s.setReg name dtype shape v) =
      if hName : name' = name then
        if hDType : dtype' = dtype then
          if hShape : shape' = shape then
            some (Tile.expandDim axis (castTile hDType hShape v))
          else
            none
        else
          none
      else
        evalOp (.expandDim axis (.ref dtype' shape' name')) s := by
  by_cases hName : name' = name
  · subst hName
    by_cases hDType : dtype' = dtype
    · subst hDType
      by_cases hShape : shape' = shape
      · subst hShape
        simp [evalOp, BlockState.setReg]
      · simp [evalOp, BlockState.setReg, hShape]
    · simp [evalOp, BlockState.setReg, hDType]
  · simp [evalOp, BlockState.setReg, hName]

@[simp 1500] theorem evalOp_expandDim_ref_setReg_ne_name
    (s : BlockState) (name name' : RegName)
    (dtype dtype' : TileDType) (shape shape' : TileShape)
    (axis : Fin (shape'.length + 1)) (v : Tile dtype shape)
    (h : name' ≠ name) :
    evalOp (.expandDim axis (.ref dtype' shape' name')) (s.setReg name dtype shape v) =
      evalOp (.expandDim axis (.ref dtype' shape' name')) s := by
  simp [h]

@[simp 1500] theorem bind_evalOp_expandDim_ref_setReg_same
    {α : Type} (s : BlockState) (name : RegName)
    (dtype : TileDType) (shape : TileShape) (axis : Fin (shape.length + 1))
    (v : Tile dtype shape)
    (f : Tile dtype (TileShape.insertAxis shape axis 1) → Option α) :
    (evalOp (.expandDim axis (.ref dtype shape name)) (s.setReg name dtype shape v)).bind f =
      f (Tile.expandDim axis v) := by
  simp

@[simp 1500] theorem bind_evalOp_expandDim_ref_of_regs
    {α : Type} (s : BlockState) (name : RegName)
    (dtype : TileDType) (shape : TileShape) (axis : Fin (shape.length + 1))
    (v : Tile dtype shape)
    (f : Tile dtype (TileShape.insertAxis shape axis 1) → Option α)
    (h : s.regs dtype shape name = some v) :
    (evalOp (.expandDim axis (.ref dtype shape name)) s).bind f =
      f (Tile.expandDim axis v) := by
  simp [h]

@[simp 1500] theorem bind_evalOp_expandDim_ref_setReg_ne_name
    {α : Type} (s : BlockState) (name name' : RegName)
    (dtype dtype' : TileDType) (shape shape' : TileShape)
    (axis : Fin (shape'.length + 1)) (v : Tile dtype shape)
    (f : Tile dtype' (TileShape.insertAxis shape' axis 1) → Option α)
    (h : name' ≠ name) :
    (evalOp (.expandDim axis (.ref dtype' shape' name')) (s.setReg name dtype shape v)).bind f =
    (evalOp (.expandDim axis (.ref dtype' shape' name')) s).bind f := by
  simp [h]

@[simp 1500] theorem bind_bind_evalOp_expandDim_ref_setReg_same
    {α β : Type} (s : BlockState) (name : RegName)
    (dtype : TileDType) (shape : TileShape) (axis : Fin (shape.length + 1))
    (v : Tile dtype shape)
    (f : Tile dtype (TileShape.insertAxis shape axis 1) → Option α)
    (g : α → Option β) :
    ((evalOp (.expandDim axis (.ref dtype shape name)) (s.setReg name dtype shape v)).bind f).bind g =
      (f (Tile.expandDim axis v)).bind g := by
  simp

@[simp 1500] theorem bind_bind_bind_evalOp_expandDim_ref_setReg_same
    {α β γ : Type} (s : BlockState) (name : RegName)
    (dtype : TileDType) (shape : TileShape) (axis : Fin (shape.length + 1))
    (v : Tile dtype shape)
    (f : Tile dtype (TileShape.insertAxis shape axis 1) → Option α)
    (g : α → Option β) (k : β → Option γ) :
    (((evalOp (.expandDim axis (.ref dtype shape name)) (s.setReg name dtype shape v)).bind f).bind g).bind k =
      ((f (Tile.expandDim axis v)).bind g).bind k := by
  simp

@[simp 1500] theorem bind_bind_bind_bind_evalOp_expandDim_ref_setReg_same
    {α β γ δ : Type} (s : BlockState) (name : RegName)
    (dtype : TileDType) (shape : TileShape) (axis : Fin (shape.length + 1))
    (v : Tile dtype shape)
    (f : Tile dtype (TileShape.insertAxis shape axis 1) → Option α)
    (g : α → Option β) (k : β → Option γ) (l : γ → Option δ) :
    ((((evalOp (.expandDim axis (.ref dtype shape name)) (s.setReg name dtype shape v)).bind f).bind g).bind k).bind l =
      (((f (Tile.expandDim axis v)).bind g).bind k).bind l := by
  simp

@[simp 1500] theorem bind_bind_evalOp_expandDim_ref_of_regs
    {α β : Type} (s : BlockState) (name : RegName)
    (dtype : TileDType) (shape : TileShape) (axis : Fin (shape.length + 1))
    (v : Tile dtype shape)
    (f : Tile dtype (TileShape.insertAxis shape axis 1) → Option α)
    (g : α → Option β) (h : s.regs dtype shape name = some v) :
    ((evalOp (.expandDim axis (.ref dtype shape name)) s).bind f).bind g =
      (f (Tile.expandDim axis v)).bind g := by
  simp [h]

@[simp 1500] theorem bind_bind_bind_evalOp_expandDim_ref_of_regs
    {α β γ : Type} (s : BlockState) (name : RegName)
    (dtype : TileDType) (shape : TileShape) (axis : Fin (shape.length + 1))
    (v : Tile dtype shape)
    (f : Tile dtype (TileShape.insertAxis shape axis 1) → Option α)
    (g : α → Option β) (k : β → Option γ)
    (h : s.regs dtype shape name = some v) :
    (((evalOp (.expandDim axis (.ref dtype shape name)) s).bind f).bind g).bind k =
      ((f (Tile.expandDim axis v)).bind g).bind k := by
  simp [h]

@[simp 1500] theorem bind_bind_bind_bind_evalOp_expandDim_ref_of_regs
    {α β γ δ : Type} (s : BlockState) (name : RegName)
    (dtype : TileDType) (shape : TileShape) (axis : Fin (shape.length + 1))
    (v : Tile dtype shape)
    (f : Tile dtype (TileShape.insertAxis shape axis 1) → Option α)
    (g : α → Option β) (k : β → Option γ) (l : γ → Option δ)
    (h : s.regs dtype shape name = some v) :
    ((((evalOp (.expandDim axis (.ref dtype shape name)) s).bind f).bind g).bind k).bind l =
      (((f (Tile.expandDim axis v)).bind g).bind k).bind l := by
  simp [h]

@[simp 1500] theorem bind_bind_evalOp_expandDim_ref_setReg_ne_name
    {α β : Type} (s : BlockState) (name name' : RegName)
    (dtype dtype' : TileDType) (shape shape' : TileShape)
    (axis : Fin (shape'.length + 1)) (v : Tile dtype shape)
    (f : Tile dtype' (TileShape.insertAxis shape' axis 1) → Option α)
    (g : α → Option β) (h : name' ≠ name) :
    ((evalOp (.expandDim axis (.ref dtype' shape' name')) (s.setReg name dtype shape v)).bind f).bind g =
      ((evalOp (.expandDim axis (.ref dtype' shape' name')) s).bind f).bind g := by
  simp [h]

@[simp 1500] theorem bind_bind_bind_evalOp_expandDim_ref_setReg_ne_name
    {α β γ : Type} (s : BlockState) (name name' : RegName)
    (dtype dtype' : TileDType) (shape shape' : TileShape)
    (axis : Fin (shape'.length + 1)) (v : Tile dtype shape)
    (f : Tile dtype' (TileShape.insertAxis shape' axis 1) → Option α)
    (g : α → Option β) (k : β → Option γ) (h : name' ≠ name) :
    (((evalOp (.expandDim axis (.ref dtype' shape' name')) (s.setReg name dtype shape v)).bind f).bind g).bind k =
      (((evalOp (.expandDim axis (.ref dtype' shape' name')) s).bind f).bind g).bind k := by
  simp [h]

@[simp 1500] theorem bind_bind_bind_bind_evalOp_expandDim_ref_setReg_ne_name
    {α β γ δ : Type} (s : BlockState) (name name' : RegName)
    (dtype dtype' : TileDType) (shape shape' : TileShape)
    (axis : Fin (shape'.length + 1)) (v : Tile dtype shape)
    (f : Tile dtype' (TileShape.insertAxis shape' axis 1) → Option α)
    (g : α → Option β) (k : β → Option γ) (l : γ → Option δ)
    (h : name' ≠ name) :
    ((((evalOp (.expandDim axis (.ref dtype' shape' name')) (s.setReg name dtype shape v)).bind f).bind g).bind k).bind l =
      ((((evalOp (.expandDim axis (.ref dtype' shape' name')) s).bind f).bind g).bind k).bind l := by
  simp [h]

@[simp 1500] theorem match_evalOp_expandDim_ref_setReg_same
    {α : Type} (s : BlockState) (name : RegName)
    (dtype : TileDType) (shape : TileShape) (axis : Fin (shape.length + 1))
    (v : Tile dtype shape)
    (f : Tile dtype (TileShape.insertAxis shape axis 1) → Option α) :
    (match evalOp (.expandDim axis (.ref dtype shape name)) (s.setReg name dtype shape v), f with
      | none, _ => none
      | some a, f => f a) =
      f (Tile.expandDim axis v) := by
  simp

@[simp 1500] theorem match_evalOp_expandDim_ref_of_regs
    {α : Type} (s : BlockState) (name : RegName)
    (dtype : TileDType) (shape : TileShape) (axis : Fin (shape.length + 1))
    (v : Tile dtype shape)
    (f : Tile dtype (TileShape.insertAxis shape axis 1) → Option α)
    (h : s.regs dtype shape name = some v) :
    (match evalOp (.expandDim axis (.ref dtype shape name)) s, f with
      | none, _ => none
      | some a, f => f a) =
      f (Tile.expandDim axis v) := by
  simp [h]

@[simp 1500] theorem match_evalOp_expandDim_ref_setReg_ne_name
    {α : Type} (s : BlockState) (name name' : RegName)
    (dtype dtype' : TileDType) (shape shape' : TileShape)
    (axis : Fin (shape'.length + 1)) (v : Tile dtype shape)
    (f : Tile dtype' (TileShape.insertAxis shape' axis 1) → Option α)
    (h : name' ≠ name) :
    (match evalOp (.expandDim axis (.ref dtype' shape' name')) (s.setReg name dtype shape v), f with
      | none, _ => none
      | some a, f => f a) =
      (match evalOp (.expandDim axis (.ref dtype' shape' name')) s, f with
        | none, _ => none
        | some a, f => f a) := by
  simp [h]

end VeriTile.Triton
