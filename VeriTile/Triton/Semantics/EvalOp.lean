/-
VeriTile.Triton.Semantics.EvalOp

Expression evaluation for the typed Triton AST.
-/

import VeriTile.Triton.Semantics.TileOps
import Batteries.Data.Nat.Bitwise

namespace VeriTile.Triton

noncomputable def evalOp : Op dtype shape → BlockState → Option (Tile dtype shape)
  | .const c, _ => some (Tile.scalar (some c : WithBot ℝ))
  | .constFloat h c, _ => some (Tile.scalar (h.ofReal c))
  | .constNat n, _ => some (Tile.scalar n)
  | .constInt n, _ => some (Tile.scalar n)
  | .constBool b, _ => some (Tile.scalar b)
  | .negInf, _ => some (Tile.scalar (none : WithBot ℝ))
  | .programId axis, s => some (Tile.scalar (s.pids axis))
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
  | .reduceMax axis keepDims a, s => do
      let va ← evalOp a s
      Tile.reduceMax axis keepDims va
  | .reduceSum axis keepDims a, s => return Tile.reduceSum axis keepDims (← evalOp a s)
  | .scan op axis a, s => return Tile.scan op axis (← evalOp a s)
  | .argMax axis a, s => return Tile.argMaxDrop axis (← evalOp a s)
  | .argMin axis a, s => return Tile.argMinDrop axis (← evalOp a s)
  | .sort axis a, s => return Tile.sortAxis axis (← evalOp a s)
  | .dot (batch := batch) a b, s => do
      let va ← evalOp a s
      let vb ← evalOp b s
      some (Tile.dot batch va vb)
  | .expandDim axis a, s => return Tile.expandDim axis (← evalOp a s)
  | .where c a b, s => do
      let vc ← evalOp c s
      let va ← evalOp a s
      let vb ← evalOp b s
      some (Tile.select vc va vb)
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
  | .makeBlockPtr region baseOffset parentShape blockShape strides offsets, _ =>
      some ⟨fun _ =>
        { region := Region.cast region
        , baseOffset := baseOffset
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
              | dtype => BlockState.defaultCarrier dtype⟩
      | .maskOther mask other => do
          let masks ← evalOp mask s
          let others ← evalOp other s
          some ⟨fun i => if masks.data i then read i else others.data i⟩
  | .natToReal a, s => return Tile.natToReal (← evalOp a s)

end VeriTile.Triton
