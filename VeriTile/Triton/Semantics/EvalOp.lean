/-
VeriTile.Triton.Semantics.EvalOp

Expression evaluation for the typed Triton AST.
-/

import VeriTile.Triton.Semantics.TileOps

namespace VeriTile.Triton

noncomputable def evalOp : Op dtype shape → BlockState → Option (Tile dtype shape)
  | .const c, _ => some (Tile.scalar (some c : WithBot ℝ))
  | .constFloat h c, _ => some (Tile.scalar (h.ofReal c))
  | .constNat n, _ => some (Tile.scalar n)
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
  | .exp a, s => return Tile.uop WithBot.realExp (← evalOp a s)
  | .log a, s => return Tile.uop WithBot.realLog (← evalOp a s)
  | .sigmoid a, s => return Tile.uop WithBot.realSigmoid (← evalOp a s)
  | .sqrt a, s => return Tile.uop WithBot.realSqrt (← evalOp a s)
  | .tanh a, s => return Tile.uop WithBot.realTanh (← evalOp a s)
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
  | .ptrBase region, _ => some (Tile.scalar (region, 0))
  | .ptrAdd bc ptr off, s => do
      let ptrs ← evalOp ptr s
      let offs ← evalOp off s
      some (Tile.ptrAdd bc ptrs offs)
  | .load region off, s => do
      let offsets ← evalOp off s
      some ⟨fun i => some (s.readMem region (offsets.data i))⟩
  | .loadMask region off mask, s => do
      let offsets ← evalOp off s
      let masks ← evalOp mask s
      some ⟨fun i =>
        let addr := offsets.data i
        if masks.data i then some (s.readMem region addr) else some (s.undef region addr)⟩
  | .loadMaskOther region off mask other, s => do
      let offsets ← evalOp off s
      let masks ← evalOp mask s
      let others ← evalOp other s
      some ⟨fun i =>
        let addr := offsets.data i
        if masks.data i then some (s.readMem region addr) else others.data i⟩
  | .loadPtr ptr, s => do
      let ptrs ← evalOp ptr s
      some ⟨fun i =>
        let p := ptrs.data i
        some (s.readMem p.1 p.2)⟩
  | .loadPtrMask ptr mask, s => do
      let ptrs ← evalOp ptr s
      let masks ← evalOp mask s
      some ⟨fun i =>
        let p := ptrs.data i
        if masks.data i then some (s.readMem p.1 p.2) else some (s.undef p.1 p.2)⟩
  | .loadPtrMaskOther ptr mask other, s => do
      let ptrs ← evalOp ptr s
      let masks ← evalOp mask s
      let others ← evalOp other s
      some ⟨fun i =>
        let p := ptrs.data i
        if masks.data i then some (s.readMem p.1 p.2) else others.data i⟩
  | .loadFloat h region off, s => do
      let offsets ← evalOp off s
      some ⟨fun i => h.ofReal (s.readMem region (offsets.data i))⟩
  | .loadFloatMask h region off mask, s => do
      let offsets ← evalOp off s
      let masks ← evalOp mask s
      some ⟨fun i =>
        let addr := offsets.data i
        if masks.data i then h.ofReal (s.readMem region addr) else h.ofReal (s.undef region addr)⟩
  | .loadFloatMaskOther h region off mask other, s => do
      let offsets ← evalOp off s
      let masks ← evalOp mask s
      let others ← evalOp other s
      some ⟨fun i =>
        let addr := offsets.data i
        if masks.data i then h.ofReal (s.readMem region addr) else others.data i⟩
  | .loadPtrFloat h ptr, s => do
      let ptrs ← evalOp ptr s
      some ⟨fun i =>
        let p := ptrs.data i
        h.ofReal (s.readMem p.1 p.2)⟩
  | .loadPtrFloatMask h ptr mask, s => do
      let ptrs ← evalOp ptr s
      let masks ← evalOp mask s
      some ⟨fun i =>
        let p := ptrs.data i
        if masks.data i then h.ofReal (s.readMem p.1 p.2) else h.ofReal (s.undef p.1 p.2)⟩
  | .loadPtrFloatMaskOther h ptr mask other, s => do
      let ptrs ← evalOp ptr s
      let masks ← evalOp mask s
      let others ← evalOp other s
      some ⟨fun i =>
        let p := ptrs.data i
        if masks.data i then h.ofReal (s.readMem p.1 p.2) else others.data i⟩
  | .natToReal a, s => return Tile.natToReal (← evalOp a s)

end VeriTile.Triton
