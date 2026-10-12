import bench.tritonbench_g.chunk_gated_attention.Proofs.RealHProgram

/-! Pointer evaluation and unmasked checked loads for the complete H source.
Both row and column strides remain symbolic. Loads preserve arbitrary real
unspecified values; the zero-extension premise is applied later, separately. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGatedAttention.RealHPointers
open VeriTile.Triton RealHMath RealHProgram
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 14000
set_option linter.unusedSimpArgs false

def matrixPointer (r : RegionName) (base rows cols nr nc sr sc offR offC : Nat) : Tile .blockPtr [nr,nc] :=
  ⟨fun _ => ⟨r,base,[rows,cols],[nr,nc],[sr,sc],[offR,offC]⟩⟩

def keyPointer (p : Layout) (l : Launch) (r : RegionName) (c : Nat) :=
  matrixPointer r (l.head*p.keyHead) p.K p.T p.BK p.BT p.keyFeature p.keyTime (l.key*p.BK) (c*p.BT)
def valuePointer (p : Layout) (l : Launch) (r : RegionName) (c : Nat) :=
  matrixPointer r (l.head*p.valueHead) p.T p.V p.BT p.BV p.valueTime p.valueFeature (c*p.BT) (l.value*p.BV)
def statePointer (p : Layout) (l : Launch) (r : RegionName) :=
  matrixPointer r (l.head*p.K*p.V) p.K p.V p.BK p.BV p.V 1 (l.key*p.BK) (l.value*p.BV)
def historyPointer (p : Layout) (l : Launch) (r : RegionName) (c : Nat) :=
  matrixPointer r (l.head*p.historyHead+c*p.K*p.V) p.K p.V p.BK p.BV p.historyKey p.historyValue (l.key*p.BK) (l.value*p.BV)

def vectorPointer (r : RegionName) (base size n stride off : Nat) : Tile .blockPtr [n] :=
  ⟨fun _ => ⟨r,base,[size],[n],[stride],[off]⟩⟩
def endpointPointer (p : Layout) (l : Launch) (r : RegionName)
    (c head stride size block start : Nat) :=
  vectorPointer r (l.head*head) (p.T*size) block stride ((c*p.BT+p.BT-1)*size+start)

theorem times_eval (u : BlockState) (name : RegName) (v n : Nat)
    (h : u.regs .nat [] name = some (Tile.scalar v)) :
    evalOp (times name n) u = some (Tile.scalar (v*n)) := by
  simp only [times,evalOp_mul,evalOp_ref,evalOp_constNat,h,Option.bind_eq_bind,Option.bind_some]
  rfl

theorem make2 (u : BlockState) (r : RegionName) (base x y : Op .nat [])
    (rows cols nr nc sr sc b i j : Nat)
    (hb : evalOp base u = some (Tile.scalar b))
    (hi : evalOp x u = some (Tile.scalar i)) (hj : evalOp y u = some (Tile.scalar j)) :
    evalOp (.makeBlockPtrDynOffsets r base [rows,cols] [nr,nc] [sr,sc] [x,y]) u =
      some (matrixPointer r b rows cols nr nc sr sc i j) := by
  simp only [makeBlockPtr2_eval,hb,hi,hj,List.mapM_cons,List.mapM_nil,Option.bind_eq_bind,
    Option.bind_some,Tile.scalar]
  rfl

theorem key_eval (p : Layout) (l : Launch) (r : RegionName) (c : Nat) (u : BlockState)
    (hh : u.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (hk : u.regs .nat [] "i_k" = some (Tile.scalar l.key))
    (hc : u.regs .nat [] "i_t" = some (Tile.scalar c)) :
    evalOp (keyPtr p r) u = some (keyPointer p l r c) :=
  make2 u r _ _ _ _ _ _ _ _ _ _ _ _ (times_eval u _ _ _ hh) (times_eval u _ _ _ hk) (times_eval u _ _ _ hc)

theorem value_eval (p : Layout) (l : Launch) (r : RegionName) (c : Nat) (u : BlockState)
    (hh : u.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (hv : u.regs .nat [] "i_v" = some (Tile.scalar l.value))
    (hc : u.regs .nat [] "i_t" = some (Tile.scalar c)) :
    evalOp (valuePtr p r) u = some (valuePointer p l r c) :=
  make2 u r _ _ _ _ _ _ _ _ _ _ _ _ (times_eval u _ _ _ hh) (times_eval u _ _ _ hc) (times_eval u _ _ _ hv)

theorem stateBase_eval (p : Layout) (l : Launch) (u : BlockState)
    (hh : u.regs .nat [] "i_bh" = some (Tile.scalar l.head)) :
    evalOp (stateBase p) u = some (Tile.scalar (l.head*p.K*p.V)) := by
  rw [stateBase,evalOp_mul,times_eval u _ _ _ hh,evalOp_constNat]
  rfl

theorem state_eval (p : Layout) (l : Launch) (r : RegionName) (u : BlockState)
    (hh : u.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (hk : u.regs .nat [] "i_k" = some (Tile.scalar l.key))
    (hv : u.regs .nat [] "i_v" = some (Tile.scalar l.value)) :
    evalOp (statePtr p r) u = some (statePointer p l r) :=
  make2 u r _ _ _ _ _ _ _ _ _ _ _ _ (stateBase_eval p l u hh) (times_eval u _ _ _ hk) (times_eval u _ _ _ hv)

theorem history_eval (p : Layout) (l : Launch) (r : RegionName) (c : Nat) (u : BlockState)
    (hh : u.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (hk : u.regs .nat [] "i_k" = some (Tile.scalar l.key))
    (hv : u.regs .nat [] "i_v" = some (Tile.scalar l.value))
    (hc : u.regs .nat [] "i_t" = some (Tile.scalar c)) :
    evalOp (historyPtr p r) u = some (historyPointer p l r c) := by
  apply make2
  · simp only [evalOp_add,evalOp_mul,evalOp_ref,evalOp_constNat,times,hh,hc,Option.bind_eq_bind,Option.bind_some]
    rfl
  · exact times_eval u _ _ _ hk
  · exact times_eval u _ _ _ hv

theorem endpointOp_eval (p : Layout) (c size block indexValue : Nat) (index : RegName) (u : BlockState)
    (hc : u.regs .nat [] "i_t" = some (Tile.scalar c))
    (hi : u.regs .nat [] index = some (Tile.scalar indexValue)) :
    evalOp (endpointOp p size block index) u =
      some (Tile.scalar ((c*p.BT+p.BT-1)*size+indexValue*block)) := by
  simp only [endpointOp,times,evalOp_add,evalOp_mul,evalOp_sub,evalOp_ref,evalOp_constNat,
    hc,hi,Option.bind_eq_bind,Option.bind_some]
  rfl

theorem endpoint_eval (p : Layout) (l : Launch) (r : RegionName)
    (c head stride size block indexValue : Nat) (index : RegName) (u : BlockState)
    (hh : u.regs .nat [] "i_bh" = some (Tile.scalar l.head))
    (hc : u.regs .nat [] "i_t" = some (Tile.scalar c))
    (hi : u.regs .nat [] index = some (Tile.scalar indexValue)) :
    evalOp (endpointPtr p r head stride size block index) u =
      some (endpointPointer p l r c head stride size block (indexValue*block)) := by
  simp only [endpointPtr,makeBlockPtr2_eval,times_eval u _ _ _ hh,
    endpointOp_eval p c size block indexValue index u hc hi,List.mapM_cons,List.mapM_nil,
    Option.bind_eq_bind,Option.bind_some,Tile.scalar]
  rfl

/-- No padding value is assumed here: each inactive lane reads its oracle at
that lane's actual address. -/
theorem load2_eval (r : RegionName) (u : BlockState) (name : RegName)
    (base rows cols nr nc sr sc offR offC : Nat)
    (hp : u.regs .blockPtr [nr,nc] name = some (matrixPointer r base rows cols nr nc sr sc offR offC)) :
    evalOp (load2 nr nc name) u = some (Tile.mat (fun (i : Fin nr) (j : Fin nc) =>
      if offR+i.val < rows ∧ offC+j.val < cols then
        u.readMem r (base+(offR+i.val)*sr+(offC+j.val)*sc)
      else u.undef r (base+(offR+i.val)*sr+(offC+j.val)*sc))) := by
  simp only [load2,evalOp,hp,Option.bind]
  apply congrArg some
  apply Tile.ext
  rintro ⟨i,j,⟨⟩⟩
  simp only [matrixPointer,Tile.mat,TileShape.indexToList,BlockPtr.address_2d_offsets,
    BlockPtr.inBounds_2d_offsets,BlockState.readMemValue_real]
  by_cases h : offR+i.val < rows ∧ offC+j.val < cols
  · simp only [h,decide_true,if_true]
    rfl
  · simp [h,BlockState.paddingValue,BlockState.unspecifiedValue]
    rfl

theorem load1_eval (r : RegionName) (u : BlockState) (name : RegName)
    (base size n stride off : Nat)
    (hp : u.regs .blockPtr [n] name = some (vectorPointer r base size n stride off)) :
    evalOp (load1 n name) u = some (Tile.vec (fun i : Fin n =>
      if off+i.val < size then u.readMem r (base+(off+i.val)*stride)
      else u.undef r (base+(off+i.val)*stride))) := by
  simp only [load1,evalOp,hp,Option.bind]
  apply congrArg some
  apply Tile.ext
  rintro ⟨i,⟨⟩⟩
  simp only [vectorPointer,Tile.vec,TileShape.indexToList,BlockPtr.address_1d,
    BlockPtr.inBounds_1d,BlockState.readMemValue_real]
  by_cases h : off+i.val < size
  · simp only [h,decide_true,if_true]
    rfl
  · simp [h,BlockState.paddingValue,BlockState.unspecifiedValue]
    rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGatedAttention.RealHPointers
