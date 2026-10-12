import bench.tritonbench_g.chunk_gated_attention.Proofs.RealHExecution
import VeriTile.Triton.Memory.TraceProofs

/-! Allocation bounds for the exact checked addresses. Inactive lanes do not
access memory, whatever their unspecified value. Endpoint vectors are checked
against their flattened T*K or T*V parent, independently of two-dimensional
feature bounds; neither time clipping nor contiguous input strides are assumed. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGatedAttention.RealHAccessSafety
open VeriTile.Triton RealHMath RealHProgram RealHPointers
noncomputable section
set_option maxHeartbeats 400000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

def MatrixBounds (r : RegionName) (base rows cols nr nc sr sc offR offC : Nat)
    (bounds : RegionBounds) : Prop :=
  ∀ (i : Fin nr) (j : Fin nc), offR+i.val < rows ∧ offC+j.val < cols →
    base+(offR+i.val)*sr+(offC+j.val)*sc < bounds r

def VectorBounds (r : RegionName) (base size n stride off : Nat) (bounds : RegionBounds) : Prop :=
  ∀ i : Fin n, off+i.val < size → base+(off+i.val)*stride < bounds r

def KeyBounds (p : Layout) (l : Launch) (r : RegionName) (c : Nat) :=
  MatrixBounds r (l.head*p.keyHead) p.K p.T p.BK p.BT p.keyFeature p.keyTime (l.key*p.BK) (c*p.BT)
def ValueBounds (p : Layout) (l : Launch) (r : RegionName) (c : Nat) :=
  MatrixBounds r (l.head*p.valueHead) p.T p.V p.BT p.BV p.valueTime p.valueFeature (c*p.BT) (l.value*p.BV)
def StateBounds (p : Layout) (l : Launch) (r : RegionName) :=
  MatrixBounds r (l.head*p.K*p.V) p.K p.V p.BK p.BV p.V 1 (l.key*p.BK) (l.value*p.BV)
def HistoryBounds (p : Layout) (l : Launch) (r : RegionName) (c : Nat) :=
  MatrixBounds r (l.head*p.historyHead+c*p.K*p.V) p.K p.V p.BK p.BV p.historyKey p.historyValue (l.key*p.BK) (l.value*p.BV)
def EndpointBounds (p : Layout) (l : Launch) (r : RegionName)
    (c head stride size block start : Nat) :=
  VectorBounds r (l.head*head) (p.T*size) block stride ((c*p.BT+p.BT-1)*size+start)

theorem load2_safe (r : RegionName) (base rows cols nr nc sr sc offR offC : Nat)
    (name : RegName) (bounds : RegionBounds) (u : BlockState)
    (hb : MatrixBounds r base rows cols nr nc sr sc offR offC bounds)
    (hp : u.regs .blockPtr [nr,nc] name = some (matrixPointer r base rows cols nr nc sr sc offR offC)) :
    (load2 nr nc name).SafeAt bounds u := by
  rw [load2,Op.SafeAt.eq_def]
  refine ⟨by simp [Op.SafeAt],trivial,?_⟩
  refine ⟨by simp [Op.MemorySafe],?_⟩
  intro ps he idx _
  rw [evalOp_ref,hp] at he
  cases he
  rcases idx with ⟨i,j,⟨⟩⟩
  simpa only [matrixPointer,TileShape.indexToList,BlockPtr.inBounds_2d_offsets,
    BlockPtr.address_2d_offsets,decide_eq_true_eq] using hb i j

theorem load1_safe (r : RegionName) (base size n stride off : Nat)
    (name : RegName) (bounds : RegionBounds) (u : BlockState)
    (hb : VectorBounds r base size n stride off bounds)
    (hp : u.regs .blockPtr [n] name = some (vectorPointer r base size n stride off)) :
    (load1 n name).SafeAt bounds u := by
  rw [load1,Op.SafeAt.eq_def]
  refine ⟨by simp [Op.SafeAt],trivial,?_⟩
  refine ⟨by simp [Op.MemorySafe],?_⟩
  intro ps he idx _
  rw [evalOp_ref,hp] at he
  cases he
  rcases idx with ⟨i,⟨⟩⟩
  simpa only [vectorPointer,TileShape.indexToList,BlockPtr.inBounds_1d,
    BlockPtr.address_1d,decide_eq_true_eq] using hb i

theorem store2_safe (r : RegionName) (base rows cols nr nc sr sc offR offC : Nat)
    (name value : RegName) (bounds : RegionBounds) (u : BlockState)
    (hb : MatrixBounds r base rows cols nr nc sr sc offR offC bounds)
    (hp : u.regs .blockPtr [nr,nc] name = some (matrixPointer r base rows cols nr nc sr sc offR offC)) :
    Stmt.TraceSafe bounds (.store .real [nr,nc] (.blockPtr (.ref .blockPtr [nr,nc] name) [0,1])
      (.ref .real [nr,nc] value) .none) u := by
  rw [Stmt.TraceSafe]
  refine ⟨by simp [MemAccess.SafeAt,Op.SafeAt],by simp [Op.SafeAt],trivial,?_⟩
  refine ⟨by simp [Op.MemorySafe],?_⟩
  intro ps he idx _
  rw [evalOp_ref,hp] at he
  cases he
  rcases idx with ⟨i,j,⟨⟩⟩
  simpa only [matrixPointer,TileShape.indexToList,BlockPtr.inBounds_2d_offsets,
    BlockPtr.address_2d_offsets,decide_eq_true_eq] using hb i j

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGatedAttention.RealHAccessSafety
