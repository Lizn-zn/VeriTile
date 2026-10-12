import bench.tritonbench_g.bmm_chunk_fwd.Proofs.RealSequence
import bench.tritonbench_g.bmm_chunk_bwd.Proofs.RealStore

/-! Exact forward output addressing and full-chunk store mask. Live padding
in a shortened final chunk is written, rather than left untouched. The shared
store fold supplies tagged readback and preservation of every other cell. -/
namespace VeriTile.Bench.TritonBenchG.BmmChunkFwd.RealStore
open VeriTile.Triton
noncomputable section
set_option linter.unusedSimpArgs false

def basePointer (region : RegionName) (batch chunk head : Nat) : Op .ptr [] :=
  .ptrAdd .nil (.ptrBase region)
    (.add .nat .nil
      (.add .nat .nil (.mul .nat .nil (.ref .nat [] "pid_b") (.constNat batch))
        (.mul .nat .nil (.ref .nat [] "pid_c") (.constNat chunk)))
      (.mul .nat .nil (.ref .nat [] "pid_h") (.constNat head)))

def pointers (BM BN row column : Nat) : Op .ptr [BM, BN] :=
  .ptrAdd .scalarL (.ref .ptr [] "out_ptr")
    (.add .nat Broadcast.nil.consL.consR
      (.mul .nat .scalarL (.constNat row) (.expandDim ⟨1, by simp⟩ (.ref .nat [BM] "offs_m")))
      (.mul .nat .scalarR (.expandDim ⟨0, by simp⟩ (.ref .nat [BN] "offs_n")) (.constNat column)))

def mask (BM BN chunk : Nat) : Op .bool [BM, BN] :=
  .boolAnd Broadcast.nil.consL.consR
    (.lt .nat .scalarR (.expandDim ⟨1, by simp⟩ (.ref .nat [BM] "offs_m")) (.constNat chunk))
    (.lt .nat .scalarR (.expandDim ⟨0, by simp⟩ (.ref .nat [BN] "offs_n")) (.constNat chunk))

def code (BM BN chunk : Nat) : Stmt :=
  .store .real [BM, BN] (.ptr (.ref .ptr [BM, BN] "out_ptrs"))
    (.ref .real [BM, BN] "out") (.mask (mask BM BN chunk))

abbrev pointerValues := BmmChunkBwd.RealResidual.pointerValues
abbrev offset := BmmChunkBwd.RealStore.offset

def active (BM BN chunk : Nat) (rows : Fin BM → Nat)
    (columns : Fin BN → Nat) (i : TileIndex [BM, BN]) : Prop :=
  rows i.1 < chunk ∧ columns i.2.1 < chunk
instance (BM BN chunk : Nat) (rows : Fin BM → Nat) (columns : Fin BN → Nat) :
    DecidablePred (active BM BN chunk rows columns) := fun _ => inferInstanceAs (Decidable (_ ∧ _))

def write (s : BlockState) (region : RegionName) (BM BN base row column chunk : Nat)
    (rows : Fin BM → Nat) (columns : Fin BN → Nat) (values : Fin BM → Fin BN → ℝ) : BlockState :=
  BmmChunkBwd.RealStore.write s region BM BN base row column chunk chunk rows columns values

theorem base_eval (s : BlockState) (region : RegionName) (batch chunk head pb pc ph : Nat)
    (hb : s.regs .nat [] "pid_b" = some (Tile.scalar pb))
    (hc : s.regs .nat [] "pid_c" = some (Tile.scalar pc))
    (hh : s.regs .nat [] "pid_h" = some (Tile.scalar ph)) :
    evalOp (basePointer region batch chunk head) s =
      some (Tile.scalar (region, pb * batch + pc * chunk + ph * head)) := by
  simp [basePointer, evalOp.eq_def, hb, hc, hh, bind, Option.bind]
  apply Tile.ext
  intro i
  simp only [Tile.ptrAdd, Tile.bop, Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.add, NumericDType.mul, Nat.zero_add]

theorem pointers_eval (s : BlockState) (region : RegionName) (BM BN base row column : Nat)
    (rows : Fin BM → Nat) (columns : Fin BN → Nat)
    (hp : s.regs .ptr [] "out_ptr" = some (Tile.scalar (region, base)))
    (hr : s.regs .nat [BM] "offs_m" = some (Tile.vec rows))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec columns)) :
    evalOp (pointers BM BN row column) s =
      some (pointerValues region BM BN base row column rows columns) := by
  simp [pointers, evalOp.eq_def, hp, hr, hn, bind, Option.bind]
  apply Tile.ext
  rintro ⟨i,j,⟨⟩⟩
  simp [Tile.ptrAdd, Tile.bop, Tile.vec, Tile.scalar, Tile.mat, pointerValues,
    BmmChunkBwd.RealResidual.pointerValues, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.add, NumericDType.mul, Tile.expandDim, TileShape.dropInsertedIndex, Nat.mul_comm, -Tile.scalar_eta]

theorem mask_eval (s : BlockState) (BM BN chunk : Nat)
    (rows : Fin BM → Nat) (columns : Fin BN → Nat)
    (hr : s.regs .nat [BM] "offs_m" = some (Tile.vec rows))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec columns)) :
    evalOp (mask BM BN chunk) s =
      some (Tile.mat fun i j => decide (rows i < chunk ∧ columns j < chunk)) := by
  simp [mask, evalOp.eq_def, hr, hn, bind, Option.bind]
  apply Tile.ext
  rintro ⟨i,j,⟨⟩⟩
  simp only [Tile.bop, Tile.cop, Tile.expandDim, Tile.vec, Tile.mat, Tile.scalar,
    ComparableDType.lt, Broadcast.leftIndex, Broadcast.rightIndex, TileShape.dropInsertedIndex]
  rfl

theorem executes (s : BlockState) (region : RegionName) (BM BN base row column chunk : Nat)
    (rows : Fin BM → Nat) (columns : Fin BN → Nat) (values : Fin BM → Fin BN → ℝ)
    (hp : s.regs .ptr [BM, BN] "out_ptrs" = some (pointerValues region BM BN base row column rows columns))
    (hv : s.regs .real [BM, BN] "out" = some (Tile.mat fun i j => some (values i j)))
    (hr : s.regs .nat [BM] "offs_m" = some (Tile.vec rows))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec columns)) :
    stepStmt (code BM BN chunk) s =
      some (write s region BM BN base row column chunk rows columns values) := by
  have hm := mask_eval s BM BN chunk rows columns hr hn
  rw [code, stepStmt]
  simp only [evalOp_ref, hp, hv, hm, bind, Option.bind_some,
    pointerValues, BmmChunkBwd.RealResidual.pointerValues, Tile.mat,
    Option.map_some, decide_eq_true_eq, BlockState.writeMemTyped_real]
  rfl

theorem frame (s : BlockState) (region : RegionName) (BM BN base row column chunk : Nat)
    (rows : Fin BM → Nat) (columns : Fin BN → Nat) (values : Fin BM → Fin BN → ℝ)
    (r : RegionName) (address : Nat)
    (h : ∀ i, active BM BN chunk rows columns i →
      ¬ (region = r ∧ offset BM BN base row column rows columns i = address)) :
    (write s region BM BN base row column chunk rows columns values).mem r address = s.mem r address :=
  BmmChunkBwd.RealStore.frame s region BM BN base row column chunk chunk rows columns values r address h

theorem readback (s : BlockState) (region : RegionName) (BM BN base row column chunk : Nat)
    (rows : Fin BM → Nat) (columns : Fin BN → Nat) (values : Fin BM → Fin BN → ℝ)
    (i : TileIndex [BM, BN]) (hi : active BM BN chunk rows columns i)
    (hinj : ∀ a b, active BM BN chunk rows columns a → active BM BN chunk rows columns b →
      offset BM BN base row column rows columns a = offset BM BN base row column rows columns b → a = b) :
    (write s region BM BN base row column chunk rows columns values).mem region
      (offset BM BN base row column rows columns i) = MemCell.real (values i.1 i.2.1) :=
  BmmChunkBwd.RealStore.readback s region BM BN base row column chunk chunk rows columns values i hi hinj

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BmmChunkFwd.RealStore
