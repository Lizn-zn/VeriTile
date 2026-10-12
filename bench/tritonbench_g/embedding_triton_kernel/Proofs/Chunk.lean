import bench.tritonbench_g.embedding_triton_kernel.Proofs.Reference
import VeriTile.Triton
namespace VeriTile.Bench.TritonBenchG.EmbeddingTritonKernel.Chunk
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 20000

def code (weight ids out : RegionName) (first last : Int) (sw so n h D N NN : Nat) : List Stmt :=
  (Erasure.source Bool.false weight ids out first last sw so n h D N NN).toAlgKernel.body

def prefixCode (weight ids out : RegionName) (first last : Int) (sw so n h D N NN : Nat) : List Stmt :=
  (code weight ids out first last sw so n h D N NN).take 3

def body (weight ids out : RegionName) (first last : Int) (sw so n h D N NN : Nat) : List Stmt :=
  match (code weight ids out first last sw so n h D N NN).drop 3 with
  | [.forRange _ _ _ _ statements] => statements
  | _ => []

/-- The chunk is extracted from the complete source, and its position/loop
bounds are checked here rather than assumed by later execution proofs. -/
theorem source_decomposition (weight ids out : RegionName) (first last : Int) (sw so n h D N NN : Nat) :
    code weight ids out first last sw so n h D N NN =
      prefixCode weight ids out first last sw so n h D N NN ++
      [.forRange "start_nn" 0 N NN (body weight ids out first last sw so n h D N NN)] := by
  rfl

structure Context (u : BlockState) (N NN D : Nat) : Prop where
  rows : u.regs .nat [NN] "offs_nn"=some ⟨fun j => u.pid*N+j.1.val⟩
  columns : u.regs .nat [D] "offs_d"=some ⟨fun j => j.1.val⟩

def row (u : BlockState) (N start : Nat) (j : TileIndex [NN,D]) : Nat := start+(u.pid*N+j.1.val)

def address (u : BlockState) (so N start : Nat) (j : TileIndex [NN,D]) : Nat := row u N start j*so+j.2.1.val

def Active (u : BlockState) (n h N start : Nat) (j : TileIndex [NN,D]) : Prop := row u N start j<n ∧ j.2.1.val<h

/-- Full tagged-cell frame for one actual source iteration. -/
theorem frame (weight ids out : RegionName) (first last : Int) (sw so n h D N NN start : Nat)
    (u v : BlockState) (hc : Context u N NN D)
    (he : stepStmts (body weight ids out first last sw so n h D N NN)
      (u.setReg "start_nn" .nat [] (Tile.scalar start))=some v)
    (r : RegionName) (o : Nat)
    (ho : ∀ j : TileIndex [NN,D], Active u n h N start j → ¬ (out=r ∧ address u so N start j=o)) :
    v.mem r o=u.mem r o := by
  rcases hc with ⟨hr,hc⟩
  simp [body,code,Erasure.source,ComputeKernel.surfaceBody,ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,Except.bind,bind,Option.bind,
    stepStmts,stepStmt,evalOp.eq_def,Tile.bop,Tile.cop,Tile.expandDim,TileShape.dropInsertedIndex,
    NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt,ComparableDType.ge,
    Bool.and_eq_true,hr,hc] at he
  subst v
  refine Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_) rfl
  intro j _ hj hwhere
  exact ho j (by simpa [Active,row] using hj) (by simpa [address,row] using hwhere)

/-- One store-active lane reads the independent signed-ID lookup formula.
Only collisions with other active lanes matter; padded columns may overlap. -/
theorem readback (weight ids out : RegionName) (first last : Int) (sw so n h D N NN start : Nat)
    (u v : BlockState) (hc : Context u N NN D)
    (he : stepStmts (body weight ids out first last sw so n h D N NN)
      (u.setReg "start_nn" .nat [] (Tile.scalar start))=some v)
    (j : TileIndex [NN,D]) (hj : Active u n h N start j)
    (hinj : ∀ k : TileIndex [NN,D], Active u n h N start k →
      address u so N start k=address u so N start j → k=j) :
    v.readMem out (address u so N start j)=Reference.value u weight ids first last sw (row u N start j) j.2.1.val := by
  rcases hc with ⟨hr,hc⟩
  simp [body,code,Erasure.source,ComputeKernel.surfaceBody,ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,Except.bind,bind,Option.bind,
    stepStmts,stepStmt,evalOp.eq_def,Tile.bop,Tile.cop,Tile.expandDim,TileShape.dropInsertedIndex,
    NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt,ComparableDType.ge,
    Bool.and_eq_true,hr,hc] at he
  subst v
  unfold address row
  rw [BlockState.scatter_readback_prop_masked_nd_of_true _ _ _ _ j
    (by simpa [Active,row] using hj) (by simpa [Active,address,row] using hinj)]
  rcases hj with ⟨hn,hd⟩
  simp only [row] at hn
  simp only [Reference.value,Reference.token]
  simp [hn,hd]
  split_ifs with hv
  · rw [Int.toNat_add (mul_nonneg (sub_nonneg.mpr hv.1) (Int.natCast_nonneg _)) (Int.natCast_nonneg _)]
    rw [Int.toNat_mul (sub_nonneg.mpr hv.1) (Int.natCast_nonneg _)]
    simp
  · norm_num

/-- Every register read in a chunk is provided by the prefix or the chunk
itself. This is successful complete execution, not a conditional readback. -/
theorem executes (weight ids out : RegionName) (first last : Int) (sw so n h D N NN start : Nat)
    (u : BlockState) (hc : Context u N NN D) :
    ∃ v, stepStmts (body weight ids out first last sw so n h D N NN)
      (u.setReg "start_nn" .nat [] (Tile.scalar start))=some v := by
  rcases hc with ⟨hr,hc⟩
  simp [body,code,Erasure.source,ComputeKernel.surfaceBody,ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,Except.bind,bind,Option.bind,
    stepStmts,stepStmt,evalOp.eq_def,Tile.bop,Tile.cop,Tile.expandDim,TileShape.dropInsertedIndex,
    NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt,ComparableDType.ge,
    Bool.and_eq_true,hr,hc]

theorem preserves_context (weight ids out : RegionName) (first last : Int) (sw so n h D N NN start : Nat)
    (u v : BlockState) (hc : Context u N NN D)
    (he : stepStmts (body weight ids out first last sw so n h D N NN)
      (u.setReg "start_nn" .nat [] (Tile.scalar start))=some v) :
    Context v N NN D ∧ v.pids=u.pids := by
  rcases hc with ⟨hr,hc⟩
  simp [body,code,Erasure.source,ComputeKernel.surfaceBody,ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,Except.bind,bind,Option.bind,
    stepStmts,stepStmt,evalOp.eq_def,Tile.bop,Tile.cop,Tile.expandDim,TileShape.dropInsertedIndex,
    NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt,ComparableDType.ge,
    Bool.and_eq_true,hr,hc] at he
  subst v
  constructor
  · constructor
    · simp [BlockState.foldl_writeMem_prop_masked_pid,BlockState.setReg,hr]
    · simp [BlockState.setReg,hc]
  · simp [BlockState.foldl_writeMem_prop_masked_pids]

/-- The complete three-statement prefix initializes the persistent lane
registers from any input register file and does not write memory. -/
theorem prefix_executes (weight ids out : RegionName) (first last : Int) (sw so n h D N NN : Nat)
    (u : BlockState) :
    ∃ v, stepStmts (prefixCode weight ids out first last sw so n h D N NN) u=some v ∧
      Context v N NN D ∧ v.pids=u.pids ∧ v.mem=u.mem := by
  simp [prefixCode,code,Erasure.source,ComputeKernel.surfaceBody,ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,Except.bind,bind,Option.bind,
    stepStmts,stepStmt,evalOp.eq_def,Tile.bop,NumericDType.add,NumericDType.mul]
  constructor
  · constructor
    · simp [BlockState.setReg]
    · simp only [BlockState.setReg]
      rfl
  · simp [BlockState.setReg]

theorem flatten_ok (weight ids out : RegionName) (first last : Int) (sw so n h D N NN : Nat) :
    (Erasure.source Bool.false weight ids out first last sw so n h D N NN).toAlgKernel.FlattenOk := by
  unfold Kernel.FlattenOk
  simp [Erasure.source,ComputeKernel.surfaceBody,ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,Except.bind,bind,
    StmtList.FlattenOk,Stmt.FlattenOk,Op.FlattenOk]
  simp [Op.FlattenOk.eq_def]

/-- A sufficient physical row stride gives exactly the active-lane collision
condition needed by the chunk readback theorem. -/
theorem no_collision (u : BlockState) (so n h N start : Nat) (j k : TileIndex [NN,D])
    (hs : min D h≤so) (hj : Active u n h N start j) (hk : Active u n h N start k)
    (he : address u so N start j=address u so N start k) : j=k := by
  obtain ⟨hr,hd⟩ := Reference.row_major_injective so D h (row u N start j) (row u N start k)
    j.2.1 k.2.1 hs hj.2 hk.2 he
  apply Prod.ext
  · apply Fin.ext
    simp only [row] at hr
    omega
  · exact Prod.ext hd (Subsingleton.elim _ _)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.EmbeddingTritonKernel.Chunk
