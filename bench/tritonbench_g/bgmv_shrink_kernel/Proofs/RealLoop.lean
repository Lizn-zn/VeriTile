import bench.tritonbench_g.bgmv_shrink_kernel.Proofs.RealFoundation
import VeriTile.Meta.StatementAudit

/-! Loop arithmetic with an explicit selected slot. The complete signed
metadata prefix must establish that slot and the invariant; this module
never substitutes natural metadata for the actual source load. -/
namespace VeriTile.Bench.TritonBenchG.BgmvShrinkKernel.RealLoop
open VeriTile.Triton
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

/-- Input element `x[k] = input[cur_batch·xm_stride + k]` (upstream addresses
the input row directly by `current_k`; its `xk_stride` argument is unused). -/
noncomputable def aElem (s : BlockState) (input_ptr : RegionName)
    (xm_stride k : Nat) : ℝ :=
  s.readMem input_ptr (s.pids 1 * xm_stride + k)

/-- LoRA-A element
`W[n,k] = lora[l0_stride·lora_index + n·lora_k_stride + k·lora_n_stride]`. -/
noncomputable def bElem (s : BlockState) (lora_ptr : RegionName)
    (slot : Nat)
    (l0_stride lora_k_stride lora_n_stride n k : Nat) : ℝ :=
  s.readMem lora_ptr
    (l0_stride * slot + n * lora_k_stride
      + k * lora_n_stride)

/-- The kernel's `current_k` rank index at loop iteration `c`, lane `e`:
`k(c,e) = c·(BLOCK_K·SPLIT_K) + (e + pid_sk·BLOCK_K)` — this program's
(`pid_sk`) slice of the rank dimension. -/
def kIdx (s : BlockState) (BLOCK_K SPLIT_K c e : Nat) : Nat :=
  c * (BLOCK_K * SPLIT_K) + (e + s.pids 0 * BLOCK_K)

/-- One K-split block's masked contribution at output lane `n`:
`Σ_{e<BLOCK_K} [k(c,e) < K] · x[k(c,e)] · W[n,k(c,e)]` (the `tl.sum` of the
masked `tiled_a · tiled_b` products; out-of-range rank lanes load `0`). -/
noncomputable def blockTerm (s : BlockState) (input_ptr lora_ptr : RegionName)
    (slot : Nat)
    (K xm_stride l0_stride lora_k_stride lora_n_stride
      BLOCK_K SPLIT_K c n : Nat) : ℝ :=
  (Finset.univ : Finset (Fin BLOCK_K)).sum fun e =>
    if kIdx s BLOCK_K SPLIT_K c e.val < K then
      aElem s input_ptr xm_stride (kIdx s BLOCK_K SPLIT_K c e.val)
        * bElem s lora_ptr slot l0_stride lora_k_stride lora_n_stride
            n (kIdx s BLOCK_K SPLIT_K c e.val)
    else 0

/-- Partial accumulator after `c` K-split blocks at lane `n`. -/
noncomputable def accPartial (s : BlockState) (input_ptr lora_ptr : RegionName)
    (slot : Nat)
    (K xm_stride l0_stride lora_k_stride lora_n_stride
      BLOCK_K SPLIT_K c n : Nat) : ℝ :=
  (Finset.range c).sum fun c' =>
    blockTerm s input_ptr lora_ptr slot K xm_stride l0_stride
      lora_k_stride lora_n_stride BLOCK_K SPLIT_K c' n

/-- One-block recurrence: `accPartial (c+1) = accPartial c + blockTerm c`. -/
theorem accPartial_succ (s : BlockState) (input_ptr lora_ptr : RegionName)
    (slot : Nat)
    (K xm_stride l0_stride lora_k_stride lora_n_stride
      BLOCK_K SPLIT_K c n : Nat) :
    accPartial s input_ptr lora_ptr slot K xm_stride l0_stride
        lora_k_stride lora_n_stride BLOCK_K SPLIT_K (c + 1) n
      = accPartial s input_ptr lora_ptr slot K xm_stride l0_stride
          lora_k_stride lora_n_stride BLOCK_K SPLIT_K c n
        + blockTerm s input_ptr lora_ptr slot K xm_stride l0_stride
            lora_k_stride lora_n_stride BLOCK_K SPLIT_K c n := by
  unfold accPartial
  rw [Finset.sum_range_succ]

/-- Trip count of `for k in range(0, K, S)`: `⌈K/S⌉`. -/
def numKIters (K S : Nat) : Nat := (K + S - 1) / S

/-- This program's scaled rank-slice contribution at output
lane `n` — `scaling · Σ_{c<⌈K/S⌉} Σ_{e<BLOCK_K} [k(c,e)<K] · x[k(c,e)] ·
W[n,k(c,e)]`, a closed form over INPUT memory. -/
noncomputable def scaledContribution (s : BlockState) (input_ptr lora_ptr : RegionName)
    (slot : Nat) (scaling : ℝ)
    (K xm_stride l0_stride lora_k_stride lora_n_stride
      BLOCK_K SPLIT_K n : Nat) : ℝ :=
  scaling * accPartial s input_ptr lora_ptr slot K xm_stride l0_stride
    lora_k_stride lora_n_stride BLOCK_K SPLIT_K
    (numKIters K (BLOCK_K * SPLIT_K)) n


/-- **K-loop invariant** (counter `i = c · BLOCK_K·SPLIT_K`). After `c`
K-split blocks: pids fixed, the prologue registers seeded, memory untouched,
and `accumulator` holds the partial rank-slice sum `accPartial c` at active
lanes (`n < N`; inactive lanes hold `0` — their `tiled_b` loads are fully
masked). -/
noncomputable def shrinkInv
    (input_ptr lora_ptr : RegionName) (slot : Nat)
    (N K xm_stride l0_stride lora_k_stride lora_n_stride
      BLOCK_N BLOCK_K SPLIT_K : Nat) (s0 : BlockState)
    (i : Nat) (s : BlockState) : Prop :=
  let c := i / (BLOCK_K * SPLIT_K)
  s.pids = s0.pids ∧ i = c * (BLOCK_K * SPLIT_K) ∧
  i < K + BLOCK_K * SPLIT_K ∧
  s.regs .nat [] "cur_batch" = some (Tile.scalar (s0.pids 1)) ∧
  s.regs .nat [BLOCK_N] "offset_n"
    = some (Tile.vec (fun j : Fin BLOCK_N => j.val)) ∧
  s.regs .nat [BLOCK_K] "offset_k"
    = some (Tile.vec (fun e : Fin BLOCK_K => e.val + s0.pids 0 * BLOCK_K)) ∧
  s.regs .ptr [] "a_ptr"
    = some (Tile.scalar ((input_ptr, s0.pids 1 * xm_stride) : RegionName × Nat)) ∧
  s.regs .ptr [] "b_ptr"
    = some (Tile.scalar
        ((lora_ptr, l0_stride * slot) : RegionName × Nat)) ∧
  s.regs .real [BLOCK_N] "accumulator"
    = some ⟨fun idx : TileIndex [BLOCK_N] =>
        some (if idx.1.val < N then
          accPartial s0 input_ptr lora_ptr slot K xm_stride l0_stride
            lora_k_stride lora_n_stride BLOCK_K SPLIT_K c idx.1.val
        else 0)⟩ ∧
  s.mem = s0.mem


set_option maxHeartbeats 4000000 in
/-- **Step**: one K-split loop iteration advances the invariant by one block —
the masked `tl.sum` adds `blockTerm c` to the partial accumulator at every
active lane (and `0` at masked lanes). -/
theorem shrink_step
    (input_ptr lora_ptr : RegionName) (slot : Nat)
    (N K xm_stride l0_stride lora_k_stride lora_n_stride
      BLOCK_N BLOCK_K SPLIT_K : Nat) (s0 : BlockState)
    (hS : 0 < BLOCK_K * SPLIT_K)
    (i : Nat) (s : BlockState) (hilt : i < K)
    (hinv : shrinkInv input_ptr lora_ptr slot N K xm_stride l0_stride
        lora_k_stride lora_n_stride BLOCK_N BLOCK_K SPLIT_K s0 i s) :
    ∃ s', stepStmts (shrinkLoopBody N K lora_k_stride lora_n_stride BLOCK_N BLOCK_K)
        (s.setReg "k" .nat [] (Tile.scalar i)) = some s'
      ∧ shrinkInv input_ptr lora_ptr slot N K xm_stride l0_stride
          lora_k_stride lora_n_stride BLOCK_N BLOCK_K SPLIT_K s0
          (i + BLOCK_K * SPLIT_K) s' := by
  obtain ⟨hpids, hi, hbound, hcb, hon, hok, hap, hbp, hacc, hmem⟩ := hinv
  have hc1 : (i + BLOCK_K * SPLIT_K) / (BLOCK_K * SPLIT_K)
      = i / (BLOCK_K * SPLIT_K) + 1 := Nat.add_div_right i hS
  set c := i / (BLOCK_K * SPLIT_K) with hcdef
  set ckF : Fin BLOCK_K → Nat :=
    fun e => i + (e.val + s0.pids 0 * BLOCK_K) with hckF
  set sk := s.setReg "k" .nat [] (Tile.scalar i) with hsk
  have hskpids : sk.pids = s0.pids := by rw [hsk, BlockState.setReg_pids, hpids]
  have hskmem : sk.mem = s0.mem := by rw [hsk]; exact hmem
  have hskrm : ∀ R o, sk.readMem R o = s0.readMem R o := by
    intro R o; unfold BlockState.readMem; rw [hskmem]
  have hk_sk : sk.regs .nat [] "k" = some (Tile.scalar i) := by
    rw [hsk, BlockState.setReg_same]
  have hcb_sk : sk.regs .nat [] "cur_batch" = some (Tile.scalar (s0.pids 1)) := by
    rw [hsk, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("cur_batch":RegName) ≠ "k" by decide)]
    exact hcb
  have hon_sk : sk.regs .nat [BLOCK_N] "offset_n"
      = some (Tile.vec (fun j : Fin BLOCK_N => j.val)) := by
    rw [hsk, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offset_n":RegName) ≠ "k" by decide)]
    exact hon
  have hok_sk : sk.regs .nat [BLOCK_K] "offset_k"
      = some (Tile.vec (fun e : Fin BLOCK_K => e.val + s0.pids 0 * BLOCK_K)) := by
    rw [hsk, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offset_k":RegName) ≠ "k" by decide)]
    exact hok
  have hap_sk : sk.regs .ptr [] "a_ptr"
      = some (Tile.scalar ((input_ptr, s0.pids 1 * xm_stride) : RegionName × Nat)) := by
    rw [hsk, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("a_ptr":RegName) ≠ "k" by decide)]
    exact hap
  have hbp_sk : sk.regs .ptr [] "b_ptr"
      = some (Tile.scalar
          ((lora_ptr, l0_stride * slot) : RegionName × Nat)) := by
    rw [hsk, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("b_ptr":RegName) ≠ "k" by decide)]
    exact hbp
  have hacc_sk : sk.regs .real [BLOCK_N] "accumulator"
      = some ⟨fun idx : TileIndex [BLOCK_N] =>
          some (if idx.1.val < N then
            accPartial s0 input_ptr lora_ptr slot K xm_stride l0_stride
              lora_k_stride lora_n_stride BLOCK_K SPLIT_K c idx.1.val
          else 0)⟩ := by
    rw [hsk, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("accumulator":RegName) ≠ "k" by decide)]
    exact hacc
  -- step current_k
  unfold shrinkLoopBody
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (currentK_eval BLOCK_K i (fun e : Fin BLOCK_K => e.val + s0.pids 0 * BLOCK_K)
          sk hk_sk hok_sk))]
  set s1 := sk.setReg "current_k" .nat [BLOCK_K] (Tile.vec ckF) with hs1
  have hck1 : s1.regs .nat [BLOCK_K] "current_k" = some (Tile.vec ckF) := by
    rw [hs1, BlockState.setReg_same]
  -- step current_k_c (max_contiguous hint erased: plain copy)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (show evalOp (Op.ref .nat [BLOCK_K] "current_k") s1 = some (Tile.vec ckF) by
          rw [evalOp_ref]; simp [hck1]))]
  set s2 := s1.setReg "current_k_c" .nat [BLOCK_K] (Tile.vec ckF) with hs2
  have hckc2 : s2.regs .nat [BLOCK_K] "current_k_c" = some (Tile.vec ckF) := by
    rw [hs2, BlockState.setReg_same]
  have hck2 : s2.regs .nat [BLOCK_K] "current_k" = some (Tile.vec ckF) := by
    rw [hs2, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("current_k":RegName) ≠ "current_k_c" by decide)]
    exact hck1
  have hap2 : s2.regs .ptr [] "a_ptr"
      = some (Tile.scalar ((input_ptr, s0.pids 1 * xm_stride) : RegionName × Nat)) := by
    rw [hs2, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("a_ptr":RegName) ≠ "current_k_c" by decide),
      hs1, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("a_ptr":RegName) ≠ "current_k" by decide)]
    exact hap_sk
  have hrm2 : ∀ R o, s2.readMem R o = s0.readMem R o := by
    intro R o
    rw [hs2, BlockState.setReg_readMem, hs1, BlockState.setReg_readMem]
    exact hskrm R o
  -- step tiled_a
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (tiledA_eval input_ptr K BLOCK_K (s0.pids 1 * xm_stride) ckF s0 s2
          hap2 hckc2 hck2 (fun o => hrm2 input_ptr o)))]
  set taT : Tile .real [BLOCK_K] :=
    ⟨fun e : TileIndex [BLOCK_K] =>
      some (if ckF e.1 < K then s0.readMem input_ptr (s0.pids 1 * xm_stride + ckF e.1)
        else 0)⟩ with htaT
  set s3 := s2.setReg "tiled_a" .real [BLOCK_K] taT with hs3
  have hon3 : s3.regs .nat [BLOCK_N] "offset_n"
      = some (Tile.vec (fun j : Fin BLOCK_N => j.val)) := by
    rw [hs3, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offset_n":RegName) ≠ "tiled_a" by decide),
      hs2, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offset_n":RegName) ≠ "current_k_c" by decide),
      hs1, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offset_n":RegName) ≠ "current_k" by decide)]
    exact hon_sk
  have hck3 : s3.regs .nat [BLOCK_K] "current_k" = some (Tile.vec ckF) := by
    rw [hs3, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("current_k":RegName) ≠ "tiled_a" by decide)]
    exact hck2
  -- step b_ptr_mask
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (bmask_eval N K BLOCK_N BLOCK_K ckF s3 hon3 hck3))]
  set bmT : Tile .bool [BLOCK_N, BLOCK_K] :=
    ⟨fun idx : TileIndex [BLOCK_N, BLOCK_K] =>
      decide (idx.1.val < N) && decide (ckF idx.2.1 < K)⟩ with hbmT
  set s4 := s3.setReg "b_ptr_mask" .bool [BLOCK_N, BLOCK_K] bmT with hs4
  have hbp4 : s4.regs .ptr [] "b_ptr"
      = some (Tile.scalar
          ((lora_ptr, l0_stride * slot) : RegionName × Nat)) := by
    rw [hs4, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("b_ptr":RegName) ≠ "b_ptr_mask" by decide),
      hs3, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("b_ptr":RegName) ≠ "tiled_a" by decide),
      hs2, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("b_ptr":RegName) ≠ "current_k_c" by decide),
      hs1, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("b_ptr":RegName) ≠ "current_k" by decide)]
    exact hbp_sk
  have hon4 : s4.regs .nat [BLOCK_N] "offset_n"
      = some (Tile.vec (fun j : Fin BLOCK_N => j.val)) := by
    rw [hs4, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offset_n":RegName) ≠ "b_ptr_mask" by decide)]
    exact hon3
  have hck4 : s4.regs .nat [BLOCK_K] "current_k" = some (Tile.vec ckF) := by
    rw [hs4, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("current_k":RegName) ≠ "b_ptr_mask" by decide)]
    exact hck3
  have hbm4 : s4.regs .bool [BLOCK_N, BLOCK_K] "b_ptr_mask" = some bmT := by
    rw [hs4, BlockState.setReg_same]
  have hrm4 : ∀ R o, s4.readMem R o = s0.readMem R o := by
    intro R o
    rw [hs4, BlockState.setReg_readMem, hs3, BlockState.setReg_readMem]
    exact hrm2 R o
  -- step tiled_b
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (tiledB_eval lora_ptr N K lora_k_stride lora_n_stride BLOCK_N BLOCK_K
          (l0_stride * slot) ckF s0 s4 hbp4 hon4 hck4
          (by rw [hbm4, hbmT]) (fun o => hrm4 lora_ptr o)))]
  set tbT : Tile .real [BLOCK_N, BLOCK_K] :=
    ⟨fun idx : TileIndex [BLOCK_N, BLOCK_K] =>
      some (if idx.1.val < N ∧ ckF idx.2.1 < K then
        s0.readMem lora_ptr
          (l0_stride * slot + idx.1.val * lora_k_stride
            + ckF idx.2.1 * lora_n_stride)
      else 0)⟩ with htbT
  set s5 := s4.setReg "tiled_b" .real [BLOCK_N, BLOCK_K] tbT with hs5
  have hta5 : s5.regs .real [BLOCK_K] "tiled_a" = some taT := by
    rw [hs5, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("tiled_a":RegName) ≠ "tiled_b" by decide),
      hs4, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("tiled_a":RegName) ≠ "b_ptr_mask" by decide),
      hs3, BlockState.setReg_same]
  have htb5 : s5.regs .real [BLOCK_N, BLOCK_K] "tiled_b" = some tbT := by
    rw [hs5, BlockState.setReg_same]
  have hacc5 : s5.regs .real [BLOCK_N] "accumulator"
      = some ⟨fun idx : TileIndex [BLOCK_N] =>
          some (if idx.1.val < N then
            accPartial s0 input_ptr lora_ptr slot K xm_stride l0_stride
              lora_k_stride lora_n_stride BLOCK_K SPLIT_K c idx.1.val
          else 0)⟩ := by
    rw [hs5, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("accumulator":RegName) ≠ "tiled_b" by decide),
      hs4, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("accumulator":RegName) ≠ "b_ptr_mask" by decide),
      hs3, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("accumulator":RegName) ≠ "tiled_a" by decide),
      hs2, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("accumulator":RegName) ≠ "current_k_c" by decide),
      hs1, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("accumulator":RegName) ≠ "current_k" by decide)]
    exact hacc_sk
  -- step accumulator
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (accadd_eval BLOCK_N BLOCK_K _ taT tbT s5 hacc5 hta5 htb5))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  simp only [shrinkInv, hc1]
  refine ⟨?_, by rw [hi]; ring, by omega, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · -- pids
    simp only [hs5, hs4, hs3, hs2, hs1, BlockState.setReg_pids]
    exact hskpids
  · -- cur_batch
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("cur_batch":RegName) ≠ "accumulator" by decide),
      hs5, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("cur_batch":RegName) ≠ "tiled_b" by decide),
      hs4, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("cur_batch":RegName) ≠ "b_ptr_mask" by decide),
      hs3, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("cur_batch":RegName) ≠ "tiled_a" by decide),
      hs2, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("cur_batch":RegName) ≠ "current_k_c" by decide),
      hs1, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("cur_batch":RegName) ≠ "current_k" by decide)]
    exact hcb_sk
  · -- offset_n
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offset_n":RegName) ≠ "accumulator" by decide),
      hs5, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offset_n":RegName) ≠ "tiled_b" by decide)]
    exact hon4
  · -- offset_k
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offset_k":RegName) ≠ "accumulator" by decide),
      hs5, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offset_k":RegName) ≠ "tiled_b" by decide),
      hs4, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offset_k":RegName) ≠ "b_ptr_mask" by decide),
      hs3, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offset_k":RegName) ≠ "tiled_a" by decide),
      hs2, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offset_k":RegName) ≠ "current_k_c" by decide),
      hs1, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("offset_k":RegName) ≠ "current_k" by decide)]
    exact hok_sk
  · -- a_ptr
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("a_ptr":RegName) ≠ "accumulator" by decide),
      hs5, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("a_ptr":RegName) ≠ "tiled_b" by decide),
      hs4, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("a_ptr":RegName) ≠ "b_ptr_mask" by decide),
      hs3, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("a_ptr":RegName) ≠ "tiled_a" by decide)]
    exact hap2
  · -- b_ptr
    simp only [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("b_ptr":RegName) ≠ "accumulator" by decide),
      hs5, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (show ("b_ptr":RegName) ≠ "tiled_b" by decide)]
    exact hbp4
  · -- accumulator advanced by one block
    rw [BlockState.setReg_same]
    apply congrArg some
    apply Tile.ext
    intro idx
    have hlane := reduceSum_lane N K BLOCK_N BLOCK_K ckF
      (fun k => s0.readMem input_ptr (s0.pids 1 * xm_stride + k))
      (fun n k => s0.readMem lora_ptr
        (l0_stride * slot + n * lora_k_stride
          + k * lora_n_stride)) idx.1
    simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex]
    rw [htaT, htbT]
    erw [hlane]
    have hkIdx : ∀ e : Fin BLOCK_K, ckF e = kIdx s0 BLOCK_K SPLIT_K c e.val := by
      intro e
      simp only [hckF, kIdx]
      omega
    by_cases hn : idx.1.val < N
    · simp only [hn, if_true, NumericDType.add, WithBot.realAdd, Option.map₂,
        Option.bind, Option.map]
      apply congrArg some
      rw [accPartial_succ]
      congr 1
      unfold blockTerm
      refine Finset.sum_congr rfl (fun e _ => ?_)
      rw [hkIdx e]
      rfl
    · simp only [hn, if_false, NumericDType.add, WithBot.realAdd, Option.map₂,
        Option.bind, Option.map]
      apply congrArg some
      norm_num
  · -- mem
    show _ = s0.mem
    simp only [hs5, hs4, hs3, hs2, hs1, BlockState.setReg_mem]
    exact hskmem


#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.BgmvShrinkKernel.RealLoop
