# Spec sheet — `bench/tritonbench_g/block_sparse_attn/BlockSparseAttn.lean`

**Python source:** `bench/tritonbench_g/block_sparse_attn/block_sparse_attn.py`

## Public theorem: `block_sparse_attn_output_closed_form_summary_general`

<details><summary>docstring</summary>

```
/-- **General (symbolic-dimension) public summary.** At *arbitrary* tile dims and
strides (`NUM_D_BLOCKS = 2`, `EVEN_M = EVEN_N = true`), the full faithful
`block_sparse_attention_kernel` surface (prologue + the CSR `forRangeDyn`
online-softmax loop + the two masked `out` stores) **realizes the genuine causal
block-sparse softmax closed form** `blockSparseAttnClosedForm` at every active
output lane, for both D-blocks (`dBlockBase = 0` / `BLOCK_D`).

This is the symbolic-dimension generalization of the former case-1 Python-shape
summary (the instance `BLOCK_M=BLOCK_N=BLOCK_D=16`, `num_heads=4`,
`num_kv_heads=2`, strides `2048/512/32/1024`, `num_layout=1`, row/col CSR strides
`3`/`4`, `total_seq_len=16`).

Honest side-conditions: clean undef (`hundef`); the CSR row-pointer window
(`hStartL`/`hEndL`/`hsle`) with a nonempty window (`hN`); positivity of the block
size (`hBN`); output-offset injectivity (`hinj`) and the two stores hitting
disjoint offsets (`hdisj`); first-key causal visibility (`hVis0`); and the
per-block selection / load-alignment advance (`hstep`, the trusted host boundary).
This is **not** the self-referential executed value: the streaming
`m_i`/`l_i`/`acc`/`acc2` recurrence is unfolded statement-by-statement
(`bsa_execG`) and proven to collapse to the closed form
(`bsa_streaming_eq_closedFormG`). -/
```
</details>

**Statement:**
```lean
theorem block_sparse_attn_output_closed_form_summary_general
    (Out Q K V : RegionName) (R C : Region .nat) (s : BlockState)
    (BLOCK_M BLOCK_D BLOCK_N num_heads num_kv_heads num_layout total_seq_len : Nat)
    (rowStrideH colStrideH sqb sqh sqm skb skh skn svb svh svn sob soh som : Nat) (scale : ℝ)
    (hundef : ∀ rg o, s.undef rg o = 0)
    (hBN : 0 < BLOCK_N)
    (start_l end_l : Nat)
    (hStartL : s.readMemValue .nat R.cast
      (s.pids 1 % num_heads % num_layout * rowStrideH + s.pids 0) = start_l)
    (hEndL : s.readMemValue .nat R.cast
      (s.pids 1 % num_heads % num_layout * rowStrideH + s.pids 0 + 1) = end_l)
    (hsle : start_l ≤ end_l) (hN : 0 < end_l - start_l)
    (hinj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_D] => outOffset s num_heads sob soh som BLOCK_M idx))
    (hdisj : ∀ a b : TileIndex [BLOCK_M, BLOCK_D],
      outOffset s num_heads sob soh som BLOCK_M a
        ≠ out2Offset s num_heads sob soh som BLOCK_M BLOCK_D b)
    (hVis0 : ∀ idx : TileIndex [BLOCK_M, BLOCK_D], active s total_seq_len BLOCK_M idx →
      selKeyGlobal s C (s.pids 1 % num_heads % num_layout) num_heads start_l BLOCK_N 0
        ≤ s.pids 0 * BLOCK_M + idx.1.val)
    (hstep : ∀ (i : Nat) (st : BlockState), start_l ≤ i → i < end_l →
      bsaInvariantG Out Q K V R C BLOCK_M BLOCK_D BLOCK_N num_heads num_kv_heads num_layout
          total_seq_len sqb sqh sqm skb skh skn svb svh svn (s.pids 0 * BLOCK_M) (end_l - start_l)
          (fun r : Fin (BLOCK_N * (end_l - start_l)) =>
            selKeyGlobal s C (s.pids 1 % num_heads % num_layout) num_heads start_l BLOCK_N r.val)
          (fun jx : TileIndex [BLOCK_M, 2 * BLOCK_D] =>
            qTileBSA s Q num_heads sqb sqh sqm BLOCK_M jx.1 jx.2.1.val)
          (fun jx : TileIndex [BLOCK_N * (end_l - start_l), 2 * BLOCK_D] =>
            kRowBSA s K num_heads num_kv_heads skb skh skn
              (selKeyGlobal s C (s.pids 1 % num_heads % num_layout) num_heads start_l BLOCK_N jx.1.val) jx.2.1.val)
          (fun jx : TileIndex [BLOCK_N * (end_l - start_l), BLOCK_D] =>
            vRowBSA s V num_heads num_kv_heads svb svh svn
              (selKeyGlobal s C (s.pids 1 % num_heads % num_layout) num_heads start_l BLOCK_N jx.1.val) (0 + jx.2.1.val))
          (fun jx : TileIndex [BLOCK_N * (end_l - start_l), BLOCK_D] =>
            vRowBSA s V num_heads num_kv_heads svb svh svn
              (selKeyGlobal s C (s.pids 1 % num_heads % num_layout) num_heads start_l BLOCK_N jx.1.val) (BLOCK_D + jx.2.1.val))
          scale s (i - start_l) st →
      ∃ st', stepStmts (bsaLoopBodyG C BLOCK_M BLOCK_D BLOCK_N num_heads colStrideH skn svn total_seq_len scale)
          (st.setReg "col_idx_idx" .nat [] (Tile.scalar i)) = some st'
        ∧ bsaInvariantG Out Q K V R C BLOCK_M BLOCK_D BLOCK_N num_heads num_kv_heads num_layout
            total_seq_len sqb sqh sqm skb skh skn svb svh svn (s.pids 0 * BLOCK_M) (end_l - start_l)
            (fun r : Fin (BLOCK_N * (end_l - start_l)) =>
              selKeyGlobal s C (s.pids 1 % num_heads % num_layout) num_heads start_l BLOCK_N r.val)
            (fun jx : TileIndex [BLOCK_M, 2 * BLOCK_D] =>
              qTileBSA s Q num_heads sqb sqh sqm BLOCK_M jx.1 jx.2.1.val)
            (fun jx : TileIndex [BLOCK_N * (end_l - start_l), 2 * BLOCK_D] =>
              kRowBSA s K num_heads num_kv_heads skb skh skn
                (selKeyGlobal s C (s.pids 1 % num_heads % num_layout) num_heads start_l BLOCK_N jx.1.val) jx.2.1.val)
            (fun jx : TileIndex [BLOCK_N * (end_l - start_l), BLOCK_D] =>
              vRowBSA s V num_heads num_kv_heads svb svh svn
                (selKeyGlobal s C (s.pids 1 % num_heads % num_layout) num_heads start_l BLOCK_N jx.1.val) (0 + jx.2.1.val))
            (fun jx : TileIndex [BLOCK_N * (end_l - start_l), BLOCK_D] =>
              vRowBSA s V num_heads num_kv_heads svb svh svn
                (selKeyGlobal s C (s.pids 1 % num_heads % num_layout) num_heads start_l BLOCK_N jx.1.val) (BLOCK_D + jx.2.1.val))
            scale s (i - start_l + 1) st') :
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := block_sparse_attention_kernel Out Q K V R C
        rowStrideH colStrideH num_layout scale sqb sqh sqm skb skh skn svb svh svn sob soh som
        num_heads num_kv_heads total_seq_len BLOCK_M BLOCK_N BLOCK_D 2 Bool.true Bool.true)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK_M, BLOCK_D] => active s total_seq_len BLOCK_M idx)
        (fun idx : TileIndex [BLOCK_M, BLOCK_D] => (Out, outOffset s num_heads sob soh som BLOCK_M idx)))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_D] =>
        blockSparseAttnClosedForm s Q K V C num_heads num_kv_heads sqb sqh sqm skb skh skn svb svh svn
          (s.pids 1 % num_heads % num_layout) num_heads start_l (end_l - start_l) (2 * BLOCK_D) BLOCK_M BLOCK_N
          0 scale idx.1 idx.2.1.val)) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := block_sparse_attention_kernel Out Q K V R C
        rowStrideH colStrideH num_layout scale sqb sqh sqm skb skh skn svb svh svn sob soh som
        num_heads num_kv_heads total_seq_len BLOCK_M BLOCK_N BLOCK_D 2 Bool.true Bool.true)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK_M, BLOCK_D] => active s total_seq_len BLOCK_M idx)
        (fun idx : TileIndex [BLOCK_M, BLOCK_D] => (Out, out2Offset s num_heads sob soh som BLOCK_M BLOCK_D idx)))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_D] =>
        blockSparseAttnClosedForm s Q K V C num_heads num_kv_heads sqb sqh sqm skb skh skn svb svh svn
          (s.pids 1 % num_heads % num_layout) num_heads start_l (end_l - start_l) (2 * BLOCK_D) BLOCK_M BLOCK_N
          BLOCK_D scale idx.1 idx.2.1.val))
```

**Assumptions / layout contracts:**
- `hundef : ∀ rg o, s.undef rg o = 0`
- `hBN : 0 < BLOCK_N`
- `hStartL : s.readMemValue .nat R.cast
      (s.pids 1 % num_heads % num_layout * rowStrideH + s.pids 0) = start_l`
- `hEndL : s.readMemValue .nat R.cast
      (s.pids 1 % num_heads % num_layout * rowStrideH + s.pids 0 + 1) = end_l`
- `hsle : start_l ≤ end_l`
- `hN : 0 < end_l - start_l`
- `hinj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_D] => outOffset s num_heads sob soh som BLOCK_M idx)`
- `hdisj : ∀ a b : TileIndex [BLOCK_M, BLOCK_D],
      outOffset s num_heads sob soh som BLOCK_M a
        ≠ out2Offset s num_heads sob soh som BLOCK_M BLOCK_D b`
- `hVis0 : ∀ idx : TileIndex [BLOCK_M, BLOCK_D], active s total_seq_len BLOCK_M idx →
      selKeyGlobal s C (s.pids 1 % num_heads % num_layout) num_heads start_l BLOCK_N 0
        ≤ s.pids 0 * BLOCK_M + idx.1.val`
- `fun jx : TileIndex [BLOCK_M, 2 * BLOCK_D] =>
            qTileBSA s Q num_heads sqb sqh sqm BLOCK_M jx.1 jx.2.1.val`
- `fun jx : TileIndex [BLOCK_M, 2 * BLOCK_D] =>
              qTileBSA s Q num_heads sqb sqh sqm BLOCK_M jx.1 jx.2.1.val`
- `fun idx : TileIndex [BLOCK_M, BLOCK_D] => active s total_seq_len BLOCK_M idx`
- `fun idx : TileIndex [BLOCK_M, BLOCK_D] => (Out, outOffset s num_heads sob soh som BLOCK_M idx)`
- `fun idx : TileIndex [BLOCK_M, BLOCK_D] => active s total_seq_len BLOCK_M idx`
- `fun idx : TileIndex [BLOCK_M, BLOCK_D] => (Out, out2Offset s num_heads sob soh som BLOCK_M BLOCK_D idx)`

**Closed-form spec defs (transitive):** `outOffset`, `out2Offset`, `active`, `selKeyGlobal`, `bsaInvariantG`, `qTileBSA`, `kRowBSA`, `vRowBSA`, `bsaLoopBodyG`, `block_sparse_attention_kernel`, `blockSparseAttnClosedForm`, `offB`, `offH`, `mIndex`, `dIndex`, `bsaMPartial`, `bsaLPartial`, `bsaOPartial`, `qBase`, `kvBase`, `rawScoreBSA`, `maskedScore`, `offHkv`, `gScore`, `headGroups`

<details><summary><code>outOffset</code></summary>

```lean
def outOffset
    (s : BlockState)
    (num_heads stride_ob stride_oh stride_om BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_D]) : Nat :=
  offB s num_heads * stride_ob + offH s num_heads * stride_oh +
    mIndex s BLOCK_M idx.1 * stride_om + dIndex idx
```
</details>

<details><summary><code>out2Offset</code></summary>

```
/-- Output offset for the second block-sparse store. -/
```
```lean
def out2Offset
    (s : BlockState)
    (num_heads stride_ob stride_oh stride_om BLOCK_M BLOCK_D : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_D]) : Nat :=
  offB s num_heads * stride_ob + offH s num_heads * stride_oh +
    mIndex s BLOCK_M idx.1 * stride_om + BLOCK_D + dIndex idx
```
</details>

<details><summary><code>active</code></summary>

```lean
def active
    (s : BlockState) (total_seq_len BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_D]) : Prop :=
  mIndex s BLOCK_M idx.1 < total_seq_len
```
</details>

<details><summary><code>selKeyGlobal</code></summary>

```
/-- Global key position of the `r`-th flattened selected key:
`col_idx(r / BLOCK_N) · BLOCK_N + (r % BLOCK_N)`, where `col_idx(b)` is the
`b`-th CSR column index visited by the program, read from
`layout_csr_col_indices` at `layout_h · stride_col + start_l + b`. -/
```
```lean
def selKeyGlobal
    (s : BlockState) (layoutCols : Region .nat)
    (layout_h stride_col start_l BLOCK_N : Nat) (r : Nat) : Nat :=
  s.readMemValue .nat (Region.cast layoutCols)
      (layout_h * stride_col + start_l + r / BLOCK_N) * BLOCK_N +
    r % BLOCK_N
```
</details>

<details><summary><code>bsaInvariantG</code></summary>

```
/-- General (symbolic-dim) CSR loop invariant after `c` selected `BLOCK_N`-blocks.
Mirrors `bsaInvariant` with every literal replaced by a symbolic dim/stride. The
math accumulators carry `M = BLOCK_M`, contraction `D = 2·BLOCK_D` (the two q/k
D-blocks), value dim `Dv = BLOCK_D`, block size `Bk = BLOCK_N`. -/
```
```lean
noncomputable def bsaInvariantG
    (Out Q K V : RegionName) (R C : Region .nat)
    (BLOCK_M BLOCK_D BLOCK_N num_heads num_kv_heads num_layout total_seq_len : Nat)
    (sqb sqh sqm skb skh skn svb svh svn : Nat)
    (qStart numKVBlocks : Nat) (gpos : Fin (BLOCK_N * numKVBlocks) → Nat)
    (Qg : TileIndex [BLOCK_M, 2 * BLOCK_D] → ℝ)
    (Kg : TileIndex [BLOCK_N * numKVBlocks, 2 * BLOCK_D] → ℝ)
    (Vg Vg2 : TileIndex [BLOCK_N * numKVBlocks, BLOCK_D] → ℝ) (scale : ℝ)
    (s0 : BlockState) (c : Nat) (s : BlockState) : Prop :=
  s.pids = s0.pids
  ∧ s.mem = s0.mem
  ∧ (∀ rg o, s.undef rg o = 0)
  ∧ s.regs .nat [] "q_seq_len" = some (Tile.scalar total_seq_len)
  ∧ s.regs .nat [] "start_m" = some (Tile.scalar (s0.pids 0))
  ∧ s.regs .nat [] "off_bh" = some (Tile.scalar (s0.pids 1))
  ∧ s.regs .nat [] "off_h" = some (Tile.scalar (s0.pids 1 % num_heads))
  ∧ s.regs .nat [] "off_b" = some (Tile.scalar (s0.pids 1 / num_heads))
  ∧ s.regs .nat [] "head_groups" = some (Tile.scalar (num_heads / num_kv_heads))
  ∧ s.regs .nat [] "off_h_kv" =
      some (Tile.scalar (s0.pids 1 % num_heads / (num_heads / num_kv_heads)))
  ∧ s.regs .nat [BLOCK_M] "offs_m" =
      some (Tile.vec (fun i : Fin BLOCK_M => s0.pids 0 * BLOCK_M + i.val))
  ∧ s.regs .nat [BLOCK_N] "offs_n" = some (Tile.vec (fun j : Fin BLOCK_N => j.val))
  ∧ s.regs .nat [BLOCK_D] "offs_d" = some (Tile.vec (fun e : Fin BLOCK_D => e.val))
  ∧ s.regs .nat [] "layout_h" = some (Tile.scalar (s0.pids 1 % num_heads % num_layout))
  ∧ s.regs .real [BLOCK_M] "m_i" =
      some (Tile.vec (fun i : Fin BLOCK_M =>
        bsaMPartial BLOCK_N qStart numKVBlocks gpos Qg Kg scale c i))
  ∧ s.regs .real [BLOCK_M] "l_i" =
      some (Tile.vec (fun i : Fin BLOCK_M =>
        (some (bsaLPartial BLOCK_N qStart numKVBlocks gpos Qg Kg scale c i) : WithBot ℝ)))
  ∧ s.regs .real [BLOCK_M, BLOCK_D] "acc" =
      some (⟨fun idx : TileIndex [BLOCK_M, BLOCK_D] =>
        (some (bsaOPartial BLOCK_N qStart numKVBlocks gpos Qg Kg Vg scale c idx /
          bsaLPartial BLOCK_N qStart numKVBlocks gpos Qg Kg scale c idx.1) : WithBot ℝ)⟩
          : Tile .real [BLOCK_M, BLOCK_D])
  ∧ s.regs .real [BLOCK_M, BLOCK_D] "acc2" =
      some (⟨fun idx : TileIndex [BLOCK_M, BLOCK_D] =>
        (some (bsaOPartial BLOCK_N qStart numKVBlocks gpos Qg Kg Vg2 scale c idx /
          bsaLPartial BLOCK_N qStart numKVBlocks gpos Qg Kg scale c idx.1) : WithBot ℝ)⟩
          : Tile .real [BLOCK_M, BLOCK_D])
  ∧ s.regs .real [BLOCK_M, BLOCK_D] "q" =
      some (⟨fun idx : TileIndex [BLOCK_M, BLOCK_D] =>
        s0.readMemValue .real Q ((s0.pids 1 / num_heads * sqb + s0.pids 1 % num_heads * sqh) +
          ((s0.pids 0 * BLOCK_M + idx.1.val) * sqm + idx.2.1.val))⟩ : Tile .real [BLOCK_M, BLOCK_D])
  ∧ s.regs .real [BLOCK_M, BLOCK_D] "q2" =
      some (⟨fun idx : TileIndex [BLOCK_M, BLOCK_D] =>
        s0.readMemValue .real Q ((s0.pids 1 / num_heads * sqb + s0.pids 1 % num_heads * sqh) +
          ((s0.pids 0 * BLOCK_M + idx.1.val) * sqm + idx.2.1.val) + BLOCK_D)⟩
          : Tile .real [BLOCK_M, BLOCK_D])
  ∧ s.regs .ptr [BLOCK_D, BLOCK_N] "k_ptrs" =
      some (⟨fun idx : TileIndex [BLOCK_D, BLOCK_N] =>
        (K, (s0.pids 1 / num_heads * skb +
            s0.pids 1 % num_heads / (num_heads / num_kv_heads) * skh) +
          (idx.2.1.val * skn + idx.1.val))⟩ : Tile .ptr [BLOCK_D, BLOCK_N])
  ∧ s.regs .ptr [BLOCK_N, BLOCK_D] "v_ptrs" =
      some (⟨fun idx : TileIndex [BLOCK_N, BLOCK_D] =>
        (V, (s0.pids 1 / num_heads * svb +
            s0.pids 1 % num_heads / (num_heads / num_kv_heads) * svh) +
          (idx.1.val * svn + idx.2.1.val))⟩ : Tile .ptr [BLOCK_N, BLOCK_D])
```
</details>

<details><summary><code>qTileBSA</code></summary>

```
/-- Q row `qStart + i`, head channel `e`, read at
`qBase + (qStart+i) · stride_qm + e`. -/
```
```lean
noncomputable def qTileBSA (s : BlockState) (Q : RegionName)
    (num_heads stride_qb stride_qh stride_qm BLOCK_M : Nat)
    (i : Fin BLOCK_M) (e : Nat) : ℝ :=
  s.readMem Q (qBase s num_heads stride_qb stride_qh +
    mIndex s BLOCK_M i * stride_qm + e)
```
</details>

<details><summary><code>kRowBSA</code></summary>

```
/-- K row at global key position `n`, head channel `e`, read at
`kvBase + n · stride_kn + e`. -/
```
```lean
noncomputable def kRowBSA (s : BlockState) (K : RegionName)
    (num_heads num_kv_heads stride_kb stride_kh stride_kn : Nat)
    (n e : Nat) : ℝ :=
  s.readMem K (kvBase s num_heads num_kv_heads stride_kb stride_kh +
    n * stride_kn + e)
```
</details>

<details><summary><code>vRowBSA</code></summary>

```
/-- V row at global key position `n`, head channel `d`, read at
`vbase + n · stride_vn + d`. The second D-block reads channel `BLOCK_D + d`. -/
```
```lean
noncomputable def vRowBSA (s : BlockState) (V : RegionName)
    (num_heads num_kv_heads stride_vb stride_vh stride_vn : Nat)
    (n d : Nat) : ℝ :=
  s.readMem V (kvBase s num_heads num_kv_heads stride_vb stride_vh +
    n * stride_vn + d)
```
</details>

<details><summary><code>bsaLoopBodyG</code></summary>

```
/-- General (symbolic-dim) CSR-loop body — the 26 lowered loop-body statements with
every dim/stride symbolic. Mirrors `bsaLoopBody` literal-for-literal against the
elaborated body dump (`NUM_D_BLOCKS = 2`, `EVEN_N = true`). -/
```
```lean
def bsaLoopBodyG (C : Region .nat)
    (BLOCK_M BLOCK_D BLOCK_N num_heads colStrideH skn svn total_seq_len : Nat) (scale : ℝ) : List Stmt :=
  [ Stmt.assign .nat [] "col_idx"
      (Op.load .nat
        (MemAccess.region C
          (Op.add NumericDType.nat Broadcast.nil
            (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "layout_h") (Op.constNat colStrideH))
            (Op.ref .nat [] "col_idx_idx")))
        MaskOpt.none),
    Stmt.assign .nat [] "start_n"
      (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "col_idx") (Op.constNat BLOCK_N)),
    Stmt.ifThenElse (Op.constBool «true»)
      [Stmt.assign .real [BLOCK_D, BLOCK_N] "k"
          (Op.load .real
            (MemAccess.ptr
              (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BLOCK_D, BLOCK_N] "k_ptrs")
                (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "start_n") (Op.constNat skn))))
            MaskOpt.none)]
      [Stmt.assign .real [BLOCK_D, BLOCK_N] "k"
          (Op.load .real
            (MemAccess.ptr
              (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BLOCK_D, BLOCK_N] "k_ptrs")
                (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "start_n") (Op.constNat skn))))
            (MaskOpt.mask
              (Op.remap [BLOCK_D, BLOCK_N] Broadcast.nil.consSame.consL.leftIndex
                (Op.lt ComparableDType.nat Broadcast.scalarR
                  (Op.add NumericDType.nat Broadcast.scalarR
                    (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_N] "offs_n"))
                    (Op.ref .nat [] "start_n"))
                  (Op.constNat total_seq_len)))))],
    Stmt.assign .real [BLOCK_M, BLOCK_N] "qk" (Op.full [BLOCK_M, BLOCK_N] (Op.const 0)),
    Stmt.assign .real [BLOCK_M, BLOCK_N] "qk"
      (Op.add NumericDType.real Broadcast.nil.consSame.consSame (Op.ref .real [BLOCK_M, BLOCK_N] "qk")
        (Op.dot (batch := []) (Op.ref .real [BLOCK_M, BLOCK_D] "q") (Op.ref .real [BLOCK_D, BLOCK_N] "k"))),
    Stmt.ifThen (Op.constBool (decide (2 ≥ 2)))
      [Stmt.ifThenElse (Op.constBool «true»)
          [Stmt.assign .real [BLOCK_D, BLOCK_N] "k"
              (Op.load .real
                (MemAccess.ptr
                  (Op.ptrAdd Broadcast.scalarR
                    (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BLOCK_D, BLOCK_N] "k_ptrs")
                      (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "start_n") (Op.constNat skn)))
                    (Op.constNat BLOCK_D)))
                MaskOpt.none)]
          [Stmt.assign .real [BLOCK_D, BLOCK_N] "k"
              (Op.load .real
                (MemAccess.ptr
                  (Op.ptrAdd Broadcast.scalarR
                    (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BLOCK_D, BLOCK_N] "k_ptrs")
                      (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "start_n") (Op.constNat skn)))
                    (Op.constNat BLOCK_D)))
                (MaskOpt.mask
                  (Op.remap [BLOCK_D, BLOCK_N] Broadcast.nil.consSame.consL.leftIndex
                    (Op.lt ComparableDType.nat Broadcast.scalarR
                      (Op.add NumericDType.nat Broadcast.scalarR
                        (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_N] "offs_n"))
                        (Op.ref .nat [] "start_n"))
                      (Op.constNat total_seq_len)))))],
        Stmt.assign .real [BLOCK_M, BLOCK_N] "qk"
          (Op.add NumericDType.real Broadcast.nil.consSame.consSame (Op.ref .real [BLOCK_M, BLOCK_N] "qk")
            (Op.dot (batch := []) (Op.ref .real [BLOCK_M, BLOCK_D] "q2") (Op.ref .real [BLOCK_D, BLOCK_N] "k")))],
    Stmt.assign .real [BLOCK_M, BLOCK_N] "qk"
      (Op.mul NumericDType.real Broadcast.scalarR (Op.ref .real [BLOCK_M, BLOCK_N] "qk") (Op.const scale)),
    Stmt.assign .real [BLOCK_M, BLOCK_N] "qk"
      (Op.add NumericDType.real Broadcast.nil.consSame.consSame (Op.ref .real [BLOCK_M, BLOCK_N] "qk")
        ((Op.ge ComparableDType.nat Broadcast.nil.consL.consR
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_M] "offs_m"))
              (Op.add NumericDType.nat Broadcast.scalarL (Op.ref .nat [] "start_n")
                (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_N] "offs_n")))).where
          ((Op.const 0).broadcast [BLOCK_M, BLOCK_N]) (Op.negInf.broadcast [BLOCK_M, BLOCK_N]))),
    Stmt.assign .real [BLOCK_M] "m_ij"
      (Op.reduceMax ⟨1, by simp⟩ «false» (Op.ref .real [BLOCK_M, BLOCK_N] "qk")),
    Stmt.assign .real [BLOCK_M, BLOCK_N] "p"
      (Op.sub NumericDType.real Broadcast.nil.consR.consSame (Op.ref .real [BLOCK_M, BLOCK_N] "qk")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BLOCK_M] "m_ij"))).exp,
    Stmt.assign .real [BLOCK_M] "l_ij"
      (Op.reduceSum ⟨1, by simp⟩ «false» (Op.ref .real [BLOCK_M, BLOCK_N] "p")),
    Stmt.assign .real [BLOCK_M] "m_i_new"
      ((Op.gt ComparableDType.real (Broadcast.consSame Broadcast.nil)
            (Op.ref .real [BLOCK_M] "m_i") (Op.ref .real [BLOCK_M] "m_ij")).where
        (Op.ref .real [BLOCK_M] "m_i") (Op.ref .real [BLOCK_M] "m_ij")),
    Stmt.assign .real [BLOCK_M] "alpha"
      (Op.sub NumericDType.real (Broadcast.consSame Broadcast.nil)
        (Op.ref .real [BLOCK_M] "m_i") (Op.ref .real [BLOCK_M] "m_i_new")).exp,
    Stmt.assign .real [BLOCK_M] "beta"
      (Op.sub NumericDType.real (Broadcast.consSame Broadcast.nil)
        (Op.ref .real [BLOCK_M] "m_ij") (Op.ref .real [BLOCK_M] "m_i_new")).exp,
    Stmt.assign .real [BLOCK_M] "l_i_new"
      (Op.add NumericDType.real (Broadcast.consSame Broadcast.nil)
        (Op.mul NumericDType.real (Broadcast.consSame Broadcast.nil)
          (Op.ref .real [BLOCK_M] "alpha") (Op.ref .real [BLOCK_M] "l_i"))
        (Op.mul NumericDType.real (Broadcast.consSame Broadcast.nil)
          (Op.ref .real [BLOCK_M] "beta") (Op.ref .real [BLOCK_M] "l_ij"))),
    Stmt.assign .real [BLOCK_M] "p_scale"
      (Op.div NumericDType.real (Broadcast.consSame Broadcast.nil)
        (Op.ref .real [BLOCK_M] "beta") (Op.ref .real [BLOCK_M] "l_i_new")),
    Stmt.assign .real [BLOCK_M, BLOCK_N] "p"
      (Op.mul NumericDType.real Broadcast.nil.consR.consSame (Op.ref .real [BLOCK_M, BLOCK_N] "p")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BLOCK_M] "p_scale"))),
    Stmt.assign .real [BLOCK_M] "acc_scale"
      (Op.mul NumericDType.real (Broadcast.consSame Broadcast.nil)
        (Op.div NumericDType.real (Broadcast.consSame Broadcast.nil)
          (Op.ref .real [BLOCK_M] "l_i") (Op.ref .real [BLOCK_M] "l_i_new"))
        (Op.ref .real [BLOCK_M] "alpha")),
    Stmt.assign .real [BLOCK_M, BLOCK_D] "acc"
      (Op.mul NumericDType.real Broadcast.nil.consR.consSame (Op.ref .real [BLOCK_M, BLOCK_D] "acc")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BLOCK_M] "acc_scale"))),
    Stmt.ifThen (Op.constBool (decide (2 ≥ 2)))
      [Stmt.assign .real [BLOCK_M, BLOCK_D] "acc2"
          (Op.mul NumericDType.real Broadcast.nil.consR.consSame (Op.ref .real [BLOCK_M, BLOCK_D] "acc2")
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BLOCK_M] "acc_scale")))],
    Stmt.assign .real [BLOCK_M, BLOCK_N] "p" (Op.ref .real [BLOCK_M, BLOCK_N] "p"),
    Stmt.ifThenElse (Op.constBool «true»)
      [Stmt.assign .real [BLOCK_N, BLOCK_D] "v"
          (Op.load .real
            (MemAccess.ptr
              (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BLOCK_N, BLOCK_D] "v_ptrs")
                (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "start_n") (Op.constNat svn))))
            MaskOpt.none)]
      [Stmt.assign .real [BLOCK_N, BLOCK_D] "v"
          (Op.load .real
            (MemAccess.ptr
              (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BLOCK_N, BLOCK_D] "v_ptrs")
                (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "start_n") (Op.constNat svn))))
            (MaskOpt.mask
              (Op.remap [BLOCK_N, BLOCK_D] Broadcast.nil.consL.consSame.leftIndex
                (Op.lt ComparableDType.nat Broadcast.scalarR
                  (Op.add NumericDType.nat Broadcast.scalarR
                    (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_N] "offs_n"))
                    (Op.ref .nat [] "start_n"))
                  (Op.constNat total_seq_len)))))],
    Stmt.assign .real [BLOCK_M, BLOCK_D] "acc"
      (Op.add NumericDType.real Broadcast.nil.consSame.consSame (Op.ref .real [BLOCK_M, BLOCK_D] "acc")
        (Op.dot (batch := []) (Op.ref .real [BLOCK_M, BLOCK_N] "p") (Op.ref .real [BLOCK_N, BLOCK_D] "v"))),
    Stmt.ifThen (Op.constBool (decide (2 ≥ 2)))
      [Stmt.ifThenElse (Op.constBool «true»)
          [Stmt.assign .real [BLOCK_N, BLOCK_D] "v"
              (Op.load .real
                (MemAccess.ptr
                  (Op.ptrAdd Broadcast.scalarR
                    (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BLOCK_N, BLOCK_D] "v_ptrs")
                      (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "start_n") (Op.constNat svn)))
                    (Op.constNat BLOCK_D)))
                MaskOpt.none)]
          [Stmt.assign .real [BLOCK_N, BLOCK_D] "v"
              (Op.load .real
                (MemAccess.ptr
                  (Op.ptrAdd Broadcast.scalarR
                    (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BLOCK_N, BLOCK_D] "v_ptrs")
                      (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "start_n") (Op.constNat svn)))
                    (Op.constNat BLOCK_D)))
                (MaskOpt.mask
                  (Op.remap [BLOCK_N, BLOCK_D] Broadcast.nil.consL.consSame.leftIndex
                    (Op.lt ComparableDType.nat Broadcast.scalarR
                      (Op.add NumericDType.nat Broadcast.scalarR
                        (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_N] "offs_n"))
                        (Op.ref .nat [] "start_n"))
                      (Op.constNat total_seq_len)))))],
        Stmt.assign .real [BLOCK_M, BLOCK_D] "acc2"
          (Op.add NumericDType.real Broadcast.nil.consSame.consSame (Op.ref .real [BLOCK_M, BLOCK_D] "acc2")
            (Op.dot (batch := []) (Op.ref .real [BLOCK_M, BLOCK_N] "p") (Op.ref .real [BLOCK_N, BLOCK_D] "v")))],
    Stmt.assign .real [BLOCK_M] "l_i" (Op.ref .real [BLOCK_M] "l_i_new"),
    Stmt.assign .real [BLOCK_M] "m_i" (Op.ref .real [BLOCK_M] "m_i_new") ]
```
</details>

<details><summary><code>block_sparse_attention_kernel</code></summary>

```
/-- Faithful transcription of `block_sparse_attn.py`'s
`block_sparse_attention_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `tl.constexpr` parameters become Lean parameters with `$(...)` at use
  sites.
- The Python `tl.static_print(f"...")` f-string payload is represented by a
  fixed debug string; `tl.static_print` is a compile-time/no-op DSL marker.
- `if NUM_D_BLOCKS >= 2:` is represented as the equivalent Bool antiquote
  `if $((NUM_D_BLOCKS >= 2 : Bool))`. -/
```
```lean
def block_sparse_attention_kernel
    (out Q K V : RegionName)
    (layout_csr_row_indices layout_csr_col_indices : Region .nat)
    (layout_csr_row_stride_h layout_csr_col_stride_h num_layout : Nat)
    (softmax_scale : ℝ)
    (stride_qb stride_qh stride_qm stride_kb stride_kh stride_kn
      stride_vb stride_vh stride_vn stride_ob stride_oh stride_om
      num_heads num_kv_heads total_seq_len BLOCK_M BLOCK_N BLOCK_D
      NUM_D_BLOCKS : Nat)
    (EVEN_M EVEN_N : Bool) :
    ComputeKernel := triton {
  tl.static_print("block_sparse_attention_kernel")
  q_seq_len = $(total_seq_len)
  start_m = tl.program_id(0)
  off_bh = tl.program_id(1)
  off_h = off_bh % $(num_heads)
  off_b = off_bh // $(num_heads)
  head_groups = $(num_heads) // $(num_kv_heads)
  off_h_kv = off_h // head_groups
  Q += off_b * $(stride_qb) + off_h * $(stride_qh)
  K += off_b * $(stride_kb) + off_h_kv * $(stride_kh)
  V += off_b * $(stride_vb) + off_h_kv * $(stride_vh)
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_n = tl.arange(0, $(BLOCK_N))
  offs_d = tl.arange(0, $(BLOCK_D))
  off_q = offs_m[:, None] * $(stride_qm) + offs_d[None, :]
  off_k = offs_n[None, :] * $(stride_kn) + offs_d[:, None]
  off_v = offs_n[:, None] * $(stride_vn) + offs_d[None, :]
  q_ptrs = Q + off_q
  k_ptrs = K + off_k
  v_ptrs = V + off_v
  m_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32)
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_D)], dtype=tl.float32)
  if $((NUM_D_BLOCKS >= 2 : Bool)) {
    acc2 = tl.zeros([$(BLOCK_M), $(BLOCK_D)], dtype=tl.float32)
  }
  if EVEN_M {
    q = tl.load(q_ptrs)
    if $((NUM_D_BLOCKS >= 2 : Bool)) {
      q2 = tl.load(q_ptrs + $(BLOCK_D))
    }
  } else {
    q = tl.load(q_ptrs, mask=offs_m[:, None] < q_seq_len)
    if $((NUM_D_BLOCKS >= 2 : Bool)) {
      q2 = tl.load(q_ptrs + $(BLOCK_D), mask=offs_m[:, None] < q_seq_len)
    }
  }
  layout_h = off_h % $(num_layout)
  layout_ptr = layout_csr_row_indices + layout_h * $(layout_csr_row_stride_h) + start_m
  start_l = (tl.load(layout_ptr)).to(tl.int32)
  end_l = (tl.load(layout_ptr + $(1))).to(tl.int32)
  for col_idx_idx in range(start_l, end_l) {
    col_idx = (tl.load(layout_csr_col_indices +
      layout_h * $(layout_csr_col_stride_h) + col_idx_idx)).to(tl.int32)
    start_n = col_idx * $(BLOCK_N)
    if EVEN_N {
      k = tl.load(k_ptrs + start_n * $(stride_kn))
    } else {
      k = tl.load(k_ptrs + start_n * $(stride_kn),
        mask=offs_n[None, :] + start_n < $(total_seq_len))
    }
    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    qk += tl.dot(q, k)
    if $((NUM_D_BLOCKS >= 2 : Bool)) {
      if EVEN_N {
        k = tl.load(k_ptrs + start_n * $(stride_kn) + $(BLOCK_D))
      } else {
        k = tl.load(k_ptrs + start_n * $(stride_kn) + $(BLOCK_D),
          mask=offs_n[None, :] + start_n < $(total_seq_len))
      }
      qk += tl.dot(q2, k)
    }
    qk *= $(softmax_scale)
    qk += tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), 0, float("-inf"))
    m_ij = tl.max(qk, 1)
    p = tl.exp(qk - m_ij[:, None])
    l_ij = tl.sum(p, 1)
    m_i_new = tl.maximum(m_i, m_ij)
    alpha = tl.exp(m_i - m_i_new)
    beta = tl.exp(m_ij - m_i_new)
    l_i_new = alpha * l_i + beta * l_ij
    p_scale = beta / l_i_new
    p = p * p_scale[:, None]
    acc_scale = l_i / l_i_new * alpha
    acc = acc * acc_scale[:, None]
    if $((NUM_D_BLOCKS >= 2 : Bool)) {
      acc2 = acc2 * acc_scale[:, None]
    }
    p = (p).to(Q.dtype.element_ty)
    if EVEN_N {
      v = tl.load(v_ptrs + start_n * $(stride_vn))
    } else {
      v = tl.load(v_ptrs + start_n * $(stride_vn),
        mask=offs_n[:, None] + start_n < $(total_seq_len))
    }
    acc += tl.dot(p, v)
    if $((NUM_D_BLOCKS >= 2 : Bool)) {
      if EVEN_N {
        v = tl.load(v_ptrs + start_n * $(stride_vn) + $(BLOCK_D))
      } else {
        v = tl.load(v_ptrs + start_n * $(stride_vn) + $(BLOCK_D),
          mask=offs_n[:, None] + start_n < $(total_seq_len))
      }
      acc2 += tl.dot(p, v)
    }
    l_i = l_i_new
    m_i = m_i_new
  }
  off_o = off_b * $(stride_ob) + off_h * $(stride_oh) +
    offs_m[:, None] * $(stride_om) + offs_d[None, :]
  out_ptrs = out + off_o
  tl.store(out_ptrs, acc, mask=offs_m[:, None] < q_seq_len)
  if $((NUM_D_BLOCKS >= 2 : Bool)) {
    tl.store(out_ptrs + $(BLOCK_D), acc2, mask=offs_m[:, None] < q_seq_len)
  }
}
```
</details>

<details><summary><code>blockSparseAttnClosedForm</code></summary>

```
/-- **Genuine closed-form block-sparse attention output** for one program.

`out[i, d] = (Σ_r w(i,r) · V[selKeyGlobal r, dChan d]) / (Σ_r w(i,r))`, where the
sum ranges over the `numSelBlocks · BLOCK_N` flattened selected keys, and
`w(i,r) = exp(softmax_scale · rawScore i (selKeyGlobal r))` when key
`selKeyGlobal r ≤ qStart + i` (causal), else `0`. `dChan d = dBlockBase + d`
selects the head channel for the chosen output D-block (`dBlockBase = 0` for the
first store, `BLOCK_D` for the second). This is `attentionRealCausal` with the
causal predicate evaluated on the global key position. -/
```
```lean
noncomputable def blockSparseAttnClosedForm
    (s : BlockState) (Q K V : RegionName) (layoutCols : Region .nat)
    (num_heads num_kv_heads
      stride_qb stride_qh stride_qm stride_kb stride_kh stride_kn
      stride_vb stride_vh stride_vn
      layout_h stride_col start_l numSelBlocks
      HEAD_DIM BLOCK_M BLOCK_N dBlockBase : Nat)
    (softmax_scale : ℝ)
    (i : Fin BLOCK_M) (d : Nat) : ℝ :=
  let n := fun r : Fin (numSelBlocks * BLOCK_N) =>
    selKeyGlobal s layoutCols layout_h stride_col start_l BLOCK_N r.val
  let w := fun r : Fin (numSelBlocks * BLOCK_N) =>
    if n r ≤ mIndex s BLOCK_M i then
      Real.exp (softmax_scale *
        rawScoreBSA s Q K num_heads num_kv_heads stride_qb stride_qh stride_qm
          stride_kb stride_kh stride_kn HEAD_DIM BLOCK_M i (n r))
    else 0
  let denom := Finset.univ.sum (fun r => w r)
  let numer := Finset.univ.sum (fun r =>
    w r * vRowBSA s V num_heads num_kv_heads stride_vb stride_vh stride_vn
      (n r) (dBlockBase + d))
  numer / denom
```
</details>

<details><summary><code>offB</code></summary>

```lean
def offB (s : BlockState) (num_heads : Nat) : Nat :=
  s.pids 1 / num_heads
```
</details>

<details><summary><code>offH</code></summary>

```lean
def offH (s : BlockState) (num_heads : Nat) : Nat :=
  s.pids 1 % num_heads
```
</details>

<details><summary><code>mIndex</code></summary>

```lean
def mIndex (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 0 * BLOCK_M + i.val
```
</details>

<details><summary><code>dIndex</code></summary>

```lean
def dIndex (idx : TileIndex [BLOCK_M, BLOCK_D]) : Nat :=
  idx.2.1.val
```
</details>

<details><summary><code>bsaMPartial</code></summary>

```
/-- Running per-row max of gathered causal masked scores over the first `k`
KV blocks. Future (gathered-global) keys enter as `⊥`. -/
```
```lean
noncomputable def bsaMPartial {M D : Nat} (Bk : Nat)
    (qStart : Nat) (numKVBlocks : Nat) (gpos : Fin (Bk * numKVBlocks) → Nat)
    (Q : TileIndex [M, D] → ℝ)
    (Kg : TileIndex [Bk * numKVBlocks, D] → ℝ) (scale : ℝ) :
    Nat → Fin M → WithBot ℝ
  | 0, _ => ⊥
  | k + 1, i =>
      if h : k + 1 ≤ numKVBlocks then
        max (bsaMPartial Bk qStart numKVBlocks gpos Q Kg scale k i)
          ((Finset.univ : Finset (Fin Bk)).sup fun jLocal =>
            maskedScore qStart gpos Q Kg scale i
              (StreamingAccumulator.blockIndex Bk numKVBlocks k h jLocal))
      else
        bsaMPartial Bk qStart numKVBlocks gpos Q Kg scale k i
```
</details>

<details><summary><code>bsaLPartial</code></summary>

```
/-- Running causal softmax normalizer, shifted by the running max. -/
```
```lean
noncomputable def bsaLPartial {M D : Nat} (Bk : Nat)
    (qStart : Nat) (numKVBlocks : Nat) (gpos : Fin (Bk * numKVBlocks) → Nat)
    (Q : TileIndex [M, D] → ℝ)
    (Kg : TileIndex [Bk * numKVBlocks, D] → ℝ) (scale : ℝ) :
    Nat → Fin M → ℝ
  | 0, _ => 0
  | k + 1, i =>
      if h : k + 1 ≤ numKVBlocks then
        let alpha :=
          (WithBot.realExp
            (Option.map₂ (fun x y : ℝ => x - y)
              (bsaMPartial Bk qStart numKVBlocks gpos Q Kg scale k i)
              (bsaMPartial Bk qStart numKVBlocks gpos Q Kg scale (k + 1) i))).unbotD 0
        alpha * bsaLPartial Bk qStart numKVBlocks gpos Q Kg scale k i +
          (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
            (WithBot.realExp
              (Option.map₂ (fun x y : ℝ => x - y)
                (maskedScore qStart gpos Q Kg scale i
                  (StreamingAccumulator.blockIndex Bk numKVBlocks k h jLocal))
                (bsaMPartial Bk qStart numKVBlocks gpos Q Kg scale (k + 1) i))).unbotD 0)
      else
        bsaLPartial Bk qStart numKVBlocks gpos Q Kg scale k i
```
</details>

<details><summary><code>bsaOPartial</code></summary>

```
/-- Running causal unnormalized output accumulator over the gathered value
stream `Vg`. The two D-blocks differ only by which `Vg` is supplied. -/
```
```lean
noncomputable def bsaOPartial {M D Dv : Nat} (Bk : Nat)
    (qStart : Nat) (numKVBlocks : Nat) (gpos : Fin (Bk * numKVBlocks) → Nat)
    (Q : TileIndex [M, D] → ℝ)
    (Kg : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (Vg : TileIndex [Bk * numKVBlocks, Dv] → ℝ) (scale : ℝ) :
    Nat → TileIndex [M, Dv] → ℝ
  | 0, _ => 0
  | k + 1, idx =>
      if h : k + 1 ≤ numKVBlocks then
        let alpha :=
          (WithBot.realExp
            (Option.map₂ (fun x y : ℝ => x - y)
              (bsaMPartial Bk qStart numKVBlocks gpos Q Kg scale k idx.1)
              (bsaMPartial Bk qStart numKVBlocks gpos Q Kg scale (k + 1) idx.1))).unbotD 0
        alpha * bsaOPartial Bk qStart numKVBlocks gpos Q Kg Vg scale k idx +
          (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
            let j := StreamingAccumulator.blockIndex Bk numKVBlocks k h jLocal
            (WithBot.realExp
              (Option.map₂ (fun x y : ℝ => x - y)
                (maskedScore qStart gpos Q Kg scale idx.1 j)
                (bsaMPartial Bk qStart numKVBlocks gpos Q Kg scale (k + 1) idx.1))).unbotD 0 *
              Vg (j, idx.2.1, PUnit.unit))
      else
        bsaOPartial Bk qStart numKVBlocks gpos Q Kg Vg scale k idx
```
</details>

<details><summary><code>qBase</code></summary>

```
/-- Q tile base offset `off_b · stride_qb + off_h · stride_qh`. -/
```
```lean
def qBase (s : BlockState) (num_heads stride_qb stride_qh : Nat) : Nat :=
  offB s num_heads * stride_qb + offH s num_heads * stride_qh
```
</details>

<details><summary><code>kvBase</code></summary>

```
/-- K/V tile base offset `off_b · stride_kb + off_h_kv · stride_kh`
(grouped-query: the KV head is `off_h // headGroups`). -/
```
```lean
def kvBase (s : BlockState) (num_heads num_kv_heads stride_kb stride_kh : Nat) :
    Nat :=
  offB s num_heads * stride_kb + offHkv s num_heads num_kv_heads * stride_kh
```
</details>

<details><summary><code>rawScoreBSA</code></summary>

```
/-- Unscaled raw score `Σ_{e<HEAD_DIM} Q[qStart+i,e] · K[n,e]` at global key `n`. -/
```
```lean
noncomputable def rawScoreBSA (s : BlockState) (Q K : RegionName)
    (num_heads num_kv_heads stride_qb stride_qh stride_qm
      stride_kb stride_kh stride_kn HEAD_DIM BLOCK_M : Nat)
    (i : Fin BLOCK_M) (n : Nat) : ℝ :=
  Finset.univ.sum (fun e : Fin HEAD_DIM =>
    qTileBSA s Q num_heads stride_qb stride_qh stride_qm BLOCK_M i e.val *
      kRowBSA s K num_heads num_kv_heads stride_kb stride_kh stride_kn n e.val)
```
</details>

<details><summary><code>maskedScore</code></summary>

```
/-- Causal masked score under the **gathered global position** predicate.
Returns `⊥` when the gathered key's global position `gpos r` is in the future
(`> qStart + i`), otherwise the ordinary gathered scaled score. -/
```
```lean
noncomputable def maskedScore {M N D Bk : Nat}
    (qStart : Nat) (gpos : Fin (Bk * N) → Nat)
    (Q : TileIndex [M, D] → ℝ) (Kg : TileIndex [Bk * N, D] → ℝ) (scale : ℝ)
    (i : Fin M) (r : Fin (Bk * N)) : WithBot ℝ :=
  if gpos r ≤ qStart + i.val then
    ((gScore Q Kg scale i r : ℝ) : WithBot ℝ)
  else
    ⊥
```
</details>

<details><summary><code>offHkv</code></summary>

```
/-- KV head index `off_h // headGroups` for grouped-query attention. -/
```
```lean
def offHkv (s : BlockState) (num_heads num_kv_heads : Nat) : Nat :=
  offH s num_heads / headGroups num_heads num_kv_heads
```
</details>

<details><summary><code>gScore</code></summary>

```
/-- Gathered scaled score: `scale · Σ_e Q[i,e] · Kg[r,e]`, reading the gathered
key tile at flat stream index `r` (no global remap on the score itself). -/
```
```lean
noncomputable def gScore {M N D Bk : Nat}
    (Q : TileIndex [M, D] → ℝ) (Kg : TileIndex [Bk * N, D] → ℝ) (scale : ℝ)
    (i : Fin M) (r : Fin (Bk * N)) : ℝ :=
  scale * Finset.univ.sum (fun d : Fin D =>
    Q (i, d, PUnit.unit) * Kg (r, d, PUnit.unit))
```
</details>

<details><summary><code>headGroups</code></summary>

```
/-- Grouped-query head map `num_heads // num_kv_heads`. -/
```
```lean
def headGroups (num_heads num_kv_heads : Nat) : Nat := num_heads / num_kv_heads
```
</details>

## Also present (pinned special-case summaries)
- `block_sparse_attn_output_store_slice_compute_correct`
- `block_sparse_attn_output_store_second_slice_compute_correct`
