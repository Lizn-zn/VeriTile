# Spec sheet — `bench/tritonbench_g/batched_vecmat_mult/Correct.lean`

**Python source:** `bench/tritonbench_g/batched_vecmat_mult/batched_vecmat_mult.py`

## Public theorem: `vecmat_correctness`

**Statement:**
```lean
specification vecmat_correctness (A B out : RegionName) (M N BM BN BK T : Nat)
    (hBK : 0 < BK) (hBN : BN ≤ N) :
    Spec.Real (originalIO A B out M N BM BN BK T ⊨ fun _ _ xs ys l => expected BM BN BK T xs ys l)
```

**Assumptions / layout contracts:**
- `hBK : 0 < BK`
- `hBN : BN ≤ N`

**Closed-form spec defs (transitive):** `originalIO`, `expected`, `original`, `inputALane`, `inputBLane`

<details><summary><code>originalIO</code></summary>

```
/-- At iteration t, read A[m,t*BK+e] and B[m,n,t*BK+e], with source B tile
order [BN,BM,BK]; store output[m,n]. All these accesses are unmasked. M is
source-unused; the actual windows and allocation bounds determine validity. -/
```
```lean
def originalIO (A B out : RegionName) (M N BM BN BK T : Nat) : StreamMasked2DKernelIO₂ where
  kernel := (TypedSource.original A B out M N (BK*T) BM BN BK).eraseDType
  projection := RealSource.projection A B out M N (BK*T) BM BN BK Bool.false
  inp1 := A
  inp2 := B
  out := out
  T := T
  B1 := BM*BK
  B2 := BN*BM*BK
  C := BM*BN
  outDType := .real
  read1 := fun pm _ t l => (pm*BM+l.val/BK)*(BK*T) + (t.val*BK+l.val%BK)
  read2 := fun pm pn t l => (pm*BM+l.val/BK%BM)*N*(BK*T) +
    (pn*BN+l.val/BK/BM)*(BK*T) + (t.val*BK+l.val%BK)
  write := fun pm pn l => (pm*BM+l.val/BN)*N + (pn*BN+l.val%BN)
  mask1 := fun _ _ _ _ => True
  mask2 := fun _ _ _ _ => True
  writeMask := fun _ _ _ => True
```
</details>

<details><summary><code>expected</code></summary>

```
/-- Independent mathematical sum over all K=T*BK input elements. -/
```
```lean
def expected (BM BN BK T : Nat) (xs : Fin T → Fin (BM*BK) → ℝ)
    (ys : Fin T → Fin (BN*BM*BK) → ℝ) (l : Fin (BM*BN)) : ℝ :=
  ∑ t : Fin T, ∑ e : Fin BK,
    xs t (inputALane BM BN BK l e) * ys t (inputBLane BM BN BK l e)
```
</details>

<details><summary><code>original</code></summary>

```lean
def original
    (A B output : RegionName)
    (_dim_m dim_n dim_k BLOCK_M BLOCK_N BLOCK_K : Nat) :
    ComputeKernel := triton {
  tl.region A = tl.float32, B = tl.float32, output = tl.float32
  m_index = tl.program_id(0)
  n_index = tl.program_id(1)
  output_tile = (m_index * $(BLOCK_M) + tl.arange(0, $(BLOCK_M)))[:, None] * $(dim_n) +
    (n_index * $(BLOCK_N) + tl.arange(0, $(BLOCK_N)))[None, :]
  vecmat = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=A.dtype.element_ty)
  k_blocks = $(dim_k) // $(BLOCK_K)
  for k_index in range(k_blocks) {
    a_tile = (m_index * $(BLOCK_M) + tl.arange(0, $(BLOCK_M)))[:, None] * $(dim_k) +
      (k_index * $(BLOCK_K) + tl.arange(0, $(BLOCK_K)))[None, :]
    a = tl.load(A + a_tile)
    b_tile = (m_index * $(BLOCK_M) + tl.arange(0, $(BLOCK_M)))[None, :, None] *
      $(dim_n) * $(dim_k) +
      (n_index * $(BLOCK_N) + tl.arange(0, $(BLOCK_N)))[:, None, None] * $(dim_k) +
      (k_index * $(BLOCK_K) + tl.arange(0, $(BLOCK_K)))[None, None, :]
    b = tl.load(B + b_tile)
    expanded_a, _ = tl.broadcast(a, b)
    vecmat += tl.trans(tl.sum(expanded_a * b, axis=2))
  }
  tl.store(output + output_tile, vecmat)
}
```
</details>

<details><summary><code>inputALane</code></summary>

```
/-- For output lane (i,j), A contributes (i,e) and B contributes (j,i,e). -/
```
```lean
def inputALane (BM BN BK : Nat) (l : Fin (BM*BN)) (e : Fin BK) : Fin (BM*BK) :=
  Lane2D.encode ((Lane2D.decode l).1, e, PUnit.unit)
```
</details>

<details><summary><code>inputBLane</code></summary>

```lean
def inputBLane (BM BN BK : Nat) (l : Fin (BM*BN)) (e : Fin BK) : Fin (BN*BM*BK) :=
  Lane2D.encode (Lane2D.encode ((Lane2D.decode l).2.1, (Lane2D.decode l).1, PUnit.unit), e, PUnit.unit)
```
</details>

## Public theorem: `vecmat_block_first_correctness`

**Statement:**
```lean
specification vecmat_block_first_correctness (A B out : RegionName) (M N BM BN BK T : Nat)
    (hBK : 0 < BK) (hBN : BN ≤ N) :
    Spec.Real (blockFirstIO A B out M N BM BN BK T ⊨ fun _ _ xs ys l => expected BM BN BK T xs ys l)
```

**Assumptions / layout contracts:**
- `hBK : 0 < BK`
- `hBN : BN ≤ N`

**Closed-form spec defs (transitive):** `blockFirstIO`, `expected`, `originalIO`, `blockFirst`, `inputALane`, `inputBLane`, `original`

<details><summary><code>blockFirstIO</code></summary>

```lean
def blockFirstIO (A B out : RegionName) (M N BM BN BK T : Nat) : StreamMasked2DKernelIO₂ :=
  { originalIO A B out M N BM BN BK T with
    kernel := (TypedSource.blockFirst A B out M N (BK*T) BM BN BK).eraseDType
    projection := RealSource.projection A B out M N (BK*T) BM BN BK Bool.true }
```
</details>

<details><summary><code>expected</code></summary>

```
/-- Independent mathematical sum over all K=T*BK input elements. -/
```
```lean
def expected (BM BN BK T : Nat) (xs : Fin T → Fin (BM*BK) → ℝ)
    (ys : Fin T → Fin (BN*BM*BK) → ℝ) (l : Fin (BM*BN)) : ℝ :=
  ∑ t : Fin T, ∑ e : Fin BK,
    xs t (inputALane BM BN BK l e) * ys t (inputBLane BM BN BK l e)
```
</details>

<details><summary><code>originalIO</code></summary>

```
/-- At iteration t, read A[m,t*BK+e] and B[m,n,t*BK+e], with source B tile
order [BN,BM,BK]; store output[m,n]. All these accesses are unmasked. M is
source-unused; the actual windows and allocation bounds determine validity. -/
```
```lean
def originalIO (A B out : RegionName) (M N BM BN BK T : Nat) : StreamMasked2DKernelIO₂ where
  kernel := (TypedSource.original A B out M N (BK*T) BM BN BK).eraseDType
  projection := RealSource.projection A B out M N (BK*T) BM BN BK Bool.false
  inp1 := A
  inp2 := B
  out := out
  T := T
  B1 := BM*BK
  B2 := BN*BM*BK
  C := BM*BN
  outDType := .real
  read1 := fun pm _ t l => (pm*BM+l.val/BK)*(BK*T) + (t.val*BK+l.val%BK)
  read2 := fun pm pn t l => (pm*BM+l.val/BK%BM)*N*(BK*T) +
    (pn*BN+l.val/BK/BM)*(BK*T) + (t.val*BK+l.val%BK)
  write := fun pm pn l => (pm*BM+l.val/BN)*N + (pn*BN+l.val%BN)
  mask1 := fun _ _ _ _ => True
  mask2 := fun _ _ _ _ => True
  writeMask := fun _ _ _ => True
```
</details>

<details><summary><code>blockFirst</code></summary>

```lean
def blockFirst
    (A B output : RegionName)
    (_dim_m dim_n dim_k BLOCK_M BLOCK_N BLOCK_K : Nat) :
    ComputeKernel := triton {
  tl.region A = tl.float32, B = tl.float32, output = tl.float32
  m_index = tl.program_id(0)
  n_index = tl.program_id(1)
  output_tile = (m_index * $(BLOCK_M) + tl.arange(0, $(BLOCK_M)))[:, None] * $(dim_n) +
    (n_index * $(BLOCK_N) + tl.arange(0, $(BLOCK_N)))[None, :]
  vecmat = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=A.dtype.element_ty)
  k_blocks = $(dim_k) // $(BLOCK_K)
  for k_index in range(k_blocks) {
    a_tile = (m_index * $(BLOCK_M) + tl.arange(0, $(BLOCK_M)))[:, None] * $(dim_k) +
      (k_index * $(BLOCK_K) + tl.arange(0, $(BLOCK_K)))[None, :]
    a = tl.load(A + a_tile)
    b_tile = (m_index * $(BLOCK_M) + tl.arange(0, $(BLOCK_M)))[None, :, None] *
      $(dim_n) * $(dim_k) +
      (n_index * $(BLOCK_N) + tl.arange(0, $(BLOCK_N)))[:, None, None] * $(dim_k) +
      (k_index * $(BLOCK_K) + tl.arange(0, $(BLOCK_K)))[None, None, :]
    b = tl.load(B + b_tile)
    expanded_a, _ = tl.broadcast(a, b)
    vecmat = tl.trans(tl.sum(expanded_a * b, axis=2)) + vecmat
  }
  tl.store(output + output_tile, vecmat)
}
```
</details>

<details><summary><code>inputALane</code></summary>

```
/-- For output lane (i,j), A contributes (i,e) and B contributes (j,i,e). -/
```
```lean
def inputALane (BM BN BK : Nat) (l : Fin (BM*BN)) (e : Fin BK) : Fin (BM*BK) :=
  Lane2D.encode ((Lane2D.decode l).1, e, PUnit.unit)
```
</details>

<details><summary><code>inputBLane</code></summary>

```lean
def inputBLane (BM BN BK : Nat) (l : Fin (BM*BN)) (e : Fin BK) : Fin (BN*BM*BK) :=
  Lane2D.encode (Lane2D.encode ((Lane2D.decode l).2.1, (Lane2D.decode l).1, PUnit.unit), e, PUnit.unit)
```
</details>

<details><summary><code>original</code></summary>

```lean
def original
    (A B output : RegionName)
    (_dim_m dim_n dim_k BLOCK_M BLOCK_N BLOCK_K : Nat) :
    ComputeKernel := triton {
  tl.region A = tl.float32, B = tl.float32, output = tl.float32
  m_index = tl.program_id(0)
  n_index = tl.program_id(1)
  output_tile = (m_index * $(BLOCK_M) + tl.arange(0, $(BLOCK_M)))[:, None] * $(dim_n) +
    (n_index * $(BLOCK_N) + tl.arange(0, $(BLOCK_N)))[None, :]
  vecmat = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=A.dtype.element_ty)
  k_blocks = $(dim_k) // $(BLOCK_K)
  for k_index in range(k_blocks) {
    a_tile = (m_index * $(BLOCK_M) + tl.arange(0, $(BLOCK_M)))[:, None] * $(dim_k) +
      (k_index * $(BLOCK_K) + tl.arange(0, $(BLOCK_K)))[None, :]
    a = tl.load(A + a_tile)
    b_tile = (m_index * $(BLOCK_M) + tl.arange(0, $(BLOCK_M)))[None, :, None] *
      $(dim_n) * $(dim_k) +
      (n_index * $(BLOCK_N) + tl.arange(0, $(BLOCK_N)))[:, None, None] * $(dim_k) +
      (k_index * $(BLOCK_K) + tl.arange(0, $(BLOCK_K)))[None, None, :]
    b = tl.load(B + b_tile)
    expanded_a, _ = tl.broadcast(a, b)
    vecmat += tl.trans(tl.sum(expanded_a * b, axis=2))
  }
  tl.store(output + output_tile, vecmat)
}
```
</details>
