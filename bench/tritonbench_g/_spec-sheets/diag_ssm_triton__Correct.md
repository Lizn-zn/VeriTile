# Spec sheet — `bench/tritonbench_g/diag_ssm_triton/Correct.lean`

**Python source:** `bench/tritonbench_g/diag_ssm_triton/diag_ssm_triton.py`

## Public theorem: `forward_correctness`

**Statement:**
```lean
specification forward_correctness (storage : Storage) (p : Layout) (length : Nat) :
    Spec.Real (contract (original storage p length) p length)
```

**Closed-form spec defs (transitive):** `Layout`, `contract`, `original`, `mask`, `complexOutput`, `offset`, `transition`, `gradOutput`, `columns`, `initial`, `output`, `Outside`, `complexBackward`, `complexInput`, `indexCommuted`

<details><summary><code>Layout</code></summary>

```lean
abbrev Layout := BackwardLayout
```
</details>

<details><summary><code>contract</code></summary>

```
/-- Actual finite access safety, successful flat execution, all three
independent reverse-scan outputs and the full tagged-cell frame. -/
```
```lean
def contract (kernel : ComputeKernel) (p : Layout) (length : Nat) : Prop :=
  (0<length → Separation p) →
  ∀ A : FlatAlloc, A.Disjoint → (∀ r, r ∉ A.regions → A.extent r=0) →
  ∀ u : BlockState, u.undef=(fun _ _ => 0) → Bounds p u length A.extent →
    kernel.toAlgKernel.TraceSafe A.extent u ∧
    ∃ v, exec (A.flattenKernel kernel.toAlgKernel) (A.flattenState u)=some v ∧
      (∀ t, t<length → ∀ j, mask p u j=Bool.true →
        complexOutput A v p.gx (offset p u t j)=Reference.inputGradient length
          (starRingEnd ℂ (transition p u j)) (fun q => gradOutput p u q j) t) ∧
      (∀ j, mask p u j=Bool.true → complexOutput A v p.gs (columns p u j)=
        Reference.reverseCarry length (starRingEnd ℂ (transition p u j)) (fun q => gradOutput p u q j) length) ∧
      (∀ j, mask p u j=Bool.true → complexOutput A v p.gl (columns p u j)=
        Reference.transitionGradient length (starRingEnd ℂ (transition p u j)) (starRingEnd ℂ (initial p u j))
          (fun q => starRingEnd ℂ (output p u q j)) (fun q => gradOutput p u q j) length) ∧
      (∀ r o, Outside p A u length r o → v.mem r o=(A.flattenState u).mem r o)
```
</details>

<details><summary><code>original</code></summary>

```lean
def original (p : Layout) (length : Nat) : ComputeKernel :=
  (TypedSource.complexBackward Bool.false p.s p.l p.y p.gs p.gx p.gl p.gy length p.batch p.dim p.B).eraseDType
```
</details>

<details><summary><code>mask</code></summary>

```lean
def mask (p : Layout) (u : BlockState) (j : Lane p) : Bool := decide (columns p u j<p.batch*p.dim)
```
</details>

<details><summary><code>complexOutput</code></summary>

```
/-- Relocate components separately; the allocation base may be odd. -/
```
```lean
def complexOutput (A : FlatAlloc) (v : BlockState) (r : RegionName) (offset : Nat) : ℂ :=
  ⟨v.readMem A.flat (A.addr r (offset*2)),v.readMem A.flat (A.addr r (offset*2+1))⟩
```
</details>

<details><summary><code>offset</code></summary>

```lean
def offset (p : Layout) (u : BlockState) (t : Nat) (j : Lane p) : Nat := t*(p.batch*p.dim)+columns p u j
```
</details>

<details><summary><code>transition</code></summary>

```lean
def transition (p : Layout) (u : BlockState) : Lane p → ℂ :=
  complexInput u p.l (fun j => columns p u j%p.dim) (mask p u)
```
</details>

<details><summary><code>gradOutput</code></summary>

```lean
def gradOutput (p : Layout) (u : BlockState) (t : Nat) : Lane p → ℂ :=
  complexInput u p.gy (offset p u t) (mask p u)

/-- Sufficient alias conditions for nonempty reverse scans. All terminal
stores follow the reads; input regions can alias one another. -/
structure Separation (p : Layout) : Prop where
  initial : p.s≠p.gx
  output : p.y≠p.gx
  gradient : p.gy≠p.gx
  terminal : p.gs≠p.gl
  initialGradient : p.gx≠p.gs
  transitionGradient : p.gx≠p.gl

structure Bounds (p : Layout) (u : BlockState) (length : Nat) (bounds : RegionBounds) : Prop where
  initial : 0<length → ∀ j, mask p u j=Bool.true → columns p u j*2+1 < bounds p.s
  transition : ∀ j, mask p u j=Bool.true → (columns p u j%p.dim)*2+1 < bounds p.l
  previous : ∀ t, t+1<length → ∀ j, mask p u j=Bool.true → offset p u t j*2+1 < bounds p.y
  gradOutput : ∀ t, t<length → ∀ j, mask p u j=Bool.true → offset p u t j*2+1 < bounds p.gy
  gradInput : ∀ t, t<length → ∀ j, mask p u j=Bool.true → offset p u t j*2+1 < bounds p.gx
  gradInitial : ∀ j, mask p u j=Bool.true → columns p u j*2+1 < bounds p.gs
  gradTransition : ∀ j, mask p u j=Bool.true → columns p u j*2+1 < bounds p.gl
```
</details>

<details><summary><code>columns</code></summary>

```lean
def columns (p : Layout) (u : BlockState) (j : Lane p) : Nat := u.pids 0*p.B+j.1.val
```
</details>

<details><summary><code>initial</code></summary>

```lean
def initial (p : Layout) (u : BlockState) : Lane p → ℂ := complexInput u p.s (columns p u) (mask p u)
```
</details>

<details><summary><code>output</code></summary>

```lean
def output (p : Layout) (u : BlockState) (t : Nat) : Lane p → ℂ :=
  complexInput u p.y (offset p u t) (mask p u)
```
</details>

<details><summary><code>Outside</code></summary>

```lean
def Outside (p : Layout) (A : FlatAlloc) (u : BlockState) (length : Nat)
    (r : RegionName) (o : Nat) : Prop :=
  r≠A.flat ∨
    (∀ t, t<length → ∀ j, mask p u j=Bool.true →
      A.addr p.gx (offset p u t j*2)≠o ∧ A.addr p.gx (offset p u t j*2+1)≠o) ∧
    (∀ j, mask p u j=Bool.true → A.addr p.gs (columns p u j*2)≠o ∧ A.addr p.gs (columns p u j*2+1)≠o) ∧
    (∀ j, mask p u j=Bool.true → A.addr p.gl (columns p u j*2)≠o ∧ A.addr p.gl (columns p u j*2+1)≠o)
```
</details>

<details><summary><code>complexBackward</code></summary>

```lean
def complexBackward (swapped : Bool) : complexBackwardSource :=
  if swapped then diagSsmComplexBackwardSource%[tl.float32, indexCommuted]
  else diagSsmComplexBackwardSource%[tl.float32, original]

end VeriTile.Bench.TritonBenchG.DiagSsmTriton.TypedSource

namespace VeriTile.Bench.TritonBenchG.DiagSsmTriton
open VeriTile.Triton

/-- Buffer names and logical sizes shared by the real and complex forward
specifications. B is the source block size; length is supplied separately. -/
structure ForwardLayout where
  s : RegionName
  x : RegionName
  y : RegionName
  l : RegionName
  batch : Nat
  dim : Nat
  B : Nat

/-- Backward buffers contain one initial state/transition per column,
time-dependent y/grad_y/grad_x and terminal grad_s/grad_lambda. -/
structure BackwardLayout where
  s : RegionName
  l : RegionName
  y : RegionName
  gs : RegionName
  gx : RegionName
  gl : RegionName
  gy : RegionName
  batch : Nat
  dim : Nat
  B : Nat
```
</details>

<details><summary><code>complexInput</code></summary>

```
/-- Read the two masked components of an original complex input. -/
```
```lean
def complexInput {B : Nat} (u : BlockState) (r : RegionName) (offset : TileIndex [B] → Nat)
    (mask : TileIndex [B] → Bool) : TileIndex [B] → ℂ := fun j =>
  ⟨if mask j then u.readMem r (offset j*2) else 0,
   if mask j then u.readMem r (offset j*2+1) else 0⟩
```
</details>

<details><summary><code>indexCommuted</code></summary>

```lean
def indexCommuted (p : Layout) (length : Nat) : ComputeKernel :=
  (TypedSource.complexBackward Bool.true p.s p.l p.y p.gs p.gx p.gl p.gy length p.batch p.dim p.B).eraseDType
```
</details>

## Public theorem: `forward_index_commuted_correctness`

**Statement:**
```lean
specification forward_index_commuted_correctness (storage : Storage) (p : Layout) (length : Nat) :
    Spec.Real (contract (indexCommuted storage p length) p length)
```

**Closed-form spec defs (transitive):** `Layout`, `contract`, `indexCommuted`, `mask`, `complexOutput`, `offset`, `transition`, `gradOutput`, `columns`, `initial`, `output`, `Outside`, `complexBackward`, `complexInput`, `original`

<details><summary><code>Layout</code></summary>

```lean
abbrev Layout := BackwardLayout
```
</details>

<details><summary><code>contract</code></summary>

```
/-- Actual finite access safety, successful flat execution, all three
independent reverse-scan outputs and the full tagged-cell frame. -/
```
```lean
def contract (kernel : ComputeKernel) (p : Layout) (length : Nat) : Prop :=
  (0<length → Separation p) →
  ∀ A : FlatAlloc, A.Disjoint → (∀ r, r ∉ A.regions → A.extent r=0) →
  ∀ u : BlockState, u.undef=(fun _ _ => 0) → Bounds p u length A.extent →
    kernel.toAlgKernel.TraceSafe A.extent u ∧
    ∃ v, exec (A.flattenKernel kernel.toAlgKernel) (A.flattenState u)=some v ∧
      (∀ t, t<length → ∀ j, mask p u j=Bool.true →
        complexOutput A v p.gx (offset p u t j)=Reference.inputGradient length
          (starRingEnd ℂ (transition p u j)) (fun q => gradOutput p u q j) t) ∧
      (∀ j, mask p u j=Bool.true → complexOutput A v p.gs (columns p u j)=
        Reference.reverseCarry length (starRingEnd ℂ (transition p u j)) (fun q => gradOutput p u q j) length) ∧
      (∀ j, mask p u j=Bool.true → complexOutput A v p.gl (columns p u j)=
        Reference.transitionGradient length (starRingEnd ℂ (transition p u j)) (starRingEnd ℂ (initial p u j))
          (fun q => starRingEnd ℂ (output p u q j)) (fun q => gradOutput p u q j) length) ∧
      (∀ r o, Outside p A u length r o → v.mem r o=(A.flattenState u).mem r o)
```
</details>

<details><summary><code>indexCommuted</code></summary>

```lean
def indexCommuted (p : Layout) (length : Nat) : ComputeKernel :=
  (TypedSource.complexBackward Bool.true p.s p.l p.y p.gs p.gx p.gl p.gy length p.batch p.dim p.B).eraseDType
```
</details>

<details><summary><code>mask</code></summary>

```lean
def mask (p : Layout) (u : BlockState) (j : Lane p) : Bool := decide (columns p u j<p.batch*p.dim)
```
</details>

<details><summary><code>complexOutput</code></summary>

```
/-- Relocate components separately; the allocation base may be odd. -/
```
```lean
def complexOutput (A : FlatAlloc) (v : BlockState) (r : RegionName) (offset : Nat) : ℂ :=
  ⟨v.readMem A.flat (A.addr r (offset*2)),v.readMem A.flat (A.addr r (offset*2+1))⟩
```
</details>

<details><summary><code>offset</code></summary>

```lean
def offset (p : Layout) (u : BlockState) (t : Nat) (j : Lane p) : Nat := t*(p.batch*p.dim)+columns p u j
```
</details>

<details><summary><code>transition</code></summary>

```lean
def transition (p : Layout) (u : BlockState) : Lane p → ℂ :=
  complexInput u p.l (fun j => columns p u j%p.dim) (mask p u)
```
</details>

<details><summary><code>gradOutput</code></summary>

```lean
def gradOutput (p : Layout) (u : BlockState) (t : Nat) : Lane p → ℂ :=
  complexInput u p.gy (offset p u t) (mask p u)

/-- Sufficient alias conditions for nonempty reverse scans. All terminal
stores follow the reads; input regions can alias one another. -/
structure Separation (p : Layout) : Prop where
  initial : p.s≠p.gx
  output : p.y≠p.gx
  gradient : p.gy≠p.gx
  terminal : p.gs≠p.gl
  initialGradient : p.gx≠p.gs
  transitionGradient : p.gx≠p.gl

structure Bounds (p : Layout) (u : BlockState) (length : Nat) (bounds : RegionBounds) : Prop where
  initial : 0<length → ∀ j, mask p u j=Bool.true → columns p u j*2+1 < bounds p.s
  transition : ∀ j, mask p u j=Bool.true → (columns p u j%p.dim)*2+1 < bounds p.l
  previous : ∀ t, t+1<length → ∀ j, mask p u j=Bool.true → offset p u t j*2+1 < bounds p.y
  gradOutput : ∀ t, t<length → ∀ j, mask p u j=Bool.true → offset p u t j*2+1 < bounds p.gy
  gradInput : ∀ t, t<length → ∀ j, mask p u j=Bool.true → offset p u t j*2+1 < bounds p.gx
  gradInitial : ∀ j, mask p u j=Bool.true → columns p u j*2+1 < bounds p.gs
  gradTransition : ∀ j, mask p u j=Bool.true → columns p u j*2+1 < bounds p.gl
```
</details>

<details><summary><code>columns</code></summary>

```lean
def columns (p : Layout) (u : BlockState) (j : Lane p) : Nat := u.pids 0*p.B+j.1.val
```
</details>

<details><summary><code>initial</code></summary>

```lean
def initial (p : Layout) (u : BlockState) : Lane p → ℂ := complexInput u p.s (columns p u) (mask p u)
```
</details>

<details><summary><code>output</code></summary>

```lean
def output (p : Layout) (u : BlockState) (t : Nat) : Lane p → ℂ :=
  complexInput u p.y (offset p u t) (mask p u)
```
</details>

<details><summary><code>Outside</code></summary>

```lean
def Outside (p : Layout) (A : FlatAlloc) (u : BlockState) (length : Nat)
    (r : RegionName) (o : Nat) : Prop :=
  r≠A.flat ∨
    (∀ t, t<length → ∀ j, mask p u j=Bool.true →
      A.addr p.gx (offset p u t j*2)≠o ∧ A.addr p.gx (offset p u t j*2+1)≠o) ∧
    (∀ j, mask p u j=Bool.true → A.addr p.gs (columns p u j*2)≠o ∧ A.addr p.gs (columns p u j*2+1)≠o) ∧
    (∀ j, mask p u j=Bool.true → A.addr p.gl (columns p u j*2)≠o ∧ A.addr p.gl (columns p u j*2+1)≠o)
```
</details>

<details><summary><code>complexBackward</code></summary>

```lean
def complexBackward (swapped : Bool) : complexBackwardSource :=
  if swapped then diagSsmComplexBackwardSource%[tl.float32, indexCommuted]
  else diagSsmComplexBackwardSource%[tl.float32, original]

end VeriTile.Bench.TritonBenchG.DiagSsmTriton.TypedSource

namespace VeriTile.Bench.TritonBenchG.DiagSsmTriton
open VeriTile.Triton

/-- Buffer names and logical sizes shared by the real and complex forward
specifications. B is the source block size; length is supplied separately. -/
structure ForwardLayout where
  s : RegionName
  x : RegionName
  y : RegionName
  l : RegionName
  batch : Nat
  dim : Nat
  B : Nat

/-- Backward buffers contain one initial state/transition per column,
time-dependent y/grad_y/grad_x and terminal grad_s/grad_lambda. -/
structure BackwardLayout where
  s : RegionName
  l : RegionName
  y : RegionName
  gs : RegionName
  gx : RegionName
  gl : RegionName
  gy : RegionName
  batch : Nat
  dim : Nat
  B : Nat
```
</details>

<details><summary><code>complexInput</code></summary>

```
/-- Read the two masked components of an original complex input. -/
```
```lean
def complexInput {B : Nat} (u : BlockState) (r : RegionName) (offset : TileIndex [B] → Nat)
    (mask : TileIndex [B] → Bool) : TileIndex [B] → ℂ := fun j =>
  ⟨if mask j then u.readMem r (offset j*2) else 0,
   if mask j then u.readMem r (offset j*2+1) else 0⟩
```
</details>

<details><summary><code>original</code></summary>

```lean
def original (p : Layout) (length : Nat) : ComputeKernel :=
  (TypedSource.complexBackward Bool.false p.s p.l p.y p.gs p.gx p.gl p.gy length p.batch p.dim p.B).eraseDType
```
</details>

## Public theorem: `backward_correctness`

**Statement:**
```lean
specification backward_correctness (storage : Storage) (p : Layout) (length : Nat) :
    Spec.Real (contract (original storage p length) p length)
```

**Closed-form spec defs (transitive):** `Layout`, `contract`, `original`, `mask`, `complexOutput`, `offset`, `transition`, `gradOutput`, `columns`, `initial`, `output`, `Outside`, `complexBackward`, `complexInput`, `indexCommuted`

<details><summary><code>Layout</code></summary>

```lean
abbrev Layout := BackwardLayout
```
</details>

<details><summary><code>contract</code></summary>

```
/-- Actual finite access safety, successful flat execution, all three
independent reverse-scan outputs and the full tagged-cell frame. -/
```
```lean
def contract (kernel : ComputeKernel) (p : Layout) (length : Nat) : Prop :=
  (0<length → Separation p) →
  ∀ A : FlatAlloc, A.Disjoint → (∀ r, r ∉ A.regions → A.extent r=0) →
  ∀ u : BlockState, u.undef=(fun _ _ => 0) → Bounds p u length A.extent →
    kernel.toAlgKernel.TraceSafe A.extent u ∧
    ∃ v, exec (A.flattenKernel kernel.toAlgKernel) (A.flattenState u)=some v ∧
      (∀ t, t<length → ∀ j, mask p u j=Bool.true →
        complexOutput A v p.gx (offset p u t j)=Reference.inputGradient length
          (starRingEnd ℂ (transition p u j)) (fun q => gradOutput p u q j) t) ∧
      (∀ j, mask p u j=Bool.true → complexOutput A v p.gs (columns p u j)=
        Reference.reverseCarry length (starRingEnd ℂ (transition p u j)) (fun q => gradOutput p u q j) length) ∧
      (∀ j, mask p u j=Bool.true → complexOutput A v p.gl (columns p u j)=
        Reference.transitionGradient length (starRingEnd ℂ (transition p u j)) (starRingEnd ℂ (initial p u j))
          (fun q => starRingEnd ℂ (output p u q j)) (fun q => gradOutput p u q j) length) ∧
      (∀ r o, Outside p A u length r o → v.mem r o=(A.flattenState u).mem r o)
```
</details>

<details><summary><code>original</code></summary>

```lean
def original (p : Layout) (length : Nat) : ComputeKernel :=
  (TypedSource.complexBackward Bool.false p.s p.l p.y p.gs p.gx p.gl p.gy length p.batch p.dim p.B).eraseDType
```
</details>

<details><summary><code>mask</code></summary>

```lean
def mask (p : Layout) (u : BlockState) (j : Lane p) : Bool := decide (columns p u j<p.batch*p.dim)
```
</details>

<details><summary><code>complexOutput</code></summary>

```
/-- Relocate components separately; the allocation base may be odd. -/
```
```lean
def complexOutput (A : FlatAlloc) (v : BlockState) (r : RegionName) (offset : Nat) : ℂ :=
  ⟨v.readMem A.flat (A.addr r (offset*2)),v.readMem A.flat (A.addr r (offset*2+1))⟩
```
</details>

<details><summary><code>offset</code></summary>

```lean
def offset (p : Layout) (u : BlockState) (t : Nat) (j : Lane p) : Nat := t*(p.batch*p.dim)+columns p u j
```
</details>

<details><summary><code>transition</code></summary>

```lean
def transition (p : Layout) (u : BlockState) : Lane p → ℂ :=
  complexInput u p.l (fun j => columns p u j%p.dim) (mask p u)
```
</details>

<details><summary><code>gradOutput</code></summary>

```lean
def gradOutput (p : Layout) (u : BlockState) (t : Nat) : Lane p → ℂ :=
  complexInput u p.gy (offset p u t) (mask p u)

/-- Sufficient alias conditions for nonempty reverse scans. All terminal
stores follow the reads; input regions can alias one another. -/
structure Separation (p : Layout) : Prop where
  initial : p.s≠p.gx
  output : p.y≠p.gx
  gradient : p.gy≠p.gx
  terminal : p.gs≠p.gl
  initialGradient : p.gx≠p.gs
  transitionGradient : p.gx≠p.gl

structure Bounds (p : Layout) (u : BlockState) (length : Nat) (bounds : RegionBounds) : Prop where
  initial : 0<length → ∀ j, mask p u j=Bool.true → columns p u j*2+1 < bounds p.s
  transition : ∀ j, mask p u j=Bool.true → (columns p u j%p.dim)*2+1 < bounds p.l
  previous : ∀ t, t+1<length → ∀ j, mask p u j=Bool.true → offset p u t j*2+1 < bounds p.y
  gradOutput : ∀ t, t<length → ∀ j, mask p u j=Bool.true → offset p u t j*2+1 < bounds p.gy
  gradInput : ∀ t, t<length → ∀ j, mask p u j=Bool.true → offset p u t j*2+1 < bounds p.gx
  gradInitial : ∀ j, mask p u j=Bool.true → columns p u j*2+1 < bounds p.gs
  gradTransition : ∀ j, mask p u j=Bool.true → columns p u j*2+1 < bounds p.gl
```
</details>

<details><summary><code>columns</code></summary>

```lean
def columns (p : Layout) (u : BlockState) (j : Lane p) : Nat := u.pids 0*p.B+j.1.val
```
</details>

<details><summary><code>initial</code></summary>

```lean
def initial (p : Layout) (u : BlockState) : Lane p → ℂ := complexInput u p.s (columns p u) (mask p u)
```
</details>

<details><summary><code>output</code></summary>

```lean
def output (p : Layout) (u : BlockState) (t : Nat) : Lane p → ℂ :=
  complexInput u p.y (offset p u t) (mask p u)
```
</details>

<details><summary><code>Outside</code></summary>

```lean
def Outside (p : Layout) (A : FlatAlloc) (u : BlockState) (length : Nat)
    (r : RegionName) (o : Nat) : Prop :=
  r≠A.flat ∨
    (∀ t, t<length → ∀ j, mask p u j=Bool.true →
      A.addr p.gx (offset p u t j*2)≠o ∧ A.addr p.gx (offset p u t j*2+1)≠o) ∧
    (∀ j, mask p u j=Bool.true → A.addr p.gs (columns p u j*2)≠o ∧ A.addr p.gs (columns p u j*2+1)≠o) ∧
    (∀ j, mask p u j=Bool.true → A.addr p.gl (columns p u j*2)≠o ∧ A.addr p.gl (columns p u j*2+1)≠o)
```
</details>

<details><summary><code>complexBackward</code></summary>

```lean
def complexBackward (swapped : Bool) : complexBackwardSource :=
  if swapped then diagSsmComplexBackwardSource%[tl.float32, indexCommuted]
  else diagSsmComplexBackwardSource%[tl.float32, original]

end VeriTile.Bench.TritonBenchG.DiagSsmTriton.TypedSource

namespace VeriTile.Bench.TritonBenchG.DiagSsmTriton
open VeriTile.Triton

/-- Buffer names and logical sizes shared by the real and complex forward
specifications. B is the source block size; length is supplied separately. -/
structure ForwardLayout where
  s : RegionName
  x : RegionName
  y : RegionName
  l : RegionName
  batch : Nat
  dim : Nat
  B : Nat

/-- Backward buffers contain one initial state/transition per column,
time-dependent y/grad_y/grad_x and terminal grad_s/grad_lambda. -/
structure BackwardLayout where
  s : RegionName
  l : RegionName
  y : RegionName
  gs : RegionName
  gx : RegionName
  gl : RegionName
  gy : RegionName
  batch : Nat
  dim : Nat
  B : Nat
```
</details>

<details><summary><code>complexInput</code></summary>

```
/-- Read the two masked components of an original complex input. -/
```
```lean
def complexInput {B : Nat} (u : BlockState) (r : RegionName) (offset : TileIndex [B] → Nat)
    (mask : TileIndex [B] → Bool) : TileIndex [B] → ℂ := fun j =>
  ⟨if mask j then u.readMem r (offset j*2) else 0,
   if mask j then u.readMem r (offset j*2+1) else 0⟩
```
</details>

<details><summary><code>indexCommuted</code></summary>

```lean
def indexCommuted (p : Layout) (length : Nat) : ComputeKernel :=
  (TypedSource.complexBackward Bool.true p.s p.l p.y p.gs p.gx p.gl p.gy length p.batch p.dim p.B).eraseDType
```
</details>

## Public theorem: `backward_index_commuted_correctness`

**Statement:**
```lean
specification backward_index_commuted_correctness (storage : Storage) (p : Layout) (length : Nat) :
    Spec.Real (contract (indexCommuted storage p length) p length)
```

**Closed-form spec defs (transitive):** `Layout`, `contract`, `indexCommuted`, `mask`, `complexOutput`, `offset`, `transition`, `gradOutput`, `columns`, `initial`, `output`, `Outside`, `complexBackward`, `complexInput`, `original`

<details><summary><code>Layout</code></summary>

```lean
abbrev Layout := BackwardLayout
```
</details>

<details><summary><code>contract</code></summary>

```
/-- Actual finite access safety, successful flat execution, all three
independent reverse-scan outputs and the full tagged-cell frame. -/
```
```lean
def contract (kernel : ComputeKernel) (p : Layout) (length : Nat) : Prop :=
  (0<length → Separation p) →
  ∀ A : FlatAlloc, A.Disjoint → (∀ r, r ∉ A.regions → A.extent r=0) →
  ∀ u : BlockState, u.undef=(fun _ _ => 0) → Bounds p u length A.extent →
    kernel.toAlgKernel.TraceSafe A.extent u ∧
    ∃ v, exec (A.flattenKernel kernel.toAlgKernel) (A.flattenState u)=some v ∧
      (∀ t, t<length → ∀ j, mask p u j=Bool.true →
        complexOutput A v p.gx (offset p u t j)=Reference.inputGradient length
          (starRingEnd ℂ (transition p u j)) (fun q => gradOutput p u q j) t) ∧
      (∀ j, mask p u j=Bool.true → complexOutput A v p.gs (columns p u j)=
        Reference.reverseCarry length (starRingEnd ℂ (transition p u j)) (fun q => gradOutput p u q j) length) ∧
      (∀ j, mask p u j=Bool.true → complexOutput A v p.gl (columns p u j)=
        Reference.transitionGradient length (starRingEnd ℂ (transition p u j)) (starRingEnd ℂ (initial p u j))
          (fun q => starRingEnd ℂ (output p u q j)) (fun q => gradOutput p u q j) length) ∧
      (∀ r o, Outside p A u length r o → v.mem r o=(A.flattenState u).mem r o)
```
</details>

<details><summary><code>indexCommuted</code></summary>

```lean
def indexCommuted (p : Layout) (length : Nat) : ComputeKernel :=
  (TypedSource.complexBackward Bool.true p.s p.l p.y p.gs p.gx p.gl p.gy length p.batch p.dim p.B).eraseDType
```
</details>

<details><summary><code>mask</code></summary>

```lean
def mask (p : Layout) (u : BlockState) (j : Lane p) : Bool := decide (columns p u j<p.batch*p.dim)
```
</details>

<details><summary><code>complexOutput</code></summary>

```
/-- Relocate components separately; the allocation base may be odd. -/
```
```lean
def complexOutput (A : FlatAlloc) (v : BlockState) (r : RegionName) (offset : Nat) : ℂ :=
  ⟨v.readMem A.flat (A.addr r (offset*2)),v.readMem A.flat (A.addr r (offset*2+1))⟩
```
</details>

<details><summary><code>offset</code></summary>

```lean
def offset (p : Layout) (u : BlockState) (t : Nat) (j : Lane p) : Nat := t*(p.batch*p.dim)+columns p u j
```
</details>

<details><summary><code>transition</code></summary>

```lean
def transition (p : Layout) (u : BlockState) : Lane p → ℂ :=
  complexInput u p.l (fun j => columns p u j%p.dim) (mask p u)
```
</details>

<details><summary><code>gradOutput</code></summary>

```lean
def gradOutput (p : Layout) (u : BlockState) (t : Nat) : Lane p → ℂ :=
  complexInput u p.gy (offset p u t) (mask p u)

/-- Sufficient alias conditions for nonempty reverse scans. All terminal
stores follow the reads; input regions can alias one another. -/
structure Separation (p : Layout) : Prop where
  initial : p.s≠p.gx
  output : p.y≠p.gx
  gradient : p.gy≠p.gx
  terminal : p.gs≠p.gl
  initialGradient : p.gx≠p.gs
  transitionGradient : p.gx≠p.gl

structure Bounds (p : Layout) (u : BlockState) (length : Nat) (bounds : RegionBounds) : Prop where
  initial : 0<length → ∀ j, mask p u j=Bool.true → columns p u j*2+1 < bounds p.s
  transition : ∀ j, mask p u j=Bool.true → (columns p u j%p.dim)*2+1 < bounds p.l
  previous : ∀ t, t+1<length → ∀ j, mask p u j=Bool.true → offset p u t j*2+1 < bounds p.y
  gradOutput : ∀ t, t<length → ∀ j, mask p u j=Bool.true → offset p u t j*2+1 < bounds p.gy
  gradInput : ∀ t, t<length → ∀ j, mask p u j=Bool.true → offset p u t j*2+1 < bounds p.gx
  gradInitial : ∀ j, mask p u j=Bool.true → columns p u j*2+1 < bounds p.gs
  gradTransition : ∀ j, mask p u j=Bool.true → columns p u j*2+1 < bounds p.gl
```
</details>

<details><summary><code>columns</code></summary>

```lean
def columns (p : Layout) (u : BlockState) (j : Lane p) : Nat := u.pids 0*p.B+j.1.val
```
</details>

<details><summary><code>initial</code></summary>

```lean
def initial (p : Layout) (u : BlockState) : Lane p → ℂ := complexInput u p.s (columns p u) (mask p u)
```
</details>

<details><summary><code>output</code></summary>

```lean
def output (p : Layout) (u : BlockState) (t : Nat) : Lane p → ℂ :=
  complexInput u p.y (offset p u t) (mask p u)
```
</details>

<details><summary><code>Outside</code></summary>

```lean
def Outside (p : Layout) (A : FlatAlloc) (u : BlockState) (length : Nat)
    (r : RegionName) (o : Nat) : Prop :=
  r≠A.flat ∨
    (∀ t, t<length → ∀ j, mask p u j=Bool.true →
      A.addr p.gx (offset p u t j*2)≠o ∧ A.addr p.gx (offset p u t j*2+1)≠o) ∧
    (∀ j, mask p u j=Bool.true → A.addr p.gs (columns p u j*2)≠o ∧ A.addr p.gs (columns p u j*2+1)≠o) ∧
    (∀ j, mask p u j=Bool.true → A.addr p.gl (columns p u j*2)≠o ∧ A.addr p.gl (columns p u j*2+1)≠o)
```
</details>

<details><summary><code>complexBackward</code></summary>

```lean
def complexBackward (swapped : Bool) : complexBackwardSource :=
  if swapped then diagSsmComplexBackwardSource%[tl.float32, indexCommuted]
  else diagSsmComplexBackwardSource%[tl.float32, original]

end VeriTile.Bench.TritonBenchG.DiagSsmTriton.TypedSource

namespace VeriTile.Bench.TritonBenchG.DiagSsmTriton
open VeriTile.Triton

/-- Buffer names and logical sizes shared by the real and complex forward
specifications. B is the source block size; length is supplied separately. -/
structure ForwardLayout where
  s : RegionName
  x : RegionName
  y : RegionName
  l : RegionName
  batch : Nat
  dim : Nat
  B : Nat

/-- Backward buffers contain one initial state/transition per column,
time-dependent y/grad_y/grad_x and terminal grad_s/grad_lambda. -/
structure BackwardLayout where
  s : RegionName
  l : RegionName
  y : RegionName
  gs : RegionName
  gx : RegionName
  gl : RegionName
  gy : RegionName
  batch : Nat
  dim : Nat
  B : Nat
```
</details>

<details><summary><code>complexInput</code></summary>

```
/-- Read the two masked components of an original complex input. -/
```
```lean
def complexInput {B : Nat} (u : BlockState) (r : RegionName) (offset : TileIndex [B] → Nat)
    (mask : TileIndex [B] → Bool) : TileIndex [B] → ℂ := fun j =>
  ⟨if mask j then u.readMem r (offset j*2) else 0,
   if mask j then u.readMem r (offset j*2+1) else 0⟩
```
</details>

<details><summary><code>original</code></summary>

```lean
def original (p : Layout) (length : Nat) : ComputeKernel :=
  (TypedSource.complexBackward Bool.false p.s p.l p.y p.gs p.gx p.gl p.gy length p.batch p.dim p.B).eraseDType
```
</details>

## Public theorem: `complex_forward_correctness`

**Statement:**
```lean
specification complex_forward_correctness (p : Layout) (length : Nat) :
    Spec.Real (contract (original p length) p length)
```

**Closed-form spec defs (transitive):** `Layout`, `contract`, `original`, `mask`, `complexOutput`, `offset`, `transition`, `gradOutput`, `columns`, `initial`, `output`, `Outside`, `complexBackward`, `complexInput`, `indexCommuted`

<details><summary><code>Layout</code></summary>

```lean
abbrev Layout := BackwardLayout
```
</details>

<details><summary><code>contract</code></summary>

```
/-- Actual finite access safety, successful flat execution, all three
independent reverse-scan outputs and the full tagged-cell frame. -/
```
```lean
def contract (kernel : ComputeKernel) (p : Layout) (length : Nat) : Prop :=
  (0<length → Separation p) →
  ∀ A : FlatAlloc, A.Disjoint → (∀ r, r ∉ A.regions → A.extent r=0) →
  ∀ u : BlockState, u.undef=(fun _ _ => 0) → Bounds p u length A.extent →
    kernel.toAlgKernel.TraceSafe A.extent u ∧
    ∃ v, exec (A.flattenKernel kernel.toAlgKernel) (A.flattenState u)=some v ∧
      (∀ t, t<length → ∀ j, mask p u j=Bool.true →
        complexOutput A v p.gx (offset p u t j)=Reference.inputGradient length
          (starRingEnd ℂ (transition p u j)) (fun q => gradOutput p u q j) t) ∧
      (∀ j, mask p u j=Bool.true → complexOutput A v p.gs (columns p u j)=
        Reference.reverseCarry length (starRingEnd ℂ (transition p u j)) (fun q => gradOutput p u q j) length) ∧
      (∀ j, mask p u j=Bool.true → complexOutput A v p.gl (columns p u j)=
        Reference.transitionGradient length (starRingEnd ℂ (transition p u j)) (starRingEnd ℂ (initial p u j))
          (fun q => starRingEnd ℂ (output p u q j)) (fun q => gradOutput p u q j) length) ∧
      (∀ r o, Outside p A u length r o → v.mem r o=(A.flattenState u).mem r o)
```
</details>

<details><summary><code>original</code></summary>

```lean
def original (p : Layout) (length : Nat) : ComputeKernel :=
  (TypedSource.complexBackward Bool.false p.s p.l p.y p.gs p.gx p.gl p.gy length p.batch p.dim p.B).eraseDType
```
</details>

<details><summary><code>mask</code></summary>

```lean
def mask (p : Layout) (u : BlockState) (j : Lane p) : Bool := decide (columns p u j<p.batch*p.dim)
```
</details>

<details><summary><code>complexOutput</code></summary>

```
/-- Relocate components separately; the allocation base may be odd. -/
```
```lean
def complexOutput (A : FlatAlloc) (v : BlockState) (r : RegionName) (offset : Nat) : ℂ :=
  ⟨v.readMem A.flat (A.addr r (offset*2)),v.readMem A.flat (A.addr r (offset*2+1))⟩
```
</details>

<details><summary><code>offset</code></summary>

```lean
def offset (p : Layout) (u : BlockState) (t : Nat) (j : Lane p) : Nat := t*(p.batch*p.dim)+columns p u j
```
</details>

<details><summary><code>transition</code></summary>

```lean
def transition (p : Layout) (u : BlockState) : Lane p → ℂ :=
  complexInput u p.l (fun j => columns p u j%p.dim) (mask p u)
```
</details>

<details><summary><code>gradOutput</code></summary>

```lean
def gradOutput (p : Layout) (u : BlockState) (t : Nat) : Lane p → ℂ :=
  complexInput u p.gy (offset p u t) (mask p u)

/-- Sufficient alias conditions for nonempty reverse scans. All terminal
stores follow the reads; input regions can alias one another. -/
structure Separation (p : Layout) : Prop where
  initial : p.s≠p.gx
  output : p.y≠p.gx
  gradient : p.gy≠p.gx
  terminal : p.gs≠p.gl
  initialGradient : p.gx≠p.gs
  transitionGradient : p.gx≠p.gl

structure Bounds (p : Layout) (u : BlockState) (length : Nat) (bounds : RegionBounds) : Prop where
  initial : 0<length → ∀ j, mask p u j=Bool.true → columns p u j*2+1 < bounds p.s
  transition : ∀ j, mask p u j=Bool.true → (columns p u j%p.dim)*2+1 < bounds p.l
  previous : ∀ t, t+1<length → ∀ j, mask p u j=Bool.true → offset p u t j*2+1 < bounds p.y
  gradOutput : ∀ t, t<length → ∀ j, mask p u j=Bool.true → offset p u t j*2+1 < bounds p.gy
  gradInput : ∀ t, t<length → ∀ j, mask p u j=Bool.true → offset p u t j*2+1 < bounds p.gx
  gradInitial : ∀ j, mask p u j=Bool.true → columns p u j*2+1 < bounds p.gs
  gradTransition : ∀ j, mask p u j=Bool.true → columns p u j*2+1 < bounds p.gl
```
</details>

<details><summary><code>columns</code></summary>

```lean
def columns (p : Layout) (u : BlockState) (j : Lane p) : Nat := u.pids 0*p.B+j.1.val
```
</details>

<details><summary><code>initial</code></summary>

```lean
def initial (p : Layout) (u : BlockState) : Lane p → ℂ := complexInput u p.s (columns p u) (mask p u)
```
</details>

<details><summary><code>output</code></summary>

```lean
def output (p : Layout) (u : BlockState) (t : Nat) : Lane p → ℂ :=
  complexInput u p.y (offset p u t) (mask p u)
```
</details>

<details><summary><code>Outside</code></summary>

```lean
def Outside (p : Layout) (A : FlatAlloc) (u : BlockState) (length : Nat)
    (r : RegionName) (o : Nat) : Prop :=
  r≠A.flat ∨
    (∀ t, t<length → ∀ j, mask p u j=Bool.true →
      A.addr p.gx (offset p u t j*2)≠o ∧ A.addr p.gx (offset p u t j*2+1)≠o) ∧
    (∀ j, mask p u j=Bool.true → A.addr p.gs (columns p u j*2)≠o ∧ A.addr p.gs (columns p u j*2+1)≠o) ∧
    (∀ j, mask p u j=Bool.true → A.addr p.gl (columns p u j*2)≠o ∧ A.addr p.gl (columns p u j*2+1)≠o)
```
</details>

<details><summary><code>complexBackward</code></summary>

```lean
def complexBackward (swapped : Bool) : complexBackwardSource :=
  if swapped then diagSsmComplexBackwardSource%[tl.float32, indexCommuted]
  else diagSsmComplexBackwardSource%[tl.float32, original]

end VeriTile.Bench.TritonBenchG.DiagSsmTriton.TypedSource

namespace VeriTile.Bench.TritonBenchG.DiagSsmTriton
open VeriTile.Triton

/-- Buffer names and logical sizes shared by the real and complex forward
specifications. B is the source block size; length is supplied separately. -/
structure ForwardLayout where
  s : RegionName
  x : RegionName
  y : RegionName
  l : RegionName
  batch : Nat
  dim : Nat
  B : Nat

/-- Backward buffers contain one initial state/transition per column,
time-dependent y/grad_y/grad_x and terminal grad_s/grad_lambda. -/
structure BackwardLayout where
  s : RegionName
  l : RegionName
  y : RegionName
  gs : RegionName
  gx : RegionName
  gl : RegionName
  gy : RegionName
  batch : Nat
  dim : Nat
  B : Nat
```
</details>

<details><summary><code>complexInput</code></summary>

```
/-- Read the two masked components of an original complex input. -/
```
```lean
def complexInput {B : Nat} (u : BlockState) (r : RegionName) (offset : TileIndex [B] → Nat)
    (mask : TileIndex [B] → Bool) : TileIndex [B] → ℂ := fun j =>
  ⟨if mask j then u.readMem r (offset j*2) else 0,
   if mask j then u.readMem r (offset j*2+1) else 0⟩
```
</details>

<details><summary><code>indexCommuted</code></summary>

```lean
def indexCommuted (p : Layout) (length : Nat) : ComputeKernel :=
  (TypedSource.complexBackward Bool.true p.s p.l p.y p.gs p.gx p.gl p.gy length p.batch p.dim p.B).eraseDType
```
</details>

## Public theorem: `complex_forward_index_commuted_correctness`

**Statement:**
```lean
specification complex_forward_index_commuted_correctness (p : Layout) (length : Nat) :
    Spec.Real (contract (indexCommuted p length) p length)
```

**Closed-form spec defs (transitive):** `Layout`, `contract`, `indexCommuted`, `mask`, `complexOutput`, `offset`, `transition`, `gradOutput`, `columns`, `initial`, `output`, `Outside`, `complexBackward`, `complexInput`, `original`

<details><summary><code>Layout</code></summary>

```lean
abbrev Layout := BackwardLayout
```
</details>

<details><summary><code>contract</code></summary>

```
/-- Actual finite access safety, successful flat execution, all three
independent reverse-scan outputs and the full tagged-cell frame. -/
```
```lean
def contract (kernel : ComputeKernel) (p : Layout) (length : Nat) : Prop :=
  (0<length → Separation p) →
  ∀ A : FlatAlloc, A.Disjoint → (∀ r, r ∉ A.regions → A.extent r=0) →
  ∀ u : BlockState, u.undef=(fun _ _ => 0) → Bounds p u length A.extent →
    kernel.toAlgKernel.TraceSafe A.extent u ∧
    ∃ v, exec (A.flattenKernel kernel.toAlgKernel) (A.flattenState u)=some v ∧
      (∀ t, t<length → ∀ j, mask p u j=Bool.true →
        complexOutput A v p.gx (offset p u t j)=Reference.inputGradient length
          (starRingEnd ℂ (transition p u j)) (fun q => gradOutput p u q j) t) ∧
      (∀ j, mask p u j=Bool.true → complexOutput A v p.gs (columns p u j)=
        Reference.reverseCarry length (starRingEnd ℂ (transition p u j)) (fun q => gradOutput p u q j) length) ∧
      (∀ j, mask p u j=Bool.true → complexOutput A v p.gl (columns p u j)=
        Reference.transitionGradient length (starRingEnd ℂ (transition p u j)) (starRingEnd ℂ (initial p u j))
          (fun q => starRingEnd ℂ (output p u q j)) (fun q => gradOutput p u q j) length) ∧
      (∀ r o, Outside p A u length r o → v.mem r o=(A.flattenState u).mem r o)
```
</details>

<details><summary><code>indexCommuted</code></summary>

```lean
def indexCommuted (p : Layout) (length : Nat) : ComputeKernel :=
  (TypedSource.complexBackward Bool.true p.s p.l p.y p.gs p.gx p.gl p.gy length p.batch p.dim p.B).eraseDType
```
</details>

<details><summary><code>mask</code></summary>

```lean
def mask (p : Layout) (u : BlockState) (j : Lane p) : Bool := decide (columns p u j<p.batch*p.dim)
```
</details>

<details><summary><code>complexOutput</code></summary>

```
/-- Relocate components separately; the allocation base may be odd. -/
```
```lean
def complexOutput (A : FlatAlloc) (v : BlockState) (r : RegionName) (offset : Nat) : ℂ :=
  ⟨v.readMem A.flat (A.addr r (offset*2)),v.readMem A.flat (A.addr r (offset*2+1))⟩
```
</details>

<details><summary><code>offset</code></summary>

```lean
def offset (p : Layout) (u : BlockState) (t : Nat) (j : Lane p) : Nat := t*(p.batch*p.dim)+columns p u j
```
</details>

<details><summary><code>transition</code></summary>

```lean
def transition (p : Layout) (u : BlockState) : Lane p → ℂ :=
  complexInput u p.l (fun j => columns p u j%p.dim) (mask p u)
```
</details>

<details><summary><code>gradOutput</code></summary>

```lean
def gradOutput (p : Layout) (u : BlockState) (t : Nat) : Lane p → ℂ :=
  complexInput u p.gy (offset p u t) (mask p u)

/-- Sufficient alias conditions for nonempty reverse scans. All terminal
stores follow the reads; input regions can alias one another. -/
structure Separation (p : Layout) : Prop where
  initial : p.s≠p.gx
  output : p.y≠p.gx
  gradient : p.gy≠p.gx
  terminal : p.gs≠p.gl
  initialGradient : p.gx≠p.gs
  transitionGradient : p.gx≠p.gl

structure Bounds (p : Layout) (u : BlockState) (length : Nat) (bounds : RegionBounds) : Prop where
  initial : 0<length → ∀ j, mask p u j=Bool.true → columns p u j*2+1 < bounds p.s
  transition : ∀ j, mask p u j=Bool.true → (columns p u j%p.dim)*2+1 < bounds p.l
  previous : ∀ t, t+1<length → ∀ j, mask p u j=Bool.true → offset p u t j*2+1 < bounds p.y
  gradOutput : ∀ t, t<length → ∀ j, mask p u j=Bool.true → offset p u t j*2+1 < bounds p.gy
  gradInput : ∀ t, t<length → ∀ j, mask p u j=Bool.true → offset p u t j*2+1 < bounds p.gx
  gradInitial : ∀ j, mask p u j=Bool.true → columns p u j*2+1 < bounds p.gs
  gradTransition : ∀ j, mask p u j=Bool.true → columns p u j*2+1 < bounds p.gl
```
</details>

<details><summary><code>columns</code></summary>

```lean
def columns (p : Layout) (u : BlockState) (j : Lane p) : Nat := u.pids 0*p.B+j.1.val
```
</details>

<details><summary><code>initial</code></summary>

```lean
def initial (p : Layout) (u : BlockState) : Lane p → ℂ := complexInput u p.s (columns p u) (mask p u)
```
</details>

<details><summary><code>output</code></summary>

```lean
def output (p : Layout) (u : BlockState) (t : Nat) : Lane p → ℂ :=
  complexInput u p.y (offset p u t) (mask p u)
```
</details>

<details><summary><code>Outside</code></summary>

```lean
def Outside (p : Layout) (A : FlatAlloc) (u : BlockState) (length : Nat)
    (r : RegionName) (o : Nat) : Prop :=
  r≠A.flat ∨
    (∀ t, t<length → ∀ j, mask p u j=Bool.true →
      A.addr p.gx (offset p u t j*2)≠o ∧ A.addr p.gx (offset p u t j*2+1)≠o) ∧
    (∀ j, mask p u j=Bool.true → A.addr p.gs (columns p u j*2)≠o ∧ A.addr p.gs (columns p u j*2+1)≠o) ∧
    (∀ j, mask p u j=Bool.true → A.addr p.gl (columns p u j*2)≠o ∧ A.addr p.gl (columns p u j*2+1)≠o)
```
</details>

<details><summary><code>complexBackward</code></summary>

```lean
def complexBackward (swapped : Bool) : complexBackwardSource :=
  if swapped then diagSsmComplexBackwardSource%[tl.float32, indexCommuted]
  else diagSsmComplexBackwardSource%[tl.float32, original]

end VeriTile.Bench.TritonBenchG.DiagSsmTriton.TypedSource

namespace VeriTile.Bench.TritonBenchG.DiagSsmTriton
open VeriTile.Triton

/-- Buffer names and logical sizes shared by the real and complex forward
specifications. B is the source block size; length is supplied separately. -/
structure ForwardLayout where
  s : RegionName
  x : RegionName
  y : RegionName
  l : RegionName
  batch : Nat
  dim : Nat
  B : Nat

/-- Backward buffers contain one initial state/transition per column,
time-dependent y/grad_y/grad_x and terminal grad_s/grad_lambda. -/
structure BackwardLayout where
  s : RegionName
  l : RegionName
  y : RegionName
  gs : RegionName
  gx : RegionName
  gl : RegionName
  gy : RegionName
  batch : Nat
  dim : Nat
  B : Nat
```
</details>

<details><summary><code>complexInput</code></summary>

```
/-- Read the two masked components of an original complex input. -/
```
```lean
def complexInput {B : Nat} (u : BlockState) (r : RegionName) (offset : TileIndex [B] → Nat)
    (mask : TileIndex [B] → Bool) : TileIndex [B] → ℂ := fun j =>
  ⟨if mask j then u.readMem r (offset j*2) else 0,
   if mask j then u.readMem r (offset j*2+1) else 0⟩
```
</details>

<details><summary><code>original</code></summary>

```lean
def original (p : Layout) (length : Nat) : ComputeKernel :=
  (TypedSource.complexBackward Bool.false p.s p.l p.y p.gs p.gx p.gl p.gy length p.batch p.dim p.B).eraseDType
```
</details>

## Public theorem: `complex_backward_correctness`

**Statement:**
```lean
specification complex_backward_correctness (p : Layout) (length : Nat) :
    Spec.Real (contract (original p length) p length)
```

**Closed-form spec defs (transitive):** `Layout`, `contract`, `original`, `mask`, `complexOutput`, `offset`, `transition`, `gradOutput`, `columns`, `initial`, `output`, `Outside`, `complexBackward`, `complexInput`, `indexCommuted`

<details><summary><code>Layout</code></summary>

```lean
abbrev Layout := BackwardLayout
```
</details>

<details><summary><code>contract</code></summary>

```
/-- Actual finite access safety, successful flat execution, all three
independent reverse-scan outputs and the full tagged-cell frame. -/
```
```lean
def contract (kernel : ComputeKernel) (p : Layout) (length : Nat) : Prop :=
  (0<length → Separation p) →
  ∀ A : FlatAlloc, A.Disjoint → (∀ r, r ∉ A.regions → A.extent r=0) →
  ∀ u : BlockState, u.undef=(fun _ _ => 0) → Bounds p u length A.extent →
    kernel.toAlgKernel.TraceSafe A.extent u ∧
    ∃ v, exec (A.flattenKernel kernel.toAlgKernel) (A.flattenState u)=some v ∧
      (∀ t, t<length → ∀ j, mask p u j=Bool.true →
        complexOutput A v p.gx (offset p u t j)=Reference.inputGradient length
          (starRingEnd ℂ (transition p u j)) (fun q => gradOutput p u q j) t) ∧
      (∀ j, mask p u j=Bool.true → complexOutput A v p.gs (columns p u j)=
        Reference.reverseCarry length (starRingEnd ℂ (transition p u j)) (fun q => gradOutput p u q j) length) ∧
      (∀ j, mask p u j=Bool.true → complexOutput A v p.gl (columns p u j)=
        Reference.transitionGradient length (starRingEnd ℂ (transition p u j)) (starRingEnd ℂ (initial p u j))
          (fun q => starRingEnd ℂ (output p u q j)) (fun q => gradOutput p u q j) length) ∧
      (∀ r o, Outside p A u length r o → v.mem r o=(A.flattenState u).mem r o)
```
</details>

<details><summary><code>original</code></summary>

```lean
def original (p : Layout) (length : Nat) : ComputeKernel :=
  (TypedSource.complexBackward Bool.false p.s p.l p.y p.gs p.gx p.gl p.gy length p.batch p.dim p.B).eraseDType
```
</details>

<details><summary><code>mask</code></summary>

```lean
def mask (p : Layout) (u : BlockState) (j : Lane p) : Bool := decide (columns p u j<p.batch*p.dim)
```
</details>

<details><summary><code>complexOutput</code></summary>

```
/-- Relocate components separately; the allocation base may be odd. -/
```
```lean
def complexOutput (A : FlatAlloc) (v : BlockState) (r : RegionName) (offset : Nat) : ℂ :=
  ⟨v.readMem A.flat (A.addr r (offset*2)),v.readMem A.flat (A.addr r (offset*2+1))⟩
```
</details>

<details><summary><code>offset</code></summary>

```lean
def offset (p : Layout) (u : BlockState) (t : Nat) (j : Lane p) : Nat := t*(p.batch*p.dim)+columns p u j
```
</details>

<details><summary><code>transition</code></summary>

```lean
def transition (p : Layout) (u : BlockState) : Lane p → ℂ :=
  complexInput u p.l (fun j => columns p u j%p.dim) (mask p u)
```
</details>

<details><summary><code>gradOutput</code></summary>

```lean
def gradOutput (p : Layout) (u : BlockState) (t : Nat) : Lane p → ℂ :=
  complexInput u p.gy (offset p u t) (mask p u)

/-- Sufficient alias conditions for nonempty reverse scans. All terminal
stores follow the reads; input regions can alias one another. -/
structure Separation (p : Layout) : Prop where
  initial : p.s≠p.gx
  output : p.y≠p.gx
  gradient : p.gy≠p.gx
  terminal : p.gs≠p.gl
  initialGradient : p.gx≠p.gs
  transitionGradient : p.gx≠p.gl

structure Bounds (p : Layout) (u : BlockState) (length : Nat) (bounds : RegionBounds) : Prop where
  initial : 0<length → ∀ j, mask p u j=Bool.true → columns p u j*2+1 < bounds p.s
  transition : ∀ j, mask p u j=Bool.true → (columns p u j%p.dim)*2+1 < bounds p.l
  previous : ∀ t, t+1<length → ∀ j, mask p u j=Bool.true → offset p u t j*2+1 < bounds p.y
  gradOutput : ∀ t, t<length → ∀ j, mask p u j=Bool.true → offset p u t j*2+1 < bounds p.gy
  gradInput : ∀ t, t<length → ∀ j, mask p u j=Bool.true → offset p u t j*2+1 < bounds p.gx
  gradInitial : ∀ j, mask p u j=Bool.true → columns p u j*2+1 < bounds p.gs
  gradTransition : ∀ j, mask p u j=Bool.true → columns p u j*2+1 < bounds p.gl
```
</details>

<details><summary><code>columns</code></summary>

```lean
def columns (p : Layout) (u : BlockState) (j : Lane p) : Nat := u.pids 0*p.B+j.1.val
```
</details>

<details><summary><code>initial</code></summary>

```lean
def initial (p : Layout) (u : BlockState) : Lane p → ℂ := complexInput u p.s (columns p u) (mask p u)
```
</details>

<details><summary><code>output</code></summary>

```lean
def output (p : Layout) (u : BlockState) (t : Nat) : Lane p → ℂ :=
  complexInput u p.y (offset p u t) (mask p u)
```
</details>

<details><summary><code>Outside</code></summary>

```lean
def Outside (p : Layout) (A : FlatAlloc) (u : BlockState) (length : Nat)
    (r : RegionName) (o : Nat) : Prop :=
  r≠A.flat ∨
    (∀ t, t<length → ∀ j, mask p u j=Bool.true →
      A.addr p.gx (offset p u t j*2)≠o ∧ A.addr p.gx (offset p u t j*2+1)≠o) ∧
    (∀ j, mask p u j=Bool.true → A.addr p.gs (columns p u j*2)≠o ∧ A.addr p.gs (columns p u j*2+1)≠o) ∧
    (∀ j, mask p u j=Bool.true → A.addr p.gl (columns p u j*2)≠o ∧ A.addr p.gl (columns p u j*2+1)≠o)
```
</details>

<details><summary><code>complexBackward</code></summary>

```lean
def complexBackward (swapped : Bool) : complexBackwardSource :=
  if swapped then diagSsmComplexBackwardSource%[tl.float32, indexCommuted]
  else diagSsmComplexBackwardSource%[tl.float32, original]

end VeriTile.Bench.TritonBenchG.DiagSsmTriton.TypedSource

namespace VeriTile.Bench.TritonBenchG.DiagSsmTriton
open VeriTile.Triton

/-- Buffer names and logical sizes shared by the real and complex forward
specifications. B is the source block size; length is supplied separately. -/
structure ForwardLayout where
  s : RegionName
  x : RegionName
  y : RegionName
  l : RegionName
  batch : Nat
  dim : Nat
  B : Nat

/-- Backward buffers contain one initial state/transition per column,
time-dependent y/grad_y/grad_x and terminal grad_s/grad_lambda. -/
structure BackwardLayout where
  s : RegionName
  l : RegionName
  y : RegionName
  gs : RegionName
  gx : RegionName
  gl : RegionName
  gy : RegionName
  batch : Nat
  dim : Nat
  B : Nat
```
</details>

<details><summary><code>complexInput</code></summary>

```
/-- Read the two masked components of an original complex input. -/
```
```lean
def complexInput {B : Nat} (u : BlockState) (r : RegionName) (offset : TileIndex [B] → Nat)
    (mask : TileIndex [B] → Bool) : TileIndex [B] → ℂ := fun j =>
  ⟨if mask j then u.readMem r (offset j*2) else 0,
   if mask j then u.readMem r (offset j*2+1) else 0⟩
```
</details>

<details><summary><code>indexCommuted</code></summary>

```lean
def indexCommuted (p : Layout) (length : Nat) : ComputeKernel :=
  (TypedSource.complexBackward Bool.true p.s p.l p.y p.gs p.gx p.gl p.gy length p.batch p.dim p.B).eraseDType
```
</details>

## Public theorem: `complex_backward_index_commuted_correctness`

**Statement:**
```lean
specification complex_backward_index_commuted_correctness (p : Layout) (length : Nat) :
    Spec.Real (contract (indexCommuted p length) p length)
```

**Closed-form spec defs (transitive):** `Layout`, `contract`, `indexCommuted`, `mask`, `complexOutput`, `offset`, `transition`, `gradOutput`, `columns`, `initial`, `output`, `Outside`, `complexBackward`, `complexInput`, `original`

<details><summary><code>Layout</code></summary>

```lean
abbrev Layout := BackwardLayout
```
</details>

<details><summary><code>contract</code></summary>

```
/-- Actual finite access safety, successful flat execution, all three
independent reverse-scan outputs and the full tagged-cell frame. -/
```
```lean
def contract (kernel : ComputeKernel) (p : Layout) (length : Nat) : Prop :=
  (0<length → Separation p) →
  ∀ A : FlatAlloc, A.Disjoint → (∀ r, r ∉ A.regions → A.extent r=0) →
  ∀ u : BlockState, u.undef=(fun _ _ => 0) → Bounds p u length A.extent →
    kernel.toAlgKernel.TraceSafe A.extent u ∧
    ∃ v, exec (A.flattenKernel kernel.toAlgKernel) (A.flattenState u)=some v ∧
      (∀ t, t<length → ∀ j, mask p u j=Bool.true →
        complexOutput A v p.gx (offset p u t j)=Reference.inputGradient length
          (starRingEnd ℂ (transition p u j)) (fun q => gradOutput p u q j) t) ∧
      (∀ j, mask p u j=Bool.true → complexOutput A v p.gs (columns p u j)=
        Reference.reverseCarry length (starRingEnd ℂ (transition p u j)) (fun q => gradOutput p u q j) length) ∧
      (∀ j, mask p u j=Bool.true → complexOutput A v p.gl (columns p u j)=
        Reference.transitionGradient length (starRingEnd ℂ (transition p u j)) (starRingEnd ℂ (initial p u j))
          (fun q => starRingEnd ℂ (output p u q j)) (fun q => gradOutput p u q j) length) ∧
      (∀ r o, Outside p A u length r o → v.mem r o=(A.flattenState u).mem r o)
```
</details>

<details><summary><code>indexCommuted</code></summary>

```lean
def indexCommuted (p : Layout) (length : Nat) : ComputeKernel :=
  (TypedSource.complexBackward Bool.true p.s p.l p.y p.gs p.gx p.gl p.gy length p.batch p.dim p.B).eraseDType
```
</details>

<details><summary><code>mask</code></summary>

```lean
def mask (p : Layout) (u : BlockState) (j : Lane p) : Bool := decide (columns p u j<p.batch*p.dim)
```
</details>

<details><summary><code>complexOutput</code></summary>

```
/-- Relocate components separately; the allocation base may be odd. -/
```
```lean
def complexOutput (A : FlatAlloc) (v : BlockState) (r : RegionName) (offset : Nat) : ℂ :=
  ⟨v.readMem A.flat (A.addr r (offset*2)),v.readMem A.flat (A.addr r (offset*2+1))⟩
```
</details>

<details><summary><code>offset</code></summary>

```lean
def offset (p : Layout) (u : BlockState) (t : Nat) (j : Lane p) : Nat := t*(p.batch*p.dim)+columns p u j
```
</details>

<details><summary><code>transition</code></summary>

```lean
def transition (p : Layout) (u : BlockState) : Lane p → ℂ :=
  complexInput u p.l (fun j => columns p u j%p.dim) (mask p u)
```
</details>

<details><summary><code>gradOutput</code></summary>

```lean
def gradOutput (p : Layout) (u : BlockState) (t : Nat) : Lane p → ℂ :=
  complexInput u p.gy (offset p u t) (mask p u)

/-- Sufficient alias conditions for nonempty reverse scans. All terminal
stores follow the reads; input regions can alias one another. -/
structure Separation (p : Layout) : Prop where
  initial : p.s≠p.gx
  output : p.y≠p.gx
  gradient : p.gy≠p.gx
  terminal : p.gs≠p.gl
  initialGradient : p.gx≠p.gs
  transitionGradient : p.gx≠p.gl

structure Bounds (p : Layout) (u : BlockState) (length : Nat) (bounds : RegionBounds) : Prop where
  initial : 0<length → ∀ j, mask p u j=Bool.true → columns p u j*2+1 < bounds p.s
  transition : ∀ j, mask p u j=Bool.true → (columns p u j%p.dim)*2+1 < bounds p.l
  previous : ∀ t, t+1<length → ∀ j, mask p u j=Bool.true → offset p u t j*2+1 < bounds p.y
  gradOutput : ∀ t, t<length → ∀ j, mask p u j=Bool.true → offset p u t j*2+1 < bounds p.gy
  gradInput : ∀ t, t<length → ∀ j, mask p u j=Bool.true → offset p u t j*2+1 < bounds p.gx
  gradInitial : ∀ j, mask p u j=Bool.true → columns p u j*2+1 < bounds p.gs
  gradTransition : ∀ j, mask p u j=Bool.true → columns p u j*2+1 < bounds p.gl
```
</details>

<details><summary><code>columns</code></summary>

```lean
def columns (p : Layout) (u : BlockState) (j : Lane p) : Nat := u.pids 0*p.B+j.1.val
```
</details>

<details><summary><code>initial</code></summary>

```lean
def initial (p : Layout) (u : BlockState) : Lane p → ℂ := complexInput u p.s (columns p u) (mask p u)
```
</details>

<details><summary><code>output</code></summary>

```lean
def output (p : Layout) (u : BlockState) (t : Nat) : Lane p → ℂ :=
  complexInput u p.y (offset p u t) (mask p u)
```
</details>

<details><summary><code>Outside</code></summary>

```lean
def Outside (p : Layout) (A : FlatAlloc) (u : BlockState) (length : Nat)
    (r : RegionName) (o : Nat) : Prop :=
  r≠A.flat ∨
    (∀ t, t<length → ∀ j, mask p u j=Bool.true →
      A.addr p.gx (offset p u t j*2)≠o ∧ A.addr p.gx (offset p u t j*2+1)≠o) ∧
    (∀ j, mask p u j=Bool.true → A.addr p.gs (columns p u j*2)≠o ∧ A.addr p.gs (columns p u j*2+1)≠o) ∧
    (∀ j, mask p u j=Bool.true → A.addr p.gl (columns p u j*2)≠o ∧ A.addr p.gl (columns p u j*2+1)≠o)
```
</details>

<details><summary><code>complexBackward</code></summary>

```lean
def complexBackward (swapped : Bool) : complexBackwardSource :=
  if swapped then diagSsmComplexBackwardSource%[tl.float32, indexCommuted]
  else diagSsmComplexBackwardSource%[tl.float32, original]

end VeriTile.Bench.TritonBenchG.DiagSsmTriton.TypedSource

namespace VeriTile.Bench.TritonBenchG.DiagSsmTriton
open VeriTile.Triton

/-- Buffer names and logical sizes shared by the real and complex forward
specifications. B is the source block size; length is supplied separately. -/
structure ForwardLayout where
  s : RegionName
  x : RegionName
  y : RegionName
  l : RegionName
  batch : Nat
  dim : Nat
  B : Nat

/-- Backward buffers contain one initial state/transition per column,
time-dependent y/grad_y/grad_x and terminal grad_s/grad_lambda. -/
structure BackwardLayout where
  s : RegionName
  l : RegionName
  y : RegionName
  gs : RegionName
  gx : RegionName
  gl : RegionName
  gy : RegionName
  batch : Nat
  dim : Nat
  B : Nat
```
</details>

<details><summary><code>complexInput</code></summary>

```
/-- Read the two masked components of an original complex input. -/
```
```lean
def complexInput {B : Nat} (u : BlockState) (r : RegionName) (offset : TileIndex [B] → Nat)
    (mask : TileIndex [B] → Bool) : TileIndex [B] → ℂ := fun j =>
  ⟨if mask j then u.readMem r (offset j*2) else 0,
   if mask j then u.readMem r (offset j*2+1) else 0⟩
```
</details>

<details><summary><code>original</code></summary>

```lean
def original (p : Layout) (length : Nat) : ComputeKernel :=
  (TypedSource.complexBackward Bool.false p.s p.l p.y p.gs p.gx p.gl p.gy length p.batch p.dim p.B).eraseDType
```
</details>
