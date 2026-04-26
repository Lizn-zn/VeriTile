# VeriTile: Verified Translation Validation of Triton Kernels via LLM-Assisted Lean Proofs

**Project Proposal**
Zenan Li (zenan.li@apodex.ai)
2026-04-25

---

## TL;DR

We propose **VeriTile**, a system for **machine-checked equivalence proofs between Triton GPU kernels**, where users supply ordinary `.py` Triton code and never interact with Lean directly. A Lean 4 deep embedding of a Triton subset captures kernel operational semantics; an LLM agent produces optimized variants and equivalence proofs (or proves user-supplied pairs equivalent); a lifter and extractor handle the round trip between `.py` Triton and Lean-embedded Triton. The system targets the **algorithm-to-implementation gap** in production ML kernel development—where engineers hand-rewrite kernels (FlashAttention, online softmax, blocked GEMM) but have no formal way to verify their rewrites preserve semantics.

---

## 1. Problem Statement

Modern ML system performance increasingly comes from **hand-rewritten kernels**: online softmax (1-pass vs 3-pass), FlashAttention (no materialized $N \times N$ attention matrix), Welford variance, blocked Cholesky, Kahan summation. These rewrites are equivalent in real arithmetic but typically introduce operations not present in the naive form—e.g., FlashAttention's running-max correction term. Engineers rewrite, eyeball, run end-to-end tests, ship. Bugs slip through routinely; PyTorch / JAX backward passes have long histories of off-by-one errors and numerical pathologies.

The current state of verification:

- **No mainstream tool verifies that two Triton kernels are equivalent.** Engineers cannot machine-check their hand-optimizations.
- **Compiler-internal verification** (TVM Z3 integration, Alive2 for LLVM) operates on **rule-based simplifications inside one compiler pass**, not on **whole-kernel rewrites**.
- **Algorithm-equivalence verification at math abstraction** (ATL, POPL'20) requires lifting kernels into a separate algebraic representation, which (a) imposes a translation gap and (b) misses implementation-level rewrites that don't fit the abstraction (layout, swizzle, register-tile reordering).
- **LLMs can propose kernel rewrites** but offer no correctness guarantee, and recent benchmarks (Vero, 2026) show LLMs solve only a small fraction of even compiler-internal verification tasks (1/27 LLVM transfer functions).

We address this gap with a system that:

1. Verifies kernel-level equivalence **directly on the Triton code** (not on a separate abstraction).
2. **Hides Lean from the end user**: `.py` Triton goes in, verified `.py` Triton comes out.
3. Uses LLMs to produce both **optimized variants** and **equivalence proofs**, with Lean kernel as the trust anchor.

---

## 2. Approach

Three architectural commitments:

**A. Deep-embedded Triton in Lean 4.** A subset of Triton (single-block, deterministic, the ops needed for algorithm-layer rewrites) is formalized as a Lean inductive type `TritonKernel` with operational semantics over a block state (memory regions, register file, program counter). Kernel equivalence is established directly in this semantic model—the verification target is exactly the user's Triton code.

**B. Zero-Lean user experience via lifter / extractor.** Users author Triton in `.py` files as today. A lifter parses the AST and produces a `TritonKernel` term wrapped in a `triton { ... }` Lean macro. LLM agents work entirely inside Lean. An extractor produces `.py` Triton from Lean terms. Users never open a `.lean` file.

**C. Two operating modes:**

- **Optimization mode.** User supplies a naive Triton kernel; the system (LLM-driven) produces an optimized variant **and** a Lean-checked equivalence proof; the user receives the optimized `.py` kernel.
- **Validation mode.** User supplies two Triton kernels (their hand-written naive and optimized versions); the system produces a Lean-checked equivalence proof; the user receives ✓ or a counterexample.

### Architecture

```
   User-facing .py code (write / read)
   ┌─────────────────────────────────────────────────┐
   │  naive_kernel.py    optimized_kernel.py         │
   │       │                     ▲                   │
   └───────┼─────────────────────┼───────────────────┘
           │ Lifter              │ Extractor
           │ (Python AST → Lean) │ (Lean → Python)
           ▼                     │
   ┌─────────────────────────────┼───────────────────┐
   │  Lean 4 layer (trusted)     │                   │
   │  ┌────────────────────────┐ │                   │
   │  │ triton { ... } macro   │ │                   │
   │  │ TritonKernel : Type    │ │                   │
   │  │ exec : ... → State     │ │                   │
   │  │                        │ │                   │
   │  │ LLM produces:          │ │                   │
   │  │   - optimized kernel   │─┘                   │
   │  │   - equivalence proof  │                     │
   │  │                        │                     │
   │  │ Lean kernel: machine-  │                     │
   │  │  checks every artifact │                     │
   │  └────────────────────────┘                     │
   └──────────────────────────────────────────────────┘
                              │
                              │ Differential test bridge
                              ▼
                      PyTorch reference
```

**Trust boundary.**

- *Trusted (machine-checked)*: the Lean operational semantics of `TritonKernel`, the equivalence proofs produced by LLM and checked by Lean kernel.
- *Untrusted (differential-tested)*: the lifter, the extractor, the Triton compiler, the GPU. Any divergence is caught by running both kernels against PyTorch reference and comparing outputs.

This is a deliberate scope: VeriTile guarantees **the two Triton ASTs are equivalent under our formal Triton semantics**. Whether Triton itself compiles them faithfully is Triton's problem—but our equivalence claim is invariant under any uniform compilation.

---

## 3. Worked Example: Online Softmax

### 3.1 The Transformation

Naive softmax requires three passes over input $x \in \mathbb{R}^n$:

$$
m  = \max_i x_i \qquad
s  = \sum_i \exp(x_i - m) \qquad
y_i = \exp(x_i - m) / s
$$

Online softmax (Milakov & Gimelshein, 2018) folds the max and the denominator together in a single pass:

$$
\begin{aligned}
\text{from } (m_{\text{old}}, d_{\text{old}}) \text{ and new } x_i: \quad
& m_{\text{new}} = \max(m_{\text{old}}, x_i) \\
& d_{\text{new}} = d_{\text{old}} \cdot \exp(m_{\text{old}} - m_{\text{new}}) + \exp(x_i - m_{\text{new}})
\end{aligned}
$$

then $y_i = \exp(x_i - m_{\text{final}}) / d_{\text{final}}$. The correction factor $d_{\text{old}} \cdot \exp(m_{\text{old}} - m_{\text{new}})$ is invented—not present in the naive form. This is the kind of rewrite VeriTile aims to verify directly on the Triton code.

### 3.2 User-Facing Files

The user writes ordinary Triton:

```python
# naive_softmax.py
import triton
import triton.language as tl

@triton.jit
def naive_softmax(X, Y, N: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * N + tl.arange(0, N)
    x = tl.load(X + offs)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    y = e / s
    tl.store(Y + offs, y)
```

```python
# online_softmax.py (user-supplied or VeriTile-generated)
@triton.jit
def online_softmax(X, Y, N: tl.constexpr):
    pid = tl.program_id(0)
    base = pid * N
    m = tl.full((), -float('inf'), dtype=tl.float32)
    d = tl.full((), 0.0, dtype=tl.float32)
    for i in range(N):
        x_i = tl.load(X + base + i)
        m_new = tl.maximum(m, x_i)
        d = d * tl.exp(m - m_new) + tl.exp(x_i - m_new)
        m = m_new
    offs = base + tl.arange(0, N)
    x = tl.load(X + offs)
    y = tl.exp(x - m) / d
    tl.store(Y + offs, y)
```

The user invokes:

```bash
$ veritile verify naive_softmax.py online_softmax.py
[lift]    naive_softmax.py    -> Lean term (triton { ... })
[lift]    online_softmax.py   -> Lean term (triton { ... })
[prove]   constructing equivalence proof (LLM agent, ~14 min)
[check]   Lean kernel: PROOF ACCEPTED
[verify]  ✓ kernels are equivalent under VeriTile-Triton operational semantics
[diff]    differential test on 10000 random inputs: max ULP diff = 3 ✓
```

### 3.3 Internal Lean Representation

Inside Lean (user does not see this), the kernels are lifted to:

```lean
def naive_softmax_kernel : TritonKernel := triton {
  @T.jit
  def main(X: Ptr float32, Y: Ptr float32, N: tl.constexpr):
      pid = tl.program_id(0)
      offs = pid * N + tl.arange(0, N)
      x = tl.load(X + offs)
      m = tl.max(x, axis=0)
      e = tl.exp(x - m)
      s = tl.sum(e, axis=0)
      y = e / s
      tl.store(Y + offs, y)
}

def online_softmax_kernel : TritonKernel := triton {
  @T.jit
  def main(X: Ptr float32, Y: Ptr float32, N: tl.constexpr):
      pid = tl.program_id(0)
      base = pid * N
      m = tl.full((), Float.neg_inf, dtype=tl.float32)
      d = tl.full((), 0.0, dtype=tl.float32)
      for i in range(N):
          x_i = tl.load(X + base + i)
          m_new = tl.maximum(m, x_i)
          d = d * tl.exp(m - m_new) + tl.exp(x_i - m_new)
          m = m_new
      offs = base + tl.arange(0, N)
      x = tl.load(X + offs)
      y = tl.exp(x - m) / d
      tl.store(Y + offs, y)
}
```

### 3.4 The Equivalence Theorem

```lean
-- Equivalence statement: for all program inputs (memory state X, kernel parameter N,
-- program_id), the two kernels produce the same output memory Y up to algebraic
-- equivalence (treating fp32 as ℝ).
theorem online_eq_naive
    (X : MemoryRegion ℝ)
    (h_X_finite : ∀ i ∈ X.range, ¬(X.read i).isNaN)
    (h_align : X.aligned 16)
    (N : ℕ) (h_N : N > 0)
    (pid : ℕ) :
    let s_naive  := exec naive_softmax_kernel
                          (initialState X N pid)
    let s_online := exec online_softmax_kernel
                          (initialState X N pid)
    ∀ i, i ∈ outputRange pid N →
         readMem s_naive.Y i = readMem s_online.Y i := by
  intro X h_X_finite h_align N h_N pid
  -- Unfold both kernels' execution traces
  simp only [exec, naive_softmax_kernel, online_softmax_kernel,
             TritonKernel.step, ...]
  -- Key lemma: scan invariant under operational semantics
  apply scan_invariant_corresponds_to_max_and_sum_of_exp
  · exact h_X_finite
  · exact h_align
  · -- the actual algebraic step using Mathlib's Real.exp lemmas
    sorry  -- LLM-target slot
```

The `sorry` is filled by an LLM agent. The proof reduces to (a) showing the two kernels' execution traces agree on the relevant memory locations after their respective loops, and (b) the algebraic identity $d_{\text{old}} \cdot \exp(m_{\text{old}} - m_{\text{new}}) \cdot \exp(x_i - m_{\text{old}}) = d_{\text{old}} \cdot \exp(x_i - m_{\text{new}})$ that justifies the rescaling. Mathlib's `Real.exp_add`, `Real.exp_sub`, and `Finset.sum_range_succ` are the workhorses; the operational-semantics part requires lemmas about `tl.load`/`tl.store` aliasing and `tl.max`/`tl.sum` block-level reductions, which are part of VeriTile's prelude.

---

## 4. Triton Subset and Operational Semantics

We formalize a deliberately small subset of Triton in Phase 1 (P1):

**Included (P1):**
- `tl.load`, `tl.store` (with explicit alignment)
- `tl.arange`, `tl.broadcast`, scalar/tensor constants
- `tl.exp`, `tl.log`, `tl.sqrt`, `tl.maximum`, basic arithmetic
- `tl.max`, `tl.sum`, `tl.min` (block-level reductions, axis=0)
- `tl.program_id`, `tl.constexpr`
- `for`, `if`, scalar / tensor variables

**Excluded (P1, may be added later):**
- `tl.atomic_*` (P3+ if needed)
- `tl.dot`, `tl.tensor` matmul primitive (P5+ for blocked GEMM)
- async copy, software pipelining
- multi-block coordination, cluster-level ops
- specialized Hopper / Blackwell ops (TMA, WGMMA)

**Operational semantics design:**

```lean
-- Block-level state
structure BlockState where
  memory   : MemoryRegion → BitVec 32 → ℝ          -- abstract fp32 → ℝ
  registers: RegMap                                 -- named scalar / tensor regs
  pid      : ℕ                                      -- program_id
  pc       : Nat                                    -- program counter

-- Single step
inductive step : TritonStmt → BlockState → BlockState → Prop where
  | load   : ...
  | store  : ...
  | reduce : ...
  ...

-- Multi-step execution
def exec (k : TritonKernel) (s₀ : BlockState) : BlockState := ...
```

Memory is modeled as a partial map abstracting fp32 storage as ℝ values (with finite-domain assumptions: no NaN, no Inf in input, derivable for all intermediate values from input invariants).

The **fundamental simplification we accept**: we model fp32 arithmetic as ℝ. This is the same trust assumption as ATL, Halide-equivalence, and most algorithm-level verification work. Floating-point soundness is left to differential testing. This is a conscious and well-justified scoping decision.

---

## 5. LLM Integration

LLM agents operate at four levels:

| Task | Input | Output | Difficulty |
|---|---|---|---|
| **Lifter assistance** | external `.py` Triton (irregular) | embedded `triton { ... }` term | low–med (parser fallback) |
| **Optimization mode** | naive `TritonKernel` | optimized `TritonKernel` + equivalence proof | high |
| **Validation mode** | two `TritonKernel` terms | equivalence proof | medium–high |
| **Counterexample explanation** | a failed proof attempt | human-readable explanation of where rewrite is unsafe | low |

We adopt Vero's harness pattern: a coding agent (Claude Code or equivalent) runs in an iterative loop with a Lean 4 MCP server, with proof skills (`apply`, `simp`, `omega`, `induction`, kernel-specific tactics) exposed as tools. Time budget per task: matched to Vero's 1 hour wall-clock, $15 cost cap.

**Key engineering bet.** Operational-semantics proofs are harder than algebraic-abstraction proofs (Vero's 1/27 baseline operates in a simpler model). We mitigate by (a) building a strong tactic library specific to VeriTile-Triton, (b) front-loading the prelude with kernel-shape-recognition lemmas (e.g., "if a kernel's body is a fold + extract pattern, reduce equivalence to fold-step equivalence"), and (c) providing the LLM with worked-example proofs as in-context demonstrations.

---

## 6. Benchmark Design — VeriTile-Bench

We construct VeriTile-Bench, modeled on Vero's task structure:

| Category | Tasks | Difficulty |
|---|---|---|
| Reduction reordering (associativity, commutativity) | 5 | low |
| Online algorithms (softmax, layernorm, Welford) | 6 | medium |
| Numerical stabilization (log-sum-exp, max-subtraction) | 4 | medium |
| Scan / fold equivalences (sequential ↔ tree, prefix sum) | 4 | medium |
| Loop fusion / fission (verifying merged passes) | 4 | medium |
| Blocked / tiled equivalences (blocked GEMM, blocked reduction) | 4 | hard |
| Full FlashAttention-style decomposition | 1 | very hard (stretch) |

**Total: ~28 tasks.** Each task is a pair of Triton kernels (`naive.py`, `optimized.py`) plus a read-only Lean spec file declaring the equivalence theorem statement. Verification: hash check on read-only spec + Lean kernel + axiom whitelist.

**Pass rate metrics:**

1. **Optimization mode end-to-end**: agent receives `naive.py`, produces `optimized.py` + proof; Lean accepts.
2. **Validation mode**: agent receives both `naive.py` and `optimized.py`; produces proof; Lean accepts.
3. **Proof-only**: human writes both kernels, LLM produces proof.

Reporting all three separates LLM's transformation capability from its proof-writing capability.

---

## 7. Timeline

| Phase | Months | Milestone |
|---|---|---|
| **P1: Triton subset semantics** | 1–3 | Lean inductive `TritonKernel`, operational semantics, ~30 reduction lemmas (load/store aliasing, reduce identities, exp/sum interactions). Hand-prove `(0+x)·1 = x`-class trivial equivalences. |
| **P2: `triton { ... }` macro** | 3–4 | Lean meta-program parses Triton-like syntax into `TritonKernel`. Round-trip: macro-parse → pretty-print → re-parse is identity on test corpus. |
| **P3: Lifter (.py → Lean)** | 4–5 | Python AST → Lean source. Handles 80% of canonical Triton patterns; remaining 20% reported with explicit "unsupported feature" error. |
| **P4: First real proof** | 5–6 | Hand-prove `online_eq_naive` for online softmax in operational semantics. **Gate: if proof exceeds ~1500 lines or stalls, scale back subset or pivot.** |
| **P5: Extractor + end-to-end demo** | 6–7 | Lean `TritonKernel` → `.py`. Full pipeline runs: lift → prove → extract → execute on GPU + diff-test. |
| **P6: LLM proof drafting** | 7–9 | LLM produces equivalence proofs given hand-written kernel pairs. **Gate: if pass rate < 5% on first 10 benchmark tasks, reframe as "verified workflow + benchmark" rather than autonomous system.** |
| **P7: LLM optimization mode** | 9–10 | LLM produces optimized kernel + proof from naive input. ≥1 fully autonomous task end-to-end. |
| **P8: Full benchmark + paper** | 10–12 | Run VeriTile-Bench across Claude / GPT-5 / Gemini; arXiv + venue submission (PLDI, OOPSLA, ASPLOS, NeurIPS Datasets&Benchmarks, MLSys). |

Every phase has explicit gate criteria. Failures reroute the contribution narrative without killing the project.

---

## 8. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Triton subset operational semantics is too large to formalize | medium | high | Restrict P1 subset aggressively (single-block, no atomics); expand only as needed |
| LLM cannot draft proofs in operational semantics | medium–high | high | Build VeriTile-specific tactic library (P1); provide rich in-context examples; reframe as "verifier with human proof assistance" if pure automation fails |
| Lifter cannot handle real-world Triton idioms | medium | medium | Cover canonical patterns (80%); reject unsupported with clear error message; widen coverage iteratively |
| Triton language evolves, breaking subset | low–medium | medium | Pin Triton version; subset is small enough that updates are tractable |
| Mathlib lacks key lemmas for fp ↔ ℝ modeling | medium | medium | Build small VeriTile prelude (P1); contribute back to Mathlib |
| Operational-semantics proof complexity makes benchmark unrunnable | medium | high | Start with trivial equivalences; complexity ramp drives subset / tactic improvements |
| Triton miscompilations confound differential tests | low | medium | Cross-validate with PyTorch eager reference; report any miscompiles upstream |
| Translation validation work overlaps with future Triton-team formal work | low | medium | First-mover advantage; engage Triton team early as collaborators rather than competitors |
| Benchmark dismissed as "too narrow" | medium | medium | Emphasize: first kernel-level translation validation benchmark for ML; show evolution from Alive2 / Vero |

---

## 9. Expected Contributions

1. **VeriTile semantics**: First Lean 4 formalization of a Triton subset with operational semantics, designed for kernel-level equivalence reasoning.
2. **VeriTile system**: Lifter + macro + extractor + LLM proof harness, presented as a CLI tool that takes `.py` Triton in and emits verified `.py` Triton out.
3. **VeriTile-Bench**: First benchmark for kernel-level translation validation in ML systems—~28 tasks paired across naive/optimized Triton.
4. **Empirical study**: First measurement of LLM capability at kernel-level operational-semantics proofs; comparison to algebraic-abstraction proofs (Vero LLVM).
5. **Verified kernel catalog**: Open-source library of formally verified Triton kernels (online softmax, Welford, etc.) directly usable as production drop-ins.

---

## 10. Positioning vs. Existing Work

| System | What it verifies | Verifier | LLM-in-loop | User-facing language |
|---|---|---|---|---|
| Alive2 | LLVM peephole rules sound | SMT | no | LLVM IR (compiler-internal) |
| TVM Z3 (PR #1367) | TVM arith rules sound | SMT | no | TVM TIR (compiler-internal) |
| Vero (LLVM) | LLVM transfer functions sound + optimal | Lean | yes | Lean (compiler-internal) |
| ATL (POPL'20) | Tensor expressions equivalent | Coq | no | ATL DSL (separate from C) |
| CompCert | Compiler preserves C semantics | Coq | no | C (production) |
| **VeriTile** | **Two Triton kernels equivalent** | **Lean** | **yes** | **Triton (production)** |

VeriTile is the unique combination: **kernel-level + LLM-driven + Lean-verified + zero-Lean user experience + operational-semantics scope**. The closest analog is "Alive2 for Triton kernels"—which does not exist. CompCert provides the closest formal-methods analog but is single-author, decade-scale, and operates on C compilation rather than ML kernel rewriting.

---

## 11. Concrete First Step (Week 1)

```lean
-- File: VeriTile/Triton/Core.lean
import Mathlib

namespace VeriTile.Triton

-- Minimal Triton ops to start
inductive Op : Type where
  | const     : ℝ → Op
  | load      : (ptr : String) → (offset : Op) → Op
  | store     : (ptr : String) → (offset : Op) → (value : Op) → Op
  | arange    : (n : ℕ) → Op
  | broadcast : Op → (n : ℕ) → Op
  | add | sub | mul | div : Op → Op → Op
  | exp       : Op → Op
  | reduceSum : Op → Op
  | reduceMax : Op → Op
  -- 6-8 more

-- Block state and operational step (sketch)
structure BlockState where
  mem  : String → ℕ → ℝ
  pid  : ℕ
  -- ...

def step : Op → BlockState → BlockState := sorry

-- Trivial first goal
example : ∀ s, step (.add (.const 1) (.const 2)) s = step (.const 3) s := by
  intro s
  rfl
```

Goal of week 1: file compiles, the trivial equivalence above goes through. Total: ~150 lines, ~1–2 days. Once this passes, every subsequent step is structural elaboration.

---

## Appendix A: Sample Failure Mode Analysis (Anticipatory)

Based on Vero's published failure analysis, we anticipate three primary failure modes for LLM agents on VeriTile-Bench, **with operational-semantics target making each more severe than Vero's bitvector setting**:

1. **Imprecise imitation** (estimated ~50%): LLM produces a kernel that resembles a known optimized form but has subtle semantic differences (different reduction order producing different fp values; subtly different handling of edge cases). Mitigation: differential test as a strong filter; rejection corpus fed back as examples.
2. **Incomplete proof** (~30%): LLM writes a correct kernel but the proof contains `sorry` or invokes unjustified lemmas. Mitigation: strict axiom whitelist enforced by checker; explicit `sorry` detection.
3. **Bypass attempts** (~20%): LLM modifies the equivalence theorem statement to make it trivially provable. Mitigation: cryptographic hash of the read-only theorem section, following Vero's pattern.

We expect VeriTile's failure profile to skew toward (2) more heavily than Vero's, because operational-semantics proofs are longer and the temptation to leave `sorry` higher.

---

## References

- Dao, T. et al. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.* NeurIPS 2022.
- Milakov, M. & Gimelshein, N. *Online normalizer calculation for softmax.* arXiv:1805.02867, 2018.
- Bernstein, G. L. et al. *ATL: A Tensor Language for Verifying Tensor Programs.* POPL 2020.
- Lopes, N. P. et al. *Alive2: Bounded Translation Validation for LLVM.* PLDI 2021.
- Tillet, P. et al. *Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations.* MAPL 2019.
- TileLang authors. *TileLang: A Composable Tiled Programming Model for AI Systems.* arXiv:2504.17577.
- Vero authors. *From Specification to Kernel Commit: Verified Code Generation on Real-World Systems.* (preprint, 2026).
- Leroy, X. *A Formally Verified Compiler Back-End.* JAR, 2009 (CompCert).
- Pnueli, A. et al. *Translation Validation.* TACAS 1998.

---

*End of proposal.*
