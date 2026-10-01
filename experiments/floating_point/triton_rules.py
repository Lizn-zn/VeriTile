"""Concrete, versioned candidates. Imported only by the GPU runner.

No autotuning, automatic input conditioning, or implicit TF32. Ordinary
arithmetic launches disable FP fusion; the FMA candidate calls tl.fma explicitly.
Different schedules/casts are different instances, not universal rule evidence.
"""
import triton
import triton.language as tl

ELEMENTWISE = (
    "ADD-COMMUTE", "MUL-COMMUTE", "ROUND-IDEM", "BF16-WIDEN-RETURN",
    "ADD-ASSOC", "MUL-ASSOC", "MUL-DISTRIB", "FMA-CONTRACT", "CANCEL",
    "DIV-RCP", "SQRT-RSQRT", "CAST-MOVE", "CAST-REMOVE", "ACC-WIDEN", "SWIGLU-FUSE",
)
ROW_RULES = ("REDUCE-REORDER", "REDUCE-SPLIT", "SCAN-REORDER", "SOFTMAX-SHIFT",
             "SOFTMAX-ONLINE", "LAYERNORM-WELFORD")
DOT_RULES = ("DOT-LOWER", "DOT-ACC-FUSE", "GEMM-SPLIT-K")
LAYOUT_RULES = ("LAYOUT-INVERSE", "STORE-LOAD-FORWARD")
SUPPORTED = set(ELEMENTWISE + ROW_RULES + DOT_RULES + LAYOUT_RULES)


@triton.jit
def rnd(x, LOW: tl.constexpr):
    if LOW:
        return x.to(tl.bfloat16).to(tl.float32)
    return x.to(tl.float32)


@triton.jit
def elementwise(A, B, C, O, N: tl.constexpr, RULE: tl.constexpr,
                SIDE: tl.constexpr, LOW: tl.constexpr, INPUT_LOW: tl.constexpr,
                BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(A + i, i < N, 0).to(tl.float32)
    b = tl.load(B + i, i < N, 0).to(tl.float32)
    c = tl.load(C + i, i < N, 0).to(tl.float32)
    if RULE == "ADD-COMMUTE":
        if SIDE == 0:
            out = rnd(a + b, LOW)
        else:
            out = rnd(b + a, LOW)
    elif RULE == "MUL-COMMUTE":
        if SIDE == 0:
            out = rnd(a * b, LOW)
        else:
            out = rnd(b * a, LOW)
    elif RULE == "ROUND-IDEM":
        if SIDE == 0:
            out = a.to(O.dtype.element_ty).to(tl.float32).to(O.dtype.element_ty)
        else:
            out = a.to(O.dtype.element_ty)
    elif RULE == "BF16-WIDEN-RETURN":
        if SIDE == 0:
            out = a.to(tl.bfloat16).to(tl.float32).to(tl.bfloat16)
        else:
            out = a.to(tl.bfloat16)
    elif RULE == "ADD-ASSOC":
        if SIDE == 0:
            out = rnd(rnd(a + b, LOW) + c, LOW)
        else:
            out = rnd(a + rnd(b + c, LOW), LOW)
    elif RULE == "MUL-ASSOC":
        if SIDE == 0:
            out = rnd(rnd(a * b, LOW) * c, LOW)
        else:
            out = rnd(a * rnd(b * c, LOW), LOW)
    elif RULE == "MUL-DISTRIB":
        if SIDE == 0:
            out = rnd(a * rnd(b + c, LOW), LOW)
        else:
            out = rnd(rnd(a * b, LOW) + rnd(a * c, LOW), LOW)
    elif RULE == "FMA-CONTRACT":
        if SIDE == 0:
            out = rnd(rnd(a * b, LOW) + c, LOW)
        else:
            out = rnd(tl.fma(a, b, c), LOW)
    elif RULE == "CANCEL":
        if SIDE == 0:
            out = rnd(rnd(a - b, LOW) + b, LOW)
        else:
            out = a
    elif RULE == "DIV-RCP":
        if SIDE == 0:
            out = rnd(tl.div_rn(a, b), LOW)
        else:
            out = rnd(a * rnd(tl.div_rn(1.0, b), LOW), LOW)
    elif RULE == "SQRT-RSQRT":
        if SIDE == 0:
            out = rnd(tl.div_rn(1.0, rnd(tl.sqrt(a), LOW)), LOW)
        else:
            out = rnd(tl.rsqrt(a), LOW)
    elif RULE == "CAST-MOVE":
        if SIDE == 0:
            out = rnd(rnd(a, True) + rnd(b, True), True)
        else:
            out = rnd(a + b, True)
    elif RULE == "CAST-REMOVE":
        if SIDE == 0:
            out = rnd(rnd(a + b, True) * c, LOW)
        else:
            out = rnd((a + b) * c, LOW)
    elif RULE == "ACC-WIDEN":
        if SIDE == 0:
            out = rnd(rnd(a + b, INPUT_LOW) + c, INPUT_LOW)
        else:
            out = (a + b) + c
    elif RULE == "SWIGLU-FUSE":
        gate = rnd(tl.div_rn(a, 1.0 + tl.exp(-a)), LOW)
        if SIDE == 0:
            out = gate  # Materialize in output format; second kernel multiplies.
        else:
            out = rnd(gate * b, LOW)
    else:
        tl.static_assert(False, "unknown elementwise rule")
    tl.store(O + i, out, i < N)


@triton.jit
def multiply_stage(G, B, O, N: tl.constexpr, LOW: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    g = tl.load(G + i, i < N, 0).to(tl.float32)
    b = tl.load(B + i, i < N, 0).to(tl.float32)
    tl.store(O + i, rnd(g * b, LOW), i < N)


@triton.jit
def copy_or_transpose(A, O, M: tl.constexpr, N: tl.constexpr,
                      TRANSPOSE: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(A + i, i < M * N, 0)
    if TRANSPOSE:
        target = (i % N) * M + i // N
    else:
        target = i
    tl.store(O + target, x, i < M * N)


@triton.jit
def welford_merge(mean1, m21, n1, mean2, m22, n2):
    n = n1 + n2
    delta = mean2 - mean1
    safe_n = tl.maximum(n, 1)
    mean = mean1 + delta * n2 / safe_n
    m2 = m21 + m22 + delta * delta * n1 * n2 / safe_n
    return mean, m2, n


@triton.jit
def row_kernel(A, O, N: tl.constexpr, RULE: tl.constexpr, SIDE: tl.constexpr,
               EPS: tl.constexpr, CHUNK: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    j = tl.arange(0, BLOCK)
    x = tl.load(A + row * N + j, j < N, 0).to(tl.float32)
    if RULE == "REDUCE-REORDER":
        if SIDE == 0:
            out = tl.sum(x, 0)
        else:
            reverse = tl.load(A + row * N + (N - 1 - j), j < N, 0).to(tl.float32)
            out = tl.sum(reverse, 0)
        tl.store(O + row, out)
    elif RULE == "REDUCE-SPLIT":
        if SIDE == 0:
            out = tl.sum(x, 0)
        else:
            out = tl.full((), 0, tl.float32)
            c = tl.arange(0, CHUNK)
            for start in range(triton.cdiv(N, CHUNK)):
                index = start * CHUNK + c
                values = tl.load(A + row * N + index, index < N, 0).to(tl.float32)
                out += tl.sum(values, 0)
        tl.store(O + row, out)
    elif RULE == "SCAN-REORDER":
        if SIDE == 0:
            # Serial prefix scan, all prefixes written. The candidate uses a tree.
            carry = tl.full((), 0, tl.float32)
            for index in range(N):
                value = tl.load(A + row * N + index).to(tl.float32)
                carry += value
                tl.store(O + row * N + index, carry)
        else:
            out = tl.cumsum(x, 0)
            tl.store(O + row * N + j, out, j < N)
    elif RULE == "SOFTMAX-SHIFT":
        if SIDE == 0:
            ex = tl.where(j < N, tl.exp(x), 0.0)
        else:
            maximum = tl.max(tl.where(j < N, x, -float("inf")), 0)
            ex = tl.where(j < N, tl.exp(x - maximum), 0.0)
        out = ex / tl.sum(ex, 0)
        tl.store(O + row * N + j, out, j < N)
    elif RULE == "SOFTMAX-ONLINE":
        if SIDE == 0:
            maximum = tl.max(tl.where(j < N, x, -float("inf")), 0)
            ex = tl.where(j < N, tl.exp(x - maximum), 0.0)
            denominator = tl.sum(ex, 0)
        else:
            maximum = tl.full((), -float("inf"), tl.float32)
            denominator = tl.full((), 0, tl.float32)
            c = tl.arange(0, CHUNK)
            for start in range(triton.cdiv(N, CHUNK)):
                index = start * CHUNK + c
                values = tl.load(A + row * N + index, index < N, -float("inf")).to(tl.float32)
                next_max = tl.maximum(maximum, tl.max(values, 0))
                denominator = denominator * tl.exp(maximum - next_max) + tl.sum(tl.exp(values - next_max), 0)
                maximum = next_max
            ex = tl.where(j < N, tl.exp(x - maximum), 0.0)
        out = ex / denominator
        tl.store(O + row * N + j, out, j < N)
    elif RULE == "LAYERNORM-WELFORD":
        if SIDE == 0:
            mean = tl.sum(x, 0) / N
            centered = tl.where(j < N, x - mean, 0.0)
            variance = tl.sum(centered * centered, 0) / N
        else:
            count = (j < N).to(tl.float32)
            mean, m2, total = tl.reduce((x, tl.full((BLOCK,), 0, tl.float32), count), 0, welford_merge)
            variance = m2 / total
        out = (x - mean) * tl.rsqrt(variance + EPS)
        tl.store(O + row * N + j, out, j < N)
    else:
        tl.static_assert(False, "unknown row rule")


@triton.jit
def dot_kernel(A, B, C, O, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
               RULE: tl.constexpr, SIDE: tl.constexpr, TILE: tl.constexpr):
    m = tl.program_id(0) * TILE + tl.arange(0, TILE)
    n = tl.program_id(1) * TILE + tl.arange(0, TILE)
    k = tl.arange(0, TILE)
    initial = tl.load(C + m[:, None] * N + n[None, :],
                      (m[:, None] < M) & (n[None, :] < N), 0).to(tl.float32)
    acc = tl.full((TILE, TILE), 0, tl.float32)
    if RULE == "DOT-LOWER" and SIDE == 0:
        for index in range(K):
            av = tl.load(A + m * K + index, m < M, 0).to(tl.float32)
            bv = tl.load(B + index * N + n, n < N, 0).to(tl.float32)
            acc = acc + av[:, None] * bv[None, :]
    elif RULE == "GEMM-SPLIT-K" and SIDE == 1:
        # Two independently accumulated contiguous halves, deterministic merge.
        acc2 = tl.full((TILE, TILE), 0, tl.float32)
        for start in range(triton.cdiv(K // 2, TILE)):
            index = start * TILE + k
            av = tl.load(A + m[:, None] * K + index[None, :],
                         (m[:, None] < M) & (index[None, :] < K // 2), 0)
            bv = tl.load(B + index[:, None] * N + n[None, :],
                         (index[:, None] < K // 2) & (n[None, :] < N), 0)
            acc = tl.dot(av, bv, acc, input_precision="ieee")
        for start in range(triton.cdiv(K - K // 2, TILE)):
            index = K // 2 + start * TILE + k
            av = tl.load(A + m[:, None] * K + index[None, :],
                         (m[:, None] < M) & (index[None, :] < K), 0)
            bv = tl.load(B + index[:, None] * N + n[None, :],
                         (index[:, None] < K) & (n[None, :] < N), 0)
            acc2 = tl.dot(av, bv, acc2, input_precision="ieee")
        acc += acc2
    else:
        if RULE == "DOT-ACC-FUSE" and SIDE == 1:
            acc = initial
        for start in range(triton.cdiv(K, TILE)):
            index = start * TILE + k
            av = tl.load(A + m[:, None] * K + index[None, :],
                         (m[:, None] < M) & (index[None, :] < K), 0)
            bv = tl.load(B + index[:, None] * N + n[None, :],
                         (index[:, None] < K) & (n[None, :] < N), 0)
            acc = tl.dot(av, bv, acc, input_precision="ieee")
    if RULE == "DOT-ACC-FUSE" and SIDE == 0:
        acc += initial
    tl.store(O + m[:, None] * N + n[None, :], acc,
             (m[:, None] < M) & (n[None, :] < N))
