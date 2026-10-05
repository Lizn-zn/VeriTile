"""Fixed-size scalar relations; no reduction, scan or algorithm-level atom.

Q rounds every arithmetic/transcendental node to compute precision. All three
inputs load in input precision and widen before arithmetic. Both results store
in output precision. Ordinary / deliberately differs from the old div_rn pair.
"""
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

SUPPORTED = {
    "ADD-ZERO", "MUL-ONE", "DIV-ONE", "DIV-MUL-RCP", "MUL-RCP-CANCEL",
    "EXP-SUB", "EXP-ZERO", "LOG-MUL", "LOG-EXP", "MAX-COMMUTE",
    "MAX-ASSOC", "MAX-IDEM", "MAX-NEG-INF", "EXP-NEG-INF-SUB",
    "EXP-SUB-INTRINSIC", "COUNT-ZERO", "COUNT-SUCCESSOR", "LOG-EXP-LIBDEVICE",
    "LOG-MUL-LIBDEVICE", "LOG-EXP-FULL-LIBDEVICE",
    "LOG-EXP-EXPM1",
    "LOG-MUL-LOG1P", "LOG-MUL-GUARDED",
}


@triton.jit
def rnd(x, PRECISION: tl.constexpr):
    if PRECISION == "bf16":
        return x.to(tl.bfloat16).to(tl.float32)
    elif PRECISION == "fp64":
        return x.to(tl.float64)
    else:
        return x.to(tl.float32)


@triton.jit
def elementwise(A, B, C, O, N: tl.constexpr, RULE: tl.constexpr,
                SIDE: tl.constexpr, PRECISION: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    if PRECISION == "fp64":
        a = tl.load(A + offs, mask, other=1).to(tl.float64)
        b = tl.load(B + offs, mask, other=1).to(tl.float64)
        c = tl.load(C + offs, mask, other=1).to(tl.float64)
    else:
        a = tl.load(A + offs, mask, other=1).to(tl.float32)
        b = tl.load(B + offs, mask, other=1).to(tl.float32)
        c = tl.load(C + offs, mask, other=1).to(tl.float32)
    zero = tl.full((BLOCK,), 0, a.dtype)
    one = tl.full((BLOCK,), 1, a.dtype)
    neginf = tl.full((BLOCK,), float("-inf"), a.dtype)
    if RULE == "COUNT-ZERO":
        if SIDE == 0:
            out = tl.full((BLOCK,), 0, tl.int32).to(tl.float32)
        else:
            out = zero
    elif RULE == "COUNT-SUCCESSOR":
        i = tl.load(A + offs, mask, other=0)
        if SIDE == 0:
            out = (i + 1).to(tl.float32)
        else:
            out = i.to(tl.float32) + 1.0
    elif RULE == "ADD-ZERO":
        if SIDE == 0:
            out = rnd(a + zero, PRECISION)
        else:
            out = a
    elif RULE == "MUL-ONE":
        if SIDE == 0:
            out = rnd(a * one, PRECISION)
        else:
            out = a
    elif RULE == "DIV-ONE":
        if SIDE == 0:
            out = rnd(a / one, PRECISION)
        else:
            out = a
    elif RULE == "DIV-MUL-RCP":
        if SIDE == 0:
            out = rnd(a / b, PRECISION)
        else:
            out = rnd(a * rnd(one / b, PRECISION), PRECISION)
    elif RULE == "MUL-RCP-CANCEL":
        if SIDE == 0:
            out = rnd(a * rnd(one / a, PRECISION), PRECISION)
        else:
            out = one
    elif RULE == "EXP-SUB":
        if SIDE == 0:
            out = rnd(libdevice.exp(rnd(a - b, PRECISION)), PRECISION)
        else:
            out = rnd(rnd(libdevice.exp(a), PRECISION) / rnd(libdevice.exp(b), PRECISION), PRECISION)
    elif RULE == "EXP-SUB-INTRINSIC":
        if SIDE == 0:
            out = tl.exp(a - b)
        else:
            out = tl.exp(a) / tl.exp(b)
    elif RULE == "EXP-ZERO":
        if SIDE == 0:
            out = rnd(tl.exp(zero), PRECISION)
        else:
            out = one
    elif RULE == "LOG-MUL":
        if SIDE == 0:
            out = rnd(tl.log(rnd(a * b, PRECISION)), PRECISION)
        else:
            out = rnd(rnd(tl.log(a), PRECISION) + rnd(tl.log(b), PRECISION), PRECISION)
    elif RULE == "LOG-MUL-LIBDEVICE":
        if SIDE == 0:
            out = rnd(libdevice.log(rnd(a * b, PRECISION)), PRECISION)
        else:
            out = rnd(rnd(libdevice.log(a), PRECISION) + rnd(libdevice.log(b), PRECISION), PRECISION)
    elif RULE == "LOG-MUL-LOG1P":
        tl.static_assert(PRECISION == "fp32", "log-product probe requires fp32")
        if SIDE == 0:
            p = a * b
            near_one = (p >= 0.5) & (p <= 1.5)
            small_a = tl.where(near_one, a, 1.0)
            small_b = tl.where(near_one, b, 1.0)
            small = libdevice.log1p(tl.fma(small_a, small_b, -1.0))
            other = libdevice.log(tl.where(near_one, 1.0, p))
            out = tl.where(near_one, small, other)
        else:
            out = libdevice.log(a) + libdevice.log(b)
    elif RULE == "LOG-MUL-GUARDED":
        tl.static_assert(PRECISION == "fp32", "guarded log-product probe requires fp32")
        p = a * b
        if SIDE == 0:
            out = libdevice.log(p)
        else:
            keep_product = (p >= 0.5) & (p <= 2.0)
            direct = libdevice.log(tl.where(keep_product, p, 1.0))
            split_a = tl.where(keep_product, 1.0, a)
            split_b = tl.where(keep_product, 1.0, b)
            split = libdevice.log(split_a) + libdevice.log(split_b)
            out = tl.where(keep_product, direct, split)
    elif RULE == "LOG-EXP":
        if SIDE == 0:
            out = rnd(tl.log(rnd(tl.exp(a), PRECISION)), PRECISION)
        else:
            out = a
    elif RULE == "LOG-EXP-LIBDEVICE":
        # Keep a distinct identity: prior LOG-EXP results tested tl.exp.
        # PR #12 rejected the fp32 tl.exp exp-sub rewrite; the examples use
        # libdevice.exp, whose log-inverse relation needs its own experiment.
        if SIDE == 0:
            out = rnd(tl.log(rnd(libdevice.exp(a), PRECISION)), PRECISION)
        else:
            out = a
    elif RULE == "LOG-EXP-FULL-LIBDEVICE":
        if SIDE == 0:
            out = rnd(libdevice.log(rnd(libdevice.exp(a), PRECISION)), PRECISION)
        else:
            out = a
    elif RULE == "LOG-EXP-EXPM1":
        tl.static_assert(PRECISION == "fp32", "expm1/log1p probe requires fp32")
        if SIDE == 0:
            near_zero = tl.abs(a) <= 0.5
            # tl.where evaluates both arms. Safe unused arguments keep expm1
            # from saturating to -1 on large negative fallback inputs.
            small_a = tl.where(near_zero, a, 0.0)
            other_a = tl.where(near_zero, 0.0, a)
            small = libdevice.log1p(libdevice.expm1(small_a))
            other = libdevice.log(libdevice.exp(other_a))
            out = tl.where(near_zero, small, other)
        else:
            out = a
    elif RULE == "MAX-COMMUTE":
        if SIDE == 0:
            out = tl.maximum(a, b)
        else:
            out = tl.maximum(b, a)
    elif RULE == "MAX-ASSOC":
        if SIDE == 0:
            out = tl.maximum(tl.maximum(a, b), c)
        else:
            out = tl.maximum(a, tl.maximum(b, c))
    elif RULE == "MAX-IDEM":
        if SIDE == 0:
            out = tl.maximum(a, a)
        else:
            out = a
    elif RULE == "MAX-NEG-INF":
        if SIDE == 0:
            out = tl.maximum(neginf, a)
        else:
            out = a
    elif RULE == "EXP-NEG-INF-SUB":
        if SIDE == 0:
            out = rnd(tl.exp(rnd(neginf - a, PRECISION)), PRECISION)
        else:
            out = zero
    else:
        tl.static_assert(False, "unknown supplemental scalar relation")
    tl.store(O + offs, out, mask)


@triton.jit
def quotient_errors(A, B, REF, CAND, ER, EC, N: tl.constexpr, BLOCK: tl.constexpr):
    """Residual oracle for fp64 operands and fp32 outputs.

Explicit fp64 FMA rounds a - output*b ONCE, retaining small residuals that
separate multiplication/subtraction would lose. The final error division is
rounded in fp64. This kernel is an oracle, never a candidate atomic relation.
"""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    a = tl.load(A + offs, mask, other=1).to(tl.float64)
    b = tl.load(B + offs, mask, other=1).to(tl.float64)
    ref = tl.load(REF + offs, mask, other=1).to(tl.float64)
    cand = tl.load(CAND + offs, mask, other=1).to(tl.float64)
    ref_error = tl.abs(tl.fma(-ref, b, a) / b)
    cand_error = tl.abs(tl.fma(-cand, b, a) / b)
    tl.store(ER + offs, ref_error, mask)
    tl.store(EC + offs, cand_error, mask)
