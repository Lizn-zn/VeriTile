"""Atomic, fixed-size expression relations. Imported only by the GPU runner.

No autotuning, automatic input conditioning, or implicit TF32. Ordinary
arithmetic launches disable FP fusion; the FMA candidate calls tl.fma explicitly.
Different schedules/casts are different instances, not universal rule evidence.
"""
import triton
import triton.language as tl

ELEMENTWISE = (
    "ADD-COMMUTE", "MUL-COMMUTE", "ROUND-IDEM", "BF16-WIDEN-RETURN",
    "ADD-ASSOC", "MUL-ASSOC", "MUL-DISTRIB", "FMA-CONTRACT", "CANCEL",
    "DIV-RCP", "SQRT-RSQRT", "CAST-MOVE", "CAST-REMOVE", "ACC-WIDEN",
)
SUPPORTED = set(ELEMENTWISE)


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
    else:
        tl.static_assert(False, "unknown atomic rule")
    tl.store(O + i, out, i < N)
