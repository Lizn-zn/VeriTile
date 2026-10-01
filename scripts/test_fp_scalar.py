"""Concrete Lean scalar semantics against independent bit/interval references.

Run after `lake build VeriTile.Triton.Float.ScalarOps`:
    python3 -m unittest scripts.test_fp_scalar
The oracle searches ordered representable values, independently of the Lean
implementation's exponent/significand rounding algorithm. No GPU is required.
"""
from fractions import Fraction
from pathlib import Path
import random
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
FRACTION_BITS = {"bf16": 7, "fp32": 23}
MODES = ("rne", "rtz", "rup", "rdn")


def power2(exponent):
    return Fraction(2 ** exponent) if exponent >= 0 else Fraction(1, 2 ** -exponent)


def magnitude(bits, fraction_bits):
    """Positive finite decoding, plus the virtual overflow neighbor 2**128."""
    exponent, fraction = divmod(bits, 1 << fraction_bits)
    if exponent == 255:
        assert fraction == 0
        return power2(128)
    if exponent == 0:
        return Fraction(fraction, 1 << fraction_bits) * power2(-126)
    return (1 + Fraction(fraction, 1 << fraction_bits)) * power2(exponent - 127)


def decode(bits, fmt):
    p = FRACTION_BITS[fmt]
    sign, positive = divmod(bits, 1 << (p + 8))
    assert positive < (255 << p), "finite inputs required"
    return (-1 if sign else 1) * magnitude(positive, p)


def reference_round(value, fmt, mode="rne", negative_zero=False, flush=False):
    """Binary search adjacent encodings; compare exact distances or direction."""
    p = FRACTION_BITS[fmt]
    sign = value < 0 or (value == 0 and negative_zero)
    value = abs(value)
    inf = 255 << p
    low, high = 0, inf
    while low < high:
        middle = (low + high) // 2
        if magnitude(middle, p) < value:
            low = middle + 1
        else:
            high = middle
    upper = low
    if value == 0:
        code = 0
    elif magnitude(upper, p) == value:
        code = upper
    else:
        lower = upper - 1
        if mode == "rne":
            distances = value - magnitude(lower, p), magnitude(upper, p) - value
            code = lower if (distances[0], lower % 2) < (distances[1], upper % 2) else upper
        elif mode == "rtz":
            code = lower
        elif mode == "rup":
            code = lower if sign else upper
        else:
            code = upper if sign else lower
    # The virtual overflow neighbor is infinity only for the matching modes.
    if value >= power2(128):
        code = inf if mode == "rne" or (mode == "rup" and not sign) or (mode == "rdn" and sign) else inf - 1
    if flush and code < (1 << p):
        code = 0
    return (int(sign) << (p + 8)) | code


def command(fmt, op, a, b=0, c=0, mode="rne", flush_in=False, flush_out=False):
    return f"{fmt} {mode} {int(flush_in)} {int(flush_out)} {op} {a} {b} {c}"


class ScalarTests(unittest.TestCase):
    def check_cases(self, cases):
        run = subprocess.run(
            ["lake", "env", "lean", "--run", "scripts/fp_scalar_driver.lean"],
            cwd=ROOT, input="\n".join(line for line, _ in cases) + "\n",
            text=True, capture_output=True, timeout=180,
        )
        self.assertEqual(run.returncode, 0, run.stderr + run.stdout[-1000:])
        actual = run.stdout.splitlines()
        self.assertEqual(len(actual), len(cases), run.stdout[-1000:])
        for result, (line, expected) in zip(actual, cases):
            self.assertEqual(int(result), expected, f"{line}: got {int(result):x}, want {expected:x}")

    def test_all_bf16_encodings_and_widen_return(self):
        cases = []
        for bits in range(1 << 16):
            nan = bits & 0x7F80 == 0x7F80 and bits & 0x7F != 0
            cases.append((command("bf16", "to_fp32", bits), 0x7FC00000 if nan else bits << 16))
            cases.append((command("fp32", "to_bf16", bits << 16), 0x7FC0 if nan else bits))
        self.check_cases(cases)

    def test_rounding_boundaries_all_modes(self):
        cases = []
        for fmt, p in FRACTION_BITS.items():
            tiny = power2(-126 - p)
            maximum = magnitude((255 << p) - 1, p)
            values = [Fraction(0), tiny / 2, tiny * Fraction(3, 2), tiny,
                      power2(-126) - tiny / 2, power2(-126),
                      1 + power2(-p - 1), 1 + 3 * power2(-p - 1),
                      2 - power2(-p - 1), maximum, maximum + power2(126 - p),
                      power2(128), power2(150), power2(-200), Fraction(1, 3)]
            for mode in MODES:
                for flush in (False, True):
                    for positive in values:
                        for sign in (False, True):
                            q = -positive if sign else positive
                            cases.append((command(fmt, "round", q.numerator, q.denominator,
                                                  int(sign), mode, flush_out=flush),
                                          reference_round(q, fmt, mode, sign, flush)))
        self.check_cases(cases)

    def test_special_values_signs_and_flush(self):
        cases = []
        for fmt, p in FRACTION_BITS.items():
            sign, inf, nan = 1 << (p + 8), 255 << p, (255 << p) | (1 << (p - 1))
            one = 127 << p
            minus_one = sign | one
            for mode in MODES:
                neg_cancel = sign if mode == "rdn" else 0
                examples = [
                    ("add", 0, sign, 0, neg_cancel), ("add", sign, sign, 0, sign),
                    ("sub", one, one, 0, neg_cancel), ("add", one, minus_one, 0, neg_cancel),
                    ("mul", sign, one, 0, sign), ("mul", 0, minus_one, 0, sign),
                    ("mul", 0, inf, 0, nan), ("div", 0, 0, 0, nan),
                    ("div", one, sign, 0, sign | inf), ("div", minus_one, inf, 0, sign),
                    ("div", inf, inf, 0, nan), ("add", inf, sign | inf, 0, nan),
                    ("fma", 0, inf, one, nan), ("fma", inf, one, sign | inf, nan),
                    ("fma", one, one, minus_one, neg_cancel), ("fma", sign, one, sign, sign),
                    ("fma", 1, 1, 0, 1 if mode == "rup" else 0),
                    ("eq", 0, sign, 0, 1), ("eq", nan, nan, 0, 0),
                    ("lt", sign | inf, one, 0, 1), ("lt", nan, one, 0, 0),
                    ("lt", sign, 0, 0, 0), ("neg", sign | inf | 1, 0, 0, inf | 1),
                    ("abs", sign | inf | 1, 0, 0, inf | 1),
                ]
                for op, a, b, c, expected in examples:
                    cases.append((command(fmt, op, a, b, c, mode), expected))
                for payload in (1, 3, 1 << (p - 1), (1 << p) - 1):
                    for operation in ("add", "sub", "mul", "div", "fma"):
                        cases.append((command(fmt, operation, sign | inf | payload, one, one, mode), nan))
            cases.extend([
                (command(fmt, "mul", sign | 1, one, flush_in=True), sign),
                (command(fmt, "mul", sign | 1, one, flush_out=True), sign),
                (command(fmt, "add", 1, 1, flush_in=True), 0),
                (command(fmt, "add", 1, 1), 2),
                (command(fmt, "div", one, 1, flush_in=True), inf),
            ])
        self.check_cases(cases)

    def test_random_finite_arithmetic_against_interval_oracle(self):
        rng = random.Random(20261001)
        cases = []
        for fmt, p in FRACTION_BITS.items():
            for mode in MODES:
                for _ in range(180):
                    bits = [rng.randrange(255 << p) | (rng.randrange(2) << (p + 8)) for _ in range(3)]
                    a, b, c = (decode(v, fmt) for v in bits)
                    for op, exact in (("add", a + b), ("sub", a - b), ("mul", a * b),
                                      ("div", a / b if b else None), ("fma", a * b + c)):
                        if exact is None:
                            continue
                        if op in ("mul", "div"):
                            negative_zero = bool((bits[0] ^ bits[1]) & (1 << (p + 8)))
                        else:
                            negative_zero = mode == "rdn"
                            if op == "add" and a == b == 0 and (bits[0] == bits[1]):
                                negative_zero = bool(bits[0])
                        cases.append((command(fmt, op, *bits, mode=mode),
                                      reference_round(exact, fmt, mode, negative_zero)))
        self.check_cases(cases)

    def test_fp32_host_differential(self):
        # Independent native float32 arithmetic, with explicit canonicalization.
        import numpy as np
        rng = np.random.default_rng(410)
        bits = rng.integers(0, 2**32, size=(1500, 2), dtype=np.uint32)
        operands = bits.view(np.float32)
        cases = []
        with np.errstate(all="ignore"):
            for op, fn in (("add", np.add), ("sub", np.subtract),
                           ("mul", np.multiply), ("div", np.divide)):
                results = fn(operands[:, 0], operands[:, 1], dtype=np.float32)
                for pair, value, result in zip(bits, results, results.view(np.uint32)):
                    expected = 0x7FC00000 if np.isnan(value) else int(result)
                    cases.append((command("fp32", op, int(pair[0]), int(pair[1])), expected))
        self.check_cases(cases)

    def test_documented_non_equivalences(self):
        cases = []
        for fmt, p in FRACTION_BITS.items():
            def bits(x):
                return reference_round(Fraction(x), fmt)
            big = 1 << (p + 1)
            # Associativity: (big + 1) + (-big) = 0; big + (1 - big) = 1.
            cases.extend([(command(fmt, "add", bits(big), bits(1)), bits(big)),
                          (command(fmt, "add", bits(1), bits(-big)), bits(1 - big)),
                          (command(fmt, "add", bits(big), bits(1 - big)), bits(1)),
                          (command(fmt, "add", bits(big), bits(-big)), bits(0))])
            # Distribution: 3 * round(big + 1) versus round(3 * big + 3).
            cases.extend([(command(fmt, "mul", bits(3), bits(big)), bits(3 * big)),
                          (command(fmt, "add", bits(3 * big), bits(3)), bits(3 * big + 4))])
            # Separate multiplication loses the low term; FMA keeps it.
            delta = Fraction(1, 128 if fmt == "bf16" else 8192)
            cases.extend([(command(fmt, "mul", bits(1 + delta), bits(1 - delta)), bits(1)),
                          (command(fmt, "add", bits(1), bits(-1)), bits(0)),
                          (command(fmt, "fma", bits(1 + delta), bits(1 - delta), bits(-1)), bits(-delta**2))])
        self.check_cases(cases)


if __name__ == "__main__":
    unittest.main()
