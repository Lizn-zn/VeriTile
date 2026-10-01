"""CPU-only wiring/oracle regressions; never create gate records.

Set TRITON_INTERPRET=1 to exercise the kernels through Triton's interpreter.
These checks cover masks, indexing and formulas, not GPU instruction rounding.
"""
from copy import deepcopy
import importlib.util
import os
from types import SimpleNamespace
import unittest

from scripts import fp_experiment as experiment

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_TRITON = importlib.util.find_spec("triton") is not None


@unittest.skipUnless(HAS_TORCH, "optional CPU oracle checks require torch")
class OracleTests(unittest.TestCase):
    def test_ulp_including_binade_boundary_and_subnormals(self):
        import torch
        values = torch.tensor([0, 1, 2, 0.5, 2.0**-140], dtype=torch.float64)
        actual = experiment.ulp(torch, values, "fp32")
        expected = torch.tensor([2.0**-149, 2.0**-23, 2.0**-22, 2.0**-24, 2.0**-149], dtype=torch.float64)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual = experiment.ulp(torch, values, "bf16")
        expected = torch.tensor([2.0**-133, 2.0**-7, 2.0**-6, 2.0**-8, 2.0**-133], dtype=torch.float64)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_domain_is_not_silently_conditioned(self):
        import torch
        inputs = [torch.tensor([-1.0, 1]), torch.tensor([0.0, 1]), torch.ones(2)]
        for rule in ("SQRT-RSQRT", "DIV-RCP"):
            with self.assertRaises(experiment.NumericEvent) as error:
                experiment.oracle(torch, rule, inputs, 1e-5)
            self.assertEqual(error.exception.status, "INCONCLUSIVE")

    def test_oracle_and_candidate_nonfinite_events_are_distinct(self):
        import torch
        x = torch.ones((2, 3))
        nan = x * float("nan")
        for reference, candidate, oracle, status in ((x, nan, x, "REJECT"),
                                                    (nan, x, x, "INCONCLUSIVE"),
                                                    (x, x, nan, "INCONCLUSIVE")):
            with self.assertRaises(experiment.NumericEvent) as error:
                experiment.observe(torch, reference, candidate, oracle, "fp32")
            self.assertEqual(error.exception.status, status)


@unittest.skipUnless(HAS_TORCH and HAS_TRITON and os.environ.get("TRITON_INTERPRET") == "1",
                     "optional wiring checks require torch, triton and TRITON_INTERPRET=1")
class TemplateWiringTests(unittest.TestCase):
    def test_all_templates_masked_rectangular_fp32(self):
        import torch
        import triton
        module = experiment.load_module(experiment.KERNELS)

        class InterpretedLaunch:
            def __init__(self, kernel):
                self.kernel = kernel

            def __getitem__(self, grid):
                def invoke(*args, **kwargs):
                    self.kernel[grid](*args, **kwargs)
                    # Production expects compiler output. This test stub is never
                    # serialized and never passed to the gate runner or importer.
                    return SimpleNamespace(asm={"ptx": "interpreter-wiring-test-only"})
                return invoke

        kernels = SimpleNamespace(**{key: getattr(module, key) for key in
                                     ("ELEMENTWISE", "ROW_RULES", "DOT_RULES", "LAYOUT_RULES")})
        for name in ("elementwise", "multiply_stage", "copy_or_transpose", "row_kernel", "dot_kernel"):
            setattr(kernels, name, InterpretedLaunch(getattr(module, name)))
        profile = experiment.validate_profile(deepcopy(experiment.load_module(experiment.DEFAULT_PROFILE).PROFILE))
        profile["shape"] = [3, 17]  # Masked row, dot tiles, transpose; non-square.
        profile["launch"]["chunk"] = 32
        fmt = profile["formats"][-1]
        generator = torch.Generator().manual_seed(823)
        for rule in profile["rules"]:
            if rule == "BF16-WIDEN-RETURN":
                continue  # Production marks this fp32 instance unsupported.
            with self.subTest(rule=rule):
                inputs = [torch.rand(experiment.shapes_for(profile, rule)[k], generator=generator) + 0.5
                          for k in ("a", "b", "c")]
                exact = experiment.oracle(torch, rule, inputs, profile["layernorm_epsilon"])
                outputs, _ = experiment.launch_pair(torch, triton, kernels, rule, inputs, profile, fmt)
                for output in outputs:
                    # Cast candidates explicitly quantize to bf16 even with fp32
                    # storage; all other cases exercise fp32 indexing/formulas.
                    tolerance = 0.02 if rule in ("CAST-MOVE", "CAST-REMOVE") else 2e-5
                    torch.testing.assert_close(output.double(), exact, rtol=tolerance, atol=tolerance)


if __name__ == "__main__":
    unittest.main()
