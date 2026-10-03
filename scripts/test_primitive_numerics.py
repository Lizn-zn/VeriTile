"""Bounded integer probes and intrinsic identity; CPU fixtures are not GPU evidence."""
from copy import deepcopy
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None

from scripts import check_numerics_supplement as runner, supplement_numerics as supplement
from scripts.test_numerics_supplement import fixture_bundle

COUNT_PROFILE = runner.ROOT / "experiments/floating_point/primitives/count_config.py"


class CPUFixtureTorch:
    def __getattr__(self, key):
        return getattr(torch, key)

    def Generator(self, device):
        return torch.Generator(device="cpu")

    def randint(self, *args, **kwargs):
        return torch.randint(*args, **{**kwargs, "device": "cpu"})

    def zeros(self, *args, **kwargs):
        return torch.zeros(*args, **{**kwargs, "device": "cpu"})


@unittest.skipIf(torch is None, "optional CPU fixtures require torch")
class CountTests(unittest.TestCase):
    def profile(self):
        return runner.validate_profile(deepcopy(runner.load_module(COUNT_PROFILE).PROFILE))

    def test_count_domain_precision_and_gate_budgets(self):
        p = self.profile()
        old = runner.load_module(runner.DEFAULT_PROFILE).PROFILE
        for key in ("shape", "seed", "replicates", "replicates_max", "batch", "gates", "launch"):
            self.assertEqual(p[key], old[key])
        for change in ("negative", "overflow", "dtype", "mixed", "distribution"):
            bad = deepcopy(p)
            if change == "negative":
                bad["distribution"]["low"] = -1
            elif change == "overflow":
                bad["distribution"]["high"] = 2**31
            elif change == "dtype":
                bad["formats"][0]["input"] = "fp32"
            elif change == "mixed":
                bad["rules"].append("EXP-SUB-INTRINSIC")
            else:
                bad["distribution"] = deepcopy(old["distribution"])
            with self.subTest(change=change), self.assertRaises(ValueError):
                runner.validate_profile(bad)

    def test_sampling_keeps_integer_input_and_bounds_in_contract(self):
        p = self.profile()
        p["shape"] = [2, 3]
        fmt = p["formats"][0]
        cpu = CPUFixtureTorch()
        for rule in p["rules"]:
            inputs = supplement.sample_inputs(cpu, p, fmt, rule, torch.Generator().manual_seed(4))
            self.assertEqual(inputs[0].dtype, torch.int32)
            self.assertTrue(bool(((inputs[0] >= 0) & (inputs[0] < 2**24)).all()))
            cfg = supplement.contract_for(p, fmt, rule, {}, runner.source_hashes(),
                                          {k: ["0" * 64] for k in ("reference", "candidate")})
            self.assertEqual(cfg["numerics"]["input_formats"]["a"], "int32")
            self.assertEqual(cfg["probe"]["family"], "constant" if rule == "COUNT-ZERO" else "uniform_integer")
            if rule == "COUNT-SUCCESSOR":
                self.assertIn("16777216", cfg["relation"]["domain"])
                changed = deepcopy(p)
                changed["distribution"]["high"] = 4096
                other = supplement.contract_for(changed, fmt, rule, {}, runner.source_hashes(),
                                                {k: ["0" * 64] for k in ("reference", "candidate")})
                self.assertNotEqual(supplement.instance_key(cfg), supplement.instance_key(other))

    def test_successor_boundary_is_not_an_unbounded_equality(self):
        a = torch.tensor([0, 4095, 2**24 - 1, 2**24 + 1], dtype=torch.int32)
        ref, cand = (a + 1).float(), a.float() + 1
        self.assertEqual((ref == cand).tolist(), [True, True, True, False])
        exact = supplement.oracle(torch, "COUNT-SUCCESSOR", [a, a, a])
        self.assertEqual(exact[-1].item(), 16777218.)
        self.assertEqual(ref[-1].item(), 16777218.)
        self.assertEqual(cand[-1].item(), 16777216.)
        from scripts.export_supplemental_rules import domain
        for rule in supplement.COUNT_RULES:
            with self.assertRaisesRegex(ValueError, "integer-range-aware"):
                domain(rule)

    def test_integer_runner_and_replay_keep_declared_sampling(self):
        for rule in supplement.COUNT_RULES:
            with self.subTest(rule=rule), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                fixture_bundle(root)
                p = self.profile()
                p.update(shape=[2, 3], replicates=4, replicates_max=4, batch=4, rules=[rule])
                manifest = runner.read_json(root / "manifest.json")
                fmt = p["formats"][0]
                name = f"{fmt['name']}__{rule}"
                manifest.update(profile=p, entries=[name])
                runner.write_json(root / "manifest.json", manifest)

                def pair(_torch, _triton, _kernels, chosen, inputs, _profile, _fmt):
                    a = inputs[0]
                    self.assertEqual(a.dtype, torch.int32)
                    out = ([torch.zeros_like(a).float()] * 2 if chosen == "COUNT-ZERO"
                           else [(a + 1).float(), a.float() + 1])
                    return out, {k: ["synthetic test, not PTX"] for k in ("reference", "candidate")}, None

                with patch.object(runner, "launch_pair", side_effect=pair), contextlib.redirect_stdout(io.StringIO()):
                    decision = runner.run_instance(CPUFixtureTorch(), None, None, p, fmt, rule, root / name,
                                                   manifest["backend"], manifest["sources"], False)
                self.assertEqual(decision, "ACCEPT")
                report = runner.replay(root)
                self.assertEqual(len(report["accepted"]), 1)
                self.assertEqual(report["rows"][0]["statistics"]["vars"]["upper"], 0.)


if __name__ == "__main__":
    unittest.main()
