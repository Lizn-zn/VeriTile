"""Domain filtering regressions; CPU fixtures never supply GPU evidence."""
import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import check_numerics as main, check_numerics_supplement as supplement
from scripts.test_numerical_gates import fixture_bundle as main_bundle
from scripts.test_numerics_supplement import fixture_bundle as supplement_bundle


class AccountingTests(unittest.TestCase):
    def test_masks_cover_every_catalogue_and_supplement_operand(self):
        domains = main.domains
        self.assertEqual(set(domains.OPERANDS), set(main.registry.load_catalog()) | set(supplement.supplemental.load_catalog()))
        for rule, relation in supplement.supplemental.load_catalog().items():
            self.assertEqual(list(domains.OPERANDS[rule]), relation["operands"])

    def test_replay_rejects_missing_invalid_or_inconsistent_counts(self):
        for runner, fixture in ((main, main_bundle), (supplement, supplement_bundle)):
            for change in ("missing", "negative", "too_many", "empty", "dtype", "totals"):
                with self.subTest(runner=runner.__name__, change=change), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    entry = fixture(root)
                    record = runner.read_json(entry / "record.json")
                    path = entry / "observations.npz"
                    with np.load(path, allow_pickle=False) as data:
                        arrays = {key: data[key] for key in data.files}
                    if change == "missing":
                        del arrays["valid_samples"]
                    elif change == "dtype":
                        arrays["valid_samples"] = arrays["valid_samples"].astype(float)
                    elif change == "totals":
                        record["sampling"]["skipped_samples"] += 1
                    else:
                        arrays["valid_samples"][0] = {"negative": -1, "too_many": 7, "empty": 0}[change]
                    np.savez_compressed(path, **arrays)
                    record["observations_sha256"] = runner.sha(path.read_bytes())
                    runner.write_json(entry / "record.json", record)
                    with self.assertRaisesRegex(ValueError, "domain sample"):
                        runner.replay(root)


@unittest.skipUnless(importlib.util.find_spec("torch"), "CPU tensor checks require torch")
class DomainTests(unittest.TestCase):
    def test_log_filters_operands_not_product_or_output(self):
        import torch
        a = torch.tensor([-1., -1., 0., 2., 4., float("inf")])
        b = torch.tensor([-1., 2., 2., 2., 2., 1.])
        valid = main.domains.mask(torch, "LOG-MUL", [a, b, torch.full_like(a, float("nan"))])
        self.assertEqual(valid.tolist(), [False, False, False, True, True, False])
        # Two negative operands have a positive product but remain outside the domain.
        # Unused c must not reject an otherwise valid tuple.
        before = a.clone()
        exact = torch.ones_like(a).double()
        ref = torch.ones_like(a)
        cand = torch.tensor([float("nan"), float("inf"), 10., 1. + 2.**-22, 1. + 2.**-21, 99.])
        obs = main.observe(torch, ref, cand, exact, "fp32", valid=valid)
        self.assertEqual(obs["delta"].tolist(), [3.])
        self.assertEqual(obs["candidate_error"], 2.**-21)
        torch.testing.assert_close(a, before)
        cand[3] = float("inf")
        obs = main.observe(torch, ref, cand, exact, "fp32", valid=valid)
        self.assertEqual(main.gates.amplification([obs["reference_error"]], [obs["candidate_error"]])[0], float("inf"))

    def test_mask_is_after_quantization_and_matches_residual_errors(self):
        import torch
        a = torch.tensor([1., 1.])
        b = torch.tensor([2.**-150, 2.], dtype=torch.float64).float()
        valid = main.domains.mask(torch, "DIV-RCP", [a, b, a])
        self.assertEqual(valid.tolist(), [False, True])
        errors = [torch.tensor([float("nan"), 0.25]), torch.tensor([float("inf"), 0.5])]
        obs = main.observe(torch, a, a, a.double(), "fp32", errors, valid)
        self.assertEqual((obs["reference_error"], obs["candidate_error"]), (0.25, 0.5))

    def test_runners_skip_tuples_and_empty_replicates_then_replay(self):
        import torch

        class CPUFixtureTorch:
            def __init__(self, all_empty):
                self.calls = 0
                self.all_empty = all_empty

            def __getattr__(self, key):
                return getattr(torch, key)

            def Generator(self, device):
                return torch.Generator(device="cpu")

            def randn(self, shape, **kwargs):
                replicate, operand = divmod(self.calls, 3)
                self.calls += 1
                # After the configured +mean, these become the exact input tuples.
                values = ([1., 1., 1., 1., 1., 1.], [0., 2., 0., 0., 2., 0.], [1.] * 6)[operand]
                if self.all_empty or replicate == 0:
                    values = [0.] * 6
                return torch.tensor(values, dtype=kwargs["dtype"]).reshape(shape) - 1.

        for runner, fixture, rule in ((main, main_bundle, "DIV-RCP"),
                                       (supplement, supplement_bundle, "LOG-MUL")):
            for all_empty in (False, True):
                with self.subTest(runner=runner.__name__, all_empty=all_empty), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    old_entry = fixture(root)
                    manifest = runner.read_json(root / "manifest.json")
                    p = manifest["profile"]
                    p.update(rules=[rule], replicates_max=8)
                    name = f"{p['formats'][0]['name']}__{rule}"
                    manifest["entries"] = [name]
                    runner.write_json(root / "manifest.json", manifest)
                    cpu = CPUFixtureTorch(all_empty)

                    def pair(_torch, _triton, _kernels, chosen, inputs, _profile, _fmt):
                        self.assertEqual(tuple(inputs[0].shape), (2, 3))
                        # Invalid values stay in the full-shaped execution inputs.
                        self.assertEqual(inputs[1].flatten().tolist(), [0., 2., 0., 0., 2., 0.])
                        exact = runner.oracle(torch, chosen, inputs).float()
                        result = ([exact, exact.clone()], {side: ["synthetic PTX"] for side in ("reference", "candidate")})
                        return (*result, None) if runner is supplement else result

                    with patch.object(runner, "launch_pair", side_effect=pair) as launched, contextlib.redirect_stdout(io.StringIO()):
                        decision = runner.run_instance(cpu, None, None, p, p["formats"][0], rule,
                                                       root / name, manifest["backend"], manifest["sources"], False)
                    record = runner.read_json(root / name / "record.json")
                    replay = runner.replay(root)
                    if all_empty:
                        self.assertEqual(decision, "INCONCLUSIVE")
                        self.assertEqual(record["sampling"]["skipped_samples"], 48)
                        self.assertEqual(cpu.calls, 24)
                        self.assertEqual(replay["accepted"], [])
                        launched.assert_not_called()
                    else:
                        self.assertEqual(decision, "ACCEPT")
                        self.assertEqual(record["sampling"], dict(attempted_replicates=5, empty_replicates=1,
                                                                  drawn_samples=30, valid_samples=8, skipped_samples=22))
                        self.assertEqual(cpu.calls, 15)  # no replacement draws
                        self.assertEqual(replay["rows"][0]["replicates"], 4)
                        self.assertEqual(replay["rows"][0]["sampling"], record["sampling"])
                        if runner is supplement:
                            runner.publish_report(root, replay, root / "report")
                            row = runner.read_json(root / "report/summary.json")["rows"][0]
                            self.assertEqual((row["valid_samples"], row["skipped_samples"]), (8, 22))


if __name__ == "__main__":
    unittest.main()
