"""Numerical regression cases and adaptive sampling checks."""
import json
from pathlib import Path
from unittest.mock import patch
import unittest

import numpy as np

from scripts import numerical_gates as gates
from scripts.test_numerical_gates import observations, profile


class SamplingTests(unittest.TestCase):
    def test_magnitude_regression_cases(self):
        fixture = json.loads((Path(__file__).parent / 'fixtures/numerical_gate_cases.json').read_text())
        n = fixture['replicates']
        for case in fixture['cases']:
            with self.subTest(family=case['family']):
                rng = np.random.default_rng(fixture['seed'])
                k = {
                    'zero': lambda: np.zeros(n),
                    'constant_small': lambda: np.ones(n),
                    'constant_large': lambda: np.full(n, 31.75),
                    'sparse': lambda: np.r_[np.zeros(n - 6), np.arange(6)],
                    'exponential': lambda: rng.exponential(.2, n),
                    'heavy': lambda: (rng.pareto(5, n) / .2) * .2,
                    'bounded': lambda: rng.uniform(0, 1, n),
                    'ties': lambda: rng.integers(0, 4, n).astype(float),
                    'nonfinite': lambda: np.r_[np.zeros(n - 1), np.inf],
                }[case['family']]()
                result = gates.vars_gate(np.ones(n), k, profile()['gates']['vars'], 123456)
                self.assertEqual(result['status'], case['status'])
                self.assertEqual(result['valid'], case['valid'])
                for actual, expected in ((result['upper'], case['U']), (result.get('xi'), case['xi'])):
                    if isinstance(expected, float):
                        self.assertAlmostEqual(actual, expected, places=11)
                    else:
                        self.assertEqual(actual, expected)

    def test_adaptive_minimum_fallback_and_nonfinite(self):
        config = profile()['gates']['vars']
        obs = observations(8)
        _, stop = gates.checkpoint(obs, config, 16, 32, 8)
        self.assertIsNone(stop)
        _, stop = gates.checkpoint(observations(16), config, 16, 32, 8)
        self.assertEqual(stop, 'empirical_fallback')
        obs['candidate_error'][0] = np.inf
        _, stop = gates.checkpoint(obs, config, 16, 32, 8)
        self.assertEqual(stop, 'nonfinite')

    def test_stability_checks_both_thresholds_and_cap(self):
        var = dict(valid=True, upper=3., return_level=2., warn_threshold=2., fail_threshold=10.)
        self.assertIsNone(gates.stopping_reason(var, 16, 16, 32, 8))
        self.assertEqual(gates.stopping_reason(var, 32, 16, 32, 8), 'maximum_replicates')
        var.update(upper=7., return_level=6.)
        self.assertEqual(gates.stopping_reason(var, 16, 16, 32, 8), 'stable')
        var.update(upper=11., return_level=10.)
        self.assertIsNone(gates.stopping_reason(var, 16, 16, 32, 8))

    def test_replay_rejects_early_truncation_and_late_stopping(self):
        config = profile()['gates']['vars']
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            gates.validate_stopping(observations(8), config, 16, 32, 8)
        with self.assertRaisesRegex(ValueError, 'continue'):
            gates.validate_stopping(observations(24), config, 16, 32, 8)
        self.assertEqual(gates.validate_stopping(observations(16), config, 16, 32, 8), 'empirical_fallback')

    def test_non_multiple_cap_finishes_full_batch(self):
        var = dict(valid=True, upper=3., return_level=2., warn_threshold=2., fail_threshold=10.)
        with patch.object(gates, 'vars_gate', return_value=var):
            self.assertEqual(gates.validate_stopping(observations(24), profile()['gates']['vars'],
                                                    16, 20, 8), 'maximum_replicates')


if __name__ == '__main__':
    unittest.main()
