#!/usr/bin/env python3
"""GPU boundary and branch diagnostics for log-product alternatives; not admission."""
import argparse
from copy import deepcopy
from pathlib import Path

if __package__:
    from . import check_numerics_supplement as runner
else:
    import check_numerics_supplement as runner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import torch
    import triton

    if not torch.cuda.is_available():
        parser.error('NVIDIA CUDA GPU required')
    profile = runner.validate_profile(deepcopy(runner.load_module(
        runner.supplemental.DIRECTORY / 'log_product_config.py').PROFILE))
    fmt = profile['formats'][0]
    kernels = runner.load_module(runner.KERNELS)
    values = [(1. + 2.**-23, 1. - 2.**-24), (2., 0.5), (0.75, 1.5),
              (0.1, 0.2), (3., 4.), (np.finfo(np.float32).max, 2.),
              (np.finfo(np.float32).tiny, np.finfo(np.float32).tiny)]
    for x in [0.5, 1.5, 2.]:
        values.extend((float(v), 1.) for v in [np.nextafter(np.float32(x), np.float32(0)),
                                              np.float32(x), np.nextafter(np.float32(x), np.float32(np.inf))])
    a, b = [torch.tensor(x, dtype=torch.float32, device='cuda') for x in zip(*values)]
    inputs = [a, b, torch.ones_like(a)]
    p = a * b
    small_profile = deepcopy(profile)
    small_profile['shape'] = [1, len(values)]
    rules = ['LOG-MUL-LIBDEVICE', 'LOG-MUL-LOG1P', 'LOG-MUL-GUARDED']
    outputs = {}
    args.output.mkdir(parents=True, exist_ok=False)
    for rule in rules:
        pair, programs, _ = runner.launch_pair(torch, triton, kernels, rule, inputs, small_profile, fmt)
        outputs[rule] = pair
        for side, texts in programs.items():
            for i, text in enumerate(texts):
                assert '.f64' not in text
                (args.output / f'{rule}.{side}.{i}.ptx').write_text(text)
    baseline, corrected, guarded = [outputs[r] for r in rules]
    same_bits = lambda x, y: torch.equal(x.view(torch.int32), y.view(torch.int32))
    assert same_bits(guarded[0], baseline[0])
    assert same_bits(corrected[1], baseline[1])
    keep = (p >= 0.5) & (p <= 2.)
    assert same_bits(guarded[1][keep], baseline[0][keep])
    assert same_bits(guarded[1][~keep], baseline[1][~keep])
    fallback = (p < 0.5) | (p > 1.5)
    assert same_bits(corrected[0][fallback], baseline[0][fallback])
    golden = runner.oracle(torch, 'LOG-MUL', inputs)
    assert baseline[0][0] == 0
    assert (corrected[0][0].double() - golden[0]).abs() < (baseline[0][0].double() - golden[0]).abs()
    # Underflow/overflow are valid-input failures, never removed from the sample.
    assert runner.domains.mask(torch, 'LOG-MUL-GUARDED', inputs).all()
    assert torch.isposinf(guarded[0][5]) and torch.isneginf(guarded[0][6])
    rows = []
    for i, (av, bv) in enumerate(values):
        rows.append({'a': float(av), 'b': float(bv), 'keep_product': bool(keep[i]),
                     **{rule: {'reference': runner.gates._number(float(pair[0][i])),
                               'candidate': runner.gates._number(float(pair[1][i]))}
                        for rule, pair in outputs.items()}})

    # The first 32 paired draws describe branch coverage and error contributions.
    # They do not replace the separate 4096-replicate two-gates experiments.
    generator = torch.Generator(device='cuda').manual_seed(runner.seed_for(profile, fmt, 'LOG-MUL'))
    groups = {'p<0.5': [], '0.5<=p<=2': [], 'p>2': []}
    counts = dict.fromkeys(groups, 0)
    valid_total = 0
    max_abs_delta = dict.fromkeys(groups, 0.)
    for _ in range(32):
        inputs = runner.supplemental.sample_inputs(torch, profile, fmt, 'LOG-MUL', generator)
        valid = runner.domains.mask(torch, 'LOG-MUL', inputs)
        exact = runner.oracle(torch, 'LOG-MUL', inputs)[valid]
        (ref, cand), _, _ = runner.launch_pair(torch, triton, kernels, 'LOG-MUL', inputs, profile, fmt)
        product = (inputs[0] * inputs[1])[valid]
        delta = (cand[valid].double() - ref[valid].double()) / runner.original.ulp(torch, exact, 'fp32')
        n = int(valid.sum())
        valid_total += n
        masks = [product < 0.5, (product >= 0.5) & (product <= 2.), product > 2.]
        for name, mask in zip(groups, masks):
            counts[name] += int(mask.sum())
            groups[name].append(float(delta[mask].sum()) / n)
            if mask.any():
                max_abs_delta[name] = max(max_abs_delta[name], float(delta[mask].abs().max()))
    evidence = {
        'scope': 'GPU boundary fixtures and 32-draw descriptive diagnostics; not statistical admission',
        'sources': runner.source_hashes(), 'check_script_sha256': runner.sha(Path(__file__).read_bytes()),
        'boundary_checks_passed': True, 'boundaries': rows,
        'branch_diagnostics': {'replicates': 32, 'valid_samples': valid_total, 'groups': {
            name: {'count': counts[name], 'fraction': counts[name] / valid_total,
                   'mean_contribution_to_total_delta': float(np.mean(data)),
                   'std_contribution_to_total_delta': float(np.std(data, ddof=1)),
                   'maximum_absolute_element_delta_ulps': max_abs_delta[name]}
            for name, data in groups.items()}},
    }
    runner.write_json(args.output / 'log_product_diagnostics.json', evidence)
    print(f'PASS: {len(rows)} boundary inputs; branch coverage and contributions recorded on 32 paired draws.')


if __name__ == '__main__':
    main()
