#!/usr/bin/env python3
"""GPU boundary and timing checks for guarded FP32 log-exp elimination."""
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
    import triton.testing

    if not torch.cuda.is_available():
        parser.error('NVIDIA CUDA GPU required')
    p = runner.validate_profile(deepcopy(runner.load_module(
        runner.supplemental.DIRECTORY / 'log_accuracy_config.py').PROFILE))
    neighbors = [float(np.nextafter(np.float32(x), np.float32(direction)))
                 for x in (-80., -0.5, 0.5, 80.) for direction in (-np.inf, np.inf)]
    values = [-120., -100., -90., -80., -20., -2., -0.5, -2.**-25,
              -0., 0., 2.**-25, 0.5, 1., 10., 80., 90., 100., *neighbors]
    a = torch.tensor(values, dtype=torch.float32, device='cuda')
    p['shape'] = [1, len(values)]
    inputs = [a, torch.zeros_like(a), torch.zeros_like(a)]
    kernels = runner.load_module(runner.KERNELS)
    outputs, programs = {}, {}
    rules = [r for pair in runner.supplemental.LOG_PAIRS for r in pair if r.startswith('LOG-EXP')]
    for rule in rules:
        (ref, cand), compiled, _ = runner.launch_pair(torch, triton, kernels, rule, inputs, p, p['formats'][0])
        outputs[rule] = (ref, cand)
        programs[rule] = compiled
    same_bits = lambda x, y: torch.equal(x.view(torch.int32), y.view(torch.int32))
    simplify = (a.abs() > 0.5) & (a.abs() <= 80.)
    for baseline, guarded in [('LOG-EXP-LIBDEVICE', 'LOG-EXP-GUARDED-INTRINSIC'),
                              ('LOG-EXP-FULL-LIBDEVICE', 'LOG-EXP-GUARDED')]:
        old, identity = outputs[baseline]
        reference, candidate = outputs[guarded]
        assert same_bits(identity, a)
        assert same_bits(reference, old), 'each backend must preserve its own reference'
        assert same_bits(candidate[~simplify], reference[~simplify])
        assert same_bits(candidate[simplify], a[simplify])
        tiny = a.abs() == 2.**-25
        assert torch.equal(reference[tiny], torch.zeros_like(a[tiny]))
        assert same_bits(candidate[tiny], reference[tiny])
        assert torch.isposinf(reference[a == 90]).all()
        assert torch.isneginf(reference[a == -120]).all()
        assert runner.domains.mask(torch, guarded, inputs).all()
    args.output.mkdir(parents=True, exist_ok=False)
    for rule, compiled in programs.items():
        for side, texts in compiled.items():
            for i, text in enumerate(texts):
                assert '.f64' not in text
                (args.output / f'{rule}.{side}.{i}.ptx').write_text(text)
    rows = [{'a': float(x), 'simplify': 0.5 < abs(float(x)) <= 80.,
             'outputs': {rule: {'reference': runner.gates._number(float(pair[0][i])),
                                'candidate': runner.gates._number(float(pair[1][i]))}
                         for rule, pair in outputs.items()}}
            for i, x in enumerate(a.cpu())]

    # Time preallocated kernels separately from sampling and gate evaluation.
    # Report both fast and fallback workloads; a conditional identity is not
    # automatically a speedup on mixed tiles.
    n, block = 4096 * 4096, p['launch']['block']
    generator = torch.Generator(device='cuda').manual_seed(20261005)
    cases = {
        'all_simplify_uniform_1_2': torch.rand(n, device='cuda', generator=generator) + 1.,
        'all_fallback_uniform_0_0.25': torch.rand(n, device='cuda', generator=generator) * 0.25,
        'mixed_normal_1_1': torch.randn(n, device='cuda', generator=generator) + 1.,
    }
    timings = []
    for guarded in ['LOG-EXP-GUARDED-INTRINSIC', 'LOG-EXP-GUARDED']:
        for name, data in cases.items():
            ref_out, cand_out = torch.empty_like(data), torch.empty_like(data)
            def launch(side, out):
                return kernels.elementwise[(triton.cdiv(n, block),)](
                    data, data, data, out, n, guarded, side, 'fp32', block,
                    num_warps=p['launch']['num_warps'], enable_fp_fusion=False)
            launch(0, ref_out)
            launch(1, cand_out)
            fast = (data.abs() > 0.5) & (data.abs() <= 80.)
            assert same_bits(cand_out[fast], data[fast])
            assert same_bits(cand_out[~fast], ref_out[~fast])
            ref_ms = triton.testing.do_bench(lambda: launch(0, ref_out), warmup=100, rep=300, quantiles=[0.5])
            cand_ms = triton.testing.do_bench(lambda: launch(1, cand_out), warmup=100, rep=300, quantiles=[0.5])
            timings.append({'rule': guarded, 'workload': name, 'elements': n, 'block': block,
                            'simplify_fraction': float(fast.float().mean()),
                            'reference_ms': ref_ms, 'candidate_ms': cand_ms,
                            'speedup': ref_ms / cand_ms})
    runner.write_json(args.output / 'boundaries.json', {
        'scope': 'deterministic GPU boundary fixture and descriptive timing; not statistical admission',
        'sources': runner.source_hashes(),
        'check_script_sha256': runner.sha(Path(__file__).read_bytes()),
        'lower': 0.5, 'upper': 80.0, 'branch_side': 'candidate',
        'reference_matches_baseline_bitwise': True, 'checks_passed': True,
        'rows': rows, 'timings': timings,
        'timing_method': 'Triton do_bench median; 100 ms warmup, 300 ms repeat; preallocated outputs; H200'})
    print(f'PASS: {len(rows)} GPU boundary inputs; fixed reference, guarded identity, tails and block paths checked.')
    for timing in timings:
        print(timing)


if __name__ == '__main__':
    main()
