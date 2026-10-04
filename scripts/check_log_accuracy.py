#!/usr/bin/env python3
"""GPU boundary checks for the fp32 expm1/log1p experiment, not admission."""
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
    p = runner.validate_profile(deepcopy(runner.load_module(
        runner.supplemental.DIRECTORY / 'log_accuracy_config.py').PROFILE))
    neighbors = [float(np.nextafter(np.float32(x), np.float32(direction)))
                 for x in (-0.5, 0.5) for direction in (-np.inf, np.inf)]
    values = [-20., -2., -0.5, -2.**-25, -0., 0., 2.**-25, 0.5, 1., 10., 90., *neighbors]
    a = torch.tensor(values, dtype=torch.float32, device='cuda')
    p['shape'] = [1, len(values)]
    inputs = [a, torch.zeros_like(a), torch.zeros_like(a)]
    kernels = runner.load_module(runner.KERNELS)
    outputs, programs = {}, {}
    for rule in ('LOG-EXP-FULL-LIBDEVICE', 'LOG-EXP-EXPM1'):
        (ref, cand), compiled, _ = runner.launch_pair(torch, triton, kernels, rule, inputs, p, p['formats'][0])
        assert torch.equal(cand.view(torch.int32), a.view(torch.int32))
        outputs[rule] = ref
        programs[rule] = compiled
    new = outputs['LOG-EXP-EXPM1']
    old = outputs['LOG-EXP-FULL-LIBDEVICE']
    outside = a.abs() > 0.5
    assert torch.equal(new[outside].view(torch.int32), old[outside].view(torch.int32))
    tiny = a.abs() == 2.**-25
    assert torch.equal(new[tiny], a[tiny]), 'tiny inputs must survive the new path'
    assert torch.isfinite(new[a <= 10]).all()
    # Overflow remains in-domain and must not disappear behind branch masking.
    assert torch.isposinf(new[a == 90]).all()
    assert runner.domains.mask(torch, 'LOG-EXP-EXPM1', inputs).all()
    for text in programs['LOG-EXP-EXPM1']['reference']:
        assert '.f64' not in text, 'reference must retain fp32 computation'
    args.output.mkdir(parents=True, exist_ok=False)
    for rule, compiled in programs.items():
        for side, texts in compiled.items():
            for i, text in enumerate(texts):
                (args.output / f'{rule}.{side}.{i}.ptx').write_text(text)
    rows = [{'a': float(x), 'baseline': runner.gates._number(float(b)),
             'expm1_path': runner.gates._number(float(y)), 'near_zero': abs(float(x)) <= 0.5}
            for x, b, y in zip(a.cpu(), old.cpu(), new.cpu())]
    runner.write_json(args.output / 'boundaries.json', {
        'scope': 'deterministic GPU boundary fixture; not statistical admission',
        'sources': runner.source_hashes(),
        'check_script_sha256': runner.sha(Path(__file__).read_bytes()),
        'threshold': 0.5, 'checks_passed': True, 'rows': rows})
    print(f'PASS: {len(rows)} GPU boundary inputs; fp32 path, fallback, tiny inputs and overflow checked.')


if __name__ == '__main__':
    main()
