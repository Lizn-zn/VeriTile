"""Fixed-budget, replayable two-gates checker. No GPU or Lean dependency.

POT uses Hosking-Wallis probability weighted moments and a conditional tail
bootstrap. Positive fitted shape parameters are NOT clipped. Confidence bounds
are model-based estimates, not distribution-free or sequential guarantees.
"""
import math

import numpy as np

VERSION = "two-gates-fixed-pwm-v1"
OBSERVATIONS = {"delta", "ulp", "reference_error", "candidate_error", "epsilon"}


def _result(status, **details):
    return {"status": status, **details}


def _number(value):
    # Strict JSON, including the explicitly defined zero-variance branches.
    return float(value) if math.isfinite(value) else ("+inf" if value > 0 else "-inf")


def bias_gate(delta, ulp, config):
    delta, ulp = np.asarray(delta, dtype=float), np.asarray(ulp, dtype=float)
    if (delta.ndim != 2 or delta.shape[0] < 2 or delta.shape[1] == 0
            or ulp.shape != delta.shape or not np.isfinite(delta).all()
            or not np.isfinite(ulp).all() or (ulp <= 0).any()):
        return _result("INCONCLUSIVE", reason="invalid or insufficient replicate/bucket observations")
    with np.errstate(over="ignore", invalid="ignore"):
        mean, std = delta.mean(axis=0), delta.std(axis=0, ddof=1)
        scale = ulp.mean(axis=0)
    if not all(np.isfinite(x).all() for x in (mean, std, scale)):
        return _result("INCONCLUSIVE", reason="nonfinite aggregate bias statistics")
    snr = np.divide(np.abs(mean), std, out=np.zeros_like(mean), where=std != 0)
    snr[(std == 0) & (mean != 0)] = np.inf
    z = math.sqrt(len(delta)) * snr
    significant = z > config["z"]
    fail = significant & (snr > config["snr"]) & (np.abs(mean) > config["ulp_floor"] * scale)
    status = "FAIL" if fail.any() else "WARN" if significant.any() else "PASS"
    return _result(status, replicates=len(delta), bucket_count=delta.shape[1],
                   mean=mean.tolist(), std=std.tolist(), ulp=scale.tolist(),
                   abs_z=[_number(v) for v in z], snr=[_number(v) for v in snr],
                   significant_buckets=np.flatnonzero(significant).tolist(),
                   failing_buckets=np.flatnonzero(fail).tolist())


def amplification(reference_error, candidate_error, epsilon):
    ref, cand, eps = (np.asarray(x, dtype=float) for x in (reference_error, candidate_error, epsilon))
    if (ref.ndim != 1 or ref.shape != cand.shape or ref.shape != eps.shape or len(ref) < 2
            or any(not np.isfinite(x).all() for x in (ref, cand, eps))
            or (ref < 0).any() or (cand < 0).any() or (eps <= 0).any()):
        raise ValueError("errors must be finite nonnegative paired vectors; epsilon must be positive")
    k = np.zeros_like(ref)
    above = cand > eps
    k[above & (ref == 0)] = np.inf
    active = above & (ref > 0)
    with np.errstate(over="ignore", divide="ignore"):
        k[active] = (cand[active] - eps[active]) / ref[active]
    return k


def pwm_fit(excess):
    y = np.sort(np.asarray(excess, dtype=float))
    if len(y) < 3 or not np.isfinite(y).all() or (y <= 0).any():
        raise ValueError("insufficient or invalid positive excesses")
    b0 = y.mean()
    b1 = np.mean(y * np.arange(len(y) - 1, -1, -1) / (len(y) - 1))
    denominator = b0 - 2 * b1
    if denominator <= 0:
        raise ValueError("degenerate PWM fit")
    xi = 2 - b0 / denominator
    scale = b0 * (1 - xi)
    if not math.isfinite(xi) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("invalid GPD parameters")
    if xi < 0 and y[-1] > -scale / xi:
        raise ValueError("fitted GPD endpoint excludes an observed excess")
    return float(xi), float(scale)


def return_level(threshold, xi, scale, horizon, rate):
    if horizon * rate <= 1:
        raise ValueError("requested return level is outside the fitted tail")
    log_t = math.log(horizon * rate)
    value = threshold + scale * (log_t if abs(xi) < 1e-10 else math.expm1(xi * log_t) / xi)
    if not math.isfinite(value):
        raise ValueError("nonfinite return level")
    return value


def vars_gate(reference_error, candidate_error, epsilon, config, seed):
    try:
        k = amplification(reference_error, candidate_error, epsilon)
    except ValueError as error:
        return _result("INCONCLUSIVE", reason=str(error))
    if np.isinf(k).any():
        return _result("FAIL", branch="infinite_amplification", count=int(np.isinf(k).sum()))
    base = {"replicates": len(k), "maximum_k": float(k.max()), "positive_count": int((k > 0).sum())}
    # Observed zero amplification is a protocol branch, NOT a fitted population bound.
    if not (k > 0).any():
        return _result("PASS", branch="all_zero_observed", **base)
    positive = k[k > 0]
    threshold = float(np.quantile(positive, config["quantile"], method="linear"))
    excess = k[k > threshold] - threshold
    base.update(threshold=threshold, exceedances=len(excess), exceedance_rate=len(excess) / len(k))
    if len(excess) < config["min_exceedances"]:
        return _result("INCONCLUSIVE", reason="insufficient strict tail exceedances", **base)
    try:
        xi, scale = pwm_fit(excess)
        level = return_level(threshold, xi, scale, config["horizon"], len(excess) / len(k))
        rng = np.random.default_rng(seed)
        levels = []
        for _ in range(config["bootstrap"]):
            bx, bs = pwm_fit(rng.choice(excess, len(excess), replace=True))
            levels.append(return_level(threshold, bx, bs, config["horizon"], len(excess) / len(k)))
        se = float(np.std(levels, ddof=1))
        upper = level + config["confidence_z"] * se
        if not math.isfinite(upper):
            raise ValueError("nonfinite bootstrap upper estimate")
    except (ValueError, OverflowError) as error:
        # Never discard failed bootstrap samples and report success on the remainder.
        return _result("INCONCLUSIVE", reason=str(error), **base)
    status = "PASS" if upper <= config["warn"] else "WARN" if upper <= config["fail"] else "FAIL"
    return _result(status, branch="pot_pwm", xi=xi, scale=scale, return_level=level,
                   bootstrap_se=se, upper=upper, **base)


def evaluate(observations, config, seed, expected_replicates, smoke=False):
    if set(observations) != OBSERVATIONS:
        raise ValueError("unexpected observation fields")
    if any(len(v) != expected_replicates for v in observations.values()):
        raise ValueError("incomplete replicate budget")
    bias = bias_gate(observations["delta"], observations["ulp"], config["bias"])
    var = vars_gate(observations["reference_error"], observations["candidate_error"],
                    observations["epsilon"], config["vars"], seed)
    statuses = {bias["status"], var["status"]}
    if smoke:
        decision = "SMOKE_ONLY"
    elif "FAIL" in statuses:
        decision = "REJECT"
    elif "INCONCLUSIVE" in statuses:
        decision = "INCONCLUSIVE"
    elif "WARN" in statuses:
        decision = "ACCEPT_WITH_WARNING" if config["warning_policy"] == "allow_warn" else "WARN_NOT_ACCEPTED"
    else:
        decision = "ACCEPT"
    return {"checker_version": VERSION, "bias": bias, "vars": var, "decision": decision}
