"""Local-ULP mean-bias budgets and error amplification with NumPy-only replay.

Adaptive sampling uses a PWM tail fit or an explicitly marked empirical fallback.
The tail model and confidence estimates are not distribution-free guarantees.
"""
import math
from statistics import NormalDist

import numpy as np

VERSION = "local-ulp-replicate-mean"
OBSERVATIONS = {"delta", "reference_error", "candidate_error"}


def _result(status, **details):
    return {"status": status, **details}


def _number(value):
    if math.isnan(value):
        return None
    return float(value) if math.isfinite(value) else ("+inf" if value > 0 else "-inf")


def bias_gate(delta, config):
    """Require every mean +/- configured SE band to lie inside +/- tau.

    z remains diagnostic. These engineering SE bands do not claim calibrated
    simultaneous or optional-stopping confidence coverage.
    """
    delta = np.asarray(delta, dtype=float)
    if delta.ndim != 2 or delta.shape[0] < 2 or delta.shape[1] == 0:
        raise ValueError("invalid or insufficient replicate/bucket observations")
    if not np.isfinite(delta).all():
        return _result("FAIL", reason="nonfinite normalized differences")
    with np.errstate(over="ignore", invalid="ignore"):
        mean, std = delta.mean(axis=0), delta.std(axis=0, ddof=1)
    if not all(np.isfinite(x).all() for x in (mean, std)):
        return _result("FAIL", reason="nonfinite aggregate bias statistics")
    snr = np.divide(np.abs(mean), std, out=np.zeros_like(mean), where=std != 0)
    snr[(std == 0) & (mean != 0)] = np.inf
    z = math.sqrt(len(delta)) * snr
    se = std / math.sqrt(len(delta))
    margin = config["se_multiplier"] * se
    upper = np.abs(mean) + margin
    lower = np.maximum(np.abs(mean) - margin, 0.0)
    if not np.isfinite(upper).all():
        return _result("FAIL", reason="nonfinite mean-bias bounds")
    fail = lower > config["tau"]
    unresolved = (upper > config["tau"]) & ~fail
    status = "FAIL" if fail.any() else "INCONCLUSIVE" if unresolved.any() else "PASS"
    return _result(status, replicates=len(delta), bucket_count=delta.shape[1],
                   mean=mean.tolist(), std=std.tolist(), units="local_ulp",
                   standard_error=se.tolist(), tau=config["tau"], se_multiplier=config["se_multiplier"],
                   interval_lower=(mean - margin).tolist(), interval_upper=(mean + margin).tolist(),
                   upper=float(upper.max()), lower=float(lower.max()),
                   abs_z=[_number(v) for v in z], snr=[_number(v) for v in snr],
                   inconclusive_buckets=np.flatnonzero(unresolved).tolist(),
                   failing_buckets=np.flatnonzero(fail).tolist())


def amplification(reference_error, candidate_error):
    """Error amplification after a fixed one-local-ULP additive allowance."""
    ref, cand = (np.asarray(x, dtype=float) for x in (reference_error, candidate_error))
    if ref.ndim != 1 or ref.shape != cand.shape or not len(ref):
        raise ValueError("errors must be nonempty paired vectors")
    valid = np.isfinite(ref) & np.isfinite(cand) & (ref >= 0) & (cand >= 0)
    k = np.full_like(ref, np.inf)
    k[valid & (cand <= 1)] = 0
    active = valid & (cand > 1) & (ref > 0)
    with np.errstate(over="ignore", divide="ignore"):
        k[active] = (cand[active] - 1) / ref[active]
    return k


def pwm_fit(excess):
    """Hosking-Wallis PWM, including the degenerate constant-excess case."""
    y = np.sort(np.asarray(excess, dtype=float))
    n = y.size
    w0 = y.mean()
    j = np.arange(1, n + 1)
    w1 = float(np.sum((n - j) / (n - 1) * y) / n)
    denominator = w0 - 2.0 * w1
    if denominator == 0.0:
        return 0.0, max(w0, 0.0)
    scale = 2.0 * w0 * w1 / denominator
    xi = 2.0 - w0 / denominator
    return float(xi), float(scale)


def return_level(threshold, xi, scale, horizon, rate):
    """Bounded/exponential-tail model: retain raw xi, clip for evaluation."""
    if scale <= 0:
        return threshold
    xi = min(xi, 0.0)
    if abs(xi) < 1e-6:
        return threshold + scale * math.log(horizon * rate)
    return threshold + (scale / xi) * ((horizon * rate) ** xi - 1.0)


def vars_gate(reference_error, candidate_error, config, seed=0):
    # Bootstrap seed 0 is independent of the input/probe seed.
    k = amplification(reference_error, candidate_error)
    base = {"replicates": len(k), "maximum_k": _number(float(k.max())),
            "positive_count": int((k > 0).sum())}
    if not np.isfinite(k).all():
        return _result("FAIL", branch="nonfinite", upper="+inf", valid=False,
                       empirical_fallback=False, **base)
    positive = np.sort(k[k > 0])
    if len(positive) >= 12:
        n_tail = min(len(positive) - 1, max(config["min_exceedances"],
                     int(round((1.0 - config["quantile"]) * len(positive)))))
        threshold = float(positive[-(n_tail + 1)])
        excess = positive[positive > threshold] - threshold
    else:
        threshold, excess = None, np.array([])
    rate = len(excess) / len(k)
    base.update(threshold=threshold, exceedances=len(excess), exceedance_rate=rate)
    if len(positive) < 12 or len(excess) < 2:
        upper = float(k.max())
        details = dict(branch="empirical_max", upper=upper, return_level=upper,
                       xi=None, valid=False, empirical_fallback=True,
                       reason="tail fit unavailable; empirical maximum is not a confidence bound")
    else:
        xi, scale = pwm_fit(excess)
        level = return_level(threshold, xi, scale, config["horizon"], rate)
        rng = np.random.default_rng(0)
        boots = []
        for _ in range(config["bootstrap"]):
            bx, bs = pwm_fit(rng.choice(excess, size=len(excess), replace=True))
            value = return_level(threshold, bx, bs, config["horizon"], rate)
            if math.isfinite(value):
                boots.append(value)
        se = float(np.std(boots)) if len(boots) > 10 else 0.0
        z = NormalDist().inv_cdf(1.0 - config["alpha"])
        upper = float(level + z * se)
        details = dict(branch="pot_pwm", xi=_number(xi), scale=_number(scale),
                       return_level=_number(level), bootstrap_se=_number(se), upper=_number(upper),
                       valid=True, empirical_fallback=False)
    status = ("FAIL" if not math.isfinite(upper) or upper > config["fail"] else
              "WARN" if upper > config["warn"] else "PASS")
    return _result(status, **details, **base)


def stopping_reason(var, count, minimum, maximum, batch):
    """Check after each full batch; the replicate cap may round upward."""
    if count % batch:
        return None
    upper = var["upper"]
    if not isinstance(upper, (float, int)) or not math.isfinite(upper):
        return "nonfinite"
    if count >= minimum:
        if not var["valid"]:
            return "empirical_fallback"
        low = 2.0 * var["return_level"] - upper
        if not (low <= var["warn_threshold"] <= upper) and not (low <= var["fail_threshold"] <= upper):
            return "stable"
    return "maximum_replicates" if count >= maximum else None


def checkpoint(observations, config, minimum, maximum, batch):
    var = vars_gate(*(observations[k] for k in ("reference_error", "candidate_error")), config)
    var.update(warn_threshold=config["warn"], fail_threshold=config["fail"])
    return var, stopping_reason(var, len(observations["candidate_error"]), minimum, maximum, batch)


def validate_stopping(observations, config, minimum, maximum, batch):
    """Recompute every checkpoint: no truncated or overrun stream can be accepted."""
    count = len(observations["candidate_error"])
    limit = math.ceil(maximum / batch) * batch
    if not count or count % batch or count > limit:
        raise ValueError("incomplete or invalid adaptive replicate budget")
    for end in range(batch, count + 1, batch):
        # Only the magnitude gate determines stopping; avoid copying bucket arrays.
        prefix = {k: observations[k][:end] for k in ("reference_error", "candidate_error")}
        _, reason = checkpoint(prefix, config, minimum, maximum, batch)
        if reason:
            if end != count:
                raise ValueError("observations continue after the first adaptive stopping point")
            return reason
    raise ValueError("incomplete adaptive replicate budget")


def evaluate(observations, config, seed, expected_replicates, smoke=False):
    if set(observations) != OBSERVATIONS:
        raise ValueError("unexpected observation fields")
    if any(len(v) != expected_replicates for v in observations.values()):
        raise ValueError("incomplete replicate budget")
    bias = bias_gate(observations["delta"], config["bias"])
    var = vars_gate(observations["reference_error"], observations["candidate_error"],
                    config["vars"], seed)
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
