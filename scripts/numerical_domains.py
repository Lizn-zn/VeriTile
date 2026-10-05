"""Input-only domain masks for scalar probes; never filter by output error."""
import numpy as np


OPERANDS = {
    **dict.fromkeys(("ADD-COMMUTE", "MUL-COMMUTE", "CANCEL", "DIV-RCP", "CAST-MOVE"), "ab"),
    **dict.fromkeys(("ADD-ASSOC", "MUL-ASSOC", "MUL-DISTRIB", "FMA-CONTRACT",
                     "CAST-REMOVE", "ACC-WIDEN"), "abc"),
    **dict.fromkeys(("ROUND-IDEM", "BF16-WIDEN-RETURN", "SQRT-RSQRT", "ADD-ZERO",
                     "MUL-ONE", "DIV-ONE", "MUL-RCP-CANCEL", "LOG-EXP", "LOG-EXP-LIBDEVICE", "LOG-EXP-FULL-LIBDEVICE",
                     "LOG-EXP-GUARDED", "MAX-IDEM", "MAX-NEG-INF", "EXP-NEG-INF-SUB"), "a"),
    **dict.fromkeys(("DIV-MUL-RCP", "EXP-SUB", "EXP-SUB-INTRINSIC", "LOG-MUL", "LOG-MUL-LIBDEVICE",
                     "LOG-MUL-LOG1P", "LOG-MUL-GUARDED", "MAX-COMMUTE"), "ab"),
    "MAX-ASSOC": "abc", "EXP-ZERO": "", "COUNT-ZERO": "", "COUNT-SUCCESSOR": "a",
}
POSITIVE = {"SQRT-RSQRT": "a", "LOG-MUL": "ab", "LOG-MUL-LIBDEVICE": "ab",
            "LOG-MUL-LOG1P": "ab", "LOG-MUL-GUARDED": "ab"}
NONZERO = {"DIV-RCP": "b", "DIV-MUL-RCP": "b", "MUL-RCP-CANCEL": "a"}


def policy(rule):
    return {
        "version": "skip-outside-domain-1",
        "finite": list(OPERANDS[rule]),
        "positive": list(POSITIVE.get(rule, "")),
        "nonzero": list(NONZERO.get(rule, "")),
        "selection": "same input-only mask for both sides and oracle, after input quantization",
        "sampling": ("constant zero" if rule == "COUNT-ZERO" else
                     "uniform int32 draws in the frozen profile interval" if rule == "COUNT-SUCCESSOR" else
                     "configured normal draws; skip invalid tuples without replacement or resampling"),
        "aggregation": "one mean and two error maxima over valid tuples per nonempty replicate",
        "empty": "skip empty replicates; insufficient nonempty replicates within draw budget is inconclusive",
        "outputs": "nonfinite results on valid inputs remain gate failures",
    }


def mask(torch, rule, inputs):
    values = dict(zip("abc", inputs))
    valid = torch.ones_like(inputs[0], dtype=torch.bool)
    for name in OPERANDS[rule]:
        valid &= torch.isfinite(values[name])
    for name in POSITIVE.get(rule, ""):
        valid &= values[name] > 0
    for name in NONZERO.get(rule, ""):
        valid &= values[name] != 0
    return valid


def summary(counts, shape):
    drawn = len(counts) * int(np.prod(shape))
    valid = sum(int(n) for n in counts)
    return {"attempted_replicates": len(counts), "empty_replicates": sum(n == 0 for n in counts),
            "drawn_samples": drawn, "valid_samples": valid, "skipped_samples": drawn - valid}


def unpack(arrays, record, profile, completed):
    """Check saved sample accounting before replaying the unchanged gates."""
    counts = arrays.pop("valid_samples", None)
    limit = ((profile["replicates_max"] + profile["batch"] - 1) // profile["batch"]) * profile["batch"]
    if (counts is None or counts.dtype != np.int64 or counts.ndim != 1
            or not completed <= len(counts) <= limit
            or np.any(counts < 0) or np.any(counts > int(np.prod(profile["shape"])))
            or np.count_nonzero(counts) != completed or counts[-1] == 0):
        raise ValueError("invalid domain sample counts")
    expected = summary(counts.tolist(), profile["shape"])
    if record.get("sampling") != expected:
        raise ValueError("domain sample totals disagree with observations")
    return expected
