"""CPU-only synthetic output-gradient diagnostics; no model/data dependencies."""

import math


def pooled(p, kind, alpha=1.0):
    n = len(p)
    if kind == "endpoint":
        return p[-1], [0.0] * (n - 1) + [1.0]
    if kind == "linear_mil":
        total = sum(p)
        s = sum(x * x for x in p) / total
        return s, [(2 * x - s) / total for x in p]
    maximum = max(alpha * x for x in p)
    w = [math.exp(alpha * x - maximum) for x in p]
    w = [x / sum(w) for x in w]
    s = sum(x * a for x, a in zip(p, w))
    return s, [a * (1 + alpha * (x - s)) for x, a in zip(p, w)]


def diagnose(p, y, kind, alpha=1.0):
    s, ds = pooled(p, kind, alpha)
    event = [(s - y) / (s * (1 - s)) * d * x * (1 - x)
             for x, d in zip(p, ds)]
    n = len(p)
    target = [y * (0.9 + 1.2 * (i / n) - 1.5 * (i / n) ** 2) / 1.14
              for i in range(1, n + 1)]
    token = [(x - t) / n for x, t in zip(p, target)]
    norm = lambda v: math.sqrt(sum(x * x for x in v))
    denominator = norm(token) * norm(event)
    cosine = sum(a * b for a, b in zip(token, event)) / denominator
    conflict = sum(a * b < 0 for a, b in zip(token, event)) / n
    ratio = 0.1 * norm(event) / norm(token)
    return s, conflict, cosine, ratio


def main():
    cases = {
        "short_spike_16": ([0.05] * 7 + [0.9] + [0.05] * 8, 1),
        "short_spike_256": ([0.05] * 127 + [0.9] + [0.05] * 128, 1),
        "long_signal_256": ([0.05] * 64 + [0.8] * 128 + [0.05] * 64, 1),
        "safe_correction_16": ([0.8] * 8 + [0.05] * 8, 0),
        "constant_16": ([0.2] * 16, 1),
        "constant_256": ([0.2] * 256, 1),
    }
    print("| Synthetic case | Pooling | Score | Opposing coordinates | Cosine | 0.1 event/token norm |")
    print("|---|---|---:|---:|---:|---:|")
    for name, (p, y) in cases.items():
        for kind, alpha in [("endpoint", 1), ("linear_mil", 1),
                            ("auto_pool", 1), ("auto_pool", 8)]:
            s, conflict, cosine, ratio = diagnose(p, y, kind, alpha)
            label = f"{kind}({alpha})" if kind == "auto_pool" else kind
            print(f"| {name} | {label} | {s:.6f} | {conflict:.2%} | {cosine:.6f} | {ratio:.6f} |")
    p = cases["short_spike_16"][0]
    padded = p + [0.999] * 240
    mask = [True] * len(p) + [False] * 240
    valid = [x for x, keep in zip(padded, mask) if keep]
    for kind in ["endpoint", "linear_mil", "auto_pool"]:
        print(f"Masked padding score delta, {kind}: {pooled(valid, kind)[0] - pooled(p, kind)[0]:.1f}")
    print("Duplicate complete bag score deltas (endpoint, MIL, auto-pool):",
          [pooled(p * 2, k)[0] - pooled(p, k)[0]
           for k in ["endpoint", "linear_mil", "auto_pool"]])
    print("Positive endpoint equilibrium (g_end=0.6/1.14, event weight=0.1):")
    for n in [16, 256, 4096]:
        print(n, (0.6 / 1.14 + 0.1 * n) / (1 + 0.1 * n))


if __name__ == "__main__":
    main()
