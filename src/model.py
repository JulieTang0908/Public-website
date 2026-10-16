"""Standalone model primitives; NumPy/SciPy are used only by the Python CLI.

All composition likelihoods average over uniformly weighted latent placements.
The browser implementation in engine.js is checked against this implementation
and small-N exhaustive enumeration, not just against the displayed curve.
"""
from math import cos, sin, pi, sqrt, log
import numpy as np
from scipy.special import gammaln, logsumexp


def pairs(N):
    if isinstance(N, bool) or not isinstance(N, int) or not 1 <= N <= 100:
        raise ValueError('N must be an integer from 1 to 100')
    return [(a, b) for b in range(N // 2 + 1) for a in range(N - 2 * b + 1)]


def motif_scene(N, motif):
    pairs(N)
    if not motif or len(motif) > 100 or any(type(v) is not int or v not in (0, 1, 2) for v in motif):
        raise ValueError('motif needs 1..100 entries in {0,1,2}')
    run = 0
    for value in [*motif, 0]:
        if value == 2:
            run += 1
        else:
            if run % 2:
                raise ValueError('Every run of 2 in the motif must contain complete rods')
            run = 0
    row = [motif[i % len(motif)] for i in range(N)]
    run = 0
    for v in reversed(row):
        if v != 2:
            break
        run += 1
    if run % 2:
        row[-1] = 0
    return row


def normal_noise(seed, length):
    """xorshift32 + Box-Muller, also implemented in engine.js.

    This algorithm is intentionally explicit so new simulations have the same
    seed meaning in Python and JavaScript. Archived NumPy noise is separate.
    """
    if type(seed) is not int or not 1 <= seed <= 2**32 - 1:
        raise ValueError('seed must be an integer in 1..2**32-1')
    state = seed

    def uniform():
        nonlocal state
        state ^= (state << 13) & 0xffffffff
        state ^= state >> 17
        state ^= (state << 5) & 0xffffffff
        state &= 0xffffffff
        return (state + .5) / 2**32

    result = []
    while len(result) < length:
        radius, angle = sqrt(-2 * log(uniform())), 2 * pi * uniform()
        result.extend([radius * cos(angle), radius * sin(angle)])
    return np.asarray(result[:length])


def posterior(z, sigma, mode='typed'):
    """log Q recurrence with three rolling length layers (O(N^2) memory)."""
    z = np.asarray(z, dtype=float)
    if z.ndim != 1 or not np.all(np.isfinite(z)):
        raise ValueError('z must contain finite cell measurements')
    N = len(z)
    legal = pairs(N)
    if not np.isfinite(sigma) or sigma <= 0 or mode not in ('typed', 'binary'):
        raise ValueError('positive sigma and typed/binary mode required')
    c = 2 if mode == 'typed' else 1
    emissions = -.5 * ((z[:, None] - [0, 1, c]) / sigma) ** 2
    if not np.all(np.isfinite(emissions)):
        raise ValueError('measurement scale exceeds floating-point range')
    shape = (N + 1, N // 2 + 1)
    prev2 = np.full(shape, -np.inf)
    prev1 = np.full(shape, -np.inf)
    prev1[0, 0] = 0
    for i in range(1, N + 1):
        next_row = prev1 + emissions[i - 1, 0]
        next_row[1:] = np.logaddexp(next_row[1:], prev1[:-1] + emissions[i - 1, 1])
        if i >= 2:
            next_row[:, 1:] = np.logaddexp(next_row[:, 1:], prev2[:, :-1] + emissions[i - 2, 2] + emissions[i - 1, 2])
        prev2, prev1 = prev1, next_row
    scores = []
    for a, b in legal:
        # R=(N-b)!/[a! b! (N-a-2b)!]; divide Q by R before normalizing.
        log_R = gammaln(N - b + 1) - gammaln(a + 1) - gammaln(b + 1) - gammaln(N - a - 2 * b + 1)
        scores.append(prev1[a, b] - log_R)
    logs = np.asarray(scores) - logsumexp(scores)
    return np.exp(logs), logs


def placements(N, a, b):
    """Exhaustive reference only; used for exact small-N feature mixtures."""
    row = [0] * N

    def visit(i, empty, short, rod):
        if i == N:
            yield row.copy()
            return
        if empty:
            row[i] = 0
            yield from visit(i + 1, empty - 1, short, rod)
        if short:
            row[i] = 1
            yield from visit(i + 1, empty, short - 1, rod)
        if rod:
            row[i:i + 2] = [2, 2]
            yield from visit(i + 2, empty, short, rod - 1)
    yield from visit(0, N - a - 2 * b, a, b)


def feature_marginals(N, samples, exact_max_N, rng):
    """Ideal F0 and F2 under the two-stage composition/placement prior."""
    legal = pairs(N)
    p0 = np.zeros(2 * N + 1)
    p2 = np.zeros(4 * N + 1)
    for a, b in legal:
        p0[a + 4 * b] += 1 / len(legal)
    exact = N <= exact_max_N
    if exact:
        for a, b in legal:
            scenes = np.asarray(list(placements(N, a, b)))
            values = (scenes[:, :-1] * scenes[:, 1:]).sum(axis=1)
            p2 += np.bincount(values, minlength=len(p2)) / (len(legal) * len(scenes))
    else:
        labels = rng.integers(len(legal), size=samples)
        for label in np.unique(labels):
            a, b = legal[label]
            n = int(np.count_nonzero(labels == label))
            tokens = np.asarray([0] * (N - a - 2 * b) + [1] * a + [2] * b)
            # Uniform labeled permutations induce uniform multiset placements.
            draws = rng.permuted(np.broadcast_to(tokens, (n, len(tokens))), axis=1)
            lengths = 1 + (draws == 2)
            starts = np.cumsum(lengths, axis=1) - lengths
            rowids = np.broadcast_to(np.arange(n)[:, None], starts.shape)
            scenes = np.zeros((n, N), dtype=int)
            scenes[rowids, starts] = draws
            rods = draws == 2
            scenes[rowids[rods], starts[rods] + 1] = 2
            values = (scenes[:, :-1] * scenes[:, 1:]).sum(axis=1)
            p2 += np.bincount(values, minlength=len(p2)) / samples
    output = {}
    for name, p in [('F0', p0), ('F2', p2)]:
        values = np.arange(len(p))
        mean = float(values @ p)
        output[name] = {'values': values.tolist(), 'probability': p.tolist(), 'mean': mean,
                        'sd': float(np.sqrt(((values - mean) ** 2) @ p)),
                        'method': 'exact' if name == 'F0' or exact else 'Monte Carlo',
                        'samples': None if name == 'F0' or exact else samples}
    return output
