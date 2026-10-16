"""Reproducible command-line simulation; can also run as a standalone file."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
try:
    from .model import pairs, motif_scene, normal_noise, posterior, feature_marginals
except ImportError:
    from model import pairs, motif_scene, normal_noise, posterior, feature_marginals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config.json'))
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('output') / 'simulation')
    parser.add_argument('--N-min', type=int)
    parser.add_argument('--N-max', type=int)
    parser.add_argument('--sigma', type=float, action='append', help='Repeat for multiple sigma values')
    parser.add_argument('--mode', choices=('typed', 'binary'), action='append')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--observations', type=Path, help='JSON {"z":[...]}: infer one supplied cell vector; truth remains unknown')
    parser.add_argument('--skip-features', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    for attr, key in [('N_min', 'N_min'), ('N_max', 'N_max'), ('seed', 'seed')]:
        if getattr(args, attr) is not None:
            config[key] = getattr(args, attr)
    if args.sigma:
        config['sigmas'] = args.sigma
    if args.mode:
        config['modes'] = args.mode
    lo, hi = config['N_min'], config['N_max']
    pairs(lo); pairs(hi)
    if lo > hi:
        raise ValueError('N_min must be <= N_max')
    if not config['sigmas'] or any(type(v) is bool or not np.isfinite(v) or v <= 0 for v in config['sigmas']):
        raise ValueError('all sigma values must be positive and finite')
    if not config['modes'] or any(v not in ('typed', 'binary') for v in config['modes']):
        raise ValueError('modes must be typed or binary')
    if type(config['F2_samples']) is not int or config['F2_samples'] < 1:
        raise ValueError('F2_samples must be a positive integer')
    if type(config['F2_exact_max_N']) is not int or not 0 <= config['F2_exact_max_N'] <= 10:
        raise ValueError('F2_exact_max_N must be in 0..10 (exhaustive reference cost)')
    noise = normal_noise(config['seed'], hi)
    supplied = None
    if args.observations:
        supplied = np.asarray(json.loads(args.observations.read_text(encoding='utf-8'))['z'], dtype=float)
        if supplied.ndim != 1 or not np.all(np.isfinite(supplied)):
            raise ValueError('z must be one finite cell vector')
        pairs(len(supplied)); lo = hi = len(supplied)
        config['N_min'] = config['N_max'] = lo
    args.output.mkdir(parents=True, exist_ok=True)
    identity = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    result = {'config': config, 'config_sha256': identity, 'noise_algorithm': 'xorshift32 + Box-Muller',
              'prior': 'uniform legal count pairs, then uniform placements', 'slices': []}
    if supplied is not None:
        result['observations_sha256'] = hashlib.sha256(args.observations.read_bytes()).hexdigest()
    with (args.output / 'posterior.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['N', 'mode', 'sigma', 'x', 'u', 'a', 'b', 'probability', 'log_probability', 'relative_bin_density'])
        for N in range(lo, hi + 1):
            legal = pairs(N); K = len(legal)
            ideal = None if supplied is not None else motif_scene(N, config['motif'])
            row = {'N': N, 'K': K, 'ideal': ideal, 'truth': None if ideal is None else {'a': ideal.count(1), 'b': ideal.count(2) // 2}, 'posteriors': {}}
            if not args.skip_features:
                rng = np.random.default_rng(np.random.SeedSequence([config['seed'], N, 2718]))
                row['features'] = feature_marginals(N, config['F2_samples'], config['F2_exact_max_N'], rng)
            for mode in config['modes']:
                for sigma in config['sigmas']:
                    signal = None if ideal is None else (np.asarray(ideal) if mode == 'typed' else (np.asarray(ideal) > 0).astype(int))
                    z = supplied if supplied is not None else signal + sigma * noise[:N]
                    p, logs = posterior(z, sigma, mode)
                    best = int(np.argmax(p))
                    row['posteriors'][f'{mode}:{sigma:g}'] = {'z': z.tolist(), 'probability': p.tolist(), 'log_probability': logs.tolist(), 'MAP': list(legal[best]), 'MAP_probability': float(p[best]), 'probability_sum': float(p.sum())}
                    for x, ((a, b), probability, log_probability) in enumerate(zip(legal, p, logs)):
                        writer.writerow([N, mode, sigma, x, x / (K - 1), a, b, probability, log_probability, probability * (K - 1)])
            result['slices'].append(row)
            if N % 10 == 0 or N == hi:
                print(f'N={N}: completed', flush=True)
    (args.output / 'simulation.json').write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(',', ':')), encoding='utf-8')
    print(f'Saved simulation.json and posterior.csv to {args.output.resolve()}')


if __name__ == '__main__':
    main()
