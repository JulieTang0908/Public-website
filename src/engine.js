/* Exact 1-D composition inference. Shared by browser workers and Node checks.
   Assumptions: independent calibrated Gaussian cell noise; no blur/overlap.
   P(a,b|N)=1/K; placements within each pair have weight 1/R(N,a,b). */
(function (root) {
  'use strict';
  const fail = message => { throw new Error(message); };
  function integer(value, name, lo, hi) {
    if (!Number.isInteger(value) || value < lo || value > hi) fail(`${name} 必须是 ${lo}～${hi} 的整数`);
    return value;
  }
  function count(N) { integer(N, 'N', 1, 100); const b = Math.floor(N / 2); return (b + 1) * (N + 1 - b); }
  function encode(N, a, b) {
    count(N); integer(a, 'a', 0, N); integer(b, 'b', 0, Math.floor(N / 2));
    if (a + 2 * b > N) fail('组成超出视野');
    return b * (N + 2 - b) + a;
  }
  function decode(N, x) {
    integer(x, 'x', 0, count(N) - 1);
    let b = 0;
    while (b < Math.floor(N / 2) && (b + 1) * (N + 1 - b) <= x) b++;
    return [x - b * (N + 2 - b), b];
  }
  function logadd(a, b) {
    if (a === -Infinity) return b;
    if (b === -Infinity) return a;
    const m = Math.max(a, b);
    return m + Math.log1p(Math.exp(-Math.abs(a - b)));
  }
  function posterior(z, sigma, mode = 'typed') {
    const N = z.length; count(N);
    if (!Array.from(z).every(Number.isFinite)) fail('读数必须全部是有限数值');
    if (!Number.isFinite(sigma) || sigma <= 0) fail('σ 必须大于 0');
    if (!['typed', 'binary'].includes(mode)) fail('观测模式无效');
    const width = N + 1, size = width * (Math.floor(N / 2) + 1);
    let prev2 = new Float64Array(size).fill(-Infinity);
    let prev1 = new Float64Array(size).fill(-Infinity);
    let next = new Float64Array(size).fill(-Infinity);
    prev1[0] = 0; // log Q(0,0,0)=0; illegal states remain -Infinity.
    const c = mode === 'typed' ? 2 : 1;
    const emission = (i, v) => -0.5 * ((z[i] - v) / sigma) ** 2;
    for (let i = 1; i <= N; i++) {
      next.fill(-Infinity);
      const empty = emission(i - 1, 0), short = emission(i - 1, 1);
      const rod = i >= 2 ? emission(i - 2, c) + emission(i - 1, c) : -Infinity;
      if (![empty, short].every(Number.isFinite) || (i >= 2 && !Number.isFinite(rod))) fail('读数与 σ 的尺度超出浮点计算范围');
      for (let b = 0; b <= Math.floor(i / 2); b++) {
        for (let a = 0; a <= i - 2 * b; a++) {
          const k = b * width + a;
          // Q_i = empty*Q_(i-1,a,b) + A*Q_(i-1,a-1,b)
          //       + B_(i-1)*B_i*Q_(i-2,a,b-1), accumulated in log space.
          let value = prev1[k] + empty;
          if (a > 0) value = logadd(value, prev1[k - 1] + short);
          if (b > 0) value = logadd(value, prev2[k - width] + rod);
          next[k] = value;
        }
      }
      [prev2, prev1, next] = [prev1, next, prev2];
    }
    const factorial = [0];
    for (let i = 1; i <= N; i++) factorial[i] = factorial[i - 1] + Math.log(i);
    const scores = [], pairs = []; let normalizer = -Infinity;
    for (let b = 0; b <= Math.floor(N / 2); b++) {
      for (let a = 0; a <= N - 2 * b; a++) {
        // log R = log((N-b)!/(a! b! (N-a-2b)!)).
        const logR = factorial[N - b] - factorial[a] - factorial[b] - factorial[N - a - 2 * b];
        const score = prev1[b * width + a] - logR;
        scores.push(score); pairs.push([a, b]); normalizer = logadd(normalizer, score);
      }
    }
    if (!Number.isFinite(normalizer)) fail('没有可归一化的有限似然');
    // Convert to mass only at the output boundary. Extremely small masses may
    // round to zero; log_probability retains their finite logarithms.
    const log_probability = scores.map(v => v - normalizer);
    const probability = log_probability.map(Math.exp);
    let best = 0; for (let x = 1; x < probability.length; x++) if (probability[x] > probability[best]) best = x;
    return {probability, log_probability, map: pairs[best], map_index: best};
  }
  function validateMotif(motif) {
    if (!Array.isArray(motif) || !motif.length || motif.length > 100 || !motif.every(v => [0, 1, 2].includes(v))) fail('基本序列需要 1～100 个 0、1、2');
    let run = 0;
    for (const value of [...motif, 0]) {
      if (value === 2) run++;
      else { if (run % 2) fail('基本序列中每段连续 2 必须能组成完整长杆'); run = 0; }
    }
    return motif;
  }
  function scene(N, motif = [0, 1, 2, 2, 1, 0]) {
    count(N); validateMotif(motif);
    const row = Array.from({length: N}, (_, i) => motif[i % motif.length]);
    let run = 0; for (let i = N - 1; i >= 0 && row[i] === 2; i--) run++;
    if (run % 2) row[N - 1] = 0; // Remove an incomplete right-boundary rod.
    return row;
  }
  function normalNoise(seed, length = 100) {
    integer(seed, '种子', 1, 4294967295); integer(length, '噪声长度', 1, 100);
    let state = seed >>> 0;
    const uniform = () => {
      state ^= state << 13; state ^= state >>> 17; state ^= state << 5; state >>>= 0;
      return (state + 0.5) / 4294967296;
    };
    const result = [];
    while (result.length < length) {
      const radius = Math.sqrt(-2 * Math.log(uniform())), angle = 2 * Math.PI * uniform();
      result.push(radius * Math.cos(angle));
      if (result.length < length) result.push(radius * Math.sin(angle));
    }
    return result;
  }
  const api = {count, encode, decode, posterior, scene, normalNoise, validateMotif};
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.CompositionEngine = api;
})(typeof self === 'object' ? self : globalThis);
