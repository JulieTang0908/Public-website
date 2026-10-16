/* UI integration: the frozen reference remains available until a new world is
   explicitly generated. A world uses one shared standard-noise prefix across N.
   Missing mode/sigma curves are then recomputed in a local Web Worker. */
const Live = (() => {
  let reference = true, generation = 0;
  const running = new Map();
  const workerCode = document.getElementById('engineSource').textContent + `
    self.onmessage = ({data}) => {
      try {
        const rows = [], logs = [];
        for (let N=1; N<=100; N++) {
          const signal=CompositionEngine.scene(N,data.motif);
          const z=signal.map((v,i)=>(data.mode==='typed'?v:Number(v>0))+data.sigma*data.noise[i]);
          const r=CompositionEngine.posterior(z,data.sigma,data.mode);
          rows.push(r.probability); logs.push(r.log_probability);
          if(N%10===0)self.postMessage({progress:N});
        }
        self.postMessage({rows,logs});
      }catch(e){self.postMessage({error:e.message});}
    };`;
  const logRows = {};
  function range() {
    const lo = Number($('Nmin').value), hi = Number($('Nmax').value);
    if (!Number.isInteger(lo) || !Number.isInteger(hi) || lo < 1 || hi > 100 || lo > hi) throw Error('N 范围需要满足 1 ≤ 起点 ≤ 终点 ≤ 100');
    return [lo, hi];
  }
  function motif() { return D.live_motif || D.config.scene_motif; }
  async function ensure(token) {
    if (currentView !== 'posterior') return;
    const sigma = Number($('noise').value), mode = $('rod').value, key = `${mode}:${sigma}`;
    if (!Number.isFinite(sigma) || sigma <= 0) throw Error('测量噪声 σ 必须是大于 0 的有限数');
    if (D.posterior[key]) return;
    if (running.has(key)) return running.get(key).promise;
    const epoch = generation, workerUrl=URL.createObjectURL(new Blob([workerCode], {type:'text/javascript'})),worker = new Worker(workerUrl);
    const entry = {};
    entry.promise = new Promise((resolve, reject) => {
      entry.reject = reject;
      worker.onmessage = ({data}) => {
        if (epoch !== generation) return;
        if (data.progress) { if (token === renderToken) $('status').textContent=`本机计算后验：${data.progress}/100 个 N…`; return; }
        worker.terminate(); URL.revokeObjectURL(workerUrl); running.delete(key);
        if (data.error) reject(Error(data.error));
        else { D.posterior[key] = data.rows; logRows[key] = data.logs; resolve(); }
      };
      worker.onerror = e => {worker.terminate();URL.revokeObjectURL(workerUrl);running.delete(key);reject(Error(e.message || '本机计算失败'));};
    });
    entry.worker = worker;entry.url=workerUrl;running.set(key, entry);
    worker.postMessage({mode, sigma, noise:D.noise, motif:motif()});
    return entry.promise;
  }
  function save(name, value, type='application/json') {
    const link=document.createElement('a'),url=URL.createObjectURL(new Blob([value],{type}));
    link.href=url;link.download=name;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  }
  function makeWorld() {
    const seed=Number($('seed').value), pattern=JSON.parse($('motif').value);
    CompositionEngine.validateMotif(pattern);
    const noise=CompositionEngine.normalNoise(seed);
    generation++;
    for (const entry of running.values()) {entry.worker.terminate();URL.revokeObjectURL(entry.url);entry.reject(Error('已切换模拟数据'));} running.clear();
    D.noise=noise;D.live_motif=pattern;D.signals=D.N_values.map(N=>CompositionEngine.scene(N,pattern));D.posterior={};
    for(const key of Object.keys(logRows))delete logRows[key];
    reference=false;D.live_seed=seed;currentView='posterior';
    $('worldStatus').textContent=`当前：本机新模拟，种子 ${seed}。各 N 共用噪声前缀；特征边际视图仍使用原先验的参考计算。`;
  }
  $('simulate').addEventListener('click',()=>{try{makeWorld();render();}catch(e){$('worldStatus').textContent=e.message;}});
  $('rangeApply').addEventListener('click',()=>render());
  $('currentOnly').addEventListener('click',()=>{$('Nmin').value=$('Nmax').value=$('Nnumber').value;render();});
  $('allN').addEventListener('click',()=>{$('Nmin').value=1;$('Nmax').value=100;render();});
  $('downloadSlice').addEventListener('click',async()=>{
    try {
      await ensure(renderToken); const N=clampN($('Nnumber').value),meta=byN[N-1],key=`${$('rod').value}:${Number($('noise').value)}`;
      const signal=D.signals[N-1],sigma=Number($('noise').value),mode=$('rod').value;
      const output={N,sigma,mode,source:reference?'frozen reference':'local simulation',seed:reference?D.config.seed:D.live_seed,
        noise_algorithm:reference?'archived NumPy normal vector':'xorshift32 + Box-Muller',prior:'uniform count pairs; uniform conditional placements',
        ideal:signal,observation:signal.map((v,i)=>(mode==='typed'?v:Number(v>0))+sigma*D.noise[i]),
        truth:{a:signal.filter(v=>v===1).length,b:signal.filter(v=>v===2).length/2},
        candidates:meta.v.map((x,i)=>({x,u:x/(meta.K-1),a:meta.a[i],b:meta.b[i],p:D.posterior[key][N-1][i],
          log_p:logRows[key]?.[N-1][i] ?? null}))};
      save(`posterior_N${N}.json`,JSON.stringify(output,null,2));
    }catch(e){$('worldStatus').textContent=e.message;}
  });
  return {range,ensure,save};
})();
