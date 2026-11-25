



import argparse, csv, importlib.util, random, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import adaselect_experiment as base

POLICIES = ('Class-Top-K','Obs.+Top-K','Rand-Obs','AdaSelect')
REGIMES = {'zero': (0.0, 0.0), 'constant_0p5': (0.5, 0.0), 'jitter_0p5_pm_0p4': (0.5,0.4)}

def run_study(out_path, seeds=range(5000,5010)):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows=[]
    cluster=base.WorkerCluster(base.MAX_WORKERS)
    try:
        for seed in seeds:
            n=12
            theta0=base.coordinator_calibration(cluster,n,seed)
            tasks=base.make_tasks(seed,'cyclic')
            for regime, (base_delay, jitter) in REGIMES.items():
                cases=list(POLICIES)
                random.Random(seed*19+len(regime)).shuffle(cases)
                for pol in cases:
                    orig_call=cluster.call
                    local_rng=random.Random(seed*1771+53)
                    def inject(worker_id,request,rpc_delay_ms=0.0):
                        delay = max(0.0, base_delay + local_rng.uniform(-jitter,jitter))
                        return orig_call(worker_id,request,rpc_delay_ms=delay)
                    cluster.call=inject
                    try:
                        row=base.run_trial(cluster,n,seed,pol,tasks,theta0,'recovery',experiment='rpc_stress')
                    finally:
                        cluster.call=orig_call
                    row['network_regime']=regime
                    row['rpc_delay_mean_ms']=base_delay
                    row['rpc_jitter_halfwidth_ms']=jitter
                    rows.append(row)
                    print(f"{seed} {regime} {pol} tcr={row['tcr']:.2f} p50={row['p50_cycle_ms']:.3f} p95={row['p95_cycle_ms']:.3f}",flush=True)
                    with out_path.open('w',newline='') as f:
                        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    finally:
        cluster.close()
    return rows

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--seeds',type=int,default=10)
    ap.add_argument('--out',default=str(HERE/'rpc_jitter_runs.csv'))
    args=ap.parse_args()
    run_study(Path(args.out),range(5000,5000+args.seeds))
