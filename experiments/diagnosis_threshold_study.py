




import argparse, csv, math, random
from pathlib import Path
import adaselect_experiment as base

HERE=Path(__file__).resolve().parent
DEFAULT={'min_age':4, 'slow_ratio':1.50, 'ref_ratio':0.90, 'interval_margin_ms':0.020}
CASES = [
    ('default', DEFAULT),
    ('age2', dict(DEFAULT,min_age=2)),('age8',dict(DEFAULT,min_age=8)),
    ('slow1p3',dict(DEFAULT,slow_ratio=1.3)),('slow1p7',dict(DEFAULT,slow_ratio=1.7)),
    ('ref0p8',dict(DEFAULT,ref_ratio=0.8)),('ref1p0',dict(DEFAULT,ref_ratio=1.0)),
    ('margin0',dict(DEFAULT,interval_margin_ms=0.0)),('margin0p04',dict(DEFAULT,interval_margin_ms=0.04)),
]

def elig_factory(conf):
    def eligible(theta,theta0,last,cls,iters,step):
        pred,_=base.estimate_state(theta,theta0,last,cls,iters)
        incumbent=min(range(len(theta)),key=lambda i:pred[i])
        age_c=step-last[incumbent][cls]
        upper_c=1.12*pred[incumbent]+base.V_MS_PER_STEP*age_c
        ret=[]
        for i in range(len(theta)):
            if i==incumbent:continue
            age_i=step-last[i][cls]
            if age_i<conf['min_age']:continue
            ref_i=iters*theta0[i][cls]
            if pred[i]<=conf['slow_ratio']*ref_i:continue
            if ref_i>=conf['ref_ratio']*pred[incumbent]:continue
            lower_i=max(0.0,0.88*pred[i]-base.V_MS_PER_STEP*age_i)
            if upper_c-lower_i<=conf['interval_margin_ms']:continue
            gain=pred[incumbent]-ref_i+upper_c-lower_i
            den=1.0+pred[i]/base.CURRENT_DEADLINE_MS
            ret.append((i,gain,min(age_i,24),den))
        return incumbent,ret
    return eligible

def run_study(out,seeds=range(6000,6010)):
    out.parent.mkdir(parents=True,exist_ok=True)
    rows=[]
    orig_eligible=base.eligible_candidates
    cluster=base.WorkerCluster(base.MAX_WORKERS)
    try:
        for seed in seeds:
            n=12
            theta0=base.coordinator_calibration(cluster,n,seed)
            tasks=base.make_tasks(seed,'cyclic')
            cases=CASES[:]
            random.Random(seed*123+11).shuffle(cases)
            for name,conf in cases:
                base.eligible_candidates=elig_factory(conf)
                try:
                    r=base.run_trial(cluster,n,seed,'AdaSelect',tasks,theta0,'recovery',experiment='eligibility_sensitivity')
                finally:
                    base.eligible_candidates=orig_eligible
                r['threshold_variant']=name
                for k,v in conf.items():r[k]=v
                rows.append(r)
                with out.open('w',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
                print(seed,name,'TCR',round(r['tcr'],2),'msg',round(r['messages_per_task'],3),'diag',round(r['diagnoses_per_task'],3),flush=True)
    finally:cluster.close();base.eligible_candidates=orig_eligible

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seeds',type=int,default=10);ap.add_argument('--out',default=str(HERE/'diagnosis_threshold_runs.csv'));a=ap.parse_args()
    run_study(Path(a.out),range(6000,6000+a.seeds))
