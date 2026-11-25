






from __future__ import annotations
import csv, random, statistics, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import adaselect_experiment as base

OUT = Path(__file__).with_name('uniform_diagnosis_comparison.csv')
FIELDS = None

def main():
    rows=[]
    cluster=base.WorkerCluster(base.MAX_WORKERS)
    original=base.select_diagnostic
    try:
        for seed in range(8000,8010):
            n=12
            theta0=base.coordinator_calibration(cluster,n,seed)
            tasks=base.make_tasks(seed,'cyclic')
            schedule=set(random.Random(seed*1447+7).sample(range(len(tasks)),27))
            def uniform_probe(policy,theta,theta0,last,cls,iters,step,n,rng,rr_orders,rr_ptr):
                return random.Random(seed*100081+step*479+31).randrange(n) if step in schedule else None
            for label in (['AdaSelect','Uniform-Probe'] if seed%2 else ['Uniform-Probe','AdaSelect']):
                base.select_diagnostic = original if label=='AdaSelect' else uniform_probe
                row=base.run_trial(cluster,n,seed,'AdaSelect',tasks,theta0,'recovery',experiment='budget_matched_diagnosis')
                row['policy']=label
                rows.append(row)
                with OUT.open('w',newline='') as fh:
                    writer=csv.DictWriter(fh,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
                print(f"seed={seed} policy={label} TCR={row['tcr']:.2f} post={row['post_tcr']:.2f} msg={row['messages_per_task']:.3f} diag={row['diagnoses_per_task']:.3f}",flush=True)
    finally:
        base.select_diagnostic=original
        cluster.close()
    print('CSV:',OUT)
    for name in ['AdaSelect','Uniform-Probe']:
        x=[r for r in rows if r['policy']==name]
        print(name, 'TCR',statistics.mean(r['tcr'] for r in x),'POST',statistics.mean(r['post_tcr'] for r in x),'MSG',statistics.mean(r['messages_per_task'] for r in x),'DIAG',statistics.mean(r['diagnoses_per_task'] for r in x),flush=True)

if __name__=='__main__':main()
