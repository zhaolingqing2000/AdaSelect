import argparse, dataclasses, os, sys
from pathlib import Path
import pandas as pd
from config import DEFAULT,QUICK
from environment import generate_environment
from simulator import run_simulation
from metrics import summarize,paired_significance

METHODS=['CBAA','LC-DTA','AS','CAA','AdaSelect']

def specs(base):
    yield 'overall',{},METHODS
    for n in [20,50,100,200,500]: yield 'scalability',{'n_agents':n},METHODS
    for x in [0,.1,.2,.3,.4]: yield 'drift',{'drift_degradation':x},METHODS
    for x in [0,.05,.1,.2]: yield 'failures',{'failure_rate':x},['AS','CAA','AdaSelect']
    for x in [0,5,10,20]: yield 'feedback_delay',{'feedback_delay':x},['AS','DF','AdaSelect']
    for x in [.1,.2,.3,.5,1.]: yield 'candidate_ratio',{'kappa':x},['AdaSelect']
    for x in [.5,.7,.8,.9,.95]: yield 'forgetting_factor',{'rho':x},['AdaSelect']
    for x in [.5,1.,1.5,2.]: yield 'arrival_rate',{'arrival_rate':x},['SS','AS','AdaSelect']
    yield 'ablation',{},['Static Full','SS','DF','AdaSelect']

PARAM_COL={'n_agents':'n_agents','drift_degradation':'degradation','failure_rate':'failure_rate','feedback_delay':'delay','kappa':'kappa','rho':'rho','arrival_rate':'arrival_rate'}

def paper_outputs(summary_dir):
    def latex_df(d):
        esc=lambda x:str(x).replace('_','\\_').replace('%','\\%').replace('&','\\&')
        cols=list(d.columns); out=['\\begin{tabular}{'+'l'+'r'*(len(cols)-1)+'}','\\toprule',' & '.join(map(esc,cols))+' \\\\','\\midrule']
        for _,r in d.iterrows():
            vals=[f'{x:.4f}' if isinstance(x,float) else str(x) for x in r]
            out.append(' & '.join(map(esc,vals))+' \\\\')
        return '\n'.join(out+['\\bottomrule','\\end{tabular}'])
    def table(file,title):
        p=summary_dir/file
        if not p.exists(): return ''
        d=pd.read_csv(p); return '\\begin{table}[ht]\\centering\n\\caption{'+title+'}\n'+latex_df(d)+'\n\\end{table}\n'
    cfg=pd.DataFrame([dataclasses.asdict(DEFAULT)]).T.reset_index(); cfg.columns=['parameter','value']
    parts=['\\begin{table}[ht]\\centering\n\\caption{Default Simulation Configuration}\n'+latex_df(cfg)+'\n\\end{table}',
      table('comparison.csv','Comparison with Representative Methods'),table('communication_by_n.csv','Communication under Increasing System Scale'),
      table('af_by_n.csv','Allocation Fitness under Increasing System Scale'),table('drift.csv','Average Fitness under Increasing Capability Drift'),
      table('failures.csv','Robustness under Dynamic Conditions'),table('candidate_ratio.csv','Communication Trade-off and Sensitivity'),table('ablation.csv','Ablation Study')]
    (summary_dir/'paper_tables.tex').write_text('\n'.join(parts),encoding='utf-8')
    notes=['# Experimental results notes','All statements below are generated from the full-mode CSV files; no values are hand-entered.']
    for fn,title in [('comparison.csv','Comparison with Existing Methods'),('scalability.csv','Scalability'),('drift.csv','Adaptation to Capability Drift'),('failures.csv','Robustness'),('candidate_ratio.csv','Communication--Quality Trade-off'),('forgetting_factor.csv','Parameter Sensitivity'),('ablation.csv','Ablation')]:
        d=pd.read_csv(summary_dir/fn); notes += [f'## {title}',d.to_markdown(index=False)]
        if 'method' in d and 'af_mean' in d:
            best=d.loc[d.af_mean.idxmax()]; notes.append(f"Highest mean AF in this table: {best['method']} ({best['af_mean']:.6f}).")
            if (d.method=='AdaSelect').any():
                ada=float(d.loc[d.method=='AdaSelect','af_mean'].max())
                notes.append(f"AF hypothesis check: {'supported' if best['method']=='AdaSelect' else 'not supported'}; best AdaSelect row is {ada:.6f}.")
        if 'method' in d and 'tcr_mean' in d:
            best=d.loc[d.tcr_mean.idxmax()]; notes.append(f"Highest mean TCR in this table: {best['method']} ({best['tcr_mean']:.6f}).")
            if (d.method=='AdaSelect').any():
                ada=float(d.loc[d.method=='AdaSelect','tcr_mean'].max())
                notes.append(f"TCR hypothesis check: {'supported' if best['method']=='AdaSelect' else 'not supported'}; best AdaSelect row is {ada:.6f}.")
        notes.append('Any manuscript claim naming a different best method must be revised; these generated tables are authoritative.')
    (summary_dir/'paper_results_notes.md').write_text('\n\n'.join(notes),encoding='utf-8')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--mode',choices=['quick','full'],required=True); args=ap.parse_args()
    base=QUICK if args.mode=='quick' else DEFAULT; seeds=range(3) if args.mode=='quick' else range(30)
    root=Path(__file__).parent/'results'; root=(root/'quick') if args.mode=='quick' else root
    rawdir=root/'raw'; sumdir=root/'summary'; rawdir.mkdir(parents=True,exist_ok=True); sumdir.mkdir(parents=True,exist_ok=True)
    allrows=[]; current_exp=None; current_rows=[]
    def flush(exp,rows):
        if exp is None: return
        filename='comparison' if exp=='overall' else exp
        raw=pd.DataFrame(rows); raw.to_csv(rawdir/f'{filename}.csv',index=False); summarize(raw).to_csv(sumdir/f'{filename}.csv',index=False)
    for exp,changes,methods in specs(base):
        if current_exp is not None and exp!=current_exp:
            flush(current_exp,current_rows); current_rows=[]
        current_exp=exp
        rows=[]; cfg=dataclasses.replace(base,**changes)
        for seed in seeds:
            env=generate_environment(cfg,seed)
            for method in methods:
                rec=run_simulation(env,cfg,method,seed); rec['experiment']=exp
                if exp=='ablation':
                    rec['adaptation_enabled']=method in {'DF','AdaSelect'}
                    rec['selective_enabled']=method in {'SS','AdaSelect'}
                for k,v in changes.items(): rec[PARAM_COL[k]]=v
                rows.append(rec); print(exp,changes,method,seed,flush=True)
        current_rows.extend(rows); allrows+=rows
    flush(current_exp,current_rows)
    allraw=pd.DataFrame(allrows); paired_significance(allraw).to_csv(sumdir/'significance.csv',index=False)
    # Required paper-specific scalability projections.
    sc=pd.read_csv(sumdir/'scalability.csv')
    sc.filter(regex='method|n_agents|messages_per_task').to_csv(sumdir/'communication_by_n.csv',index=False)
    sc.filter(regex='method|n_agents|af_').to_csv(sumdir/'af_by_n.csv',index=False)
    if args.mode=='full': paper_outputs(sumdir)

if __name__=='__main__': main()
