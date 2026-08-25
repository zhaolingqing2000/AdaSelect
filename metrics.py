import math
import numpy as np
from scipy import stats

IDCOLS={'experiment','method','n_agents','degradation','failure_rate','delay','kappa','rho','arrival_rate'}

def summarize(df):
    group=[c for c in df.columns if c in IDCOLS]
    numeric=[c for c in df.select_dtypes(include=np.number).columns if c not in group and c!='seed']
    rows=[]
    for keys,g in df.groupby(group,dropna=False,sort=False):
        row=dict(zip(group,keys if isinstance(keys,tuple) else (keys,)))
        for c in numeric:
            x=g[c].dropna().to_numpy(); n=len(x); mu=float(np.mean(x)) if n else np.nan; sd=float(np.std(x,ddof=1)) if n>1 else 0.
            h=float(stats.t.ppf(.975,n-1)*sd/math.sqrt(n)) if n>1 else 0.
            row.update({f'{c}_mean':mu,f'{c}_std':sd,f'{c}_ci95_low':mu-h,f'{c}_ci95_high':mu+h})
        rows.append(row)
    import pandas as pd
    return pd.DataFrame(rows)

def paired_significance(raw):
    rows=[]; base=raw[raw.experiment=='overall']
    for other in ['AS','CAA']:
      for metric in ['af','tcr']:
        p=base[base.method=='AdaSelect'].set_index('seed')[metric]; q=base[base.method==other].set_index('seed')[metric]
        d=(p-q).dropna(); normal_p=stats.shapiro(d).pvalue if len(d)>=3 else 0
        if normal_p>=.05:
            test='paired t-test'; pv=stats.ttest_rel(p.loc[d.index],q.loc[d.index]).pvalue; effect=d.mean()/d.std(ddof=1) if d.std(ddof=1)>0 else np.inf
        else:
            test='Wilcoxon signed-rank'; pv=stats.wilcoxon(d).pvalue if np.any(d) else 1.; effect=(np.sum(d>0)-np.sum(d<0))/len(d)
        lo,hi=stats.t.interval(.95,len(d)-1,loc=d.mean(),scale=stats.sem(d)) if len(d)>1 else (d.mean(),d.mean())
        rows.append({'comparison':f'AdaSelect vs {other}','metric':metric,'normality_p':normal_p,'test':test,'p_value':pv,'effect_size':effect,'ci95_low':lo,'ci95_high':hi})
    import pandas as pd
    return pd.DataFrame(rows)

