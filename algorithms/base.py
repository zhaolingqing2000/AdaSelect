import math, time
import numpy as np

ADAPTIVE={'AdaSelect','CBAA','AS','DF'}
SELECTIVE={'AdaSelect','AS','SS'}

class Allocator:
    def __init__(self,name,cfg,rng): self.name=name; self.cfg=cfg; self.rng=rng
    def fitness(self,est,task): return np.sum(task.weights*np.minimum(est/task.requirement,1),axis=1)
    def choose(self,agents,task):
        start=time.perf_counter_ns(); avail=np.array([a.id for a in agents if a.available],dtype=int)
        if not len(avail): return None,{},(time.perf_counter_ns()-start)/1e6
        est=np.stack([agents[i].estimate for i in avail]); loads=np.array([agents[i].workload(self.cfg.l_max) for i in avail])
        fit=self.fitness(est,task)
        if self.name=='AS': rank=.7*est.mean(1)-.3*loads
        else: rank=fit-self.cfg.beta*loads
        if self.name in {'CBAA','CAA','DF','Static Full'}: cand_idx=np.arange(len(avail))
        elif self.name=='LC-DTA': cand_idx=np.sort(self.rng.choice(len(avail),math.ceil(math.sqrt(len(avail))),replace=False))
        else:
            k=min(len(avail),max(1,math.ceil(self.cfg.kappa*self.cfg.n_agents)))
            cand_idx=np.array(sorted(range(len(avail)),key=lambda x:(-rank[x],avail[x]))[:k])
        cids=avail[cand_idx]; ce=est[cand_idx]; cl=loads[cand_idx]; cf=fit[cand_idx]
        ratios=ce/np.maximum(task.requirement,1e-6); pred=task.load/np.maximum(1e-6,ratios.mean(1)); pl=pred*(1+cl); cost=pl/pl.max()
        if self.name=='AS': util=.7*ce.mean(1)-.3*cl-self.cfg.eta*cost
        else: util=cf-self.cfg.beta*cl-self.cfg.eta*cost
        wi=sorted(range(len(cids)),key=lambda x:(-util[x],cids[x]))[0]
        nav=len(avail); k=len(cids)
        if self.name=='CBAA': msg=nav+nav+2*(nav-1)
        elif self.name in {'CAA','Static Full'}: msg=nav+nav+1
        else: msg=2*k+1
        # Evaluation-only full B and candidate quality; never used for decision.
        allrat=est/np.maximum(task.requirement,1e-6); allp=task.load/np.maximum(1e-6,allrat.mean(1))*(1+loads); alle=allp/allp.max()
        fullb=fit-self.cfg.beta*loads-self.cfg.eta*alle; candb=util
        meta={'estimated_fitness':float(cf[wi]),'messages':int(msg),'k':k,'navail':nav,
              'b_candidate':float(candb.max()),'b_full':float(fullb.max())}
        return int(cids[wi]),meta,(time.perf_counter_ns()-start)/1e6

