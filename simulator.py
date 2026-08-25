import copy, heapq
import numpy as np
from models import Agent
from algorithms.base import Allocator,ADAPTIVE

def run_simulation(env,cfg,method,seed):
    agents=[Agent(i,c.copy(),c.copy(),c.copy()) for i,c in enumerate(env.nominal)]
    rng=np.random.default_rng(seed+1000003+sum(map(ord,method)))
    alg=Allocator(method,cfg,rng); events=[]; seq=0
    def push(t,kind,data):
        nonlocal seq; seq+=1; heapq.heappush(events,(t,seq,kind,data))
    for task in env.tasks: push(task.arrival,'arrival',(task.id,0))
    for f in env.failures: push(f.start,'failure',f); push(f.end,'recovery',f)
    drift_index=int(cfg.drift_start*cfg.n_tasks); drift_done=False
    active={}; pending=[]; arrival_count=0
    m={'success':0,'failed':0,'attempts':0,'messages':0,'reallocations':0,
       'af':[],'actual':[],'latency':[],'cee':[],'workload':[],'waiting':[],
       'q':[],'r':[],'q_excluded':0,'af_by_arrival':[]}
    now=0.
    while events:
        now,_,kind,data=heapq.heappop(events)
        for a in agents: a.advance(now)
        if kind=='failure':
            a=agents[data.agent_id]; a.available=False
            for tid in list(a.jobs):
                attempt=active.get(tid,{}).get('attempt',0); a.jobs.pop(tid,None); active.pop(tid,None)
                m['reallocations']+=1
                if attempt<cfg.max_attempts: push(now,'arrival',(tid,attempt))
                else: m['failed']+=1
        elif kind=='recovery': agents[data.agent_id].available=True
        elif kind=='feedback':
            aid,obs,mask=data
            if method in ADAPTIVE:
                agents[aid].estimate[mask]=cfg.rho*agents[aid].estimate[mask]+(1-cfg.rho)*obs[mask]
        elif kind=='completion':
            tid,aid,attempt,token=data
            if tid not in active or active[tid]['token']!=token: continue
            a=agents[aid]; a.jobs.pop(tid,None); rec=active.pop(tid); task=env.tasks[tid]
            success=task.success_u[attempt] < np.clip(.15+.85*rec['truefit'],0,1)
            if success: m['success']+=1
            else: m['failed']+=1
            obs=np.clip(a.true+task.obs_noise[attempt],0,1); mask=task.requirement>.1
            target=arrival_count+cfg.feedback_delay
            if cfg.feedback_delay==0: push(now,'feedback',(aid,obs,mask))
            else: pending.append((target,aid,obs,mask))
        elif kind=='arrival':
            tid,attempt=data; task=env.tasks[tid]
            if attempt==0:
                arrival_count+=1
                if not drift_done and arrival_count>drift_index:
                    before=np.stack([a.estimate.copy() for a in agents])
                    for i in env.drift_agents: agents[i].true*=1-cfg.drift_degradation
                    assert np.array_equal(before,np.stack([a.estimate for a in agents]))
                    drift_done=True
                due=[x for x in pending if x[0]<=arrival_count]; pending=[x for x in pending if x[0]>arrival_count]
                for _,aid,obs,mask in due: push(now,'feedback',(aid,obs,mask))
                m['cee'].append(float(np.mean(np.abs(np.stack([a.estimate for a in agents])-np.stack([a.true for a in agents])))))
                m['workload'].append(float(np.mean([a.workload(cfg.l_max) for a in agents])))
            winner,meta,lat=alg.choose(agents,task)
            if winner is None:
                if attempt+1<cfg.max_attempts: push(now+1e-9,'arrival',(tid,attempt+1))
                else: m['failed']+=1
                continue
            m['attempts']+=1; m['messages']+=meta['messages']; m['latency'].append(lat); m['af'].append(meta['estimated_fitness'])
            m['r'].append(meta['k']/meta['navail'])
            if method=='AdaSelect' and meta['b_full']>0:
                assert meta['b_candidate']<=meta['b_full']+1e-12
                m['q'].append(meta['b_candidate']/meta['b_full'])
            elif method=='AdaSelect': m['q_excluded']+=1
            a=agents[winner]; load_before=a.workload(cfg.l_max)
            truefit=float(np.sum(task.weights*np.minimum(a.true/task.requirement,1))); m['actual'].append(truefit)
            rate=max(.1,truefit)/(1+load_before); service=task.load/rate*task.exec_noise[attempt]
            token=(tid,attempt,seq); active[tid]={'token':token,'attempt':attempt+1,'truefit':truefit}
            a.jobs[tid]=[task.load,now,now+service]; push(now+service,'completion',(tid,winner,attempt,token))
            m['waiting'].append(0.)
            if attempt==0: m['af_by_arrival'].append(meta['estimated_fitness'])
    total=cfg.n_tasks
    def mean(x): return float(np.mean(x)) if x else float('nan')
    pre=m['af_by_arrival'][max(0,drift_index-200):drift_index]; threshold=.95*mean(pre); recovery=float('nan')
    vals=m['af_by_arrival']
    for i in range(drift_index+49,len(vals)):
        if mean(vals[i-49:i+1])>=threshold: recovery=i-drift_index+1; break
    return {'method':method,'seed':seed,'tcr':m['success']/total,'af':mean(m['af']),'actual_fitness':mean(m['actual']),
      'latency_ms':mean(m['latency']),'messages_per_task':m['messages']/total,'reallocations':m['reallocations'],
      'cee':mean(m['cee']),'recovery_time':recovery,'q':mean(m['q']),'r':mean(m['r']),
      'q_excluded':m['q_excluded'],'mean_waiting_time':mean(m['waiting']),'mean_workload':mean(m['workload']),
      'attempts':m['attempts'],'successes':m['success'],'failed':m['failed']}
