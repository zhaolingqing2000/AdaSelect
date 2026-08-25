import numpy as np
from models import Task,Failure,Environment

def generate_environment(cfg,seed):
    r=np.random.default_rng(seed)
    nominal=r.uniform(.2,1.,(cfg.n_agents,cfg.d))
    arrivals=np.cumsum(r.exponential(1/cfg.arrival_rate,cfg.n_tasks))
    tasks=[]
    for j,t in enumerate(arrivals):
        req=r.uniform(.1,.9,cfg.d); z=r.exponential(1.,cfg.d)
        tasks.append(Task(j,float(t),req,z/z.sum(),float(r.uniform(.6,1.4)),
            r.lognormal(0,.05,cfg.max_attempts),r.random(cfg.max_attempts),
            r.normal(0,cfg.observation_noise_std,(cfg.max_attempts,cfg.d))))
    drift=tuple(sorted(r.choice(cfg.n_agents,int(cfg.drift_fraction*cfg.n_agents),replace=False).tolist()))
    failures=[]
    if cfg.failure_rate>0:
        horizon=float(arrivals[-1])+20
        rate=-np.log(1-cfg.failure_rate)/100.
        for i in range(cfg.n_agents):
            t=float(r.exponential(1/rate))
            while t<horizon:
                dur=float(r.uniform(3,8)); failures.append(Failure(i,t,t+dur)); t+=dur+float(r.exponential(1/rate))
    return Environment(nominal,tuple(tasks),tuple(sorted(failures,key=lambda x:(x.start,x.agent_id))),drift)

