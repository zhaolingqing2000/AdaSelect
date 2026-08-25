from dataclasses import dataclass, field
import numpy as np

@dataclass(frozen=True)
class Task:
    id:int; arrival:float; requirement:np.ndarray; weights:np.ndarray; load:float
    exec_noise:np.ndarray; success_u:np.ndarray; obs_noise:np.ndarray

@dataclass
class Agent:
    id:int; nominal:np.ndarray; true:np.ndarray; estimate:np.ndarray
    available:bool=True; jobs:dict=field(default_factory=dict)
    def workload(self,lmax): return min(sum(x[0] for x in self.jobs.values())/lmax,1.)
    def advance(self,t):
        for x in self.jobs.values():
            rem,last,end=x
            x[0]=0. if t>=end else rem*max(0.,end-t)/max(1e-12,end-last)
            x[1]=t

@dataclass(frozen=True)
class Failure:
    agent_id:int; start:float; end:float

@dataclass(frozen=True)
class Environment:
    nominal:np.ndarray; tasks:tuple; failures:tuple; drift_agents:tuple
