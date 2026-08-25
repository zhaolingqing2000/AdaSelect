import dataclasses
import numpy as np
from config import Config
from environment import generate_environment
from models import Agent
from algorithms.base import Allocator
from simulator import run_simulation

def small(**kw): return dataclasses.replace(Config(n_agents=12,n_tasks=40,failure_rate=0),**kw)

def test_environment_bounds_and_determinism():
    c=small(); a=generate_environment(c,7); b=generate_environment(c,7)
    assert np.array_equal(a.nominal,b.nominal) and np.all((a.nominal>=0)&(a.nominal<=1))
    assert all(np.all((t.requirement>=0)&(t.requirement<=1)) and np.isclose(t.weights.sum(),1) for t in a.tasks)

def test_fitness_cost_candidates_and_unavailable():
    c=small(kappa=1); e=generate_environment(c,1); agents=[Agent(i,x.copy(),x.copy(),x.copy()) for i,x in enumerate(e.nominal)]
    agents[2].available=False; alg=Allocator('AdaSelect',c,np.random.default_rng(1)); win,m,_=alg.choose(agents,e.tasks[0])
    assert win!=2 and m['k']==m['navail']==11 and 0<=m['estimated_fitness']<=1
    assert m['b_candidate']<=m['b_full']+1e-12 and m['messages']>=0

def test_rho_one_no_update_and_zero_drift():
    c=small(rho=1,drift_degradation=0,feedback_delay=0); e=generate_environment(c,2)
    before=e.nominal.copy(); r=run_simulation(e,c,'AdaSelect',2)
    assert np.array_equal(before,e.nominal) and r['q']<=1+1e-12

def test_delay_is_arrival_queue_and_metrics_bounds():
    c=small(feedback_delay=5); e=generate_environment(c,3); r=run_simulation(e,c,'AdaSelect',3)
    assert 0<=r['tcr']<=1 and 0<=r['af']<=1 and 0<=r['actual_fitness']<=1
    assert 0<=r['mean_workload']<=1 and r['messages_per_task']>=0

