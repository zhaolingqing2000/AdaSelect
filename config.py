from dataclasses import dataclass, replace

@dataclass(frozen=True)
class Config:
    n_agents:int=200; n_tasks:int=2000; d:int=5
    kappa:float=.2; rho:float=.8; alpha:float=1.; beta:float=.25; eta:float=.2
    arrival_rate:float=1.; drift_fraction:float=.3; drift_degradation:float=.3
    drift_start:float=.4; failure_rate:float=.1; feedback_delay:int=5
    observation_noise_std:float=.05; l_max:float=3.; max_attempts:int=3

DEFAULT=Config()
QUICK=replace(DEFAULT,n_tasks=300)

