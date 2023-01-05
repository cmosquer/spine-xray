import wandb
from dataclasses import dataclass
from .._utils import set_sweep_name

@dataclass
class WandbSweep:
    entity: str
    project: str
    config: dict
    pipeline: str
    run_count: int = 50
    sweep_id: str = ''
    
    def __post_init__(self):
        if not self.sweep_id:
            self.config['name'] = set_sweep_name()
            self.sweep_id =  wandb.sweep(sweep=self.config, entity=self.entity, project=self.project,) 


@dataclass
class WandbSingleRun:
    entity: str
    project: str
    config: dict
    pipeline: str
    
