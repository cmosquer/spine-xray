import torch
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from training_module._utils import set_run_name, set_run_config
from training_module.logger import wandb_login, WandbSingleRun, WandbSweep
from training_module.pipeline import model_pipeline_single, model_pipeline_sweep_DCR, model_pipeline_sweep_HBR


def run(wandb_class: WandbSingleRun or WandbSweep):
    run_name = set_run_name()
    if isinstance(wandb_class, WandbSingleRun):
        with wandb.init(project=wandb_class.project, 
                        entity=wandb_class.entity, 
                        config=wandb_class.config, 
                        name=run_name,
                        tags=[wandb_class.config['model_architecture']['model'], "Single Run"]):
            model_pipeline_single(wandb_class)
    elif isinstance(wandb_class, WandbSweep) and wandb_class.pipeline == 'DCR':
        wandb.agent(sweep_id=wandb_class.sweep_id, function=model_pipeline_sweep_DCR, count=wandb_class.run_count, entity=wandb_class.entity, project=wandb_class.project,)
    elif isinstance(wandb_class, WandbSweep) and wandb_class.pipeline == 'HBR':
        wandb.agent(sweep_id=wandb_class.sweep_id, function=model_pipeline_sweep_HBR, count=wandb_class.run_count, entity=wandb_class.entity, project=wandb_class.project,)


def single_run():
    model_config = {
        "model": 'unet_sm',
        "backbone": 'efficientnet-b4',
        "encoder_weights": 'imagenet',
        "decoder_use_batchnorm": False,
        "levels": 5
    }

    run_config = set_run_config(epochs=1, 
                                model = model_config, 
                                dataset = 'Leeds Sports Pose Dataset', 
                                optimizer = 'adam', 
                                loss_func = 'heatmap_loss', 
                                lr = 0.0003419, 
                                batch_size = 4, 
                                sigma = 0.0656, 
                                weight_decay = 0, 
                                patience = 10)
        
    wandb_single_run = WandbSingleRun(entity='brunocruzfranchi', 
                                      project='landmarkDetectionHeatmaps', 
                                      config=run_config,
                                      pipeline = 'HBR')
    
    run(wandb_single_run)


@hydra.main(version_base=None, config_path='./configs', config_name='config')
def sweep_run_HBR(cfg: DictConfig):
    
    wandb_config = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
        
    sweep_config = hydra.compose(overrides=["+sweep=heatmaps"])
    
    sweep_config = OmegaConf.to_container(
        sweep_config.sweep, resolve=True, throw_on_missing=True
    )
    
    wandb_sweep = WandbSweep(entity=wandb_config['entity'], project=wandb_config['project'], 
                             config=sweep_config, pipeline='HBR', run_count=150, 
                             sweep_id= '' if wandb_config['sweep_id'] is None else wandb_config['sweep_id'])    
    
    run(wandb_sweep)   


@hydra.main(version_base=None, config_path='./configs', config_name='config')
def sweep_run_DCR(cfg: DictConfig):
    
    wandb_config = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
        
    sweep_config = hydra.compose(overrides=["+sweep=direct"])
    
    sweep_config = OmegaConf.to_container(
        sweep_config.sweep, resolve=True, throw_on_missing=True
    )
    
    wandb_sweep = WandbSweep(entity=wandb_config['entity'], project=wandb_config['project'], 
                             config=sweep_config, pipeline='DCR', run_count=150, 
                             sweep_id= '' if wandb_config['sweep_id'] is None else wandb_config['sweep_id'])
    
    run(wandb_sweep)   


if __name__ == "__main__":
    global device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    wandb_login()
    sweep_run_HBR()
    