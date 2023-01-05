import torch
from typing import Dict
from itertools import islice
from datetime import datetime

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def average_loss(losses):
    """Calculate the average of per-location losses.
    Args:
        losses (Tensor): Predictions (B x L)
    """

    denom = losses.numel()

    # Prevent division by zero
    if isinstance(denom, int):
        denom = max(denom, 1)
    else:
        denom = denom.clamp(1)

    return losses.sum() / denom


def euclidean_losses(actual, target):
    """Calculate the Euclidean losses for multi-point samples.
    Each sample must contain `n` points, each with `d` dimensions.
    Args:
        actual (Tensor): Predictions (B x L x D)
        target (Tensor): Ground truth target (B x L x D)
    Returns:
        Tensor: Losses (B x L)
    """
    assert actual.size() == target.size(), 'input tensors must have the same size'
    return torch.linalg.norm(actual - target, ord=2, dim=-1, keepdim=False)  # Euclidean Norm para vectores


def set_run_name() -> str:
    now = datetime.today().strftime('%Y%m%d_%I:%M%p')
    return "SpineLandmarkDetection_" + now

def set_sweep_name() -> str:
    now = datetime.today().strftime('%Y%m%d_%I:%M%p')
    return "Sweep_SpineLandmarkDetection_" + now

def set_run_config(epochs='', model='', dataset='', optimizer='', loss_func='', lr='', batch_size='', sigma='', weight_decay='', patience='') -> Dict:
    wandb_config = {
        'epochs': epochs,
        "model_architecture": model,
        "datasets": dataset,
        'optimizer': optimizer,
        'loss_func': loss_func,
        "learning_rate": lr,
        "batch_size": batch_size,
        "sigma": sigma,
        'weight_decay': weight_decay,
        'patience': patience,
    }
    return wandb_config
  

def set_sweep_config(method, metric_name, goal, run_parameters):
    sweep_config = {
        'name': set_run_name(), 
        'method': method,
        'metric': {
            'name': metric_name, 
            'goal': goal
            }
        }

    sweep_config['parameters'] = run_parameters
    
    return sweep_config


def chunks(data, SIZE=10000):
    it = iter(data)
    for i in range(0, len(data), SIZE):
        yield {k:data[k] for k in islice(it, SIZE)}