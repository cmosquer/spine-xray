import gc
import math
import wandb
import torch
import hydra
from ._utils import set_run_name
from .models import get_model, set_optimizer, set_loss_function
from .dataset import get_spine_datasets, get_test_spine_dataset
from .training import train, train_heatmap, val, val_heatmap, model_selection_metrics, EarlyStopping
from .logger import log_model, wandb_prediction_table, wandb_test_heatmaps_table, wandb_heatmaps_table, log_wandb_table, log_model, WandbSingleRun


def model_pipeline_single(wandb_class: WandbSingleRun):
    
    cfg = (hydra.compose(overrides=["+landmarks=default"])).landmarks
    
    gc.collect()
    torch.cuda.empty_cache()

    # This config will be set by Sweep Controller
    config = wandb.config

    model = get_model(config.model_architecture)

    # Wandb start to register all data associated to the model
    wandb.watch(model, log="all")

    # Obtain train, validation, and testing DataLayer
    train_loader, val_loader = get_spine_datasets(config, type_normalization=cfg.type_normalization)
    test_loader = get_test_spine_dataset(config, type_normalization=cfg.type_normalization)

    # Make the loss and optimizer
    loss_func = set_loss_function(config.loss_func)
    optimizer = set_optimizer(model, config.optimizer, config)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 
                                                            factor=0.1, 
                                                            verbose=True,
                                                            patience=config.lr_patience)

    early_stopping = EarlyStopping(patience=config.patience, verbose=True)

    table = []

    if wandb_class.pipeline == 'DCR':
        for epoch in range(config.epochs):
            metrics_avg = train(model, train_loader, loss_func, optimizer, epoch)
            val_metrics_avg = val(model, val_loader, loss_func, epoch)

            wandb_prediction_table(table, model, test_loader, loss_func, epoch)

            scheduler.step(val_metrics_avg['val/val_avg_loss'])
            early_stopping(val_metrics_avg['val/val_avg_loss'], model)

            lr = {'epoch/learning_rate': optimizer.param_groups[0]['lr']}

            wandb.log({**metrics_avg, **val_metrics_avg, **lr})

            print(f"Train Avg Loss: {metrics_avg['train/train_avg_loss']:.3f}, "
                    f"Train Median Loss: {metrics_avg['train/train_median_loss']:.3f}, "
                    f"Valid Avg Loss: {val_metrics_avg['val/val_avg_loss']:3f}, "
                    f"Valid Median Loss: {val_metrics_avg['val/val_median_loss']:.3f}, "
                    f"Learning rate: {lr['epoch/learning_rate']:8f}, "
                    f"Epoch: {epoch + 1}")
            
            if early_stopping.early_stop:
                print("Early stopping")
                break
        
    elif wandb_class.pipeline == 'HBR':
        heatmap_table = []

        # Start training
        for epoch in range(config.epochs):
            metrics_avg = train_heatmap(model, train_loader, loss_func, optimizer, epoch, config.sigma)

            val_metrics_avg = val_heatmap(model, val_loader, loss_func, epoch, config.sigma)

            wandb_heatmaps_table(table=table, heatmap_table=heatmap_table, model=model,
                                loader=test_loader, loss_func=loss_func, epoch=epoch, num_example=2,
                                normalization=2, sigma=config.sigma)

            scheduler.step(val_metrics_avg['val/val_avg_loss'])
            early_stopping(val_metrics_avg['val/val_avg_loss'], model)

            lr = {'epoch/learning_rate': optimizer.param_groups[0]['lr']}

            wandb.log({**metrics_avg, **val_metrics_avg, **lr})

            print(f"Train Avg Loss: {metrics_avg['train/train_avg_loss']:.3f}, "
                    f"Train Median Loss: {metrics_avg['train/train_median_loss']:.3f}, "
                    f"Valid Avg Loss: {val_metrics_avg['val/val_avg_loss']:3f}, "
                    f"Valid Median Loss: {val_metrics_avg['val/val_median_loss']:.3f}, "
                    f"Learning rate: {lr['epoch/learning_rate']:8f}, "
                    f"Epoch: {epoch + 1}")

            if early_stopping.early_stop:
                print("Early stopping")
                break
            
        log_wandb_table(heatmap_table, "heatmaps_table")
        del heatmap_table

    log_wandb_table(table)
    log_model(config.model_architecture['model'],'./models/checkpoint.pt')

    del model
    del table
    
    
def model_pipeline_sweep_DCR():
    
    cfg = (hydra.compose(overrides=["+landmarks=default"])).landmarks
    
    gc.collect()
    torch.cuda.empty_cache()
    
    with wandb.init(name=set_run_name(), tags=['Sweep', 'DCR']):

        config = wandb.config
        # This config will be set by Sweep Controller

        model = get_model(config.model_architecture)

        # Wandb start to register all data associated to the model
        wandb.watch(model, log="all")

        dataset_table = []
                
        # Obtain train, validation, and testing DataLayer
        train_loader, val_loader = get_spine_datasets(config_hyp=config, 
                                                      type_normalization=cfg.type_normalization, 
                                                      wandb_table=dataset_table, 
                                                      seed=None, custom_split=config.dataset['random'])
        
        test_loader = get_test_spine_dataset(config, type_normalization=cfg.type_normalization, wandb_table=dataset_table)
        
        log_wandb_table(dataset_table, "Dataset/Table Dataset")

        # Make the loss and optimizer
        loss_func = set_loss_function(config.loss_func)
        optimizer = set_optimizer(model, config.optimizer, config)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, verbose=True, patience=config.lr_patience)
        early_stopping = EarlyStopping(patience=config.patience, verbose=True, delta = 0.001)

        table = []
        
        try:
            # Start training
            for epoch in range(config.epochs):
                metrics_avg = train(model, train_loader, loss_func, optimizer, epoch, cfg.type_normalization, cfg.num_keypoints)

                val_metrics_avg = val(model, val_loader, loss_func, epoch, cfg.type_normalization, cfg.num_keypoints)

                scheduler.step(val_metrics_avg['val/val_avg_loss'])
                early_stopping(val_metrics_avg['val/val_avg_loss'], model)

                lr = {'epoch/learning_rate': optimizer.param_groups[0]['lr']}

                wandb.log({**metrics_avg, **val_metrics_avg, **lr})

                print(f"Train Avg Loss: {metrics_avg['train/train_avg_loss']:.3f}, "
                        f"Train Median Loss: {metrics_avg['train/train_median_loss']:.3f}, "
                        f"Valid Avg Loss: {val_metrics_avg['val/val_avg_loss']:3f}, "
                        f"Valid Median Loss: {val_metrics_avg['val/val_median_loss']:.3f}, "
                        f"Learning rate: {lr['epoch/learning_rate']:8f}, "
                        f"Epoch: {epoch + 1}")
                
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
                
        except RuntimeError as e:
            print(e)
            del model
            gc.collect()
            torch.cuda.empty_cache()
            return
        
        log_model(config.model_architecture, './models/checkpoint.pt')
        
        wandb_prediction_table(table, model, test_loader, loss_func, epoch)
        
        log_wandb_table(table)
        
    del model
    del table
    
    gc.collect()
    torch.cuda.empty_cache()


def model_pipeline_sweep_HBR():
    
    cfg = (hydra.compose(overrides=["+landmarks=default"])).landmarks
    
    gc.collect()
    torch.cuda.empty_cache()
    
    with wandb.init(name=set_run_name(), tags=['Sweep', 'HBR']):

        # This config will be set by Sweep Controller
        config = wandb.config

        model = get_model(config.model_architecture)

        # Wandb start to register all data associated to the model
        wandb.watch(model, log="all")

        dataset_table = []
                
        # Obtain train, validation, and testing DataLayer
        train_loader, val_loader = get_spine_datasets(config_hyp=config, 
                                                      type_normalization=cfg.type_normalization, 
                                                      wandb_table=dataset_table, 
                                                      seed=None, 
                                                      custom_split=config.dataset['random'])
        
        test_loader = get_test_spine_dataset(config, wandb_table=dataset_table)
        
        log_wandb_table(dataset_table, "Dataset/Table Dataset")
        
        # Make the loss and optimizer
        loss_func = set_loss_function(config.loss_func)
        
        optimizer = set_optimizer(model, config.optimizer, config)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, verbose=True, patience=config.lr_patience)

        early_stopping = EarlyStopping(patience=config.patience, verbose=True, delta = 0.00001)

        table = []
        heatmap_table = []
        val_pck_table = []
        
        # Start training
        try:
            for epoch in range(config.epochs):

                metrics_avg = train_heatmap(model, train_loader, loss_func, optimizer, epoch, config.sigma, cfg.type_normalization)
                val_metrics_avg, avg_pck = val_heatmap(model, val_loader, loss_func, epoch, config.sigma, cfg.type_normalization, config.delta)

                val_pck_table.append(avg_pck)
                
                scheduler.step(val_metrics_avg['val/val_avg_loss'])
                
                if math.isnan(val_metrics_avg['val/val_avg_loss']):
                    raise ValueError("Nan values")
                
                early_stopping(val_metrics_avg['val/val_avg_loss'], model)
                
                lr = {'epoch/learning_rate': optimizer.param_groups[0]['lr']}

                wandb.log({**metrics_avg, **val_metrics_avg, **lr})

                print(f"Train Avg Loss: {metrics_avg['train/train_avg_loss']:.3f}, "
                        f"Train Median Loss: {metrics_avg['train/train_median_loss']:.3f}, "
                        f"Valid Avg Loss: {val_metrics_avg['val/val_avg_loss']:3f}, "
                        f"Valid Median Loss: {val_metrics_avg['val/val_median_loss']:.3f}, "
                        f"Learning rate: {lr['epoch/learning_rate']:8f}, "
                        f"Epoch: {epoch + 1}")

                if early_stopping.early_stop:
                    print("Early stopping")
                    break
                
        except RuntimeError as e:
            print(e)
            del model
            gc.collect()
            torch.cuda.empty_cache()
            return
        except ValueError as v:
            print(v)
            del model
            gc.collect()
            torch.cuda.empty_cache()
            return
        
        log_model(config.model_architecture, './models/checkpoint.pt')
        
        all_pck_values = []
        
        for epoch_pck in val_pck_table:
            for values in epoch_pck:
                all_pck_values.append(values)
        
        log_wandb_table(all_pck_values, "val/PCK table")
        
        metrics_model_selection, localization_error_table, euclidean_error_table, median_per_landmark = model_selection_metrics(model, val_loader, cfg.type_normalization)

        log_wandb_table(median_per_landmark, "Model_selection/Median of Neighbourhood error per landmarks table")

        log_wandb_table(localization_error_table, "Model_selection/Neighbourhood error table")
        
        log_wandb_table(euclidean_error_table, "Model_selection/Euclidean error table")
        
        wandb.log(metrics_model_selection)
        
        wandb_test_heatmaps_table(table=table, heatmap_table=heatmap_table, model=model, loader=test_loader, 
                                  epoch=epoch, num_example=cfg.num_examples_log, normalization=cfg.type_normalization)
            
        log_wandb_table(heatmap_table, "Heatmaps/Heatmaps Table")
        
        log_wandb_table(table, "Predictions/Predictions Table")
        
        del model
        del heatmap_table
        del table
    
        gc.collect()
        torch.cuda.empty_cache()

