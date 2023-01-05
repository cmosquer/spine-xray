import math
import torch
import wandb
import operator
import itertools
import statistics
import numpy as np
from tqdm import tqdm
from .._utils import euclidean_losses, chunks
from kornia.losses import js_div_loss_2d
from kornia.geometry.subpix.dsnt import render_gaussian2d

device = 'cuda' if torch.cuda.is_available() else 'cpu'

keypoints_labels = ["C2OT", "C1AE", "C1PE", "C2CE", "C2AI", "C2PI", "C7AS", "C7PS", "C7CE",
                    "C7AI", "C7PI", "T1AS", "T1PS", "T1CE", "T1AI", "T1PI", "T5AS", "T5PS", "T12AI", "T12PI",
                    "L1AS", "L1PS", "L4AS", "L4PS", "L4AI", "L4PI", "S1AS", "S1MI", "S1PS", "F1HC", "F2HC"]


def train(model, train_loader, loss_func, optimizer, epoch, normalization, number_kpts):
    cant_images, acm_loss, acm_mse  = 0, 0, 0
    losses, mse_values = [], []
    n_steps_per_epoch = math.ceil(len(train_loader.dataset) / train_loader.batch_size)

    model.train()
    
    with tqdm(train_loader, unit="batch") as tepoch:
        for step, (images, labels) in enumerate(tepoch):
            
            tepoch.set_description(f"Epoch {epoch}")
            
            images, targets = images.to(device), labels.to(device)
            images, targets = images.type(torch.cuda.FloatTensor), targets.type(torch.cuda.FloatTensor)

            if normalization in [0, 1]:
                boolean_index = targets[:,:,0] != 0
            if normalization == 2:
                boolean_index = targets[:,:,0] != -1
            
            # forward propagation
            landmark_pred = model(images)
            landmark_pred = torch.reshape(landmark_pred, (landmark_pred.shape[0], number_kpts, 2))

            train_loss = loss_func(landmark_pred[np.where(boolean_index.cpu())], targets[np.where(boolean_index.cpu())])
            losses.append(train_loss.item())
            acm_loss += train_loss.item()
            
            mse = torch.nn.functional.mse_loss(landmark_pred[np.where(boolean_index.cpu())], targets[np.where(boolean_index.cpu())])
            mse_values.append(mse.item())
            acm_mse += mse.item()
            
            # backward propagation
            optimizer.zero_grad()
            train_loss.backward()

            optimizer.step()

            cant_images += len(images)
            
            metrics = {"train/train_step_loss": train_loss,
                        "train/mse_step": mse,
                        "train/epoch": (step + 1 + (n_steps_per_epoch * epoch)) / n_steps_per_epoch,
                        "train/cant_images": cant_images}
            
            tepoch.set_postfix(loss=losses[-1], mse=mse_values[-1])
            
            if step + 1 <= n_steps_per_epoch:
                wandb.log(metrics)

    metrics_avg = {
        "train/train_avg_loss": acm_loss / n_steps_per_epoch,
        "train/mse_avg": acm_mse / n_steps_per_epoch,
        "train/train_median_loss": statistics.median(losses),
        "train/mse_median": statistics.median(mse_values),
        "train/epoch": epoch + 1,
    }

    return metrics_avg


def val(model, val_loader, loss_func, epoch, normalization, number_kpts):

    val_loss, acm_loss, acm_mse = 0., 0., 0.
    losses, mse_values = [], []
    n_steps_per_epoch = math.ceil(len(val_loader.dataset) / val_loader.batch_size)
   
    model.eval()

    with torch.inference_mode():
        for batch_idx, (images, targets) in enumerate(val_loader):
            images, targets = images.to(device), targets.to(device)
            images, targets = images.type(torch.cuda.FloatTensor), targets.type(torch.cuda.FloatTensor)

            if normalization in [0,1]:
                boolean_index = targets[:,:,0] != 0
            if normalization == 2:
                boolean_index = targets[:,:,0] != -1
            
            # forward propagation
            landmark_pred = model(images)
            landmark_pred = torch.reshape(landmark_pred, (landmark_pred.shape[0], number_kpts, 2))

            val_loss = loss_func(landmark_pred[np.where(boolean_index.cpu())], targets[np.where(boolean_index.cpu())])
            losses.append(val_loss.item())
            acm_loss += val_loss.item()
            
            mse = torch.nn.functional.mse_loss(landmark_pred[np.where(boolean_index.cpu())], targets[np.where(boolean_index.cpu())])
            mse_values.append(mse.item())
            acm_mse += mse.item()

    val_metrics = {"val/val_avg_loss": acm_loss / n_steps_per_epoch,
                   "val/mse_avg": acm_mse / n_steps_per_epoch,
                   "val/val_median_loss": statistics.median(losses),
                   "val/mse_median": statistics.median(mse_values),
                   "val/epoch": epoch + 1}
    
    return val_metrics


def train_heatmap(model, train_loader, loss_func, optimizer, epoch, sigma, normalization):
    cant_images, acm_loss, acm_mse = 0, 0, 0
    losses, mse_values = [], []
    n_steps_per_epoch = math.ceil(len(train_loader.dataset) / train_loader.batch_size)

    model.train()
    
    with tqdm(train_loader, unit="batch") as tepoch:
        for step, (images, targets) in enumerate(tepoch):
            
            tepoch.set_description(f"Epoch {epoch}")
            
            images, targets = images.to(device), targets.to(device)
            images, targets = images.type(torch.cuda.FloatTensor), targets.type(torch.cuda.FloatTensor)

            # forward propagation
            heatmap_pred, landmark_pred = model(images)

            if normalization in [0, 1]:
                boolean_index = targets[:,:,0] != 0
            if normalization == 2:
                boolean_index = targets[:,:,0] != -1
            
            heatmap_gt = render_gaussian2d(targets[np.where(boolean_index.cpu())],
                                        torch.tensor([sigma, sigma], 
                                                        dtype=torch.float32, 
                                                        device=device),
                                        heatmap_pred.shape[2:], 
                                        normalized_coordinates=True)

            euclidean_loss = euclidean_losses(targets[np.where(boolean_index.cpu())], landmark_pred[np.where(boolean_index.cpu())])

            js_loss = js_div_loss_2d(heatmap_gt[None, :, :, :], heatmap_pred[np.where(boolean_index.cpu())][None, :, :, :], reduction='none')

            train_loss = loss_func(euclidean_loss + js_loss[0])
            
            losses.append(train_loss.item())

            acm_loss += train_loss.item()

            mse = torch.nn.functional.mse_loss(landmark_pred[np.where(boolean_index.cpu())], targets[np.where(boolean_index.cpu())])

            mse_values.append(mse.item())

            acm_mse += mse

            # backward propagation
            optimizer.zero_grad()
            train_loss.backward()

            optimizer.step()

            cant_images += len(images)

            metrics = {"train/train_step_loss": train_loss,
                        "train/mse_step": mse,
                        "train/epoch": (step + 1 + (n_steps_per_epoch * epoch)) / n_steps_per_epoch,
                        "train/cant_images": cant_images}
            
            tepoch.set_postfix(loss=losses[-1], mse=mse_values[-1])
            
            if step + 1 <= n_steps_per_epoch:
                wandb.log(metrics)

    metrics_avg = {
        "train/train_avg_loss": acm_loss / n_steps_per_epoch,
        "train/mse_avg": acm_mse / n_steps_per_epoch,
        "train/train_median_loss": statistics.median(losses),
        "train/mse_median": statistics.median(mse_values),
        "train/epoch": epoch + 1,
    }

    return metrics_avg


def val_heatmap(model, val_loader, loss_func, epoch, sigma, normalization, delta):

    val_loss, acm_mse = 0., 0. 
    mse_values, losses = [], []
    n_steps_per_epoch = math.ceil(len(val_loader.dataset) / val_loader.batch_size)

    pck_values = dict((label,[]) for label in keypoints_labels)
    
    model.eval()

    with torch.inference_mode():
        for batch_idx, (images, targets) in enumerate(val_loader):
            
            images, targets = images.to(device), targets.to(device)
            images, targets = images.type(torch.cuda.FloatTensor), targets.type(torch.cuda.FloatTensor)

            # forward propagation
            heatmap_pred, landmark_pred = model(images)

            array_labels = np.array(list(itertools.chain.from_iterable(itertools.repeat(keypoints_labels, images.shape[0])))).reshape((images.shape[0],len(keypoints_labels)))

            if normalization == 0:
                boolean_index = targets[:,:,0] != 0
            if normalization == 2:
                boolean_index = targets[:,:,0] != -1
                
            kpts_labels_selected = array_labels[np.where(boolean_index.cpu())]
            
            heatmap_gt = render_gaussian2d(targets[np.where(boolean_index.cpu())],
                                        torch.tensor([sigma, sigma], 
                                                        dtype=torch.float32, 
                                                        device=device),
                                        heatmap_pred.shape[2:], 
                                        normalized_coordinates=True)

            euclidean_loss = euclidean_losses(targets[np.where(boolean_index.cpu())], landmark_pred[np.where(boolean_index.cpu())])
        
            mse = torch.nn.functional.mse_loss(landmark_pred[np.where(boolean_index.cpu())], targets[np.where(boolean_index.cpu())])
            
            mse_values.append(mse.item())
            acm_mse += mse.item()

            js_loss = js_div_loss_2d(heatmap_gt[None, :, :, :], heatmap_pred[np.where(boolean_index.cpu())][None, :, :, :], reduction='none')

            loss = loss_func(euclidean_loss + js_loss[0])
            
            losses.append(loss.item())
            
            compute_pck(pck_values, euclidean_loss, kpts_labels_selected, delta)
            
            val_loss += loss.item()

    val_metrics = {"val/val_avg_loss": val_loss / n_steps_per_epoch,
                   "val/mse_avg": acm_mse / n_steps_per_epoch,
                   "val/val_median_loss": statistics.median(losses),
                   "val/mse_median": statistics.median(mse_values),
                   "val/epoch": epoch + 1}

    return val_metrics, compute_mean_pck(pck_values, epoch + 1)


def compute_pck(pck_values, euclidean_distances, selected_labels, delta):

    for distance, label in zip(euclidean_distances, selected_labels):
        pck_values[label].append(1 if distance < delta else 0)
    
    return pck_values


def compute_mean_pck(pck_values, epoch):
    
    pck_values = {k: [statistics.mean(values) * 100] if len(values)>0 else [0] for k, values in pck_values.items()}
    
    pck_values = [{'Landmark':list(item.keys())[0], 'PCK': list(item.values())[0], 'epoch': epoch} for item in chunks(pck_values, 1)] 
    
    return pck_values
    

def model_selection_metrics(model, val_loader, normalization):

    model.eval()

    localization_error = dict((label,[]) for label in keypoints_labels)
    
    euclidean_error = dict((label,[]) for label in keypoints_labels)
    
    with torch.inference_mode():
        for batch_idx, (images, targets) in enumerate(val_loader):
            
            images, targets = images.to(device), targets.to(device)
            images, targets = images.type(torch.cuda.FloatTensor), targets.type(torch.cuda.FloatTensor)

            array_labels = np.array(list(itertools.chain.from_iterable(itertools.repeat(keypoints_labels, images.shape[0])))).reshape((images.shape[0],len(keypoints_labels)))
            
            # forward propagation
            _ , landmark_pred = model(images)

            if normalization in [0, 1]:
                boolean_index = targets[:,:,0] != 0
            if normalization == 2:
                boolean_index = targets[:,:,0] != -1

            kpts_labels_selected = array_labels[np.where(boolean_index.cpu())]
            
            euclidean_loss = euclidean_losses(targets[np.where(boolean_index.cpu())], landmark_pred[np.where(boolean_index.cpu())])
            
            weighted_loss = weighted_localization_error(euclidean_loss, targets, boolean_index)    
            
            for label, loss in zip(kpts_labels_selected, weighted_loss):
                localization_error[label].append(loss)
        
            for label, eu_loss in zip(kpts_labels_selected, euclidean_loss):
                euclidean_error[label].append(eu_loss.item())
            
        median_values = {k: statistics.median(values) if len(values)>0 else [0] for k, values in localization_error.items()}
        
        metrics = {
            "Model_selection/Maximum median Neighbourhood Error of landmarks": max(median_values.items(), key=operator.itemgetter(1))[1],
            "Model_selection/Landmark with maximum median Neighbourhood error": max(median_values.items(), key=operator.itemgetter(1))[0],
        }

        median_values = [{'Landmark':list(item.keys())[0], 'Median Value': list(item.values())[0]} for item in chunks(median_values, 1)] 

    return metrics, [localization_error], [euclidean_error], median_values
        
        
def compute_min_localization_error(idx_kpt, targets):
    distance_error = []
    for idx, coordinates in enumerate(targets):
        if idx != idx_kpt:
            distance_error.append((euclidean_losses(targets[idx_kpt], coordinates)).detach().cpu().numpy())
    return min(np.array(distance_error))


def weighted_localization_error(euclidean_loss, targets, mask):
    weighted_denominator = []

    for idx_batch, kpts_mask in enumerate(mask):
        for idx_kpt, bool_kpt in enumerate(kpts_mask):
            if(bool_kpt):
                weighted_denominator.append(compute_min_localization_error(idx_kpt, targets[idx_batch]))
    
    weighted_denominator = np.array(weighted_denominator)
    
    return euclidean_loss.detach().cpu().numpy()/weighted_denominator
    
    
