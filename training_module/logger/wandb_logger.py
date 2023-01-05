import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
from kornia.geometry.subpix.dsnt import render_gaussian2d
from kornia.losses import js_div_loss_2d

import wandb

from .._utils import euclidean_losses
from ..helpers import getBoundingBox, normalized_to_pixel_coordinates

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def wandb_bbox(image, keypoints, name_bbox, normalization=1):
    """ Receives an image with their keypoints associated, and returns a wandb Image object
    with the bounding box integrated to the image. The bounding box is created using the keypoints

    Args:
        image (_type_): image in which the bounding box is going to be integrated
        keypoints (_type_): keypoints associated to the image
        nameBbox (_type_): name of the bounding box created
        normalization (int, optional): Determines the type of normalization apply to keypoints. Defaults to 1. If set to 1 [0,1] or [-1,1] if set 2 as a parameter

    Returns:
        wandb.Image : wandb Image object with the bounding box integrated to the image.
    """

    # TODO: Normalize each coordinate to their corresponded value using the size associated to the image (height or width)

    if normalization == 1:
        keypoints = (keypoints * image.shape[0]) + image.shape[0]
    elif normalization == 2:
        keypoints = normalized_to_pixel_coordinates(keypoints, image.shape[:2])

    x_min, y_min, x_max, y_max = getBoundingBox(keypoints, expand_px=0)

    all_boxes = []

    box_data = {
        "position": {
            "minX": x_min,
            "minY": y_min,
            "maxX": x_max,
            "maxY": y_max
        },
        "class_id": 1,
        "box_caption": 'person',
        "domain": "pixel"
    }

    all_boxes.append(box_data)

    return wandb.Image(image, boxes={
        name_bbox: {
            "box_data": all_boxes,
            "class_labels": {1: "person"},
        }
    })


def wandb_landmarks(image, keypoints, normalization: int = 1):
    """ Creates an image with keypoints integrated on it. Reference: https://github.com/wandb/gitbook/blob/master/library/log.md

    Args:
        image (_type_): image in which the keypoints are going to be plotted
        keypoints (_type_): keypoints associated to the image
         normalization (int, optional): Determines the type of normalization apply to keypoints. Defaults to 1. If set to 1 [0,1] or [-1,1] if set 2 as a parameter

    Returns:
        wandb.Image: wandb Image object with the bounding box integrated to the image.
    """

    # TODO: Normalize each coordinate to their corresponded value using the size associated to the image (height or
    #  width)

    if normalization == 1:
        keypoints = (keypoints * image.shape[0]) + image.shape[0]
    elif normalization == 2:
        keypoints = normalized_to_pixel_coordinates(keypoints, image.shape[:2])

    plt.gcf().canvas.get_renderer()
    fig, ax = plt.subplots()
    ax.imshow(image, 'gray')
    ax.scatter(keypoints[:, 0], keypoints[:, 1])
    wandb_image = wandb.Image(fig)
    plt.close()
    return wandb_image


def wandb_landmarks_plotly(image, gt_keypoints, predicted_kpts, normalization=1):
    """ Creates a plotly object with ground truth and predicted keypoints plot onto the image. Reference: https://github.com/wandb/gitbook/blob/master/library/log.md

    Args:
        image (_type_): image in which the keypoints are going to be plotted
        keypoints (_type_): keypoints associated to the image
        normalization (int, optional): Determines the type of normalization apply to keypoints. Defaults to 1. If set to 1 [0,1] or [-1,1] if set 2 as a parameter

    Returns:
        wandb.Html: wandb HTML object with all parameters plot on a plotly figure 
    """

    # TODO: Normalize each coordinate to their corresponded value using the size associated to the image (height or width)

    if normalization == 1:
        gt_keypoints = (gt_keypoints * image.shape[0]) + image.shape[0]
        predicted_kpts = (predicted_kpts * image.shape[0]) + image.shape[0]
        boolean_index = gt_keypoints[:, 0] != 0
    elif normalization == 2:
        gt_keypoints = normalized_to_pixel_coordinates(
            gt_keypoints, (image.shape[1], image.shape[0]))
        predicted_kpts = normalized_to_pixel_coordinates(
            predicted_kpts, (image.shape[1], image.shape[0]))
        boolean_index = gt_keypoints[:, 0] != 0

    plt.gcf().canvas.get_renderer()

    fig = px.imshow(image, color_continuous_scale='gray', labels={})
    fig.update_traces(name='Image', showlegend=False,
                      selector=dict(name='coloraxis'))

    keypoints_labels = ["C2OT", "C1AE", "C1PE", "C2CE", "C2AI", "C2PI", "C7AS", "C7PS", "C7CE",
                        "C7AI", "C7PI", "T1AS", "T1PS", "T1CE", "T1AI", "T1PI", "T5AS", "T5PS", "T12AI", "T12PI",
                        "L1AS", "L1PS", "L4AS", "L4PS", "L4AI", "L4PI", "S1AS", "S1MI", "S1PS", "F1HC", "F2HC"]

    fig.add_trace(go.Scatter(x=gt_keypoints[boolean_index, 0],
                             y=gt_keypoints[boolean_index, 1],
                             text=list(np.array(keypoints_labels)
                                       [boolean_index]),
                             mode="markers",
                             name='Ground Truth'),)

    fig.add_trace(go.Scatter(x=predicted_kpts[:, 0],
                             y=predicted_kpts[:, 1],
                             text=keypoints_labels,
                             mode="markers",
                             name='Predicted Landmarks'),)

    fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
    fig.update_traces(showlegend=True, selector=dict(type='scatter'))
    fig.update_coloraxes(showscale=False)

    fig.update_layout(
        autosize=False,
        width=image.shape[1],
        height=image.shape[0],
        margin=dict(l=5, r=5, b=0, t=25),
        # paper_bgcolor="white",
    )

    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=0.92,
        xanchor="right",
        x=1
    ))

    path_to_plotly_html = "./temp/plotly_figure.html"

    # Write Plotly figure to HTML
    # Setting auto_play to False prevents animated Plotly charts from playing in the table automatically
    fig.write_html(path_to_plotly_html, auto_play=False)

    # Add Plotly figure as HTML file into Table
    return wandb.Html(path_to_plotly_html)


def wandb_landmarks_plotly(image, keypoints, normalization: int = 1):
    """ Creates a plotly object with ground truth and predicted keypoints plot onto the image. Reference: https://github.com/wandb/gitbook/blob/master/library/log.md

    Args:
        image (_type_): image in which the keypoints are going to be plotted
        keypoints (_type_): keypoints associated to the image
        normalization (int, optional): Determines the type of normalization apply to keypoints. Defaults to 1. If set to 1 [0,1] or [-1,1] if set 2 as a parameter

    Returns:
        wandb.Html: wandb HTML object with all parameters plot on a plotly figure 
    """

    # TODO: Normalize each coordinate to their corresponded value using the size associated to the image (height or width)

    if normalization == 1:
        keypoints = (keypoints * image.shape[0]) + image.shape[0]
    elif normalization == 2:
        keypoints = normalized_to_pixel_coordinates(
            keypoints, (image.shape[1], image.shape[0]))

    plt.gcf().canvas.get_renderer()

    fig = px.imshow(image, color_continuous_scale='gray', labels={})
    fig.update_traces(name='Image', showlegend=False,
                      selector=dict(name='coloraxis'))

    keypoints_labels = ["C2OT", "C1AE", "C1PE", "C2CE", "C2AI", "C2PI", "C7AS", "C7PS", "C7CE",
                        "C7AI", "C7PI", "T1AS", "T1PS", "T1CE", "T1AI", "T1PI", "T5AS", "T5PS", "T12AI", "T12PI",
                        "L1AS", "L1PS", "L4AS", "L4PS", "L4AI", "L4PI", "S1AS", "S1MI", "S1PS", "F1HC", "F2HC"]

    fig.add_trace(go.Scatter(x=keypoints[:, 0],
                             y=keypoints[:, 1],
                             text=list(np.array(keypoints_labels)),
                             mode="markers",
                             name='Predicted Landmarks'),)

    fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
    fig.update_traces(showlegend=True, selector=dict(type='scatter'))
    fig.update_coloraxes(showscale=False)

    fig.update_layout(
        autosize=False,
        width=image.shape[1],
        height=image.shape[0],
        margin=dict(l=5, r=5, b=0, t=25),
        # paper_bgcolor="white",
    )

    fig.update_layout(legend=dict(
        orientation="h", yanchor="bottom", y=0.92, xanchor="right", x=1))

    path_to_plotly_html = "./temp/plotly_figure.html"

    # Write Plotly figure to HTML
    # Setting auto_play to False prevents animated Plotly charts from playing in the table automatically
    fig.write_html(path_to_plotly_html, auto_play=False)

    # Add Plotly figure as HTML file into Table
    return wandb.Html(path_to_plotly_html)


def wandb_heatmaps_plot(heatmaps, grouped: bool = True, mask=None):
    """_summary_

    Args:
        heatmaps (_type_): _description_
        grouped (bool, optional): _description_. Defaults to True.
        mask (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    if grouped:
        fig = plt.figure(figsize=(10, 3))
        plt.subplots_adjust(hspace=0.01)

        ncols = 8
        nrows = 4

        for n, heatmap in enumerate(heatmaps):
            subplot = plt.subplot(nrows, ncols, n + 1)
            subplot.set_xlabel("")
            subplot.set_xticklabels([])
            subplot.set_yticklabels([])
            subplot.tick_params(left=False, bottom=False)
            plt.imshow(heatmap.detach().cpu())

        wandb_image = wandb.Image(fig)
        plt.close()
    else:
        heatmaps_list = []
        keypoints_labels = ["C2OT", "C1AE", "C1PE", "C2CE", "C2AI", "C2PI", "C7AS", "C7PS", "C7CE",
                            "C7AI", "C7PI", "T1AS", "T1PS", "T1CE", "T1AI", "T1PI", "T5AS", "T5PS", "T12AI", "T12PI",
                            "L1AS", "L1PS", "L4AS", "L4PS", "L4AI", "L4PI", "S1AS", "S1MI", "S1PS", "F1HC", "F2HC"]

        if mask is not None:
            keypoints_labels = list(np.array(keypoints_labels)[mask.cpu()])

        for heatmap, label in zip(heatmaps, keypoints_labels):
            plt.imshow(heatmap.detach().cpu())
            plt.title(label)
            plt.axis('off')
            plt.tick_params(left=False, bottom=False)
            heatmaps_list.append(wandb.Image(plt))
        wandb_image = heatmaps_list
    return wandb_image


def wandb_prediction(image, targets, outputs, loss, epoch, normalization=1):
    """_summary_

    Args:
        image (_type_): _description_
        targets (_type_): _description_
        outputs (_type_): _description_
        loss (_type_): _description_
        epoch (_type_): _description_
        normalization (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    prediction_info = {

        'Landmark ground truth': wandb_landmarks(
            image[0].permute(1, 2, 0).detach().cpu().numpy(),
            targets[0].detach().cpu().numpy(),
            normalization),

        'Landmark prediction': wandb_landmarks(
            image[0].permute(1, 2, 0).detach().cpu().numpy(),
            outputs[0].detach().cpu().numpy(),
            normalization),

        'Bbox ground truth': wandb_bbox(image[0].permute(1, 2, 0).detach().cpu().numpy(),
                                        targets[0].detach().cpu().numpy(),
                                        'ground truth',
                                        normalization),

        'Bbox prediction': wandb_bbox(image[0].permute(1, 2, 0).detach().cpu().numpy(),
                                      outputs[0].detach().cpu().numpy(),
                                      'prediction',
                                      normalization),
        'Loss': loss,

        'Epoch': epoch + 1
    }

    return prediction_info


def wandb_prediction_pyplot(image, targets, outputs, loss, epoch, normalization=2):
    """_summary_

    Args:
        image (_type_): _description_
        targets (_type_): _description_
        outputs (_type_): _description_
        loss (_type_): _description_
        epoch (_type_): _description_
        normalization (int, optional): _description_. Defaults to 2.

    Returns:
        _type_: _description_
    """
    prediction_info = {

        'Epoch': epoch + 1,

        'Spine Rx': wandb_landmarks_plotly(
            image=image[0].detach().cpu().numpy(),
            gt_keypoints=targets.detach().cpu().numpy(),
            predicted_kpts=outputs.detach().cpu().numpy(),
            normalization=normalization),

        'Euclidean Loss per landmark': loss['euclidean_loss'],

        'Avg Euclidean Loss': torch.mean(loss['euclidean_loss']),

        'MSE': loss['mse'],

        'Loss': loss['loss'],

    }

    return prediction_info


def wandb_prediction_heatmap(image, targets, outputs, loss, epoch, normalization=2):
    """_summary_

    Args:
        image (_type_): _description_
        targets (_type_): _description_
        outputs (_type_): _description_
        loss (_type_): _description_
        epoch (_type_): _description_
        normalization (int, optional): _description_. Defaults to 2.

    Returns:
        _type_: _description_
    """
    prediction_info = {
        'Epoch': epoch + 1,

        'Spine Rx': wandb_landmarks_plotly(
            image=image[0].detach().cpu().numpy(),
            gt_keypoints=targets.detach().cpu().numpy(),
            predicted_kpts=outputs.detach().cpu().numpy(),
            normalization=normalization),

        'Euclidean Loss per landmark': loss['euclidean_loss'],

        'Avg Euclidean Loss': torch.mean(loss['euclidean_loss']),

        'Jensen Shannon per landmark': loss['js_loss'],

        'Avg Jensen Shannon Loss': torch.mean(loss['js_loss']),

        'MSE': loss['mse'],

        'Loss': loss['loss'],

    }

    return prediction_info


def wandb_prediction_heatmap(image, keypoints, epoch, normalization=2):

    prediction_info = {
        'Epoch': epoch + 1,

        'Spine Rx': wandb_landmarks_plotly(
            image=image[0].detach().cpu().numpy(),
            keypoints=keypoints.detach().cpu().numpy(),
            normalization=normalization
            ),
    }

    return prediction_info


def wandb_heatmaps(heatmap_target, heatmap_pred, loss, epoch, mask=None):
    """_summary_

    Args:
        heatmap_target (_type_): _description_
        heatmap_pred (_type_): _description_
        loss (_type_): _description_
        epoch (_type_): _description_
        mask (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    heatmaps_info = {

        'Heatmap ground truth': wandb_heatmaps_plot(heatmap_target, grouped=False, mask=mask),

        'Heatmap model output': wandb_heatmaps_plot(heatmap_pred, grouped=False),

        'Euclidean Loss per landmark': loss['euclidean_loss'],

        'Avg Euclidean Loss': torch.mean(loss['euclidean_loss']),

        'Jensen Shannon per landmark': loss['js_loss'],

        'Avg Jensen Shannon Loss': torch.mean(loss['js_loss']),

        'MSE': loss['mse'],

        'Loss': loss['loss'],

        'Epoch': epoch + 1
    }

    return heatmaps_info


def wandb_prediction_table(table, model, loader, loss_func, epoch: int, num_example: int = 2, normalization: int = 2):
    """_summary_

    Args:
        table (_type_): _description_
        model (_type_): _description_
        loader (_type_): _description_
        loss_func (_type_): _description_
        epoch (int): _description_
        num_example (int, optional): _description_. Defaults to 2.
        normalization (bool, optional): _description_. Defaults to True.
    """
    # This function receives a table and add a specific number of examples as wandb objects;

    model.eval()

    with torch.inference_mode():
        for idx_batch, (images, targets) in enumerate(loader):

            if idx_batch + 1 == num_example:
                break

            images, targets = images.to(device), targets.to(device)
            images, targets = images.type(
                torch.cuda.FloatTensor), targets.type(torch.cuda.FloatTensor)

            if normalization == 0 or normalization == 1:
                boolean_index = targets[:, 0] != 0
            if normalization == 2:
                boolean_index = targets[:, 0] != -1

            # forward propagation
            landmark_pred = model(images)
            landmark_pred = torch.reshape(
                landmark_pred, (landmark_pred.shape[0], int(os.getenv('NUM_KPTS')), 2))

            val_loss = loss_func(landmark_pred[np.where(
                boolean_index.cpu())], targets[np.where(boolean_index.cpu())])

            mse = torch.nn.functional.mse_loss(
                landmark_pred[0][boolean_index], targets[boolean_index])

            loss = {'loss': val_loss.item(), 'mse': mse.item()}

            table.append(wandb_prediction_pyplot(
                image=images, targets=targets, outputs=landmark_pred[0],
                loss=loss, epoch=epoch, normalization=normalization))


def wandb_heatmaps_table(table, heatmap_table, model, loader, loss_func, epoch: int, num_example: int = 2, normalization: int = 2, sigma=0.0005):
    """_summary_

    Args:
        table (_type_): _description_
        heatmap_table (_type_): _description_
        model (_type_): _description_
        loader (_type_): _description_
        loss_func (_type_): _description_
        epoch (int): _description_
        num_example (int, optional): _description_. Defaults to 2.
        normalization (int, optional): _description_. Defaults to 2.
        sigma (float, optional): _description_. Defaults to 0.0005.
    """
    # This function receives a table and add a specific number of examples as wandb objects;
    model.eval()

    with torch.inference_mode():
        for idx_batch, (images, targets) in enumerate(loader):

            if idx_batch == num_example:
                break

            images, targets = images.to(device), targets.to(device)
            images, targets = images[0].type(
                torch.cuda.FloatTensor), targets[0].type(torch.cuda.FloatTensor)

            # forward propagation
            heatmap_pred, landmark_pred = model(images[None, :])

            if normalization == 0:
                boolean_index = targets[:, 0] != 0
            if normalization == 2:
                boolean_index = targets[:, 0] != -1

            heatmap_gt = render_gaussian2d(
                targets[boolean_index],
                torch.tensor([sigma, sigma],
                             dtype=torch.float32,
                             device=device),
                heatmap_pred.shape[2:],
                normalized_coordinates=True)

            euclidean_loss = euclidean_losses(
                targets[boolean_index], landmark_pred[0][boolean_index])

            js_loss = js_div_loss_2d(
                heatmap_gt[None, :, :, :], heatmap_pred[:, boolean_index], reduction='none')

            loss = loss_func(euclidean_loss + js_loss[0])

            mse = torch.nn.functional.mse_loss(
                landmark_pred[0][boolean_index], targets[boolean_index])

            loss = {'euclidean_loss': euclidean_loss,
                    'js_loss': js_loss, 'loss': loss, 'mse': mse}

            table.append(wandb_prediction_heatmap(image=images,
                                                  targets=targets,
                                                  outputs=landmark_pred[0],
                                                  loss=loss,
                                                  epoch=epoch,
                                                  normalization=normalization))

            heatmap_table.append(wandb_heatmaps(
                heatmap_gt, heatmap_pred[0], loss, epoch, boolean_index))


def wandb_test_heatmaps_table(table, heatmap_table, model, loader, epoch: int, num_example: int = 2, normalization=2):
    """_summary_

    Args:
        table (_type_): _description_
        heatmap_table (_type_): _description_
        model (_type_): _description_
        loader (_type_): _description_
        loss_func (_type_): _description_
        epoch (int): _description_
        num_example (int, optional): _description_. Defaults to 2.
        normalization (int, optional): _description_. Defaults to 2.
        sigma (float, optional): _description_. Defaults to 0.0005.
    """

    # This function receives a table and add a specific number of examples as wandb objects;
    model.eval()

    with torch.inference_mode():
        for idx_batch, images in enumerate(loader):
            
            if idx_batch == num_example:
                break

            images = images.to(device)
            images = images[0].type(torch.cuda.FloatTensor)
            
            # forward propagation
            heatmap_pred, landmark_pred = model(images[None, :])

            table.append(wandb_prediction_heatmap(image=images, keypoints=landmark_pred[0], epoch=epoch, normalization=normalization))
            heatmap_table.append(
                {
                    'Epoch': epoch + 1,
                    'Heatmap model output': wandb_heatmaps_plot(heatmap_pred[0], grouped=False),
                }
            )


def log_model_artifact(model, input_size, config):
    """_summary_

    Args:
        model (_type_): _description_
        input_size (_type_): _description_
        config (_type_): _description_
    """

    dummy_input = torch.zeros(input_size, device=device)

    # Save the model in the exchangeable ONNX format
    torch.onnx.export(model, dummy_input, './models/' + "model_" + wandb.run.group + '_' +
                      config.model_architecture + ".onnx")

    artifact = wandb.Artifact(config.model_architecture, type='model')
    artifact.add_file('./models/' + "model_" + wandb.run.group +
                      '_' + config.model_architecture + ".onnx")
    wandb.log_artifact(artifact)


def log_model(model_config, path):
    """_summary_

    Args:
        model_config (_type_): _description_
        path (_type_): _description_
    """
    artifact = wandb.Artifact(model_config['model'], type='model')
    artifact.add_file(path)
    wandb.log_artifact(artifact)


def wandb_login():
    """_summary_
    """
    wandb.login(key="272ec3ec3b36f8a715c8ccdf3d905481d92ec3b1",
                host='https://api.wandb.ai', relogin=True)


def log_wandb_table(table, table_name="Table/Table"):
    """_summary_

    Args:
        table (_type_): _description_
        table_name (str, optional): _description_. Defaults to "Table/Table".
    """
    # Add table data to a DataFrame
    table_df = pd.DataFrame(table)

    columns = list(table_df.columns)

    # Prediction table
    pred_table = wandb.Table(data=table_df, columns=columns)

    # Log Table to W&B
    wandb.log({table_name: pred_table})
