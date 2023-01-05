import ssl

import hydra
import segmentation_models_pytorch as smp
import torch
import torchvision
from torch import nn

from .._utils import average_loss
from ..models import UNet, get_inception_v3_heatmap
from ._utils import device


def get_model(architecture):

    cfg = (hydra.compose(overrides=["+landmarks=default"])).landmarks

    if architecture['model'] == 'VGG16':
        model = torchvision.models.vgg16(
            weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
        model.classifier[6] = nn.Linear(
            in_features=4096, out_features=2 * cfg.num_keypoints)
    elif architecture['model'] == 'VGG19':
        model = torchvision.models.vgg19(
            weights=torchvision.models.VGG19_Weights.IMAGENET1K_V1)
        # Freeze parameters so we don't backprop through them
        for param in model.parameters():
            param.requires_grad = False
        model.features[0] = nn.Conv2d(1, 64, kernel_size=(
            3, 3), stride=(1, 1), padding=(1, 1))
        model.classifier[6] = nn.Linear(
            in_features=4096, out_features=2*cfg.num_keypoints, bias=True)
    elif architecture['model'] == 'RESNET18':
        model = torchvision.models.resnet18(
            weights=torchvision.models.resnet.ResNet18_Weights.DEFAULT, num_classes=28)
        # Freeze parameters so we don't backprop through them
        for param in model.parameters():
            param.requires_grad = False

        model.conv1 = nn.Conv2d(1, 64, kernel_size=(
            7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    elif architecture['model'] == 'RESNET34':
        model = torchvision.models.resnet34(
            weights=torchvision.models.resnet.ResNet34_Weights.DEFAULT)
        # Freeze parameters so we don't backprop through them
        for param in model.parameters():
            param.requires_grad = False

        model.conv1 = nn.Conv2d(1, 64, kernel_size=(
            7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = nn.Linear(in_features=512, out_features=2*cfg.num_keypoints)
    elif architecture['model'] == 'RESNET50':
        model = torchvision.models.resnet50(
            weights=torchvision.models.resnet.ResNet50_Weights.DEFAULT)
        # Freeze parameters so we don't backprop through them
        for param in model.parameters():
            param.requires_grad = False

        model.conv1 = nn.Conv2d(1, 64, kernel_size=(
            7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = nn.Linear(
            in_features=2048, out_features=2*cfg.num_keypoints, bias=True)
    elif architecture['model'] == 'INCEPTIONV3':
        model = get_inception_v3_heatmap()
    elif architecture['model'] == 'UNET':
        if architecture['pooling'] == 'avg_pooling':
            model = UNet(init_features=architecture['init_features'], output_channels=cfg.num_keypoints,
                         levels=architecture['levels'], avg_polling=True)
        elif architecture['pooling'] == 'max_pooling':
            model = UNet(init_features=architecture['init_features'], output_channels=cfg.num_keypoints,
                         levels=architecture['levels'])
    elif architecture['model'] == 'UNET_SM':
        ssl._create_default_https_context = ssl._create_unverified_context
        model = smp.Unet(encoder_name=architecture['backbone'],
                         encoder_depth=architecture['levels'],
                         encoder_weights=architecture['encoder_weights'],
                         #decoder_use_batchnorm=architecture['decoder_use_batchnorm'],
                         in_channels=1,
                         classes=cfg.num_keypoints,
                         activation="dsnt")
    elif architecture['model'] == 'FPN_SM':
        ssl._create_default_https_context = ssl._create_unverified_context
        model = smp.FPN(encoder_name=architecture['backbone'],
                        encoder_depth=architecture['levels'],
                        encoder_weights=architecture['encoder_weights'],
                        # decoder_use_batchnorm=architecture['decoder_use_batchnorm'],
                        in_channels=1,
                        classes=cfg.num_keypoints,
                        activation="dsnt")

    model = model.to(device)

    return model


def set_optimizer(model, optimizer_name, config):
    params = [p for p in model.parameters() if p.requires_grad]
    if optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(
            params, lr=config.learning_rate, momentum=0.8, weight_decay=config.weight_decay)
    elif optimizer_name == 'adam':
        optimizer = torch.optim.Adam(
            params, lr=config.learning_rate, weight_decay=config.weight_decay)
    elif optimizer_name == 'adadelta':
        optimizer = torch.optim.Adadelta(
            params, lr=config.learning_rate, weight_decay=config.weight_decay)
    elif optimizer_name == 'rmsprop':
        optimizer = torch.optim.RMSprop(
            params, lr=config.learning_rate, weight_decay=config.weight_decay, momentum=0.8)

    return optimizer


def set_loss_function(loss_function_name):
    if loss_function_name == 'rmse':
        return nn.MSELoss()
    elif loss_function_name == 'mse':
        return nn.MSELoss()
    elif loss_function_name == 'wing_loss':
        return nn.MSELoss()
    elif loss_function_name == 'heatmap_loss':
        return average_loss
