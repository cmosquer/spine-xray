import copy
import os

import albumentations as A
import hydra
import numpy as np
import pandas as pd
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from .class_dataset import SpineDataset, SpineRXImages


def get_transform(config_hyp, type_dataset: str = 'train', wandb_table=None) -> A.Compose:
    """ Returns a specific transformation depending on type of dataset """
    if config_hyp.model_architecture['model'] == 'VGG19' or config_hyp.model_architecture['model'] == 'RESNET34' or config_hyp.model_architecture['model'] == 'RESNET50':
        if type_dataset == 'train':
            if config_hyp.dataset['data_augmentation']:
                wandb_table['Data Augmentation'] = 'True'
                wandb_table['Transformations'] = "Horizontal Flip (p=0.3), GaussNoise (p=0.2), MotionBlur(p=0.2), Blur (p=0.1), RandomBrightnessContrast (p=0.1), Resize (224x224)"
                return A.Compose(
                    [
                        A.HorizontalFlip(p=0.3),
                        A.GaussNoise(p=0.2),
                        A.OneOf([
                            A.MotionBlur(blur_limit=5, p=0.2),
                            A.Blur(blur_limit=3, p=0.1),
                        ], p=0.2),
                        A.RandomBrightnessContrast(p=0.1),
                        A.Resize(height=224, width=224),
                        A.Normalize(mean=(0., 0., 0.), std=(
                            1., 1., 1.), max_pixel_value=255.),
                        ToTensorV2(),
                    ],
                    keypoint_params=A.KeypointParams(
                        format="xy", remove_invisible=False)
                )
            else:
                wandb_table['Data Augmentation'] = 'False'
                wandb_table['Transformations'] = "Resize (224x224)"
                return A.Compose(
                    [
                        A.Resize(height=224, width=224),
                        A.Normalize(mean=(0., 0., 0.), std=(
                            1., 1., 1.), max_pixel_value=255.),
                        ToTensorV2(),
                    ],
                    keypoint_params=A.KeypointParams(
                        format="xy", remove_invisible=False)
                )
        elif type_dataset == 'val' or type_dataset == 'test':
            wandb_table['Data Augmentation'] = 'False'
            wandb_table['Transformations'] = "Resize (224x224)"
            return A.Compose(
                [
                    A.Resize(height=224, width=224),
                    A.Normalize(mean=(0., 0., 0.), std=(
                        1., 1., 1.), max_pixel_value=255.),
                    ToTensorV2(),
                ],
                keypoint_params=A.KeypointParams(
                    format="xy", remove_invisible=False)
            )
    elif config_hyp.model_architecture['model'] in ['UNET_SM', 'FPN_SM']:
        if type_dataset == 'train':
            if config_hyp.dataset['data_augmentation']:
                wandb_table['Data Augmentation'] = 'True'
                wandb_table['Transformations'] = "Horizontal Flip (p=0.3), GaussNoise (p=0.2), MotionBlur(p=0.2), Blur (p=0.1), RandomBrightnessContrast (p=0.1), Rotate [-30,30] (p=0.3),Resize (928x416)"
                return A.Compose(
                    [
                        A.HorizontalFlip(p=0.3),
                        A.GaussNoise(p=0.2),
                        A.OneOf([
                            A.MotionBlur(blur_limit=5, p=0.2),
                            A.Blur(blur_limit=3, p=0.1),
                        ], p=0.2),
                        A.RandomBrightnessContrast(p=0.1),
                        A.Resize(height=928, width=416),
                        A.Rotate(limit=[-30,30], p=0.3),
                        A.Normalize(mean=(0., 0., 0.), std=(
                            1., 1., 1.), max_pixel_value=255.),
                        ToTensorV2(),
                    ],
                    keypoint_params=A.KeypointParams(
                        format="xy", remove_invisible=False)
                )
            else:
                wandb_table['Data Augmentation'] = 'False'
                wandb_table['Transformations'] = "Resize (928x416)"
                return A.Compose(
                    [
                        A.Resize(height=928, width=416),
                        A.Normalize(mean=(0., 0., 0.), std=(
                            1., 1., 1.), max_pixel_value=255.),
                        ToTensorV2(),
                    ],
                    keypoint_params=A.KeypointParams(
                        format="xy", remove_invisible=False)
                )
        elif type_dataset == 'train_right_sagittal':
            if config_hyp.dataset['data_augmentation']:
                wandb_table['Data Augmentation'] = 'True'
                wandb_table['Transformations'] = "GaussNoise (p=0.2), MotionBlur(p=0.2), Blur (p=0.1), RandomBrightnessContrast (p=0.1), Rotate [-30,30] (p=0.3),Resize (928x416)"
                transforms = {
                    "RIGHT_SAGITTAL": A.Compose(
                    [
                        A.GaussNoise(p=0.2),
                        A.OneOf([
                            A.MotionBlur(blur_limit=5, p=0.2),
                            A.Blur(blur_limit=3, p=0.1),
                        ], p=0.2),
                        A.RandomBrightnessContrast(p=0.1),
                        A.Resize(height=928, width=416),
                        A.Rotate(limit=[-30,30], p=0.3),
                        A.Normalize(mean=(0., 0., 0.), std=(
                            1., 1., 1.), max_pixel_value=255.),
                        ToTensorV2(),
                    ],
                    keypoint_params=A.KeypointParams(
                        format="xy", remove_invisible=False)
                    ),
                    "LEFT_SAGITTAL": A.Compose(
                    [
                        A.HorizontalFlip(p=1),
                        A.GaussNoise(p=0.2),
                        A.OneOf([
                            A.MotionBlur(blur_limit=5, p=0.2),
                            A.Blur(blur_limit=3, p=0.1),
                        ], p=0.2),
                        A.RandomBrightnessContrast(p=0.1),
                        A.Resize(height=928, width=416),
                        A.Rotate(limit=[-30,30], p=0.3),
                        A.Normalize(mean=(0., 0., 0.), std=(
                            1., 1., 1.), max_pixel_value=255.),
                        ToTensorV2(),
                    ],
                    keypoint_params=A.KeypointParams(
                        format="xy", remove_invisible=False)
                    ),
                    }
                return transforms
        elif type_dataset == 'val':
            wandb_table['Data Augmentation'] = 'False'
            wandb_table['Transformations'] = "Resize (928x416)"
            return A.Compose(
                [
                    A.Resize(height=928, width=416),
                    A.Normalize(mean=(0., 0., 0.), std=(
                        1., 1., 1.), max_pixel_value=255.),
                    ToTensorV2(),
                ],
                keypoint_params=A.KeypointParams(
                    format="xy", remove_invisible=False)
            )
        elif type_dataset == 'test':
            wandb_table['Data Augmentation'] = 'False'
            wandb_table['Transformations'] = "Resize (928x416)"
            return A.Compose(
                [
                    A.Resize(height=928, width=416),
                    A.Normalize(mean=(0., 0., 0.), std=(
                        1., 1., 1.), max_pixel_value=255.),
                    ToTensorV2(),
                ]
            )


def get_spine_datasets(config_hyp, type_normalization: int = 0, wandb_table=None, seed=None, custom_split=True) -> tuple[DataLoader, DataLoader]:

    config_dataset = (hydra.compose(overrides=["+dataset=default"])).dataset

    wandb_table.append({'Dataset': 'Train'})
    wandb_table.append({'Dataset': 'Validation'})

    dataset = SpineDataset(root=config_dataset.images_path,
                           ann_file=config_dataset.dataset_json,
                           transform=get_transform(config_hyp, type_dataset='train_right_sagittal',
                                                   wandb_table=wandb_table[find_dataset_idx(wandb_table, 'Train')]),
                           type_normalization=type_normalization)

    if custom_split:
        info = pd.read_csv(config_dataset.dicom_metadata)
        dl_train, dl_val = custom_split_dataset(dataset=dataset, batch_size=config_hyp.dataset['batch_size'],
                                                split=config_hyp.dataset['split'], shuffle=config_hyp.dataset['random'], seed=seed,
                                                dataset_info=info, join_on='FileNameJPG', column_filter=config_hyp.dataset['split_wise'])
                                                
        wandb_table[0]['Length'] = len(dl_train.sampler.indices)
        wandb_table[0]['IDs'] = dl_train.sampler.indices

        wandb_table[1]['Length'] = len(dl_val.sampler.indices)
        wandb_table[1]['IDs'] = dl_val.sampler.indices
        
        dl_val.dataset.transform = get_transform(
            config_hyp, type_dataset='val', wandb_table=wandb_table[find_dataset_idx(wandb_table, 'Validation')])
    else:
        dl_train, dl_val = split_dataset(dataset=dataset, batch_size=config_hyp.dataset['batch_size'],
                                         split=config_hyp.dataset['split'], shuffle=config_hyp.dataset['shuffle'], seed=seed)
        
        wandb_table[0]['Length'] = len(dl_train.sampler.indices)
        wandb_table[0]['IDs'] = np.array(dl_train.sampler.indices, dtype=int)

        wandb_table[1]['Length'] = len(dl_val.sampler.indices)
        wandb_table[1]['IDs'] = np.array(dl_val.sampler.indices, dtype=int)
        
        dl_val.dataset.transform = get_transform(
            config_hyp, type_dataset='val', wandb_table=wandb_table[find_dataset_idx(wandb_table, 'Validation')])

    return dl_train, dl_val


def get_test_spine_dataset(config_hyp, wandb_table=None) -> DataLoader:

    config_dataset = (hydra.compose(overrides=["+dataset=default"])).dataset

    if wandb_table is not None:
        wandb_table.append({'Dataset': 'Test'})
        test_dataset = SpineRXImages(root=config_dataset.test_images_path,
                                     transform=get_transform(config_hyp,
                                                             type_dataset='test',
                                                             wandb_table=wandb_table[find_dataset_idx(wandb_table, 'Test')]))
    else:
        test_dataset = SpineRXImages(root=config_dataset.test_images_path, 
                                     transform=get_transform(config_hyp, type_dataset='test'))

    dl_test = DataLoader(test_dataset, batch_size=config_hyp.dataset['batch_size'], shuffle=True)

    return dl_test


def split_dataset(dataset, batch_size, split, shuffle, seed=None):
    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)

    indices = list(range(dataset_size))
    id_split = int(np.floor(split * dataset_size))

    if shuffle:
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(indices)

    train_indices, val_indices = indices[id_split:], indices[:id_split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler)
    validation_loader = DataLoader(copy.deepcopy(
        dataset), batch_size=batch_size, sampler=valid_sampler)

    return train_loader, validation_loader


def custom_split_dataset(dataset, batch_size, split, shuffle, seed=None, dataset_info=None, join_on=None, column_filter=None):
    df = pd.DataFrame(np.stack([dataset.ids, dataset.paths]).T, columns=[
                      'IDs', 'FileName'])
    df = pd.merge(df, dataset_info, left_on='FileName',
                  right_on=join_on, how='left')
    df = df.drop_duplicates(subset=[column_filter])

    dataset.ids = np.array(df['IDs'], dtype=int)
    dataset.paths = np.array(df['FileName'], dtype=str)

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    id_split = int(np.floor(split * dataset_size))

    if shuffle:
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(indices)

    train_indices, val_indices = indices[id_split:], indices[:id_split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler)
    validation_loader = DataLoader(copy.deepcopy(
        dataset), batch_size=batch_size, sampler=valid_sampler)

    return train_loader, validation_loader


def find_dataset_idx(list_dataset, name_dataset) -> int:
    return int([idx for idx, dataset in enumerate(list_dataset) if dataset['Dataset'] == name_dataset][0])


# ---------------------------------------------------------------------
#  Deprecated
# def get_data(config_hyp, type_dataset: str = 'train', type_normalization: int = 0, wandb_table=None) -> SpineDataset:
#     """ Creates a SportDataset object depending on type of dataset needed """
#     load_dotenv()
#     if type_dataset == 'train':
#         train_dataset = SpineDataset(root=os.getenv('IMG_DATA_DIR'),
#                                      annFile=os.getenv('TRAIN_ANNO_DATA_DIR'),
#                                      transform=get_transform(
#                                          config_hyp, type_dataset, wandb_table),
#                                      type_normalization=type_normalization)
#         wandb_table['Dataset'] = 'Train'
#         wandb_table['Length'] = len(train_dataset)
#         wandb_table['Type Normalization'] = type_normalization
#         wandb_table['Root'] = os.getenv('IMG_DATA_DIR')
#         wandb_table['Annotation File'] = os.getenv('TRAIN_ANNO_DATA_DIR')
#         return train_dataset
#     elif type_dataset == 'val':
#         val_dataset = SpineDataset(root=os.getenv('IMG_DATA_DIR'),
#                                    annFile=os.getenv('VAL_ANNO_DATA_DIR'),
#                                    transform=get_transform(
#                                        config_hyp, type_dataset, wandb_table),
#                                    type_normalization=type_normalization)
#         wandb_table['Dataset'] = 'Validation'
#         wandb_table['Length'] = len(val_dataset)
#         wandb_table['Type Normalization'] = type_normalization
#         wandb_table['Root'] = os.getenv('IMG_DATA_DIR')
#         wandb_table['Annotation File'] = os.getenv('VAL_ANNO_DATA_DIR')
#         return val_dataset
#     elif type_dataset == 'test':
#         test_dataset = SpineDataset(root=os.getenv('IMG_DATA_DIR'),
#                                     annFile=os.getenv('TEST_ANNO_DATA_DIR'),
#                                     transform=get_transform(
#                                         config_hyp, type_dataset, wandb_table),
#                                     type_normalization=type_normalization)
#         wandb_table['Dataset'] = 'Test'
#         wandb_table['Length'] = len(test_dataset)
#         wandb_table['Type Normalization'] = type_normalization
#         wandb_table['Root'] = os.getenv('IMG_DATA_DIR')
#         wandb_table['Annotation File'] = os.getenv('VAL_ANNO_DATA_DIR')
#         return test_dataset
# def get_dataloader(data: SpineDataset, config_hyp, is_train: bool) -> DataLoader:
#     """ Returns a pytorch Dataloader made with configuration received as parameters """
#     return DataLoader(dataset=data, batch_size=config_hyp.dataset['batch_size'], shuffle=True if is_train else False, num_workers=1)
# ---------------------------------------------------------------------
