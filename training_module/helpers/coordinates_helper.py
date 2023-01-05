import torch

def pixel_to_normalized_coordinates(coords, img_size):
    """ Normalized pixel value coordinates to values between [-1,1]

    Args:
        coords (Tensor or List[(int,int)]): Coordinates in pixel values
        img_size (Tuple(int)): Tuple with width and height image values

    Returns:
        coords: Returns normalized coordinates
    """
    if torch.is_tensor(coords):
        img_size = img_size.clone().detach().flip(-1)
    return ((2 * coords) / img_size) - 1


def normalized_to_pixel_coordinates(coords, img_size):
    """ Convertes normalized coordinates values to its pixel values

    Args:
        coords (Tensor or List[(int,int)]): Normalized coordinates values
        img_size (Tuple(int)): Tuple with width and height image valuess

    Returns:
        coords: Returns coordinates in pixel values 
    """
    if torch.is_tensor(coords):
        img_size = img_size.clone().detach().flip(-1)
    return ((coords + 1) / 2) * img_size
