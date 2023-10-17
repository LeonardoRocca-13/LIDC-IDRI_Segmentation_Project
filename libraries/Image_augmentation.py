import albumentations as Alb
import tensorflow as tf
import numpy as np
import cv2


def Image_augmentation(image, mask):
    """
    Applica trasformazioni di data augmentation a un'immagine e la sua maschera corrispondente.
    Args:
        image (np.array): L'immagine originale.
        mask (np.array): La maschera corrispondente.

    Returns:
        tuple: Una tupla contenente l'immagine augmentata e la maschera augmentata.
    """
    result = image_augmentation(image=image, mask=mask)
    aug_img = result['image']
    aug_mask = result['mask']

    aug_img = np.expand_dims(np.array(aug_img, dtype=np.float32) / 255.0, axis=-1)
    aug_mask = tf.one_hot(tf.cast(aug_mask, tf.int32), depth=3, axis=-1)

    return aug_img, aug_mask


image_augmentation = Alb.Compose(
    [
        Alb.Resize(256, 256),
        Alb.Rotate((1, 359), border_mode=cv2.BORDER_CONSTANT, value=0),
        Alb.HorizontalFlip(p=0.5),
        Alb.RandomBrightnessContrast(brightness_limit=(-0.2, 0.3), contrast_limit=(-0.2, 0.3)),
    ]
)
