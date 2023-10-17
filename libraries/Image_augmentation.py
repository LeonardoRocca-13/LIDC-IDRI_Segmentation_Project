import albumentations as Alb
import tensorflow as tf
import numpy as np
import cv2


def Image_augmentation(image, mask):
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
        Alb.RandomBrightnessContrast(brightness_limit=(-0.2, 0.25), contrast_limit=(-0.2, 0.25)),
    ]
)
