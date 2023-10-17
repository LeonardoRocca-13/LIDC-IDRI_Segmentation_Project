from PIL import Image
import numpy as np
import random
import os

from libraries.Image_augmentation import Image_augmentation


def Image_generator(split: str = 'training'):
    """
    Generatore di immagini e maschere a partire dai dati del dataset.
    Args:
        split (str, optional): Specifica il tipo di divisione dei dati (es. 'training', 'validation', 'testing').

    Yields:
        tuple: Una tupla contenente l'immagine e la maschera corrispondente.
    """
    patients = os.listdir(f'resources/{split}')
    random.shuffle(patients)  # Shuffle the list of patients

    for patient in patients:
        img_dir = f'resources/{split}/{patient}/images'
        if os.path.exists(img_dir):
            for file in os.listdir(img_dir):
                img = np.array(Image.open(f'resources/{split}/{patient}/images/{file}').convert('L'))
                mask = np.array(Image.open(f'resources/{split}/{patient}/masks/{file}').convert('L'))

                aug_img, aug_mask = Image_augmentation(image=img, mask=mask)

                yield aug_img, aug_mask
