from skimage.draw import polygon
import matplotlib.pyplot as plt
import pylidc as pl
import numpy as np
import imageio
import shutil
import os


def Create_dataset():
    """
    Crea un dataset a partire dalle scansioni dei pazienti.
    Salva le immagini delle scansioni e le relative maschere.
    """
    start_patient_id = 'LIDC-IDRI-0001'
    end_patient_id = 'LIDC-IDRI-1012'

    scans = pl.query(pl.Scan).filter(
        pl.Scan.patient_id >= start_patient_id,
        pl.Scan.patient_id <= end_patient_id).all()

    scans.sort(key=lambda x: x.patient_id)

    for scan in scans:  # Scorro tutte le scansioni del dataset
        pt_id = scan.patient_id.split('-')[-1]

        masks = {}

        for nodule in scan.annotations:  # Prendo tutte le annotazioni dei noduli della scansione
            if nodule.malignancy >= 3:
                label = 1  # Malignant Nodule
            else:
                label = 2  # Benign Nodule

            nodule_scans = nodule.scan.load_all_dicom_images()
            contours = sorted(nodule.contours, key=lambda c: c.image_z_position)
            fnames = nodule.scan.sorted_dicom_file_names.split(',')
            index_of_contour = [fnames.index(c.dicom_file_name) for c in contours]

            for index, contours in zip(index_of_contour, contours):
                if index not in masks:
                    masks[index] = np.zeros((512, 512), dtype=np.uint8)

                coords = contours.coords.split('\n')
                x_coords, y_coords = zip(*(map(int, coord.split(',')) for coord in coords if coord))
                rr, cc = polygon(y_coords, x_coords)
                masks[index][rr, cc] = label

        for index, mask in masks.items():
            os.makedirs(os.path.dirname(f'resources/{pt_id}/masks/{index}.png'), exist_ok=True)
            os.makedirs(os.path.dirname(f'resources/{pt_id}/images/{index}.png'), exist_ok=True)

            plt.imsave(f'resources/{pt_id}/images/{index}.png', nodule_scans[index].pixel_array, cmap=plt.cm.gray)
            imageio.imsave(f'resources/{pt_id}/masks/{index}.png', mask)


def Shuffle_dataset():
    """
    Mescola il dataset e suddividilo in training, validation e testing set.
    """
    patient_folders = [folder for folder in os.listdir('../resources')[:-1] if os.path.isdir(f'resources/{folder}/images')]
    np.random.shuffle(patient_folders)

    num_patients = len(patient_folders)
    num_train = int(0.8 * num_patients)
    num_val = int(0.1 * num_patients)
    num_test = num_patients - num_train - num_val

    train_folders = patient_folders[:num_train]
    val_folders = patient_folders[num_train:num_train + num_val]
    test_folders = patient_folders[num_train + num_val:]

    os.makedirs('../resources/training', exist_ok=True)
    os.makedirs('../resources/validation', exist_ok=True)
    os.makedirs('../resources/testing', exist_ok=True)

    for i in train_folders:
        source_path = f'resources/{i}'
        target_path = f'resources/training/{i}'
        shutil.move(source_path, target_path)

    for i in val_folders:
        source_path = f'resources/{i}'
        target_path = f'resources/validation/{i}'
        shutil.move(source_path, target_path)

    for i in test_folders:
        source_path = f'resources/{i}'
        target_path = f'resources/testing/{i}'
        shutil.move(source_path, target_path)
