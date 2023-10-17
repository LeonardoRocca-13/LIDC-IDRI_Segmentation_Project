from keras.layers import Conv2D, Conv2DTranspose, SeparableConv2D, BatchNormalization, Activation, MaxPooling2D, add, UpSampling2D
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping, TerminateOnNaN
from tensorflow.data import Dataset
from keras.metrics import F1Score
from keras.optimizers import Adam
from keras.models import Model
import albumentations as Alb
import tensorflow as tf
from PIL import Image
import numpy as np
import keras
import cv2
import os
import pickle

from libraries.Image_generator import Image_generator

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    checkpoint_filepath = 'checkpoints/model_at_epoch_{epoch:03d}.h5'

    callbacks = [
        ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_best_only=True
        ),
        EarlyStopping(
            patience=10,
            restore_best_weights=True
        ),
        TerminateOnNaN(),
        LearningRateScheduler(lambda epoch: 1e-3 * tf.math.exp(-0.1 * (epoch // 10)))
    ]

    train_dataset = Dataset.from_generator(
        lambda: Image_generator('training'),
        output_signature=(
            tf.TensorSpec(shape=(256, 256, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(256, 256, 3), dtype=tf.float32),
        )
    )

    val_dataset = Dataset.from_generator(
        lambda: Image_generator('validation'),
        output_signature=(
            tf.TensorSpec(shape=(256, 256, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(256, 256, 3), dtype=tf.float32),
        )
    )

    train_dataset = train_dataset.shuffle(buffer_size=15_000).batch(64)
    val_dataset = val_dataset.shuffle(buffer_size=15_000).batch(64)


    def get_unet_model(input_shape, num_classes):
        inputs = tf.keras.layers.Input(shape=input_shape)

        ### [First half of the network: downsampling inputs] ###

        # Entry block
        x = Conv2D(32, 3, strides=2, padding="same")(inputs)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        previous_block_activation = x  # Set aside residual

        # Blocks 1, 2, 3 are identical apart from the feature depth.
        for filters in [64, 128, 256]:
            x = Activation("relu")(x)
            x = SeparableConv2D(filters, 3, padding="same")(x)
            x = BatchNormalization()(x)

            x = Activation("relu")(x)
            x = SeparableConv2D(filters, 3, padding="same")(x)
            x = BatchNormalization()(x)

            x = MaxPooling2D(3, strides=2, padding="same")(x)

            # Project residual
            residual = Conv2D(filters, 1, strides=2, padding="same")(previous_block_activation)
            x = add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        ### [Second half of the network: upsampling inputs] ###

        for filters in [256, 128, 64, 32]:
            x = Activation("relu")(x)
            x = Conv2DTranspose(filters, 3, padding="same")(x)
            x = BatchNormalization()(x)

            x = Activation("relu")(x)
            x = Conv2DTranspose(filters, 3, padding="same")(x)
            x = BatchNormalization()(x)

            x = UpSampling2D(2)(x)

            # Project residual
            residual = UpSampling2D(2)(previous_block_activation)
            residual = Conv2D(filters, 1, padding="same")(residual)
            x = add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        # Add a per-pixel classification layer
        outputs = Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

        # Define the model
        model = Model(inputs, outputs)
        return model


    f1_score = F1Score(average='macro')


    def f1_score_custom(y_true, y_pred):
        y_true = tf.reshape(y_true, (-1, 3))
        y_pred = tf.reshape(y_pred, (-1, 3))

        return f1_score(y_true, y_pred)


    metrics = [
        keras.metrics.categorical_accuracy,
        f1_score_custom,
        keras.metrics.TruePositives(name='tp'),
        keras.metrics.FalsePositives(name='fp'),
        keras.metrics.TrueNegatives(name='tn'),
        keras.metrics.FalseNegatives(name='fn'),
        keras.metrics.BinaryAccuracy(name='accuracy'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc'),
        keras.metrics.AUC(name='prc', curve='PR')  # precision-recall curve
    ]

    u_net = get_unet_model((256, 256, 1), 3)

    weights = {0: 0.5004178292268188, 1: 766.2575319317873, 2: 2740.648423505425}
    weights_2 = {0: 0.3336118861512125, 1: 510.8383546211915, 2: 1827.0989490036166}

    u_net.compile(
        optimizer=Adam(),
        loss=keras.losses.categorical_focal_crossentropy,
        metrics=metrics
    )

    u_net.load_weights('models/u-net/u-net_crs_normal.keras')

    history = u_net.fit(
        x=train_dataset,
        validation_data=val_dataset,
        callbacks=callbacks,
        epochs=20
    )

    with open('../models/u-net/u-net_focal_normal_parte2.pkl', 'wb') as file:
        pickle.dump(history.history, file)

    u_net.save('u-net_focal_normal_parte2.keras')
