from keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping, TerminateOnNaN
from keras.layers import Conv2D, Conv2DTranspose, BatchNormalization
from keras.losses import categorical_crossentropy, categorical_focal_crossentropy
from keras.models import Sequential
from tensorflow.data import Dataset
from keras.activations import relu
from keras.metrics import F1Score
from keras.optimizers import Adam
import tensorflow as tf
import pickle
import keras

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

    train_dataset = train_dataset.shuffle(buffer_size=2_000).batch(64)
    val_dataset = val_dataset.shuffle(buffer_size=2_000).batch(64)

    fcnn = Sequential(
        [
            Conv2D(filters=64, kernel_size=3, strides=2, padding="same", activation=relu, input_shape=(256, 256, 1)),
            Conv2D(filters=64, kernel_size=3, padding="same", activation=relu),
            BatchNormalization(),
            Conv2D(filters=128, kernel_size=3, strides=2, padding="same", activation=relu),
            Conv2D(filters=128, kernel_size=3, padding="same", activation=relu),
            BatchNormalization(),
            Conv2D(filters=256, kernel_size=3, strides=2, padding="same", activation=relu),
            Conv2D(filters=256, kernel_size=3, padding="same", activation=relu),
            BatchNormalization(),
            Conv2D(filters=512, kernel_size=3, strides=2, padding="same", activation=relu),
            Conv2D(filters=512, kernel_size=3, padding="same", activation=relu),

            Conv2DTranspose(filters=512, kernel_size=3, padding="same", activation=relu),
            Conv2DTranspose(filters=512, kernel_size=3, padding="same", strides=2, activation=relu),
            BatchNormalization(),
            Conv2DTranspose(filters=256, kernel_size=3, padding="same", activation=relu),
            Conv2DTranspose(filters=256, kernel_size=3, padding="same", strides=2, activation=relu),
            BatchNormalization(),
            Conv2DTranspose(filters=128, kernel_size=3, padding="same", activation=relu),
            Conv2DTranspose(filters=128, kernel_size=3, padding="same", strides=2, activation=relu),
            BatchNormalization(),
            Conv2DTranspose(filters=64, kernel_size=3, padding="same", activation=relu),
            Conv2DTranspose(filters=64, kernel_size=3, padding="same", strides=2, activation=relu),

            Conv2D(filters=3, kernel_size=3, padding="same", activation="softmax")
        ]
    )

    f1_score = F1Score(average='macro')


    def f1_score_custom(y_true, y_pred):
        y_true = tf.reshape(y_true, (-1, 3))
        y_pred = tf.reshape(y_pred, (-1, 3))

        return f1_score(y_true, y_pred)


    weights = {0: 0.5004178292268188, 1: 766.2575319317873, 2: 2740.648423505425}
    weights_2 = {0: 0.3336118861512125, 1: 510.8383546211915, 2: 1827.0989490036166}

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

    fcnn.compile(
        optimizer=Adam(),
        loss=categorical_crossentropy,
        metrics=metrics
    )

    # fcnn.load_weights('models/fcnn/fcnn_focal_normal_parte2.keras')

    history = fcnn.fit(
        x=train_dataset,
        validation_data=val_dataset,
        callbacks=callbacks,
        epochs=15
    )

    with open('prova.pkl', 'wb') as file:
        pickle.dump(history.history, file)

    fcnn.save('prova.keras')


