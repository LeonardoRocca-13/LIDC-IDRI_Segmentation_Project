from keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping, TerminateOnNaN
from tensorflow.data import Dataset
from keras.optimizers import Adam
import tensorflow as tf
import pickle
import keras

from libraries.Image_generator import Image_generator
from libraries.Models_creation import Create_fcnn_model, Create_unet_model


f1_score_custom = keras.metrics.F1Score(average='macro')
def F1_score_custom(y_true, y_pred):
    y_true = tf.reshape(y_true, (-1, 3))
    y_pred = tf.reshape(y_pred, (-1, 3))

    return f1_score_custom(y_true, y_pred)


def Train_FCNN_Model(loss: str, save_path: str, buffer_size: int, batch_size: int, epochs: int, learning_rate: float):
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        checkpoint_filepath = save_path + '/FCNN_{epoch:03d}.h5'

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
            LearningRateScheduler(lambda epoch: learning_rate * tf.math.exp(-0.1 * (epoch // 10)))
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

        train_dataset = train_dataset.shuffle(buffer_size=buffer_size).batch(batch_size)
        val_dataset = val_dataset.shuffle(buffer_size=buffer_size).batch(batch_size)

        # weights = {0: 0.5004178292268188, 1: 766.2575319317873, 2: 2740.648423505425}

        metrics = [
            keras.metrics.categorical_accuracy,
            f1_score_custom,
            keras.metrics.TruePositives(name='tp'),
            keras.metrics.FalsePositives(name='fp'),
            keras.metrics.TrueNegatives(name='tn'),
            keras.metrics.FalseNegatives(name='fn'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc'),
            keras.metrics.AUC(name='prc', curve='PR')  # precision-recall curve
        ]

        try:
            fcnn = Create_fcnn_model((256, 256, 1), 3)

            fcnn.compile(
                optimizer=Adam(),
                loss=loss,
                metrics=metrics
            )

            history = fcnn.fit(
                x=train_dataset,
                validation_data=val_dataset,
                callbacks=callbacks,
                epochs=epochs
            )
        except keras.applications.xception.Xception:
            print('Parametri non validi!')
            exit(-1)

    with open(save_path + '/FCNN_history.pkl', 'wb') as file:
        pickle.dump(history.history, file)

    fcnn.save(save_path + '/FCNN.keras')


def Train_UNET_Model(loss: str, save_path: str, buffer_size: int, batch_size: int, epochs: int, learning_rate: float):
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        checkpoint_filepath = save_path + '/UNET_{epoch:03d}.h5'

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
            LearningRateScheduler(lambda epoch: learning_rate * tf.math.exp(-0.1 * (epoch // 10)))
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

        train_dataset = train_dataset.shuffle(buffer_size=buffer_size).batch(batch_size)
        val_dataset = val_dataset.shuffle(buffer_size=buffer_size).batch(batch_size)

        # weights = {0: 0.5004178292268188, 1: 766.2575319317873, 2: 2740.648423505425}

        metrics = [
            keras.metrics.categorical_accuracy,
            f1_score_custom,
            keras.metrics.TruePositives(name='tp'),
            keras.metrics.FalsePositives(name='fp'),
            keras.metrics.TrueNegatives(name='tn'),
            keras.metrics.FalseNegatives(name='fn'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc'),
            keras.metrics.AUC(name='prc', curve='PR')  # precision-recall curve
        ]

        try:

            unet = Create_unet_model((256, 256, 1), 3)

            unet.compile(
                optimizer=Adam(),
                loss=loss,
                metrics=metrics
            )

            history = unet.fit(
                x=train_dataset,
                validation_data=val_dataset,
                callbacks=callbacks,
                epochs=epochs
            )

        except keras.applications.xception.Xception:
            print('Parametri non validi!')
            exit(-1)

    with open(save_path + '/UNET_history.pkl', 'wb') as file:
        pickle.dump(history.history, file)

    unet.save(save_path + '/UNET.keras')
