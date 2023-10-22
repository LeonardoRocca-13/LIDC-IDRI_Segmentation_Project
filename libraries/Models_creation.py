from keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, SeparableConv2D, Activation, MaxPooling2D, add, UpSampling2D
from keras.models import Model, Sequential
from keras.activations import relu
import tensorflow as tf


def Create_fcnn_model(input_shape: tuple, num_classes: int):
    """
    Crea un modello FCNN.
    Args:
        input_shape (tuple): Dimensioni dell'input (altezza, larghezza, canali).
        num_classes (int): Numero di classi di output.

    Returns:
        tf.keras.Model: Il modello FCNN.
    """
    model = Sequential(
        [
            # Sezione di encoding
            Conv2D(filters=64, kernel_size=3, strides=2, padding="same", activation=relu, input_shape=input_shape),
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

            # Sezione di decoding
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

            # Strato di output
            Conv2D(filters=num_classes, kernel_size=3, padding="same", activation="softmax")
        ]
    )

    return model


def Create_unet_model(input_shape: tuple, num_classes: int):
    """
    Crea un modello U-Net.
    Args:
        input_shape (tuple): Dimensioni dell'input (altezza, larghezza, canali).
        num_classes (int): Numero di classi di output.

    Returns:
        tf.keras.Model: Il modello U-Net modificato.
    """
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Prima metà della rete: downsampling degli input
    x = Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation(relu)(x)

    previous_block_activation = x

    for filters in [64, 128, 256]:
        for _ in range(2):
            x = Activation(relu)(x)
            x = SeparableConv2D(filters, 3, padding="same")(x)
            x = BatchNormalization()(x)

        x = MaxPooling2D(3, strides=2, padding="same")(x)

        # Proiezione del residuo
        residual = Conv2D(filters, 1, strides=2, padding="same")(previous_block_activation)
        x = add([x, residual])
        previous_block_activation = x

    # Seconda metà della rete: upsampling degli input
    for filters in [256, 128, 64, 32]:
        for _ in range(2):
            x = Activation(relu)(x)
            x = Conv2DTranspose(filters, 3, padding="same")(x)
            x = BatchNormalization()(x)

        x = UpSampling2D(2)(x)

        # Proiezione del residuo
        residual = UpSampling2D(2)(previous_block_activation)
        residual = Conv2D(filters, 1, padding="same")(residual)
        x = add([x, residual])
        previous_block_activation = x

    # Aggiunge un livello di classificazione per pixel
    outputs = Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

    # Definisce il modello
    model = Model(inputs, outputs)

    return model
