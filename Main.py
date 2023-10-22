import keras

from libraries.Train_model_pipeline import Train_FCNN_Model, Train_UNET_Model

if __name__ == '__main__':
    Train_FCNN_Model(
        loss=keras.losses.categorical_focal_crossentropy(),
        save_path='fcnn-path',
        buffer_size=2000,
        batch_size=64,
        epochs=20,
        learning_rate=1e-3
    )

    Train_UNET_Model(
        loss=keras.losses.categorical_crossentropy(),
        save_path='unet-path',
        buffer_size=2000,
        batch_size=64,
        epochs=20,
        learning_rate=1e-3
    )
