import tensorflow as tf
from tensorflow import keras


def training(model, trainX, testX, trainY, testY, hyper_params):
    lr = hyper_params['lr']
    epochs = hyper_params['epochs']
    batch_size = hyper_params['batch_size']

    aug = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest")

    print("[INFO] compiling model...")
    opt = tf.keras.optimizers.Adam(learning_rate=lr, decay=lr/epochs)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])

    print("[INFO] training head...")
    H=model.fit(
        aug.flow(trainX, trainY, batch_size=batch_size),
        steps_per_epoch=len(trainX) // batch_size,
        validation_data=(testX, testY),
        validation_steps=len(testX) // batch_size,
        epochs=epochs)

    model.save('classifier_model.h5')
    print('[INFO] Model Saved to Disk !')

    return model