import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import Input
import tensorflow.keras.layers as layers


def fine_tune_architecture():
    baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

    headModel = baseModel.output
    headModel = layers.AveragePooling2D(pool_size=(7, 7))(headModel)
    headModel = layers.Flatten(name="flatten")(headModel)
    headModel = layers.Dense(128, activation="relu")(headModel)
    headModel = layers.Dropout(0.5)(headModel)
    headModel = layers.Dense(2, activation="softmax")(headModel)

    model = tf.keras.Model(inputs=baseModel.input, outputs=headModel)

    # loop over all layers in the base model and freeze them so they will
    # *not* be updated during the first training process
    for layer in baseModel.layers:
        layer.trainable = False

    return model


fine_tune_architecture()
