import os

def build_cnn_model(img_size=128, num_classes=1):
    """
    Builds the 4-stage Conv2D + BatchNorm deepfake detection baseline model
    from the project notebooks.
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

    layers = [
        Conv2D(32, (3, 3), activation="relu", input_shape=(img_size, img_size, 3)),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        Conv2D(64, (3, 3), activation="relu"),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        Conv2D(128, (3, 3), activation="relu"),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        Conv2D(256, (3, 3), activation="relu"),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        Flatten(),
        Dense(512, activation="relu"),
        Dropout(0.3),
        Dense(256, activation="relu"),
        Dropout(0.3)
    ]

    if num_classes == 1:
        layers.append(Dense(1, activation="sigmoid"))
        loss = "binary_crossentropy"
    else:
        layers.append(Dense(num_classes, activation="softmax"))
        loss = "categorical_crossentropy"

    model = Sequential(layers)
    model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
    return model


def load_cnn_model(weights_path=None, img_size=128, num_classes=1):
    """
    Instantiates CNN and optionally loads weights from .h5 or SavedModel format.
    """
    from tensorflow.keras.models import load_model

    has_weights = False
    if weights_path and os.path.exists(weights_path):
        try:
            model = load_model(weights_path)
            has_weights = True
            return model, has_weights
        except Exception as e:
            print(f"Warning: Failed to load Keras model directly: {e}, compiling fresh architecture...")

    model = build_cnn_model(img_size=img_size, num_classes=num_classes)
    if weights_path and os.path.exists(weights_path):
        try:
            model.load_weights(weights_path)
            has_weights = True
        except Exception as e:
            print(f"Warning: Failed to load weights: {e}")

    return model, has_weights
