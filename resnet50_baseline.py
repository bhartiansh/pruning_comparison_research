from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model

def build_resnet50_baseline(input_shape=(224, 224, 3), num_classes=200):
    base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=Input(shape=input_shape))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False  # Freeze base layers initially

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model