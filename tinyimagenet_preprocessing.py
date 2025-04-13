from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_tiny_imagenet_data(data_dir='./tiny-imagenet-200', target_size=(224, 224), batch_size=64):
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    train_gen = datagen.flow_from_directory(
        directory=f"{data_dir}/train",
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    val_gen = datagen.flow_from_directory(
        directory=f"{data_dir}/train",
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    return train_gen, val_gen