import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_tiny_imagenet_data(data_dir='./tiny-imagenet-200', target_size=(224, 224), batch_size=64):
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    datagen = ImageDataGenerator(rescale=1./255)

    train_gen = datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    val_gen = datagen.flow_from_directory(
        val_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    return train_gen, val_gen