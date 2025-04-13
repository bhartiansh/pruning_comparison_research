from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_tiny_imagenet_data(base_dir='./tiny-imagenet-200', target_size=(224, 224), batch_size=64):
    train_dir = f"{base_dir}/train"
    val_dir = f"{base_dir}/val"

    train_datagen = ImageDataGenerator(rescale=1./255,
                                       rotation_range=20,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       horizontal_flip=True)

    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical')

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical')

    return train_generator, val_generator