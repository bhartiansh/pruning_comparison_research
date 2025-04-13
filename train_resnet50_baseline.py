from tinyimagenet_preprocessing import load_tiny_imagenet_data
from resnet50_baseline import build_resnet50_baseline
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

# Ensure models folder exists
os.makedirs('models', exist_ok=True)

# Load data
train_gen, val_gen = load_tiny_imagenet_data()

# Build model
model = build_resnet50_baseline(num_classes=200)

# Callbacks
checkpoint = ModelCheckpoint(
    filepath='models/resnet50_best_model.keras',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max'
)
early_stop = EarlyStopping(patience=5, monitor='val_accuracy', restore_best_weights=True)

# Train
model.fit(
    train_gen,
    epochs=30,
    validation_data=val_gen,
    callbacks=[checkpoint, early_stop]
)

# Save final model
model.save('models/resnet50_final_model.keras')