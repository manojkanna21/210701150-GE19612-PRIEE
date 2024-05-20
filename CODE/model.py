import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# Define constants
folder_path = r"Medicinal plant dataset"
batch_size = 32
image_size = (224, 224)
seed = 42

# Data augmentation settings
data_augmentation = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,  # Split dataset into training and validation
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

# Create train and validation generators with data augmentation
train_generator = data_augmentation.flow_from_directory(
    folder_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    seed=seed
)

validation_generator = data_augmentation.flow_from_directory(
    folder_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    seed=seed
)

# Define the model architecture
model = Sequential([
    Input(shape=(224, 224, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'),  # Increase complexity by adding more units
    Dense(128, activation='relu'),
    Dense(20, activation='softmax')
])

# Compile the model with Adam optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Learning rate scheduling and early stopping callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.00001, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model with callbacks
history = model.fit(train_generator, epochs=15, validation_data=validation_generator, callbacks=[reduce_lr, early_stop])

# Evaluate the model
loss, accuracy = model.evaluate(validation_generator)
print("Validation Accuracy:", accuracy)
