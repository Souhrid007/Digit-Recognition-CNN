# train_model.py
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Build an improved CNN model with Dropout to prevent overfitting
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dropout(0.5),  # Dropout layer added to prevent overfitting
    Dense(128, activation='relu'),
    Dropout(0.5),  # Dropout layer added after dense layer
    Dense(10, activation='softmax')
])

# Compile the model
from tensorflow.keras.optimizers import Adam

# Use a smaller learning rate
model.compile(optimizer=Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])


# Use data augmentation to improve generalization
datagen = ImageDataGenerator(
    rotation_range=15,  # Increase the range of rotation
    width_shift_range=0.2,  # Increase the width shift
    height_shift_range=0.2,  # Increase the height shift
    zoom_range=0.2,  # Increase zoom range
    shear_range=0.2,  # Add shear transformation
    horizontal_flip=True  # Add horizontal flipping
)


datagen.fit(x_train)

# Train the model
model.fit(datagen.flow(x_train, y_train, batch_size=64, shuffle=True), epochs=10, validation_data=(x_test, y_test))

# Save the improved model
model.save('model1.h5')
print("Improved model saved as model1.h5")
