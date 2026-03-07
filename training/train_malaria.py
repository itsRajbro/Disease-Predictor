import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 10

TRAIN_DIR = "../datasets/malaria/train"
VAL_DIR   = "../datasets/malaria/val"

# Labels (alphabetical): Parasitized, Uninfected → index 0, 1
datagen = ImageDataGenerator(rescale=1./255)

train_data = datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE,
    batch_size=BATCH_SIZE, class_mode='binary'
)
val_data = datagen.flow_from_directory(
    VAL_DIR, target_size=IMG_SIZE,
    batch_size=BATCH_SIZE, class_mode='binary'
)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # binary
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_data, validation_data=val_data, epochs=EPOCHS)
model.save("../models/malaria_model.h5")

import json
with open("../models/malaria_classes.json", "w") as f:
    json.dump(train_data.class_indices, f)

print("✅ Malaria model saved!")
print("Class indices:", train_data.class_indices)