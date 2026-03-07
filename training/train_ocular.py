import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 10

TRAIN_DIR = "../datasets/ocular/train"
VAL_DIR   = "../datasets/ocular/val"

# 8 classes (alphabetical): A, C, D, G, H, M, N, O
datagen = ImageDataGenerator(rescale=1./255)

train_data = datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE,
    batch_size=BATCH_SIZE, class_mode='categorical'
)
val_data = datagen.flow_from_directory(
    VAL_DIR, target_size=IMG_SIZE,
    batch_size=BATCH_SIZE, class_mode='categorical'
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
    Dense(8, activation='softmax')  # 8 classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data, validation_data=val_data, epochs=EPOCHS)
model.save("../models/ocular_model.h5")

import json
with open("../models/ocular_classes.json", "w") as f:
    json.dump(train_data.class_indices, f)

print("✅ Ocular model saved!")
print("Class indices:", train_data.class_indices)