from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Data Generator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Training Data

train_generator = train_datagen.flow_from_directory(
    'dataset',
    target_size=(128,128),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

# Validation Data

val_generator = train_datagen.flow_from_directory(
    'dataset',
    target_size=(128,128),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# CNN Model

model = Sequential()

model.add(
    Conv2D(
        32,
        (3,3),
        activation='relu',
        input_shape=(128,128,3)
    )
)

model.add(
    MaxPooling2D(pool_size=(2,2))
)

model.add(
    Conv2D(
        64,
        (3,3),
        activation='relu'
    )
)

model.add(
    MaxPooling2D(pool_size=(2,2))
)

model.add(Flatten())

model.add(
    Dense(
        128,
        activation='relu'
    )
)

model.add(
    Dense(
        1,
        activation='sigmoid'
    )
)

# Compile Model

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train Model

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=5
)

# Save Model

model.save('model.h5')

print("Model Saved Successfully")

# Accuracy Graph

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title('Model Accuracy')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.legend(['Train','Validation'])

plt.show()