import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

# Load the Fer2013 dataset
data = pd.read_csv("C:\\Users\\mukes\\Downloads\\fer2013.csv")

# Extract pixels and emotions from the dataset
pixels = data['pixels'].tolist()
emotions = pd.get_dummies(data['emotion']).to_numpy()

# Convert pixel values to images
images = []
for pixel_sequence in pixels:
    pixel_array = np.asarray([int(pixel) for pixel in pixel_sequence.split(' ')])
    image = np.reshape(pixel_array, (48, 48, 1))
    images.append(image)
images = np.asarray(images)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, emotions, test_size=0.2, random_state=42)

# Normalize the pixel values
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Define the model architecture
model = Sequential()

# Add a Flatten layer to convert input shape from (None, 48, 48, 1) to (None, 2304)
model.add(Flatten(input_shape=(48, 48, 1)))

# Add a Dense layer with 128 units and relu activation
model.add(Dense(128, activation='relu'))

# Add a Dropout layer with rate 0.5
model.add(Dropout(0.5))

# Add a Dense layer with 64 units and relu activation
model.add(Dense(64, activation='relu'))

# Add a Dropout layer with rate 0.5
model.add(Dropout(0.5))

# Add a Dense layer with 7 units and softmax activation
model.add(Dense(7, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Define the checkpoint to save the model with the best accuracy during training
checkpoint = ModelCheckpoint('facial_expression_recognition.h5', monitor='val_accuracy', save_best_only=True, mode='max')

# Train the model
history = model.fit(X_train, y_train, epochs=200, batch_size=64, validation_data=(X_test, y_test), callbacks=[checkpoint])

# Plot the training and validation accuracy and loss curves
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Save the model to a h5 file
if not os.path.exists('models'):
    os.mkdir('models')
model.save('models/facial_expression_recognition.h5')

# Load the saved model
saved_model = load_model('models/facial_expression_recognition.h5')

# Evaluate the model on the test data
test_loss, test_acc = saved_model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)
