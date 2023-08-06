import cv2
import numpy as np
from keras.models import load_model

# Load the saved model
model = load_model("C:\\Users\\mukes\\PycharmProjects\\Emotion Detection\\facial_expression_recognition.h5")

# Define the emotions
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Create a video capture object
cap = cv2.VideoCapture(0)

while True:
    # Read the current frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect the face in the current frame
    face_cascade = cv2.CascadeClassifier("C:\\Users\\mukes\\PycharmProjects\\Emotion Detection\\haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # For each face detected, predict the emotion
    for (x, y, w, h) in faces:
        # Extract the face from the frame and resize it to 48x48 pixels
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        # Normalize the pixel values between 0 and 1
        roi_gray = roi_gray / 255.0

        # Reshape the face to match the input shape of the model
        roi_gray = roi_gray.reshape(1, 48, 48, 1)

        # Predict the emotion using the trained model
        predicted_emotion = model.predict(roi_gray)

        # Get the emotion with the highest predicted probability
        emotion = emotions[np.argmax(predicted_emotion)]

        # Draw a rectangle around the face and display the predicted emotion
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the current frame with the predicted emotion
    cv2.imshow('Facial Expression Recognition', frame)

    # Exit the program when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
