import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

# Load the pre-trained MobileNetV2 model
model = MobileNetV2(weights='imagenet')

# Initialize the camera
camera = cv2.VideoCapture(0)  # Change to the correct camera index if needed

# Check if the camera opened successfully
if not camera.isOpened():
    print("Error: Could not open camera.")
    exit()

# Function to preprocess the frame
def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (224, 224))
    frame_preprocessed = preprocess_input(frame_resized)
    return np.expand_dims(frame_preprocessed, axis=0)

while True:
    # Capture frame-by-frame
    ret, frame = camera.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Preprocess the frame for model prediction
    preprocessed_frame = preprocess_frame(frame)

    # Perform the prediction
    predictions = model.predict(preprocessed_frame)
    decoded_predictions = decode_predictions(predictions, top=1)[0]

    # Get the predicted label and confidence
    label = decoded_predictions[0][1]
    confidence = decoded_predictions[0][2]

    # Display the label and confidence on the frame
    label_text = f"{label}: {confidence:.2f}"
    cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Camera Feed', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the camera and close windows
camera.release()
cv2.destroyAllWindows()
