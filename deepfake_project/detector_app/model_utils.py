# detector_app/model_utils.py
import numpy as np
import cv2
import os
import tensorflow as tf
from django.conf import settings # To get BASE_DIR

# --- Constants from your notebook ---
IMG_SIZE = 128
FRAMES_PER_VIDEO = 10
MODEL_FILE_NAME = 'best_deepfake_model.keras' # Or 'best_deepfake_model.keras'
MODEL_PATH = os.path.join(settings.BASE_DIR, 'detector_app', MODEL_FILE_NAME)

# --- Load the model once when Django starts ---
# This is better for performance than loading it on every request
try:
    print(f"Attempting to load model from: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    loaded_model = tf.keras.models.load_model(MODEL_PATH)
    print("Deepfake detection model loaded successfully.")
except Exception as e:
    print(f"Error loading the deepfake model: {e}")
    loaded_model = None # Handle this case in your view

# --- Your create_model function (if needed for loading, but usually load_model is enough) ---
# If your .h5 or .keras file needs custom objects, you might need to define them here
# or pass them to load_model via custom_objects argument.
# For a standard model saved with model.save(), this is usually not needed.

# --- Your prediction function, adapted ---
def predict_single_video(video_path):
    if not loaded_model:
        return {"error": "Model not loaded. Cannot predict."}
    if not os.path.exists(video_path):
        return {"error": f"Video file not found at {video_path}"}

    try:
        vidcap = cv2.VideoCapture(video_path)
        if not vidcap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return {"error": f"Could not open video file."}

        frames = []
        success, image = vidcap.read()
        count = 0
        while success and count < FRAMES_PER_VIDEO:
            try:
                if image is None: # Check if frame is empty
                    print(f"Warning: Empty frame read at count {count} for video {video_path}")
                    success, image = vidcap.read() # Try reading next frame
                    continue
                image_resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
                frames.append(image_resized)
            except cv2.error as e:
                print(f"OpenCV error processing frame {count} for video {video_path}: {e}")
                # Potentially skip this frame or handle error
            success, image = vidcap.read()
            count += 1
        vidcap.release()

        if len(frames) == FRAMES_PER_VIDEO:
            video_array = np.array(frames) # No normalization here if your model expects 0-255
                                          # Or / 255.0 if it expects 0-1 (check your training)
            video_tensor = np.expand_dims(video_array, axis=0) # Add batch dimension
            
            prediction_prob = loaded_model.predict(video_tensor)[0][0] # Get the single probability value

            threshold = 0.5 # Or whatever threshold you used
            if prediction_prob > threshold:
                return {"prediction": "Fake", "probability": float(prediction_prob)}
            else:
                return {"prediction": "Real", "probability": float(prediction_prob)}
        elif len(frames) < FRAMES_PER_VIDEO and len(frames) > 0:
             # Handle videos with fewer frames than expected, e.g., by padding
             # For simplicity, we'll say not enough frames.
             # Or, you could pad with zeros or last frame if your model can handle it.
            print(f"Warning: Video {video_path} had {len(frames)} frames, expected {FRAMES_PER_VIDEO}.")
            # Pad frames if necessary (example with zero padding)
            while len(frames) < FRAMES_PER_VIDEO:
                frames.append(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)) # Zero padding
            
            video_array = np.array(frames)
            video_tensor = np.expand_dims(video_array, axis=0)
            prediction_prob = loaded_model.predict(video_tensor)[0][0]
            threshold = 0.5
            if prediction_prob > threshold:
                return {"prediction": "Fake", "probability": float(prediction_prob)}
            else:
                return {"prediction": "Real", "probability": float(prediction_prob)}

        else:
            print(f"Error: Video {video_path} had {len(frames)} frames. Not enough frames for prediction or no frames extracted.")
            return {"error": f"Not enough frames ({len(frames)}) extracted for prediction. Required {FRAMES_PER_VIDEO}."}
    except Exception as e:
        print(f"Error processing video {video_path} for prediction: {e}")
        return {"error": f"An error occurred during prediction: {str(e)}"}