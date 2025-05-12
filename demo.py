# --- Imports ---

import os
import numpy as np
import cv2
import torch
import dlib
import face_recognition
from torchvision import transforms
from tqdm import tqdm
#from dataset.loader import normalize_data
from loader import normalize_data
from decord import VideoReader, cpu
import tensorflow as tf
from sklearn.model_selection import train_test_split
import torch.nn as nn
import matplotlib.pyplot as plt # Added for visualization
import timm #added from myside
from timm import create_model
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- GenConViT Model (same as before) ---

def load_genconvit(config, net, ed_weight, vae_weight, fp16):
    model = GenConViT(
        config,
        ed= ed_weight,
        vae= vae_weight, 
        net=net,
        fp16=fp16
    )

    model.to(device)
    model.eval()
    if fp16:
        model.half()

    return model


def face_rec(frames, p=None, klass=None):
    temp_face = np.zeros((len(frames), 224, 224, 3), dtype=np.uint8)
    count = 0
    mod = "cnn" if dlib.DLIB_USE_CUDA else "hog"

    for _, frame in tqdm(enumerate(frames), total=len(frames)):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        face_locations = face_recognition.face_locations(
            frame, number_of_times_to_upsample=0, model=mod
        )

        for face_location in face_locations:
            if count < len(frames):
                top, right, bottom, left = face_location
                face_image = frame[top:bottom, left:right]
                face_image = cv2.resize(
                    face_image, (224, 224), interpolation=cv2.INTER_AREA
                )
                face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

                temp_face[count] = face_image
                count += 1
            else:
                break

    return ([], 0) if count == 0 else (temp_face[:count], count)


def preprocess_frame(frame):
    df_tensor = torch.tensor(frame, device=device).float()
    df_tensor = df_tensor.permute((0, 3, 1, 2))

    for i in range(len(df_tensor)):
        df_tensor[i] = normalize_data()["vid"](df_tensor[i] / 255.0)

    return df_tensor


def pred_vid(df, model):
    with torch.no_grad():
        return max_prediction_value(torch.sigmoid(model(df).squeeze()))


def max_prediction_value(y_pred):
    # Finds the index and value of the maximum prediction value.
    mean_val = torch.mean(y_pred, dim=0)
    return (
        torch.argmax(mean_val).item(),
        mean_val[0].item()
        if mean_val[0] > mean_val[1]
        else abs(1 - mean_val[1]).item(),
    )


def real_or_fake(prediction):
    return {0: "REAL", 1: "FAKE"}[prediction ^ 1]


def extract_frames(video_file, frames_nums=15):
    vr = VideoReader(video_file, ctx=cpu(0))
    step_size = max(1, len(vr) // frames_nums)  # Calculate the step size between frames
    return vr.get_batch(
        list(range(0, len(vr), step_size))[:frames_nums]
    ).asnumpy()  # seek frames with step_size


def df_face(vid, num_frames, net):
    img = extract_frames(vid, num_frames)
    face, count = face_rec(img)
    return preprocess_frame(face) if count > 0 else []


def is_video(vid):
    print('IS FILE', os.path.isfile(vid))
    return os.path.isfile(vid) and vid.endswith(
        tuple([".avi", ".mp4", ".mpg", ".mpeg", ".mov"])
    )


def set_result():
    return {
        "video": {
            "name": [],
            "pred": [],
            "klass": [],
            "pred_label": [],
            "correct_label": [],
        }
    }


def store_result(
    result, filename, y, y_val, klass, correct_label=None, compression=None
):
    result["video"]["name"].append(filename)
    result["video"]["pred"].append(y_val)
    result["video"]["klass"].append(klass.lower())
    result["video"]["pred_label"].append(real_or_fake(y))

    if correct_label is not None:
        result["video"]["correct_label"].append(correct_label)

    if compression is not None:
        result["video"]["compression"].append(compression)

    return result



class GenConViT(nn.Module):

    def __init__(self, config, ed, vae, net, fp16):
        super(GenConViT, self).__init__()
        self.net = net
        self.fp16 = fp16
        if self.net=='ed':
            try:
                from genconvit_ed import GenConViTED
                self.model_ed = GenConViTED(config)
                self.checkpoint_ed = torch.load(f'weight/{ed}.pth', map_location=torch.device('cpu'))

                if 'state_dict' in self.checkpoint_ed:
                    self.model_ed.load_state_dict(self.checkpoint_ed['state_dict'])
                else:
                    self.model_ed.load_state_dict(self.checkpoint_ed)

                self.model_ed.eval()
                if self.fp16:
                    self.model_ed.half()
            except FileNotFoundError:
                raise Exception(f"Error: weight/{ed}.pth file not found.")
        elif self.net=='vae':
            try:
                from genconvit_vae import GenConViTVAE
                self.model_vae = GenConViTVAE(config)
                self.checkpoint_vae = torch.load(f'weight/{vae}.pth', map_location=torch.device('cpu'))

                if 'state_dict' in self.checkpoint_vae:
                    self.model_vae.load_state_dict(self.checkpoint_vae['state_dict'])
                else:
                    self.model_vae.load_state_dict(self.checkpoint_vae)
                    
                self.model_vae.eval()
                if self.fp16:
                    self.model_vae.half()
            except FileNotFoundError:
                raise Exception(f"Error: weight/{vae}.pth file not found.")
        else:
            try:
                from genconvit_ed import GenConViTED
                from genconvit_vae import GenConViTVAE
                self.model_ed = GenConViTED(config)
                self.model_vae = GenConViTVAE(config)
                self.checkpoint_ed = torch.load(f'weight/{ed}.pth', map_location=torch.device('cpu'))
                self.checkpoint_vae = torch.load(f'weight/{vae}.pth', map_location=torch.device('cpu'))
                if 'state_dict' in self.checkpoint_ed:
                    self.model_ed.load_state_dict(self.checkpoint_ed['state_dict'])
                else:
                    self.model_ed.load_state_dict(self.checkpoint_ed)
                if 'state_dict' in self.checkpoint_vae:
                    self.model_vae.load_state_dict(self.checkpoint_vae['state_dict'])
                else:
                    self.model_vae.load_state_dict(self.checkpoint_vae)
                self.model_ed.eval()
                self.model_vae.eval()
                if self.fp16:
                    self.model_ed.half()
                    self.model_vae.half()
            except FileNotFoundError as e:
                raise Exception(f"Error: Model weights file not found.")


    def forward(self, x):
        if self.net == 'ed' :
            x = self.model_ed(x)
        elif self.net == 'vae':
            x,_ = self.model_vae(x)
        else:
            x1 = self.model_ed(x)
            x2,_ = self.model_vae(x)
            x =  torch.cat((x1, x2), dim=0) #(x1+x2)/2 #
        return x


# --- Combined Model ---

IMG_SIZE = 128
FRAMES_PER_VIDEO = 5

def create_combined_model(genconvit_config, genconvit_ed_weight, genconvit_vae_weight, genconvit_net,genconvit_fp16 ,input_shape=(FRAMES_PER_VIDEO, 224, 224, 3)):
    
    
    genconvit_model = load_genconvit(genconvit_config, genconvit_net, genconvit_ed_weight, genconvit_vae_weight, genconvit_fp16)
    base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(include_top=False, weights ='imagenet',input_shape=(IMG_SIZE,IMG_SIZE, 3))

    base_model.trainable = True

    inputs = tf.keras.Input(shape=input_shape)

    # GenConViT Feature Extraction 
    def genconvit_feature_extractor(x):
          
        x_np = x.numpy()
        #print("shape of input",x_np.shape) # (FRAMES_PER_VIDEO,224,224,3)
        x_tensor = preprocess_frame(x_np)
        with torch.no_grad():
            features = genconvit_model(x_tensor)
        #print("shape of tensor",features.shape)
        return features.cpu().numpy()
    
    genconvit_layer = tf.keras.layers.Lambda(genconvit_feature_extractor)

    genconvit_output = tf.keras.layers.TimeDistributed(genconvit_layer)(inputs) #(None,FRAMES_PER_VIDEO,2,768)

    # Reshape tensor 
    if genconvit_net == 'both':
        print("The shape in if condition",genconvit_output.shape)
        reshape_layer = tf.keras.layers.Reshape((FRAMES_PER_VIDEO,2,768,1))

        reshaped_output = reshape_layer(genconvit_output)
    else: 
         print("The shape in else condition",genconvit_output.shape)
         reshape_layer = tf.keras.layers.Reshape((FRAMES_PER_VIDEO,1,768,1))
         reshaped_output = reshape_layer(genconvit_output)
    # CNN (EfficientNet) for Spatial Feature Extraction
    cnn_output= tf.keras.layers.TimeDistributed(base_model)(reshaped_output) # Apply base model to each frame's GenConViT output.

    # LSTM for Temporal Modeling
    x = tf.keras.layers.GlobalAveragePooling3D()(cnn_output)
    x = tf.keras.layers.Reshape((1,-1))(x)
    x = tf.keras.layers.LSTM(256)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    combined_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return combined_model

# --- Data Loading and Preprocessing ---

def load_and_preprocess_videos(path, label):
    videos = []
    labels = []
    for filename in os.listdir(path):
        if filename.endswith(".mp4") or filename.endswith(".avi"):
            try:
                videcap = cv2.VideoCapture(os.path.join(path, filename))
                frames = []
                success, image = videcap.read()
                count = 0
                while success and count < FRAMES_PER_VIDEO:
                    image = cv2.resize(image, (224, 224))
                    frames.append(image)
                    success, image = videcap.read()
                    count += 1
                videcap.release()

                if len(frames) == FRAMES_PER_VIDEO:
                    videos.append(np.array(frames))
                    labels.append(label)

            except Exception as e:
                print(f"Error processing video {filename}: {type(e).__name__} - {e}")

    return np.array(videos), np.array(labels)

# --- Training and Evaluation ---

# Main execution block (for notebook)
if __name__ == '__main__':
    # Define paths and model parameters
    real_videos_path = r"C:\Users\rajku\Downloads\archive\Celeb-real"
    fake_videos_path = r"C:\Users\rajku\Downloads\archive\Celeb-synthesis"
    

    genconvit_config = {
        'image_size': 224,
        'patch_size': 16,
        'num_classes': 2,
        'dim': 768,
        'depth': 12,
        'heads': 12,
        'mlp_dim': 3072,
        'dropout': 0.1,
        'emb_dropout': 0.1,
    }

    genconvit_ed_weight = 'ed_best'
    genconvit_vae_weight = 'vae_best'
    genconvit_net = 'both' # 'ed', 'vae', 'both'
    genconvit_fp16 = False

    # Load and preprocess data
    real_videos, real_labels = load_and_preprocess_videos(real_videos_path, 0)
    fake_videos, fake_labels = load_and_preprocess_videos(fake_videos_path, 1)

    # Combine real and fake data
    X = np.concatenate([real_videos, fake_videos])
    Y = np.concatenate([real_labels, fake_labels])

    # Split data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Create the combined model
    combined_model = create_combined_model(genconvit_config, genconvit_ed_weight, genconvit_vae_weight, genconvit_net, genconvit_fp16 )

    # Compile and train the model
    combined_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = combined_model.fit(
        X_train, Y_train,
        epochs=2, batch_size=8,  # Adjust based on your resources
        validation_data=(X_test, Y_test)
    )

    # Evaluation and Visualization
    _, accuracy = combined_model.evaluate(X_test, Y_test)
    print(f"Test Accuracy: {accuracy}")

    # Plotting the training history
    def plot_history(history):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    plot_history(history)


    # Prediction on new video
    def predict_video(video_path):
        try:
            vidcap = cv2.VideoCapture(video_path)
            frames = []
            success, image = vidcap.read()
            count = 0
            while success and count < FRAMES_PER_VIDEO:
                image = cv2.resize(image, (224, 224))
                frames.append(image)
                success, image = vidcap.read()
                count += 1
            vidcap.release()
            if len(frames) == FRAMES_PER_VIDEO:
                video = np.expand_dims(np.array(frames), axis=0)
                prediction = combined_model.predict(video)[0][0]
                return prediction
            else:
                return None
        except Exception as e:
            print(f"Error processing video: {e}")
            return None

    
    # Example Prediction
    prediction = predict_video(r"archive\Celeb-synthesis\id16_id31_0007.mp4")
     # replace with test video path
    if prediction is not None:
        if prediction > 0.5:
            print("Prediction: Fake (Probability:", prediction, ")")
        else:
            print("Prediction: Real (Probability:", prediction, ")")
    else:
      print("Error processing video.")


