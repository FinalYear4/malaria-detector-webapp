import os
import io
from PIL import Image
import base64
import logging
import cv2
import numpy as np

import torch
import torch.nn as nn
from torchvision import transforms
import segmentation_models_pytorch as smp

from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# --- Configuration ---
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_CHECKPOINT_FILENAME = "best_malaria_classifier_checkpoint.pth"
MODEL_PATH = os.path.join(APP_ROOT, MODEL_CHECKPOINT_FILENAME)

IMAGE_SIZE = 384
# For production servers like Render's free tier, default to CPU.
DEVICE = torch.device("cpu")
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Global variables ---
model = None
class_names = []
classwise_thresholds = {}
target_layer_for_grad_cam = None

# --- Model Definition ---
class MalariaSpeciesClassifier(nn.Module):
    def __init__(self, pretrained_encoder, num_target_classes):
        super().__init__()
        self.encoder = pretrained_encoder
        num_encoder_features = self.encoder.out_channels[-1]
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier_head = nn.Sequential(
            nn.Linear(num_encoder_features, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_target_classes)
        )
    def forward(self, x):
        features_list = self.encoder(x)
        deepest_features = features_list[-1]
        pooled_features = self.avgpool(deepest_features).squeeze(-1).squeeze(-1)
        output = self.classifier_head(pooled_features)
        return output

# --- Model Loading Function ---
def load_model_from_checkpoint():
    global model, class_names, classwise_thresholds, target_layer_for_grad_cam
    if model is not None:
        logging.info("Model is already loaded.")
        return

    logging.info(f"Attempting to load checkpoint from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        # This error is critical in production. If Git LFS worked, the file should be here.
        raise FileNotFoundError(f"Checkpoint file not found at {MODEL_PATH}. Ensure Git LFS is configured correctly and the file is tracked.")

    try:
        # Load checkpoint to CPU, as DEVICE is set to "cpu" for server compatibility.
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        class_names = checkpoint['class_names']
        num_classes = checkpoint['num_classes']
        classwise_thresholds = checkpoint['classwise_thresholds']
        model_state_dict = checkpoint['state_dict']

        logging.info(f"Loaded from checkpoint: {len(class_names)} classes -> {class_names}")
        logging.info(f"Loaded class-wise thresholds: {classwise_thresholds}")

        seg_model_full = smp.UnetPlusPlus(
            encoder_name="efficientnet-b4", encoder_weights=None, in_channels=3, classes=1
        )
        segmentation_encoder = seg_model_full.encoder
        model = MalariaSpeciesClassifier(segmentation_encoder, num_classes)
        model.load_state_dict(model_state_dict)
        model.to(DEVICE)
        model.eval()
        logging.info(f"Model loaded to {DEVICE} and set to evaluation mode.")

        # Set Grad-CAM target layer
        if hasattr(model.encoder, '_blocks') and model.encoder._blocks:
            last_block = model.encoder._blocks[-1]
            if hasattr(last_block, '_bn2'): target_layer_for_grad_cam = [last_block._bn2]
            else: target_layer_for_grad_cam = [last_block]
        elif hasattr(model.encoder, '_conv_head'):
            target_layer_for_grad_cam = [model.encoder._conv_head]
        
        if target_layer_for_grad_cam:
            logging.info(f"Grad-CAM target layer set to: {type(target_layer_for_grad_cam[0])}")
        else:
            logging.error("Could not find a suitable layer for Grad-CAM.")
            
    except Exception as e:
        logging.error(f"Error loading model from checkpoint: {e}", exc_info=True)
        model = None
        raise

# --- Image Processing & Grad-CAM ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def transform_image_for_model_input(pil_image):
    mean_vals = [0.485, 0.456, 0.406]
    std_vals = [0.229, 0.224, 0.225]
    model_input_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE + 32),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_vals, std=std_vals)
    ])
    return model_input_transform(pil_image).unsqueeze(0).to(DEVICE)

def generate_grad_cam_overlay(input_tensor, original_pil_image, target_class_index):
    if not target_layer_for_grad_cam:
        return None
    try:
        base_image_for_overlay = np.array(original_pil_image.resize((IMAGE_SIZE, IMAGE_SIZE)), dtype=np.float32) / 255.0

        with GradCAM(model=model, target_layers=target_layer_for_grad_cam) as cam:
            targets = [ClassifierOutputTarget(target_class_index)]
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
            
            visualization = show_cam_on_image(base_image_for_overlay, grayscale_cam, use_rgb=True)
            
            cam_image_pil = Image.fromarray(visualization)
            buffered = io.BytesIO()
            cam_image_pil.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
            
    except Exception as e:
        logging.error(f"Error generating Grad-CAM: {e}", exc_info=True)
        return None

# --- Flask App Initialization & Routes ---
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# --- SERVER STARTUP LOGIC ---
# This block runs once when the Gunicorn server process starts.
try:
    load_model_from_checkpoint()
except Exception as e:
    logging.critical(f"FATAL: Could not load the model checkpoint at startup. The application will not be able to serve predictions. Error: {e}", exc_info=True)
# --- END SERVER STARTUP LOGIC ---

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', error_message="Model is not available. Please contact the administrator.")

    if 'file' not in request.files: return redirect(request.url)
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename): return redirect(request.url)

    try:
        filename = secure_filename(file.filename)
        img_bytes = file.read()
        original_pil_image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        input_tensor = transform_image_for_model_input(original_pil_image.copy())

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            confidence, predicted_idx_tensor = torch.max(probabilities, 0)

        predicted_idx = predicted_idx_tensor.item()
        confidence_score = confidence.item()
        predicted_class_name = class_names[predicted_idx]
        
        is_unconfident = False
        # Ensure thresholds are standard floats for comparison, not numpy types
        threshold_used = float(classwise_thresholds.get(predicted_class_name, 0.70))
        
        if confidence_score < threshold_used:
            final_predicted_class = "Unknown / Unconfident"
            is_unconfident = True
            logging.info(f"Prediction for {filename} ({predicted_class_name}) is below threshold. Confidence: {confidence_score:.4f} < Threshold: {threshold_used:.4f}. Classifying as 'Unknown'.")
        else:
            final_predicted_class = predicted_class_name
            logging.info(f"Prediction for {filename}: {final_predicted_class} with confidence {confidence_score:.4f} >= Threshold {threshold_used:.4f}.")
        
        all_class_probs = {name: f"{prob.item()*100:.2f}%" for name, prob in zip(class_names, probabilities)}
        
        grad_cam_base64 = None
        if not is_unconfident and target_layer_for_grad_cam:
            grad_cam_base64 = generate_grad_cam_overlay(
                input_tensor=input_tensor,
                original_pil_image=original_pil_image,
                target_class_index=predicted_idx
            )

        img_base64 = base64.b64encode(img_bytes).decode('utf-8')

        return render_template('result.html',
                               predicted_class=final_predicted_class,
                               confidence=f"{confidence_score*100:.2f}%",
                               all_probs=all_class_probs,
                               image_data=img_base64,
                               grad_cam_image_data=grad_cam_base64,
                               is_unconfident=is_unconfident,
                               threshold=f"{threshold_used*100:.2f}%",
                               original_prediction=predicted_class_name,
                               filename=filename)
    except Exception as e:
        logging.error(f"Error during prediction for {file.filename}: {e}", exc_info=True)
        return render_template('index.html', error_message="An error occurred during prediction.")

@app.route('/tutorials')
def tutorials_page():
    return render_template('tutorials.html')

@app.route('/faq')
def faq_page():
    return render_template('faq_page.html')

@app.route('/contact')
def contact_page():
    return render_template('contact.html')

# This block is used for local development and will not be used by Gunicorn on Render.
if __name__ == '__main__':
    logging.info("Starting Malaria Species Predictor Flask app for local development...")
    from waitress import serve
    serve(app, host="0.0.0.0", port=5000)