import io
import os
import base64
import json
import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Attempt to load custom trained model
model_path = 'best_pest_model.pth'
class_names_path = 'class_names.json'
is_custom_model = False

weights_enum = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights_enum)
class_names = []

if os.path.exists(model_path) and os.path.exists(class_names_path):
    print("Loading custom trained IP102 ResNet18...")
    with open(class_names_path, 'r') as f:
        class_names = json.load(f)
    
    num_classes = len(class_names)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    is_custom_model = True
else:
    print("Custom model not found. Using pre-trained ImageNet ResNet18 for demonstration.")
    class_names = weights_enum.meta["categories"]

model = model.to(device)
model.eval()

# Grad-CAM setup
target_layers = [model.layer4[-1]]
cam = GradCAM(model=model, target_layers=target_layers)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


with open('pest_db.json', 'r') as f:
    pest_db = json.load(f)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # Process for inference
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)
    
    confidence_val = confidence.item() * 100
    
    if is_custom_model:
        predicted_class = class_names[predicted_idx.item()].title()
    else:
        predicted_class = class_names[predicted_idx.item()].title()
    
    if predicted_class not in pest_db:
        import copy
        pest_db[predicted_class] = copy.deepcopy(pest_db["Default"])
        pest_db[predicted_class]["scientific_name"] = f"Identified as: {predicted_class}"
        if not is_custom_model:
            pest_db[predicted_class]["fun_fact"] = f"The AI recognized this as a '{predicted_class}'. Since the specific pest model isn't trained yet, it's using a general AI model."
        else:
            pest_db[predicted_class]["fun_fact"] = f"Identified {predicted_class} with {confidence_val:.1f}% confidence."
        pest_db[predicted_class]["impact"]["loss_desc"] = "Agricultural data pending for this specific species."
        pest_db[predicted_class]["impact"]["infestation_level"] = "Identified Infestation"
    # Generate Grad-CAM
    # Re-enable grads for CAM
    input_tensor.requires_grad_(True)
    grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(predicted_idx.item())])
    grayscale_cam = grayscale_cam[0, :]
    
    # Overlay CAM on original image
    img_np = np.array(image.resize((224, 224))) / 255.0
    cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
    
    # Convert CAM image to Base64 to send to frontend
    cam_pil = Image.fromarray(cam_image)
    buffered = io.BytesIO()
    cam_pil.save(buffered, format="JPEG")
    cam_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    # Prepare original image base64
    orig_pil = image.resize((224, 224))
    orig_buffered = io.BytesIO()
    orig_pil.save(orig_buffered, format="JPEG")
    orig_base64 = base64.b64encode(orig_buffered.getvalue()).decode("utf-8")

    # Get DB info
    db_info = pest_db.get(predicted_class, pest_db["Default"])
    
    return {
        "class_name": predicted_class,
        "confidence": round(confidence_val, 2),
        "heatmap_base64": f"data:image/jpeg;base64,{cam_base64}",
        "original_base64": f"data:image/jpeg;base64,{orig_base64}",
        "details": db_info
    }

# Serve static files
app.mount("/assets", StaticFiles(directory="."), name="assets")

@app.get("/")
async def read_index():
    return FileResponse('index.html')

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
